#!/usr/bin/env python3
"""CPU-only admission for a sealed E1 frozen-base KNN projection bundle.

This is a GPU gate preflight, not merely a shape checker.  It binds the
projection manifest, metadata policy, original source trace, real projected
event bytes, and the replayed final active-set witness before a native runner
may construct its CUDA/GTS image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "e1-frozen-base-knn-projection-manifest-v1"
METADATA_SCHEMA = "e1-frozen-base-knn-projection-bundle-v1"
ADMISSION_SCHEMA = "e1-frozen-base-knn-projection-admission-v2"
MAGIC = b"E1GTRC01"
VERSION = 1
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
INSERT, DELETE, KNN, RANGE = 1, 2, 3, 4
OP_NAME = {INSERT: "insert", DELETE: "delete", KNN: "knn", RANGE: "range"}
REQUIRED_MANIFEST_FILES = {
    "metadata.json",
    "trace.e1gtrc",
    "pool.i16",
    "queries.i16",
    "stable_id_to_pool_row.i32",
    "initial_base_stable_ids.i32",
}
EXPECTED_ALLOWED = ["insert", "knn"]
EXPECTED_EXCLUDED = ["delete", "range"]
EXPECTED_FILTER_POLICY = (
    "Retain only source insert/KNN events in source order; exclude all source deletes "
    "because they are immutable-base deletions, and exclude range because this is KNN-only."
)
EXPECTED_ORACLE_ROLE = "source provenance only; never a projection oracle"


class Fail(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Fail(message)


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def require_sha(value: Any, name: str) -> str:
    require(is_sha256(value), f"{name} must be a lowercase SHA-256 hex digest")
    return str(value)


def regular_nonsymlink(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise Fail(f"missing required file: {path}") from exc
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode),
            f"required file must be a regular non-symlink: {path}")


def real_directory(path: Path, label: str) -> Path:
    require(path.is_dir() and not path.is_symlink(), f"{label} must be a real directory: {path}")
    return path.resolve()


def sha256_file(path: Path) -> str:
    regular_nonsymlink(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_exact(path: Path, expected_bytes: int) -> bytes:
    regular_nonsymlink(path)
    actual = path.stat().st_size
    require(actual == expected_bytes,
            f"wrong byte size for {path.name}: got {actual}, expected {expected_bytes}")
    payload = path.read_bytes()
    require(len(payload) == expected_bytes, f"short read for {path.name}")
    return payload


def read_json(path: Path, label: str) -> dict[str, Any]:
    regular_nonsymlink(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Fail(f"invalid {label} JSON: {exc}") from exc
    require(isinstance(value, dict), f"{label} root must be an object")
    return value


def active_hash(active: set[int]) -> str:
    return hashlib.sha256("".join(f"{sid}\n" for sid in sorted(active)).encode("ascii")).hexdigest()


def header_from_values(values: tuple[Any, ...]) -> dict[str, Any]:
    magic, version, dimension, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = values
    return {
        "magic": magic.decode("ascii", errors="strict"),
        "version": version,
        "dimension": dimension,
        "base_n": base_n,
        "reservoir_n": reservoir_n,
        "pool_n": pool_n,
        "query_n": query_n,
        "k": k,
        "radius": radius,
        "event_count": event_count,
    }


def validate_header(header: dict[str, Any], label: str) -> None:
    expected = {"magic", "version", "dimension", "base_n", "reservoir_n",
                "pool_n", "query_n", "k", "radius", "event_count"}
    require(set(header) == expected, f"{label} header fields drift")
    require(header["magic"] == MAGIC.decode("ascii") and header["version"] == VERSION,
            f"{label} magic/version mismatch")
    for key in ("dimension", "base_n", "reservoir_n", "pool_n", "query_n", "k", "event_count"):
        require(isinstance(header[key], int) and not isinstance(header[key], bool) and header[key] >= 0,
                f"{label} {key} must be a nonnegative integer")
    require(header["dimension"] > 0 and header["base_n"] > 0 and
            header["pool_n"] >= header["base_n"] and header["query_n"] > 0 and header["k"] > 0,
            f"{label} has invalid dimensions")
    require(header["reservoir_n"] == header["pool_n"] - header["base_n"],
            f"{label} reservoir/pool/base contract mismatch")
    require(isinstance(header["radius"], (int, float)) and not isinstance(header["radius"], bool),
            f"{label} radius must be numeric")


def parse_trace(path: Path, allowed_ops: set[int], label: str) -> tuple[dict[str, Any], list[tuple[int, int, int]], bytes]:
    regular_nonsymlink(path)
    raw = path.read_bytes()
    require(len(raw) >= HEADER.size, f"{label} trace is shorter than header")
    values = HEADER.unpack_from(raw)
    require(values[0] == MAGIC and values[1] == VERSION, f"{label} trace magic/version mismatch")
    header = header_from_values(values)
    validate_header(header, label)
    expected_bytes = HEADER.size + header["event_count"] * EVENT.size
    require(len(raw) == expected_bytes,
            f"{label} trace byte count does not match event count")
    events: list[tuple[int, int, int]] = []
    for position in range(header["event_count"]):
        op_index, opcode, argument = EVENT.unpack_from(raw, HEADER.size + position * EVENT.size)
        require(op_index == position, f"{label} event index is non-contiguous at {position}")
        require(opcode in allowed_ops, f"{label} illegal opcode {opcode} at event {position}")
        events.append((op_index, opcode, argument))
    return header, events, raw


def require_same_header(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    validate_header(left, f"{label}/left")
    validate_header(right, f"{label}/right")
    require(left == right, f"{label} header mismatch")


def safe_manifest_name(name: Any) -> str:
    require(isinstance(name, str) and name and Path(name).name == name and name not in {".", ".."},
            "manifest files_sha256 key is not a safe basename")
    return name


def parse_i32(path: Path, count: int) -> list[int]:
    return list(struct.unpack(f"<{count}i", read_exact(path, count * 4)))


def verify_manifest_files(bundle: Path, manifest: dict[str, Any]) -> dict[str, str]:
    file_map = manifest.get("files_sha256")
    require(isinstance(file_map, dict), "manifest.files_sha256 must be an object")
    names = {safe_manifest_name(name) for name in file_map}
    missing = REQUIRED_MANIFEST_FILES - names
    require(not missing, f"manifest.files_sha256 omits required files: {sorted(missing)}")
    hashes: dict[str, str] = {}
    for name in sorted(names):
        expected = require_sha(file_map[name], f"manifest.files_sha256[{name!r}]")
        actual = sha256_file(bundle / name)
        require(actual == expected, f"manifest payload hash mismatch: {name}")
        hashes[name] = actual
    return hashes


def validate_projection_metadata(bundle: Path, manifest: dict[str, Any],
                                 payload_hashes: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = bundle / "metadata.json"
    metadata = read_json(metadata_path, "metadata")
    require(metadata.get("schema") == METADATA_SCHEMA, "metadata schema mismatch")
    require(payload_hashes["metadata.json"] == sha256_file(metadata_path),
            "metadata hash is not bound by manifest")
    projection = metadata.get("projection")
    require(isinstance(projection, dict), "metadata.projection must be an object")
    require(projection.get("allowed_ops") == EXPECTED_ALLOWED,
            "metadata projection allowed_ops must be exactly insert,knn")
    require(projection.get("excluded_ops") == EXPECTED_EXCLUDED,
            "metadata projection excluded_ops must be exactly delete,range")
    require(projection.get("filter_policy") == EXPECTED_FILTER_POLICY,
            "metadata projection filter policy drift")
    require(projection.get("source_quantized_oracle_role") == EXPECTED_ORACLE_ROLE,
            "metadata source oracle role drift")

    projected_header = metadata.get("header")
    source_header = metadata.get("source_header")
    require(isinstance(projected_header, dict) and isinstance(source_header, dict),
            "metadata must contain projected and source headers")
    validate_header(projected_header, "metadata projected")
    validate_header(source_header, "metadata source")

    for name in ("source_trace_sha256", "source_event_stream_sha256",
                 "source_manifest_sha256", "source_metadata_sha256"):
        require_sha(metadata.get(name), f"metadata.{name}")
    for name in ("projection_trace_sha256", "projection_event_stream_sha256",
                 "source_indices_sha256", "stable_id_mapping_sha256",
                 "initial_base_stable_ids_sha256", "final_active_set_sha256"):
        require_sha(projection.get(name), f"metadata.projection.{name}")
    for name in ("source_indices_count", "first_source_index", "last_source_index",
                 "final_active_count"):
        require(isinstance(projection.get(name), int) and not isinstance(projection.get(name), bool),
                f"metadata.projection.{name} must be an integer")
    require(projection["source_indices_count"] > 0 and projection["first_source_index"] >= 0 and
            projection["last_source_index"] >= projection["first_source_index"] and
            projection["final_active_count"] > 0,
            "metadata projection witness has invalid counts")

    require(manifest.get("projection_trace_sha256") == projection["projection_trace_sha256"],
            "manifest/metadata projection trace hash disagreement")
    require(manifest.get("source_trace_sha256") == metadata["source_trace_sha256"],
            "manifest/metadata source trace hash disagreement")
    require(payload_hashes["trace.e1gtrc"] == projection["projection_trace_sha256"],
            "metadata projection trace hash is not actual trace hash")
    require(payload_hashes["stable_id_to_pool_row.i32"] == projection["stable_id_mapping_sha256"],
            "metadata mapping hash mismatch")
    require(payload_hashes["initial_base_stable_ids.i32"] ==
            projection["initial_base_stable_ids_sha256"],
            "metadata base-ID hash mismatch")
    return metadata, projection


def validate_binary_inputs(bundle: Path, header: dict[str, Any]) -> tuple[list[int], list[int]]:
    read_exact(bundle / "pool.i16", header["pool_n"] * header["dimension"] * 2)
    read_exact(bundle / "queries.i16", header["query_n"] * header["dimension"] * 2)
    mapping = parse_i32(bundle / "stable_id_to_pool_row.i32", header["pool_n"])
    base_ids = parse_i32(bundle / "initial_base_stable_ids.i32", header["base_n"])
    require(len(set(mapping)) == header["pool_n"] and all(0 <= row < header["pool_n"] for row in mapping),
            "stableID->pool-row mapping must be a full physical-row bijection")
    require(len(set(base_ids)) == header["base_n"] and
            all(0 <= stable < header["pool_n"] for stable in base_ids),
            "initial base IDs must be distinct mapped stable IDs")
    return mapping, base_ids


def replay_projected(header: dict[str, Any], events: list[tuple[int, int, int]],
                     base_ids: list[int]) -> tuple[Counter[str], set[int]]:
    active = set(base_ids)
    counts: Counter[str] = Counter()
    for index, opcode, argument in events:
        if opcode == INSERT:
            require(header["base_n"] <= argument < header["pool_n"],
                    f"projected insert {index} is outside declared reservoir")
            require(argument not in active, f"projected insert {index} repeats a live stable ID")
            active.add(argument)
        else:
            require(0 <= argument < header["query_n"],
                    f"projected KNN {index} query id is outside matrix")
        counts[OP_NAME[opcode]] += 1
    require(set(counts) == {"insert", "knn"} and counts["insert"] > 0 and counts["knn"] > 0,
            "projection must contain positive insert and KNN counts only")
    return counts, active


def validate_source_policy(metadata: dict[str, Any], projection: dict[str, Any],
                           projected_header: dict[str, Any],
                           projected_raw: bytes, projected_events: list[tuple[int, int, int]],
                           base_ids: list[int], bundle: Path) -> Counter[str]:
    source_root_value = metadata.get("source_bundle")
    require(isinstance(source_root_value, str) and source_root_value,
            "metadata.source_bundle must be a nonempty path")
    source = real_directory(Path(source_root_value), "metadata.source_bundle")
    source_trace = source / "trace.e1gtrc"
    source_manifest = source / "manifest.json"
    source_metadata = source / "metadata.json"
    require(sha256_file(source_trace) == metadata["source_trace_sha256"], "source trace hash mismatch")
    require(sha256_file(source_manifest) == metadata["source_manifest_sha256"], "source manifest hash mismatch")
    require(sha256_file(source_metadata) == metadata["source_metadata_sha256"], "source metadata hash mismatch")
    source_header, source_events, source_raw = parse_trace(
        source_trace, {INSERT, DELETE, KNN, RANGE}, "source")
    require_same_header(source_header, metadata["source_header"], "actual source/metadata source")
    require(hashlib.sha256(source_raw[HEADER.size:]).hexdigest() == metadata["source_event_stream_sha256"],
            "source event-stream hash mismatch")

    # The narrow projection is valid only if every omitted delete targets an
    # initial immutable-base ID in the original source replay.
    source_base = parse_i32(source / "initial_base_stable_ids.i32", source_header["base_n"])
    require(sha256_file(source / "initial_base_stable_ids.i32") ==
            sha256_file(bundle / "initial_base_stable_ids.i32"),
            "source/projected initial base payload differs")
    require(sha256_file(source / "stable_id_to_pool_row.i32") ==
            sha256_file(bundle / "stable_id_to_pool_row.i32"),
            "source/projected stable mapping differs")
    require(sha256_file(source / "pool.i16") == sha256_file(bundle / "pool.i16"),
            "source/projected pool differs")
    require(sha256_file(source / "queries.i16") == sha256_file(bundle / "queries.i16"),
            "source/projected queries differ")
    require(source_base == base_ids, "source/projected initial base IDs differ")

    active = set(source_base)
    base_set = set(source_base)
    source_counts: Counter[str] = Counter()
    filtered: list[tuple[int, int, int]] = []
    source_indices: list[int] = []
    for old_index, opcode, argument in source_events:
        if opcode == INSERT:
            require(source_header["base_n"] <= argument < source_header["pool_n"],
                    f"source insert {old_index} outside reservoir")
            require(argument not in active, f"source insert {old_index} repeats a live stable ID")
            active.add(argument)
        elif opcode == DELETE:
            require(argument in active, f"source delete {old_index} targets inactive stable ID")
            require(argument in base_set,
                    f"source delete {old_index} is mutable; immutable-base projection would be unsound")
            active.remove(argument)
        else:
            require(0 <= argument < source_header["query_n"],
                    f"source query {old_index} id outside matrix")
        source_counts[OP_NAME[opcode]] += 1
        if opcode in (INSERT, KNN):
            source_indices.append(old_index)
            filtered.append((len(filtered), opcode, argument))

    require(source_counts["delete"] > 0 and source_counts["range"] > 0,
            "source policy provenance must contain excluded delete and range events")
    expected_counts = metadata.get("trace_counts")
    require(isinstance(expected_counts, dict), "metadata.trace_counts must be an object")
    require(expected_counts.get("source") == dict(source_counts),
            "metadata source trace_counts mismatch")
    projection_counts = Counter(OP_NAME[opcode] for _, opcode, _ in projected_events)
    require(expected_counts.get("projection") == dict(projection_counts),
            "metadata projection trace_counts mismatch")
    require(source_counts["insert"] == projection_counts["insert"] and
            source_counts["knn"] == projection_counts["knn"],
            "projection did not retain all source inserts/KNNs")

    require(source_header["event_count"] == sum(source_counts.values()) and
            projected_header["event_count"] == len(filtered) == len(projected_events),
            "header event count / trace count mismatch")
    for key in ("magic", "version", "dimension", "base_n", "reservoir_n", "pool_n", "query_n", "k", "radius"):
        require(source_header[key] == projected_header[key],
                f"source/projected header changed non-event field {key}")
    require(source_header["event_count"] > projected_header["event_count"],
            "projection source header does not contain excluded operations")

    expected_raw = bytearray()
    expected_raw.extend(HEADER.pack(
        MAGIC, VERSION, projected_header["dimension"], projected_header["base_n"],
        projected_header["reservoir_n"], projected_header["pool_n"], projected_header["query_n"],
        projected_header["k"], projected_header["radius"], len(filtered)))
    for event in filtered:
        expected_raw.extend(EVENT.pack(*event))
    require(bytes(expected_raw) == projected_raw,
            "projected trace is not exactly the ordered source insert/KNN projection")
    require(hashlib.sha256("".join(f"{index}\n" for index in source_indices).encode("ascii")).hexdigest() ==
            projection["source_indices_sha256"],
            "projection source-index hash mismatch")
    require(projection["source_indices_count"] == len(source_indices) and
            projection["first_source_index"] == source_indices[0] and
            projection["last_source_index"] == source_indices[-1],
            "projection source-index metadata mismatch")
    return source_counts


def validate_bundle(bundle_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    bundle = real_directory(bundle_path, "bundle")
    manifest = read_json(bundle / "manifest.json", "manifest")
    require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema mismatch")
    require_sha(manifest.get("source_trace_sha256"), "manifest.source_trace_sha256")
    require_sha(manifest.get("projection_trace_sha256"), "manifest.projection_trace_sha256")
    payload_hashes = verify_manifest_files(bundle, manifest)
    metadata, projection = validate_projection_metadata(bundle, manifest, payload_hashes)

    projected_header, projected_events, projected_raw = parse_trace(
        bundle / "trace.e1gtrc", {INSERT, KNN}, "projected")
    require_same_header(projected_header, metadata["header"], "actual projected/metadata projected")
    require(hashlib.sha256(projected_raw[HEADER.size:]).hexdigest() ==
            projection["projection_event_stream_sha256"],
            "metadata projection event-stream hash mismatch")
    mapping, base_ids = validate_binary_inputs(bundle, projected_header)
    del mapping  # only its byte-level bijection is needed at admission.
    projection_counts, final_active = replay_projected(projected_header, projected_events, base_ids)
    require(metadata["trace_counts"]["projection"] == dict(projection_counts),
            "metadata projected trace counts mismatch")
    require(projection["final_active_count"] == len(final_active),
            "metadata final active count mismatch")
    computed_active_hash = active_hash(final_active)
    require(projection["final_active_set_sha256"] == computed_active_hash,
            "metadata final active-set hash mismatch")
    validate_source_policy(metadata, projection, projected_header, projected_raw,
                           projected_events, base_ids, bundle)

    hashes = {
        "manifest_sha256": sha256_file(bundle / "manifest.json"),
        "metadata_sha256": payload_hashes["metadata.json"],
        "trace_sha256": payload_hashes["trace.e1gtrc"],
        "pool_sha256": payload_hashes["pool.i16"],
        "queries_sha256": payload_hashes["queries.i16"],
        "mapping_sha256": payload_hashes["stable_id_to_pool_row.i32"],
        "base_ids_sha256": payload_hashes["initial_base_stable_ids.i32"],
        "projection_event_stream_sha256": projection["projection_event_stream_sha256"],
        "source_trace_sha256": metadata["source_trace_sha256"],
        "source_event_stream_sha256": metadata["source_event_stream_sha256"],
    }
    info: dict[str, Any] = {
        "bundle_realpath": str(bundle),
        "dimension": projected_header["dimension"],
        "base_n": projected_header["base_n"],
        "pool_n": projected_header["pool_n"],
        "query_n": projected_header["query_n"],
        "k": projected_header["k"],
        "event_count": projected_header["event_count"],
        "insert_count": projection_counts["insert"],
        "knn_count": projection_counts["knn"],
        "final_active_count": len(final_active),
        "final_active_set_sha256": computed_active_hash,
        "source_event_count": metadata["source_header"]["event_count"],
    }
    return hashes, info


def write_admission(path: Path, hashes: dict[str, str], info: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lines = [
        f"schema={ADMISSION_SCHEMA}",
        "status=PASS",
        f"bundle_realpath={info['bundle_realpath']}",
        f"projection_manifest_schema={MANIFEST_SCHEMA}",
        f"projection_metadata_schema={METADATA_SCHEMA}",
        "ops=insert,knn",
        "excluded_ops=delete,range",
        "base_immutable=true",
        "direct_sidecar_allowed=false",
        "legacy_routing_allowed=false",
    ]
    for name in ("dimension", "base_n", "pool_n", "query_n", "k", "event_count",
                 "insert_count", "knn_count", "final_active_count", "source_event_count"):
        lines.append(f"{name}={info[name]}")
    lines.append(f"final_active_set_sha256={info['final_active_set_sha256']}")
    for name in sorted(hashes):
        lines.append(f"{name}={hashes[name]}")
    payload = "\n".join(lines) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".frozen-base-preflight.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="strict CPU-only admission for E1 frozen-base KNN projection")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        hashes, info = validate_bundle(args.bundle)
        write_admission(args.out, hashes, info)
        print("PASS"
              f" events={info['event_count']}"
              f" inserts={info['insert_count']}"
              f" knn={info['knn_count']}"
              f" final_active={info['final_active_count']}"
              f" final_hash={info['final_active_set_sha256']}"
              f" out={args.out}")
        return 0
    except Fail as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

