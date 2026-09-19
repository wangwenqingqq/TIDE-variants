#!/usr/bin/env python3
"""Seal an ordinal-only calibration/test split for the Safe-C1 HNSW protocol.

CPU-only, pure stdlib, and intentionally minimal.  The membership rule uses
only KNN ordinal in the frozen projection trace: zero-based even ordinals are
calibration and odd ordinals are held-out.  Native-GTS recall fields are read
only after membership is fixed to report the requested aggregate targets.
This program neither imports nor executes HNSW, CUDA, GTS, NVML, or a
user-owned Python environment.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAGIC = b"E1GTRC01"
VERSION = 1
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
INSERT, KNN = 1, 3
SPLIT_SCHEMA = "safe-c1-hnsw-calibration-test-split-v1"


class Fail(RuntimeError):
    pass


@dataclass(frozen=True)
class QueryObservation:
    knn_ordinal: int
    op_index: int
    query_id: int
    global_delta_live: int
    base_overlap_count: int
    merged_overlap_count: int
    base_exact_match: bool
    merged_exact_match: bool


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Fail(message)


def regular_nonsymlink(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise Fail(f"missing {label}: {path}") from exc
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode),
            f"{label} must be a regular non-symlink: {path}")


def real_directory(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise Fail(f"missing {label}: {path}") from exc
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode),
            f"{label} must be a real directory: {path}")
    return path.resolve()


def sha256_file(path: Path, label: str) -> str:
    regular_nonsymlink(path, label)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    regular_nonsymlink(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Fail(f"invalid {label} JSON: {exc}") from exc
    require(isinstance(value, dict), f"{label} JSON root must be an object")
    return value


def is_nonbool_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_nonbool_int(value: Any, label: str, minimum: int = 0) -> int:
    require(is_nonbool_int(value) and value >= minimum,
            f"{label} must be an integer >= {minimum}")
    return int(value)


def require_bool(value: Any, label: str) -> bool:
    require(isinstance(value, bool), f"{label} must be boolean")
    return value


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(ch in "0123456789abcdef" for ch in value),
            f"{label} must be a lowercase SHA-256 digest")
    return value


def parse_projection(bundle: Path) -> tuple[dict[str, Any], dict[str, Any], list[tuple[int, int, int]], str, str]:
    bundle = real_directory(bundle, "projection bundle")
    manifest_path = bundle / "manifest.json"
    metadata_path = bundle / "metadata.json"
    trace_path = bundle / "trace.e1gtrc"
    manifest = read_json_object(manifest_path, "projection manifest")
    metadata = read_json_object(metadata_path, "projection metadata")
    require(manifest.get("schema") == "e1-frozen-base-knn-projection-manifest-v1",
            "unexpected projection manifest schema")
    require(metadata.get("schema") == "e1-frozen-base-knn-projection-bundle-v1",
            "unexpected projection metadata schema")
    files = manifest.get("files_sha256")
    require(isinstance(files, dict), "manifest.files_sha256 must be an object")
    for name in ("metadata.json", "trace.e1gtrc"):
        expected = require_sha256(files.get(name), f"manifest.files_sha256[{name!r}]")
        actual = sha256_file(bundle / name, f"projection {name}")
        require(actual == expected, f"manifest hash mismatch for {name}")
    trace_sha = sha256_file(trace_path, "projection trace")
    projection = metadata.get("projection")
    require(isinstance(projection, dict), "metadata.projection must be an object")
    require(trace_sha == require_sha256(projection.get("projection_trace_sha256"),
                                        "metadata projection trace sha"),
            "metadata projection trace hash does not bind trace bytes")
    require(trace_sha == require_sha256(manifest.get("projection_trace_sha256"),
                                        "manifest projection trace sha"),
            "manifest projection trace hash does not bind trace bytes")
    require(metadata.get("scope") ==
            "Safe-C1 immutable-base KNN gate only; not the full E1 workload, not a base-delete, range, rebuild, direct-sidecar, or performance claim.",
            "projection scope drift")
    require(projection.get("allowed_ops") == ["insert", "knn"],
            "projection must permit only insert and knn")
    raw = trace_path.read_bytes()
    require(len(raw) >= HEADER.size, "projection trace shorter than header")
    magic, version, dimension, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = HEADER.unpack_from(raw)
    del dimension, base_n, reservoir_n, pool_n, radius
    require(magic == MAGIC and version == VERSION, "projection trace magic/version mismatch")
    require(is_nonbool_int(query_n) and query_n > 0, "projection query_n invalid")
    require(is_nonbool_int(k) and k > 0, "projection k invalid")
    require(is_nonbool_int(event_count) and event_count > 0, "projection event_count invalid")
    require(len(raw) == HEADER.size + event_count * EVENT.size,
            "projection trace byte count/event_count mismatch")
    events: list[tuple[int, int, int]] = []
    for position in range(event_count):
        op_index, opcode, argument = EVENT.unpack_from(raw, HEADER.size + position * EVENT.size)
        require(op_index == position, f"projection trace op_index not contiguous at {position}")
        require(opcode in {INSERT, KNN}, f"projection trace has non-projected opcode at {position}")
        if opcode == KNN:
            require(0 <= argument < query_n, f"KNN query id out of range at {position}")
        events.append((op_index, opcode, argument))
    header = metadata.get("header")
    require(isinstance(header, dict), "metadata.header must be an object")
    for key, actual in (("query_n", query_n), ("k", k), ("event_count", event_count)):
        require(header.get(key) == actual, f"metadata/header trace mismatch for {key}")
    return manifest, metadata, events, trace_sha, sha256_file(metadata_path, "projection metadata")


def parse_native_recall(engine_path: Path, validator_path: Path,
                        events: list[tuple[int, int, int]], trace_sha: str,
                        metadata_sha: str, k: int) -> tuple[list[QueryObservation], dict[str, str]]:
    regular_nonsymlink(engine_path, "Native-GTS recall JSONL")
    validator = read_json_object(validator_path, "Native-GTS independent validator")
    require(validator.get("schema") == "fair-safe-c1-native-candidate-recall-output-validator-v1",
            "unexpected Native-GTS validator schema")
    require(validator.get("status") == "PASS", "Native-GTS validator is not PASS")
    require(validator.get("trace_sha256") == trace_sha,
            "Native-GTS validator trace hash does not match projection")
    require(validator.get("metadata_sha256") == metadata_sha,
            "Native-GTS validator metadata hash does not match projection")
    require(validator.get("output") == str(engine_path.resolve()),
            "Native-GTS validator output path does not bind supplied JSONL")
    try:
        lines = engine_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise Fail(f"cannot read Native-GTS recall JSONL: {exc}") from exc
    require(len(lines) == len(events),
            "Native-GTS recall JSONL record count does not match projection trace")
    observations: list[QueryObservation] = []
    global_delta_live = 0
    for position, ((op_index, opcode, argument), line) in enumerate(zip(events, lines)):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Fail(f"invalid Native-GTS JSONL record at line {position + 1}: {exc}") from exc
        require(isinstance(record, dict), f"Native-GTS JSONL record {position} is not an object")
        require(record.get("op_index") == op_index,
                f"Native-GTS JSONL op_index mismatch at trace event {position}")
        if opcode == INSERT:
            require(record.get("record") == "update" and record.get("op") == "insert",
                    f"Native-GTS insertion record mismatch at op {op_index}")
            require(record.get("stable_id") == argument,
                    f"Native-GTS inserted stable_id mismatch at op {op_index}")
            require(record.get("placement") == "global_delta" and record.get("base_mutated") is False,
                    f"Native-GTS insertion violates frozen-base/global-delta scope at op {op_index}")
            global_delta_live += 1
            continue
        require(record.get("record") == "knn", f"Native-GTS KNN record mismatch at op {op_index}")
        require(record.get("query_id") == argument, f"Native-GTS query id mismatch at op {op_index}")
        require(record.get("k") == k, f"Native-GTS K mismatch at op {op_index}")
        require(record.get("global_delta_live") == global_delta_live,
                f"Native-GTS delta cardinality mismatch at op {op_index}")
        require(record.get("base_immutable") is True and record.get("direct_sidecar_used") is False,
                f"Native-GTS scope flags drift at op {op_index}")
        require(record.get("native_gts_candidate_path_used_for_trace") is True and
                record.get("typed_exact_full_immutable_base_fallback_used_for_trace") is False,
                f"Native-GTS path flags drift at op {op_index}")
        base_overlap = require_nonbool_int(record.get("base_overlap_count"),
                                           f"base_overlap_count at op {op_index}")
        merged_overlap = require_nonbool_int(record.get("merged_overlap_count"),
                                             f"merged_overlap_count at op {op_index}")
        require(base_overlap <= k and merged_overlap <= k,
                f"Native-GTS overlap exceeds K at op {op_index}")
        observations.append(QueryObservation(
            knn_ordinal=len(observations),
            op_index=op_index,
            query_id=argument,
            global_delta_live=global_delta_live,
            base_overlap_count=base_overlap,
            merged_overlap_count=merged_overlap,
            base_exact_match=require_bool(record.get("base_exact_match"),
                                          f"base_exact_match at op {op_index}"),
            merged_exact_match=require_bool(record.get("merged_exact_match"),
                                            f"merged_exact_match at op {op_index}"),
        ))
    require(observations, "projection contains no KNN records")
    totals = {
        "knn": len(observations),
        "base_overlap_sum": sum(item.base_overlap_count for item in observations),
        "merged_overlap_sum": sum(item.merged_overlap_count for item in observations),
        "base_exact_match_count": sum(item.base_exact_match for item in observations),
        "merged_exact_match_count": sum(item.merged_exact_match for item in observations),
    }
    for key, value in totals.items():
        require(validator.get(key) == value,
                f"Native-GTS validator {key} disagrees with JSONL-derived metric")
    require(validator.get("validated_events") == len(events),
            "Native-GTS validator event count mismatch")
    return observations, {
        "native_engine_jsonl_sha256": sha256_file(engine_path, "Native-GTS recall JSONL"),
        "native_independent_validation_json_sha256": sha256_file(
            validator_path, "Native-GTS independent validator"),
    }


def ordinal_split(observations: list[QueryObservation]) -> tuple[list[QueryObservation], list[QueryObservation]]:
    """Membership is deliberately independent of every Native-GTS quality field."""
    calibration = [item for item in observations if item.knn_ordinal % 2 == 0]
    held_out = [item for item in observations if item.knn_ordinal % 2 == 1]
    require(len(calibration) + len(held_out) == len(observations), "split lost a KNN")
    require(not {item.knn_ordinal for item in calibration} & {item.knn_ordinal for item in held_out},
            "calibration/held-out ordinal overlap")
    require(len(calibration) == len(held_out),
            "this sealed trace must yield an equal even/odd KNN split")
    return calibration, held_out


def rate(numerator: int, denominator: int) -> dict[str, Any]:
    require(denominator > 0, "zero rate denominator")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "decimal_12": f"{numerator / denominator:.12f}",
    }


def native_target(items: list[QueryObservation], k: int) -> dict[str, Any]:
    return {
        "knn_count": len(items),
        "k": k,
        "base_overlap": rate(sum(item.base_overlap_count for item in items), len(items) * k),
        "merged_overlap": rate(sum(item.merged_overlap_count for item in items), len(items) * k),
        "base_exact_result_sets": rate(sum(item.base_exact_match for item in items), len(items)),
        "merged_exact_result_sets": rate(sum(item.merged_exact_match for item in items), len(items)),
    }


def render_members(items: list[QueryObservation]) -> list[dict[str, int]]:
    return [{
        "knn_ordinal": item.knn_ordinal,
        "op_index": item.op_index,
        "query_id": item.query_id,
        "global_delta_live": item.global_delta_live,
    } for item in items]


def membership_sha256(items: list[QueryObservation]) -> str:
    payload = "".join(
        f"{item.knn_ordinal}\t{item.op_index}\t{item.query_id}\n"
        for item in items
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def write_root_owned_new_file(output: Path, value: dict[str, Any]) -> None:
    require(os.geteuid() == 0, "must run as root so the sealed split is root-owned")
    parent = real_directory(output.parent, "output parent")
    require(output.parent.resolve() == parent, "output parent resolution drift")
    try:
        existing = output.lstat()
    except FileNotFoundError:
        existing = None
    require(existing is None, f"refusing to overwrite existing split: {output}")
    payload = canonical_json_bytes(value)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chown(temporary, 0, 0)
        os.replace(temporary, output)
        os.chown(output, 0, 0)
        os.chmod(output, 0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    final_stat = output.lstat()
    require(stat.S_ISREG(final_stat.st_mode) and not stat.S_ISLNK(final_stat.st_mode),
            "written split is not a regular file")
    require(final_stat.st_uid == 0 and final_stat.st_gid == 0 and
            stat.S_IMODE(final_stat.st_mode) == 0o600,
            "written split is not root:root mode 0600")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="CPU-only ordinal-only HNSW calibration/test split derivation; no HNSW/GPU execution.")
    parser.add_argument("--bundle", type=Path,
                        default=root / "inputs/e1_frozen_base_knn_projection_v1")
    parser.add_argument("--native-engine", type=Path,
                        default=root / "runs/safe-c1-native-recall-e1-v1-20260729/engine.jsonl")
    parser.add_argument("--native-validator", type=Path,
                        default=root / "runs/safe-c1-native-recall-e1-v1-20260729/independent_validation.json")
    parser.add_argument("--output", type=Path,
                        default=root / "inputs/hnsw_frozen_base_calibration_test_split_v1.json")
    args = parser.parse_args()
    require(args.output.is_absolute(), "output path must be absolute")
    manifest, metadata, events, trace_sha, metadata_sha = parse_projection(args.bundle)
    k = require_nonbool_int(metadata["header"].get("k"), "projection K", minimum=1)
    observations, native_hashes = parse_native_recall(
        args.native_engine, args.native_validator, events, trace_sha, metadata_sha, k)
    calibration, held_out = ordinal_split(observations)
    manifest_sha = sha256_file(args.bundle.resolve() / "manifest.json", "projection manifest")
    payload: dict[str, Any] = {
        "schema": SPLIT_SCHEMA,
        "scope": (
            "Pre-registered CPU-only calibration/test query allocation for a frozen immutable-base, "
            "insertion-only, KNN-only quality comparison. It does not execute HNSW, CUDA, GTS, NVML, "
            "or any user-owned environment; it makes no timing, full-E1, deletion, range, rebuild, "
            "direct-sidecar, or exactness claim."
        ),
        "selection_policy": {
            "rule": (
                "zero-based KNN ordinal in projected trace order: even ordinal => calibration; "
                "odd ordinal => held_out"
            ),
            "membership_uses_only": (
                "KNN ordinal; neither HNSW output nor Native-GTS overlap/exactness/candidate-count "
                "fields participate in membership allocation"
            ),
            "full_event_binding": (
                "all projected events are parsed and matched one-for-one against the Native-GTS JSONL; "
                "only KNN events are assigned, while preceding insert events determine recorded global_delta_live"
            ),
            "quality_optimization": "none",
        },
        "input_paths": {
            "bundle": str(args.bundle.resolve()),
            "native_engine_jsonl": str(args.native_engine.resolve()),
            "native_independent_validator": str(args.native_validator.resolve()),
        },
        "input_sha256": {
            "bundle_manifest_json_sha256": manifest_sha,
            "bundle_metadata_json_sha256": metadata_sha,
            "projection_trace_sha256": trace_sha,
            "source_trace_sha256": require_sha256(manifest.get("source_trace_sha256"),
                                                  "manifest source trace sha"),
            **native_hashes,
        },
        "projection": {
            "event_count": len(events),
            "knn_count": len(observations),
            "k": k,
            "trace_sha256": trace_sha,
            "metadata_sha256": metadata_sha,
        },
        "membership_sha256_scope": (
            "SHA-256 of LF-delimited ASCII lines in KNN-ordinal order: "
            "knn_ordinal<TAB>op_index<TAB>query_id<LF>."
        ),
        "native_gts_overlap_targets": {
            "calibration": native_target(calibration, k),
            "held_out": native_target(held_out, k),
            "interpretation": (
                "These requested aggregate targets are computed only after the ordinal-only membership "
                "is fixed from the sealed passed Native-GTS candidate-recall JSONL. Select HNSW efSearch "
                "only on calibration merged_overlap; lock it before a single held-out report."
            ),
        },
        "splits": {
            "calibration": {
                "member_count": len(calibration),
                "member_sha256": membership_sha256(calibration),
                "members": render_members(calibration),
            },
            "held_out": {
                "member_count": len(held_out),
                "member_sha256": membership_sha256(held_out),
                "members": render_members(held_out),
            },
        },
        "split_sha256_scope": (
            "SHA-256 of canonical UTF-8 JSON for this object with split_sha256 omitted; "
            "sort_keys=true, separators=(',', ':'), trailing LF."
        ),
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    payload["split_sha256"] = digest
    write_root_owned_new_file(args.output, payload)
    print(f"PASS output={args.output.resolve()}")
    print(f"PASS split_sha256={digest}")
    print(f"PASS calibration_knn={len(calibration)} held_out_knn={len(held_out)}")
    print("PASS membership_rule=even_knn_ordinal_calibration_odd_knn_ordinal_held_out")
    print("PASS no_hnsw_no_gpu_no_user_environment_execution")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Fail as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
