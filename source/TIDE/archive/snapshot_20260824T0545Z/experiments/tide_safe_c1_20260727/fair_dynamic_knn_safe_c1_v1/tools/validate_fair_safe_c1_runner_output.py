#!/usr/bin/python3.12
"""Pure-stdlib independent CPU validator for Safe-C1 delta-only output.

The validator replays the sealed frozen-base/insertion-only/KNN-only projection
from raw bundle bytes.  It uses exact integer squared L2 and requires the
runner to declare its typed full-immutable-base correctness fallback.  It does
not import NumPy, the archived GTS environment, the source-provenance oracle
JSONL, or runner-reported answers as an oracle.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

ADMISSION_SCHEMA = "e1-frozen-base-knn-projection-admission-v2"
MAGIC = b"E1GTRC01"
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
INSERT, KNN = 1, 3


class Fail(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Fail(message)


def safe_bytes(path: Path, label: str) -> bytes:
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing {label}: {path}")
    return path.read_bytes()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_le_array(path: Path, typecode: str, width: int, label: str) -> array.array:
    raw = safe_bytes(path, label)
    require(len(raw) % width == 0, f"{label} byte count is not a multiple of {width}")
    values = array.array(typecode)
    require(values.itemsize == width, f"host {typecode!r} width is not {width}")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def active_hash(active: bytearray) -> str:
    return hashlib.sha256(
        "".join(f"{stable}\n" for stable, present in enumerate(active) if present).encode("ascii")
    ).hexdigest()


def active_count(active: bytearray) -> int:
    return active.count(1)


def read_env(path: Path) -> dict[str, str]:
    raw = safe_bytes(path, "admission").decode("ascii")
    values: dict[str, str] = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        require("=" in line and line.index("=") > 0, "malformed admission line")
        key, value = line.split("=", 1)
        require(key not in values and value, f"invalid duplicated/empty admission key {key}")
        values[key] = value
    return values


def need(env: dict[str, str], key: str) -> str:
    require(key in env, f"admission omits {key}")
    return env[key]


def as_nonnegative(env: dict[str, str], key: str) -> int:
    value = need(env, key)
    require(value.isdecimal(), f"admission integer {key} is not decimal")
    return int(value)


def parse_bundle(bundle: Path) -> tuple[dict[str, int], list[tuple[int, int, int]], array.array,
                                         array.array, array.array, array.array]:
    raw = safe_bytes(bundle / "trace.e1gtrc", "trace")
    require(len(raw) >= HEADER.size, "truncated trace")
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, _radius, event_count = HEADER.unpack_from(raw)
    require(magic == MAGIC and version == 1, "trace magic/version mismatch")
    require(dim > 0 and base_n > 0 and pool_n >= base_n and reservoir_n == pool_n - base_n and
            query_n > 0 and k > 0, "invalid trace header")
    require(len(raw) == HEADER.size + event_count * EVENT.size, "trace byte count mismatch")
    events: list[tuple[int, int, int]] = []
    for offset in range(event_count):
        index, opcode, argument = EVENT.unpack_from(raw, HEADER.size + offset * EVENT.size)
        require(index == offset and opcode in (INSERT, KNN), f"illegal projection event at {offset}")
        if opcode == INSERT:
            require(base_n <= argument < pool_n, f"projection insert outside mutable reservoir at {offset}")
        else:
            require(0 <= argument < query_n, f"projection KNN query outside matrix at {offset}")
        events.append((index, opcode, argument))

    pool = read_le_array(bundle / "pool.i16", "h", 2, "pool")
    queries = read_le_array(bundle / "queries.i16", "h", 2, "queries")
    mapping = read_le_array(bundle / "stable_id_to_pool_row.i32", "i", 4, "mapping")
    base = read_le_array(bundle / "initial_base_stable_ids.i32", "i", 4, "base IDs")
    require(len(pool) == pool_n * dim and len(queries) == query_n * dim and
            len(mapping) == pool_n and len(base) == base_n, "bundle matrix shape mismatch")
    require(len(set(mapping)) == pool_n and min(mapping) >= 0 and max(mapping) < pool_n,
            "stableID->pool-row mapping is not bijective")
    require(len(set(base)) == base_n and min(base) >= 0 and max(base) < pool_n,
            "invalid initial base IDs")
    return {
        "dimension": dim,
        "base_n": base_n,
        "pool_n": pool_n,
        "query_n": query_n,
        "k": k,
        "event_count": event_count,
    }, events, pool, queries, mapping, base


def canonical_topk(pool: array.array, mapping: array.array, queries: array.array,
                   query_id: int, active: bytearray, dimension: int, k: int) -> list[list[int]]:
    require(0 <= query_id and (query_id + 1) * dimension <= len(queries), "query ID outside matrix")
    q_offset = query_id * dimension
    best: list[tuple[int, int]] = []  # (distance_sq, stable_id), ascending canonical order
    for stable, present in enumerate(active):
        if not present:
            continue
        row = int(mapping[stable])
        p_offset = row * dimension
        distance_sq = 0
        for coordinate in range(dimension):
            delta = int(pool[p_offset + coordinate]) - int(queries[q_offset + coordinate])
            distance_sq += delta * delta
        item = (distance_sq, stable)
        if len(best) < k:
            best.append(item)
            if len(best) == k:
                best.sort()
        elif item < best[-1]:
            best[-1] = item
            best.sort()
    require(len(best) == k, "active population is smaller than k")
    return [[stable, distance_sq] for distance_sq, stable in best]


def parse_rows(value: Any, label: str, k: int) -> list[list[int]]:
    require(isinstance(value, list) and len(value) == k, f"{label} must contain exactly k results")
    rows: list[list[int]] = []
    prior: tuple[int, int] | None = None
    seen: set[int] = set()
    for index, row in enumerate(value):
        require(isinstance(row, list) and len(row) == 2, f"{label}[{index}] is not [stable,distance]")
        stable, distance = row
        require(isinstance(stable, int) and not isinstance(stable, bool) and stable >= 0,
                f"{label}[{index}] has invalid stable ID")
        require(isinstance(distance, int) and not isinstance(distance, bool) and distance >= 0,
                f"{label}[{index}] has invalid distance")
        require(stable not in seen, f"{label} has duplicate stable ID")
        seen.add(stable)
        key = (distance, stable)
        require(prior is None or prior < key, f"{label} is not canonical (distance,stable) order")
        prior = key
        rows.append([stable, distance])
    return rows


def load_records(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSONL: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(line.endswith("\n"), f"JSONL line {line_number} lacks LF terminator")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise Fail(f"invalid JSONL at line {line_number}: {error}") from error
            require(isinstance(row, dict), f"JSONL line {line_number} is not an object")
            records.append(row)
    return records


def validate_admission_and_bytes(bundle: Path, admission: Path) -> dict[str, str]:
    env = read_env(admission)
    require(need(env, "schema") == ADMISSION_SCHEMA and need(env, "status") == "PASS",
            "admission schema/status mismatch")
    require(Path(need(env, "bundle_realpath")).resolve() == bundle.resolve(),
            "admission bundle path mismatch")
    require(need(env, "ops") == "insert,knn" and need(env, "excluded_ops") == "delete,range",
            "admission policy mismatch")
    required_hashes = {
        "manifest.json": "manifest_sha256",
        "metadata.json": "metadata_sha256",
        "trace.e1gtrc": "trace_sha256",
        "pool.i16": "pool_sha256",
        "queries.i16": "queries_sha256",
        "stable_id_to_pool_row.i32": "mapping_sha256",
        "initial_base_stable_ids.i32": "base_ids_sha256",
    }
    for filename, key in required_hashes.items():
        require(sha256_file(bundle / filename) == need(env, key),
                f"bundle drifted after admission: {filename}")
    return env


def validate(args: argparse.Namespace) -> dict[str, Any]:
    bundle = args.bundle.resolve()
    require(bundle.is_dir() and not bundle.is_symlink(), "bundle must be real directory")
    env = validate_admission_and_bytes(bundle, args.admission)
    header, events, pool, queries, mapping, base_ids = parse_bundle(bundle)
    for key, actual in header.items():
        require(as_nonnegative(env, key) == actual, f"admission/header mismatch: {key}")
    raw_trace = safe_bytes(bundle / "trace.e1gtrc", "trace")
    require(hashlib.sha256(raw_trace[HEADER.size:]).hexdigest() ==
            need(env, "projection_event_stream_sha256"), "event-stream hash mismatch")

    records = load_records(args.output)
    require(len(records) == len(events), "JSONL record count does not equal trace event count")
    active = bytearray(header["pool_n"])
    for stable in base_ids:
        active[int(stable)] = 1
    base_active = bytearray(active)
    inserts = 0
    knn = 0
    sequence = hashlib.sha256()

    for expected_event, record in zip(events, records):
        op_index, opcode, argument = expected_event
        require(record.get("op_index") == op_index, f"output op_index mismatch at {op_index}")
        if opcode == INSERT:
            require(record.get("record") == "update" and record.get("op") == "insert",
                    f"insert record shape mismatch at {op_index}")
            require(record.get("stable_id") == argument and
                    record.get("placement") == "global_delta" and
                    record.get("base_mutated") is False,
                    f"insert semantics mismatch at {op_index}")
            require(0 <= argument < header["pool_n"] and not active[argument],
                    f"trace insertion invalid at {op_index}")
            active[argument] = 1
            inserts += 1
            continue

        require(record.get("record") == "knn" and record.get("query_id") == argument and
                record.get("k") == header["k"], f"KNN record shape mismatch at {op_index}")
        require(record.get("base_path") == "exact_full_immutable_base_range_fallback_no_gts_receipt" and
                record.get("base_query_mode") == "exact_full_immutable_base_range_fallback" and
                record.get("base_exact_full_immutable_fallback") is True and
                record.get("full_immutable_base_candidate_count") == header["base_n"] and
                record.get("native_gts_knn_used_for_trace") is False,
                f"KNN exact-base fallback declaration mismatch at {op_index}")
        require(record.get("base_immutable") is True and
                record.get("direct_sidecar_used") is False and
                record.get("global_delta_live") == inserts and
                "native_base" not in record,
                f"KNN Safe-C1 flags mismatch at {op_index}")
        actual_base = parse_rows(record.get("base_exact_topk"), f"base_exact_topk event {op_index}", header["k"])
        actual_merged = parse_rows(record.get("merged"), f"merged event {op_index}", header["k"])
        expected_base = canonical_topk(pool, mapping, queries, argument, base_active,
                                       header["dimension"], header["k"])
        expected_full = canonical_topk(pool, mapping, queries, argument, active,
                                       header["dimension"], header["k"])
        require(actual_base == expected_base, f"exact immutable-base top-k oracle mismatch at {op_index}")
        require(actual_merged == expected_full, f"merged int64 oracle mismatch at {op_index}")
        sequence.update(f"{op_index}:{argument}:{active_hash(active)}:".encode("ascii"))
        for stable, distance in expected_full:
            sequence.update(f"{stable}:{distance},".encode("ascii"))
        sequence.update(b"\n")
        knn += 1

    final_hash = active_hash(active)
    require(inserts == as_nonnegative(env, "insert_count") and
            knn == as_nonnegative(env, "knn_count"), "output operation counts mismatch admission")
    require(active_count(active) == as_nonnegative(env, "final_active_count") and
            final_hash == need(env, "final_active_set_sha256"), "output final active witness mismatch admission")

    require(args.summary.is_file() and not args.summary.is_symlink(), "unsafe/missing summary")
    try:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise Fail(f"invalid summary JSON: {error}") from error
    require(isinstance(summary, dict), "summary is not an object")
    require(summary.get("schema") == "fair-safe-c1-delta-knn-correctness-v3" and
            summary.get("mode") == "correctness" and summary.get("status") == "PASS",
            "summary schema/mode/status mismatch")
    checks: dict[str, Any] = {
        "event_count": len(events),
        "insert_count": inserts,
        "knn_count": knn,
        "global_delta_live": inserts,
        "final_active_count": active_count(active),
        "final_active_set_sha256": final_hash,
        "metadata_sha256": need(env, "metadata_sha256"),
        "trace_sha256": need(env, "trace_sha256"),
        "projection_event_stream_sha256": need(env, "projection_event_stream_sha256"),
    }
    for key, expected in checks.items():
        require(summary.get(key) == expected, f"summary mismatch for {key}")
    require(summary.get("base_immutable") is True and
            summary.get("base_query_mode") == "exact_full_immutable_base_range_fallback" and
            summary.get("base_exact_full_immutable_fallback") is True and
            summary.get("native_gts_knn_used_for_trace") is False and
            summary.get("direct_sidecar_used") is False and
            summary.get("legacy_routing_used") is False and
            summary.get("timing_claim") is False,
            "summary safety flags mismatch")

    return {
        "schema": "fair-safe-c1-delta-knn-output-validator-v3",
        "status": "PASS",
        "bundle": str(bundle),
        "output": str(args.output.resolve()),
        "trace_sha256": need(env, "trace_sha256"),
        "metadata_sha256": need(env, "metadata_sha256"),
        "validated_events": len(events),
        "insert": inserts,
        "knn": knn,
        "final_active_count": active_count(active),
        "final_active_set_sha256": final_hash,
        "canonical_int64_knn_sequence_sha256": sequence.hexdigest(),
        "oracle": "pure-stdlib exact integer squared L2; canonical order=(distance_sq,stable_id); independently checks typed immutable-base fallback output replay",
    }


def write_once(path: Path, result: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite validator output: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".output-validator.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="optional new validation JSON")
    args = parser.parse_args()
    try:
        result = validate(args)
        if args.out is not None:
            write_once(args.out.resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Fail as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
