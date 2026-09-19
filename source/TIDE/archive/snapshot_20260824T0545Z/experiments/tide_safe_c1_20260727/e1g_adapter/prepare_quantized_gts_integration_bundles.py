#!/usr/bin/env python3
"""Build E1-GI-B quantized GTS-integration input bundles from E1-G0 traces.

This is CPU-only input preparation for a future C++ base-tree runner.  It is
not a GTS execution or performance result.  Coordinates are quantized using
one fixed, documented transform: int16(rint(float * 100)), with no clipping.
The event bytes are preserved exactly; the trace header's range radius is
converted into quantized-coordinate units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"E1GTRC01"
VERSION = 1
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
OP_BY_CODE = {1: "insert", 2: "delete", 3: "knn", 4: "range"}
SCALE = 100.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_set_sha256(active: set[int]) -> str:
    return hashlib.sha256("".join(f"{sid}\n" for sid in sorted(active)).encode("ascii")).hexdigest()


def quantize(array: np.ndarray, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    if not np.isfinite(array).all():
        raise ValueError(f"{label}: non-finite input cannot be quantized")
    rounded = np.rint(array.astype(np.float64, copy=False) * SCALE)
    info = np.iinfo(np.int16)
    lo, hi = int(rounded.min()), int(rounded.max())
    if lo < info.min or hi > info.max:
        raise OverflowError(f"{label}: scale={SCALE} gives [{lo},{hi}] outside int16 [{info.min},{info.max}]; refusing to clip")
    out = np.asarray(rounded, dtype="<i2", order="C")
    stats = {
        "float_min": float(np.min(array)), "float_max": float(np.max(array)),
        "quantized_min": lo, "quantized_max": hi, "overflow": False,
    }
    return out, stats


def parse_trace(source: Path, metadata: dict[str, Any]) -> tuple[tuple[Any, ...], bytes, list[tuple[int, int, int]]]:
    blob = (source / "trace.e1gtrc").read_bytes()
    if len(blob) < HEADER.size:
        raise ValueError(f"{source}: trace shorter than fixed header")
    header = HEADER.unpack_from(blob, 0)
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = header
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"{source}: unsupported trace magic/version")
    expected = HEADER.size + EVENT.size * event_count
    if len(blob) != expected:
        raise ValueError(f"{source}: trace size {len(blob)} != {expected}")
    meta_header = metadata.get("header", {})
    for name, observed in {"dim": dim, "base_n": base_n, "reservoir_n": reservoir_n, "pool_n": pool_n, "query_n": query_n, "k": k, "event_count": event_count}.items():
        if name in meta_header and int(meta_header[name]) != int(observed):
            raise ValueError(f"{source}: metadata/header mismatch for {name}")
    events: list[tuple[int, int, int]] = []
    for pos in range(event_count):
        op_index, opcode, argument = EVENT.unpack_from(blob, HEADER.size + pos * EVENT.size)
        if opcode not in OP_BY_CODE:
            raise ValueError(f"{source}: unknown op code {opcode} at event {pos}")
        if op_index != pos:
            raise ValueError(f"{source}: noncontiguous op_index {op_index} at event {pos}")
        events.append((int(op_index), int(opcode), int(argument)))
    return header, blob[HEADER.size:], events


def validate_stable_contract(events: list[tuple[int, int, int]], base_n: int, reservoir_n: int, pool_n: int, query_n: int) -> dict[str, int]:
    active = set(range(base_n))
    seen_insert: set[int] = set()
    counts = {"insert": 0, "delete": 0, "knn": 0, "range": 0}
    for oi, code, arg in events:
        op = OP_BY_CODE[code]
        if op == "insert":
            if not (base_n <= arg < base_n + reservoir_n <= pool_n):
                raise ValueError(f"trace op {oi}: insert {arg} is not in the disjoint reservoir")
            if arg in active or arg in seen_insert:
                raise ValueError(f"trace op {oi}: duplicate/reinserted stable ID {arg}")
            active.add(arg); seen_insert.add(arg)
        elif op == "delete":
            if arg not in active:
                raise ValueError(f"trace op {oi}: delete non-live stable ID {arg}")
            active.remove(arg)
        else:
            if not (0 <= arg < query_n):
                raise ValueError(f"trace op {oi}: query ID {arg} outside [0,{query_n})")
        counts[op] += 1
    counts["final_active_count"] = len(active)
    return counts


def exact(pool: np.ndarray, query: np.ndarray, active: set[int], op: str, radius: float, k: int) -> list[list[float | int]]:
    ids = np.fromiter(sorted(active), dtype=np.int64)
    # Avoid int16/int32 overflow: all differences/sums are int64, then sqrt float64.
    delta = pool[ids].astype(np.int64, copy=False) - query.astype(np.int64, copy=False)
    distance = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.int64).astype(np.float64))
    if op == "knn":
        chosen = np.lexsort((ids, distance))[: min(k, len(ids))]
    elif op == "range":
        keep = np.flatnonzero(distance <= radius + 1e-12)
        chosen = keep[np.lexsort((ids[keep], distance[keep]))] if len(keep) else np.empty(0, dtype=np.int64)
    else:
        raise ValueError(op)
    return [[int(ids[i]), float(distance[i])] for i in chosen]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def quantized_oracle(pool: np.ndarray, queries: np.ndarray, events: list[tuple[int, int, int]], base_n: int, radius: float, k: int) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    active = set(range(base_n))
    rows: list[dict[str, Any]] = []
    count = {"update": 0, "knn": 0, "range": 0}
    for oi, code, arg in events:
        op = OP_BY_CODE[code]
        if op in {"insert", "delete"}:
            if op == "insert": active.add(arg)
            else: active.remove(arg)
            rows.append({"record": "update", "op_index": oi, "op": op, "stable_id": arg, "active_count": len(active), "verifier_active_set_sha256": stable_set_sha256(active)})
            count["update"] += 1
        else:
            rows.append({"record": "query", "op_index": oi, "kind": op, "query_id": arg, "results": exact(pool, queries[arg], active, op, radius, k), "active_count": len(active), "verifier_active_set_sha256": stable_set_sha256(active)})
            count[op] += 1
    return rows, count, stable_set_sha256(active)


def copy_required(source: Path, destination: Path, name: str) -> None:
    input_path = source / name
    if input_path.is_file():
        shutil.copyfile(input_path, destination / name)


def prepare_one(source: Path, destination: Path, script_sha: str) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    original_header, event_bytes, events = parse_trace(source, metadata)
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, original_radius, event_count = original_header
    del magic, version
    if event_count != len(events): raise AssertionError("event count")
    raw_pool = np.fromfile(source / "pool.f32", dtype="<f4")
    raw_queries = np.fromfile(source / "queries.f32", dtype="<f4")
    if len(raw_pool) != pool_n * dim or len(raw_queries) != query_n * dim:
        raise ValueError(f"{source}: payload shape mismatch")
    pool_float = raw_pool.reshape(pool_n, dim)
    query_float = raw_queries.reshape(query_n, dim)
    pool_i16, pool_stats = quantize(pool_float, "pool")
    query_i16, query_stats = quantize(query_float, "queries")
    trace_counts = validate_stable_contract(events, base_n, reservoir_n, pool_n, query_n)
    qradius = float(np.float32(float(original_radius) * SCALE))
    destination.mkdir(parents=True)
    (destination / "pool.i16").write_bytes(pool_i16.tobytes(order="C"))
    (destination / "queries.i16").write_bytes(query_i16.tobytes(order="C"))
    # Preserve operation records byte-for-byte; only the radius has unit conversion.
    qheader = HEADER.pack(MAGIC, VERSION, dim, base_n, reservoir_n, pool_n, query_n, k, np.float32(qradius), event_count)
    (destination / "trace.e1gtrc").write_bytes(qheader + event_bytes)
    copy_required(source, destination, "initial_base_stable_ids.i32")
    copy_required(source, destination, "stable_id_to_pool_row.i32")
    expected_rows, oracle_counts, final_active_hash = quantized_oracle(pool_i16, query_i16, events, base_n, qradius, k)
    write_jsonl(destination / "quantized_oracle_expected.jsonl", expected_rows)
    event_sha = hashlib.sha256(event_bytes).hexdigest()
    metadata_out: dict[str, Any] = {
        "schema": "e1gi-b-quantized-gts-integration-bundle-v1",
        "scope": "CPU-prepared input for a future C++ GTS base-tree replay; NOT a GTS integration or performance result",
        "gpu_used": False,
        "quantization": {
            "scale": SCALE,
            "formula": "int16(rint(float64(x) * 100.0))",
            "rounding": "IEEE-754 round-to-nearest, ties-to-even (numpy.rint)",
            "overflow_policy": "reject; no clipping or saturation permitted",
            "coordinate_type": "little-endian signed int16",
            "pool": pool_stats,
            "queries": query_stats,
        },
        "metric_contract": {
            "metric": "L2",
            "coordinate_space": "quantized int16 coordinates",
            "distance_arithmetic": "int64 squared-difference accumulation then float64 sqrt in independent oracle/verifier",
            "distance_units": "quantized-coordinate units",
            "range_radius_original_float": float(original_radius),
            "range_radius_quantized": qradius,
            "k": k,
            "range_inclusive": True,
            "topk_tie_break": "(distance, stable_id)",
        },
        "trace_binary_format": {
            "endianness": "little",
            "header_struct": "<8sI6IfQ", "header_bytes": HEADER.size,
            "magic_ascii": MAGIC.decode("ascii"), "version": VERSION,
            "event_struct": "<IB3xi", "event_bytes": EVENT.size,
            "op_codes": {"1": "insert", "2": "delete", "3": "knn", "4": "range"},
            "event_argument": "stable_id for insert/delete; query_id for knn/range",
        },
        "header": {"magic": MAGIC.decode("ascii"), "version": VERSION, "dim": dim, "base_n": base_n, "reservoir_n": reservoir_n, "pool_n": pool_n, "query_n": query_n, "k": k, "radius": qradius, "event_count": event_count},
        "stable_id_contract": {
            "initial_base_ids": "[0, base_n)", "reservoir_ids": "[base_n, base_n + reservoir_n)",
            "mapping": "identity stable_id_to_pool_row.i32", "delete_semantics": "stable ID, never logical rank",
            "query_semantics": "external query_id into queries.i16",
        },
        "trace_counts": trace_counts,
        "event_stream": {"source_event_stream_sha256": event_sha, "quantized_event_stream_sha256": event_sha, "byte_identical": True},
        "source_prepared_bundle": {"path": str(source), "metadata_sha256": sha256_file(source / "metadata.json"), "trace_sha256": sha256_file(source / "trace.e1gtrc"), "manifest_sha256": sha256_file(source / "manifest.json") if (source / "manifest.json").is_file() else None},
        "payload_files": {"pool": "pool.i16", "queries": "queries.i16", "trace": "trace.e1gtrc", "oracle": "quantized_oracle_expected.jsonl", "initial_base_ids": "initial_base_stable_ids.i32", "stable_id_mapping": "stable_id_to_pool_row.i32"},
        "quantized_oracle": {"file": "quantized_oracle_expected.jsonl", "query_counts": oracle_counts, "final_active_count": trace_counts["final_active_count"], "verifier_active_set_sha256": final_active_hash, "note": "independent exact oracle over quantized vectors; not an engine result"},
        "preparation_script_sha256": script_sha,
    }
    (destination / "metadata.json").write_text(json.dumps(metadata_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = sorted(p for p in destination.iterdir() if p.is_file())
    manifest = {"schema": "e1gi-b-quantized-gts-integration-manifest-v1", "status": "prepared_input_not_gts_result", "gpu_used": False, "files_sha256": {p.name: sha256_file(p) for p in files}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"source": str(source), "output": str(destination), "events": event_count, "dim": dim, "trace_counts": trace_counts, "oracle_counts": oracle_counts, "qradius": qradius, "event_bytes_identical": True}


def discover(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "metadata.json").is_file() and (p / "trace.e1gtrc").is_file() and (p / "pool.f32").is_file() and (p / "queries.f32").is_file())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-root", type=Path, help="directory containing prepared E1-G0 child bundles")
    p.add_argument("--input", action="append", type=Path, help="single prepared E1-G0 bundle; repeatable")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    sources = list(args.input or [])
    if args.prepared_root:
        sources.extend(discover(args.prepared_root.resolve()))
    sources = list(dict.fromkeys(x.resolve() for x in sources))
    if not sources:
        raise SystemExit("provide --prepared-root or --input")
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite output root: {out}")
    out.mkdir(parents=True)
    script_sha = sha256_file(Path(__file__).resolve())
    results: list[dict[str, Any]] = []
    try:
        for source in sources:
            results.append(prepare_one(source, out / source.name, script_sha))
    except Exception:
        (out / "FAILED.txt").write_text("quantized batch failed; do not treat partial outputs as complete\n", encoding="utf-8")
        raise
    batch = {"schema": "e1gi-b-quantized-gts-integration-batch-v1", "scope": "prepared inputs for future C++ GTS base-tree replay; NOT GTS results", "gpu_used": False, "scale": SCALE, "prepared_count": len(results), "results": results, "preparation_script_sha256": script_sha}
    (out / "batch_metadata.json").write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "prepared_count": len(results), "scale": SCALE, "gpu_used": False, "scope": "E1-GI-B prepared inputs only"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
