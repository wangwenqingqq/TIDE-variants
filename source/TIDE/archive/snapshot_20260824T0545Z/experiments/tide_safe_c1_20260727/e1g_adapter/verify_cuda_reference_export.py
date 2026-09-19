#!/usr/bin/env python3
"""Independently verify an E1-G0 CUDA Safe-C1 executor export.

Inputs are a prepared compact trace bundle and engine JSONL.  The verifier
replays stable-ID updates from the binary trace and computes exact L2 answers
from immutable pool.f32 / queries.f32 using NumPy.  It never consumes or trusts
an engine-provided summary, recall value, or final-state claim.

Expected engine JSONL records (one per binary event):
  {"record":"update", "op_index":N, "op":"insert|delete", "stable_id":ID}
  {"record":"query", "op_index":N, "kind":"knn|range", "query_id":Q,
   "results":[[stable_id, distance], ...]}
Optional `{record:"meta"}` / `{record:"summary"}` lines are ignored and cannot
substitute for operation records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"E1GTRC01"
VERSION = 1
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
OP_BY_CODE = {1: "insert", 2: "delete", 3: "knn", 4: "range"}


def active_set_sha256(active: set[int]) -> str:
    return hashlib.sha256("".join(f"{sid}\n" for sid in sorted(active)).encode("ascii")).hexdigest()


def error(kind: str, **kwargs: Any) -> dict[str, Any]:
    return {"type": kind, **kwargs}


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usable: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(error("invalid_json", line=lineno, reason=str(exc)))
                continue
            if not isinstance(obj, dict):
                errors.append(error("invalid_record_type", line=lineno, reason="JSON record must be an object"))
                continue
            # Never use a summary as evidence.  It is merely ignored.
            if obj.get("record") in {"meta", "summary"}:
                continue
            usable.append(obj)
    return usable, errors


def require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def parse_results(row: dict[str, Any]) -> list[tuple[int, float]]:
    raw = row.get("results")
    if not isinstance(raw, list):
        raise ValueError("results must be a JSON list")
    output: list[tuple[int, float]] = []
    for pos, item in enumerate(raw):
        if isinstance(item, dict):
            sid, distance = item.get("stable_id"), item.get("distance")
        elif isinstance(item, list) and len(item) == 2:
            sid, distance = item
        else:
            raise ValueError(f"results[{pos}] must be [stable_id,distance] or an object")
        stable_id = require_int(sid, f"results[{pos}].stable_id")
        d = float(distance)
        if not np.isfinite(d):
            raise ValueError(f"results[{pos}].distance must be finite")
        output.append((stable_id, d))
    if len({sid for sid, _ in output}) != len(output):
        raise ValueError("duplicate stable ID in query results")
    return output


def exact_l2(pool: np.ndarray, query: np.ndarray, active: set[int], kind: str, radius: float, k: int) -> list[tuple[int, float]]:
    ids = np.fromiter(sorted(active), dtype=np.int64)
    # float64 reference arithmetic is intentionally independent of CUDA execution.
    delta = pool[ids].astype(np.float64, copy=False) - query.astype(np.float64, copy=False)
    distance = np.sqrt(np.sum(delta * delta, axis=1))
    if kind == "knn":
        index = np.lexsort((ids, distance))[: min(k, len(ids))]
    elif kind == "range":
        keep = np.flatnonzero(distance <= radius + 1e-12)
        index = keep[np.lexsort((ids[keep], distance[keep]))] if len(keep) else np.empty(0, dtype=np.int64)
    else:
        raise ValueError(f"unknown query kind {kind}")
    return [(int(ids[i]), float(distance[i])) for i in index]


def load_prepared(bundle: Path) -> tuple[dict[str, Any], list[tuple[int, str, int]], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prep_errors: list[dict[str, Any]] = []
    header_meta = metadata.get("header", {})
    blob = (bundle / "trace.e1gtrc").read_bytes()
    if len(blob) < HEADER.size:
        raise ValueError("trace.e1gtrc is shorter than its fixed header")
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = HEADER.unpack_from(blob, 0)
    fields = {
        "magic": magic.decode("ascii", errors="replace"), "version": version, "dim": dim,
        "base_n": base_n, "reservoir_n": reservoir_n, "pool_n": pool_n,
        "query_n": query_n, "k": k, "event_count": event_count,
    }
    if magic != MAGIC or version != VERSION:
        prep_errors.append(error("unsupported_trace_header", observed=fields, expected_magic=MAGIC.decode("ascii"), expected_version=VERSION))
    for name, observed in fields.items():
        if name in header_meta and header_meta[name] != observed:
            prep_errors.append(error("metadata_header_mismatch", field=name, metadata=header_meta[name], binary=observed))
    if "radius" in header_meta and not np.isclose(np.float32(header_meta["radius"]), np.float32(radius), rtol=0.0, atol=0.0):
        prep_errors.append(error("metadata_header_mismatch", field="radius", metadata=header_meta["radius"], binary=float(radius)))
    expected_bytes = HEADER.size + EVENT.size * event_count
    if len(blob) != expected_bytes:
        prep_errors.append(error("trace_size_mismatch", bytes=len(blob), expected_bytes=expected_bytes))
    events: list[tuple[int, str, int]] = []
    if len(blob) >= expected_bytes:
        for i in range(event_count):
            op_index, code, argument = EVENT.unpack_from(blob, HEADER.size + i * EVENT.size)
            op = OP_BY_CODE.get(code)
            if op is None:
                prep_errors.append(error("unknown_op_code", event_position=i, op_index=op_index, op_code=code))
                op = f"unknown_{code}"
            if op_index != i:
                prep_errors.append(error("noncontiguous_op_index", event_position=i, op_index=op_index))
            events.append((int(op_index), op, int(argument)))
    pool_path, query_path = bundle / "pool.f32", bundle / "queries.f32"
    required_pool_count = pool_n * dim
    required_query_count = query_n * dim
    pool_raw = np.fromfile(pool_path, dtype="<f4")
    query_raw = np.fromfile(query_path, dtype="<f4")
    if len(pool_raw) != required_pool_count:
        prep_errors.append(error("pool_size_mismatch", values=len(pool_raw), expected_values=required_pool_count))
        pool = np.empty((0, dim), dtype=np.float32)
    else:
        pool = pool_raw.reshape(pool_n, dim)
    if len(query_raw) != required_query_count:
        prep_errors.append(error("query_size_mismatch", values=len(query_raw), expected_values=required_query_count))
        queries = np.empty((0, dim), dtype=np.float32)
    else:
        queries = query_raw.reshape(query_n, dim)
    metadata["_binary_header"] = {**fields, "radius": float(radius), "header_bytes": HEADER.size, "event_bytes": EVENT.size}
    return metadata, events, pool, queries, prep_errors


def verify(bundle: Path, engine_path: Path, atol: float, rtol: float) -> dict[str, Any]:
    metadata, events, pool, queries, prep_errors = load_prepared(bundle)
    header = metadata["_binary_header"]
    engine_records, parse_errors = load_jsonl(engine_path)
    errors = [*prep_errors, *parse_errors]
    by_op_index: dict[int, dict[str, Any]] = {}
    for line_no, row in enumerate(engine_records, 1):
        try:
            oi = require_int(row.get("op_index"), "op_index")
        except ValueError as exc:
            errors.append(error("bad_engine_record", record_number=line_no, reason=str(exc)))
            continue
        if oi in by_op_index:
            errors.append(error("duplicate_engine_record", op_index=oi))
        else:
            by_op_index[oi] = row

    base_n = int(header["base_n"])
    reservoir_n = int(header["reservoir_n"])
    pool_n = int(header["pool_n"])
    query_n = int(header["query_n"])
    k = int(header["k"])
    radius = float(header["radius"])
    active = set(range(base_n))
    checked = {"update": 0, "knn": 0, "range": 0}
    comparisons = {"knn": 0, "range": 0}

    for oi, op, argument in events:
        row = by_op_index.pop(oi, None)
        if row is None:
            errors.append(error("missing_engine_record", op_index=oi, op=op))
        elif op in {"insert", "delete"}:
            try:
                if row.get("record") != "update":
                    raise ValueError("record must be update")
                if row.get("op") != op:
                    raise ValueError(f"op mismatch: expected {op}, got {row.get('op')}")
                stable_id = require_int(row.get("stable_id"), "stable_id")
                if stable_id != argument:
                    raise ValueError(f"stable_id mismatch: expected {argument}, got {stable_id}")
            except ValueError as exc:
                errors.append(error("update_metadata_mismatch", op_index=oi, reason=str(exc)))
        elif op in {"knn", "range"}:
            try:
                if row.get("record") != "query":
                    raise ValueError("record must be query")
                if row.get("kind") != op:
                    raise ValueError(f"kind mismatch: expected {op}, got {row.get('kind')}")
                query_id = require_int(row.get("query_id"), "query_id")
                if query_id != argument:
                    raise ValueError(f"query_id mismatch: expected {argument}, got {query_id}")
                if not (0 <= argument < query_n):
                    raise ValueError(f"binary query ID {argument} outside [0,{query_n})")
                observed = parse_results(row)
                expected = exact_l2(pool, queries[argument], active, op, radius, k)
                comparisons[op] += 1
                if op == "knn":
                    if len(observed) != len(expected):
                        errors.append(error("knn_length_mismatch", op_index=oi, expected=len(expected), observed=len(observed)))
                    for position, (got, want) in enumerate(zip(observed, expected)):
                        if got[0] != want[0] or not np.isclose(got[1], want[1], atol=atol, rtol=rtol):
                            errors.append(error("knn_mismatch", op_index=oi, position=position, expected=[want[0], want[1]], observed=[got[0], got[1]]))
                            break
                else:
                    actual_by_id = {sid: distance for sid, distance in observed}
                    expected_by_id = {sid: distance for sid, distance in expected}
                    missing = sorted(set(expected_by_id) - set(actual_by_id))
                    extra = sorted(set(actual_by_id) - set(expected_by_id))
                    bad_distance = [sid for sid in sorted(set(actual_by_id) & set(expected_by_id)) if not np.isclose(actual_by_id[sid], expected_by_id[sid], atol=atol, rtol=rtol)]
                    if missing or extra or bad_distance:
                        errors.append(error("range_mismatch", op_index=oi, missing=missing[:20], extra=extra[:20], bad_distance_ids=bad_distance[:20], counts={"missing": len(missing), "extra": len(extra), "bad_distance": len(bad_distance)}))
            except ValueError as exc:
                errors.append(error("query_metadata_or_result_error", op_index=oi, reason=str(exc)))
        else:
            errors.append(error("unsupported_binary_op", op_index=oi, op=op))

        # Replay only the binary trace; this is deliberately independent of the engine's state claims.
        if op == "insert":
            if not (base_n <= argument < base_n + reservoir_n <= pool_n):
                errors.append(error("invalid_insert_id_in_binary_trace", op_index=oi, stable_id=argument))
            elif argument in active:
                errors.append(error("duplicate_insert_in_binary_trace", op_index=oi, stable_id=argument))
            else:
                active.add(argument)
                checked["update"] += 1
        elif op == "delete":
            if argument not in active:
                errors.append(error("delete_nonlive_id_in_binary_trace", op_index=oi, stable_id=argument))
            else:
                active.remove(argument)
                checked["update"] += 1
        elif op in {"knn", "range"}:
            checked[op] += 1

    if by_op_index:
        errors.append(error("unexpected_engine_records", count=len(by_op_index), op_indices=sorted(by_op_index)[:50]))
    return {
        "schema": "e1g0-cuda-reference-export-verification-v1",
        "scope": "independent NumPy exact verifier for E1-G0 CUDA reference export; NOT GTS integration evidence",
        "gpu_used": False,
        "prepared_bundle": str(bundle),
        "engine_jsonl": str(engine_path),
        "header": header,
        "pass": not errors,
        "checked_events": checked,
        "exact_query_comparisons": comparisons,
        "final_active_count": len(active),
        "verifier_active_set_sha256": active_set_sha256(active),
        "error_count": len(errors),
        "errors": errors[:200],
        "ignored_engine_summary_lines": True,
        "tolerance": {"atol": atol, "rtol": rtol},
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-bundle", required=True, type=Path)
    p.add_argument("--engine-jsonl", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-5)
    args = p.parse_args()
    result = verify(args.prepared_bundle.resolve(), args.engine_jsonl.resolve(), args.atol, args.rtol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": result["pass"], "error_count": result["error_count"], "out": str(args.out.resolve()), "gpu_used": False}, sort_keys=True))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
