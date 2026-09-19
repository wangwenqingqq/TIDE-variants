#!/usr/bin/env python3
"""Independently verify a Safe-C1 CUDA/GTS stable-ID result export.

The verifier replays `stable_trace.jsonl` over immutable pool vectors and
recomputes exact L2 range/top-k answers.  It intentionally does not trust
`oracle_expected.jsonl`, so it can be used after a real CUDA integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def set_hash(active: set[int]) -> str:
    return hashlib.sha256("".join(f"{sid}\n" for sid in sorted(active)).encode("ascii")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if raw:
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}:{lineno}: expected object")
                rows.append(obj)
    return rows


def exact(pool: np.ndarray, query: np.ndarray, active: set[int], kind: str, radius: float, k: int) -> list[tuple[int, float]]:
    ids = np.asarray(sorted(active), dtype=np.int64)
    dist = np.sqrt(np.sum((pool[ids].astype(np.float64, copy=False) - query.astype(np.float64, copy=False)) ** 2, axis=1))
    if kind == "knn":
        order = np.lexsort((ids, dist))[: min(k, len(ids))]
    elif kind == "range":
        keep = np.flatnonzero(dist <= radius + 1e-12)
        order = keep[np.lexsort((ids[keep], dist[keep]))] if len(keep) else np.empty(0, dtype=np.int64)
    else:
        raise ValueError(f"unknown query kind {kind}")
    return [(int(ids[i]), float(dist[i])) for i in order]


def parse_results(row: dict[str, Any]) -> list[tuple[int, float]]:
    raw = row.get("results")
    if not isinstance(raw, list):
        raise ValueError("missing list-valued `results`")
    out: list[tuple[int, float]] = []
    for x in raw:
        if isinstance(x, dict):
            sid, dist = x.get("stable_id"), x.get("distance")
        elif isinstance(x, (list, tuple)) and len(x) == 2:
            sid, dist = x
        else:
            raise ValueError("every result must be [stable_id,distance] or an object")
        if isinstance(sid, bool) or not isinstance(sid, int):
            raise ValueError("stable_id must be an integer")
        d = float(dist)
        if not np.isfinite(d):
            raise ValueError("distance must be finite")
        out.append((sid, d))
    if len({sid for sid, _ in out}) != len(out):
        raise ValueError("duplicate stable ID in one query result")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--engine-results", required=True, type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-5)
    args = p.parse_args()
    bundle = args.bundle.resolve()
    contract = json.loads((bundle / "contract.json").read_text(encoding="utf-8"))
    dim = int(contract["dimension"])
    pool = np.fromfile(bundle / "pool.f32", dtype="<f4").reshape(int(contract["pool_n"]), dim)
    queries = np.fromfile(bundle / "queries.f32", dtype="<f4").reshape(int(contract["query_n"]), dim)
    trace = read_jsonl(bundle / "stable_trace.jsonl")
    actual_rows = [x for x in read_jsonl(args.engine_results.resolve()) if x.get("record") != "meta"]
    by_index: dict[int, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for raw in actual_rows:
        if "op_index" not in raw:
            errors.append({"type": "bad_engine_record", "reason": "missing_op_index"})
            continue
        oi = int(raw["op_index"])
        if oi in by_index:
            errors.append({"type": "duplicate_engine_record", "op_index": oi})
        else:
            by_index[oi] = raw

    active = set(range(int(contract["base_n"])))
    checked = {"update": 0, "knn": 0, "range": 0}
    for event in trace:
        oi, op = int(event["op_index"]), str(event["op"])
        row = by_index.pop(oi, None)
        if row is None:
            errors.append({"type": "missing_engine_record", "op_index": oi, "op": op})
            # Still replay trace to retain correct oracle state.
        elif op in {"insert", "delete"}:
            expected_id = int(event["stable_id"])
            if row.get("record") != "update" or row.get("op") != op or int(row.get("stable_id", -1)) != expected_id:
                errors.append({"type": "update_metadata_mismatch", "op_index": oi, "expected": {"op": op, "stable_id": expected_id}, "actual": row})
            if "active_set_sha256" in row and row["active_set_sha256"] != (set_hash(active | {expected_id}) if op == "insert" else set_hash(active - {expected_id})):
                errors.append({"type": "active_hash_mismatch", "op_index": oi})
        else:
            qid = int(event["query_id"])
            if row.get("record") != "query" or row.get("kind") != op or int(row.get("query_id", -1)) != qid:
                errors.append({"type": "query_metadata_mismatch", "op_index": oi, "expected": {"kind": op, "query_id": qid}, "actual": row})
            else:
                try:
                    observed = parse_results(row)
                    expected = exact(pool, queries[qid], active, op, float(contract["radius"]), int(contract["k"]))
                    if op == "knn":
                        if len(observed) != len(expected):
                            errors.append({"type": "knn_length_mismatch", "op_index": oi, "expected": len(expected), "actual": len(observed)})
                        for pos, (got, want) in enumerate(zip(observed, expected)):
                            if got[0] != want[0] or not np.isclose(got[1], want[1], atol=args.atol, rtol=args.rtol):
                                errors.append({"type": "knn_mismatch", "op_index": oi, "position": pos, "expected": want, "actual": got})
                                break
                    else:
                        got_by_id = {sid: dist for sid, dist in observed}
                        want_by_id = {sid: dist for sid, dist in expected}
                        missing = sorted(set(want_by_id) - set(got_by_id))
                        extra = sorted(set(got_by_id) - set(want_by_id))
                        bad_dist = [sid for sid in sorted(set(got_by_id) & set(want_by_id)) if not np.isclose(got_by_id[sid], want_by_id[sid], atol=args.atol, rtol=args.rtol)]
                        if missing or extra or bad_dist:
                            errors.append({"type": "range_mismatch", "op_index": oi, "missing": missing[:20], "extra": extra[:20], "bad_distance_ids": bad_dist[:20], "counts": {"missing": len(missing), "extra": len(extra), "bad_distance": len(bad_dist)}})
                except Exception as exc:  # make malformed exporter output auditable
                    errors.append({"type": "result_parse_error", "op_index": oi, "reason": str(exc)})
        if op == "insert":
            active.add(int(event["stable_id"]))
            checked["update"] += 1
        elif op == "delete":
            active.remove(int(event["stable_id"]))
            checked["update"] += 1
        else:
            checked[op] += 1
    if by_index:
        errors.append({"type": "unexpected_engine_records", "op_indices": sorted(by_index)[:20], "count": len(by_index)})

    result = {
        "schema": "e1g-export-verification-v1",
        "bundle": str(bundle),
        "engine_results": str(args.engine_results.resolve()),
        "pass": not errors,
        "checked": checked,
        "final_active_count": len(active),
        "final_active_set_sha256": set_hash(active),
        "error_count": len(errors),
        "errors": errors[:50],
        "gpu_used_by_verifier": False,
    }
    serialized = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
