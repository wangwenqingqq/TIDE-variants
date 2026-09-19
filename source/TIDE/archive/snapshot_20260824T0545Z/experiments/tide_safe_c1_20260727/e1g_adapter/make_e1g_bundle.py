#!/usr/bin/env python3
"""Create a stable-ID/disjoint-reservoir E1-G input bundle.

This adapter deliberately does *not* translate the trace into the legacy GTS
update format: that format has logical-rank deletes and count-only output.  It
instead creates the contract that a Safe-C1 CUDA implementation must consume
and an exact expected-answer stream for independent verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_set_hash(active: set[int]) -> str:
    payload = "".join(f"{sid}\n" for sid in sorted(active)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected JSON object")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False))
            fh.write("\n")


def exact_knn(pool: np.ndarray, query: np.ndarray, active: set[int], k: int) -> list[list[float | int]]:
    ids = np.asarray(sorted(active), dtype=np.int64)
    data = pool[ids].astype(np.float64, copy=False)
    q = query.astype(np.float64, copy=False)
    dist = np.sqrt(np.sum((data - q) ** 2, axis=1))
    order = np.lexsort((ids, dist))[: min(k, len(ids))]
    return [[int(ids[i]), float(dist[i])] for i in order]


def exact_range(pool: np.ndarray, query: np.ndarray, active: set[int], radius: float) -> list[list[float | int]]:
    ids = np.asarray(sorted(active), dtype=np.int64)
    data = pool[ids].astype(np.float64, copy=False)
    q = query.astype(np.float64, copy=False)
    dist = np.sqrt(np.sum((data - q) ** 2, axis=1))
    keep = np.flatnonzero(dist <= radius + 1e-12)
    if len(keep) == 0:
        return []
    order = keep[np.lexsort((ids[keep], dist[keep]))]
    return [[int(ids[i]), float(dist[i])] for i in order]


def translated_event(event: dict[str, Any]) -> dict[str, Any]:
    op = str(event["op"])
    out: dict[str, Any] = {"op_index": int(event["op_index"]), "op": op}
    if op in {"insert", "delete"}:
        stable_id = int(event["id"])
        out["stable_id"] = stable_id
        if op == "insert":
            # In the immutable pool contract stable ID is the pool row.
            out["pool_row"] = stable_id
    elif op in {"knn", "range"}:
        out["query_id"] = int(event["query_id"])
        out["query_row"] = int(event["query_id"])
    else:
        raise ValueError(f"unsupported op {op!r}")
    if "label" in event:
        out["label"] = str(event["label"])
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path, help="one safe_c1_oracle seed directory")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--radius", type=float, default=3.0)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    run_dir: Path = args.run_dir.resolve()
    out: Path = args.out.resolve()
    required = [run_dir / name for name in ("pool.npy", "queries.npy", "trace.jsonl", "summary.json")]
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise SystemExit("missing required inputs: " + ", ".join(missing))
    if out.exists():
        if not args.force:
            raise SystemExit(f"output already exists: {out}; pass --force to replace")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    data_meta = summary["data"]
    base_n = int(data_meta["base_n"])
    reservoir_n = int(data_meta["reservoir_n"])
    dim = int(data_meta["dimension"])
    pool = np.load(run_dir / "pool.npy")
    queries = np.load(run_dir / "queries.npy")
    if pool.ndim != 2 or pool.shape != (base_n + reservoir_n, dim):
        raise ValueError(f"pool shape {pool.shape} violates ({base_n + reservoir_n}, {dim})")
    if queries.ndim != 2 or queries.shape[1] != dim:
        raise ValueError(f"query shape {queries.shape} violates dimension {dim}")
    if not np.issubdtype(pool.dtype, np.floating) or not np.issubdtype(queries.dtype, np.floating):
        raise ValueError("pool/queries must be floating point")
    pool = np.asarray(pool, dtype=np.float32, order="C")
    queries = np.asarray(queries, dtype=np.float32, order="C")

    trace = read_jsonl(run_dir / "trace.jsonl")
    active: set[int] = set(range(base_n))
    ever_inserted: set[int] = set()
    stable_trace: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    counts = {"insert": 0, "delete": 0, "knn": 0, "range": 0}

    for expected_index, event in enumerate(trace):
        if int(event.get("op_index", -1)) != expected_index:
            raise ValueError(f"trace op_index is not contiguous at list index {expected_index}")
        converted = translated_event(event)
        stable_trace.append(converted)
        op = converted["op"]
        if op == "insert":
            sid = int(converted["stable_id"])
            if not (base_n <= sid < base_n + reservoir_n):
                raise ValueError(f"op {expected_index}: insert {sid} is not in the disjoint reservoir")
            if sid in active or sid in ever_inserted:
                raise ValueError(f"op {expected_index}: duplicate/reinserted stable ID {sid}")
            active.add(sid)
            ever_inserted.add(sid)
            expected.append({
                "record": "update", "op_index": expected_index, "op": "insert", "stable_id": sid,
                "active_count": len(active), "active_set_sha256": stable_set_hash(active),
            })
        elif op == "delete":
            sid = int(converted["stable_id"])
            if sid not in active:
                raise ValueError(f"op {expected_index}: delete of non-live stable ID {sid}")
            active.remove(sid)
            expected.append({
                "record": "update", "op_index": expected_index, "op": "delete", "stable_id": sid,
                "active_count": len(active), "active_set_sha256": stable_set_hash(active),
            })
        elif op == "knn":
            qid = int(converted["query_id"])
            if not (0 <= qid < len(queries)):
                raise ValueError(f"op {expected_index}: bad query ID {qid}")
            expected.append({
                "record": "query", "op_index": expected_index, "kind": "knn", "query_id": qid,
                "results": exact_knn(pool, queries[qid], active, args.k),
                "active_count": len(active), "active_set_sha256": stable_set_hash(active),
            })
        elif op == "range":
            qid = int(converted["query_id"])
            if not (0 <= qid < len(queries)):
                raise ValueError(f"op {expected_index}: bad query ID {qid}")
            expected.append({
                "record": "query", "op_index": expected_index, "kind": "range", "query_id": qid,
                "results": exact_range(pool, queries[qid], active, args.radius),
                "active_count": len(active), "active_set_sha256": stable_set_hash(active),
            })
        counts[op] += 1

    # Binary layout is raw row-major float32 / int32.  The contract records dimensions.
    (out / "pool.f32").write_bytes(pool.tobytes(order="C"))
    (out / "queries.f32").write_bytes(queries.tobytes(order="C"))
    np.asarray(range(base_n), dtype="<i4").tofile(out / "initial_base_stable_ids.i32")
    np.arange(base_n + reservoir_n, dtype="<i4").tofile(out / "stable_id_to_pool_row.i32")
    write_jsonl(out / "stable_trace.jsonl", stable_trace)
    write_jsonl(out / "oracle_expected.jsonl", expected)

    contract: dict[str, Any] = {
        "schema": "e1g-safe-c1-stable-id-contract-v1",
        "purpose": "E1-G CUDA/GTS replay precondition; no legacy logical-rank update semantics",
        "metric": "L2",
        "dimension": dim,
        "base_n": base_n,
        "reservoir_n": reservoir_n,
        "pool_n": int(pool.shape[0]),
        "query_n": int(queries.shape[0]),
        "radius": float(args.radius),
        "k": int(args.k),
        "stable_id_contract": {
            "base": "stable IDs [0, base_n) are present when the frozen tree is built",
            "reservoir": "stable IDs [base_n, base_n+reservoir_n) are disjoint from the tree at build time",
            "mapping": "stable_id_to_pool_row.i32; current bundle uses identity mapping",
            "deletes": "delete records address stable IDs, never live-rank/logical IDs",
            "queries": "external query vectors are addressed by query_id, never pool-row IDs",
        },
        "engine_export_schema": {
            "update": {"record": "update", "op_index": "int", "op": "insert|delete", "stable_id": "int"},
            "query": {"record": "query", "op_index": "int", "kind": "knn|range", "query_id": "int", "results": "[[stable_id, distance], ...]"},
            "constraint": "exactly one engine record for every stable_trace event; no count-only result accepted",
        },
        "source_run": str(run_dir),
        "source_files": {name: sha256_file(run_dir / name) for name in ("pool.npy", "queries.npy", "trace.jsonl", "summary.json")},
        "trace_counts": counts,
        "final_active_count": len(active),
        "final_active_set_sha256": stable_set_hash(active),
        "legacy_update_file_emitted": False,
        "gpu_used_to_create_bundle": False,
    }
    (out / "contract.json").write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    output_files = [p for p in sorted(out.iterdir()) if p.is_file()]
    manifest = {
        "schema": "e1g-bundle-manifest-v1",
        "files": {p.name: sha256_file(p) for p in output_files},
        "contract_sha256": sha256_file(out / "contract.json"),
        "gpu_used": False,
        "status": "ready_for_safe_cuda_integration_not_a_gts_result",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bundle": str(out), "events": len(trace), "counts": counts, "final_live": len(active), "gpu_used": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
