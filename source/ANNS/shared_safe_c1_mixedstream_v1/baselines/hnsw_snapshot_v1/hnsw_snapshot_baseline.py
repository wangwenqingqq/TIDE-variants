#!/usr/bin/env python3
"""Independent snapshot-correct CPU HNSW baseline for TIDE/Safe-C1.

This tool deliberately does not read any Safe-C1 V5--V8 bundle or result.  A
trace is generated from raw public SIFT files, with stable IDs and an explicit
live-set transition.  At every query it compares hnswlib against a freshly
computed exact live-set top-k oracle.

The initial supported use is development smoke testing.  It is not a GPU
comparison and it is not a final/held-out result generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(obj))


def fvecs_shape(path: Path) -> tuple[int, int]:
    size = path.stat().st_size
    with path.open("rb") as f:
        raw = f.read(4)
    if len(raw) != 4:
        raise ValueError(f"empty/truncated fvecs: {path}")
    d = int(np.frombuffer(raw, dtype="<i4")[0])
    if d <= 0:
        raise ValueError(f"invalid fvecs dimension {d}: {path}")
    row = 4 * (d + 1)
    if size % row:
        raise ValueError(f"fvecs byte size not divisible by row width: {path}")
    return size // row, d


def open_fvecs(path: Path) -> np.ndarray:
    n, d = fvecs_shape(path)
    record = np.dtype([("dim", "<i4"), ("vec", "<f4", (d,))])
    arr = np.memmap(path, dtype=record, mode="r", shape=(n,))
    # Catch an input with variable or corrupt dimensions before using it.
    probes = np.unique(np.array([0, n // 2, n - 1], dtype=np.int64))
    if not np.all(arr[probes]["dim"] == d):
        raise ValueError(f"inconsistent fvecs dimensions: {path}")
    return arr["vec"]


def percentile_ns(values: list[int], p: float) -> int | None:
    if not values:
        return None
    return int(np.percentile(np.asarray(values, dtype=np.float64), p, method="linear"))


def stable_set_sha256(ids: Iterable[int]) -> str:
    a = np.asarray(sorted(ids), dtype="<u8")
    return hashlib.sha256(a.tobytes()).hexdigest()


def exact_topk(vectors: np.ndarray, ids: np.ndarray, query: np.ndarray, k: int) -> tuple[list[int], list[float]]:
    # float64 keeps the oracle independent of the approximate index's internal
    # floating-point accumulation.  Tie break is fixed by stable external ID.
    diff = vectors.astype(np.float64, copy=False) - query.astype(np.float64, copy=False)
    distances = np.einsum("ij,ij->i", diff, diff, dtype=np.float64)
    if k > len(ids):
        raise ValueError(f"k={k} exceeds live set={len(ids)}")
    small = np.argpartition(distances, k - 1)[:k]
    cutoff = float(np.max(distances[small]))
    strict = np.flatnonzero(distances < cutoff)
    equal = np.flatnonzero(distances == cutoff)
    need = k - len(strict)
    equal = equal[np.argsort(ids[equal], kind="stable")[:need]]
    chosen = np.concatenate((strict, equal))
    order = np.lexsort((ids[chosen], distances[chosen]))
    chosen = chosen[order]
    return [int(x) for x in ids[chosen]], [float(x) for x in distances[chosen]]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_trace(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            obj = json.loads(line)
            if obj.get("ordinal") != len(records):
                raise ValueError(f"trace ordinal mismatch at line {line_no}")
            records.append(obj)
    return records


def get_cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def cmd_make(args: argparse.Namespace) -> None:
    base_path = Path(args.base).resolve()
    learn_path = Path(args.learn).resolve()
    query_path = Path(args.query).resolve()
    out_dir = Path(args.out_dir).resolve()
    protocol_id = args.protocol_id
    if not protocol_id.startswith("hnsw-dev-"):
        raise ValueError("This v1 tool accepts only a development protocol id beginning hnsw-dev-.")
    if args.base_n <= 0 or args.arrival_n <= 0 or args.warmup <= 0 or args.pairs <= 0 or args.query_every <= 0:
        raise ValueError("all workload counts must be positive")
    if args.arrival_n < args.warmup + args.pairs:
        raise ValueError("arrival_n must cover warmup + pairs")

    base_total, d = fvecs_shape(base_path)
    learn_total, learn_d = fvecs_shape(learn_path)
    query_total, query_d = fvecs_shape(query_path)
    if not (d == learn_d == query_d):
        raise ValueError(f"dimension mismatch: base={d}, learn={learn_d}, query={query_d}")
    if args.base_n > base_total:
        raise ValueError("base_n exceeds base file")
    if args.arrival_offset < 0 or args.arrival_offset + args.arrival_n > learn_total:
        raise ValueError("arrival selection exceeds learn file")
    expected_queries = args.pairs // args.query_every
    if args.query_n != expected_queries:
        raise ValueError(f"query_n must equal pairs/query_every={expected_queries} for a fixed cadence")
    if args.query_offset < 0 or args.query_offset + args.query_n > query_total:
        raise ValueError("query selection exceeds query file")
    if out_dir.exists():
        raise FileExistsError(f"refuse to overwrite protocol directory: {out_dir}")

    rng = np.random.default_rng(args.seed)
    trace: list[dict[str, Any]] = []
    active_arrivals: list[int] = []
    ordinal = 0
    for a in range(args.warmup):
        label = args.base_n + a
        trace.append({"ordinal": ordinal, "op": "I", "arrival_offset": args.arrival_offset + a, "label": label})
        active_arrivals.append(label)
        ordinal += 1
    for pair in range(args.pairs):
        a = args.warmup + pair
        label = args.base_n + a
        trace.append({"ordinal": ordinal, "op": "I", "arrival_offset": args.arrival_offset + a, "label": label})
        ordinal += 1
        # The deletion target is active before this pair's insertion.  It can
        # never be the just-inserted label; this preserves immediate-visibility
        # pressure at every snapshot.
        victim_idx = int(rng.integers(0, len(active_arrivals)))
        victim = active_arrivals.pop(victim_idx)
        trace.append({"ordinal": ordinal, "op": "D", "label": victim})
        ordinal += 1
        active_arrivals.append(label)
        if (pair + 1) % args.query_every == 0:
            q = args.query_offset + ((pair + 1) // args.query_every - 1)
            trace.append({"ordinal": ordinal, "op": "Q", "query_offset": q, "after_pair": pair + 1})
            ordinal += 1

    out_dir.mkdir(parents=True, exist_ok=False)
    trace_path = out_dir / "trace.jsonl"
    trace_bytes = b"".join(canonical_json_bytes(r) for r in trace)
    atomic_write_bytes(trace_path, trace_bytes)
    manifest = {
        "schema": "tide-hnsw-snapshot-protocol-v1",
        "protocol_id": protocol_id,
        "evidence_scope": "development_smoke_only",
        "claim_boundary": [
            "semantic/API smoke test only",
            "not a held-out test",
            "not a GPU performance comparison",
            "not a Safe-C1 performance result",
        ],
        "dataset": "SIFT1M-L2",
        "dimension": d,
        "selection": {
            "base": {"path": str(base_path), "sha256": sha256_file(base_path), "row_start": 0, "row_count": args.base_n},
            "arrival": {"path": str(learn_path), "sha256": sha256_file(learn_path), "row_start": args.arrival_offset, "row_count": args.arrival_n},
            "query": {"path": str(query_path), "sha256": sha256_file(query_path), "row_start": args.query_offset, "row_count": args.query_n},
        },
        "workload": {
            "seed": args.seed,
            "base_n": args.base_n,
            "arrival_n": args.arrival_n,
            "warmup_inserts": args.warmup,
            "insert_delete_pairs": args.pairs,
            "query_every_pairs": args.query_every,
            "query_count": args.query_n,
            "k": args.k,
            "base_deletes": "forbidden",
            "deletion_rule": "each D selects a pre-pair active arrival label; deleted labels never reinsert",
            "query_visibility": "each Q must reflect all prior I/D operations",
        },
        "artifacts": {"trace": {"path": "trace.jsonl", "sha256": hashlib.sha256(trace_bytes).hexdigest(), "bytes": len(trace_bytes)}},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"status": "PROTOCOL_CREATED", "directory": str(out_dir), "trace_sha256": manifest["artifacts"]["trace"]["sha256"], "query_count": args.query_n}, sort_keys=True))


def require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(f"{field} mismatch: actual={actual!r} expected={expected!r}")


def cmd_run(args: argparse.Namespace) -> None:
    protocol_dir = Path(args.protocol_dir).resolve()
    manifest_path = protocol_dir / "manifest.json"
    trace_path = protocol_dir / "trace.jsonl"
    manifest = load_json(manifest_path)
    require_equal(manifest.get("schema"), "tide-hnsw-snapshot-protocol-v1", "schema")
    require_equal(manifest.get("evidence_scope"), "development_smoke_only", "evidence_scope")
    trace_digest = sha256_file(trace_path)
    require_equal(trace_digest, manifest["artifacts"]["trace"]["sha256"], "trace sha256")
    records = read_trace(trace_path)
    if len(records) == 0:
        raise ValueError("empty trace")
    s = manifest["selection"]
    for role in ("base", "arrival", "query"):
        p = Path(s[role]["path"])
        require_equal(sha256_file(p), s[role]["sha256"], f"{role} data sha256")
    base = open_fvecs(Path(s["base"]["path"]))[: int(s["base"]["row_count"])]
    a_start, a_n = int(s["arrival"]["row_start"]), int(s["arrival"]["row_count"])
    arrivals = open_fvecs(Path(s["arrival"]["path"]))[a_start:a_start + a_n]
    q_start, q_n = int(s["query"]["row_start"]), int(s["query"]["row_count"])
    queries = open_fvecs(Path(s["query"]["path"]))[q_start:q_start + q_n]
    w = manifest["workload"]
    base_n, k = int(w["base_n"]), int(w["k"])
    require_equal(len(base), base_n, "base vector count")
    require_equal(len(arrivals), a_n, "arrival vector count")
    all_vectors = np.concatenate((np.asarray(base, dtype=np.float32), np.asarray(arrivals, dtype=np.float32)), axis=0)

    import hnswlib
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"refuse to overwrite run directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    # hnswlib labels are immutable for this trace: a label is inserted once,
    # may be marked deleted once, and is never reused.
    index = hnswlib.Index(space="l2", dim=int(manifest["dimension"]))
    t0 = time.perf_counter_ns()
    index.init_index(max_elements=base_n + a_n, M=args.M, ef_construction=args.ef_construction, random_seed=args.index_seed, allow_replace_deleted=False)
    index.set_ef(args.ef_search)
    index.set_num_threads(1)
    index.add_items(base, np.arange(base_n, dtype=np.int64), num_threads=1)
    build_ns = time.perf_counter_ns() - t0

    active_arrivals: set[int] = set()
    insert_ns: list[int] = []
    delete_ns: list[int] = []
    query_ns: list[int] = []
    exact_ns: list[int] = []
    qrecords: list[dict[str, Any]] = []
    failure: str | None = None
    for r in records:
        op = r["op"]
        if op == "I":
            label = int(r["label"])
            a_idx = int(r["arrival_offset"]) - a_start
            if not (0 <= a_idx < a_n) or label != base_n + a_idx:
                raise ValueError(f"invalid insert mapping at ordinal {r['ordinal']}")
            if label in active_arrivals:
                raise ValueError(f"duplicate active insert label={label}")
            t = time.perf_counter_ns()
            index.add_items(arrivals[a_idx:a_idx + 1], np.asarray([label], dtype=np.int64), num_threads=1)
            insert_ns.append(time.perf_counter_ns() - t)
            active_arrivals.add(label)
        elif op == "D":
            label = int(r["label"])
            if label not in active_arrivals:
                raise ValueError(f"delete not-active arrival label={label}")
            t = time.perf_counter_ns()
            index.mark_deleted(label)
            delete_ns.append(time.perf_counter_ns() - t)
            active_arrivals.remove(label)
        elif op == "Q":
            q_idx = int(r["query_offset"]) - q_start
            if not (0 <= q_idx < q_n):
                raise ValueError(f"invalid query mapping at ordinal {r['ordinal']}")
            active_ids = np.asarray(list(range(base_n)) + sorted(active_arrivals), dtype=np.int64)
            active_sha = stable_set_sha256(active_ids.tolist())
            t = time.perf_counter_ns()
            exact_ids, exact_dist = exact_topk(all_vectors[active_ids], active_ids, queries[q_idx], k)
            exact_elapsed = time.perf_counter_ns() - t
            t = time.perf_counter_ns()
            labels, dists = index.knn_query(queries[q_idx:q_idx + 1], k=k, num_threads=1)
            hnsw_elapsed = time.perf_counter_ns() - t
            returned = [int(x) for x in labels[0].tolist()]
            returned_d = [float(x) for x in dists[0].tolist()]
            live = set(active_ids.tolist())
            stale = sorted(set(returned) - live)
            recall = len(set(returned).intersection(exact_ids)) / float(k)
            qr = {
                "ordinal": int(r["ordinal"]), "query_offset": int(r["query_offset"]), "after_pair": int(r["after_pair"]),
                "active_count": int(len(active_ids)), "active_ids_sha256": active_sha,
                "exact_ids": exact_ids, "exact_squared_l2": exact_dist,
                "hnsw_ids": returned, "hnsw_distances": returned_d,
                "recall_at_k": recall, "stale_or_inactive_returned": stale,
                "exact_elapsed_ns": exact_elapsed, "hnsw_query_elapsed_ns": hnsw_elapsed,
            }
            qrecords.append(qr)
            exact_ns.append(exact_elapsed)
            query_ns.append(hnsw_elapsed)
            if stale:
                failure = f"visibility violation at ordinal {r['ordinal']}: {stale}"
                break
        else:
            raise ValueError(f"unknown trace op {op}")
    q_bytes = b"".join(canonical_json_bytes(x) for x in qrecords)
    atomic_write_bytes(out_dir / "query_records.jsonl", q_bytes)
    recalls = [float(q["recall_at_k"]) for q in qrecords]
    summary = {
        "schema": "tide-hnsw-snapshot-run-v1",
        "status": "PASS_DEVELOPMENT_SMOKE" if failure is None else "FAIL_VISIBILITY",
        "evidence_scope": "development_smoke_only",
        "claim_boundary": manifest["claim_boundary"],
        "protocol": {"directory": str(protocol_dir), "manifest_sha256": sha256_file(manifest_path), "trace_sha256": trace_digest},
        "hnsw": {"M": args.M, "ef_construction": args.ef_construction, "ef_search": args.ef_search, "index_seed": args.index_seed, "threads": 1, "module_path": str(Path(hnswlib.__file__).resolve()), "module_sha256": sha256_file(Path(hnswlib.__file__).resolve())},
        "environment": {"hostname": socket.gethostname(), "cpu_model": get_cpu_model(), "python": sys.version, "platform": platform.platform(), "gpu_used": False, "thread_env": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "timing_ns": {"build": build_ns, "insert": {"count": len(insert_ns), "p50": percentile_ns(insert_ns, 50), "p95": percentile_ns(insert_ns, 95)}, "delete": {"count": len(delete_ns), "p50": percentile_ns(delete_ns, 50), "p95": percentile_ns(delete_ns, 95)}, "hnsw_query": {"count": len(query_ns), "p50": percentile_ns(query_ns, 50), "p95": percentile_ns(query_ns, 95)}, "exact_oracle": {"count": len(exact_ns), "p50": percentile_ns(exact_ns, 50), "p95": percentile_ns(exact_ns, 95)}},
        "correctness": {"queries_expected": int(w["query_count"]), "queries_completed": len(qrecords), "mean_recall_at_k": float(np.mean(recalls)) if recalls else None, "min_recall_at_k": float(np.min(recalls)) if recalls else None, "visibility_violations": sum(1 for q in qrecords if q["stale_or_inactive_returned"]), "query_records_sha256": hashlib.sha256(q_bytes).hexdigest()},
        "failure": failure,
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(out_dir / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "run_dir": str(out_dir), "queries_completed": len(qrecords), "mean_recall_at_k": summary["correctness"]["mean_recall_at_k"], "visibility_violations": summary["correctness"]["visibility_violations"]}, sort_keys=True))
    if failure:
        raise SystemExit(2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    make = sub.add_parser("make", help="create a deterministic development-only protocol")
    make.add_argument("--base", required=True)
    make.add_argument("--learn", required=True)
    make.add_argument("--query", required=True)
    make.add_argument("--out-dir", required=True)
    make.add_argument("--protocol-id", required=True)
    make.add_argument("--base-n", type=int, required=True)
    make.add_argument("--arrival-n", type=int, required=True)
    make.add_argument("--arrival-offset", type=int, default=0)
    make.add_argument("--warmup", type=int, required=True)
    make.add_argument("--pairs", type=int, required=True)
    make.add_argument("--query-every", type=int, required=True)
    make.add_argument("--query-offset", type=int, default=0)
    make.add_argument("--query-n", type=int, required=True)
    make.add_argument("--k", type=int, default=10)
    make.add_argument("--seed", type=int, required=True)
    make.set_defaults(func=cmd_make)
    run = sub.add_parser("run", help="run hnswlib and exact snapshot oracle")
    run.add_argument("--protocol-dir", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--M", type=int, required=True)
    run.add_argument("--ef-construction", type=int, required=True)
    run.add_argument("--ef-search", type=int, required=True)
    run.add_argument("--index-seed", type=int, default=17)
    run.set_defaults(func=cmd_run)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
