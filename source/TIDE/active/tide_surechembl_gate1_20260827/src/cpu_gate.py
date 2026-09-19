#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import resource
import socket
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from FPSim2 import FPSim2Engine
from FPSim2.FPSim2lib import GenericSearch


K_FETCH = 64
K_FINAL = 10
CAL_CORRECT = list(range(64))
SEALED = list(range(512, 768))
CORRECT = CAL_CORRECT + SEALED


@dataclass(frozen=True)
class Hit:
    mol_id: int
    num: int
    den: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def result_hash(hits: list[Hit]) -> str:
    s = ";".join(f"{h.mol_id}:{h.num}/{h.den}" for h in hits)
    return sha256_bytes(s.encode())


def better(a: Hit, b: Hit) -> bool:
    lhs = a.num * b.den
    rhs = b.num * a.den
    return lhs > rhs or (lhs == rhs and a.mol_id < b.mol_id)


def insert_top10(out: list[Hit], hit: Hit) -> None:
    for old in out:
        if old.mol_id == hit.mol_id:
            return
    pos = 0
    while pos < len(out) and better(out[pos], hit):
        pos += 1
    out.insert(pos, hit)
    if len(out) > K_FINAL:
        out.pop()


def exact_hits(db: np.ndarray, idxs: np.ndarray, query: np.ndarray) -> list[Hit]:
    if len(idxs) == 0:
        return []
    rows = db[idxs]
    inter = np.bitwise_count(np.bitwise_and(rows[:, 1:5], query[1:5])).sum(
        axis=1, dtype=np.uint16
    )
    den = query[5].astype(np.uint16) + rows[:, 5].astype(np.uint16) - inter
    out: list[Hit] = []
    qid = int(query[0])
    for row, num, d in zip(rows, inter, den, strict=True):
        mid = int(row[0])
        if mid == qid:
            continue
        insert_top10(out, Hit(mid, int(num), max(1, int(d))))
    return out


def split_ranges(n: int, workers: int) -> list[tuple[int, int]]:
    chunk = max(1, n // workers)
    ranges = [(x, min(x + chunk, n)) for x in range(0, n, chunk)]
    ranges[-1] = (ranges[-1][0], n)
    return ranges


def raw_search(db: np.ndarray, query: np.ndarray, workers: int) -> np.ndarray:
    ranges = split_ranges(len(db), workers)
    if workers == 1:
        parts = [GenericSearch(query, db, 0.0, K_FETCH, 0, *ranges[0])]
    else:
        parts = []
        with cf.ThreadPoolExecutor(max_workers=workers) as exe:
            futures = [
                exe.submit(GenericSearch, query, db, 0.0, K_FETCH, 0, a, b)
                for a, b in ranges
            ]
            for f in cf.as_completed(futures):
                parts.append(f.result())
    return np.concatenate(parts) if parts else np.empty((0,), dtype=np.uint32)


def search_one(db: np.ndarray, query: np.ndarray, workers: int) -> list[Hit]:
    raw = raw_search(db, query, workers)
    idxs = raw["idx"].astype(np.int64, copy=False)
    return exact_hits(db, idxs, query)


def merge_hits(parts: Iterable[list[Hit]]) -> list[Hit]:
    out: list[Hit] = []
    for part in parts:
        for h in part:
            insert_top10(out, h)
    return out


class SearchState:
    def __init__(self, root: Path, need_union: bool = True, need_base: bool = True):
        self.root = root
        self.manifest = json.loads((root / "data/stage_a/stage_a_data_manifest.json").read_text())
        # FPSim2's pybind boundary is not safe with NumPy's memmap subclass on
        # this pinned build.  These two artifacts are small, so materialize
        # ordinary contiguous ndarrays and keep that boundary explicit.
        self.queries = np.fromfile(
            root / "data/stage_a/queries_u64x6.bin", dtype="<u8"
        ).reshape(self.manifest["query_rows"], 6)
        self.delta = np.fromfile(
            root / "data/stage_a/delta_u64x6.bin", dtype="<u8"
        ).reshape(self.manifest["delta_rows"], 6)
        self.base_engine = FPSim2Engine(str(root / "data/2026-08-18/fpsim2_fingerprints.h5")) if need_base else None
        self.union_engine = FPSim2Engine(str(root / "data/2026-08-25/fpsim2_fingerprints.h5")) if need_union else None

    def fresh(self, qi: int) -> list[Hit]:
        assert self.union_engine is not None
        query = np.array(self.queries[qi], dtype=np.uint64, copy=True)
        return search_one(self.union_engine.fps, query, 8)

    def overlay(self, qi: int) -> list[Hit]:
        assert self.base_engine is not None
        q = np.array(self.queries[qi], dtype=np.uint64, copy=True)
        return merge_hits((search_one(self.base_engine.fps, q, 8), search_one(self.delta, q, 1)))


def load_oracle(path: Path, wanted: set[int]) -> dict[int, list[Hit]]:
    out: dict[int, list[Hit]] = {q: [] for q in wanted}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            qi = int(row["query_index"])
            if qi in wanted:
                out[qi].append(Hit(int(row["result_id"]), int(row["num"]), int(row["den"])))
    for q in out:
        if len(out[q]) != K_FINAL:
            raise RuntimeError(f"oracle query {q} has {len(out[q])} rows")
    return out


def cmd_correctness(args: argparse.Namespace) -> None:
    state = SearchState(args.root)
    oracle = load_oracle(args.oracle, set(CORRECT))
    rows = []
    mismatch = 0
    for qi in CORRECT:
        for variant, fn in (("FRESH_UNION", state.fresh), ("BASE_PLUS_DELTA", state.overlay)):
            t0 = time.perf_counter_ns()
            got = fn(qi)
            ns = time.perf_counter_ns() - t0
            exp = oracle[qi]
            ok = got == exp
            mismatch += 0 if ok else 1
            rows.append({
                "query_index": qi,
                "variant": variant,
                "latency_ns": ns,
                "ok": int(ok),
                "got_hash": result_hash(got),
                "oracle_hash": result_hash(exp),
                "got": ";".join(f"{h.mol_id}:{h.num}/{h.den}" for h in got),
                "oracle": ";".join(f"{h.mol_id}:{h.num}/{h.den}" for h in exp),
            })
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summary = {"queries": len(CORRECT), "variant_rows": len(rows), "mismatches": mismatch}
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if mismatch:
        raise SystemExit(2)


def cmd_query_process(args: argparse.Namespace) -> None:
    state = SearchState(args.root)
    variants = [("FRESH_UNION", state.fresh), ("BASE_PLUS_DELTA", state.overlay)]
    if args.process_index % 2:
        variants.reverse()
    for _, fn in variants:
        for qi in range(16):
            fn(qi)
    rows = []
    for slot, (name, fn) in enumerate(variants):
        for qi in SEALED:
            t0 = time.perf_counter_ns()
            hits = fn(qi)
            ns = time.perf_counter_ns() - t0
            rows.append({
                "process_index": args.process_index,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "order_slot": slot,
                "variant": name,
                "query_index": qi,
                "latency_ns": ns,
                "result_hash": result_hash(hits),
            })
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for name, _ in variants:
        vals = [r["latency_ns"] / 1e6 for r in rows if r["variant"] == name]
        print(name, "n", len(vals), "median_ms", statistics.median(vals),
              "p95_ms", float(np.percentile(vals, 95)))


def cmd_update(args: argparse.Namespace) -> None:
    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if args.variant == "FRESH_UNION":
        t0 = time.perf_counter_ns()
        engine = FPSim2Engine(str(args.root / "data/2026-08-25/fpsim2_fingerprints.h5"))
        publish = engine.fps
        if publish.shape[0] != 41126808:
            raise RuntimeError("fresh union row mismatch")
    else:
        base = FPSim2Engine(str(args.root / "data/2026-08-18/fpsim2_fingerprints.h5"))
        if base.fps.shape[0] != 41106048:
            raise RuntimeError("base row mismatch")
        manifest = json.loads((args.root / "data/stage_a/stage_a_data_manifest.json").read_text())
        t0 = time.perf_counter_ns()
        publish = np.fromfile(args.root / "data/stage_a/delta_u64x6.bin", dtype="<u8").reshape(-1, 6)
        if publish.shape != (manifest["delta_rows"], 6):
            raise RuntimeError("delta shape mismatch")
        if np.any(publish[1:, 5] < publish[:-1, 5]):
            publish = np.ascontiguousarray(publish[np.lexsort((publish[:, 0], publish[:, 5]))])
        if publish.shape[0] != 20760:
            raise RuntimeError("delta row mismatch")
    elapsed = time.perf_counter_ns() - t0
    end_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "variant": args.variant,
        "process_index": args.process_index,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "elapsed_ns": elapsed,
        "ru_maxrss_before_kib": start_rss,
        "ru_maxrss_after_kib": end_rss,
        "published_rows": int(publish.shape[0]),
        "published_bytes": int(publish.nbytes),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def combined_hash(results: list[list[Hit]]) -> str:
    return sha256_bytes("|".join(result_hash(x) for x in results).encode())


def cmd_stage_b_cpu_bench(args: argparse.Namespace) -> None:
    state = SearchState(args.root, need_union=True, need_base=False)
    assert state.union_engine is not None
    db = state.union_engine.fps
    rows = []

    # Interactive comparator: one request uses all eight declared workers.
    for i in range(-10, 50):
        qi = 512 + ((i + 10) * 17) % 256
        t0 = time.perf_counter_ns()
        hits = state.fresh(qi)
        ns = time.perf_counter_ns() - t0
        if i >= 0:
            rows.append({
                "process_index": args.process_index,
                "mode": "INTERACTIVE",
                "iteration": i,
                "batch": 1,
                "host_ms": ns / 1e6,
                "qps": 1e9 / ns,
                "result_hash": result_hash(hits),
            })

    # Sustained comparator: eight concurrent one-worker requests, 64 requests
    # per retained batch, using one persistent service executor.
    with cf.ThreadPoolExecutor(max_workers=8) as exe:
        for i in range(-10, 50):
            start = ((i + 10) * 17) % 256
            qis = [512 + ((start + j) % 256) for j in range(64)]
            queries = [np.array(state.queries[q], dtype=np.uint64, copy=True) for q in qis]
            t0 = time.perf_counter_ns()
            futures = [exe.submit(search_one, db, q, 1) for q in queries]
            results = [f.result() for f in futures]
            ns = time.perf_counter_ns() - t0
            if i >= 0:
                rows.append({
                    "process_index": args.process_index,
                    "mode": "SUSTAINED64",
                    "iteration": i,
                    "batch": 64,
                    "host_ms": ns / 1e6,
                    "qps": 64e9 / ns,
                    "result_hash": combined_hash(results),
                })

    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for mode in ("INTERACTIVE", "SUSTAINED64"):
        vals = [float(r["host_ms"]) for r in rows if r["mode"] == mode]
        qps = [float(r["qps"]) for r in rows if r["mode"] == mode]
        print(mode, "n", len(vals), "median_ms", statistics.median(vals),
              "p95_ms", float(np.percentile(vals, 95)),
              "median_qps", statistics.median(qps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("correctness")
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.set_defaults(func=cmd_correctness)
    p = sub.add_parser("query-process")
    p.add_argument("--process-index", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_query_process)
    p = sub.add_parser("update")
    p.add_argument("--variant", choices=["FRESH_UNION", "BASE_PLUS_DELTA"], required=True)
    p.add_argument("--process-index", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_update)
    p = sub.add_parser("stage-b-cpu-bench")
    p.add_argument("--process-index", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_stage_b_cpu_bench)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
