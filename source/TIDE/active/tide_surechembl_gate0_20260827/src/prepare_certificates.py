#!/usr/bin/env python3
"""Build frozen block metadata and exact-answer task metrics for Gate-0."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


BLOCK_SIZES = (32, 128, 512, 2048)
CERTIFICATE_BYTES = 48  # 32-byte OR + packed/aligned min/max/start/length.


def percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p, method="linear"))


def read_exact_csv(path: Path, query_count: int) -> tuple[np.ndarray, ...]:
    ids = np.empty((query_count, 10), dtype=np.int64)
    num = np.empty((query_count, 10), dtype=np.uint16)
    den = np.empty((query_count, 10), dtype=np.uint16)
    seen = np.zeros((query_count, 10), dtype=bool)
    with path.open() as f:
        for row in csv.DictReader(f):
            q, rank = int(row["query_index"]), int(row["rank"])
            if not (0 <= q < query_count and 0 <= rank < 10):
                raise RuntimeError(f"out-of-range row in {path}: {row}")
            ids[q, rank] = int(row["result_id"])
            num[q, rank] = int(row["num"])
            den[q, rank] = int(row["den"])
            seen[q, rank] = True
    if not seen.all():
        raise RuntimeError(f"missing exact rows in {path}: {np.size(seen)-seen.sum()}")
    return ids, num, den


def build_or_blocks(fp: np.memmap, pc: np.memmap, block_size: int, out: Path) -> dict:
    n = len(fp)
    nblocks = (n + block_size - 1) // block_size
    full_blocks = n // block_size
    block_or = np.memmap(out / f"block{block_size}_or_u64x4.bin", mode="w+",
                         dtype="<u8", shape=(nblocks, 4))
    if full_blocks:
        view = fp[: full_blocks * block_size].reshape(full_blocks, block_size, 4)
        block_or[:full_blocks] = np.bitwise_or.reduce(view, axis=1)
    if full_blocks < nblocks:
        block_or[full_blocks] = np.bitwise_or.reduce(
            fp[full_blocks * block_size :], axis=0
        )
    block_or.flush()

    starts = np.arange(nblocks, dtype=np.uint64) * block_size
    ends = np.minimum(starts + block_size, n).astype(np.uint64)
    block_min = np.asarray(pc[starts], dtype="<u2")
    block_max = np.asarray(pc[ends - 1], dtype="<u2")
    block_len = np.asarray(ends - starts, dtype="<u4")
    block_min.tofile(out / f"block{block_size}_minpc_u16.bin")
    block_max.tofile(out / f"block{block_size}_maxpc_u16.bin")
    block_len.tofile(out / f"block{block_size}_len_u32.bin")
    return {
        "block_size": block_size,
        "blocks": int(nblocks),
        "certificate_bytes_per_block": CERTIFICATE_BYTES,
        "certificate_bytes_per_query": int(nblocks * CERTIFICATE_BYTES),
        "or_file": str(out / f"block{block_size}_or_u64x4.bin"),
        "minpc_file": str(out / f"block{block_size}_minpc_u16.bin"),
        "maxpc_file": str(out / f"block{block_size}_maxpc_u16.bin"),
        "len_file": str(out / f"block{block_size}_len_u32.bin"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--formal", type=Path, required=True)
    args = ap.parse_args()
    prepared = args.root / "data" / "prepared"
    cert = prepared / "certificates"
    cert.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((prepared / "data_manifest.json").read_text())
    nq = manifest["counts"]["calibration_queries"] + manifest["counts"]["test_queries"]
    n = manifest["counts"]["union"]
    fp = np.memmap(prepared / "union_fp_u64x4.bin", mode="r", dtype="<u8",
                   shape=(n, 4))
    pc = np.memmap(prepared / "union_popcnt_u16.bin", mode="r", dtype="<u2",
                   shape=(n,))
    qpc = np.memmap(prepared / "queries_popcnt_u16.bin", mode="r", dtype="<u2",
                    shape=(nq,))
    qids = np.memmap(prepared / "queries_id_i64.bin", mode="r", dtype="<i8",
                     shape=(nq,))
    if np.any(pc[1:] < pc[:-1]):
        raise RuntimeError("official FPSim2 row order is not monotone in population count")

    union_ids, union_num, union_den = read_exact_csv(args.formal / "union.csv", nq)
    base_ids, _, _ = read_exact_csv(args.formal / "base.csv", nq)
    kth_num = union_num[:, 9].copy()
    kth_den = union_den[:, 9].copy()
    kth_num.astype("<u2").tofile(cert / "kth_num_u16.bin")
    kth_den.astype("<u2").tofile(cert / "kth_den_u16.bin")

    added = np.fromfile(prepared / "added_ids_i64.bin", dtype="<i8")
    added.sort()
    delta_hit = np.isin(union_ids, added, assume_unique=False)
    self_leak = union_ids == qids[:, None]
    if np.any(self_leak):
        raise RuntimeError(f"self ID leaked into {int(self_leak.sum())} exact results")
    any_delta = delta_hit.any(axis=1)
    delta_per_query = delta_hit.sum(axis=1)
    stale_overlap = np.array(
        [len(set(union_ids[q]).intersection(base_ids[q])) for q in range(nq)],
        dtype=np.int16,
    )

    hist = np.bincount(np.asarray(pc, dtype=np.int64), minlength=257)
    cumulative = np.concatenate(([0], np.cumsum(hist, dtype=np.uint64)))
    lo = np.zeros(nq, dtype=np.int16)
    hi = np.full(nq, 256, dtype=np.int16)
    positive = kth_num > 0
    for i in np.flatnonzero(positive):
        a, num, den = int(qpc[i]), int(kth_num[i]), int(kth_den[i])
        lo[i] = math.ceil(a * num / den)
        hi[i] = min(256, (a * den) // num)
    popcount_candidates = cumulative[hi.astype(np.int64) + 1] - cumulative[lo]
    popcount_candidates.astype("<u8").tofile(cert / "popcount_candidate_count_u64.bin")
    popcount_bytes = popcount_candidates * 32
    popcount_bytes.astype("<u8").tofile(cert / "popcount_candidate_bytes_u64.bin")

    t0 = time.time()
    blocks = []
    for bs in BLOCK_SIZES:
        print(f"building block size {bs}", flush=True)
        blocks.append(build_or_blocks(fp, pc, bs, cert))

    cal = slice(0, manifest["counts"]["calibration_queries"])
    test = slice(manifest["counts"]["calibration_queries"], nq)
    summary = {
        "experiment_id": manifest["experiment_id"],
        "formal_exact_dir": str(args.formal),
        "certificate_bytes_per_block": CERTIFICATE_BYTES,
        "blocks": blocks,
        "counts": manifest["counts"],
        "task": {
            "calibration_any_delta_fraction": float(any_delta[cal].mean()),
            "test_any_delta_fraction": float(any_delta[test].mean()),
            "test_delta_results_per_query_median": percentile(delta_per_query[test], 50),
            "test_delta_results_per_query_p90": percentile(delta_per_query[test], 90),
            "test_stale_top10_recall_median": percentile(stale_overlap[test] / 10.0, 50),
            "test_stale_top10_recall_p10": percentile(stale_overlap[test] / 10.0, 10),
            "test_stale_top10_recall_mean": float((stale_overlap[test] / 10.0).mean()),
            "self_result_count": int(self_leak.sum()),
        },
        "exact_threshold": {
            "test_similarity_median": percentile(kth_num[test] / kth_den[test], 50),
            "test_similarity_p10": percentile(kth_num[test] / kth_den[test], 10),
            "test_similarity_p90": percentile(kth_num[test] / kth_den[test], 90),
            "test_query_popcount_median": percentile(qpc[test], 50),
        },
        "popcount_baseline": {
            "test_candidates_median": percentile(popcount_candidates[test], 50),
            "test_candidates_p90": percentile(popcount_candidates[test], 90),
            "test_scan_reduction_median": percentile(n / popcount_candidates[test], 50),
            "test_scan_reduction_p10_query": percentile(n / popcount_candidates[test], 10),
            "test_candidate_bytes_median": percentile(popcount_bytes[test], 50),
            "test_candidate_bytes_p90": percentile(popcount_bytes[test], 90),
        },
        "metadata_build_seconds": time.time() - t0,
    }
    (args.root / "results" / "pre_certificate_analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
