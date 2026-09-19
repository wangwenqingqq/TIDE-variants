#!/usr/bin/env python3
"""Independent NumPy verification of Gate-0 certificate counts and bounds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


BLOCK_SIZES = (32, 128, 512, 2048)
VERIFY_QUERIES = np.arange(512, 576, dtype=np.int64)


def bitcount_rows(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).sum(axis=1, dtype=np.uint16)
    table = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    return table[x.view(np.uint8).reshape(len(x), -1)].sum(axis=1, dtype=np.uint16)


def load_cert_csv(path: Path) -> dict[int, tuple[int, int]]:
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[int(row["query_index"])] = (
                int(row["surviving_blocks"]), int(row["surviving_items"])
            )
    return out


def load_exact_rows(path: Path, nq: int) -> np.ndarray:
    rows = np.empty((nq, 10), dtype=np.uint32)
    seen = np.zeros((nq, 10), dtype=bool)
    with path.open() as f:
        for r in csv.DictReader(f):
            q, rank = int(r["query_index"]), int(r["rank"])
            rows[q, rank] = int(r["row"])
            seen[q, rank] = True
    if not seen.all():
        raise RuntimeError("missing exact result rows")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cert-run", type=Path, required=True)
    ap.add_argument("--exact-run", type=Path, required=True)
    args = ap.parse_args()
    p = args.root / "data" / "prepared"
    manifest = json.loads((p / "data_manifest.json").read_text())
    n, nq = manifest["counts"]["union"], 4608
    fp = np.memmap(p / "union_fp_u64x4.bin", mode="r", dtype="<u8", shape=(n, 4))
    pc = np.memmap(p / "union_popcnt_u16.bin", mode="r", dtype="<u2", shape=(n,))
    qfp = np.memmap(p / "queries_fp_u64x4.bin", mode="r", dtype="<u8", shape=(nq, 4))
    qpc = np.memmap(p / "queries_popcnt_u16.bin", mode="r", dtype="<u2", shape=(nq,))
    num = np.fromfile(p / "certificates/kth_num_u16.bin", dtype="<u2")
    den = np.fromfile(p / "certificates/kth_den_u16.bin", dtype="<u2")
    exact_rows = load_exact_rows(args.exact_run / "union.csv", nq)

    report = {"verify_query_indices": VERIFY_QUERIES.tolist(), "block_sizes": {}}
    total_count_mismatch = total_top10_false_prune = total_sample_bound_violation = 0
    for bs in BLOCK_SIZES:
        blocks = (n + bs - 1) // bs
        bor = np.memmap(p / f"certificates/block{bs}_or_u64x4.bin", mode="r",
                        dtype="<u8", shape=(blocks, 4))
        bmin = np.memmap(p / f"certificates/block{bs}_minpc_u16.bin", mode="r",
                         dtype="<u2", shape=(blocks,))
        bmax = np.memmap(p / f"certificates/block{bs}_maxpc_u16.bin", mode="r",
                         dtype="<u2", shape=(blocks,))
        blen = np.memmap(p / f"certificates/block{bs}_len_u32.bin", mode="r",
                         dtype="<u4", shape=(blocks,))
        formal = load_cert_csv(args.cert_run / f"block{bs}.csv")
        count_mismatch = 0
        sample_bound_violation = 0
        sample_blocks = 0
        for q in VERIFY_QUERIES:
            cmax = bitcount_rows(np.bitwise_and(bor, qfp[q]))
            m = np.minimum(np.uint32(qpc[q]), cmax.astype(np.uint32))
            bstar = np.minimum(np.maximum(m, bmin.astype(np.uint32)), bmax.astype(np.uint32))
            cstar = np.minimum(m, bstar)
            ubden = qpc[q].astype(np.uint32) + bstar - cstar
            ubden[ubden == 0] = 1
            survive = cstar * int(den[q]) >= int(num[q]) * ubden
            got = (int(survive.sum()), int(blen[survive].sum(dtype=np.uint64)))
            if got != formal[int(q)]:
                count_mismatch += 1

            pruned = np.flatnonzero(~survive)
            if len(pruned):
                # Deterministic, query-dependent evenly spread audit blocks.
                seed = int.from_bytes(hashlib.sha256(f"{bs}:{int(q)}".encode()).digest()[:8], "little")
                rng = np.random.default_rng(seed)
                chosen = rng.choice(pruned, size=min(64, len(pruned)), replace=False)
                for b in chosen:
                    begin, end = int(b) * bs, min(n, (int(b) + 1) * bs)
                    inter = bitcount_rows(np.bitwise_and(fp[begin:end], qfp[q])).astype(np.uint32)
                    actual_den = int(qpc[q]) + pc[begin:end].astype(np.uint32) - inter
                    actual_den[actual_den == 0] = 1
                    # Every actual member must be <= the stored block upper bound.
                    bad = inter * int(ubden[b]) > int(cstar[b]) * actual_den
                    sample_bound_violation += int(np.count_nonzero(bad))
                    sample_blocks += 1

        # Full-query exact top-10 result blocks must all survive the bound.
        top10_false_prune = 0
        for q in range(nq):
            b = (exact_rows[q] // bs).astype(np.int64)
            cmax = bitcount_rows(np.bitwise_and(bor[b], qfp[q])).astype(np.uint32)
            m = np.minimum(np.uint32(qpc[q]), cmax)
            bstar = np.minimum(np.maximum(m, bmin[b].astype(np.uint32)),
                               bmax[b].astype(np.uint32))
            cstar = np.minimum(m, bstar)
            ubden = int(qpc[q]) + bstar - cstar
            ubden[ubden == 0] = 1
            survives = cstar * int(den[q]) >= int(num[q]) * ubden
            top10_false_prune += int(np.count_nonzero(~survives))

        report["block_sizes"][str(bs)] = {
            "independent_count_query_mismatches": count_mismatch,
            "sampled_pruned_blocks": sample_blocks,
            "sampled_member_bound_violations": sample_bound_violation,
            "full_query_top10_false_prunes": top10_false_prune,
        }
        total_count_mismatch += count_mismatch
        total_sample_bound_violation += sample_bound_violation
        total_top10_false_prune += top10_false_prune
    report["totals"] = {
        "independent_count_query_mismatches": total_count_mismatch,
        "sampled_member_bound_violations": total_sample_bound_violation,
        "full_query_top10_false_prunes": total_top10_false_prune,
    }
    print(json.dumps(report, indent=2))
    if any(report["totals"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
