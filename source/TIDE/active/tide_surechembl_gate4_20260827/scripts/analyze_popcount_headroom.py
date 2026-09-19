#!/usr/bin/env python3
"""Measure exact population-count interval headroom for threshold search."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tables as tb


THRESHOLDS = ((7, 10), (4, 5))


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def interval(query_popcount: int, num: int, den: int) -> tuple[int, int]:
    lower = ceil_div(num * query_popcount, den)
    upper = (den * query_popcount) // num
    return max(0, lower), min(256, upper)


def counts_from_h5(path: Path) -> np.ndarray:
    counts = np.zeros(257, dtype=np.int64)
    with tb.open_file(path, "r") as handle:
        bins = handle.root.config[4]
        rows = int(handle.root.fps.nrows)
    for popcount, (begin, end) in bins:
        counts[int(popcount)] = int(end) - int(begin)
    if int(counts.sum()) != rows:
        raise RuntimeError(f"bin count mismatch: {counts.sum()} != {rows}")
    return counts


def counts_from_u64x6(path: Path) -> np.ndarray:
    rows = np.memmap(path, dtype="<u8", mode="r").reshape(-1, 6)
    return np.bincount(rows[:, 5].astype(np.int64), minlength=257)


def percentiles(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def summarize(query_popcounts: np.ndarray, counts: np.ndarray) -> dict:
    total = int(counts.sum())
    cumulative = np.concatenate(([0], np.cumsum(counts)))
    output = {}
    for num, den in THRESHOLDS:
        lowers = np.empty(query_popcounts.size, dtype=np.int64)
        uppers = np.empty(query_popcounts.size, dtype=np.int64)
        eligible = np.empty(query_popcounts.size, dtype=np.int64)
        for index, query_popcount in enumerate(query_popcounts):
            lower, upper = interval(int(query_popcount), num, den)
            lowers[index] = lower
            uppers[index] = upper
            eligible[index] = cumulative[upper + 1] - cumulative[lower]
        fractions = eligible.astype(np.float64) / total
        key = f"{num / den:.2f}"
        output[key] = {
            "threshold_rational": [num, den],
            "eligible_rows": percentiles(eligible),
            "eligible_fraction": percentiles(fractions),
            "pruned_fraction": percentiles(1.0 - fractions),
            "lower_popcount": percentiles(lowers),
            "upper_popcount": percentiles(uppers),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-h5", type=Path, required=True)
    parser.add_argument("--union-h5", type=Path, required=True)
    parser.add_argument("--queries-u64x6", type=Path, required=True)
    parser.add_argument("--delta-u64x6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queries = np.memmap(args.queries_u64x6, dtype="<u8", mode="r").reshape(-1, 6)
    if queries.shape[0] != 4608:
        raise RuntimeError(f"unexpected query count: {queries.shape[0]}")
    query_popcounts = queries[:, 5].astype(np.int64)
    base_counts = counts_from_h5(args.base_h5)
    union_counts = counts_from_h5(args.union_h5)
    delta_counts = counts_from_u64x6(args.delta_u64x6)
    if not np.array_equal(base_counts + delta_counts, union_counts):
        raise RuntimeError("base + delta population-count bins do not equal union")

    split_ranges = {"calibration": (0, 512), "sealed_test": (512, 4608)}
    splits = {}
    for split, (begin, end) in split_ranges.items():
        splits[split] = summarize(query_popcounts[begin:end], union_counts)

    delta_nonempty = np.flatnonzero(delta_counts)
    result = {
        "experiment_id": "tide_20260827_segmented_popcount_headroom",
        "rows": {
            "base": int(base_counts.sum()),
            "delta": int(delta_counts.sum()),
            "union": int(union_counts.sum()),
        },
        "query_count": int(queries.shape[0]),
        "query_popcount": percentiles(query_popcounts),
        "delta_bins": {
            "nonempty_count": int(delta_nonempty.size),
            "min": int(delta_nonempty.min()),
            "max": int(delta_nonempty.max()),
            "row_count_per_nonempty_bin": percentiles(delta_counts[delta_nonempty]),
        },
        "splits": splits,
    }
    sealed = splits["sealed_test"]
    gates = {
        "G4_A_HEADROOM70": (
            sealed["0.70"]["eligible_fraction"]["median"] <= 0.85
            and sealed["0.70"]["eligible_fraction"]["p95"] <= 0.95
        ),
        "G4_A_HEADROOM80": (
            sealed["0.80"]["eligible_fraction"]["median"] <= 0.65
            and sealed["0.80"]["eligible_fraction"]["p95"] <= 0.85
        ),
    }
    result["gates"] = gates
    result["headroom_pass"] = all(gates.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
