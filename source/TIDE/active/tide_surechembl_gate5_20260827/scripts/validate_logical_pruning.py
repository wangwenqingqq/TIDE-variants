#!/usr/bin/env python3
"""Validate exact BitBound composition across 64 immutable Gate-5 runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "min": min(values),
        "median": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-pc", type=Path, required=True)
    parser.add_argument("--union-pc", type=Path, required=True)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_pc = np.fromfile(args.base_pc, dtype=np.uint16)
    union_pc = np.fromfile(args.union_pc, dtype=np.uint16)
    delta = np.fromfile(args.delta, dtype=np.uint64).reshape(-1, 6)
    queries = np.fromfile(args.queries, dtype=np.uint64).reshape(-1, 6)
    base_counts = np.bincount(base_pc, minlength=257).astype(np.int64)
    union_counts = np.bincount(union_pc, minlength=257).astype(np.int64)
    run_counts = np.zeros((64, 257), dtype=np.int64)
    for row in delta:
        run = splitmix64(int(row[0])) % 64
        run_counts[run, int(row[5])] += 1
    if not np.array_equal(base_counts + run_counts.sum(axis=0), union_counts):
        raise ValueError("base plus logical runs does not equal fresh union bins")

    thresholds = ((7, 10), (4, 5))
    mismatches = 0
    prefix_mismatches = 0
    report = {}
    for num, den in thresholds:
        eligible_fraction = []
        unbounded_over_bounded = []
        for query in queries:
            query_pc = int(query[5])
            lower = max(0, (num * query_pc + den - 1) // den)
            upper = min(256, (den * query_pc) // num)
            compacted = int(union_counts[lower : upper + 1].sum())
            logical = int(base_counts[lower : upper + 1].sum()) + int(
                run_counts[:, lower : upper + 1].sum()
            )
            if compacted != logical:
                mismatches += 1
            prefix = int(base_counts[lower : upper + 1].sum())
            for run in range(64):
                prefix += int(run_counts[run, lower : upper + 1].sum())
            if prefix != compacted:
                prefix_mismatches += 1
            fraction = logical / len(union_pc)
            eligible_fraction.append(fraction)
            unbounded_over_bounded.append(len(union_pc) / max(logical, 1))
        report[f"{num}/{den}"] = {
            "eligible_fraction": summarize(eligible_fraction),
            "pruned_fraction": summarize([1.0 - value for value in eligible_fraction]),
            "unbounded_over_bounded_candidate_ratio": summarize(unbounded_over_bounded),
        }

    result = {
        "experiment_id": "tide_20260827_gate5_logical_pruning_attribution",
        "base_rows": int(len(base_pc)),
        "delta_rows": int(len(delta)),
        "union_rows": int(len(union_pc)),
        "queries": int(len(queries)),
        "runs": 64,
        "query_threshold_cases": int(len(queries) * len(thresholds)),
        "logical_vs_compacted_candidate_mismatches": mismatches,
        "prefix_reconstruction_mismatches": prefix_mismatches,
        "run_row_min": int(run_counts.sum(axis=1).min()),
        "run_row_max": int(run_counts.sum(axis=1).max()),
        "thresholds": report,
        "G5_N_SPECIFIC": mismatches == 0 and prefix_mismatches == 0,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
