#!/usr/bin/env python3
"""Summarize the frozen Gate-5 compacted versus 64-run A/B."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "p10": percentile(values, 0.10),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "mean": statistics.fmean(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("pair*_*.csv"))
    if len(files) != 12:
        raise ValueError(f"expected 12 process files, found {len(files)}")
    rows_by_pair_variant: dict[tuple[int, str], list[dict[str, str]]] = {}
    marginal: dict[tuple[str, str], list[float]] = defaultdict(list)
    orders: dict[int, list[str]] = defaultdict(list)
    correctness_mismatches = 0
    candidate_mismatches = 0
    reference: dict[tuple[str, str], tuple[str, str]] = {}
    for path in files:
        stem = path.stem
        pair = int(stem.split("_")[0][4:])
        variant = stem.split("_")[2]
        rows = load(path)
        rows_by_pair_variant[(pair, variant)] = rows
        orders[pair].append((int(stem.split("_")[1][3:]), variant))
        for row in rows:
            threshold = f"{row['threshold_num']}/{row['threshold_den']}"
            marginal[(variant, threshold)].append(float(row["service_ms"]))
            key = (threshold, row["query_index"])
            value = (row["hit_count"], row["result_hash"])
            if key in reference and reference[key] != value:
                correctness_mismatches += 1
            else:
                reference[key] = value
            candidate = row["candidate_rows"]
            candidate_key = ("candidate:" + threshold, row["query_index"])
            candidate_value = (candidate, candidate)
            if candidate_key in reference and reference[candidate_key] != candidate_value:
                candidate_mismatches += 1
            else:
                reference[candidate_key] = candidate_value

    per_pair: list[dict[str, object]] = []
    log_ratios_by_threshold: dict[str, list[float]] = defaultdict(list)
    combined_logs: list[float] = []
    for pair in range(6):
        compacted = rows_by_pair_variant[(pair, "compacted")]
        runs = rows_by_pair_variant[(pair, "runs")]
        row = {"pair": pair, "order": [v for _, v in sorted(orders[pair])]}
        pair_logs = []
        for threshold in ("7/10", "4/5"):
            compacted_values = [
                float(item["service_ms"])
                for item in compacted
                if f"{item['threshold_num']}/{item['threshold_den']}" == threshold
            ]
            run_values = [
                float(item["service_ms"])
                for item in runs
                if f"{item['threshold_num']}/{item['threshold_den']}" == threshold
            ]
            ratio = percentile(run_values, 0.95) / percentile(compacted_values, 0.95)
            row[threshold] = {
                "compacted_p95_ms": percentile(compacted_values, 0.95),
                "runs_p95_ms": percentile(run_values, 0.95),
                "runs_over_compacted": ratio,
            }
            log_ratios_by_threshold[threshold].append(math.log(ratio))
            pair_logs.append(math.log(ratio))
        combined = statistics.fmean(pair_logs)
        combined_logs.append(combined)
        row["combined_geomean_ratio"] = math.exp(combined)
        per_pair.append(row)

    rng = random.Random(args.seed)

    def interval(log_values: list[float]) -> tuple[float, float]:
        estimates = []
        for _ in range(args.bootstrap):
            sample = [rng.choice(log_values) for _ in log_values]
            estimates.append(math.exp(statistics.fmean(sample)))
        return percentile(estimates, 0.025), percentile(estimates, 0.975)

    paired = {}
    gate = True
    for threshold, logs in log_ratios_by_threshold.items():
        ratio = math.exp(statistics.fmean(logs))
        low, high = interval(logs)
        paired[threshold] = {
            "geomean_ratio": ratio,
            "bootstrap_95": [low, high],
            "process_pair_wins": sum(value < 0 for value in logs),
            "pair_count": len(logs),
        }
        gate = gate and ratio <= 1.10 and high <= 1.15
    combined_ratio = math.exp(statistics.fmean(combined_logs))
    combined_interval = interval(combined_logs)
    result = {
        "experiment_id": "tide_20260827_gate5_fragmentation_ab",
        "estimator": "per-pair service-latency p95 ratio; process-cluster bootstrap",
        "files": [str(path) for path in files],
        "correctness_mismatches": correctness_mismatches,
        "candidate_count_mismatches": candidate_mismatches,
        "marginal": {
            variant: {
                threshold: summarize(marginal[(variant, threshold)])
                for threshold in ("7/10", "4/5")
            }
            for variant in ("compacted", "runs")
        },
        "per_pair": per_pair,
        "paired": paired,
        "combined": {
            "geomean_ratio": combined_ratio,
            "bootstrap_95": list(combined_interval),
            "process_pair_wins": sum(value < 0 for value in combined_logs),
            "pair_count": len(combined_logs),
        },
        "G5_FRAGMENT": gate
        and correctness_mismatches == 0
        and candidate_mismatches == 0,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["paired"], indent=2))
    print(json.dumps({"combined": result["combined"], "G5_FRAGMENT": result["G5_FRAGMENT"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
