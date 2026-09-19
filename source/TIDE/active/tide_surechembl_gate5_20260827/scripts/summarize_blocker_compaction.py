#!/usr/bin/env python3
"""Summarize the frozen six-pair canonical compaction campaign."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    processes = []
    pair_ratios = []
    for pair in range(args.pairs):
        pair_values = []
        expected_orders = (
            ("control-first", "mixed-first")
            if pair % 2 == 0
            else ("mixed-first", "control-first")
        )
        for position, expected_order in enumerate(expected_orders):
            path = args.input / f"pair{pair}_pos{position}_{expected_order}.json"
            value = json.loads(path.read_text())
            value["path"] = str(path)
            value["expected_order"] = expected_order
            value["eligible"] = (
                value.get("order") == expected_order
                and value.get("B5_COMPACTION_PROCESS") is True
            )
            processes.append(value)
            pair_values.append(float(value["mixed_over_control_p99"]))
        pair_ratios.append(geomean(pair_values))

    rng = random.Random(args.seed)
    bootstrap = []
    for _ in range(args.bootstrap):
        selected = [rng.choice(pair_ratios) for _ in pair_ratios]
        bootstrap.append(geomean(selected))
    estimate = geomean(pair_ratios)
    interval = [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)]
    all_processes_valid = all(value["eligible"] for value in processes)
    result = {
        "experiment_id": "tide_20260828_gate5_blocker_compaction",
        "pairs": args.pairs,
        "processes": len(processes),
        "orders": [value["order"] for value in processes],
        "pair_p99_ratios": pair_ratios,
        "paired_geomean_p99_ratio": estimate,
        "process_cluster_bootstrap_95": interval,
        "process_wins": sum(
            float(value["mixed_over_control_p99"]) <= 1.0 for value in processes
        ),
        "row_mismatches": sum(int(value["row_mismatches"]) for value in processes),
        "byte_mismatches": sum(int(value["byte_mismatches"]) for value in processes),
        "query_mismatches": sum(
            int(value["control_mismatches"]) + int(value["mixed_mismatches"])
            for value in processes
        ),
        "overflow_events": sum(
            int(value["control_overflow"]) + int(value["mixed_overflow"])
            for value in processes
        ),
        "minimum_captures_during_compaction": min(
            int(value["captures_during_compaction"]) for value in processes
        ),
        "maximum_snapshot_capture_us": max(
            float(value["max_snapshot_capture_us"]) for value in processes
        ),
        "all_processes_valid": all_processes_valid,
        "B5_COMPACTION_CLOSURE": all_processes_valid and interval[1] <= 1.25,
        "process_records": processes,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["B5_COMPACTION_CLOSURE"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
