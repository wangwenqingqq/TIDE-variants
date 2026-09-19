#!/usr/bin/env python3
"""Validate and summarize two formal nvMolKit complete-result processes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics


THRESHOLDS = ((7, 10), (4, 5))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load(path: pathlib.Path) -> dict[tuple[int, int, int], dict[str, str]]:
    rows: dict[tuple[int, int, int], dict[str, str]] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (
                int(row["query"]),
                int(row["threshold_num"]),
                int(row["threshold_den"]),
            )
            if key in rows:
                raise ValueError(f"duplicate request key {key} in {path}")
            rows[key] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--oracle", type=pathlib.Path, required=True)
    parser.add_argument("--queries", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if len(args.run) != 2:
        raise ValueError("the frozen formal contract requires exactly two processes")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")

    oracle_all = load(args.oracle)
    expected_keys = {
        (query, numerator, denominator)
        for query in range(args.queries)
        for numerator, denominator in THRESHOLDS
    }
    if not expected_keys.issubset(oracle_all):
        raise ValueError("oracle is missing a frozen request")
    runs = [load(path) for path in args.run]
    if any(set(run) != expected_keys for run in runs):
        raise ValueError("a formal process request set differs from the frozen contract")

    for key in expected_keys:
        expected = oracle_all[key]
        for run in runs:
            observed = run[key]
            if (
                observed["returned_count"] != expected["hits"]
                or observed["id_hash"] != expected["id_hash"]
                or observed["duplicate_id"] != "0"
            ):
                raise ValueError(f"exact integer-oracle mismatch for {key}")

    result: dict[str, object] = {
        "format": "nvmolkit060_exact_complete_result_summary_v1",
        "queries": args.queries,
        "requests_per_process": 2 * args.queries,
        "processes": 2,
        "exact_oracle_pass": True,
        "duplicate_id_requests": 0,
        "timing_scope": (
            "complete service wall: query H2D, released full double similarity matrix, "
            "GPU threshold/nonzero/stable-ID gather, result D2H, synchronization; "
            "setup and host sort/hash/oracle validation excluded"
        ),
        "thresholds": {},
    }
    for numerator, denominator in THRESHOLDS:
        process_summaries = []
        for run in runs:
            matching = [
                row
                for (query, num, den), row in run.items()
                if num == numerator and den == denominator
            ]
            wall = [1000 * float(row["service_seconds"]) for row in matching]
            gpu = [1000 * float(row["gpu_seconds"]) for row in matching]
            process_summaries.append(
                {
                    "p10_ms": percentile(wall, 0.10),
                    "median_ms": statistics.median(wall),
                    "p90_ms": percentile(wall, 0.90),
                    "p95_ms": percentile(wall, 0.95),
                    "min_ms": min(wall),
                    "max_ms": max(wall),
                    "median_gpu_ms": statistics.median(gpu),
                }
            )
        result["thresholds"][f"{numerator}_{denominator}"] = {
            "processes": process_summaries,
            "geometric_mean_process_median_ms": math.sqrt(
                process_summaries[0]["median_ms"]
                * process_summaries[1]["median_ms"]
            ),
            "conservative_max_process_p95_ms": max(
                process_summaries[0]["p95_ms"], process_summaries[1]["p95_ms"]
            ),
        }

    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
