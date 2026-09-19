#!/usr/bin/env python3
"""Validate GPUSimilarity runs against an independent exact integer oracle."""

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
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--queries", type=int, required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.rows <= 0 or args.queries <= 0 or args.cap <= 0:
        raise ValueError("rows, queries, and cap must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")

    oracle_all = load(args.oracle)
    expected_keys = {
        (query, numerator, denominator)
        for query in range(args.queries)
        for numerator, denominator in THRESHOLDS
    }
    if not expected_keys.issubset(oracle_all):
        raise ValueError("oracle is missing requests from the frozen contract")
    oracle = {key: oracle_all[key] for key in expected_keys}
    runs = [load(path) for path in args.run]
    if any(set(run) != expected_keys for run in runs):
        raise ValueError("run request set differs from the frozen contract")

    max_observed_hits = 0
    for key, expected in oracle.items():
        max_observed_hits = max(max_observed_hits, int(expected["hits"]))
        for run in runs:
            observed = run[key]
            if (observed["returned_count"], observed["id_hash"]) != (
                expected["hits"],
                expected["id_hash"],
            ):
                raise ValueError(f"exact integer-oracle mismatch for {key}")
            if observed["duplicate_id"] != "0":
                raise ValueError(f"duplicate output ID for {key}")
            approximate = int(observed["approximate_count"])
            returned = int(observed["returned_count"])
            if approximate != returned:
                raise ValueError(f"approximate/returned count mismatch for {key}")
            if returned >= args.cap:
                raise ValueError(f"result cap reached for {key}")

    result: dict[str, object] = {
        "format": "gpusimilarity_direct_exact_summary_v1",
        "rows": args.rows,
        "queries": args.queries,
        "requests_per_process": 2 * args.queries,
        "processes": len(runs),
        "exact_oracle_pass": True,
        "duplicate_id_requests": 0,
        "result_cap": args.cap,
        "maximum_oracle_hits": max_observed_hits,
        "timing_scope": (
            "synchronous released FingerprintDB::search; setup and numeric-ID "
            "decode/sort/hash excluded"
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
            latencies = [1000.0 * float(row["search_seconds"]) for row in matching]
            validation = [
                1000.0 * float(row["validation_seconds"]) for row in matching
            ]
            process_summaries.append(
                {
                    "mean_ms": statistics.mean(latencies),
                    "median_ms": statistics.median(latencies),
                    "p95_ms": percentile(latencies, 0.95),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "median_validation_ms": statistics.median(validation),
                }
            )
        threshold_result: dict[str, object] = {"processes": process_summaries}
        if len(process_summaries) == 2:
            threshold_result["geometric_mean_process_median_ms"] = math.sqrt(
                process_summaries[0]["median_ms"]
                * process_summaries[1]["median_ms"]
            )
            threshold_result["conservative_max_process_p95_ms"] = max(
                process_summaries[0]["p95_ms"],
                process_summaries[1]["p95_ms"],
            )
        result["thresholds"][f"{numerator}_{denominator}"] = threshold_result

    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
