#!/usr/bin/env python3
"""Validate and summarize direction-balanced nvMolKit/TIDE process pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics


THRESHOLDS = ((7, 10), (4, 5))


def load(path: pathlib.Path) -> dict[tuple[int, int, int], dict[str, str]]:
    result = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (int(row["query"]), int(row["threshold_num"]), int(row["threshold_den"]))
            if key in result:
                raise ValueError(f"duplicate request {key} in {path}")
            result[key] = row
    return result


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = p * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def stats(values: list[float]) -> dict[str, float]:
    return {
        "p10_ms": percentile(values, 0.10),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvmolkit", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--tide", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--tide-reference", type=pathlib.Path, required=True)
    parser.add_argument("--integer-oracle", type=pathlib.Path, required=True)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if len(args.nvmolkit) != 2 or len(args.tide) != 2:
        raise ValueError("exactly two process files per system are required")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")

    expected_keys = {
        (q, n, d) for q in range(args.queries) for n, d in THRESHOLDS
    }
    nvm_runs = [load(path) for path in args.nvmolkit]
    tide_runs = [load(path) for path in args.tide]
    reference = load(args.tide_reference)
    integer_oracle = load(args.integer_oracle)
    for run in nvm_runs + tide_runs:
        if set(run) != expected_keys:
            raise ValueError("retained request set differs from the frozen contract")

    for key in expected_keys:
        expected_integer = integer_oracle[key]
        for run in nvm_runs:
            observed = run[key]
            if (
                observed["returned_count"] != expected_integer["hits"]
                or observed["id_hash"] != expected_integer["id_hash"]
                or observed["duplicate_id"] != "0"
            ):
                raise ValueError(f"nvMolKit exact mismatch {key}")
        expected_tide = reference[key]
        for run in tide_runs:
            observed = run[key]
            if observed["variant"] != "logical64" or observed["overflow"] != "0":
                raise ValueError(f"TIDE variant/overflow mismatch {key}")
            for field in ("candidate_rows", "hits", "result_hash"):
                if observed[field] != expected_tide[field]:
                    raise ValueError(f"TIDE exact reference mismatch {key} field={field}")

    output = {
        "format": "nvmolkit060_tide_same_gpu_paired_v1",
        "queries": args.queries,
        "requests_per_process": 2 * args.queries,
        "process_pairs": 2,
        "exact_pass": True,
        "thresholds": {},
        "timer_caveat": (
            "both timers end after complete host result availability; TIDE additionally "
            "includes host hit sort/hash while nvMolKit excludes host sort/hash/oracle validation"
        ),
    }
    all_gate = True
    for numerator, denominator in THRESHOLDS:
        process_rows = []
        pair_ratios = []
        for index in range(2):
            nvm_values = [
                1000 * float(row["service_seconds"])
                for (q, n, d), row in nvm_runs[index].items()
                if (n, d) == (numerator, denominator)
            ]
            tide_values = [
                float(row["service_ms"])
                for (q, n, d), row in tide_runs[index].items()
                if (n, d) == (numerator, denominator)
            ]
            nvm_stats = stats(nvm_values)
            tide_stats = stats(tide_values)
            ratio = nvm_stats["median_ms"] / tide_stats["median_ms"]
            pair_ratios.append(ratio)
            process_rows.append(
                {
                    "pair": index + 1,
                    "system_order": "nvMolKit->TIDE" if index == 0 else "TIDE->nvMolKit",
                    "nvmolkit": nvm_stats,
                    "tide_logical64": tide_stats,
                    "median_speedup_tide_over_nvmolkit": ratio,
                }
            )
        geometric = math.sqrt(pair_ratios[0] * pair_ratios[1])
        gate = min(pair_ratios) >= 10.0 and geometric >= 10.0
        all_gate = all_gate and gate
        output["thresholds"][f"{numerator}_{denominator}"] = {
            "pairs": process_rows,
            "geometric_mean_paired_median_speedup_tide_over_nvmolkit": geometric,
            "minimum_pair_speedup": min(pair_ratios),
            "decisive_10x_gate": gate,
        }
    output["all_thresholds_decisive_gate"] = all_gate
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if all_gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
