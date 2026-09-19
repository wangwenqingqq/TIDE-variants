#!/usr/bin/env python3
"""Validate and summarize the frozen Gate-4 fresh-process GPU campaign."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


VARIANTS = ("released", "flat", "segmented")
THRESHOLDS = ((7, 10), (4, 5))


def distribution(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    order_rows = []
    for line in (args.run_dir / "order.csv").read_text().splitlines()[1:]:
        process, slot, variant, first, second, filename = line.split(",")
        order_rows.append(
            {
                "process_index": process,
                "slot": slot,
                "variant": variant,
                "threshold_order": f"{first},{second}",
                "file": filename,
            }
        )
    records = []
    file_errors = []
    for order in order_rows:
        path = args.run_dir / order["file"]
        rows = list(csv.DictReader(path.open())) if path.exists() else []
        if len(rows) != 512:
            file_errors.append({"file": str(path), "rows": len(rows)})
        for row in rows:
            records.append(
                {
                    "process": int(row["process_index"]),
                    "variant": row["variant"],
                    "threshold": (int(row["threshold_num"]), int(row["threshold_den"])),
                    "query": int(row["query_index"]),
                    "service_ms": float(row["service_ms"]),
                    "kernel_ms": float(row["kernel_ms"]),
                    "device_ms": float(row["device_ms"]),
                    "hits": int(row["hit_count"]),
                    "hash": int(row["result_hash"]),
                    "file": order["file"],
                }
            )

    keyed = {}
    duplicate_keys = []
    for row in records:
        key = (row["process"], row["variant"], row["threshold"], row["query"])
        if key in keyed:
            duplicate_keys.append(key)
        keyed[key] = row

    expected_keys = {
        (process, variant, threshold, query)
        for process in range(6)
        for variant in VARIANTS
        for threshold in THRESHOLDS
        for query in range(512, 768)
    }
    missing_keys = sorted(expected_keys - set(keyed))
    unexpected_keys = sorted(set(keyed) - expected_keys)
    result_mismatches = []
    for process in range(6):
        for threshold in THRESHOLDS:
            for query in range(512, 768):
                rows = [keyed.get((process, variant, threshold, query)) for variant in VARIANTS]
                if any(row is None for row in rows):
                    continue
                signatures = {(row["hits"], row["hash"]) for row in rows}
                if len(signatures) != 1:
                    result_mismatches.append(
                        {
                            "process": process,
                            "threshold": threshold,
                            "query": query,
                            "signatures": {
                                variant: (row["hits"], row["hash"])
                                for variant, row in zip(VARIANTS, rows, strict=True)
                            },
                        }
                    )

    summary_by_variant = {}
    process_summary = defaultdict(dict)
    for variant in VARIANTS:
        summary_by_variant[variant] = {}
        for threshold in THRESHOLDS:
            label = f"{threshold[0] / threshold[1]:.2f}"
            selected = np.array(
                [
                    row["service_ms"]
                    for row in records
                    if row["variant"] == variant and row["threshold"] == threshold
                ],
                dtype=np.float64,
            )
            summary_by_variant[variant][label] = {
                "service_ms": distribution(selected),
                "kernel_ms": distribution(
                    np.array(
                        [
                            row["kernel_ms"]
                            for row in records
                            if row["variant"] == variant
                            and row["threshold"] == threshold
                        ]
                    )
                ),
                "device_ms": distribution(
                    np.array(
                        [
                            row["device_ms"]
                            for row in records
                            if row["variant"] == variant
                            and row["threshold"] == threshold
                        ]
                    )
                ),
            }
            for process in range(6):
                process_values = np.array(
                    [
                        row["service_ms"]
                        for row in records
                        if row["variant"] == variant
                        and row["threshold"] == threshold
                        and row["process"] == process
                    ]
                )
                process_summary[process][f"{variant}_{label}"] = distribution(
                    process_values
                )

    paired_log_ratios = {threshold: defaultdict(list) for threshold in THRESHOLDS}
    combined_logs = defaultdict(list)
    for process in range(6):
        for threshold in THRESHOLDS:
            for query in range(512, 768):
                flat = keyed[(process, "flat", threshold, query)]["service_ms"]
                segmented = keyed[(process, "segmented", threshold, query)]["service_ms"]
                value = float(np.log(segmented / flat))
                paired_log_ratios[threshold][process].append(value)
                combined_logs[process].append(value)

    process_geomeans = {
        f"{threshold[0] / threshold[1]:.2f}": {
            str(process): float(np.exp(np.mean(paired_log_ratios[threshold][process])))
            for process in range(6)
        }
        for threshold in THRESHOLDS
    }
    process_geomeans["combined"] = {
        str(process): float(np.exp(np.mean(combined_logs[process])))
        for process in range(6)
    }

    rng = np.random.default_rng(20260827)
    bootstrap = {"0.70": [], "0.80": [], "combined": []}
    for _ in range(10000):
        sampled = rng.integers(0, 6, size=6)
        for threshold in THRESHOLDS:
            label = f"{threshold[0] / threshold[1]:.2f}"
            logs = np.concatenate([paired_log_ratios[threshold][int(p)] for p in sampled])
            bootstrap[label].append(float(np.exp(np.mean(logs))))
        logs = np.concatenate([combined_logs[int(p)] for p in sampled])
        bootstrap["combined"].append(float(np.exp(np.mean(logs))))

    paired = {}
    for label, values in bootstrap.items():
        array = np.array(values)
        if label == "combined":
            point_logs = np.concatenate([combined_logs[p] for p in range(6)])
        else:
            threshold = (7, 10) if label == "0.70" else (4, 5)
            point_logs = np.concatenate(
                [paired_log_ratios[threshold][p] for p in range(6)]
            )
        paired[label] = {
            "geometric_mean_ratio": float(np.exp(np.mean(point_logs))),
            "bootstrap_95_low": float(np.percentile(array, 2.5)),
            "bootstrap_95_high": float(np.percentile(array, 97.5)),
            "process_geometric_means": process_geomeans[label],
        }

    marginal_ratios = {}
    released_ratios = {}
    query_gate_components = {}
    for threshold in THRESHOLDS:
        label = f"{threshold[0] / threshold[1]:.2f}"
        flat_p95 = summary_by_variant["flat"][label]["service_ms"]["p95"]
        segmented_p95 = summary_by_variant["segmented"][label]["service_ms"]["p95"]
        released_p95 = summary_by_variant["released"][label]["service_ms"]["p95"]
        marginal_ratios[label] = segmented_p95 / flat_p95
        released_ratios[label] = {
            "flat_over_released_p95": flat_p95 / released_p95,
            "segmented_over_released_p95": segmented_p95 / released_p95,
        }
        query_gate_components[label] = marginal_ratios[label] <= 1.10

    contract_ok = not (
        file_errors
        or duplicate_keys
        or missing_keys
        or unexpected_keys
        or result_mismatches
    )
    query_gate = (
        contract_ok
        and all(query_gate_components.values())
        and paired["combined"]["bootstrap_95_high"] <= 1.10
    )
    result = {
        "experiment_id": "tide_20260827_segmented_popcount_single_gpu",
        "raw_record_count": len(records),
        "fresh_processes": 6,
        "retained_requests_per_variant": len(records) // 3,
        "contract_validation": {
            "file_error_count": len(file_errors),
            "duplicate_key_count": len(duplicate_keys),
            "missing_key_count": len(missing_keys),
            "unexpected_key_count": len(unexpected_keys),
            "result_mismatch_count": len(result_mismatches),
            "passed": contract_ok,
        },
        "marginal": summary_by_variant,
        "marginal_segmented_over_flat_p95": marginal_ratios,
        "released_context_p95_ratios": released_ratios,
        "paired_segmented_over_flat": paired,
        "per_process": {str(key): value for key, value in process_summary.items()},
        "gates": {
            "G4_B_RELEASED_TIMING_RESULTS_IDENTICAL": contract_ok,
            "G4_B_QUERY": query_gate,
        },
        "first_failures": {
            "files": file_errors[:3],
            "missing": missing_keys[:3],
            "unexpected": unexpected_keys[:3],
            "results": result_mismatches[:3],
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["contract_validation"], indent=2))
    print(json.dumps(result["marginal_segmented_over_flat_p95"], indent=2))
    print(json.dumps(result["paired_segmented_over_flat"], indent=2))
    print(json.dumps(result["gates"], indent=2))
    if not contract_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
