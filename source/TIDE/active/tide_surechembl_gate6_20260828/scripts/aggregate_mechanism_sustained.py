#!/usr/bin/env python3
"""Aggregate the frozen six-pair Gate-6 mechanism sustained control.

The comparison intentionally excludes candidate_rows from the correctness
observable because the unbounded control is designed to admit more candidates.
All final result observables remain part of the exact equality check.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/mechanism_sustained/formal_surechembl_latest_v1"
RAW = ROOT / "results/raw/20260829T041400Z_mechanism_unbounded_sustained_formal_v1"

SOURCE_SHA256 = "dc3ee151e4e55a6dea342e6f36d039b4c46f9dc8773d669d4ffca54c9ff76d83"
BINARY_SHA256 = "f8700667528abb89332a7b4bed1d909086f7ec85cf1abd7acac2a0163c864239"
SASS_BODY_SHA256 = "92027514dfbd14260b084e919d1d16a2237f9169b0afab97b0ec144cfd8ff264"

KEY = ("task", "reader", "cycle", "threshold_num", "threshold_den", "query")
OBSERVABLE = ("hits", "result_hash", "overflow", "completed")
INT_FIELDS = set(KEY + OBSERVABLE + ("candidate_rows",))


def percentile(values: Iterable[float], q: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("empty percentile input")
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def stats(values: list[float]) -> dict[str, float]:
    return {
        "p10": percentile(values, 0.10),
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "mean": sum(values) / len(values),
    }


def geometric_mean(values: Iterable[float]) -> float:
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def exact_cluster_ci(pair_ratios: list[float]) -> dict[str, object]:
    replicates = [
        geometric_mean(pair_ratios[i] for i in sample)
        for sample in itertools.product(range(len(pair_ratios)), repeat=len(pair_ratios))
    ]
    return {
        "lower_95": percentile(replicates, 0.025),
        "upper_95": percentile(replicates, 0.975),
        "replicates": len(replicates),
        "method": "exact enumeration of all 6^6 pair-cluster resamples",
    }


def load_csv(path: Path) -> dict[tuple[int, ...], dict[str, object]]:
    rows: dict[tuple[int, ...], dict[str, object]] = {}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = {}
            for name, value in raw.items():
                row[name] = int(value) if name in INT_FIELDS else float(value) if name in {"service_ms", "kernel_ms"} else value
            key = tuple(int(row[name]) for name in KEY)
            if key in rows:
                raise ValueError(f"duplicate key in {path}: {key}")
            rows[key] = row
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    pairs: dict[str, object] = {}
    throughput_ratios: list[float] = []
    all_equal = True
    all_clean = True
    all_unbounded_not_smaller = True
    pooled_service = {"logical64": {"7/10": [], "4/5": [], "combined": []}, "unbounded64": {"7/10": [], "4/5": [], "combined": []}}
    pooled_candidate_ratios = {"7/10": [], "4/5": [], "combined": []}
    pair_p99_ratios = {"7/10": [], "4/5": [], "combined": []}

    for pair in range(1, 7):
        variants: dict[str, object] = {}
        rows = {}
        for variant in ("logical64", "unbounded64"):
            csv_path = OUT / f"pair{pair}_{variant}.csv"
            summary_path = Path(f"{csv_path}.summary.json")
            setup_path = Path(f"{csv_path}.setup.json")
            rows[variant] = load_csv(csv_path)
            summary = json.loads(summary_path.read_text())
            setup = json.loads(setup_path.read_text())
            exit_code = int((RAW / f"pair{pair}_{variant}.exit_code.txt").read_text().strip())
            stderr_empty = (RAW / f"pair{pair}_{variant}.stderr.log").stat().st_size == 0
            before_empty = (RAW / f"pair{pair}_{variant}.compute_apps_before.csv").stat().st_size == 0
            after_empty = (RAW / f"pair{pair}_{variant}.compute_apps_after.csv").stat().st_size == 0
            process_clean = bool(summary["pass"] and summary["missing_samples"] == 0 and summary["overflow_requests"] == 0 and exit_code == 0 and stderr_empty and before_empty and after_empty)
            all_clean &= process_clean

            latency: dict[str, object] = {}
            for label, threshold in (("7/10", (7, 10)), ("4/5", (4, 5))):
                values = [float(r["service_ms"]) for r in rows[variant].values() if (r["threshold_num"], r["threshold_den"]) == threshold]
                pooled_service[variant][label].extend(values)
                pooled_service[variant]["combined"].extend(values)
                latency[label] = {"count": len(values), "service_ms": stats(values)}
            combined = [float(r["service_ms"]) for r in rows[variant].values()]
            latency["combined"] = {"count": len(combined), "service_ms": stats(combined)}
            variants[variant] = {
                "throughput_qps": summary["throughput_qps"],
                "retained_wall_s": summary["retained_wall_s"],
                "process_clean": process_clean,
                "setup": setup,
                "latency": latency,
                "csv_sha256": sha256(csv_path),
            }

        logical = rows["logical64"]
        unbounded = rows["unbounded64"]
        same_keys = logical.keys() == unbounded.keys()
        mismatch_count = 0
        unbounded_smaller_count = 0
        candidate_ratio_by_threshold = {"7/10": [], "4/5": [], "combined": []}
        if same_keys:
            for key in logical:
                left, right = logical[key], unbounded[key]
                if any(left[field] != right[field] for field in OBSERVABLE):
                    mismatch_count += 1
                if int(right["candidate_rows"]) < int(left["candidate_rows"]):
                    unbounded_smaller_count += 1
                ratio = int(right["candidate_rows"]) / int(left["candidate_rows"])
                label = f"{left['threshold_num']}/{left['threshold_den']}"
                candidate_ratio_by_threshold[label].append(ratio)
                candidate_ratio_by_threshold["combined"].append(ratio)
                pooled_candidate_ratios[label].append(ratio)
                pooled_candidate_ratios["combined"].append(ratio)
        complete_equal = same_keys and mismatch_count == 0
        all_equal &= complete_equal
        all_unbounded_not_smaller &= unbounded_smaller_count == 0

        ratio = float(variants["unbounded64"]["throughput_qps"]) / float(variants["logical64"]["throughput_qps"])
        throughput_ratios.append(ratio)
        for label in ("7/10", "4/5", "combined"):
            l_p99 = variants["logical64"]["latency"][label]["service_ms"]["p99"]
            u_p99 = variants["unbounded64"]["latency"][label]["service_ms"]["p99"]
            pair_p99_ratios[label].append(u_p99 / l_p99)

        order = (RAW / f"pair{pair}.order.txt").read_text().strip().replace(" ", ",")
        pairs[str(pair)] = {
            "order": order,
            "same_request_keys": same_keys,
            "complete_observables_equal": complete_equal,
            "observable_mismatch_count": mismatch_count,
            "unbounded_candidate_smaller_count": unbounded_smaller_count,
            "variants": variants,
            "candidate_rows_unbounded64_over_logical64": {label: stats(values) for label, values in candidate_ratio_by_threshold.items()},
        }

    throughput_ci = exact_cluster_ci(throughput_ratios)
    throughput_gmean = geometric_mean(throughput_ratios)
    causal_pass = bool(all_equal and all_clean and all_unbounded_not_smaller and throughput_gmean <= 1.0 and throughput_ci["upper_95"] <= 1.0)
    sustained_latency = {}
    for label in ("7/10", "4/5", "combined"):
        logical_stats = stats(pooled_service["logical64"][label])
        unbounded_stats = stats(pooled_service["unbounded64"][label])
        sustained_latency[label] = {
            "within_pair_p99_unbounded64_over_logical64": pair_p99_ratios[label],
            "geometric_mean_p99_ratio": geometric_mean(pair_p99_ratios[label]),
            "marginal_pooled_p99_ratio": unbounded_stats["p99"] / logical_stats["p99"],
            "pooled_logical64_service_ms": logical_stats,
            "pooled_unbounded64_service_ms": unbounded_stats,
        }

    summary = {
        "experiment_id": "tide_20260829_gate6_mechanism_sustained_surechembl_latest",
        "evidence_state": "measured",
        "scope": "latest SureChEMBL, four host readers/private streams, 4096 complete retained requests per process, six direction-balanced logical64-vs-unbounded64 pairs, physical GPU2",
        "frozen_contract": "MECHANISM_CONTROL_CARD_20260829.md",
        "candidate": {
            "source_sha256": SOURCE_SHA256,
            "binary_sha256": BINARY_SHA256,
            "selected_words4_instruction_body_sha256": SASS_BODY_SHA256,
            "selected_instruction_body_identical_between_variants": True,
            "resources": {"registers": 34, "stack_bytes": 0, "shared_bytes": 0, "local_bytes": 0, "constant_bytes": 952},
        },
        "correctness_observable": list(KEY + OBSERVABLE),
        "candidate_rows_excluded_from_correctness_reason": "unbounded64 intentionally disables the exact population-count bound and must admit no fewer candidates",
        "all_complete_observables_equal": all_equal,
        "all_processes_complete_and_clean": all_clean,
        "all_unbounded_candidate_counts_not_smaller": all_unbounded_not_smaller,
        "pairs": pairs,
        "throughput": {
            "within_pair_unbounded64_over_logical64": throughput_ratios,
            "geometric_mean_ratio": throughput_gmean,
            "pair_cluster_bootstrap_95_ci": throughput_ci,
            "unbounded64_process_wins": sum(r > 1.0 for r in throughput_ratios),
            "order_split_geometric_mean": {
                "logical64_first": geometric_mean(throughput_ratios[i] for i in (0, 2, 4)),
                "unbounded64_first": geometric_mean(throughput_ratios[i] for i in (1, 3, 5)),
            },
        },
        "sustained_latency": sustained_latency,
        "candidate_work": {label: stats(values) for label, values in pooled_candidate_ratios.items()},
        "causal_control_gate": {
            "throughput_point_ratio_max": 1.0,
            "throughput_upper_95_max": 1.0,
            "observed_throughput_ratio": throughput_gmean,
            "observed_throughput_upper_95": throughput_ci["upper_95"],
            "pass": causal_pass,
        },
        "SureChEMBL_G6_MECHANISM_SUSTAINED_CONTROL": causal_pass,
        "aggregate_G6_MECHANISM": "PARTIAL_SURECHEMBL_ONLY" if causal_pass else "FAIL",
        "aggregate_caveat": "ChEMBL 37 source-integrity blocker prevents the frozen second-corpus mechanism gate.",
    }
    summary_path = OUT / "FORMAL_SUSTAINED_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    raw_summary = {
        "primary_summary": str(summary_path.relative_to(ROOT)),
        "primary_summary_sha256": sha256(summary_path),
        "source_files": sorted(str(p.relative_to(ROOT)) for p in OUT.glob("pair*")),
        "decision": "SURECHEMBL_CAUSAL_CONTROL_PASS" if causal_pass else "SURECHEMBL_CAUSAL_CONTROL_FAIL",
    }
    (RAW / "FORMAL_AGGREGATE.json").write_text(json.dumps(raw_summary, indent=2) + "\n")
    print(json.dumps({
        "all_complete_observables_equal": all_equal,
        "all_processes_complete_and_clean": all_clean,
        "throughput_geometric_mean_unbounded_over_logical": throughput_gmean,
        "throughput_exact_ci": throughput_ci,
        "combined_p99_ratio": sustained_latency["combined"]["geometric_mean_p99_ratio"],
        "candidate_ratio_median": summary["candidate_work"]["combined"]["median"],
        "pass": causal_pass,
        "summary": str(summary_path),
    }, indent=2))


if __name__ == "__main__":
    main()
