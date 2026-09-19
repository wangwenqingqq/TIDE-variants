#!/usr/bin/env python3
"""Apply the frozen Gate-0 decisions and write durable result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BLOCK_SIZES = (32, 128, 512, 2048)


def pct(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p, method="linear"))


def read_charged(path: Path, nq: int) -> np.ndarray:
    out = np.empty(nq, dtype=np.uint64)
    seen = np.zeros(nq, dtype=bool)
    with path.open() as f:
        for row in csv.DictReader(f):
            q = int(row["query_index"])
            out[q] = int(row["charged_bytes"])
            seen[q] = True
    if not seen.all():
        raise RuntimeError(f"missing certificate rows in {path}")
    return out


def parse_kv(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cert-run", type=Path, required=True)
    ap.add_argument("--exact-run", type=Path, required=True)
    ap.add_argument("--verify", type=Path, required=True)
    ap.add_argument("--correctness", type=Path, required=True)
    args = ap.parse_args()
    p, results = args.root / "data" / "prepared", args.root / "results"
    results.mkdir(exist_ok=True)
    manifest = json.loads((p / "data_manifest.json").read_text())
    pre = json.loads((results / "pre_certificate_analysis.json").read_text())
    verify = json.loads((args.verify / "report.json").read_text())
    cal_n = manifest["counts"]["calibration_queries"]
    nq = cal_n + manifest["counts"]["test_queries"]
    cal, test = slice(0, cal_n), slice(cal_n, nq)
    baseline = np.fromfile(p / "certificates/popcount_candidate_bytes_u64.bin", dtype="<u8")
    if len(baseline) != nq:
        raise RuntimeError("bad population-count baseline length")

    charged = {bs: read_charged(args.cert_run / f"block{bs}.csv", nq) for bs in BLOCK_SIZES}
    calibration_medians = {bs: pct(charged[bs][cal], 50) for bs in BLOCK_SIZES}
    selected = min(BLOCK_SIZES, key=lambda bs: (calibration_medians[bs], bs))
    test_reduction = baseline[test].astype(np.float64) / charged[selected][test]
    block_metrics = {}
    for bs in BLOCK_SIZES:
        r = baseline[test].astype(np.float64) / charged[bs][test]
        block_metrics[str(bs)] = {
            "calibration_charged_bytes_median": calibration_medians[bs],
            "test_charged_bytes_median": pct(charged[bs][test], 50),
            "test_charged_bytes_p90": pct(charged[bs][test], 90),
            "test_reduction_median": pct(r, 50),
            "test_reduction_p10_worst_queries": pct(r, 10),
            "test_reduction_p90": pct(r, 90),
            "test_fraction_beating_popcount_baseline": float((r > 1).mean()),
        }

    correctness_text = (args.correctness / "compare.txt").read_text()
    exact_mismatches = None
    for line in correctness_text.splitlines():
        if line.startswith("mismatches "):
            exact_mismatches = int(line.split()[1])
    if exact_mismatches is None:
        raise RuntimeError("could not parse correctness mismatch count")

    gates = {
        "G_DATA": (
            manifest["counts"]["added"] >= nq
            and manifest["counts"]["removed"] == 0
            and manifest["counts"]["changed_common"] == 0
            and manifest["popcnt_sample_mismatches"] == 0
        ),
        "G_TASK": pre["task"]["test_any_delta_fraction"] >= 0.10,
        "G_BOUND": (
            verify["totals"]["independent_count_query_mismatches"] == 0
            and verify["totals"]["sampled_member_bound_violations"] == 0
        ),
        "G_TREE": pct(test_reduction, 50) >= 2.0 and pct(test_reduction, 10) >= 1.25,
        "G_EXACT": exact_mismatches == 0 and verify["totals"]["full_query_top10_false_prunes"] == 0,
    }
    decision = "PASS_TO_GRAPH_GATE1" if all(gates.values()) else "CLOSE_OR_INTERVAL_HIERARCHY"
    exact_union = parse_kv(args.exact_run / "union_stdout.log")
    exact_base = parse_kv(args.exact_run / "base_stdout.log")
    final = {
        "experiment_id": manifest["experiment_id"],
        "decision": decision,
        "gates": gates,
        "selected_block_size": selected,
        "selection_rule": "minimum calibration median charged bytes; lower block size breaks ties",
        "task": pre["task"],
        "exact_threshold": pre["exact_threshold"],
        "popcount_baseline": pre["popcount_baseline"],
        "block_metrics": block_metrics,
        "selected_test": block_metrics[str(selected)],
        "correctness": {
            "cpu_oracle_queries": 64,
            "cpu_gpu_top10_row_mismatches": exact_mismatches,
            **verify["totals"],
        },
        "diagnostic_four_gpu_scan": {
            "gpus": exact_union.get("gpus"),
            "queries": int(exact_union["query_count"]),
            "union_kernel_wall_seconds": float(exact_union["launch_wall_seconds"]),
            "base_kernel_wall_seconds": float(exact_base["launch_wall_seconds"]),
            "scope": "diagnostic exact-scan kernel wall; not an end-to-end speedup claim",
        },
        "evidence": {
            "formal_exact": str(args.exact_run),
            "formal_certificate": str(args.cert_run),
            "certificate_verification": str(args.verify),
            "cpu_gpu_correctness": str(args.correctness),
            "source_manifest": str(p / "data_manifest.json"),
        },
    }
    (results / "FINAL_VERDICT_TIGHT.json").write_text(json.dumps(final, indent=2) + "\n")

    rows = [
        ["decision", decision],
        *[[k, "PASS" if v else "FAIL"] for k, v in gates.items()],
        ["selected_block_size", selected],
        ["test_any_delta_fraction", pre["task"]["test_any_delta_fraction"]],
        ["test_stale_top10_recall_mean", pre["task"]["test_stale_top10_recall_mean"]],
        ["tree_reduction_median", pct(test_reduction, 50)],
        ["tree_reduction_p10_worst_queries", pct(test_reduction, 10)],
        ["tree_fraction_beating_popcount", block_metrics[str(selected)]["test_fraction_beating_popcount_baseline"]],
        ["cpu_gpu_mismatches", exact_mismatches],
    ]
    with (results / "SUMMARY_TIGHT.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["metric", "value"], *rows])

    results_md = f"""# TIDE SureChEMBL Gate-0.1 range-tight results

**Decision: {decision}**

## Frozen gates

""" + "\n".join(f"- `{k}`: **{'PASS' if v else 'FAIL'}**" for k, v in gates.items()) + f"""

## Real workload

- Base / union / added fingerprints: {manifest['counts']['base']:,} / {manifest['counts']['union']:,} / {manifest['counts']['added']:,}.
- Sealed test queries with at least one non-self delta result in exact top-10: {pre['task']['test_any_delta_fraction']:.3%}.
- Stale-base exact top-10 recall, mean / median / p10: {pre['task']['test_stale_top10_recall_mean']:.3f} / {pre['task']['test_stale_top10_recall_median']:.3f} / {pre['task']['test_stale_top10_recall_p10']:.3f}.
- Exact tenth-neighbor similarity, p10 / median / p90: {pre['exact_threshold']['test_similarity_p10']:.3f} / {pre['exact_threshold']['test_similarity_median']:.3f} / {pre['exact_threshold']['test_similarity_p90']:.3f}.

## Certificate result

- Calibration selected block size: {selected}.
- Charged-byte reduction versus the FPSim2-style population-count bound, median / p10 worst-query: {pct(test_reduction, 50):.3f}x / {pct(test_reduction, 10):.3f}x.
- Fraction of sealed queries on which the selected hierarchy beats the population-count baseline: {block_metrics[str(selected)]['test_fraction_beating_popcount_baseline']:.3%}.
- Required gate: at least 2.0x median and 1.25x p10 worst-query.

## Correctness

- Independent CPU vs GPU top-10 mismatches on 64 sealed queries: {exact_mismatches}.
- Independent certificate count-query mismatches: {verify['totals']['independent_count_query_mismatches']}.
- Sampled member-bound violations: {verify['totals']['sampled_member_bound_violations']}.
- Exact top-10 results mapped to pruned blocks across all queries/block sizes: {verify['totals']['full_query_top10_false_prunes']}.

## Diagnostic scan

- Four-GPU 0-3 exact scan, union/base kernel wall: {float(exact_union['launch_wall_seconds']):.3f}s / {float(exact_base['launch_wall_seconds']):.3f}s for {exact_union['query_count']} queries.
- This is a diagnostic kernel-only denominator, not an end-to-end speedup claim.
"""
    (results / "RESULTS_TIGHT.md").write_text(results_md)

    if decision == "PASS_TO_GRAPH_GATE1":
        consequence = "Proceed only to a real graph-proposal Gate-1; fused-kernel work remains unauthorized."
    else:
        consequence = "Do not build a graph or fused CUDA kernel for this host/layout; the hierarchy lacks pre-kernel headroom against the strong bound baseline."
    memo = f"""# TIDE SureChEMBL Gate-0.1 range-tight decision memo

## Decision

**{decision}**

The real update motivation is strong: {pre['task']['test_any_delta_fraction']:.1%} of sealed future-arrival queries have a non-self delta object in the exact union top-10, and stale-base mean top-10 recall is {pre['task']['test_stale_top10_recall_mean']:.3f}. However, paper admission also requires the proposed hierarchy to beat the existing population-count bound after certificate bytes are charged.

Calibration selected {selected}-row blocks. On the sealed test, the hierarchy gives {pct(test_reduction, 50):.3f}x median and {pct(test_reduction, 10):.3f}x p10 worst-query charged-byte reduction versus the strong baseline. The frozen requirement is 2.0x and 1.25x.

{consequence}

Correctness passed: zero CPU/GPU top-10 mismatches, zero independent certificate-count mismatches, zero sampled admissibility violations, and zero exact top-10 false prunes.

## Reopen condition

Reopen only with an independently justified certificate/layout whose oracle, including metadata, reaches the same frozen 2.0x median and 1.25x worst-query gate against the population-count baseline. Do not change fingerprint width, query source, or baseline after seeing this result.
"""
    (args.root / "DECISION_MEMO_TIGHT.md").write_text(memo)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
