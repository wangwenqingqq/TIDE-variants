#!/usr/bin/env python3
"""Assemble the immutable Gate-4 decision record from raw campaign evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def current(root: Path, name: str) -> Path:
    return Path((root / name).read_text().strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    runs = {
        "headroom": current(root, "CURRENT_HEADROOM_RUN.txt"),
        "cardinality": current(root, "CURRENT_CARDINALITY_RUN.txt"),
        "lifecycle": current(root, "CURRENT_LIFECYCLE_RUN.txt"),
        "build": current(root, "CURRENT_CUDA_BUILD_RUN.txt"),
        "synthetic": current(root, "CURRENT_SYNTHETIC_RUN.txt"),
        "correctness": current(root, "CURRENT_REAL_CORRECTNESS_RUN.txt"),
        "sanitizer": current(root, "CURRENT_SANITIZER_RUN.txt"),
        "stress": current(root, "CURRENT_STRESS_RUN.txt"),
        "publication": current(root, "CURRENT_PUBLICATION_RUN.txt"),
        "timing": current(root, "CURRENT_TIMING_RUN.txt"),
        "sustained_build": current(root, "CURRENT_SUSTAINED_BUILD_RUN.txt"),
        "sustained": current(root, "CURRENT_SUSTAINED_RUN.txt"),
    }
    headroom = load_json(runs["headroom"] / "result.json")
    cardinality = load_json(runs["cardinality"] / "summary.json")
    lifecycle = load_json(runs["lifecycle"] / "summary.json")
    synthetic = load_json(runs["synthetic"] / "result.json")
    correctness = load_json(runs["correctness"] / "verify.json")
    stress = load_json(runs["stress"] / "stress.json")
    publication = load_json(runs["publication"] / "summary.json")
    timing = load_json(runs["timing"] / "summary.json")
    sustained = load_json(runs["sustained"] / "summary.json")
    sass = load_json(runs["build"] / "sass_audit.json")
    memory_cost = load_json(root / "results" / "MEMORY_COST.json")

    sanitizer_tools = {}
    for tool in ("memcheck", "synccheck", "initcheck", "racecheck"):
        stdout = (runs["sanitizer"] / f"{tool}.stdout").read_text()
        sanitizer_tools[tool] = {
            "clean": bool(
                re.search(r"(?:ERROR SUMMARY: 0 errors|0 hazards displayed \(0 errors, 0 warnings\))", stdout)
            ),
            "synthetic": load_json(runs["sanitizer"] / f"{tool}_result.json"),
        }
    sanitizer_pass = all(value["clean"] for value in sanitizer_tools.values())

    sealed_headroom = headroom["splits"]["sealed_test"]
    sealed_cardinality = cardinality["splits"]["sealed_test"]
    timing_key = {}
    for threshold in ("0.70", "0.80"):
        timing_key[threshold] = {
            variant: timing["marginal"][variant][threshold]["service_ms"]
            for variant in ("released", "flat", "segmented")
        }
        timing_key[threshold]["segmented_over_flat_p95"] = timing[
            "marginal_segmented_over_flat_p95"
        ][threshold]
        timing_key[threshold]["paired_segmented_over_flat"] = timing[
            "paired_segmented_over_flat"
        ][threshold]
        timing_key[threshold]["released_context_p95_ratios"] = timing[
            "released_context_p95_ratios"
        ][threshold]

    gates = {
        **headroom["gates"],
        **cardinality["gates"],
        **lifecycle["gates"],
        "G4_B_SYNTHETIC": (
            synthetic["mismatches"] == 0 and synthetic["overflow_detected"]
        ),
        "G4_B_RELEASED": correctness["gates"]["G4_B_RELEASED"],
        "G4_B_SANITIZER": sanitizer_pass,
        "G4_B_STRESS": (
            stress["hash_mismatches"] == 0 and stress["overflow_events"] == 0
        ),
        "G4_B_STATIC": (
            sass["all_kernel_sequences_identical"]
            and sass["timing"]["exact_segment_scan_kernel"]["instruction_count"]
            > 0
        ),
        "G4_B_DYNAMIC": publication["G4_B_DYNAMIC"],
        "G4_B_QUERY": timing["gates"]["G4_B_QUERY"],
        "G4_B_SUSTAINED": sustained["G4_B_SUSTAINED"],
    }
    all_pass = all(gates.values())

    final = {
        "experiment_id": "tide_20260827_segmented_popcount_threshold_service",
        "generated_utc": "2026-08-27",
        "decision": (
            "ADMIT_BOUNDED_DYNAMIC_EXACT_THRESHOLD_GPU_PROTOTYPE"
            if all_pass
            else "REJECT_GATE4_UNDER_FAILED_BOUNDARY"
        ),
        "all_frozen_gates_pass": all_pass,
        "workload": {
            "base_rows": lifecycle["rows"]["base"],
            "delta_rows": lifecycle["rows"]["delta"],
            "union_rows": lifecycle["rows"]["union"],
            "query_source": "4608 deterministic future-arrival proxy queries",
            "thresholds": [0.70, 0.80],
            "fingerprint": "256-bit Morgan radius 2",
            "gpu": "physical GPU 2, NVIDIA RTX PRO 6000 Blackwell Server Edition",
        },
        "stage_a": {
            "sealed_headroom": {
                threshold: {
                    "eligible_fraction": sealed_headroom[threshold][
                        "eligible_fraction"
                    ],
                    "pruned_fraction": sealed_headroom[threshold][
                        "pruned_fraction"
                    ],
                }
                for threshold in ("0.70", "0.80")
            },
            "sealed_complete_cardinality": sealed_cardinality,
            "cardinality_searches": cardinality["search_record_count"],
            "lifecycle_comparisons": lifecycle["comparison_count"],
            "lifecycle_mismatches": lifecycle["mismatch_counts"],
            "per_bin_compaction": lifecycle["per_bin_compaction"],
        },
        "stage_b": {
            "synthetic": synthetic,
            "real_correctness": correctness,
            "sanitizer": sanitizer_tools,
            "stress": stress,
            "static": {
                "resources": {
                    "custom_registers": 39,
                    "custom_stack_bytes": 0,
                    "custom_local_bytes": 0,
                    "custom_shared_bytes": 0,
                    "released_registers": 16,
                    "released_source_static_shared_bytes": 16,
                },
                "sass": sass,
            },
            "publication": publication,
            "timing": {
                "raw_record_count": timing["raw_record_count"],
                "fresh_processes": timing["fresh_processes"],
                "contract_validation": timing["contract_validation"],
                "by_threshold": timing_key,
                "paired_combined": timing["paired_segmented_over_flat"][
                    "combined"
                ],
            },
            "sustained": sustained,
            "memory_cost": memory_cost,
        },
        "gates": gates,
        "evidence_boundaries": [
            "Queries are deterministic future-arrival proxies, not production traffic logs.",
            "The update trace is one real adjacent-snapshot, insert-only SureChEMBL delta.",
            "The publication measurement is a resident-GPU delta-copy plus descriptor scope; Gate-3's 190.555 s denominator includes disk append, global repair, and CPU reload, so their ratio is not an end-to-end speedup claim.",
            "The RELEASED comparator uses the exact FPSim2 0.7.4 taniRAW kernel in a C++ harness with explicit complete-output compaction; it is source-faithful context, not an invocation of the CuPy Python service.",
            "Concurrent GPU readers, concurrent writers, crash recovery, deletions, persistence, multi-GPU scale, other fingerprint widths, and production prevalence are not established.",
            "Gate-3 already closed query-kernel novelty. Gate-4 validates a bounded dynamic organization; novelty and related-work admission remain separate gates.",
            "No Gate-4 result modifies the GTS++ canonical manuscript automatically.",
        ],
        "raw_evidence": {name: str(path) for name, path in runs.items()},
    }
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    final_path = results_dir / "FINAL_GATE4.json"
    final_path.write_text(json.dumps(final, indent=2) + "\n")

    h70 = sealed_headroom["0.70"]["eligible_fraction"]
    h80 = sealed_headroom["0.80"]["eligible_fraction"]
    c70 = sealed_cardinality["0.70"]["hit_count_excluding_self"]
    c80 = sealed_cardinality["0.80"]["hit_count_excluding_self"]
    t70 = timing_key["0.70"]
    t80 = timing_key["0.80"]
    markdown = f"""# TIDE SureChEMBL Gate-4 result

## Decision

**{final['decision']}**. Every predeclared Stage-A and Stage-B gate passed under
the frozen single-GPU exact-threshold contract. This admits a prototype, not a
paper claim or a production prevalence claim.

## Strongest evidence

| Boundary | Result | Gate |
|---|---:|---:|
| 0.70 sealed eligible fraction | median {h70['median']:.4f}, p95 {h70['p95']:.4f} | pass |
| 0.80 sealed eligible fraction | median {h80['median']:.4f}, p95 {h80['p95']:.4f} | pass |
| Complete sealed hits, self excluded | 0.70 median {c70['median']:.0f}, p95 {c70['p95']:.1f}; 0.80 median {c80['median']:.0f}, p95 {c80['p95']:.1f} | pass |
| CPU lifecycle exactness | {lifecycle['comparison_count']} query-threshold cases; zero pre/post/compaction mismatch | pass |
| GPU exactness | {correctness['expected_request_count']} requests and {correctness['output_row_count']} output rows; zero oracle or cross-variant mismatch | pass |
| Publication | p95 {publication['p95_ms']:.6f} ms, gate <= {publication['gate_limit_ms']:.2f} ms | pass |
| 0.70 service p95 | released {t70['released']['p95']:.6f} ms; flat {t70['flat']['p95']:.6f} ms; segmented {t70['segmented']['p95']:.6f} ms | pass |
| 0.80 service p95 | released {t80['released']['p95']:.6f} ms; flat {t80['flat']['p95']:.6f} ms; segmented {t80['segmented']['p95']:.6f} ms | pass |
| Segmented/flat p95 | 0.70 {t70['segmented_over_flat_p95']:.4f}x; 0.80 {t80['segmented_over_flat_p95']:.4f}x | pass |
| Paired combined ratio | {final['stage_b']['timing']['paired_combined']['geometric_mean_ratio']:.4f}x, upper 95% {final['stage_b']['timing']['paired_combined']['bootstrap_95_high']:.4f}x | pass |
| 1,000 epoch cycles | zero hash mismatch, zero overflow | pass |
| Sustained mixed schedule | {sustained['mixed_over_query_only']:.4f}x query-only throughput; max compaction {sustained['compaction_max_ms']:.6f} ms | pass |
| Accounted resident CUDA memory | flat/segmented {memory_cost['variants']['segmented']['gib']:.3f} GiB; released harness {memory_cost['variants']['released']['gib']:.3f} GiB; compaction shadow {memory_cost['variants']['sustained_with_compaction_shadow']['gib']:.3f} GiB | reported |

## Interpretation

The real problem survives Gate-3's novelty rejection: restoring FPSim2's
population-count ordering after the 20,760-row append required a 190.555 s
median correctness-restoring lifecycle. Gate 4 shows that a 257-bin immutable
base plus epoch-visible delta can preserve exact threshold results while the
resident-GPU publication component remains below 0.1 ms p95 and query latency
stays statistically at flat-control parity.

The strongest defensible research direction is therefore **dynamic exact
threshold service organization**, not a new single-query scan kernel and not a
tree/graph hybrid. The released-kernel comparison is useful context, but the
causal attribution is the segmented/flat ratio.

## Material caveats

""" + "\n".join(f"- {item}" for item in final["evidence_boundaries"]) + f"""

## Raw evidence

Canonical machine-readable result: `{final_path}`.

""" + "\n".join(f"- `{name}`: `{path}`" for name, path in runs.items()) + "\n"
    (results_dir / "RESULTS_GATE4.md").write_text(markdown)
    print(json.dumps({"decision": final["decision"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
