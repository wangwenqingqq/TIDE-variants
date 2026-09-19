"""Preserve the valid per-run pinning campaign and compare the sole staging-lifetime change."""
import argparse
import csv
import gzip
import json
from pathlib import Path

from analyze import BATCHES, POLICIES, digest, gm, require


def rows(path):
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing existing ablation artifact")
    before = json.loads((args.before / "summary.json").read_text())
    after = json.loads((args.after / "summary.json").read_text())
    require(before["counts"] == after["counts"], "ablation workload count mismatch")
    require(digest(args.before / "input_manifest.json") == digest(args.after / "input_manifest.json"), "ablation input changed")
    semantic_fields = ("rows", "candidate_rows", "hits", "output_bytes", "result_hash", "epoch", "batch", "threshold", "first_query", "policy", "rotation", "mode", "phase", "cap_mib")
    compared = 0
    for file in ("sequential_requests.csv.gz", "concurrent_requests.csv.gz"):
        a, b = rows(args.before / file), rows(args.after / file)
        require(len(a) == len(b), "ablation request schedule mismatch")
        for x, y in zip(a, b):
            # Growth can acquire a different epoch after the implementation improves writer latency.
            if x["mode"] == "growth":
                continue
            require(all(x[k] == y[k] for k in semantic_fields), "fixed-visible ablation semantic mismatch")
            compared += 1
    a = list(csv.DictReader((args.before / "maintenance.csv").open()))
    b = list(csv.DictReader((args.after / "maintenance.csv").open()))
    require(len(a) == len(b), "ablation maintenance event count mismatch")
    for x, y in zip(a, b):
        for key in ("rotation", "mode", "policy", "batch", "epoch", "status", "rows", "runs", "merges", "deferred", "uploaded_bytes", "merge_host_rw_bytes", "cap_mib"):
            require(x[key] == y[key], "ablation maintenance work mismatch: " + key)
    comparison = {}
    for batch in BATCHES:
        comparison[str(batch)] = {}
        for policy in POLICIES:
            first = before["concurrency"][str(batch)][policy]
            second = after["concurrency"][str(batch)][policy]
            comparison[str(batch)][policy] = {
                "per_run_pinning_shadow_over_static_service": first["shadow_over_static_service"],
                "reused_pinning_shadow_over_static_service": second["shadow_over_static_service"],
                "per_run_pinning_shadow_over_static_p99": first["shadow_over_static_p99"],
                "reused_pinning_shadow_over_static_p99": second["shadow_over_static_p99"],
                "p99_ratio_of_ratios": gm(b / a for a, b in zip(first["shadow_over_static_p99"]["process_ratios"],
                                                               second["shadow_over_static_p99"]["process_ratios"]))}
    result = {"change": "Only move 8 MiB pinned staging registration/unregistration from each Run upload to Writer lifetime",
              "fixed_visible_batch_semantics_compared": compared, "maintenance_work_events_compared": len(a),
              "workload_counts_per_campaign": before["counts"], "comparison": comparison,
              "before_summary_sha256": digest(args.before / "summary.json"),
              "after_summary_sha256": digest(args.after / "summary.json"), "analysis_sha256": digest(Path(__file__)),
              "limitations": ["separate consecutive campaigns, not randomized interleaving", "ratio-of-ratios is descriptive, not a formal causal confidence interval",
                              "growth visible epoch changes with publication speed and must not be treated as same-input latency comparison",
                              "maintenance overlap exposure changes naturally when writer speed changes"]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"fixed_visible_batches_compared": compared, "maintenance_events_compared": len(a),
                      "Q1_all_delta": comparison["1"]["all_delta"]}, indent=2))


if __name__ == "__main__":
    main()
