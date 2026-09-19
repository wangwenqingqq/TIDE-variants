"""Audit all retained samples before deriving a bounded pilot summary."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as stat

POLICIES = ("all_delta", "periodic2", "size_tiered", "compact")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def gm(values):
    values = list(values)
    if not values or any(x <= 0 for x in values):
        raise ValueError("geometric means need positive observations")
    return math.exp(stat.mean(math.log(x) for x in values))


def process_interval(values):
    logs = [math.log(x) for x in values]
    center = stat.mean(logs)
    radius = 3.182446305284263 * stat.stdev(logs) / math.sqrt(4)
    return {"process_ratios": values, "geomean": math.exp(center),
            "process_log_t95": [math.exp(center - radius), math.exp(center + radius)]}


def audit_queries(rows, expected):
    if len(rows) != expected:
        raise ValueError("request row count mismatch")
    keys = [(r["policy"], int(r["epoch"]), int(r["threshold"]), int(r["query"])) for r in rows]
    if len(set(keys)) != expected:
        raise ValueError("duplicate or missing request keys")
    for row in rows:
        if row["cpu_complete_match"] != "1" or row["overflow"] != "0":
            raise ValueError("incorrect or truncated output found")
        if row["hits"] != row["observed_hits"]:
            raise ValueError("observed/materialized hit count differs")
        if int(row["tracked_live_device_bytes"]) > 256 << 20:
            raise ValueError("budget exceeded")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--cpu-ready", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, output = args.raw, args.output
    complete = json.loads((raw / "COLLECTION_COMPLETE.json").read_text())
    manifest = json.loads((raw / "MANIFEST.json").read_text())
    source = json.loads(args.input_manifest.read_text())
    cpu = json.loads(args.cpu_ready.read_text())
    if digest(args.cpu_ready) != manifest["cpu_ready_sha256"]:
        raise ValueError("CPU receipt hash mismatch")
    for name in manifest["commands"]:
        record = json.loads((raw / f"{name}.process.json").read_text())
        if record["failure"] is not None or record["returncode"] != 0:
            raise ValueError("failed process in completed campaign")
    for name in ("01_fixture", "02_memcheck", "02_synccheck"):
        audit_queries(read_csv(raw / f"{name}.queries.csv"), 1792)
        if "completed_requests=1792 old_epoch_checks=48" not in (raw / f"{name}.log").read_text():
            raise ValueError("fixture completion line absent")
    for name in ("02_memcheck", "02_synccheck"):
        if "ERROR SUMMARY: 0 errors" not in (raw / f"{name}.log").read_text():
            raise ValueError("sanitizer failed or did not finish")
    if "overflow_guard_pass=1 empty_slice_pass=1 requested_hits=65537" not in (raw / "00_guards.log").read_text():
        raise ValueError("guard test completion absent")
    samples, maintenance, warmups = [], [], []
    by_process = []
    final_shapes = {}
    for process in range(4):
        rows = read_csv(raw / f"pilot{process}.queries.csv")
        audit_queries(rows, 1792)
        index = {(r["policy"], int(r["epoch"]), int(r["threshold"]), int(r["query"])): r for r in rows}
        for epoch in range(7):
            for threshold in (70, 80):
                for query in range(32):
                    group = [index[(p, epoch, threshold, query)] for p in POLICIES]
                    for key in ("rows", "candidate_rows", "hits", "result_hash", "output_bytes"):
                        if len({r[key] for r in group}) != 1:
                            raise ValueError("same-epoch controls differ in semantic work: " + key)
        for row in rows:
            row["process"] = process
            row["query_arrival_epoch"] = source["query_labels"][int(row["query"])]["arrival_epoch"]
        mrows = read_csv(raw / f"pilot{process}.maintenance.csv")
        if len(mrows) != 28:
            raise ValueError("maintenance event count mismatch")
        if {(r["policy"], int(r["epoch"])) for r in mrows} != {(p, e) for p in POLICIES for e in range(7)}:
            raise ValueError("maintenance event keys incomplete")
        if sum(int(r["old_epoch_complete_checks"]) for r in mrows) != 48:
            raise ValueError("old-owner checks incomplete")
        for row in mrows:
            row["process"] = process
            if int(row["tracked_peak_device_bytes"]) > 256 << 20:
                raise ValueError("peak budget exceeded")
            if int(row["epoch"]) == 6:
                final_shapes[row["policy"]] = int(row["runs"])
        wrows = read_csv(raw / f"pilot{process}.warmup.csv")
        if len(wrows) != 448 or any(r["overflow"] != "0" or r["cpu_complete_match"] != "1" for r in wrows):
            raise ValueError("warmup evidence incomplete or incorrect")
        if {(r["policy"], int(r["epoch"]), int(r["threshold"]), int(r["query"])) for r in wrows} != {
                (p, e, t, q) for p in POLICIES for e in range(7) for t in (70, 80) for q in range(8)}:
            raise ValueError("warmup request keys incomplete")
        for row in wrows: row["process"] = process
        samples.extend(rows); maintenance.extend(mrows); warmups.extend(wrows); by_process.append(index)
    comparisons = {}
    for scope, epochs in (("all_updated_epochs", range(1, 7)), ("final_epoch", [6])):
        comparisons[scope] = {}
        for policy in POLICIES[:-1]:
            comparisons[scope][policy] = {}
            for threshold in (70, 80):
                metrics = {}
                for metric in ("service_ms", "kernel_ms"):
                    ratios = []
                    for index in by_process:
                        pairs = [(index[(policy, e, threshold, q)], index[("compact", e, threshold, q)])
                                 for e in epochs for q in range(32)]
                        # Zero-candidate kernels perform no launch; avoid dividing zero/zero.
                        ratios.append(gm(float(a[metric]) / float(b[metric]) for a, b in pairs
                                         if float(a[metric]) > 0 and float(b[metric]) > 0))
                    metrics[metric + "_policy_over_compact"] = process_interval(ratios)
                comparisons[scope][policy][str(threshold)] = metrics
    policy_totals = {}
    for policy in POLICIES:
        qrows = [r for r in samples if r["policy"] == policy and int(r["epoch"]) > 0]
        mrows = [r for r in maintenance if r["policy"] == policy and int(r["epoch"]) > 0]
        per_process = []
        for process in range(4):
            relevant = [r for r in mrows if r["process"] == process]
            per_process.append({key: sum(float(r[key]) for r in relevant)
                                for key in ("total_ms", "host_build_ms", "upload_ms", "uploaded_bytes",
                                            "merge_host_read_write_bytes", "merges")})
        policy_totals[policy] = {"final_runs": final_shapes[policy],
            "query_service_median_ms": stat.median(float(r["service_ms"]) for r in qrows),
            "query_kernel_median_ms": stat.median(float(r["kernel_ms"]) for r in qrows),
            "mean_candidate_rows": stat.mean(int(r["candidate_rows"]) for r in qrows),
            "mean_tail_threads": stat.mean(int(r["tail_threads"]) for r in qrows),
            "mean_descriptor_bytes": stat.mean(int(r["descriptor_bytes"]) for r in qrows),
            "max_peak_tracked_device_bytes": max(int(r["tracked_peak_device_bytes"]) for r in mrows),
            "max_observed_cuda_used_bytes": max(int(r["cuda_observed_used_bytes"]) for r in mrows),
            "max_process_rss_kib": max(int(r["host_maxrss_kib"]) for r in mrows),
            "maintenance_totals_by_process": per_process}
    device = json.loads((raw / "pilot0.pre.json").read_text())["gpu"][0]
    summary = {"scope": complete["scope"], "gpu": {"index": manifest["physical_gpu"], "uuid": manifest["gpu_uuid"],
        "name": device[2], "driver": device[3], "total_memory_mib": int(device[4])},
        "arrival_rows": [r["rows"] for r in source["runs"]], "final_rows": sum(r["rows"] for r in source["runs"]),
        "measurements": {"fresh_processes": 4, "complete_measured_queries": len(samples),
            "complete_warmup_queries": len(warmups), "fixture_queries_including_sanitizers": 5376,
            "old_epoch_checks_measured_processes": 192, "old_epoch_checks_fixture_processes": 144,
            "maintenance_events_including_initialization": len(maintenance), "gpu_memcheck": "0 errors",
            "gpu_synccheck": "0 errors", "overflow_and_empty_slice_guards": "pass",
            "all_control_candidate_counts_and_outputs_equal": True, "gpu_wrapper_wall_s": complete["budget_wall_s"]},
        "policy_totals": policy_totals, "paired_comparisons": comparisons,
        "estimator": "Geometric mean paired-query latency ratio within each process, then 4 process replicates; df=3 log-t interval. Ratio policy/compact >1 means compact faster.",
        "limitations": ["sequential, not sustained concurrent service", "Q=1 only, 256-bit, thresholds .7/.8",
            "consistent-ID sampled finite history; not the 41M full corpus", "controls are predeclared, not tuned on a separate development trace",
            "budget is not binding; no near-capacity evidence", "host bucket merge is a reference control, not optimized GPU compaction",
            "256 MiB is explicit data/workspace allocation only, not a whole-VRAM cap; observed whole-device used memory reaches about 670 MiB",
            "maintenance total_ms ends at publish readiness, excludes later old-reader checks and old-owner reclamation; not full lifecycle latency",
            "queries come from sampled arrival cohorts, not an external holdout", "scan bytes are logical fp+popcount loads, not measured DRAM traffic",
            "old_epoch_logical_bytes includes shared data and is not exclusive unreclaimed bytes", "four short processes on a shared host do not establish production stability"]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for filename, rows in (("queries.csv", samples), ("maintenance.csv", maintenance), ("warmup.csv", warmups)):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    (output / "input_manifest.json").write_text(json.dumps(source, indent=2) + "\n")
    receipt = {"raw_files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                             for p in sorted(raw.iterdir()) if p.is_file()},
               "cpu_ready_sha256": digest(args.cpu_ready), "source_hashes": cpu["sources"],
               "binary_sha256": cpu["binary_sha256"], "runner_sha256": manifest["runner_sha256"],
               "analysis_sha256": digest(Path(__file__)), "started_utc": manifest["start_utc"],
               "cpu_commands": [{"argv": r["argv"], "returncode": r["returncode"], "wall_s": r["wall_s"],
                                 "log_sha256": r["log_sha256"]} for r in cpu["commands"]]}
    (output / "evidence_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"measurements": summary["measurements"], "policy_totals": policy_totals,
                      "paired_comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
