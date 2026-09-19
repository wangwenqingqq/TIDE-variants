"""Audit every batch, byte ledger and process before summarizing four process replicates."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics as st

POLICIES = ("all_delta", "periodic2", "size_tiered", "compact")
BATCHES = (1, 8, 64)
CAPS = (1280, 2048)


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def gm(values):
    values = list(values)
    require(bool(values) and min(values) > 0, "invalid positive observations")
    return math.exp(st.mean(map(math.log, values)))


def interval(values):
    require(len(values) == 4, "exactly four process replicates required")
    logs = list(map(math.log, values))
    center = st.mean(logs)
    radius = 3.182446305284263 * st.stdev(logs) / 2
    return {"process_ratios": values, "geomean": math.exp(center),
            "process_log_t95": [math.exp(center - radius), math.exp(center + radius)]}


def quantile(values, p):
    values = sorted(values)
    at = (len(values) - 1) * p
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] + (at - lo) * (values[hi] - values[lo])


def groups(rows, keys):
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row)
    return out


def audit_request(r):
    require(r["cpu_complete_match"] == "1", "incomplete or unequal output")
    require(int(r["hits"]) * 16 == int(r["output_bytes"]), "output byte ledger")
    require(int(r["candidate_rows"]) * 34 == int(r["scan_fp_pc_bytes"]), "scan byte ledger")
    require(int(r["descriptors"]) * 48 == int(r["descriptor_bytes"]), "descriptor byte ledger")
    require(float(r["end_ms"]) >= float(r["device_complete_ms"]) >= float(r["begin_ms"]), "request timeline")
    require(abs(float(r["end_ms"]) - float(r["begin_ms"]) - float(r["service_ms"])) < 1e-5, "service timer")
    require(float(r["query_queue_ms"]) == 0, "unexpected closed-loop query queue")


def audit_arena(path):
    live, outstanding, peak = 0, {}, 0
    for r in read_csv(path):
        off, size = int(r["offset"]), int(r["bytes"])
        if r["event"] == "alloc":
            require(all(off + size <= old or old + n <= off for old, n in outstanding.items()), "overlapping live arena allocations")
            require(off not in outstanding, "duplicate allocation offset")
            outstanding[off] = size
            live += size
        elif r["event"] == "release":
            require(r["event"] == "release" and outstanding.pop(off, None) == size, "release mismatch")
            live -= size
        else:
            require(r["event"] == "refuse" and off < size, "invalid arena refusal")
        require(live == int(r["live_bytes"]), "arena running balance")
        peak = max(peak, live)
    require(not outstanding and live == 0, "arena final leak")
    return peak


def audit_maintenance(r):
    begin, ready, pub, done = [float(r[k]) for k in ("begin_ms", "ready_ms", "published_ms", "writer_done_ms")]
    require(begin <= ready <= pub <= done, "maintenance publication timeline")
    require(float(r["old_epoch_owner_released_ms"]) > 0, "old epoch never released")
    require(float(r["exclusive_actual_reclaim_ms"]) >= done, "reclaim precedes writer completion")
    require(float(r["maintenance_queue_ms"]) >= 0, "negative maintenance queue")
    unique = sum(int(r[k]) for k in ("shared_bytes", "current_only_bytes", "reader_only_at_publish_bytes", "other_held_at_publish_bytes"))
    # Runtime::bytes, independently derived from 48-byte queries/slices and 16-byte hits.
    workspace = 64 * 48 + 64 * 65 * 48 + 64 * 65536 * 16 + 512
    require(unique + workspace == int(r["live_at_publish_bytes"]), "unique-owner arena balance")


def write_gzip_csv(path, rows):
    # Deterministic bytes across reruns: empty embedded filename and mtime=0.
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cpu-ready", type=Path, required=True)
    parser.add_argument("--cpu-gate1m", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw
    complete = json.loads((raw / "COLLECTION_COMPLETE.json").read_text())
    manifest = json.loads((raw / "MANIFEST.json").read_text())
    cpu = json.loads(args.cpu_ready.read_text())
    source = json.loads(args.input_manifest.read_text())
    require(digest(args.cpu_ready) == manifest["cpu_ready_sha256"], "CPU receipt mismatch")
    require(digest(args.cpu_gate1m) == manifest["cpu_gate1m_sha256"], "million gate receipt mismatch")
    require(digest(args.input_manifest) == cpu["input_hashes"]["MANIFEST.json"], "input manifest mismatch")
    require(cpu["binary_sha256"] == manifest["binary_sha256"], "binary identity mismatch")
    require(len(manifest["commands"]) == 21, "campaign command count")
    memory = {}
    for name in manifest["commands"]:
        record = json.loads((raw / f"{name}.process.json").read_text())
        require(record["failure"] is None and record["returncode"] == 0, "failed process in complete campaign")
        require(not record["post"]["apps"], "live GPU process after child")
        cap = record["total_cap_mib"] * 1048576
        require(all(int(s["gpu"][0][5]) * 1048576 <= cap for s in record["samples"]), "NVML budget exceeded")
        mem = json.loads((raw / f"{name}.memory.json").read_text())
        require(mem["observed_whole_card_peak_bytes"] <= cap and mem["total_cap_bytes"] == cap, "runtime whole-card cap")
        require(mem["final_live_arena_bytes"] == 0, "logical leak")
        require(audit_arena(raw / f"{name}.arena.csv") == mem["peak_logical_arena_bytes"], "arena peak ledger")
        mem["nvml_peak_observed_bytes"] = max(int(s["gpu"][0][5]) * 1048576 for s in record["samples"])
        memory[name] = mem
        log = (raw / f"{name}.log").read_text()
        if "guards" in name:
            require("GUARDS_PASS injected=5 budget_refusal=2 overflow=1 pending_GPU_publish=1 cancellation=1 arena_restored=1" in log, "guards incomplete")
        if "memcheck" in name:
            require("LEAK SUMMARY: 0 bytes leaked in 0 allocations" in log, "leak check incomplete")
        if any(s in name for s in ("memcheck", "synccheck")):
            require("ERROR SUMMARY: 0 errors" in log, "sanitizer error")
        if "racecheck" in name:
            require("RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" in log, "racecheck error")
        for r in read_csv(raw / f"{name}.requests.csv"):
            audit_request(r)

    sequential, concurrent, maintenance = [], [], []
    seq_index = {}
    for rotation in range(4):
        for cap in CAPS:
            name = f"seq_r{rotation}_cap{cap}"
            rows = read_csv(raw / f"{name}.requests.csv")
            require(len(rows) == 8176, "sequential batch count")
            index = {}
            for r in rows:
                r["cap_mib"] = cap
                key = (r["phase"], r["policy"], int(r["batch"]), int(r["epoch"]), int(r["threshold"]), int(r["first_query"]))
                require(key not in index, "duplicate sequential batch")
                index[key] = r
            for phase in ("warmup", "measured"):
                for batch in BATCHES:
                    for e in range(7):
                        for t in (70, 80):
                            for first in range(0, 64, batch):
                                arms = [index[(phase, p, batch, e, t, first)] for p in POLICIES]
                                for field in ("rows", "candidate_rows", "hits", "output_bytes", "result_hash"):
                                    require(len({r[field] for r in arms}) == 1, "control semantic mismatch: " + field)
            seq_index[(rotation, cap)] = index
            sequential.extend(rows)
            mrows = read_csv(raw / f"{name}.maintenance.csv")
            require(len(mrows) == 84, "sequential maintenance count")
            for r in mrows:
                r["cap_mib"] = cap
                audit_maintenance(r)
            maintenance.extend(mrows)
        rows = read_csv(raw / f"concurrent_r{rotation}.requests.csv")
        grouped = groups(rows, ("mode", "policy", "batch", "phase"))
        for batch in BATCHES:
            for policy in POLICIES:
                baseline = grouped[("static", policy, str(batch), "measured")]
                shadow = grouped[("shadow", policy, str(batch), "measured")]
                growth = grouped[("growth", policy, str(batch), "measured")]
                require(len(baseline) == len(shadow) == len(growth) == 256, "concurrent batch count")
                for group in (baseline, shadow, growth):
                    for i, r in enumerate(group):
                        require(int(r["first_query"]) == (i * batch) % 64 and int(r["threshold"]) == (80 if i % 2 else 70), "concurrent request schedule")
                for a, b in zip(baseline, shadow):
                    for key in ("epoch", "rows", "candidate_rows", "hits", "output_bytes", "result_hash"):
                        require(a[key] == b[key], "shadow changed visible query work")
                    require(a["epoch"] == "0", "static epoch changed")
                require(all(a["epoch"] == "6" for a in grouped[("growth", policy, str(batch), "final_check")]), "growth final epoch missing")
        for r in rows:
            r["cap_mib"] = 2048
        concurrent.extend(rows)
        mrows = read_csv(raw / f"concurrent_r{rotation}.maintenance.csv")
        require(len(mrows) == 144, "concurrent maintenance event count")
        mi = {(r["mode"], r["policy"], r["batch"], r["epoch"]): r for r in mrows}
        for batch in BATCHES:
            for policy in POLICIES:
                for e in range(1, 7):
                    a, b = [mi[(mode, policy, str(batch), str(e))] for mode in ("shadow", "growth")]
                    for key in ("uploaded_bytes", "merge_host_rw_bytes", "rows", "runs", "merges", "deferred"):
                        require(a[key] == b[key], "shadow/growth maintenance work mismatch: " + key)
        for r in mrows:
            r["cap_mib"] = 2048
            audit_maintenance(r)
        maintenance.extend(mrows)

    comparisons = {}
    for cap in CAPS:
        comparisons[str(cap)] = {}
        for batch in BATCHES:
            comparisons[str(cap)][str(batch)] = {}
            for policy in POLICIES[:-1]:
                result = {}
                for t in (70, 80):
                    ratios = []
                    for rotation in range(4):
                        idx = seq_index[(rotation, cap)]
                        pairs = [(idx[("measured", policy, batch, e, t, first)], idx[("measured", "compact", batch, e, t, first)])
                                 for e in range(1, 7) for first in range(0, 64, batch)]
                        ratios.append(gm(float(a["service_ms"]) / float(b["service_ms"]) for a, b in pairs))
                    result[str(t)] = interval(ratios)
                comparisons[str(cap)][str(batch)][policy] = result

    concurrent_summary = {}
    measured_concurrent = [r for r in concurrent if r["phase"] == "measured"]
    cg = groups(measured_concurrent, ("rotation", "mode", "policy", "batch"))
    mg = groups([r for r in maintenance if r["mode"] != "sequential"], ("rotation", "mode", "policy", "batch"))
    for batch in BATCHES:
        concurrent_summary[str(batch)] = {}
        for policy in POLICIES:
            result = {"shadow_over_static_service": None, "exposure_by_process": [], "process_mode_stats": []}
            ratios = []
            p99ratios = []
            for rotation in range(4):
                base = cg[(str(rotation), "static", policy, str(batch))]
                shadow = cg[(str(rotation), "shadow", policy, str(batch))]
                ratios.append(gm(float(a["service_ms"]) / float(b["service_ms"]) for a, b in zip(shadow, base)))
                p99ratios.append(quantile([float(r["service_ms"]) for r in shadow], .99) /
                                 quantile([float(r["service_ms"]) for r in base], .99))
                for mode in ("static", "shadow", "growth"):
                    reqs = cg[(str(rotation), mode, policy, str(batch))]
                    vals = [float(r["service_ms"]) for r in reqs]
                    result["process_mode_stats"].append({"rotation": rotation, "mode": mode,
                        "median_batch_ms": st.median(vals), "p99_batch_ms": quantile(vals, .99),
                        "closed_loop_wall_ms": float(reqs[-1]["end_ms"]) - float(reqs[0]["begin_ms"]),
                        "sum_service_ms": sum(vals), "visible_epochs": sorted(set(int(r["epoch"]) for r in reqs)),
                        "max_maintenance_backlog": max(int(r["maintenance_backlog"]) for r in reqs)})
                    if mode == "static":
                        continue
                    events = mg[(str(rotation), mode, policy, str(batch))]
                    exposed = [r for r in reqs if any(float(r["begin_ms"]) < float(m["writer_done_ms"]) and
                               float(r["end_ms"]) > float(m["begin_ms"]) for m in events)]
                    event_counts = [sum(float(r["begin_ms"]) < float(m["writer_done_ms"]) and
                                        float(r["end_ms"]) > float(m["begin_ms"]) for r in reqs) for m in events]
                    result["exposure_by_process"].append({"rotation": rotation, "mode": mode,
                        "overlapping_batches": len(exposed), "total_batches": 256, "per_event_overlaps": event_counts,
                        "maintenance_prep_ms": sum(float(m["writer_done_ms"]) - float(m["begin_ms"]) for m in events),
                        "maintenance_queue_ms": sum(float(m["maintenance_queue_ms"]) for m in events)})
            result["shadow_over_static_service"] = interval(ratios)
            result["shadow_over_static_p99"] = interval(p99ratios)
            concurrent_summary[str(batch)][policy] = result

    policy_totals = {}
    for cap in CAPS:
        policy_totals[str(cap)] = {}
        for policy in POLICIES:
            rows = [r for r in maintenance if r["mode"] == "sequential" and r["cap_mib"] == cap and r["policy"] == policy and int(r["epoch"]) > 0]
            policy_totals[str(cap)][policy] = {
                "final_runs_observed": sorted({int(r["runs"]) for r in rows if r["epoch"] == "6"}),
                "deferred_merges": sum(int(r["deferred"]) for r in rows),
                "per_process_batch_totals": [{"rotation": int(key[0]), "batch": int(key[1]),
                    **{field: sum(float(r[field]) for r in group) for field in
                       ("host_build_ms", "metadata_ms", "staging_ms", "h2d_ms", "uploaded_bytes", "merge_host_rw_bytes", "lifecycle_ms")}}
                    for key, group in groups(rows, ("rotation", "batch")).items()]}

    measured_seq = [r for r in sequential if r["phase"] == "measured"]
    summary = {"scope": complete["scope"], "final_rows": sum(r["rows"] for r in source["runs"]),
        "arrival_rows": [r["rows"] for r in source["runs"]],
        "counts": {"measured_batches": len(measured_seq) + len(measured_concurrent),
                   "sequential_measured_queries": sum(int(r["batch"]) for r in measured_seq),
                   "concurrent_measured_queries": sum(int(r["batch"]) for r in measured_concurrent),
                   "all_logged_complete_queries": sum(int(r["batch"]) for r in sequential + concurrent),
                   "maintenance_events": len(maintenance), "performance_fresh_processes": 12},
        "correctness": {"complete_oracle_vectors": 2688, "memcheck": "0 errors and 0 bytes leaked",
                        "synccheck": "0 errors", "racecheck": "0 hazards/errors/warnings",
                        "fault_budget_overflow_pending_GPU_cancellation": "pass",
                        "all_arena_allocations_balanced": True, "all_observed_total_caps_respected": True,
                        "shadow_same_visible_queries_and_equal_maintenance_bytes": True},
        "sequential_policy_over_compact": comparisons, "sequential_policy_totals": policy_totals,
        "concurrency": concurrent_summary, "memory_by_process": memory,
        "gpu_uuid": manifest["gpu_uuid"], "wrapper_wall_s": complete["wrapper_wall_s"],
        "estimator": "Paired batch geometric-mean ratio per process, then 4 process replicates; df=3 log-t95 interval. Ratios >1 mean numerator slower.",
        "limitations": ["256-bit prepared fingerprint artifact; not raw-molecule end-to-end verification", "64 cohort-derived queries, no external holdout",
            "six finite natural arrivals, not production steady state; no deletes or crash recovery",
            "preset ordinary controls, not independently tuned on a development trace; no new mechanism claim",
            "closed-loop single reader; query queue zero by construction; oracle comparisons create dispatch gaps",
            "mode order static/shadow/growth is fixed; shared-host scheduling and temporal drift remain confounders",
            "maintenance wall overlap is exposure, not proof of simultaneous GPU kernel and H2D execution",
            "racecheck tests shared-memory hazards, not host concurrency correctness in general",
            "whole-card VRAM is sampled and observed at phases, not a hardware-enforced partition; 64 MiB headroom",
            "arena reclaim means a suballocation is reusable; physical reservation is held until process exit",
            "shadow retains fixed base separately; its full reclamation lifecycle is intentionally different",
            "old Gate0 service omitted an O(N) host metadata scan; no cross-gate speedup is computed",
            "four short process replicates do not establish long-term production stability"]}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_gzip_csv(args.output / "sequential_requests.csv.gz", sequential)
    write_gzip_csv(args.output / "concurrent_requests.csv.gz", concurrent)
    with (args.output / "maintenance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(maintenance[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(maintenance)
    (args.output / "input_manifest.json").write_text(json.dumps(source, indent=2) + "\n")
    receipt = {"raw_files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(raw.iterdir()) if p.is_file()},
               "source_hashes": cpu["sources"], "binary_sha256": cpu["binary_sha256"],
               "cpu_ready_sha256": digest(args.cpu_ready), "cpu_gate1m_sha256": digest(args.cpu_gate1m),
               "runner_sha256": manifest["runner_sha256"], "analysis_sha256": digest(Path(__file__)),
               "started_utc": manifest["start_utc"], "completed_scope": complete}
    (args.output / "evidence_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    attempts = {}
    for name in ("campaign_v1_gpu0", "campaign_v2_gpu0"):
        path = raw.parent / name
        require(path.is_dir() and not (path / "COLLECTION_COMPLETE.json").exists(), "retained interrupted attempt missing")
        attempts[name] = {"included_in_primary_estimates": False,
            "manifest": json.loads((path / "MANIFEST.json").read_text()),
            "raw_files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(path.iterdir()) if p.is_file()}}
    (args.output / "attempts_receipt.json").write_text(json.dumps(attempts, indent=2) + "\n")
    print(json.dumps({"counts": summary["counts"], "correctness": summary["correctness"],
                      "sequential_policy_over_compact": comparisons}, indent=2))


if __name__ == "__main__":
    main()
