"""Deterministic, standard-library Gate 2 metrics and pre-registered selection."""
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values, p):
    v = sorted(values)
    if not v:
        raise ValueError("empty population")
    return v[max(0, math.ceil(p * len(v)) - 1)]


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def summarize(prefix):
    prefix = str(prefix)
    meta = json.loads(Path(prefix + ".case.json").read_text())
    arrivals = rows(prefix + ".arrivals.csv")
    requests = rows(prefix + ".requests.csv")
    maintenance = rows(prefix + ".maintenance.csv")
    coverage = rows(prefix + ".coverage.csv")
    memory = json.loads(Path(prefix + ".memory.json").read_text())
    n = meta["scheduled_batches"]
    if len(arrivals) != n or len(requests) != n or meta["completed_batches"] != n or meta["dropped_batches"]:
        raise ValueError("lost or duplicated request")
    dynamic = meta["mode"] in ("shadow", "growth")
    if [int(x["epoch"]) for x in maintenance] != (list(range(1, 7)) if dynamic else []):
        raise ValueError("missing/duplicate maintenance epoch")
    for i, (a, r) in enumerate(zip(arrivals, requests)):
        if int(a["index"]) != i or int(r["cpu_complete_match"]) != 1 or int(r["first_query"]) != i*8 % 64:
            raise ValueError("request identity/correctness mismatch")
        if int(r["batch"]) != 8 or int(r["threshold"]) != 80:
            raise ValueError("unexpected query workload")
        scheduled = (float(r["begin_ms"]) if meta["mode"] == "calibrate" else
                     meta["start_ms"] + i*meta["interval_ms"])
        if abs(float(a["arrival_ms"])-scheduled) > .0001:
            raise ValueError("arrival depends on execution or is corrupt")
        if abs(float(a["response_ms"])-(float(a["returned_ms"])-scheduled)) > .0001:
            raise ValueError("response excludes queue")
        if float(a["queue_ms"]) < -.0001 or float(a["full_service_ms"]) <= 0:
            raise ValueError("invalid request clocks")
        if (float(a["returned_ms"]) <= meta["start_ms"] + meta["window_ms"]) != bool(int(a["within_window"])):
            raise ValueError("window accounting mismatch")
    if memory["observed_whole_card_peak_bytes"] > memory["total_cap_bytes"] or memory["final_live_arena_bytes"]:
        raise ValueError("memory budget/release failed")
    # Audit arena offsets, overlap and exact allocation/release balance.
    live = {}
    for event in rows(prefix + ".arena.csv"):
        kind, off, size = event["event"], int(event["offset"]), int(event["bytes"])
        if kind == "alloc":
            if any(off < b+s and b < off+size for b, s in live.items()):
                raise ValueError("overlapping arena allocation")
            live[off] = size
        elif kind == "release":
            if live.pop(off, None) != size:
                raise ValueError("arena invalid/double release")
        if sum(live.values()) != int(event["live_bytes"]):
            raise ValueError("arena byte ledger mismatch")
    if live:
        raise ValueError("arena leak")
    for m in maintenance:
        if abs(float(m["arrival_ms"])-(meta["start_ms"]+750+(int(m["epoch"])-1)*1000)) > .0001:
            raise ValueError("maintenance arrival is not fixed")
        classified = sum(int(m[k]) for k in ("shared_bytes", "current_only_bytes", "reader_only_at_publish_bytes", "other_held_at_publish_bytes"))
        # Workspace is the same unique residual at all publication snapshots.
        if classified > int(m["live_at_publish_bytes"]):
            raise ValueError("owner byte accounting underflow")
        if float(m["old_epoch_owner_released_ms"]) < float(m["last_reader_device_done_ms"]):
            raise ValueError("owner reclaimed before GPU completion")
    recomputed_coverage = all(bool(int(c["within_window"])) for c in coverage)
    if dynamic and len(coverage) != 6:
        raise ValueError("coverage ledger missing")
    if meta["mode"] == "growth":
        recomputed_coverage &= all(meta["seen_epochs"])
    if recomputed_coverage != meta["coverage_pass"]:
        raise ValueError("coverage flag mismatch")
    completed = sum(int(a["within_window"]) for a in arrivals)
    response = [float(a["response_ms"]) for a in arrivals]
    service = [float(a["full_service_ms"]) for a in arrivals]
    result = dict(meta, window_completed_batches=completed, completion_fraction=completed/n,
                  window_queries_per_s=completed*8/(meta["window_ms"]/1000),
                  p99_response_ms=percentile(response, .99), p50_response_ms=percentile(response, .5),
                  max_response_ms=max(response), p99_service_ms=percentile(service, .99),
                  mean_service_ms=sum(service)/n,
                  p99_queue_ms=percentile([float(a["queue_ms"]) for a in arrivals], .99),
                  max_waiting_batches=max(int(a["waiting_batches"]) for a in arrivals),
                  max_maintenance_backlog=max(int(a["maintenance_backlog"]) for a in arrivals),
                  max_publication_lag_ms=max([float(m["published_ms"])-float(m["arrival_ms"]) for m in maintenance] or [0]),
                  max_maintenance_queue_ms=max([float(m["maintenance_queue_ms"]) for m in maintenance] or [0]),
                  uploaded_bytes=sum(int(m["uploaded_bytes"]) for m in maintenance),
                  merged_host_rw_bytes=sum(int(m["merge_host_rw_bytes"]) for m in maintenance),
                  merges=sum(int(m["merges"]) for m in maintenance),
                  deferred=sum(int(m["deferred"]) for m in maintenance),
                  end_runs=int(maintenance[-1]["runs"]) if maintenance else int(requests[-1]["runs"]),
                  peak_vram_bytes=memory["observed_whole_card_peak_bytes"],
                  host_maxrss_kib=memory["host_maxrss_kib"])
    return result


def passes(case, deadline):
    return (case["coverage_pass"] and case["cpu_complete_match"] and
            case["completion_fraction"] >= .99 and case["max_publication_lag_ms"] <= 1000 and
            case["p99_response_ms"] <= deadline)


def choose(cases, deadline, family):
    candidates = [c for c in cases if c["policy"].startswith(family)]
    if not candidates:
        raise ValueError("missing tuning family")
    return min(candidates, key=lambda c: (not c["coverage_pass"], not passes(c, deadline),
                                         c["p99_response_ms"], c["uploaded_bytes"],
                                         int(c["policy"][len(family):])))["policy"]


def interval_for(mean_ms):
    return max(.1, math.ceil(mean_ms/.8/.01)*.01)
