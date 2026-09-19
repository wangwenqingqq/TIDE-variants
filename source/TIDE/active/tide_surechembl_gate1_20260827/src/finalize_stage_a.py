#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
from pathlib import Path

import numpy as np


def pct(x: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(x, dtype=np.float64), p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--correctness-run", type=Path, required=True)
    ap.add_argument("--query-run", type=Path, required=True)
    ap.add_argument("--update-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()

    correctness = json.loads((args.correctness_run / "summary.json").read_text())
    rows = list(csv.DictReader((args.query_run / "all_samples.csv").open()))
    by_variant: dict[str, list[float]] = {}
    for v in ("FRESH_UNION", "BASE_PLUS_DELTA"):
        by_variant[v] = [int(r["latency_ns"]) / 1e6 for r in rows if r["variant"] == v]

    proc = []
    for i in range(6):
        fresh = [int(r["latency_ns"]) / 1e6 for r in rows
                 if int(r["process_index"]) == i and r["variant"] == "FRESH_UNION"]
        overlay = [int(r["latency_ns"]) / 1e6 for r in rows
                   if int(r["process_index"]) == i and r["variant"] == "BASE_PLUS_DELTA"]
        fm, om = statistics.median(fresh), statistics.median(overlay)
        proc.append({"process_index": i, "fresh_median_ms": fm,
                     "overlay_median_ms": om, "ratio": om / fm})
    ratios = np.asarray([p["ratio"] for p in proc], dtype=np.float64)
    rng = np.random.default_rng(20260827)
    boot = np.empty(100000, dtype=np.float64)
    logs = np.log(ratios)
    for i in range(len(boot)):
        boot[i] = math.exp(float(rng.choice(logs, size=len(logs), replace=True).mean()))
    paired_geom = math.exp(float(logs.mean()))
    paired_ci = [pct(boot.tolist(), 2.5), pct(boot.tolist(), 97.5)]

    updates: dict[str, list[float]] = {}
    for v in ("FRESH_UNION", "BASE_PLUS_DELTA"):
        updates[v] = [
            json.loads(Path(p).read_text())["elapsed_ns"] / 1e6
            for p in glob.glob(str(args.update_run / f"pair_*_{v}.json"))
        ]
    fresh_artifact = args.root / "data/2026-08-25/fpsim2_fingerprints.h5"
    delta_artifact = args.root / "data/stage_a/delta_u64x6.bin"

    fresh_med = pct(by_variant["FRESH_UNION"], 50)
    overlay_med = pct(by_variant["BASE_PLUS_DELTA"], 50)
    fresh_p95 = pct(by_variant["FRESH_UNION"], 95)
    overlay_p95 = pct(by_variant["BASE_PLUS_DELTA"], 95)
    update_ratio = pct(updates["FRESH_UNION"], 50) / pct(updates["BASE_PLUS_DELTA"], 50)
    bytes_ratio = fresh_artifact.stat().st_size / delta_artifact.stat().st_size

    gates = {
        "G_A_EXACT": correctness["mismatches"] == 0,
        "G_A_UPDATE_TIME": update_ratio >= 100.0,
        "G_A_UPDATE_BYTES": bytes_ratio >= 100.0,
        "G_A_QUERY_MEDIAN": overlay_med / fresh_med <= 1.10,
        "G_A_QUERY_P95": overlay_p95 / fresh_p95 <= 1.15,
        "G_A_PROCESS": int(np.count_nonzero(ratios <= 1.10)) >= 5 and paired_ci[1] <= 1.10,
    }
    decision = "ADMIT_STAGE_B" if all(gates.values()) else "CLOSE_TIDE_DYNAMIC_OVERLAY"
    result = {
        "experiment_id": "tide_20260827_append_delta_service_top10",
        "decision": decision,
        "gates": gates,
        "correctness": correctness,
        "query": {
            "samples_per_variant": len(by_variant["FRESH_UNION"]),
            "fresh_union_ms": {"p10": pct(by_variant["FRESH_UNION"], 10),
                               "median": fresh_med, "p90": pct(by_variant["FRESH_UNION"], 90),
                               "p95": fresh_p95},
            "base_plus_delta_ms": {"p10": pct(by_variant["BASE_PLUS_DELTA"], 10),
                                   "median": overlay_med, "p90": pct(by_variant["BASE_PLUS_DELTA"], 90),
                                   "p95": overlay_p95},
            "median_ratio": overlay_med / fresh_med,
            "p95_ratio": overlay_p95 / fresh_p95,
            "processes": proc,
            "process_wins_at_1_10": int(np.count_nonzero(ratios <= 1.10)),
            "paired_geomean_ratio": paired_geom,
            "paired_bootstrap_95ci": paired_ci,
        },
        "update": {
            "fresh_union_load_ms": {"median": pct(updates["FRESH_UNION"], 50),
                                    "p90": pct(updates["FRESH_UNION"], 90)},
            "delta_publish_ms": {"median": pct(updates["BASE_PLUS_DELTA"], 50),
                                 "p90": pct(updates["BASE_PLUS_DELTA"], 90)},
            "time_ratio": update_ratio,
            "fresh_union_artifact_bytes": fresh_artifact.stat().st_size,
            "delta_artifact_bytes": delta_artifact.stat().st_size,
            "bytes_ratio": bytes_ratio,
            "scope": "warm-cache local artifact to in-memory publication; excludes network and fingerprint generation",
        },
        "evidence": {
            "correctness": str(args.correctness_run),
            "query": str(args.query_run),
            "update": str(args.update_run),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    md = f"""# Gate-1 Stage-A result

Decision: **{decision}**

All CPU overlay admission gates {'passed' if all(gates.values()) else 'did not pass'}.

- Exact correctness: {correctness['mismatches']} mismatches across {correctness['variant_rows']} variant-query rows.
- Fresh-union latency: {fresh_med:.3f} ms median, {fresh_p95:.3f} ms p95.
- Base-plus-delta latency: {overlay_med:.3f} ms median, {overlay_p95:.3f} ms p95.
- Candidate/keeper ratios: {overlay_med/fresh_med:.4f}x median and {overlay_p95/fresh_p95:.4f}x p95.
- Process gate: {int(np.count_nonzero(ratios <= 1.10))}/6 process medians within 1.10x; paired geometric mean {paired_geom:.4f}x, bootstrap 95% CI [{paired_ci[0]:.4f}, {paired_ci[1]:.4f}].
- Local-artifact publication: {pct(updates['FRESH_UNION'],50):.3f} ms full fresh union versus {pct(updates['BASE_PLUS_DELTA'],50):.6f} ms delta, a {update_ratio:.1f}x ratio.
- Artifact bytes: {fresh_artifact.stat().st_size:,} full union versus {delta_artifact.stat().st_size:,} delta, a {bytes_ratio:.1f}x ratio.

This establishes a bounded insert-only dynamic-overlay systems case. It does not
establish novelty, deletion/compaction behavior, production-query popularity,
or whole-request latency. Stage B is admitted only under the frozen GPU gates.
"""
    args.markdown.write_text(md)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
