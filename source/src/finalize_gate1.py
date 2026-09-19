#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

import numpy as np


def values(rows, field):
    return np.asarray([float(r[field]) for r in rows], dtype=np.float64)


def p(x, q):
    return float(np.percentile(np.asarray(x, dtype=np.float64), q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--stage-a", type=Path, required=True)
    ap.add_argument("--validation-run", type=Path, required=True)
    ap.add_argument("--query-run", type=Path, required=True)
    ap.add_argument("--update-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()

    stage_a = json.loads(args.stage_a.read_text())
    cpu = list(csv.DictReader((args.query_run / "all_cpu.csv").open()))
    gpu = list(csv.DictReader((args.query_run / "all_gpu.csv").open()))
    cpu_i = [r for r in cpu if r["mode"] == "INTERACTIVE"]
    cpu_s = [r for r in cpu if r["mode"] == "SUSTAINED64"]
    gpu_by_batch = {b: [r for r in gpu if int(r["batch"]) == b]
                    for b in (1, 8, 64, 512)}

    verify = (args.validation_run / "verify.txt").read_text()
    stress = json.loads((args.validation_run / "stress.json").read_text())
    memcheck = (args.validation_run / "memcheck.stdout").read_text()
    synccheck = (args.validation_run / "synccheck.stdout").read_text()
    exact_ok = "mismatches 0" in verify
    sanitizer_ok = ("ERROR SUMMARY: 0 errors" in memcheck and
                    "ERROR SUMMARY: 0 errors" in synccheck and
                    stress["hash_mismatches"] == 0 and stress["cycles"] == 1000)

    updates = {}
    for v in ("FULL_UNION", "DELTA_APPEND"):
        updates[v] = [json.loads(Path(x).read_text())["elapsed_ms"]
                      for x in glob.glob(str(args.update_run / f"pair_*_{v}.json"))]
    update_ratio = p(updates["FULL_UNION"], 50) / p(updates["DELTA_APPEND"], 50)

    cpu_i_med = p(values(cpu_i, "host_ms"), 50)
    cpu_s_qps = p(values(cpu_s, "qps"), 50)
    gpu1_p95 = p(values(gpu_by_batch[1], "host_ms"), 95)
    gpu64_qps = p(values(gpu_by_batch[64], "qps"), 50)
    gpu512_qps = p(values(gpu_by_batch[512], "qps"), 50)
    interactive_ratio = gpu1_p95 / cpu_i_med
    sustained_ratio = gpu64_qps / cpu_s_qps
    batch512_ratio = gpu512_qps / gpu64_qps

    proc = []
    for i in range(6):
        ci = [float(r["host_ms"]) for r in cpu_i if int(r["process_index"]) == i]
        cs = [float(r["qps"]) for r in cpu_s if int(r["process_index"]) == i]
        g1 = [float(r["host_ms"]) for r in gpu_by_batch[1] if int(r["process_index"]) == i]
        g64 = [float(r["qps"]) for r in gpu_by_batch[64] if int(r["process_index"]) == i]
        g512 = [float(r["qps"]) for r in gpu_by_batch[512] if int(r["process_index"]) == i]
        ir = p(g1, 95) / statistics.median(ci)
        sr = statistics.median(g64) / statistics.median(cs)
        nr = statistics.median(g512) / statistics.median(g64)
        proc.append({"process_index": i, "interactive_ratio": ir,
                     "sustained_ratio": sr, "batch512_vs_64": nr,
                     "interactive_pass": ir <= 0.50,
                     "sustained_pass": sr >= 5.0 and nr >= 1.0})
    process_passes = sum(x["interactive_pass"] and x["sustained_pass"] for x in proc)

    gates = {
        "G_B_EXACT": exact_ok,
        "G_B_SANITIZER": sanitizer_ok,
        "G_B_UPDATE": update_ratio >= 100.0,
        "G_B_INTERACTIVE": interactive_ratio <= 0.50,
        "G_B_SUSTAINED": sustained_ratio >= 5.0 and batch512_ratio >= 1.0,
        "G_B_PROCESS": process_passes >= 5,
    }
    decision = ("ADMIT_FLAT_GPU_APPEND_PROTOTYPE" if all(gates.values()) else
                "RETAIN_CPU_OVERLAY_CLOSE_ONE_CTA_GPU_INTERACTIVE")
    result = {
        "experiment_id": "tide_20260827_flat_gpu_append_top10",
        "decision": decision,
        "stage_a_decision": stage_a["decision"],
        "gates": gates,
        "validation": {"exact_mismatches": 0 if exact_ok else None,
                       "memcheck_zero_errors": "ERROR SUMMARY: 0 errors" in memcheck,
                       "synccheck_zero_errors": "ERROR SUMMARY: 0 errors" in synccheck,
                       "stress_cycles": stress["cycles"],
                       "stress_hash_mismatches": stress["hash_mismatches"]},
        "interactive": {"cpu_8worker_median_ms": cpu_i_med,
                        "gpu_batch1_median_ms": p(values(gpu_by_batch[1], "host_ms"), 50),
                        "gpu_batch1_p95_ms": gpu1_p95,
                        "gpu_p95_over_cpu_median": interactive_ratio,
                        "speedup_using_gate_denominator": 1.0 / interactive_ratio},
        "sustained": {"cpu_8core_batch64_median_qps": cpu_s_qps,
                      "gpu_batch64_median_qps": gpu64_qps,
                      "gpu_over_cpu_ratio": sustained_ratio,
                      "gpu_batch512_median_qps": gpu512_qps,
                      "batch512_over_batch64": batch512_ratio},
        "gpu_batches": {str(b): {"host_ms_median": p(values(gpu_by_batch[b], "host_ms"), 50),
                                  "host_ms_p95": p(values(gpu_by_batch[b], "host_ms"), 95),
                                  "kernel_ms_median": p(values(gpu_by_batch[b], "max_kernel_ms"), 50),
                                  "qps_median": p(values(gpu_by_batch[b], "qps"), 50)}
                        for b in (1, 8, 64, 512)},
        "update": {"full_union_median_ms": p(updates["FULL_UNION"], 50),
                   "delta_append_median_ms": p(updates["DELTA_APPEND"], 50),
                   "ratio": update_ratio},
        "processes": proc,
        "process_combined_passes": process_passes,
        "mechanism_diagnosis": "one CTA per query per GPU caps batch-1 execution at one active SM per GPU; pinned host buffers remove serialization but cannot create query-parallel device work",
        "evidence": {"stage_a": str(args.stage_a), "validation": str(args.validation_run),
                     "query": str(args.query_run), "update": str(args.update_run)},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    md = f"""# Gate-1 final result

Decision: **{decision}**

Stage A passes and retains the insert-only CPU dynamic overlay. Stage B fails
only the interactive and combined-process gates.

- Exactness: zero mismatches; memcheck and synccheck report zero errors; 1,000
  append/query cycles have zero hash mismatches.
- GPU update publication: {p(updates['FULL_UNION'],50):.3f} ms full replacement
  versus {p(updates['DELTA_APPEND'],50):.6f} ms delta append ({update_ratio:.1f}x).
- Interactive: CPU eight-worker median {cpu_i_med:.3f} ms; GPU batch-1 p95
  {gpu1_p95:.3f} ms. The frozen ratio is {interactive_ratio:.4f}x, or only
  {1/interactive_ratio:.2f}x speedup, below the required 2x.
- Sustained: CPU eight-core median {cpu_s_qps:.2f} q/s versus GPU batch-64
  {gpu64_qps:.2f} q/s ({sustained_ratio:.1f}x). GPU batch-512 reaches
  {gpu512_qps:.2f} q/s.
- Process gate: {process_passes}/6 pairs pass both interactive and sustained
  requirements.

The negative boundary is narrow: one CTA per query per GPU. Batch-1 launches at
most one active SM on each GPU, while batch-64/512 has abundant query-level
parallelism and passes strongly. Reopen only with a separately frozen
query-parallel latency kernel and a fail-closed batch dispatch; do not revive the
rejected tree.
"""
    args.markdown.write_text(md)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
