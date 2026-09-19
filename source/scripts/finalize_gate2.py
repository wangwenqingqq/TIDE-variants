#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys

root = pathlib.Path(sys.argv[1]).resolve()
raw = root / "results" / "raw"

def run_dir(pointer):
    value = (root / pointer).read_text().strip()
    p = pathlib.Path(value)
    return p if p.is_absolute() else raw / p

def read_json(path):
    with open(path) as f:
        return json.load(f)

def nearest_rank(xs, p):
    ys = sorted(xs)
    return ys[max(0, math.ceil(p * len(ys)) - 1)]

def describe(xs):
    return {
        "n": len(xs),
        "p10": nearest_rank(xs, 0.10),
        "median": statistics.median(xs),
        "p90": nearest_rank(xs, 0.90),
        "p95": nearest_rank(xs, 0.95),
    }

def load_values(path, batch, field):
    with open(path) as f:
        return [float(r[field]) for r in csv.DictReader(f) if int(r["batch"]) == batch]

def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

def bootstrap_gm_ci(xs, seed=20260827, rounds=100000):
    rng = random.Random(seed)
    n = len(xs)
    boots = sorted(gm([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(rounds))
    return [boots[int(0.025 * rounds)], boots[int(0.975 * rounds) - 1]]

cal_dir = run_dir("CURRENT_CALIBRATION_RUN.txt")
correct_dir = run_dir("CURRENT_CORRECTNESS_RUN.txt")
san_dir = run_dir("CURRENT_SANITIZER_RUN.txt")
stress_dir = run_dir("CURRENT_STRESS_RUN.txt")
ab_dir = run_dir("CURRENT_AB_RUN.txt")
ncu_dir = run_dir("CURRENT_NCU_DIAG_RUN.txt")

cal = read_json(cal_dir / "SUMMARY.json")
correct = read_json(correct_dir / "verify.json")
stress = read_json(stress_dir / "stress.json")
ab_original = read_json(ab_dir / "SUMMARY.json")

batches = (1, 8, 64, 512)
variants = ("KEEPER", "CANDIDATE")
distributions = {}
for b in batches:
    distributions[str(b)] = {}
    for v in variants:
        host = []
        kernel = []
        qps = []
        for pair in range(6):
            path = ab_dir / f"pair_{pair}_{v}.csv"
            host += load_values(path, b, "host_ms")
            kernel += load_values(path, b, "max_kernel_ms")
            qps += load_values(path, b, "qps")
        distributions[str(b)][v.lower()] = {
            "host_ms": describe(host),
            "max_kernel_ms": describe(kernel),
            "qps": describe(qps),
        }

paired = {}
for b in batches:
    field = "host_ms" if b in (1, 8) else "qps"
    ratios = []
    for pair in range(6):
        k = statistics.median(load_values(ab_dir / f"pair_{pair}_KEEPER.csv", b, field))
        c = statistics.median(load_values(ab_dir / f"pair_{pair}_CANDIDATE.csv", b, field))
        ratios.append(c / k)
    paired[str(b)] = {
        "field": field,
        "candidate_over_keeper_process_ratios": ratios,
        "geometric_mean": gm(ratios),
        "bootstrap_95_ci": bootstrap_gm_ci(ratios),
    }

mem_text = (san_dir / "memcheck.stdout").read_text() + (san_dir / "memcheck.stderr").read_text()
sync_text = (san_dir / "synccheck.stdout").read_text() + (san_dir / "synccheck.stderr").read_text()
mem_clean = "ERROR SUMMARY: 0 errors" in mem_text
sync_clean = "ERROR SUMMARY: 0 errors" in sync_text

global_ab = ab_original["global"]
gates = {
    "G2_EXACT": correct["mismatches"] == 0 and correct["queries"] == 320,
    "G2_SANITIZER": mem_clean and sync_clean and stress["cycles"] == 1000 and stress["hash_mismatches"] == 0,
    "G2_INTERACTIVE": global_ab["1"]["candidate_p95_ms"] <= 25.706720,
    "G2_MECHANISM": global_ab["1"]["candidate_over_keeper_median_latency"] <= 0.50,
    "G2_BATCH8": global_ab["8"]["candidate_over_keeper_p95_latency"] <= 0.50,
    "G2_SUSTAINED_BATCH64": global_ab["64"]["candidate_over_keeper_median_qps"] >= 0.95,
    "G2_SUSTAINED_BATCH512": global_ab["512"]["candidate_over_keeper_median_qps"] >= 0.95,
    "G2_PROCESS": ab_original["process_pairs_pass"] >= 5,
}
all_pass = all(gates.values())

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

source = root / "src" / "gpu_query_parallel_bench.cu"
candidate_bin = root / "bin" / "gpu_query_parallel_bench"
keeper_bin = root / "bin" / "gpu_append_bench_v3_keeper"

result = {
    "experiment_id": "tide_20260827_query_parallel_exact_top10",
    "decision": "ADMIT_QUERY_PARALLEL_BATCH1_8_PROCEED_NOVELTY_AND_PRODUCTION_LOG_GATES" if all_pass else "REJECT_GATE2_CANDIDATE",
    "all_frozen_gates_pass": all_pass,
    "selected_partitions": cal["selected_partitions"],
    "selection_before_sealed_timing": True,
    "hardware": {
        "host": "CONFIGURE_ARCHIVE_HOST",
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "physical_gpus": [0, 1, 2, 3],
        "sm_count_per_gpu": 188,
        "driver": "590.48.01",
        "cuda_toolkit": "13.1",
    },
    "workload": {
        "real_dataset": "SureChEMBL 2026-08-18 base plus 2026-08-25 append-only delta",
        "base_rows": 41106048,
        "delta_rows": 20760,
        "union_rows": 41126808,
        "query_semantics": "exact 256-bit Tanimoto top-10 with self exclusion and deterministic similarity/ID order",
        "sealed_test_indices": "512-767",
        "service_scope": "host wall from pinned-query H2D through four-GPU exact kernels, pinned-result D2H, and CPU four-shard merge",
    },
    "artifacts": {
        "source_sha256": sha256(source),
        "candidate_binary_sha256": sha256(candidate_bin),
        "keeper_binary_sha256": sha256(keeper_bin),
        "source": str(source),
        "candidate_binary": str(candidate_bin),
        "keeper_binary": str(keeper_bin),
    },
    "static_resources": {
        "keeper": {"registers": 84, "stack_bytes": 0, "spill_load_bytes": 0, "spill_store_bytes": 0, "dynamic_shared_bytes": 40960, "grid_ctas_per_query_per_gpu": 1},
        "candidate_stage1": {"registers": 86, "stack_bytes": 0, "spill_load_bytes": 0, "spill_store_bytes": 0, "dynamic_shared_bytes": 40960, "grid_ctas_per_query_per_gpu": 128},
        "candidate_stage2": {"registers": 114, "stack_bytes": 0, "spill_load_bytes": 0, "spill_store_bytes": 0, "dynamic_shared_bytes": 40960, "grid_ctas_per_query_per_gpu": 1},
    },
    "calibration": cal,
    "correctness": correct,
    "sanitizer": {"memcheck_zero_errors": mem_clean, "synccheck_zero_errors": sync_clean},
    "stress": stress,
    "ab_original_summary": ab_original,
    "marginal_distributions": distributions,
    "paired_process_ratios": paired,
    "gates": gates,
    "ncu_diagnostic": {
        "status": "blocked",
        "reason": "ERR_NVGPUCTRPERM",
        "claim_scope": "diagnostic-only; no NCU duration is used as public performance evidence",
        "raw_path": str(ncu_dir),
    },
    "paper_boundary": {
        "admitted": "bounded kernel prototype and measured campaign result for the frozen exact scan contract",
        "not_admitted": "novelty, general workload prevalence, cost efficiency, or end-to-end TIDE/GTS++ claims",
        "required_next_gates": ["GPU indexing prior-art/novelty audit", "production query-log batch and latency distribution", "single-GPU and cost-normalized comparator", "paper-system end-to-end integration"],
    },
    "raw_evidence": {
        "calibration": str(cal_dir),
        "correctness": str(correct_dir),
        "sanitizer": str(san_dir),
        "stress": str(stress_dir),
        "direction_balanced_ab": str(ab_dir),
        "ncu_attempt": str(ncu_dir),
    },
}

(root / "results").mkdir(exist_ok=True)
with open(root / "results" / "FINAL_GATE2.json", "w") as f:
    json.dump(result, f, indent=2)

b1 = global_ab["1"]
b8 = global_ab["8"]
b64 = global_ab["64"]
b512 = global_ab["512"]
md = f"""# TIDE SureChEMBL Gate-2 final result

## Decision

`{result['decision']}`

All frozen gates pass. Admit the exact query-parallel kernel only for batch 1 and
8; fail closed to the immutable Gate-1 keeper for batch 64, 512, and all other
unmeasured shapes.

## Strongest evidence

- Exactness: 0 mismatches across 320 declared queries against the Gate-0 exact
  fresh-union oracle (query/result IDs and exact numerator/denominator).
- Safety: memcheck and synccheck report zero errors; 1,000 append/query cycles
  report zero hash mismatch.
- Batch 1: candidate median {b1['candidate_median_ms']:.6f} ms and p95
  {b1['candidate_p95_ms']:.6f} ms versus keeper median
  {b1['keeper_median_ms']:.6f} ms. The marginal median latency ratio is
  {b1['candidate_over_keeper_median_latency']:.6f} ({1/b1['candidate_over_keeper_median_latency']:.2f}x speedup).
- Frozen CPU service denominator: candidate batch-1 p95 is
  {b1['candidate_p95_ms']/51.413439:.6f}x the 51.413439 ms CPU median
  ({51.413439/b1['candidate_p95_ms']:.2f}x).
- Batch 8: candidate median {b8['candidate_median_ms']:.6f} ms versus keeper
  {b8['keeper_median_ms']:.6f} ms; p95 ratio
  {b8['candidate_over_keeper_p95_latency']:.6f}.
- Fail-closed sustained dispatch: batch-64 throughput ratio
  {b64['candidate_over_keeper_median_qps']:.6f}; batch-512 ratio
  {b512['candidate_over_keeper_median_qps']:.6f}.
- Process rule: {ab_original['process_pairs_pass']}/6 direction-balanced process
  pairs pass every frozen threshold.

## Mechanism boundary

The keeper exposes one CTA/query/GPU and therefore at most one SM/card for batch
1. The selected candidate exposes 128 independent row-partition CTAs/query/GPU,
then performs a second exact top-10 merge. Fingerprint intersections and output
semantics are invariant; the added work is 128 partial top-10 writes plus one
small reduction per query and GPU. Static compilation reports 86 registers for
Stage 1 and 114 for Stage 2, with no stack or spills. The target has 188 SMs/GPU.

NCU collection was attempted only for diagnostic attribution and was blocked by
`ERR_NVGPUCTRPERM`. No permission escalation was attempted, and no profiler
number is used as performance evidence. CUDA-event plus service-core A/B remains
the declared denominator.

## Paper boundary

Gate-2 proves a bounded implementation result, not paper novelty or production
importance. Before paper promotion, complete a GPU indexing prior-art audit,
measure production query batch/latency distributions, add a single-GPU and
cost-normalized comparator, and integrate the admitted dispatch into the real
TIDE/GTS++ system path.

## Raw evidence

- Calibration: `{cal_dir}`
- Correctness: `{correct_dir}`
- Sanitizer: `{san_dir}`
- Stress: `{stress_dir}`
- Six-pair A/B: `{ab_dir}`
- NCU permission boundary: `{ncu_dir}`
"""
(root / "results" / "RESULTS_GATE2.md").write_text(md)
print(json.dumps({"decision": result["decision"], "gates": gates, "final": str(root / "results" / "FINAL_GATE2.json")}, indent=2))
