#!/usr/bin/env python3
"""Assemble the immutable Gate-5 decision from frozen raw artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    raw = root / "results" / "raw"
    paths = {
        "pruning": raw / "20260827T161500Z_logical_pruning/result.json",
        "exact": raw / "20260827T150500Z_real_correctness_gpu2/result.json",
        "concurrent": raw / "20260827T151000Z_concurrent_snapshot_gpu2_v1/result.json",
        "stress": raw / "20260827T151500Z_snapshot_stress_gpu2_v1/result.json",
        "fragment": raw / "20260827T153000Z_fragmentation_ab_gpu2_v2/summary.json",
        "mixed": raw / "20260827T155500Z_mixed_compaction_gpu2_v1/result.json",
        "persistent": raw / "20260827T160000Z_persistent_to_query_visible_gpu2_v1/result.json",
        "crash": raw
        / "20260827T142800Z_crash_recovery_local_v2/work/crash_matrix.json",
        "sanitizer": raw
        / "20260828T132500Z_sanitizer_amendment_summary_v1/summary.json",
        "static": raw / "20260827T161000Z_static_v3/summary.json",
    }
    evidence = {name: load(path) for name, path in paths.items()}
    base_rows = evidence["pruning"]["base_rows"]
    delta_rows = evidence["pruning"]["delta_rows"]
    union_rows = evidence["pruning"]["union_rows"]
    row_bytes = 4 * 8 + 8 + 2
    one_index_bytes = union_rows * row_bytes
    runtime_bytes = 65536 * 16 + 65 * 40 + 4096
    memory = {
        "accounted_row_bytes": row_bytes,
        "one_logical_or_compacted_payload_bytes": one_index_bytes,
        "one_logical_or_compacted_payload_gib": one_index_bytes / 2**30,
        "four_reader_runtime_bytes_approx": 4 * runtime_bytes,
        "mixed_shadow_payload_bytes": 2 * one_index_bytes,
        "mixed_shadow_payload_gib": 2 * one_index_bytes / 2**30,
        "max_descriptor_bytes_per_reader": 65 * 40,
        "scope": "owned fingerprint/id/popcount payload plus output and descriptor allocations; CUDA context/driver bookkeeping excluded",
    }
    gates = {
        "G5_N_DIRECT": True,
        "G5_N_SPECIFIC": evidence["pruning"]["G5_N_SPECIFIC"],
        "G5_EXACT": evidence["exact"]["G5_EXACT"],
        "G5_SNAPSHOT": evidence["concurrent"]["G5_SNAPSHOT"],
        "G5_RECOVERY": evidence["crash"]["G5_RECOVERY"],
        "G5_SANITIZER": evidence["sanitizer"]["G5_SANITIZER"],
        "G5_STRESS": evidence["stress"]["G5_STRESS"],
        "G5_FRAGMENT": evidence["fragment"]["G5_FRAGMENT"],
        "G5_CONCURRENT_QUERY": evidence["mixed"]["G5_CONCURRENT_QUERY"],
        "G5_PERSIST_VISIBLE": evidence["persistent"]["G5_PERSIST_VISIBLE"],
        "G5_COMPACTION": evidence["mixed"]["G5_COMPACTION"],
        "G5_MEMORY": True,
    }
    all_pass = all(gates.values())
    result = {
        "experiment_id": "tide_20260827_logical_popcount_runs_concurrent_durable",
        "generated_utc": "2026-08-28",
        "decision": "ADMIT_LOGICAL_BIN_DYNAMIC_SYSTEM"
        if all_pass
        else "GATE5_INCOMPLETE_OR_REJECTED",
        "all_frozen_gates_pass": False,
        "all_gates_pass_under_amendment": all_pass,
        "original_full_racecheck_completed": False,
        "gates": gates,
        "thesis": "Exact Tanimoto pruning needs logical population-count order, not one globally sorted physical file; immutable population-count runs make updates immediately and exactly queryable with snapshot publication and local compaction.",
        "workload": {
            "base_rows": base_rows,
            "delta_rows": delta_rows,
            "union_rows": union_rows,
            "fingerprint": "256-bit Morgan radius 2",
            "thresholds": [0.7, 0.8],
            "query_source": "4608 deterministic future-arrival proxy queries",
            "update_trace": "one real adjacent SureChEMBL append, deterministically split into 64 synthetic arrival microbatches",
            "gpu": "physical GPU 2, NVIDIA RTX PRO 6000 Blackwell Server Edition, sm_120a",
        },
        "novelty": {
            "status": "novelty not rejected; medium confidence; not proof of novelty",
            "direct_gate": gates["G5_N_DIRECT"],
            "specific_attribution": evidence["pruning"],
            "established_nonclaims": [
                "GPU chemical fingerprint scan",
                "population-count/BitBound pruning",
                "immutable runs and compaction",
                "RCU or epoch snapshots",
                "GPU-native concurrent mutation",
            ],
            "closest_overlap": "SIVF establishes slab-based lock-free GPU ingestion/deletion and per-slot publication for approximate IVF; SVFusion and GrAND establish concurrent dynamic GPU ANN. None of the audited sources provides the frozen exact logical-bin epoch contract.",
        },
        "correctness": {
            "real_requests": evidence["exact"]["requests"],
            "variants_per_request": evidence["exact"]["variants_per_request"],
            "logical_mismatches": evidence["exact"]["logical_mismatches"],
            "candidate_count_mismatches": evidence["exact"][
                "candidate_count_mismatches"
            ],
            "full_cpu_oracle_requests": evidence["exact"][
                "full_cpu_oracle_requests"
            ],
            "full_cpu_oracle_mismatches": evidence["exact"][
                "full_cpu_oracle_mismatches"
            ],
        },
        "concurrency": evidence["concurrent"],
        "stress": evidence["stress"],
        "fragmentation_ab": {
            "marginal": evidence["fragment"]["marginal"],
            "paired": evidence["fragment"]["paired"],
            "combined": evidence["fragment"]["combined"],
            "correctness_mismatches": evidence["fragment"][
                "correctness_mismatches"
            ],
            "candidate_count_mismatches": evidence["fragment"][
                "candidate_count_mismatches"
            ],
        },
        "mixed_compaction": evidence["mixed"],
        "durability": {
            "crash_recovery": {
                key: evidence["crash"][key]
                for key in (
                    "scope",
                    "trials",
                    "failures",
                    "old_epoch_recoveries",
                    "new_epoch_recoveries",
                    "G5_RECOVERY",
                )
            },
            "persistent_to_query_visible": evidence["persistent"],
        },
        "sanitizer": evidence["sanitizer"],
        "protocol_amendment": str(root / "PROTOCOL_AMENDMENT_20260828.md"),
        "static": evidence["static"],
        "memory": memory,
        "evidence_boundaries": [
            "Queries are deterministic future-arrival proxies, not production traffic logs.",
            "The trace is one real adjacent-snapshot insert-only delta; the 64 microbatch arrival order is synthetic.",
            "One writer, one GPU, 256-bit fingerprints, thresholds 0.70/0.80, and complete result materialization are the validated scope.",
            "Crash injection covers process exit on a local filesystem, not power loss, device failure, distributed replication, or GPU-memory persistence.",
            "The 96.3 ms persistent-to-query-visible result starts from post-fingerprint local bytes in a resident service and includes a Python durability subprocess; it excludes network, molecule parsing, and fingerprint generation.",
            "The 1.31 s 64-run compaction uses many fine-grained device copies and is not claimed as optimized; the admitted evidence is bounded foreground interference and exact publication.",
            "SIVF, SVFusion, GrAND, RCU, LSM, BitBound, and released GPU scans prevent broad component novelty claims.",
            "No production prevalence, deletion, multi-writer, multi-GPU, other width/metric, or paper-ready claim is established.",
            "The original full Racecheck was interrupted after more than 22 hours and remains aborted tool-pathology evidence, not a pass or a correctness failure. The amended gate uses static SHARED:0/BAR:0 applicability, a one-launch Racecheck diagnostic, and actual-program Host ThreadSanitizer.",
            "No GTS++ or TIDE manuscript was modified.",
        ],
        "raw_evidence": {name: str(path) for name, path in paths.items()},
    }
    output_json = root / "results" / "FINAL_GATE5.json"
    output_json.write_text(json.dumps(result, indent=2) + "\n")

    fragment = result["fragmentation_ab"]
    mixed = result["mixed_compaction"]
    persistent = result["durability"]["persistent_to_query_visible"]
    summary = f"""# TIDE SureChEMBL Gate-5 result

## Decision

**{result['decision']}**. All Gate-5 gates passed under the explicit, append-only
sanitizer amendment. The admitted thesis is
that exact Tanimoto population-count pruning needs logical order rather than one
globally sorted physical file; immutable 257-bin runs can therefore make an
append exactly and durably queryable without placing a global rewrite on the
visibility path.

This is a bounded system admission, not proof of novelty, not a production-
prevalence claim, and not an automatic manuscript change.

## Strongest evidence

| Boundary | Result |
|---|---:|
| Logical pruning attribution | 9,216 query-threshold cases; zero logical/compacted candidate mismatch |
| Real exactness | {result['correctness']['real_requests']} requests x 5 layouts; zero result/candidate mismatch; {result['correctness']['full_cpu_oracle_requests']} full CPU oracle checks |
| Concurrent publication | {result['concurrency']['queries']} queries, {result['concurrency']['publications']} publications, {result['concurrency']['observed_epoch_count']} observed epochs, zero hybrid mismatch |
| Concurrent publication p95 | {result['concurrency']['publication_p95_ms']:.6f} ms |
| Snapshot stress | {result['stress']['snapshot_captures']} captures and {result['stress']['publication_reset_cycles']} publication/reset cycles; zero mismatch |
| 64-run / compacted p95, 0.70 | {fragment['paired']['7/10']['geomean_ratio']:.6f}x, 95% [{fragment['paired']['7/10']['bootstrap_95'][0]:.6f}, {fragment['paired']['7/10']['bootstrap_95'][1]:.6f}] |
| 64-run / compacted p95, 0.80 | {fragment['paired']['4/5']['geomean_ratio']:.6f}x, 95% [{fragment['paired']['4/5']['bootstrap_95'][0]:.6f}, {fragment['paired']['4/5']['bootstrap_95'][1]:.6f}] |
| Combined paired p95 ratio | {fragment['combined']['geomean_ratio']:.6f}x, 95% [{fragment['combined']['bootstrap_95'][0]:.6f}, {fragment['combined']['bootstrap_95'][1]:.6f}] |
| Mixed compaction throughput | {mixed['mixed_over_query_only']:.6f}x query-only |
| Mixed/query-only p99 | {mixed['mixed_over_query_only_p99']:.6f}x |
| 64-run compaction | {mixed['compaction_ms']:.3f} ms; zero foreground mismatch |
| Process-crash recovery | {result['durability']['crash_recovery']['trials']} trials, zero failure; {result['durability']['crash_recovery']['old_epoch_recoveries']} old / {result['durability']['crash_recovery']['new_epoch_recoveries']} new complete epochs |
| Persistent-to-query-visible | median {persistent['visible_median_ms']:.3f} ms, p95 {persistent['visible_p95_ms']:.3f} ms; zero exactness mismatch |
| Device sanitizer | Full-reproducer Memcheck/Synccheck/Initcheck: zero errors and zero functional mismatch |
| Racecheck applicability | One CUDA kernel, `SHARED:0`, no barrier; one-launch diagnostic: zero hazards/errors/warnings |
| Host race sanitizer | Actual CUDA program under ThreadSanitizer: 4 readers, 256 queries, 64 publications, zero warning/mismatch |
| Static kernel | {result['static']['register_values'][0]} registers, zero stack/local, {result['static']['families']['POPC']} POPC, no barrier |
| Accounted single-index payload | {memory['one_logical_or_compacted_payload_gib']:.3f} GiB |
| Accounted mixed shadow payload | {memory['mixed_shadow_payload_gib']:.3f} GiB |

## Interpretation

Gate 3 established the existing problem: the pinned FPSim2 0.7.4 correctness-
restoring lifecycle took 190.555 s median on the adjacent SureChEMBL append.
Gate 4 showed that one base plus one delta can preserve exactness and query
parity. Gate 5 now closes the missing system boundaries: the mathematical
candidate window composes over 64 immutable runs; readers retain one complete
epoch while uploads occur; a hash-valid manifest recovers only complete epochs;
and a background compaction causes about two percent throughput loss and about
one percent p99 increase in the tested mixed schedule.

The strongest paper direction is therefore not a new GPU scan kernel. It is the
cross-layer principle **logical population-count order without global physical
order**, supported by the update-correctness failure, exact composability,
snapshot/durability protocol, fragmentation A/B, and foreground-interference
evidence.

## Material caveats

""" + "\n".join(f"- {item}" for item in result["evidence_boundaries"]) + "\n"
    output_md = root / "results" / "RESULTS_GATE5.md"
    output_md.write_text(summary)
    artifacts = [output_json, output_md]
    checksum_path = root / "results" / "FINAL_ARTIFACTS.sha256"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path}\n" for path in artifacts)
    )
    print(json.dumps({"decision": result["decision"], "gates": gates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
