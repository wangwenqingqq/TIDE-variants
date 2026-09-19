#!/usr/bin/env python3
"""Freeze a CPU-only, auditable performance-environment protocol for static AABB."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
HELDOUT = ROOT / "provenance/static_aabb_heldout_protocol_v1.json"
OUT = ROOT / "provenance/static_aabb_heldout_perf_evidence_protocol_v1.json"
V21_RUNNER = ROOT / "tools/run_diskguard_v2_1_m128.sh"
V21_BIN = ROOT / "bin/static_aabb_perf_probe_v2_1_diskguard"
V21_SRC = ROOT / "src/static_aabb_perf_probe_v2_1_diskguard.cu"
V21_AUDIT = ROOT / "tools/audit_diskguard_v2_1_sources.py"
EXPECTED = {
    "perf_binary": "e587ad822de007880e1caae8ea89072c07a5b786dcb5d15357ecde61ce577cde",
    "perf_source": "5dc92a9660bc168e1325edae7a1c9d41f59a93e036e83f57f2ccd9d55fa6819f",
}


def sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_new(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing pre-existing output: {path}")
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def main() -> int:
    for path in (HELDOUT, V21_RUNNER, V21_BIN, V21_SRC, V21_AUDIT):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or unsafe input: {path}")
    heldout = json.loads(HELDOUT.read_text())
    if heldout.get("schema") != "safe-c2-static-aabb-heldout-protocol-v1":
        raise RuntimeError("unexpected heldout protocol schema")
    if heldout.get("status") != "FROZEN_CPU_ONLY_PREPARED":
        raise RuntimeError("heldout protocol is not frozen CPU-only")
    algorithm = heldout.get("algorithm_freeze", {}).get("source_and_binary_artifacts", {})
    if algorithm.get("perf_binary", {}).get("sha256") != EXPECTED["perf_binary"]:
        raise RuntimeError("heldout protocol perf binary mismatch")
    if algorithm.get("perf_source", {}).get("sha256") != EXPECTED["perf_source"]:
        raise RuntimeError("heldout protocol perf source mismatch")
    if sha256(V21_BIN) != EXPECTED["perf_binary"] or sha256(V21_SRC) != EXPECTED["perf_source"]:
        raise RuntimeError("v2.1 artifact no longer matches frozen heldout protocol")
    heldout_ids = heldout.get("heldout_input", {}).get("gt_gated_heldout_performance", {})
    if heldout_ids.get("query_count") != 256 or not heldout_ids.get("ids_file", {}).get("sha256"):
        raise RuntimeError("heldout performance input missing")

    protocol = {
        "schema": "safe-c2-static-aabb-heldout-perf-evidence-protocol-v1",
        "status": "FROZEN_CPU_ONLY_PREPARED",
        "scope": (
            "minimum-intrusion, auditable paired-performance environment protocol for the already "
            "frozen static-AABB standard-SIFT development-heldout test; it records rather than "
            "locks GPU state"
        ),
        "parent_heldout_protocol": {
            "path": str(HELDOUT),
            "sha256": sha256(HELDOUT),
            "status": heldout["status"],
        },
        "algorithm_binding": {
            "implementation": "unmodified v2.1 SIFT-specific disk-guard performance binary",
            "perf_binary": {"path": str(V21_BIN), "sha256": sha256(V21_BIN), "bytes": V21_BIN.stat().st_size},
            "perf_source": {"path": str(V21_SRC), "sha256": sha256(V21_SRC), "bytes": V21_SRC.stat().st_size},
            "source_audit": {"path": str(V21_AUDIT), "sha256": sha256(V21_AUDIT)},
            "legacy_exploratory_runner": {"path": str(V21_RUNNER), "sha256": sha256(V21_RUNNER)},
            "must_not_modify_or_rebuild_v2_1": True,
        },
        "heldout_binding": {
            "ids_file": heldout_ids["ids_file"],
            "query_count": 256,
            "GT_role": (
                "fixed standard top-10 label for the pre-timing correctness gate only; "
                "not a timing, clock, projection, or calibration signal"
            ),
        },
        "successor_runner_requirements": {
            "required": True,
            "reason": (
                "the legacy v2.1 runner explicitly labels its results exploratory and does not "
                "record timing-interval GPU telemetry or continuous process observations"
            ),
            "must_be_new_path": True,
            "must_be_frozen_by_cpu_only_runner_freeze": True,
            "must_invoke_unmodified_v2_1_binary": True,
            "must_not_overwrite_any_v2_1_run_directory": True,
            "physical_gpu_binding": (
                "resolve and record a physical GPU UUID first, then set CUDA_VISIBLE_DEVICES to "
                "that UUID; do not rely only on an ordinal such as CUDA_VISIBLE_DEVICES=0"
            ),
            "preflight": [
                "run the frozen CPU-only heldout preflight and retain its exact stdout JSON",
                "verify the heldout IDs, v2.1 binary, v2.1 source, and source-audit SHA256 values",
                "take at least three idle/process samples before target launch; abort if a foreign compute PID is observed",
                "record GPU UUID, PCI bus ID, model, driver, compute mode, persistence state, power limit, and raw pre-launch nvidia-smi XML",
            ],
            "measurement": {
                "warmups_per_method": 5,
                "paired_repetitions_per_invocation": 30,
                "independent_process_invocations": 5,
                "order": (
                    "within each invocation, alternate baseline_then_static_aabb for even rep "
                    "and static_aabb_then_baseline for odd rep; record both raw arrays and "
                    "a derived per-pair ratio"
                ),
                "aggregation": (
                    "primary statistic: per-invocation median of the 30 pairwise "
                    "static_aabb_gpu_ms / baseline_gpu_ms ratios; report all five invocation "
                    "medians and their median/range, never a cherry-picked invocation"
                ),
                "absolute_latency_rule": (
                    "without approved clock control, absolute milliseconds are descriptive only "
                    "and must not be presented as stable hardware latency"
                ),
            },
            "postconditions": [
                "require the existing GT correctness gate, zero mismatches, and byte-identical static snapshots",
                "retain raw result.json, stdout/stderr, snapshot SHA256 list, terminal status, and a hash manifest",
                "derive timing_pairs.json from result.json without changing or reordering its raw timing arrays",
            ],
        },
        "telemetry_contract": {
            "collector_policy": "read-only nvidia-smi queries only; this protocol does not authorize clock, power-limit, or compute-mode changes",
            "target_sample_interval_ms": 100,
            "max_accepted_inter_sample_gap_ms": 500,
            "coverage": (
                "start continuous sampling before target launch and retain it until after target exit; "
                "sample processes and GPU state on the same physical GPU UUID throughout"
            ),
            "gpu_telemetry_csv_required_columns": [
                "wall_utc_ns",
                "monotonic_ns",
                "sample_seq",
                "phase",
                "gpu_index",
                "gpu_uuid",
                "pstate",
                "clock_sm_mhz",
                "clock_graphics_mhz",
                "clock_memory_mhz",
                "power_draw_w",
                "power_limit_w",
                "temperature_gpu_c",
                "utilization_gpu_pct",
                "utilization_memory_pct",
                "memory_used_mib",
            ],
            "process_samples_jsonl_required_fields": [
                "wall_utc_ns",
                "monotonic_ns",
                "sample_seq",
                "phase",
                "gpu_uuid",
                "compute_processes",
                "raw_query_output",
            ],
            "raw_gpu_state_files": [
                "gpu_state_before.xml",
                "gpu_state_after.xml",
            ],
            "required_telemetry_summary": [
                "sample count, first/last monotonic timestamp, and maximum inter-sample gap",
                "min/max/median SM clock, graphics clock, memory clock, power draw, temperature, and GPU utilization",
                "set of observed P-states",
                "all observed compute PIDs and whether any PID other than the target was sampled",
            ],
        },
        "isolation_and_clock_evidence": {
            "required_launch_metadata": [
                "target PID",
                "physical GPU UUID/index/PCI bus ID",
                "CUDA_VISIBLE_DEVICES exact value",
                "compute mode observed before and after",
                "target command line with hashes of binary, IDs, and protocol",
                "target_start_monotonic_ns and target_exit_monotonic_ns",
            ],
            "sampled_idle_guard": (
                "a clean sampled process log means no foreign compute PID was observed at the "
                "sampling instants; it is not a proof of hardware-level exclusivity"
            ),
            "clock_control": {
                "required_record": {
                    "runner_attempted_clock_lock": False,
                    "runner_attempted_application_clocks": False,
                    "runner_attempted_power_limit_change": False,
                    "runner_attempted_compute_mode_change": False,
                    "evidence_label": "DVFS_UNCONTROLLED_RECORDED",
                },
                "interpretation": (
                    "the runner can prove only that it did not request control changes and can "
                    "record observed clocks; it cannot infer a globally unlocked device from "
                    "one snapshot"
                ),
            },
            "evidence_grades": {
                "CONDITIONAL_PAIRED_DVFS_UNCONTROLLED": (
                    "all hashes, telemetry coverage, and sampled process checks pass; usable only "
                    "for a conditionally stated paired relative result under the recorded DVFS trace"
                ),
                "EXPLORATORY_TELEMETRY_INSUFFICIENT": (
                    "missing telemetry/process coverage, large sampling gaps, or missing requested fields; "
                    "do not use for a paper performance claim"
                ),
                "INVALID_FOREIGN_COMPUTE_INTERFERENCE": (
                    "a non-target compute PID is observed while the target is alive; do not use the "
                    "run for a performance claim"
                ),
                "HARDWARE_CONTROLLED": (
                    "not available under this protocol. It would require a separately approved "
                    "protocol and evidence for exclusive isolation plus clock/power policy control"
                ),
            },
        },
        "terminal_contract": {
            "success_status": "COMPLETE_CONDITIONAL_DVFS_UNCONTROLLED",
            "success_exit_code": 0,
            "formal_claim_eligible": False,
            "failure_rule": "any preflight, guard, telemetry, binary, or result failure must leave a FAILED terminal JSON; no result may be silently promoted",
        },
        "required_run_artifacts": [
            "preflight.json",
            "runner_freeze_reference.json",
            "launch.json",
            "environment.json",
            "gpu_state_before.xml",
            "gpu_state_after.xml",
            "gpu_telemetry.csv",
            "gpu_process_samples.jsonl",
            "telemetry_summary.json",
            "timing_pairs.json",
            "stdout.log",
            "stderr.log",
            "result.json",
            "snapshot_sha256.txt",
            "terminal.json",
            "manifest.json",
        ],
        "claim_boundaries": [
            "static raw-float32 L2 AABB on standard SIFT1M only",
            "no C3 insertion/update/AABB-maintenance correctness claim",
            "no general-metric-space claim from this protocol",
            "no fixed-clock or stable absolute-latency claim without a separate approved hardware-control protocol",
            "no formal claim from protocol freeze alone",
        ],
        "hard_boundaries": {
            "gpu_executed_for_this_protocol": False,
            "nvidia_smi_called_for_this_protocol": False,
            "clock_or_power_or_compute_mode_changed_for_this_protocol": False,
            "old_c2_validation_or_sealed_accessed": False,
            "does_not_authorize_gpu_execution": True,
        },
        "formal_claim_eligible": False,
    }
    atomic_write_new(OUT, json.dumps(protocol, sort_keys=True, indent=2) + "\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL_STATIC_AABB_HELDOUT_PERF_EVIDENCE_PROTOCOL_V1: {error}", file=sys.stderr)
        raise SystemExit(2)
