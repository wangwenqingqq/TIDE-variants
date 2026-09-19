#!/usr/bin/env python3
"""Freeze source/binary/held-out bindings for the timeline-attributed perf batch."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
OUT = ROOT / "provenance/static_aabb_timeline_perf_protocol_v3.json"
IDS = ROOT / "inputs/standard_sift_static_aabb_perfheldout256_v2.ids"
SELECTION = ROOT / "provenance/static_aabb_perfheldout_selection_v2.json"
SOURCE = ROOT / "src/static_aabb_perf_probe_v3_timeline.cu"
BINARY = ROOT / "bin/static_aabb_perf_probe_v3_timeline"
SOURCE_AUDIT = ROOT / "provenance/static_aabb_perf_probe_v3_timeline_source_audit_v1.json"
BUILD = ROOT / "provenance/static_aabb_perf_probe_v3_timeline_build_v1.json"
POSTRUN_AUDIT = ROOT / "tools/audit_static_aabb_heldout_perf_result_v3.py"
PARENT_SOURCE = ROOT / "src/static_aabb_perf_probe_v2_1_diskguard.cu"
PARENT_BINARY = ROOT / "bin/static_aabb_perf_probe_v2_1_diskguard"
DISK_CONTRACT = ROOT / "provenance/sift_integer_disk_contract_v1.json"
BASE = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs")
QUERY = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs")
GT = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs")
PREDECESSOR_TERMINAL = ROOT / "runs/static_aabb_heldout_perf_evidence_v1_20260728T062931Z_d605dc/terminal.json"


def sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid regular artifact: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def art(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def atomic(path: Path, obj: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing pre-existing protocol: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, sort_keys=True, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    selection = json.loads(SELECTION.read_text())
    source_audit = json.loads(SOURCE_AUDIT.read_text())
    build = json.loads(BUILD.read_text())
    predecessor = json.loads(PREDECESSOR_TERMINAL.read_text())
    ids = [int(x) for x in IDS.read_text().splitlines()]
    if (selection.get("status") != "FROZEN_CPU_ONLY" or
            source_audit.get("status") != "PASS_SOURCE_ONLY_AUDIT" or
            build.get("status") != "COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION" or
            predecessor.get("status") != "FAILED"):
        raise RuntimeError("upstream artifact status")
    if len(ids) != 256 or ids != sorted(ids) or len(set(ids)) != 256 or any(5000 <= x <= 5259 for x in ids):
        raise RuntimeError("fresh heldout IDs")
    if build.get("source", {}).get("sha256") != sha(SOURCE) or build.get("binary", {}).get("sha256") != sha(BINARY):
        raise RuntimeError("build binding")
    if source_audit.get("source_v3", {}).get("sha256") != sha(SOURCE):
        raise RuntimeError("source audit binding")
    obj: dict[str, object] = {
        "schema": "safe-c2-static-aabb-timeline-perf-protocol-v3",
        "status": "FROZEN_CPU_ONLY_PREPARED",
        "scope": (
            "one fresh standard-SIFT1M static raw-float32 L2 AABB paired-performance "
            "batch with source-defined CUDA-event timing envelopes and read-only telemetry"
        ),
        "formal_claim_eligible": False,
        "algorithm_binding": {
            "candidate": "static AABB lower-bound pruning",
            "baseline": "archived radial traversal",
            "metric": "raw float32 L2",
            "dimensions": 128,
            "disk_guard": "SIFT integer-coordinate next-up then directed-round-up square guard",
            "parent_source": art(PARENT_SOURCE),
            "parent_binary": art(PARENT_BINARY),
            "timeline_source": art(SOURCE),
            "timeline_binary": art(BINARY),
            "source_audit": art(SOURCE_AUDIT),
            "build": art(BUILD),
            "disk_contract": art(DISK_CONTRACT),
        },
        "fresh_heldout_binding": {
            "ids": art(IDS),
            "selection": art(SELECTION),
            "count": len(ids),
            "first_id": ids[0],
            "last_id": ids[-1],
            "disjoint_from_predecessor_observed_interval": [5000, 5259],
            "selection_rule": "CPU binary64 strict GT rank-10/rank-11 L2 boundary only; no traversal, timing, AABB, or prior-run reads",
        },
        "dataset_binding": {"base": art(BASE), "query": art(QUERY), "groundtruth": art(GT)},
        "predecessor_failure_boundary": {
            "failed_terminal": art(PREDECESSOR_TERMINAL),
            "interpretation": "failed predecessor timings are not used; this protocol has new IDs, binary, runner, and verifier",
        },
        "timeline_contract": {
            "CLI": "--run-token required and bound to runner-generated token",
            "clock_id": "CLOCK_MONOTONIC",
            "envelope": "after final warmup synchronization/cleanup and before any post-measurement snapshot/result writing",
            "operations": {
                "count": 60,
                "reps": 30,
                "methods_per_rep": ["baseline", "static_aabb"],
                "ordering": "baseline_then_static_aabb on even reps, static_aabb_then_baseline on odd reps",
                "fields": ["op_index", "rep", "method", "order", "host_start_ns", "host_end_ns", "cuda_ms"],
            },
            "publish": "benchmark_timeline.json written atomically after all operations and before snapshots",
        },
        "telemetry_contract": {
            "target_gpu": "physical UUID only; runner records physical index/UUID/PCI",
            "sample_period_target_ms": 100,
            "max_gap_ms": 500,
            "read_only_commands": [
                "nvidia-smi --query-gpu",
                "nvidia-smi --query-compute-apps",
                "nvidia-smi -q -x",
            ],
            "attribution": (
                "post-run verifier attributes telemetry to the source-defined timeline envelope; "
                "Popen start/exit are identity/lifecycle bounds, not the benchmark envelope"
            ),
            "acceptance": [
                "at least three prelaunch empty compute-process samples",
                "at least three empty samples after each child exits",
                "no non-target PID during any child lifetime",
                "timeline PID/token bound to the actual Popen child",
                "within each timeline envelope target PID is visible, alone, at both boundaries within 500 ms and with no internal gap above 500 ms",
                "all numeric telemetry fields present; query failures fail closed",
                "runner changes no application clocks, power limit, persistence, or compute mode",
            ],
            "evidence_grade": "CONDITIONAL_PAIRED_DVFS_UNCONTROLLED",
        },
        "matching_postrun_auditor": art(POSTRUN_AUDIT),
        "claim_boundaries": [
            "only static raw-float32 L2 AABB on standard SIFT1M",
            "only the frozen fresh query set and recorded unlocked-DVFS trace",
            "no stable absolute latency, fixed-clock, or hardware-exclusive claim",
            "no general metric-space, C3 insertion/update, calibration, drift, or deployment claim",
        ],
        "required_run_artifacts": [
            "preflight.json", "runner_freeze_reference.json", "environment.json",
            "gpu_state_before.xml", "gpu_state_after.xml", "gpu_telemetry.csv",
            "gpu_process_samples.jsonl", "launch.json", "timing_pairs.json",
            "telemetry_attribution.json", "result.json", "postrun_audit.json",
            "manifest.json", "terminal.json",
        ],
        "execution_boundary": {
            "does_not_authorize_gpu_execution": True,
            "gpu_executed_for_protocol_freeze": False,
            "nvidia_smi_called_for_protocol_freeze": False,
            "successor_output_prefix": "static_aabb_timeline_perf_v3_",
            "old_c2_validation_or_sealed_accessed": False,
        },
        "freezer": art(Path(__file__).resolve()),
    }
    atomic(OUT, obj)
    print(json.dumps({"status": obj["status"], "protocol": str(OUT), "sha256": sha(OUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_STATIC_AABB_TIMELINE_PERF_PROTOCOL_V3: {exc}", file=sys.stderr)
        raise SystemExit(2)
