#!/usr/bin/env python3
"""Freeze the fresh held-out SIFT static-AABB performance protocol (CPU-only)."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
OUT = ROOT / "provenance/static_aabb_heldout_perf_evidence_protocol_v2.json"
IDS = ROOT / "inputs/standard_sift_static_aabb_perfheldout256_v2.ids"
SELECTION = ROOT / "provenance/static_aabb_perfheldout_selection_v2.json"
BINARY = ROOT / "bin/static_aabb_perf_probe_v2_1_diskguard"
SOURCE = ROOT / "src/static_aabb_perf_probe_v2_1_diskguard.cu"
SOURCE_AUDIT = ROOT / "tools/audit_diskguard_v2_1_sources.py"
DISK_CONTRACT = ROOT / "provenance/sift_integer_disk_contract_v1.json"
BASE = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs")
QUERY = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs")
GT = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs")
PREDECESSOR_TERMINAL = ROOT / "runs/static_aabb_heldout_perf_evidence_v1_20260728T062931Z_d605dc/terminal.json"


def sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"not a regular file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def art(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def atomic(path: Path, obj: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing pre-existing output: {path}")
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
    if selection.get("status") != "FROZEN_CPU_ONLY":
        raise RuntimeError("selection state")
    ids = [int(x) for x in IDS.read_text().splitlines()]
    if len(ids) != 256 or ids != sorted(ids) or len(set(ids)) != 256:
        raise RuntimeError("fresh IDs")
    if any(5000 <= q <= 5259 for q in ids):
        raise RuntimeError("predecessor interval overlap")
    predecessor = json.loads(PREDECESSOR_TERMINAL.read_text())
    if predecessor.get("status") != "FAILED":
        raise RuntimeError("expected predecessor failed terminal")
    obj: dict[str, object] = {
        "schema": "safe-c2-static-aabb-heldout-perf-protocol-v2",
        "status": "FROZEN_CPU_ONLY_PREPARED",
        "scope": (
            "single fresh standard-SIFT1M static raw-float32 L2 AABB paired-performance "
            "telemetry batch; conditional relative evidence only"
        ),
        "formal_claim_eligible": False,
        "algorithm_binding": {
            "method": "static AABB lower-bound pruning",
            "metric": "raw float32 L2",
            "dimensions": 128,
            "disk_guard": "SIFT integer-coordinate next-up then directed-round-up square guard",
            "binary": art(BINARY),
            "source": art(SOURCE),
            "source_audit": art(SOURCE_AUDIT),
            "disk_contract": art(DISK_CONTRACT),
        },
        "fresh_heldout_binding": {
            "ids": art(IDS),
            "selection": art(SELECTION),
            "count": len(ids),
            "first_id": ids[0],
            "last_id": ids[-1],
            "selection_only_uses": "CPU binary64 GT rank-10/rank-11 strict-boundary eligibility",
        },
        "dataset_binding": {"base": art(BASE), "query": art(QUERY), "groundtruth": art(GT)},
        "predecessor_failure_boundary": {
            "terminal": art(PREDECESSOR_TERMINAL),
            "status": "FAILED",
            "interpretation": (
                "predecessor timing artifacts are excluded from any result claim; "
                "this successor uses disjoint fresh IDs and a new runner/auditor"
            ),
        },
        "measurement": {
            "warmups_per_process": 5,
            "paired_repetitions_per_process": 30,
            "independent_serial_processes": 5,
            "pair_order": "alternating baseline_then_static_aabb on even reps and reverse on odd reps",
            "target_gpu_selection": "physical UUID, not ordinal",
        },
        "telemetry_contract": {
            "sample_period_target_ms": 100,
            "max_inter_sample_gap_ms": 500,
            "phase_semantics": (
                "phase=target for the full Popen lifetime of the actual CUDA binary; "
                "target-phase telemetry must cover Popen start/end within 500 ms"
            ),
            "pid_semantics": (
                "nvidia-smi compute-process visibility is an auxiliary identity/isolation "
                "observation, not a proxy for CPU initialization or result-write lifetime"
            ),
            "acceptance": [
                "three prelaunch samples with no compute process",
                "no non-target compute PID in any target-phase sample",
                "no compute PID in prelaunch/posttarget samples",
                "each actual target PID observed at least once in target phase",
                "numeric telemetry complete and global/target-phase gaps at most 500 ms",
                "no runner clock/power/compute-mode change",
            ],
            "evidence_label": "CONDITIONAL_PAIRED_DVFS_UNCONTROLLED",
        },
        "claim_boundaries": [
            "static raw-float32 L2 AABB on standard SIFT1M only",
            "no C3 insertion/update/ancestor-maintenance claim",
            "no general metric-space claim",
            "no fixed-clock, hardware-isolated, or stable absolute-latency claim",
            "no calibration/tuning from predecessor timing artifacts",
        ],
        "required_successor_artifacts": [
            "preflight.json", "runner_freeze_reference.json", "environment.json",
            "gpu_state_before.xml", "gpu_state_after.xml", "gpu_telemetry.csv",
            "gpu_process_samples.jsonl", "telemetry_summary.json", "launch.json",
            "timing_pairs.json", "result.json", "manifest.json", "terminal.json",
            "postrun_audit.json",
        ],
        "execution_boundary": {
            "does_not_authorize_gpu_execution": True,
            "gpu_executed_for_protocol_freeze": False,
            "nvidia_smi_called_for_protocol_freeze": False,
            "does_not_modify_old_c2_validation_or_sealed": True,
            "successor_output_prefix": "static_aabb_heldout_perf_evidence_v3_",
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
        print(f"FAIL_STATIC_AABB_HELDOUT_PERF_PROTOCOL_V2: {exc}", file=sys.stderr)
        raise SystemExit(2)
