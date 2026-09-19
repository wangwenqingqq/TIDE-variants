#!/usr/bin/env python3
"""CPU-only integrity preflight for the frozen held-out correctness execution contract."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
BASE_PROTOCOL = ROOT / "provenance/static_aabb_heldout_protocol_v1.json"
EXECUTION = ROOT / "provenance/static_aabb_heldout_correctness_execution_v1.json"
AUDIT = ROOT / "tools/audit_static_aabb_heldout_correctness_source_v1.py"
EXACT_IDS = ROOT / "inputs/standard_sift_static_aabb_heldout_exact64_v1.ids"
FULL_IDS = ROOT / "inputs/standard_sift_static_aabb_heldout256_v1.ids"
PLANNED_BINARY = ROOT / "bin/static_aabb_topk_heldout_v1"


def sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid regular file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ids(path: Path) -> list[int]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid IDs: {path}")
    out = [int(line) for line in path.read_text().splitlines()]
    if out != sorted(out) or len(set(out)) != len(out):
        raise RuntimeError("noncanonical IDs")
    return out


def main() -> int:
    for path in (BASE_PROTOCOL, EXECUTION, AUDIT, EXACT_IDS, FULL_IDS):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"invalid input: {path}")
    if PLANNED_BINARY.exists() or PLANNED_BINARY.is_symlink():
        raise RuntimeError("binary now exists; source-only execution preflight is intentionally stale")

    base = json.loads(BASE_PROTOCOL.read_text())
    execution = json.loads(EXECUTION.read_text())
    if execution.get("schema") != "safe-c2-static-aabb-heldout-correctness-execution-v1":
        raise RuntimeError("execution schema")
    if execution.get("status") != "FROZEN_CPU_ONLY_SOURCE_BOUND_NOT_BUILT":
        raise RuntimeError("execution status")
    if execution.get("formal_claim_eligible") is not False:
        raise RuntimeError("formal eligibility")
    inherited = execution["inherits_heldout_protocol"]
    if inherited["sha256"] != sha256(BASE_PROTOCOL) or inherited["old_v2_1_runner_reuse_allowed"] is not False:
        raise RuntimeError("base protocol binding")
    exact = ids(EXACT_IDS)
    full = ids(FULL_IDS)
    fixed = execution["fixed_input_split"]
    if (len(exact) != 64 or len(full) != 256 or exact != full[:64] or
            fixed["gpu_heldout_correctness"]["ids_file"]["sha256"] != sha256(EXACT_IDS) or
            fixed["gt_gated_heldout_performance"]["ids_file"]["sha256"] != sha256(FULL_IDS)):
        raise RuntimeError("fixed input split")
    if fixed["gpu_heldout_correctness"]["ids"] != exact:
        raise RuntimeError("exact64 IDs binding")
    if execution["implementation_freeze"]["projection"]["dimensions"] != 128:
        raise RuntimeError("projection dimension")
    audit_now = json.loads(subprocess.check_output([str(AUDIT)], text=True))
    embedded = execution["implementation_freeze"]["source_audit"]
    if embedded["sha256"] != sha256(AUDIT) or embedded["result"] != audit_now:
        raise RuntimeError("source audit changed")
    build = execution["build_contract"]
    if build["status"] != "NOT_BUILT_NO_BINARY_EXECUTION" or build["planned_binary"]["sha256"] is not None:
        raise RuntimeError("build status")
    if "-Xcompiler=-fopenmp" not in build["required_command_argv"]:
        raise RuntimeError("OpenMP build flag missing")
    runner = execution["required_successor_runner"]
    if runner["status"] != "NOT_IMPLEMENTED" or not runner["terminal_requirements"]:
        raise RuntimeError("runner disposition")
    result = {
        "schema": "safe-c2-static-aabb-heldout-correctness-execution-preflight-v1",
        "status": "PASS_CPU_ONLY_SOURCE_BOUND_NOT_BUILT",
        "scope": "verification of source-bound correctness execution contract; no compile, GPU execution, nvidia-smi, timing, old-C2 validation, or sealed access",
        "base_protocol": {"path": str(BASE_PROTOCOL), "sha256": sha256(BASE_PROTOCOL)},
        "execution_contract": {"path": str(EXECUTION), "sha256": sha256(EXECUTION)},
        "source_audit": audit_now,
        "exact64": {"path": str(EXACT_IDS), "sha256": sha256(EXACT_IDS), "count": len(exact)},
        "full256": {"path": str(FULL_IDS), "sha256": sha256(FULL_IDS), "count": len(full)},
        "planned_binary_exists": False,
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
        "formal_claim_eligible": False,
        "next_required_action": "implement/audit the successor runner, then perform an isolated CPU-only build that creates a new binary build card before any GPU execution",
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_STATIC_AABB_HELDOUT_CORRECTNESS_EXECUTION_PREFLIGHT_V1: {exc}", file=sys.stderr)
        raise SystemExit(2)
