#!/usr/bin/env python3
"""CPU-only audit of the held-out static-AABB correctness runner."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
RUNNER = ROOT / "tools/run_static_aabb_heldout_correctness_v1.py"
CONTRACT = ROOT / "provenance/static_aabb_heldout_correctness_execution_buildbound_v1.json"
PREFLIGHT = ROOT / "provenance/static_aabb_heldout_correctness_execution_buildbound_preflight_v1.json"
SOURCE_AUDIT = ROOT / "tools/audit_static_aabb_heldout_correctness_v1.py"
BINARY = ROOT / "bin/static_aabb_topk_heldout_v1"
EXPECTED = {
    "contract": "b72b70e1f743162346e95bec09b77afec732017d1d4360a387843620a7eb6f71",
    "preflight": "912230814062826b5486b2958eae1f4699fcf1d73198b0dfc95370ce7f0a9089",
    "source_audit": "234004d39836aace8531e68dbf08f9991f5537a2b645b9b0c489957f71f0537e",
    "binary": "d10ec9a319927fbd35be27ad55d3cd8cc2cc75e4bd0aa257382e11fc0824a3de",
}
def sha(p: Path) -> str:
    if not p.is_file() or p.is_symlink():
        raise RuntimeError(f"invalid regular file: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""):
            h.update(x)
    return h.hexdigest()
def main() -> int:
    files = {"runner": RUNNER, "contract": CONTRACT, "preflight": PREFLIGHT, "source_audit": SOURCE_AUDIT, "binary": BINARY}
    got = {k: sha(v) for k,v in files.items()}
    for k,v in EXPECTED.items():
        if got[k] != v: raise RuntimeError(f"frozen artifact mismatch: {k}")
    s = RUNNER.read_text()
    required = [
        "--preflight-only", "cpu_preflight()", "nvidia_smi()", "gpu_guard.json",
        "PREPARING", "RUNNING", "FAILED", "COMPLETE_HELDOUT_CORRECTNESS_ONLY",
        "CUDA_VISIBLE_DEVICES\": GPU_UUID", "OMP_NUM_THREADS", "snapshot_pre_post_byte_equal",
        "PASS_STATIC_AABB_HELDOUT_CORRECTNESS_V1", "execution_contract", "execution_preflight",
        "no_process_kill_or_clock_change", "timeout=3600",
    ]
    for item in required:
        if item not in s: raise RuntimeError(f"runner missing: {item}")
    for forbidden in ("--lock-gpu-clocks", "nvidia-smi -rgc", "os.kill(", "pkill", "killall", "c2_speculative_fallback_v4/validation", "c2_speculative_fallback_v4/sealed"):
        if forbidden in s: raise RuntimeError(f"forbidden runner token: {forbidden}")
    preflight_branch = s.find("if args.preflight_only:")
    gpu_branch = s.find("nvsmi = nvidia_smi()")
    if preflight_branch < 0 or gpu_branch < 0 or preflight_branch > gpu_branch:
        raise RuntimeError("preflight-only could invoke nvidia-smi")
    contract = json.loads(CONTRACT.read_text())
    if contract.get("status") != "BUILD_BOUND_CPU_ONLY_EXECUTION_NOT_AUTHORIZED":
        raise RuntimeError("contract status")
    if "gpu_guard.json" not in contract.get("execution_gate", {}).get("required_runner_artifacts", []):
        raise RuntimeError("contract guard requirement")
    preflight = json.loads(PREFLIGHT.read_text())
    if preflight.get("status") != "PASS_CPU_ONLY_BUILD_BOUND_EXECUTION_BLOCKED":
        raise RuntimeError("buildbound preflight status")
    out = {
        "schema": "safe-c2-static-aabb-heldout-correctness-runner-audit-v1",
        "status": "PASS_CPU_ONLY_RUNNER_AUDITED",
        "scope": "static runner audit only; no nvidia-smi or CUDA binary execution",
        "runner_sha256": got["runner"],
        "binary_sha256": got["binary"],
        "buildbound_contract_sha256": got["contract"],
        "buildbound_preflight_sha256": got["preflight"],
        "source_audit_sha256": got["source_audit"],
        "required_run_artifacts": ["preflight.json", "gpu_guard.json", "terminal.json", "manifest.json"],
        "formal_claim_eligible": False,
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
    }
    print(json.dumps(out, sort_keys=True))
    return 0
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_STATIC_AABB_HELDOUT_CORRECTNESS_RUNNER_AUDIT_V1: {exc}", file=sys.stderr)
        raise SystemExit(2)
