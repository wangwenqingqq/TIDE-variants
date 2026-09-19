#!/usr/bin/env python3
"""CPU-only integrity preflight for the build-bound correctness contract."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
CONTRACT = ROOT / "provenance/static_aabb_heldout_correctness_execution_buildbound_v1.json"
BUILD_CARD = ROOT / "provenance/static_aabb_heldout_correctness_build_v2.json"
BINARY = ROOT / "bin/static_aabb_topk_heldout_v1"
PARENT_AUDIT = ROOT / "tools/audit_static_aabb_heldout_correctness_v1.py"
EXTRA_AUDIT = ROOT / "tools/audit_static_aabb_heldout_correctness_source_v1.py"
EXACT_IDS = ROOT / "inputs/standard_sift_static_aabb_heldout_exact64_v1.ids"
FULL_IDS = ROOT / "inputs/standard_sift_static_aabb_heldout256_v1.ids"


def sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid regular file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    for path in (CONTRACT, BUILD_CARD, BINARY, PARENT_AUDIT, EXTRA_AUDIT, EXACT_IDS, FULL_IDS):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"invalid input: {path}")
    contract = json.loads(CONTRACT.read_text())
    card = json.loads(BUILD_CARD.read_text())
    if contract.get("status") != "BUILD_BOUND_CPU_ONLY_EXECUTION_NOT_AUTHORIZED":
        raise RuntimeError("contract status")
    if contract.get("formal_claim_eligible") is not False:
        raise RuntimeError("formal status")
    if card.get("status") != "COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION":
        raise RuntimeError("build card status")
    if card["binary"]["sha256"] != sha256(BINARY):
        raise RuntimeError("binary mismatch")
    bound = contract["build_binding"]
    if bound["build_card"]["sha256"] != sha256(BUILD_CARD) or bound["binary"]["sha256"] != sha256(BINARY):
        raise RuntimeError("build binding")
    exact = [int(line) for line in EXACT_IDS.read_text().splitlines()]
    full = [int(line) for line in FULL_IDS.read_text().splitlines()]
    fixed = contract["fixed_input_split"]
    if (len(exact) != 64 or len(full) != 256 or exact != full[:64] or
            fixed["exact64"]["sha256"] != sha256(EXACT_IDS) or
            fixed["full256_for_separate_future_perf_only"]["sha256"] != sha256(FULL_IDS)):
        raise RuntimeError("input split")
    parent = json.loads(subprocess.check_output([str(PARENT_AUDIT)], text=True))
    extra = json.loads(subprocess.check_output([str(EXTRA_AUDIT)], text=True))
    if parent != bound["source_audits"]["build_card_audit"]["result"]:
        raise RuntimeError("parent audit changed")
    if extra != bound["source_audits"]["additional_protocol_audit"]["result"]:
        raise RuntimeError("additional audit changed")
    gate = contract["execution_gate"]
    if gate["status"] != "BLOCKED_UNTIL_SEPARATE_RUNNER_AUDITED" or not gate["required_runner_artifacts"]:
        raise RuntimeError("execution gate")
    result = {
        "schema": "safe-c2-static-aabb-heldout-correctness-buildbound-preflight-v1",
        "status": "PASS_CPU_ONLY_BUILD_BOUND_EXECUTION_BLOCKED",
        "scope": "integrity verification only; no GPU execution, nvidia-smi, timing, old-C2 validation, or sealed access",
        "execution_contract": {"path": str(CONTRACT), "sha256": sha256(CONTRACT)},
        "build_card": {"path": str(BUILD_CARD), "sha256": sha256(BUILD_CARD)},
        "binary": {"path": str(BINARY), "sha256": sha256(BINARY), "bytes": BINARY.stat().st_size},
        "exact64_count": len(exact),
        "full256_count": len(full),
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
        "formal_claim_eligible": False,
        "next_required_action": "wait for and audit the separate successor runner/guard/terminal contract before any GPU invocation",
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_STATIC_AABB_HELDOUT_CORRECTNESS_BUILDBOUND_PREFLIGHT_V1: {exc}", file=sys.stderr)
        raise SystemExit(2)
