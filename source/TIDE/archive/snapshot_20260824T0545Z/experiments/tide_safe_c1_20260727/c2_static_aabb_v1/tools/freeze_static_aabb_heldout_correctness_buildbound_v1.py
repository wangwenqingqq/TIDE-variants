#!/usr/bin/env python3
"""Bind the independently built held-out correctness binary without authorizing execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
BASE_PROTOCOL = ROOT / "provenance/static_aabb_heldout_protocol_v1.json"
SOURCEBOUND = ROOT / "provenance/static_aabb_heldout_correctness_execution_v1.json"
BUILD_CARD = ROOT / "provenance/static_aabb_heldout_correctness_build_v2.json"
OUT = ROOT / "provenance/static_aabb_heldout_correctness_execution_buildbound_v1.json"
PARENT_AUDIT = ROOT / "tools/audit_static_aabb_heldout_correctness_v1.py"
EXTRA_AUDIT = ROOT / "tools/audit_static_aabb_heldout_correctness_source_v1.py"
BINARY = ROOT / "bin/static_aabb_topk_heldout_v1"
EXACT_IDS = ROOT / "inputs/standard_sift_static_aabb_heldout_exact64_v1.ids"
FULL_IDS = ROOT / "inputs/standard_sift_static_aabb_heldout256_v1.ids"
EXPECTED_BINARY_SHA = "d10ec9a319927fbd35be27ad55d3cd8cc2cc75e4bd0aa257382e11fc0824a3de"


def sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid regular file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing pre-existing output: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    for path in (BASE_PROTOCOL, SOURCEBOUND, BUILD_CARD, PARENT_AUDIT, EXTRA_AUDIT, BINARY, EXACT_IDS, FULL_IDS):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"invalid required artifact: {path}")
    if OUT.exists() or OUT.is_symlink():
        raise RuntimeError("build-bound execution contract already exists")

    base = json.loads(BASE_PROTOCOL.read_text())
    sourcebound = json.loads(SOURCEBOUND.read_text())
    if base.get("status") != "FROZEN_CPU_ONLY_PREPARED" or base.get("formal_claim_eligible") is not False:
        raise RuntimeError("base protocol")
    if sourcebound.get("status") != "FROZEN_CPU_ONLY_SOURCE_BOUND_NOT_BUILT":
        raise RuntimeError("source-bound predecessor status")
    if sourcebound["inherits_heldout_protocol"]["sha256"] != sha256(BASE_PROTOCOL):
        raise RuntimeError("source-bound predecessor base binding")
    if sourcebound["build_contract"]["planned_binary"]["path"] != str(BINARY):
        raise RuntimeError("planned binary path")

    card = json.loads(BUILD_CARD.read_text())
    if card.get("schema") != "safe-c2-static-aabb-heldout-correctness-build-v2":
        raise RuntimeError("build-card schema")
    if card.get("status") != "COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION":
        raise RuntimeError("build-card status")
    if card.get("gpu_binary_executed") is not False or card.get("nvidia_smi_called") is not False:
        raise RuntimeError("build-card execution claim")
    if card["binary"]["path"] != str(BINARY) or card["binary"]["sha256"] != sha256(BINARY):
        raise RuntimeError("build-card binary binding")
    if card["binary"]["sha256"] != EXPECTED_BINARY_SHA:
        raise RuntimeError("unexpected binary SHA")
    for key, path_key in (("source", "source"), ("bound_header", "bound_header"), ("search_header", "search_header"), ("sift_integer_contract", "sift_integer_contract")):
        entry = card[key]
        if entry["sha256"] != sha256(Path(entry["path"])):
            raise RuntimeError(f"build-card artifact mismatch: {key}")
    required_flags = ["-std=c++17", "-O3", "--ftz=false", "-arch=sm_120", "-Xcompiler=-fopenmp"]
    if card.get("compile_flags") != required_flags:
        raise RuntimeError("unexpected v2 compile flags")

    parent_audit = json.loads(subprocess.check_output([str(PARENT_AUDIT)], text=True))
    extra_audit = json.loads(subprocess.check_output([str(EXTRA_AUDIT)], text=True))
    if parent_audit.get("status") != "PASS" or extra_audit.get("status") != "PASS_CPU_ONLY_SOURCE_AUDIT":
        raise RuntimeError("source audit failed")
    if card["source_audit"]["artifact"]["sha256"] != sha256(PARENT_AUDIT):
        raise RuntimeError("build-card parent audit tool")
    if card["source_audit"]["result"] != parent_audit:
        raise RuntimeError("build-card parent audit result")
    exact = [int(line) for line in EXACT_IDS.read_text().splitlines()]
    full = [int(line) for line in FULL_IDS.read_text().splitlines()]
    if len(exact) != 64 or len(full) != 256 or exact != full[:64]:
        raise RuntimeError("frozen split")

    contract = {
        "schema": "safe-c2-static-aabb-heldout-correctness-execution-buildbound-v1",
        "status": "BUILD_BOUND_CPU_ONLY_EXECUTION_NOT_AUTHORIZED",
        "scope": (
            "binary-bound successor contract for the 64-query static-AABB held-out correctness checker; "
            "compile provenance only, not a GPU correctness result or timing experiment"
        ),
        "predecessors": {
            "base_heldout_protocol": {"path": str(BASE_PROTOCOL), "sha256": sha256(BASE_PROTOCOL)},
            "source_bound_execution_contract": {"path": str(SOURCEBOUND), "sha256": sha256(SOURCEBOUND)},
        },
        "fixed_input_split": {
            "exact64": {"path": str(EXACT_IDS), "sha256": sha256(EXACT_IDS), "ids": exact, "count": 64},
            "full256_for_separate_future_perf_only": {
                "path": str(FULL_IDS), "sha256": sha256(FULL_IDS), "count": 256,
                "GT_role": "fixed standard label only; not calibration; no performance runner is bound here",
            },
            "immutable": True,
        },
        "build_binding": {
            "build_card": {"path": str(BUILD_CARD), "sha256": sha256(BUILD_CARD), "record": card},
            "binary": {"path": str(BINARY), "sha256": sha256(BINARY), "bytes": BINARY.stat().st_size},
            "source_audits": {
                "build_card_audit": {"path": str(PARENT_AUDIT), "sha256": sha256(PARENT_AUDIT), "result": parent_audit},
                "additional_protocol_audit": {"path": str(EXTRA_AUDIT), "sha256": sha256(EXTRA_AUDIT), "result": extra_audit},
            },
            "m": 128,
            "projection_policy": "all raw coordinates, no query/GT calibration",
        },
        "compile_provenance_delta": {
            "source_bound_prebuild_template": sourcebound["build_contract"]["required_command_argv"],
            "actual_build_v2_flags": card["compile_flags"],
            "status": "RECORDED_TEXTUAL_ARCH_FLAG_DELTA",
            "explanation": (
                "The prebuild template used --generate-code=arch=compute_120,code=[compute_120,sm_120]; "
                "the independently recorded v2 build used native -arch=sm_120. Both target SM120 and "
                "retain C++17/O3/FTZ-disabled/OpenMP requirements, but the textual difference is preserved "
                "rather than silently treated as identical."
            ),
        },
        "execution_gate": {
            "status": "BLOCKED_UNTIL_SEPARATE_RUNNER_AUDITED",
            "required_runner_artifacts": ["preflight.json", "gpu_guard.json", "terminal.json", "manifest.json"],
            "terminal_pass_status": "PASS_STATIC_AABB_HELDOUT_CORRECTNESS_V1",
            "terminal_pass_conditions": [
                "baseline and static-AABB invalid/duplicate/oracle-set/distance mismatches all equal zero",
                "baseline/static-AABB ID sets equal",
                "host AABB cover violations equal zero",
                "static snapshot is byte-identical before/after",
            ],
            "shared_GPU_rule": "refuse nonempty compute-process guard; do not kill or disturb others",
        },
        "nonclaims": [
            "No GPU binary execution was performed by this build or binding.",
            "No held-out result or formal paper evidence exists until the separate runner has an auditable guard and terminal manifest.",
            "This does not cover updates/C3 direct insertion or general metric spaces.",
            "The 256-query GT-gated performance runner remains separate and unimplemented.",
        ],
        "formal_claim_eligible": False,
    }
    atomic_write(OUT, json.dumps(contract, sort_keys=True, indent=2) + "\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_STATIC_AABB_HELDOUT_CORRECTNESS_BUILDBOUND_FREEZE_V1: {exc}", file=sys.stderr)
        raise SystemExit(2)
