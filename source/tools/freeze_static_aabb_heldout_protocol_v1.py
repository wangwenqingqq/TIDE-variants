#!/usr/bin/env python3
"""Freeze the static-AABB development-held-out protocol after CPU-only selection."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
SELECTION = ROOT / "provenance/static_aabb_heldout_selection_v1.json"
IDS = ROOT / "inputs/standard_sift_static_aabb_heldout256_v1.ids"
EXACT_IDS = ROOT / "inputs/standard_sift_static_aabb_heldout_exact64_v1.ids"
OUT = ROOT / "provenance/static_aabb_heldout_protocol_v1.json"
CONTRACT = ROOT / "provenance/sift_integer_disk_contract_v1.json"
SOURCE_AUDIT = ROOT / "tools/audit_diskguard_v2_1_sources.py"
RUNNER = ROOT / "tools/run_diskguard_v2_1_m128.sh"
ARTIFACTS = {
    "bound_header": ROOT / "include/aabb_static_bound_v2_diskguard.cuh",
    "search_header": ROOT / "include/search_static_aabb_v2_diskguard.cuh",
    "topk_source": ROOT / "src/static_aabb_topk_canary_v2_1_diskguard.cu",
    "perf_source": ROOT / "src/static_aabb_perf_probe_v2_1_diskguard.cu",
    "topk_binary": ROOT / "bin/static_aabb_topk_canary_v2_1_diskguard",
    "perf_binary": ROOT / "bin/static_aabb_perf_probe_v2_1_diskguard",
}
EXPECTED = {
    "bound_header": "1747b53494ab9f9ed6632a528e9a7da17edba2b2f2846c9192b5c4347eb69626",
    "search_header": "3ed53b958b2611c6419376c69741f27fd7f6763b9d41bf1fb96daba98341da7f",
    "topk_source": "42398aa18e34091dcc96c05cc4c7edea23c85ae6dd3dc830da22877186729af3",
    "perf_source": "5dc92a9660bc168e1325edae7a1c9d41f59a93e036e83f57f2ccd9d55fa6819f",
    "topk_binary": "d33e0fb3ba7757195d376bc6a7a5b579d9578bf0e8a6928cfa2670ae640bcf19",
    "perf_binary": "e587ad822de007880e1caae8ea89072c07a5b786dcb5d15357ecde61ce577cde",
}


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
        raise RuntimeError(f"refusing pre-existing protocol: {path}")
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
    if OUT.exists() or OUT.is_symlink():
        raise RuntimeError("protocol already exists; refusing overwrite")
    for path in (SELECTION, IDS, EXACT_IDS, CONTRACT, SOURCE_AUDIT, RUNNER, *ARTIFACTS.values()):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or symlink input: {path}")

    selection = json.loads(SELECTION.read_text())
    if selection.get("schema") != "safe-c2-static-aabb-standard-sift-heldout-selection-v1":
        raise RuntimeError("unexpected selection schema")
    if selection.get("status") != "FROZEN_CPU_ONLY":
        raise RuntimeError("selection not frozen")
    selected = selection.get("selected_ids")
    if not isinstance(selected, list) or len(selected) != 256 or selected != sorted(selected) or len(set(selected)) != 256:
        raise RuntimeError("invalid selected IDs")
    if any(qid < 5000 or qid >= 10000 or 0 <= qid <= 31 or 1000 <= qid <= 1259 for qid in selected):
        raise RuntimeError("selected ID violates exclusion")
    if selection["output_ids"]["sha256"] != sha256(IDS):
        raise RuntimeError("selection/IDS hash mismatch")
    exact = selection.get("gpu_heldout_correctness_subset", {})
    exact_ids = exact.get("selected_ids")
    if (not isinstance(exact_ids, list) or len(exact_ids) != 64 or exact_ids != selected[:64] or
            exact.get("ids_sha256") != sha256(EXACT_IDS)):
        raise RuntimeError("exact-oracle heldout subset mismatch")

    contract = json.loads(CONTRACT.read_text())
    if contract.get("status") != "PASS_CPU_ONLY":
        raise RuntimeError("disk contract is not CPU-pass")
    audit_result = json.loads(subprocess.check_output([str(SOURCE_AUDIT)], text=True))
    if audit_result.get("status") != "PASS":
        raise RuntimeError("v2.1 source audit failed")
    frozen = {}
    for name, path in ARTIFACTS.items():
        digest = sha256(path)
        if digest != EXPECTED[name]:
            raise RuntimeError(f"unexpected frozen {name} SHA: {digest}")
        frozen[name] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}

    runner_text = RUNNER.read_text()
    if "EXPLORATORY" not in runner_text or "formal_claim_eligible" not in runner_text:
        raise RuntimeError("expected v2.1 exploratory runner markers absent")
    protocol = {
        "schema": "safe-c2-static-aabb-heldout-protocol-v1",
        "status": "FROZEN_CPU_ONLY_PREPARED",
        "scope": (
            "pre-registered development-held-out evaluation protocol for static raw-float32 L2 "
            "coordinate-envelope pruning on standard SIFT1M only"
        ),
        "hard_boundaries": {
            "static_only": True,
            "updates_or_insertions_in_scope": False,
            "old_c2_validation_or_sealed_accessed": False,
            "gpu_executed_for_this_protocol": False,
            "nvidia_smi_called_for_this_protocol": False,
            "does_not_authorize_gpu_execution": True,
        },
        "dataset_contract": {
            "sift_integer_disk_contract": {
                "path": str(CONTRACT),
                "sha256": sha256(CONTRACT),
                "status": contract["status"],
                "scope": contract["scope"],
            },
            "input_hashes": selection["inputs"],
        },
        "algorithm_freeze": {
            "variant": "v2.1 static AABB with SIFT-specific one-ULP upward disk guard",
            "projection_dimensions": 128,
            "projection_policy": (
                "all 128 raw coordinates; no projection-width tuning, learned calibration, "
                "query-dependent calibration, drift detector, or re-calibration"
            ),
            "source_and_binary_artifacts": frozen,
            "source_audit": {
                "path": str(SOURCE_AUDIT),
                "sha256": sha256(SOURCE_AUDIT),
                "result": audit_result,
            },
        },
        "heldout_input": {
            "selection_metadata": {"path": str(SELECTION), "sha256": sha256(SELECTION)},
            "ids_file": {"path": str(IDS), "sha256": sha256(IDS), "bytes": IDS.stat().st_size},
            "query_count": 256,
            "selected_ids": selected,
            "selection_policy": selection["selection_policy"],
            "gpu_heldout_correctness": {
                "ids_file": {"path": str(EXACT_IDS), "sha256": sha256(EXACT_IDS), "bytes": EXACT_IDS.stat().st_size},
                "query_count": 64,
                "selected_ids": exact_ids,
                "relation_to_heldout256": "ordered prefix of the same frozen heldout256 list",
                "oracle": "independent CPU exact top-10 over all base vectors; GT must not be used as the oracle",
            },
            "gt_gated_heldout_performance": {
                "ids_file": {"path": str(IDS), "sha256": sha256(IDS), "bytes": IDS.stat().st_size},
                "query_count": 256,
                "GT_role": (
                    "fixed standard SIFT top-10 label for a pre-timing correctness gate only; "
                    "not a projection-width, coordinate, disk-guard, or timing calibration signal"
                ),
            },
        },
        "predeclared_evaluation": {
            "gpu_heldout_correctness": [
                "use the fixed exact64 ordered prefix of the frozen heldout256 list",
                "use a fresh successor runner with an independent CPU exact top-10 oracle over all 1,000,000 base vectors; do not use GT as the oracle",
                "require zero invalid IDs, duplicates, exact-oracle set mismatches, and distance mismatches for both radial baseline and static AABB",
                "require baseline/static-AABB ID-set equality and byte-identical static snapshots before/after",
                "require host AABB cover violations equal zero",
            ],
            "gt_gated_heldout_performance": [
                "use all fixed heldout256 IDs only after the independent exact64 correctness gate passes",
                "GT may be read only as the fixed standard top-10 label for the pre-timing gate; it must not tune any parameter or select timing repetitions",
                "paired alternating baseline/AABB trials with fixed warmups/repetitions declared before execution",
                "record GPU clock, power, memory, process guard, CUDA/driver, and per-repetition paired timings",
                "do not report a formal speed claim if clock/isolation telemetry is inadequate",
            ],
            "decision_rule": (
                "no source, parameter, query-ID, or metric change after first held-out execution; "
                "a system failure can be rerun only with a new run manifest explaining the failure"
            ),
        },
        "current_runner_disposition": {
            "path": str(RUNNER),
            "sha256": sha256(RUNNER),
            "reuse_for_heldout_formal_execution": False,
            "reason": (
                "the existing v2.1 runner and perf result schema explicitly label outputs EXPLORATORY "
                "and formal_claim_eligible=false; a separately frozen successor runner with auditable "
                "preflight/guard/terminal/manifest status is required"
            ),
        },
        "known_limitations": [
            "This is development-held-out, not a blinded or externally sealed test.",
            "The exact CPU oracle is intentionally limited to the declared 64-query prefix; it is a correctness canary, not exhaustive proof over all 10,000 queries.",
            "Tie-free selection evaluates exact top-10 behavior only where the public GT rank-10/rank-11 boundary is strict; it is not an all-query recall estimate.",
            "The numerical disk guard is justified only by the recorded SIFT integer contract, not arbitrary float vectors or general metric spaces.",
            "Static AABB evidence does not establish correctness under C3 direct insertion, updates, ancestor interval changes, or AABB maintenance.",
            "No GPU execution, performance result, or formal paper claim is created by this protocol freeze.",
        ],
        "formal_claim_eligible": False,
    }
    atomic_write(OUT, json.dumps(protocol, sort_keys=True, indent=2) + "\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_STATIC_AABB_HELDOUT_PROTOCOL_V1: {exc}", file=sys.stderr)
        raise SystemExit(2)
