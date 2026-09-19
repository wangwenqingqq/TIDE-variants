#!/usr/bin/env python3
"""Fail-closed CPU-only audit for the Safe-C2 v4 source-only successor."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
INVENTORY = ROOT / "provenance/source_only_inventory_v1.json"
SOURCE = ROOT / "src/gts_speculative_fallback_v4_sift1m.cu"
SELECTION = ROOT / "inputs/fresh_selection_pre_gt_v1"
EXPECTED_FORBIDDEN_DIRS = ("bin", "runs", ".locks", "preledger_receipts")
EXPECTED_TOP = {
    "schema", "status", "selection_pre_gt_path", "selection_pre_gt_sha256",
    "v3_forensic_record_path", "v3_forensic_record_sha256", "formal_protocol_path",
    "formal_protocol_sha256", "sealed_v2_test_forbidden", "v2_sealed_test_ids_sha256_forbidden",
    "base_fvecs_path", "base_fvecs_sha256", "query_fvecs_path", "query_fvecs_sha256",
    "query_fvecs_count", "groundtruth_ivecs_path", "groundtruth_ivecs_sha256",
    "groundtruth_width", "mapping_path", "mapping_sha256", "calibration_ids_path",
    "calibration_ids_sha256", "calibration_query_count", "validation_ids_path",
    "validation_ids_sha256", "validation_query_count", "sealed_test_ids_path",
    "sealed_test_ids_sha256", "sealed_test_query_count",
}


class AuditError(RuntimeError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def direct_file(path: Path, label: str) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise AuditError(f"{label} is not a direct canonical file: {path}")


def audit_inventory() -> int:
    direct_file(INVENTORY, "source-only inventory")
    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    if data.get("schema") != "safe-c2-v4-source-only-inventory-v1":
        raise AuditError("inventory schema mismatch")
    if data.get("status") != "SOURCE_ONLY_PRE_GT_NO_GPU_NO_BINARY_NO_LEDGER":
        raise AuditError("inventory status mismatch")
    entries = data.get("entries")
    if not isinstance(entries, list) or len(entries) < 12:
        raise AuditError("inventory entry count too small")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise AuditError("non-object inventory entry")
        path = Path(entry.get("path", ""))
        if str(path) in seen:
            raise AuditError(f"duplicate inventory path {path}")
        seen.add(str(path))
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise AuditError(f"inventory escapes v4 root: {path}") from exc
        direct_file(path, f"inventory entry {path.name}")
        if entry.get("sha256") != sha(path) or entry.get("bytes") != path.stat().st_size:
            raise AuditError(f"inventory mismatch: {path}")
    return len(entries)


def audit_cpp_source() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    required = (
        "class StrictJsonParser",
        "duplicate JSON object key",
        "require_top_field",
        "reject_unknown_or_missing_top_keys",
        "--validate-workload-only",
        "write_workload_validation_receipt",
        "O_EXCL",
        "normal_execution_phase_entered",
        "if (normal_execution_phase_entered)",
        "seen_ids.insert(value)",
    )
    for token in required:
        if token not in text:
            raise AuditError(f"missing C++ source token: {token}")
    if ".find(" in text or "manifest_string_field" in text or "manifest_true_field" in text:
        raise AuditError("legacy substring workload parser remains in v4 source")
    main_start = text.index("int safe_c2_main(int argc, char** argv)")
    ordered = (
        text.index("const WorkloadBinding workload = validate_workload_binding(args);", main_start),
        text.index("if (args.validate_workload_only)", main_start),
        text.index("write_workload_validation_receipt(args, workload);", main_start),
        text.index("normal_execution_phase_entered = true;", main_start),
        text.index("const Fvecs base = read_fvecs", main_start),
    )
    if tuple(sorted(ordered)) != ordered:
        raise AuditError("validation-only parser/hash path is not ordered before normal dataset/CUDA phase")
    # Failures before the normal phase must not call cleanup CUDA APIs.
    catch = text[text.index("} catch (const std::exception& error)"):text.index("\n}\n\n}  // namespace safe_c2", text.index("} catch (const std::exception& error)"))]
    if "if (normal_execution_phase_entered)" not in catch:
        raise AuditError("validation-only catch path is not CUDA-cleanup gated")
    if "constexpr int kQueryN = 12'524;" not in text:
        raise AuditError("v4 source/query selection cardinality mismatch")


def audit_selection() -> dict[str, object]:
    manifest = SELECTION / "selection_manifest_pre_gt.json"
    artifacts = SELECTION / "selection_artifacts.sha256"
    direct_file(manifest, "pre-GT selection manifest")
    direct_file(artifacts, "pre-GT selection artifact hash list")
    listed: dict[str, str] = {}
    for line in artifacts.read_text(encoding="ascii").splitlines():
        parts = line.split("  ")
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise AuditError("malformed selection artifact hash row")
        listed[parts[1]] = parts[0]
    actual_names = {p.name for p in SELECTION.iterdir() if p.is_file() and p.name != "selection_artifacts.sha256"}
    if set(listed) != actual_names:
        raise AuditError("selection artifact list differs from actual files")
    for name, digest in listed.items():
        if sha(SELECTION / name) != digest:
            raise AuditError(f"selection artifact hash mismatch: {name}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("status") != "FRESH_IDS_FROZEN_PRE_GT_NO_C2_NO_GPU":
        raise AuditError("selection is not explicitly pre-GT/source-only")
    if data.get("derived", {}).get("compact_query_fvecs_count") != 12524:
        raise AuditError("selection count mismatch")
    counts = data.get("filtering_counts", {})
    expected_counts = {
        "input_rows": 100000, "reserved_drift_prefix_reject": 3072,
        "v2_payload_reject": 9670, "v3_original_reject": 10000,
        "v3_payload_alias_reject": 21, "global_duplicate_non_v3_reject": 174,
        "learn_global_unique": 99766, "fresh_clean_candidates": 77063,
    }
    if counts != expected_counts:
        raise AuditError(f"selection quarantine counts mismatch: {counts}")
    mappings = (SELECTION / "local_to_original_learn_id.tsv").read_text(encoding="ascii").splitlines()
    if len(mappings) != 12524:
        raise AuditError("fresh mapping cardinality mismatch")
    v3_map = Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v3/inputs/sift_learn_compact10k_v1/local_to_original_learn_id.tsv")
    v3_rows = v3_map.read_text(encoding="ascii").splitlines()
    v3_originals = {int(line.split("\t")[1]) for line in v3_rows}
    v3_payloads = {line.split("\t")[2] for line in v3_rows}
    fresh_originals: set[int] = set()
    fresh_payloads: set[str] = set()
    for local, line in enumerate(mappings):
        parts = line.split("\t")
        if len(parts) != 3 or parts[0] != str(local):
            raise AuditError("fresh mapping malformed/order mismatch")
        oid, payload = int(parts[1]), parts[2]
        if oid in fresh_originals or payload in fresh_payloads:
            raise AuditError("fresh mapping duplicate")
        fresh_originals.add(oid); fresh_payloads.add(payload)
    if fresh_originals & v3_originals or fresh_payloads & v3_payloads:
        raise AuditError("v3 compact leakage into v4 fresh selection")
    stage_seen: set[int] = set()
    for stage, target in (("qualification_canary", 1024), ("calibration", 2500), ("validation", 2500), ("sealed_test", 6500)):
        lines = (SELECTION / f"{stage}.candidate_local_ids.txt").read_text(encoding="ascii").splitlines()
        if len(lines) != target:
            raise AuditError(f"candidate stage count mismatch: {stage}")
        ids = {int(x) for x in lines}
        if len(ids) != target or stage_seen & ids:
            raise AuditError(f"candidate stage non-disjoint: {stage}")
        stage_seen |= ids
    if stage_seen != set(range(12524)):
        raise AuditError("candidate stages do not exactly partition fresh selection")
    return {"selection_manifest_sha256": sha(manifest), "selection_count": len(mappings)}


def audit_protocols() -> None:
    workflow = json.loads((ROOT / "protocols/c2_v4_formal_workflow_pre_gt_v1.json").read_text(encoding="utf-8"))
    top = set(workflow.get("future_final_workload_manifest_contract", {}).get("exact_top_level_fields", []))
    if top != EXPECTED_TOP:
        raise AuditError("formal workflow exact top-level field contract drift")
    if workflow.get("status") != "SOURCE_ONLY_PRE_GT_SELECTION_FROZEN_NO_GT_NO_BINARY_NO_LEDGER":
        raise AuditError("formal workflow status drift")
    preledger = json.loads((ROOT / "protocols/c2_v4_preledger_parser_compatibility_gate_v1.json").read_text(encoding="utf-8"))
    if preledger.get("status") != "DESIGNED_SOURCE_ONLY_NO_FINAL_BINARY_NO_LEDGER":
        raise AuditError("preledger status drift")
    canary = json.loads((ROOT / "protocols/c2_v4_independent_canary_protocol_v1.json").read_text(encoding="utf-8"))
    if canary.get("status") != "DESIGNED_SOURCE_ONLY_NO_CANARY_EXECUTION":
        raise AuditError("canary status drift")


def audit_no_execution_state() -> None:
    for name in EXPECTED_FORBIDDEN_DIRS:
        if (ROOT / name).exists() or (ROOT / name).is_symlink():
            raise AuditError(f"forbidden execution directory exists: {name}")
    forbidden_names = []
    for path in ROOT.rglob("*"):
        if path.is_dir() and path.name == "__pycache__":
            forbidden_names.append(str(path))
        if path.is_file() and (path.suffix in {".ivecs", ".nsys-rep"} or path.name.endswith(".ledger.json") or path.name.startswith("GTS_safe_c2")):
            forbidden_names.append(str(path))
    if forbidden_names:
        raise AuditError(f"unexpected executable/GT/cache artifact(s): {forbidden_names}")
    guard = ROOT / "tools/c2_v4_guard_failure_integrity_source_only.sh"
    direct_file(guard, "source-only guard cleanup contract")
    text = guard.read_text(encoding="utf-8")
    if '--execute) echo \'REFUSED:' not in text or "nvidia-smi" in text:
        raise AuditError("source-only guard execution refusal/absence contract broken")


def main() -> int:
    inventory_count = audit_inventory()
    audit_cpp_source()
    selection = audit_selection()
    audit_protocols()
    audit_no_execution_state()
    print(json.dumps({
        "status": "PASS_C2_V4_SOURCE_ONLY_AUDIT",
        "inventory_entries": inventory_count,
        **selection,
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
        "formal_claim_eligible": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VERIFY_C2_V4_SOURCE_ONLY_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
