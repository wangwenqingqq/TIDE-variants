#!/usr/bin/env python3
"""Fail-closed finalizer for Safe-C2 v4 CPU oracle / deterministic tie admission.

This program reads only the frozen pre-GT selection, independent CPU-oracle
artifacts, and v2/v3 quarantine witnesses. It never imports CUDA, invokes any
binary, calls nvidia-smi, reads a C2 result/gamma/timing/trace artifact, or
creates a ledger/run directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
SELECTION_DIR = ROOT / "inputs/fresh_selection_pre_gt_v1"
SELECTION = SELECTION_DIR / "selection_manifest_pre_gt.json"
V3_MAPPING = Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v3/inputs/sift_learn_compact10k_v1/local_to_original_learn_id.tsv")
V2_QUERY = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs")
BASE = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs")
FRESH_QUERY = SELECTION_DIR / "sift_learn_fresh12524.fvecs"
MAPPING = SELECTION_DIR / "local_to_original_learn_id.tsv"
V3_FORENSIC = ROOT / "provenance/v3_execution_forensic_boundary_v1.json"
FINAL_DIR = ROOT / "inputs/final_workload_v2"
V1_FINALIZER_FAILURE = Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/provenance/c2_v4_oracle_finalizer_v1_failure_20260728T025823Z.json")
V1_FINALIZER_FAILURE_SHA = "dab313bf7f6de16176b2cba9dec3027e08f13ca8b81e07f752379997c4dad055"
PRELEDGER_V1_FAILURE = Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/provenance/c2_v4_preledger_gate_v1_failure_manifest_tmp_path.json")
PRELEDGER_V1_FAILURE_SHA = "2c9cc4f85f92c68b65631ec5800560475559edccdc5d9ed0625e8f00de99340f"

DIM = 128
N_BASE = 1_000_000
N_QUERY = 12_524
TOP = 101
WIDTH = 100
RECORD_BYTES = 4 + DIM * 4
AUDIT_MAGIC = 0x31564F3443545347
AUDIT_VERSION = 1
V2_SEALED_FORBIDDEN = "50ccb28263bf23e499b9c50e4d3e9800fca3949aa5b5ea8a454237d5987701ce"
SELECTION_SHA = "98e3627c7dba033f5ab73b9df90b794789c9e39046491e5cc1745b6ac87b7c39"
BASE_SHA = "21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816"
FRESH_QUERY_SHA = "6b8e480bcd7247e13d1a8b4248d2e86d619abea6919bc9d27fced93e6c3fe3c2"

STAGES = (
    ("qualification_canary", 1024, 980),
    ("calibration", 2500, 2400),
    ("validation", 2500, 2400),
    ("sealed_test", 6500, 6200),
)
WORKLOAD_FIELDS = {
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


class FinalizeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def direct_file(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise FinalizeError(f"{label} is not a direct canonical regular file: {path}")
    return path


def direct_dir(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise FinalizeError(f"{label} is not a direct canonical directory: {path}")
    return path


def require_sha(path: Path, expected: str, label: str) -> str:
    direct_file(path, label)
    observed = sha256_file(path)
    if observed != expected:
        raise FinalizeError(f"{label} SHA-256 mismatch: expected {expected}, got {observed}")
    return observed


def load_json(path: Path, label: str) -> dict[str, Any]:
    direct_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FinalizeError(f"{label} invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizeError(f"{label} root must be an object")
    return value


def canonical_id_lines(path: Path, expected_count: int, label: str) -> list[int]:
    direct_file(path, label)
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise FinalizeError(f"{label} is not ASCII") from exc
    if len(lines) != expected_count:
        raise FinalizeError(f"{label} count mismatch {len(lines)} != {expected_count}")
    result: list[int] = []
    seen: set[int] = set()
    for row, text in enumerate(lines):
        if not text or not text.isdecimal() or (len(text) > 1 and text[0] == "0"):
            raise FinalizeError(f"{label} noncanonical ID at row {row}")
        value = int(text)
        if not 0 <= value < N_QUERY or value in seen:
            raise FinalizeError(f"{label} duplicate/out-of-range ID at row {row}")
        seen.add(value)
        result.append(value)
    return result


def parse_mapping(path: Path, expected_rows: int, label: str) -> dict[int, tuple[int, str]]:
    direct_file(path, label)
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) != expected_rows:
        raise FinalizeError(f"{label} row count mismatch")
    out: dict[int, tuple[int, str]] = {}
    originals: set[int] = set()
    payloads: set[str] = set()
    for expected_local, line in enumerate(lines):
        fields = line.split("\t")
        if len(fields) != 3:
            raise FinalizeError(f"{label} malformed row {expected_local}")
        local_text, original_text, digest = fields
        if local_text != str(expected_local) or not original_text.isdecimal():
            raise FinalizeError(f"{label} noncanonical row {expected_local}")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise FinalizeError(f"{label} bad payload hash at row {expected_local}")
        original = int(original_text)
        if original in originals or digest in payloads:
            raise FinalizeError(f"{label} duplicate original/payload at row {expected_local}")
        out[expected_local] = (original, digest)
        originals.add(original)
        payloads.add(digest)
    return out


def payload_hashes(path: Path, expected_rows: int, label: str) -> set[str]:
    direct_file(path, label)
    expected_bytes = expected_rows * RECORD_BYTES
    if path.stat().st_size != expected_bytes:
        raise FinalizeError(f"{label} record-size mismatch")
    result: set[str] = set()
    with path.open("rb") as handle:
        for row in range(expected_rows):
            record = handle.read(RECORD_BYTES)
            if len(record) != RECORD_BYTES:
                raise FinalizeError(f"{label} truncated row {row}")
            if struct.unpack("<i", record[:4])[0] != DIM:
                raise FinalizeError(f"{label} dimension mismatch row {row}")
            result.add(hashlib.sha256(record[4:]).hexdigest())
        if handle.read(1):
            raise FinalizeError(f"{label} trailing byte")
    return result


def stage_candidate_members(selection: dict[str, Any], mapping: dict[int, tuple[int, str]]) -> dict[str, list[int]]:
    metadata = selection.get("candidate_splits")
    if not isinstance(metadata, dict) or set(metadata) != {name for name, _, _ in STAGES}:
        raise FinalizeError("candidate split metadata mismatch")
    all_local: set[int] = set()
    output: dict[str, list[int]] = {}
    for name, expected_count, _minimum in STAGES:
        item = metadata[name]
        if not isinstance(item, dict) or item.get("candidate_count") != expected_count:
            raise FinalizeError(f"candidate count metadata mismatch: {name}")
        ids_path = Path(item.get("candidate_local_ids_path", ""))
        map_path = Path(item.get("candidate_mapping_path", ""))
        original_path = Path(item.get("candidate_original_ids_path", ""))
        require_sha(ids_path, item.get("candidate_local_ids_sha256"), f"{name} candidate IDs")
        require_sha(map_path, item.get("candidate_mapping_sha256"), f"{name} candidate mapping")
        require_sha(original_path, item.get("candidate_original_ids_sha256"), f"{name} candidate originals")
        ids = canonical_id_lines(ids_path, expected_count, f"{name} candidate IDs")
        if all_local.intersection(ids):
            raise FinalizeError(f"candidate split local overlap: {name}")
        all_local.update(ids)

        map_lines = map_path.read_text(encoding="ascii").splitlines()
        original_lines = original_path.read_text(encoding="ascii").splitlines()
        if len(map_lines) != expected_count or len(original_lines) != expected_count:
            raise FinalizeError(f"candidate mapping/original count mismatch: {name}")
        for offset, local in enumerate(ids):
            fields = map_lines[offset].split("\t")
            if len(fields) != 3 or fields[0] != str(local):
                raise FinalizeError(f"candidate mapping malformed: {name}:{offset}")
            original, digest = mapping[local]
            if fields[1] != str(original) or fields[2] != digest or original_lines[offset] != str(original):
                raise FinalizeError(f"candidate mapping does not bind global mapping: {name}:{offset}")
        output[name] = ids
    if all_local != set(range(N_QUERY)):
        raise FinalizeError("candidate splits do not exactly partition fresh query IDs")
    return output


def float_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def parse_oracle(oracle_dir: Path) -> tuple[list[bool], dict[str, int], dict[str, str]]:
    direct_dir(oracle_dir, "oracle directory")
    gt_path = oracle_dir / "groundtruth_fp32_top100.ivecs"
    audit_path = oracle_dir / "top101_int64_fp32_audit.bin"
    admission_path = oracle_dir / "tie_admission.tsv"
    summary_path = oracle_dir / "oracle_summary.json"
    for path, label in ((gt_path, "oracle groundtruth"), (audit_path, "oracle audit"),
                        (admission_path, "oracle admission"), (summary_path, "oracle summary")):
        direct_file(path, label)

    summary = load_json(summary_path, "oracle summary")
    expected_summary = {
        "schema": "safe-c2-v4-cpu-exact-fp32-int64-oracle-v1",
        "status": "COMPLETE_CPU_ONLY_ORACLE_UNADMITTED",
        "dimension": DIM, "base_count": N_BASE, "query_count": N_QUERY,
        "top_width": TOP, "groundtruth_width": WIDTH,
        "gpu_binary_executed": False, "nvidia_smi_called": False,
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise FinalizeError(f"oracle summary field mismatch: {key}")

    expected_gt_bytes = N_QUERY * (4 + WIDTH * 4)
    if gt_path.stat().st_size != expected_gt_bytes:
        raise FinalizeError("oracle GT size mismatch")
    gt_bytes = gt_path.read_bytes()
    admission_lines = admission_path.read_text(encoding="ascii").splitlines()
    header = ("local_id\texact_rank10_d2\texact_rank11_d2\tfp32_rank10_bits\tfp32_rank11_bits\t"
              "exact_boundary_tie\tfp32_boundary_tie\ttop10_idset_agree\t"
              "fp32_top101_collision_different_d2\teligible")
    if len(admission_lines) != N_QUERY + 1 or admission_lines[0] != header:
        raise FinalizeError("oracle admission header/row count mismatch")

    tie_rows: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []
    for qid, line in enumerate(admission_lines[1:]):
        fields = line.split("\t")
        if len(fields) != 10:
            raise FinalizeError(f"oracle admission malformed row {qid}")
        try:
            values = tuple(int(value) for value in fields)
        except ValueError as exc:
            raise FinalizeError(f"oracle admission noninteger row {qid}") from exc
        if values[0] != qid or any(flag not in (0, 1) for flag in values[5:]):
            raise FinalizeError(f"oracle admission canonical/flag mismatch row {qid}")
        tie_rows.append(values)

    expected_audit_bytes = 8 + 6 * 4 + N_QUERY * (4 + TOP * 32)
    if audit_path.stat().st_size != expected_audit_bytes:
        raise FinalizeError("oracle audit binary size mismatch")
    raw = audit_path.read_bytes()
    offset = 0
    magic, = struct.unpack_from("<Q", raw, offset); offset += 8
    version, dim, base_count, query_count, top, width = struct.unpack_from("<iiiiii", raw, offset); offset += 24
    if (magic, version, dim, base_count, query_count, top, width) != (
            AUDIT_MAGIC, AUDIT_VERSION, DIM, N_BASE, N_QUERY, TOP, WIDTH):
        raise FinalizeError("oracle audit header mismatch")

    eligible: list[bool] = []
    stats = {
        "eligible_total": 0,
        "exact_boundary_ties": 0,
        "fp32_boundary_ties": 0,
        "top10_idset_mismatches": 0,
        "fp32_top101_collisions_different_d2": 0,
    }
    for qid in range(N_QUERY):
        audit_qid, = struct.unpack_from("<i", raw, offset); offset += 4
        if audit_qid != qid:
            raise FinalizeError(f"oracle audit local ID mismatch at {qid}")
        exact: list[tuple[int, int, int, float]] = []
        fp: list[tuple[int, int, int, float]] = []
        for _rank in range(TOP):
            exact_id, exact_d2, exact_bits, fp_id, fp_d2, fp_bits = struct.unpack_from("<iqIiqI", raw, offset)
            offset += 32
            exact_value = float_from_bits(exact_bits)
            fp_value = float_from_bits(fp_bits)
            if (not 0 <= exact_id < N_BASE or not 0 <= fp_id < N_BASE or exact_d2 < 0 or fp_d2 < 0 or
                    not math.isfinite(exact_value) or not math.isfinite(fp_value) or exact_value < 0 or fp_value < 0):
                raise FinalizeError(f"oracle audit invalid candidate at query {qid}")
            exact.append((exact_id, exact_d2, exact_bits, exact_value))
            fp.append((fp_id, fp_d2, fp_bits, fp_value))
        if len({row[0] for row in exact}) != TOP or len({row[0] for row in fp}) != TOP:
            raise FinalizeError(f"oracle audit duplicate top candidate at query {qid}")
        if any((exact[i - 1][1], exact[i - 1][0]) > (exact[i][1], exact[i][0]) for i in range(1, TOP)):
            raise FinalizeError(f"oracle exact ordering mismatch query {qid}")
        if any((fp[i - 1][3], fp[i - 1][0]) > (fp[i][3], fp[i][0]) for i in range(1, TOP)):
            raise FinalizeError(f"oracle fp32 ordering mismatch query {qid}")
        exact_by_id = {row[0]: row[1:] for row in exact}
        for fp_id, fp_d2, fp_bits, fp_value in fp:
            if fp_id in exact_by_id and exact_by_id[fp_id] != (fp_d2, fp_bits, fp_value):
                raise FinalizeError(f"oracle int64/fp record disagreement query {qid}")

        exact_boundary = int(exact[9][1] == exact[10][1])
        fp_boundary = int(fp[9][3] == fp[10][3])
        top10_agree = int({row[0] for row in exact[:10]} == {row[0] for row in fp[:10]})
        fp_collision = 0
        begin = 0
        while begin < TOP:
            end = begin + 1
            while end < TOP and fp[end][3] == fp[begin][3]:
                end += 1
            if len({row[1] for row in fp[begin:end]}) > 1:
                fp_collision = 1
            begin = end
        row = tie_rows[qid]
        expected_row = (exact[9][1], exact[10][1], fp[9][2], fp[10][2],
                        exact_boundary, fp_boundary, top10_agree, fp_collision,
                        int(not exact_boundary and not fp_boundary and top10_agree and not fp_collision))
        if row[1:] != expected_row:
            raise FinalizeError(f"oracle admission/audit disagreement query {qid}")

        gt_offset = qid * (4 + WIDTH * 4)
        gt_width, = struct.unpack_from("<i", gt_bytes, gt_offset)
        gt_ids = list(struct.unpack_from("<" + "i" * WIDTH, gt_bytes, gt_offset + 4))
        if gt_width != WIDTH or gt_ids != [row[0] for row in fp[:WIDTH]]:
            raise FinalizeError(f"oracle GT/audit disagreement query {qid}")
        if len(set(gt_ids)) != WIDTH:
            raise FinalizeError(f"oracle GT duplicate ID query {qid}")

        is_eligible = bool(row[-1])
        eligible.append(is_eligible)
        stats["eligible_total"] += is_eligible
        stats["exact_boundary_ties"] += exact_boundary
        stats["fp32_boundary_ties"] += fp_boundary
        stats["top10_idset_mismatches"] += 1 - top10_agree
        stats["fp32_top101_collisions_different_d2"] += fp_collision
    if offset != len(raw):
        raise FinalizeError("oracle audit trailing bytes")
    for key, value in stats.items():
        if summary.get(key) != value:
            raise FinalizeError(f"oracle summary aggregate mismatch: {key}")
    return eligible, stats, {
        "groundtruth_path": str(gt_path), "groundtruth_sha256": sha256_file(gt_path),
        "audit_path": str(audit_path), "audit_sha256": sha256_file(audit_path),
        "admission_path": str(admission_path), "admission_sha256": sha256_file(admission_path),
        "summary_path": str(summary_path), "summary_sha256": sha256_file(summary_path),
    }


def assert_no_c2_execution_state() -> None:
    # The v1 parser-only gate is an immutable, known failure. A built binary is
    # also expected now, but no normal run/ledger/lock is allowed.
    forbidden = ("runs", ".locks")
    present = [str(ROOT / name) for name in forbidden if (ROOT / name).exists() or (ROOT / name).is_symlink()]
    if present:
        raise FinalizeError(f"unexpected C2 normal execution state: {present}")
    receipts = ROOT / "preledger_receipts"
    direct_dir(receipts, "preledger receipt root")
    children = sorted(child.name for child in receipts.iterdir())
    if children != ["c2_v4_parser_gate_v1"]:
        raise FinalizeError(f"unexpected preledger receipt children: {children}")
    failed = receipts / "c2_v4_parser_gate_v1"
    direct_dir(failed, "v1 failed preledger parent")
    if (failed / "receipt.json").exists() or (failed / "receipt.json").is_symlink():
        raise FinalizeError("v1 failed preledger parent unexpectedly contains a receipt")
    for name in ("launcher_card.json", "launcher.log"):
        direct_file(failed / name, "v1 failed preledger evidence")


def write_text(path: Path, text: str) -> None:
    with path.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def final_workload_manifest(*, protocol: Path, protocol_sha: str, oracle: dict[str, str],
                            selection: dict[str, Any], stage_paths: dict[str, Path]) -> dict[str, Any]:
    source = selection.get("source", {})
    derived = selection.get("derived", {})
    if not isinstance(source, dict) or not isinstance(derived, dict):
        raise FinalizeError("selection source/derived metadata invalid")
    manifest: dict[str, Any] = {
        "schema": "gts-v4-fresh-sift-learn-workload-v1",
        "status": "READY_FOR_C2_V4_FORMAL_AFTER_PRELEDGER_PARSER_GATE",
        "selection_pre_gt_path": str(SELECTION),
        "selection_pre_gt_sha256": sha256_file(SELECTION),
        "v3_forensic_record_path": str(V3_FORENSIC),
        "v3_forensic_record_sha256": sha256_file(V3_FORENSIC),
        "formal_protocol_path": str(protocol),
        "formal_protocol_sha256": protocol_sha,
        "sealed_v2_test_forbidden": True,
        "v2_sealed_test_ids_sha256_forbidden": V2_SEALED_FORBIDDEN,
        "base_fvecs_path": str(BASE),
        "base_fvecs_sha256": sha256_file(BASE),
        "query_fvecs_path": str(FRESH_QUERY),
        "query_fvecs_sha256": sha256_file(FRESH_QUERY),
        "query_fvecs_count": N_QUERY,
        "groundtruth_ivecs_path": oracle["groundtruth_path"],
        "groundtruth_ivecs_sha256": oracle["groundtruth_sha256"],
        "groundtruth_width": WIDTH,
        "mapping_path": str(MAPPING),
        "mapping_sha256": sha256_file(MAPPING),
        "calibration_ids_path": str(stage_paths["calibration"]),
        "calibration_ids_sha256": sha256_file(stage_paths["calibration"]),
        "calibration_query_count": len(stage_paths["calibration"].read_text(encoding="ascii").splitlines()),
        "validation_ids_path": str(stage_paths["validation"]),
        "validation_ids_sha256": sha256_file(stage_paths["validation"]),
        "validation_query_count": len(stage_paths["validation"].read_text(encoding="ascii").splitlines()),
        "sealed_test_ids_path": str(stage_paths["sealed_test"]),
        "sealed_test_ids_sha256": sha256_file(stage_paths["sealed_test"]),
        "sealed_test_query_count": len(stage_paths["sealed_test"].read_text(encoding="ascii").splitlines()),
    }
    if set(manifest) != WORKLOAD_FIELDS:
        raise FinalizeError("internal workload manifest field drift")
    if source.get("base_fvecs_sha256") != manifest["base_fvecs_sha256"]:
        raise FinalizeError("base SHA does not match frozen selection")
    if derived.get("compact_query_fvecs_sha256") != manifest["query_fvecs_sha256"]:
        raise FinalizeError("query SHA does not match frozen selection")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--oracle-source", type=Path, required=True)
    parser.add_argument("--oracle-binary", type=Path, required=True)
    parser.add_argument("--oracle-command", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()

    direct_dir(ROOT, "v4 root")
    direct_dir(SELECTION_DIR, "pre-GT selection directory")
    direct_file(args.oracle_source, "CPU oracle source")
    direct_file(args.oracle_binary, "CPU oracle binary")
    direct_file(args.oracle_command, "CPU oracle command record")
    direct_file(args.protocol, "oracle-admission protocol")
    if args.oracle_source != ROOT / "tools/c2_v4_cpu_exact_oracle.cpp":
        raise FinalizeError("unexpected oracle source path")
    if args.oracle_binary != ROOT / "tools/c2_v4_cpu_exact_oracle.bin":
        raise FinalizeError("unexpected oracle binary path")
    if not str(args.oracle_dir).startswith(str(SELECTION_DIR) + "/"):
        raise FinalizeError("oracle output must live below frozen selection directory")
    if args.protocol != ROOT / "protocols/c2_v4_oracle_admission_protocol_v1.json":
        raise FinalizeError("unexpected oracle-admission protocol path")
    if FINAL_DIR.exists() or FINAL_DIR.is_symlink():
        raise FinalizeError("refuse overwrite/append final workload directory")

    assert_no_c2_execution_state()
    require_sha(SELECTION, SELECTION_SHA, "pre-GT selection manifest")
    require_sha(BASE, BASE_SHA, "SIFT1M base")
    require_sha(FRESH_QUERY, FRESH_QUERY_SHA, "fresh compact query")
    selection = load_json(SELECTION, "pre-GT selection")
    if selection.get("schema") != "gts-v4-fresh-sift-learn-selection-pre-gt-v1" or selection.get("status") != "FRESH_IDS_FROZEN_PRE_GT_NO_C2_NO_GPU":
        raise FinalizeError("pre-GT selection identity/status mismatch")
    require_sha(V1_FINALIZER_FAILURE, V1_FINALIZER_FAILURE_SHA, "v1 finalizer failure receipt")
    v1_failure = load_json(V1_FINALIZER_FAILURE, "v1 finalizer failure receipt")
    if v1_failure.get("status") != "FAILED_PRE_FINALIZATION_NO_FORMAL_STATE_CREATED":
        raise FinalizeError("v1 finalizer failure receipt state mismatch")
    require_sha(PRELEDGER_V1_FAILURE, PRELEDGER_V1_FAILURE_SHA, "v1 preledger failure receipt")
    preledger_failure = load_json(PRELEDGER_V1_FAILURE, "v1 preledger failure receipt")
    if preledger_failure.get("status") != "FAILED_PRELEDGER_NO_RECEIPT_NO_FORMAL_LEDGER":
        raise FinalizeError("v1 preledger failure receipt state mismatch")
    protocol = load_json(args.protocol, "oracle-admission protocol")
    if (protocol.get("schema") != "safe-c2-v4-oracle-admission-protocol-v1" or
            protocol.get("status") != "FROZEN_CPU_ONLY_ORACLE_AND_TIE_ADMISSION_PENDING" or
            protocol.get("canonical_root") != str(ROOT) or
            protocol.get("pre_gt_selection", {}).get("sha256") != SELECTION_SHA):
        raise FinalizeError("oracle-admission protocol identity/binding mismatch")
    if protocol.get("data_contract", {}).get("base_fvecs_sha256") != BASE_SHA:
        raise FinalizeError("oracle-admission protocol base binding mismatch")

    mapping = parse_mapping(MAPPING, N_QUERY, "fresh local-to-original mapping")
    candidates = stage_candidate_members(selection, mapping)
    eligible, oracle_stats, oracle = parse_oracle(args.oracle_dir)

    v3_sha = selection.get("source", {}).get("v3_mapping_sha256")
    if not isinstance(v3_sha, str):
        raise FinalizeError("selection lacks v3 mapping SHA")
    require_sha(V3_MAPPING, v3_sha, "v3 quarantine mapping")
    v3_mapping = parse_mapping(V3_MAPPING, 10_000, "v3 quarantine mapping")
    v3_originals = {original for original, _digest in v3_mapping.values()}
    v3_payloads = {digest for _original, digest in v3_mapping.values()}
    v2_sha = selection.get("source", {}).get("v2_query_fvecs_sha256")
    if not isinstance(v2_sha, str):
        raise FinalizeError("selection lacks v2 query SHA")
    require_sha(V2_QUERY, v2_sha, "v2 query payload source")
    v2_payloads = payload_hashes(V2_QUERY, 10_000, "v2 query payload source")
    if len(v2_payloads) != 9_999:
        raise FinalizeError("v2 payload denylist cardinality mismatch")

    final_ids: dict[str, list[int]] = {}
    final_records: dict[str, list[tuple[int, int, str]]] = {}
    seen_local: set[int] = set()
    seen_original: set[int] = set()
    seen_payload: set[str] = set()
    per_stage_counts: dict[str, int] = {}
    for name, _candidate_count, minimum in STAGES:
        ids = [local for local in candidates[name] if eligible[local]]
        if len(ids) < minimum:
            raise FinalizeError(f"{name} eligible count {len(ids)} below predeclared minimum {minimum}")
        records = [(local, *mapping[local]) for local in ids]
        locals_here = {local for local, _original, _payload in records}
        originals_here = {original for _local, original, _payload in records}
        payloads_here = {payload for _local, _original, payload in records}
        if len(locals_here) != len(records) or len(originals_here) != len(records) or len(payloads_here) != len(records):
            raise FinalizeError(f"{name} final split contains duplicate identity")
        if seen_local.intersection(locals_here) or seen_original.intersection(originals_here) or seen_payload.intersection(payloads_here):
            raise FinalizeError(f"{name} final split overlaps a prior final split")
        if any(original < 3072 for original in originals_here):
            raise FinalizeError(f"{name} reserved drift original leaked")
        if originals_here.intersection(v3_originals) or payloads_here.intersection(v3_payloads) or payloads_here.intersection(v2_payloads):
            raise FinalizeError(f"{name} v2/v3 quarantine leakage")
        seen_local.update(locals_here)
        seen_original.update(originals_here)
        seen_payload.update(payloads_here)
        final_ids[name] = ids
        final_records[name] = records
        per_stage_counts[name] = len(ids)

    parent = FINAL_DIR.parent
    direct_dir(parent, "final workload parent")
    temporary = parent / (FINAL_DIR.name + ".tmp_oracle_admission")
    if temporary.exists() or temporary.is_symlink():
        raise FinalizeError(f"temporary output already exists: {temporary}")
    temporary.mkdir(mode=0o750)
    try:
        stage_paths: dict[str, Path] = {}
        for name, _candidate_count, _minimum in STAGES:
            ids_path = temporary / f"{name}.ids"
            witness_path = temporary / f"{name}.mapping.tsv"
            write_text(ids_path, "".join(f"{local}\n" for local in final_ids[name]))
            write_text(witness_path, "".join(f"{local}\t{original}\t{payload}\n" for local, original, payload in final_records[name]))
            stage_paths[name] = ids_path

        protocol_sha = sha256_file(args.protocol)
        admission_receipt = {
            "schema": "safe-c2-v4-oracle-admission-receipt-v1",
            "status": "PASS_CPU_ONLY_TIE_ADMISSION_NO_FORMAL_EXECUTION",
            "scope": "CPU-only independent exact oracle and deterministic no-replacement admission. No CUDA/GPU/preledger/ledger/stage execution occurred.",
            "selection_pre_gt": {"path": str(SELECTION), "sha256": sha256_file(SELECTION)},
            "oracle_protocol": {"path": str(args.protocol), "sha256": protocol_sha},
            "oracle_source": {"path": str(args.oracle_source), "sha256": sha256_file(args.oracle_source)},
            "oracle_binary": {"path": str(args.oracle_binary), "sha256": sha256_file(args.oracle_binary)},
            "oracle_command": {"path": str(args.oracle_command), "sha256": sha256_file(args.oracle_command)},
            "admission_finalizer_v3": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "v1_finalizer_failure": {"path": str(V1_FINALIZER_FAILURE), "sha256": sha256_file(V1_FINALIZER_FAILURE)},
            "preledger_v1_failure": {"path": str(PRELEDGER_V1_FAILURE), "sha256": sha256_file(PRELEDGER_V1_FAILURE)},
            "final_workload_generation": "v2_manifest_paths_are_post-rename canonical paths",
            "oracle_outputs": oracle,
            "oracle_aggregate": oracle_stats,
            "final_split_counts": per_stage_counts,
            "minimums": {name: minimum for name, _count, minimum in STAGES},
            "all_final_ids_are_candidate_subsets": True,
            "pairwise_disjoint_local_original_payload": True,
            "v2_v3_quarantine_checked": True,
            "no_replacement_or_cross_split_substitution": True,
            "gpu_binary_executed": False,
            "nvidia_smi_called": False,
            "formal_claim_eligible": False,
            "next_gate": "Independently pin final Safe-C2 v4 binary/source and pass --validate-workload-only receipt verification before any ledger/GPU action."
        }
        write_json(temporary / "oracle_admission_receipt.json", admission_receipt)

        manifest = final_workload_manifest(protocol=args.protocol, protocol_sha=protocol_sha, oracle=oracle,
                                           selection=selection, stage_paths=stage_paths)
        # The files are currently in the temporary directory, but the manifest
        # must bind their canonical post-rename locations, not transient paths.
        for formal_stage in ("calibration", "validation", "sealed_test"):
            manifest[f"{formal_stage}_ids_path"] = str(FINAL_DIR / stage_paths[formal_stage].name)
        write_json(temporary / "workload_manifest.json", manifest)

        hashes = {
            child.name: sha256_file(child)
            for child in sorted(temporary.iterdir()) if child.is_file() and child.name != "final_artifacts.sha256"
        }
        write_text(temporary / "final_artifacts.sha256", "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())))
        os.replace(temporary, FINAL_DIR)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    result = {
        "status": "PASS_C2_V4_CPU_ORACLE_ADMISSION",
        "final_workload_dir": str(FINAL_DIR),
        "final_manifest": str(FINAL_DIR / "workload_manifest.json"),
        "final_manifest_sha256": sha256_file(FINAL_DIR / "workload_manifest.json"),
        "final_split_counts": per_stage_counts,
        "oracle_stats": oracle_stats,
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
        "formal_claim_eligible": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FINALIZE_C2_V4_CPU_ORACLE_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
