#!/usr/bin/env python3
"""Read-only Fable5 constructed-branch-coverage trace preflight before NVML action.

This tool validates only bytes, ABI, layout, identity, and the required 12-op
candidate shape.  Native GTS sibling-boundary certificate outcomes are absent
from E1GTRC02 and deliberately remain NOT_ESTABLISHED_STATICALLY here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

CANONICAL_ROOT = Path("/workspace/experiments/tide_safe_c1_20260730/safe_c1_fable5_matched_v1")
CONTRACT_NAME = "fable5_branchcov_matched_trace_preflight_v1.json"
BRANCH_VARIANT_ID = "constructed_branch_coverage_micro_witness_v1"
BRANCH_SOURCE_BUNDLE = "fable5_branch_coverage_e1_20260730"
CANDIDATE_NAME = "fable5_matched_trace_candidate_v1.json"
TRACE_NAME = "trace.fable5.e1gtrc"
CONTRACT_SCHEMA = "fable5-matched-trace-preflight-contract-v1"
CANDIDATE_SCHEMA = "fable5-matched-trace-candidate-v1"
CANDIDATE_STATUS = "CPU_CANDIDATE_NOT_NATIVE_RUNTIME_VALIDATED"
REPORT_SCHEMA = "fable5-branchcov-matched-trace-preflight-report-v1"
PASS_STATUS = "PASS_FABLE5_BRANCHCOV_MATCHED_TRACE_STATIC_PREFLIGHT"
FAIL_STATUS = "FAIL_FABLE5_BRANCHCOV_MATCHED_TRACE_STATIC_PREFLIGHT"
TRACE_MAGIC = b"E1GTRC02"
TRACE_VERSION = 2
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3si")
OP_INSERT = 1
OP_DELETE = 2
OP_KNN = 3
OP_RANGE = 4
OP_REBUILD = 5

EXPECTED_HEADER: dict[str, Any] = {
    "magic": "E1GTRC02",
    "version": 2,
    "dimension": 32,
    "base_n": 4096,
    "reservoir_n": 2048,
    "pool_n": 6144,
    "query_n": 3,
    "k": 10,
    "radius": 300.0,
    "event_count": 12,
}
EXPECTED_SEQUENCE = [
    {"op_index": 0, "op": "INSERT", "slot": "A"},
    {"op_index": 1, "op": "KNN", "slot": "query_A"},
    {"op_index": 2, "op": "INSERT", "slot": "B"},
    {"op_index": 3, "op": "INSERT", "slot": "R"},
    {"op_index": 4, "op": "KNN", "slot": "query_mid"},
    {"op_index": 5, "op": "DELETE", "slot": "A"},
    {"op_index": 6, "op": "DELETE", "slot": "B"},
    {"op_index": 7, "op": "INSERT", "slot": "C"},
    {"op_index": 8, "op": "KNN", "slot": "query_C"},
    {"op_index": 9, "op": "DELETE", "slot": "base_delete"},
    {"op_index": 10, "op": "REBUILD", "slot": "rebuild"},
    {"op_index": 11, "op": "KNN", "slot": "query_post"},
]
OP_NAME = {OP_INSERT: "INSERT", OP_DELETE: "DELETE", OP_KNN: "KNN", OP_RANGE: "RANGE", OP_REBUILD: "REBUILD"}
EXPECTED_RUNTIME_REQUIREMENTS: dict[str, Any] = {
    "requires_A_B_C_same_certified_leaf": True,
    "requires_R_native_certificate_leaf": -1,
    "requires_B_capacity_fallback_to_global_delta": True,
    "requires_R_certificate_reject_to_global_delta": True,
    "requires_A_and_C_direct_visibility_in_exact_results": True,
    "requires_both_policies_same_trace_and_full_active_oracle": True,
}


class PreflightError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def resolved_under(path: Path, parent: Path, label: str, direct_child: bool = False) -> Path:
    resolved = path.resolve(strict=True)
    base = parent.resolve(strict=True)
    try:
        relative = resolved.relative_to(base)
    except ValueError as exc:
        raise PreflightError(f"{label}_outside_required_root:{resolved}") from exc
    if direct_child and len(relative.parts) != 1:
        raise PreflightError(f"{label}_must_be_direct_child_of:{base}")
    return resolved


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"invalid_json:{label}:{type(exc).__name__}") from exc
    require(isinstance(value, dict), f"json_not_object:{label}")
    return value


def expect_exact_mapping(value: Any, expected: dict[str, Any], label: str) -> None:
    require(value == expected, f"unexpected_{label}")


def parse_header(raw: bytes) -> tuple[dict[str, Any], list[tuple[int, int, bytes, int]]]:
    require(len(raw) >= HEADER.size, "truncated_trace_header")
    magic, version, dimension, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = HEADER.unpack_from(raw, 0)
    header = {
        "magic": magic.decode("ascii", errors="replace"),
        "version": int(version),
        "dimension": int(dimension),
        "base_n": int(base_n),
        "reservoir_n": int(reservoir_n),
        "pool_n": int(pool_n),
        "query_n": int(query_n),
        "k": int(k),
        "radius": float(radius),
        "event_count": int(event_count),
    }
    require(magic == TRACE_MAGIC and version == TRACE_VERSION, "unsupported_trace_magic_or_version")
    require(math.isfinite(float(radius)), "nonfinite_trace_radius")
    expect_exact_mapping(header, EXPECTED_HEADER, "fable5_32d_trace_header")
    expected_bytes = HEADER.size + event_count * EVENT.size
    require(len(raw) == expected_bytes, f"wrong_trace_byte_count:expected={expected_bytes}:actual={len(raw)}")
    events: list[tuple[int, int, bytes, int]] = []
    for index in range(event_count):
        op_index, op, reserved, argument = EVENT.unpack_from(raw, HEADER.size + index * EVENT.size)
        require(op_index == index, f"noncontiguous_event_index:{index}")
        require(reserved == b"\x00\x00\x00", f"nonzero_reserved_event_bytes:{index}")
        require(op in OP_NAME, f"unknown_event_opcode:{index}:{op}")
        events.append((int(op_index), int(op), reserved, int(argument)))
    return header, events


def parse_and_validate_trace(trace: Path) -> tuple[dict[str, Any], dict[str, int]]:
    header, events = parse_header(trace.read_bytes())
    require(len(events) == len(EXPECTED_SEQUENCE), "fable5_trace_not_exact_12_events")
    for expected, actual in zip(EXPECTED_SEQUENCE, events):
        index, op, _reserved, argument = actual
        require(index == expected["op_index"], f"wrong_event_index:{expected['op_index']}")
        expected_op = expected["op"]
        require(OP_NAME[op] == expected_op, f"wrong_event_opcode:{index}:expected={expected_op}:actual={OP_NAME[op]}")
        if expected_op == "REBUILD":
            require(argument == 0, "rebuild_argument_must_be_zero")
    # Give the fields explicit role names, but do not assign a certificate
    # outcome to any inserted ID.  That requires native frozen-tree execution.
    slots = {
        "A": events[0][3],
        "query_A": events[1][3],
        "B": events[2][3],
        "R": events[3][3],
        "query_mid": events[4][3],
        "C": events[7][3],
        "query_C": events[8][3],
        "base_delete": events[9][3],
        "query_post": events[11][3],
    }
    inserted = [slots[name] for name in ("A", "B", "R", "C")]
    require(len(set(inserted)) == 4, "insert_slots_must_be_distinct")
    require(all(header["base_n"] <= ident < header["pool_n"] for ident in inserted), "insert_slot_outside_reservoir")
    require(events[5][3] == slots["A"], "delete_A_does_not_match_insert_A")
    require(events[6][3] == slots["B"], "delete_B_does_not_match_insert_B")
    require(0 <= slots["base_delete"] < header["base_n"], "base_delete_not_in_initial_base")
    require(all(0 <= slots[name] < header["query_n"] for name in ("query_A", "query_mid", "query_C", "query_post")), "query_slot_outside_query_pool")
    # The ABI carries query IDs, not query observability receipts. The four
    # query slots may legitimately differ; A/C visibility is a later native
    # traversal+exact-oracle runtime requirement, not a static label.
    # The structural active set after immediate rebuild must retain R and C,
    # while A/B are deleted and the base deletion is absent.
    active_after_rebuild_delta = {"added": sorted([slots["R"], slots["C"]]), "removed_base": slots["base_delete"], "deleted": sorted([slots["A"], slots["B"]])}
    return header, {**slots, "_active_after_rebuild_delta": active_after_rebuild_delta}  # type: ignore[dict-item]


def validate_identity_layout(bundle: Path, header: dict[str, Any]) -> None:
    mapping_path = bundle / "stable_id_to_pool_row.i32"
    initial_path = bundle / "initial_base_stable_ids.i32"
    mapping = [value[0] for value in struct.iter_unpack("<i", mapping_path.read_bytes())]
    initial = [value[0] for value in struct.iter_unpack("<i", initial_path.read_bytes())]
    require(len(mapping) == header["pool_n"], "mapping_length_does_not_match_header")
    require(len(initial) == header["base_n"], "initial_base_length_does_not_match_header")
    require(mapping == list(range(header["pool_n"])), "stable_id_to_pool_row_not_identity")
    require(initial == list(range(header["base_n"])), "initial_base_not_identity_prefix")


def validate_static_contract(contract: dict[str, Any], root: Path, contract_path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    require(contract.get("schema") == CONTRACT_SCHEMA, "wrong_preflight_contract_schema")
    require(contract.get("status") == "STATIC_PRE_NVML_CONTRACT_ONLY", "wrong_preflight_contract_status")
    require(contract.get("root") == str(root), "preflight_contract_root_mismatch")
    require(contract.get("execution_boundary") == {"gpu_used": False, "cuda_binary_executed": False, "nvml_queried": False}, "preflight_contract_execution_boundary_invalid")
    abi = contract.get("trace_abi")
    require(isinstance(abi, dict), "missing_trace_abi")
    expect_exact_mapping(abi.get("header"), EXPECTED_HEADER, "contract_trace_header")
    require(abi.get("header_bytes") == HEADER.size and abi.get("event_bytes") == EVENT.size, "contract_abi_size_mismatch")
    require(contract.get("event_sequence") == EXPECTED_SEQUENCE, "contract_event_sequence_mismatch")
    require(contract.get("candidate_contract") == {
        "filename": CANDIDATE_NAME,
        "schema": CANDIDATE_SCHEMA,
        "required_status": CANDIDATE_STATUS,
    }, "contract_candidate_contract_shape_mismatch")
    require(contract.get("runtime_requirements") == EXPECTED_RUNTIME_REQUIREMENTS, "contract_runtime_requirements_mismatch")
    limitations = contract.get("static_limitations")
    require(isinstance(limitations, list) and "native_certificate_outcomes_not_established_statically" in limitations, "contract_missing_certificate_static_limitation")
    layout = contract.get("bundle_layout")
    require(isinstance(layout, dict), "missing_bundle_layout")
    require(layout.get("trace_filename") == TRACE_NAME, "contract_trace_filename_mismatch")
    require(layout.get("stable_id_layout") == "identity_full_immutable_pool_seeded_stable_ids_v5", "contract_identity_layout_mismatch")
    raw_files = layout.get("required_files")
    require(isinstance(raw_files, dict) and set(raw_files) == {"pool.i16", "queries.i16", "stable_id_to_pool_row.i32", "initial_base_stable_ids.i32"}, "contract_required_file_set_mismatch")
    bundle_files: dict[str, dict[str, Any]] = {}
    for name, detail in raw_files.items():
        require(isinstance(detail, dict) and isinstance(detail.get("bytes"), int) and detail["bytes"] >= 0 and is_sha256(detail.get("sha256")), f"invalid_contract_file_hash:{name}")
        bundle_files[name] = {"bytes": detail["bytes"], "sha256": detail["sha256"]}
    raw_sources = contract.get("root_source_hashes")
    require(isinstance(raw_sources, dict) and set(raw_sources) == {"src/safe_c1_dynamic_gts.cu", "include/safe_c1_search_native_routing_certificate.hpp", "src/fable5_matched_gpu_runner_v1.cu"}, "contract_root_source_hash_set_mismatch")
    source_hashes: dict[str, str] = {}
    for relative, digest in raw_sources.items():
        require(isinstance(relative, str) and is_sha256(digest), f"invalid_contract_source_hash:{relative}")
        candidate = root / relative
        resolved_under(candidate, root, f"contract_source:{relative}")
        require(candidate.is_file() and sha256_file(candidate) == digest, f"root_source_hash_mismatch:{relative}")
        source_hashes[relative] = digest
    require(contract_path == root / "manifests" / CONTRACT_NAME, "contract_path_not_canonical")
    variant = contract.get("variant")
    require(isinstance(variant, dict), "branchcov_variant_missing")
    require(variant.get("id") == BRANCH_VARIANT_ID, "branchcov_variant_id_mismatch")
    require(variant.get("source_bundle") == BRANCH_SOURCE_BUNDLE, "branchcov_source_bundle_mismatch")
    require("performance" in variant.get("prohibited_claims", []), "branchcov_performance_prohibition_missing")
    witness = variant.get("witness")
    witness_sha = variant.get("witness_sha256")
    require(isinstance(witness, str) and isinstance(witness_sha, str), "branchcov_witness_binding_missing")
    witness_path = root / witness
    require(witness_path.is_file() and sha256_file(witness_path) == witness_sha, "branchcov_witness_binding_mismatch")
    return source_hashes, bundle_files


def validate_candidate_contract(candidate: dict[str, Any], candidate_path: Path, candidate_sha: str, root: Path, bundle: Path, trace: Path, trace_sha: str, header: dict[str, Any], slots: dict[str, Any], source_hashes: dict[str, str], bundle_files: dict[str, dict[str, Any]], leaf_capacity: int) -> dict[str, Any]:
    require(sha256_file(candidate_path) == candidate_sha, "candidate_contract_sha256_mismatch")
    require(candidate.get("schema") == CANDIDATE_SCHEMA, "wrong_candidate_contract_schema")
    require(candidate.get("status") == CANDIDATE_STATUS, "candidate_contract_not_cpu_candidate_only")
    require(candidate.get("trace") == {"filename": TRACE_NAME, "sha256": trace_sha}, "candidate_trace_binding_mismatch")
    expect_exact_mapping(candidate.get("trace_header"), header, "candidate_trace_header")
    require(candidate.get("leaf_capacity") == leaf_capacity == 1, "candidate_leaf_capacity_mismatch")
    require(candidate.get("stable_id_layout") == "identity_full_immutable_pool_seeded_stable_ids_v5", "candidate_identity_layout_mismatch")
    expected_payload = {name: detail["sha256"] for name, detail in bundle_files.items()}
    expect_exact_mapping(candidate.get("immutable_payload_sha256"), expected_payload, "candidate_immutable_payload_hashes")
    expect_exact_mapping(candidate.get("root_source_hashes"), source_hashes, "candidate_root_source_hashes")
    require(candidate.get("archived_core_sha256") == source_hashes["src/safe_c1_dynamic_gts.cu"], "candidate_archived_core_sha256_mismatch")
    expected_slots = {name: slots[name] for name in ("A", "B", "R", "C", "query_A", "query_mid", "query_C", "base_delete", "query_post")}
    expect_exact_mapping(candidate.get("slot_ids"), expected_slots, "candidate_slot_ids")
    expect_exact_mapping(candidate.get("runtime_requirements"), EXPECTED_RUNTIME_REQUIREMENTS, "candidate_runtime_requirements")
    require(candidate.get("static_certificate_outcomes") == "NOT_ESTABLISHED_STATICALLY", "candidate_must_not_claim_static_certificate_outcomes")
    expected_contract_binding = {"relative_path": f"manifests/{CONTRACT_NAME}", "sha256": sha256_file(root / "manifests" / CONTRACT_NAME)}
    expect_exact_mapping(candidate.get("preflight_contract"), expected_contract_binding, "candidate_branchcov_preflight_contract")
    source = candidate.get("candidate_source")
    require(isinstance(source, dict), "candidate_source_missing")
    relative = source.get("relative_path")
    digest = source.get("sha256")
    require(isinstance(relative, str) and relative.startswith("tools/") and ".." not in Path(relative).parts and is_sha256(digest), "candidate_source_identity_invalid")
    source_path = root / relative
    resolved_under(source_path, root, "candidate_source")
    require(source_path.is_file() and sha256_file(source_path) == digest, "candidate_source_sha256_mismatch")
    require(candidate_path == bundle / CANDIDATE_NAME, "candidate_contract_not_bundle_local")
    require(trace == bundle / TRACE_NAME, "trace_not_bundle_local_canonical_name")
    return {"candidate_contract_sha256": candidate_sha, "candidate_source": {"relative_path": relative, "sha256": digest}}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    require(root == CANONICAL_ROOT, "root_must_equal_isolated_fable5_root")
    contract_path = args.contract.resolve(strict=True)
    bundle = resolved_under(args.bundle, root / "bundles", "bundle", direct_child=True)
    require(bundle.name.startswith("fable5_branchcov_"), "branchcov_candidate_bundle_name_required")
    trace = args.trace.resolve(strict=True)
    candidate_path = args.candidate_contract.resolve(strict=True)
    require(is_sha256(args.trace_sha256), "trace_sha256_must_be_lowercase_hex")
    require(is_sha256(args.candidate_contract_sha256), "candidate_contract_sha256_must_be_lowercase_hex")
    require(args.leaf_capacity == 1, "fable5_matched_trace_requires_leaf_capacity_one")
    require(contract_path == root / "manifests" / CONTRACT_NAME, "contract_must_be_canonical")
    contract = read_json_object(contract_path, "preflight_contract")
    source_hashes, bundle_files = validate_static_contract(contract, root, contract_path)
    require(trace == bundle / TRACE_NAME, "trace_must_be_canonical_bundle_local_file")
    require(candidate_path == bundle / CANDIDATE_NAME, "candidate_contract_must_be_canonical_bundle_local_file")
    for name, detail in bundle_files.items():
        path = bundle / name
        require(path.is_file(), f"missing_bundle_file:{name}")
        require(path.stat().st_size == detail["bytes"], f"bundle_byte_count_mismatch:{name}")
        require(sha256_file(path) == detail["sha256"], f"bundle_sha256_mismatch:{name}")
    require(sha256_file(trace) == args.trace_sha256, "trace_sha256_mismatch")
    header, slots = parse_and_validate_trace(trace)
    validate_identity_layout(bundle, header)
    candidate = read_json_object(candidate_path, "candidate_contract")
    candidate_identity = validate_candidate_contract(candidate, candidate_path, args.candidate_contract_sha256, root, bundle, trace, args.trace_sha256, header, slots, source_hashes, bundle_files, args.leaf_capacity)
    return {
        "schema": REPORT_SCHEMA,
        "status": PASS_STATUS,
        "execution_boundary": {"gpu_used": False, "cuda_binary_executed": False, "nvml_queried": False},
        "scope": "Read-only ABI/layout/identity/event-sequence gate for a constructed branch-coverage micro-witness before run-directory creation and NVML; not a native certificate, GPU correctness, real-workload, or performance result.",
        "root": str(root),
        "bundle": str(bundle),
        "trace": str(trace),
        "trace_sha256": args.trace_sha256,
        "candidate_contract": str(candidate_path),
        "header": header,
        "slot_ids": {name: value for name, value in slots.items() if not name.startswith("_")},
        "structural_active_after_rebuild": slots["_active_after_rebuild_delta"],
        "bundle_file_sha256": {name: detail["sha256"] for name, detail in bundle_files.items()},
        "root_source_hashes": source_hashes,
        "candidate_identity": candidate_identity,
        "certificate_outcome": {
            "status": "NOT_ESTABLISHED_STATICALLY",
            "reason": "E1GTRC02 carries only opcodes/IDs. Native frozen GTS execution must emit certificates and traversal receipts for A/B/C/R.",
            "runtime_requirements_only": EXPECTED_RUNTIME_REQUIREMENTS,
        },
        "errors": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--trace-sha256", required=True)
    parser.add_argument("--candidate-contract", type=Path, required=True)
    parser.add_argument("--candidate-contract-sha256", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--leaf-capacity", type=int, required=True)
    args = parser.parse_args()
    report: dict[str, Any]
    try:
        report = run(args)
    except Exception as exc:
        report = {
            "schema": REPORT_SCHEMA,
            "status": FAIL_STATUS,
            "execution_boundary": {"gpu_used": False, "cuda_binary_executed": False, "nvml_queried": False},
            "scope": "Read-only fail-closed pre-NVML trace gate; no native certificate outcome is established statically.",
            "certificate_outcome": {"status": "NOT_ESTABLISHED_STATICALLY"},
            "errors": [f"{type(exc).__name__}:{exc}"],
        }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["status"] == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
