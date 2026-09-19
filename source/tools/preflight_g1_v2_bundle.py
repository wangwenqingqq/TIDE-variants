#!/usr/bin/env python3
"""CPU-only verifier for new Safe-C1 G1 v2 bounded-tie bundles.

`--mode witness` authorizes only a bundle finalized by a new-v2 certificate
selection artifact. `--mode certificate-selection` authorizes only a fresh,
pending new-v2 selection bundle for the separate guarded selection runner.
Neither mode launches CUDA/GPU work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safe_c1_v2_bundle_contract import (
    BOUNDED_TIE_ABORT,
    BOUNDED_TIE_DOMAIN,
    CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
    DYNAMIC_K_BOUNDARY_POLICY,
    STATIC_BASE_POLICY,
    BundleContractError,
    OP_DELETE,
    OP_INSERT,
    OP_KNN,
    load_current_runner_identity_stable_id_layout,
    load_i16,
    parse_trace,
    sha256,
    validate_tie_free_witness,
)

CERTIFICATE = {
    "schema": "safe-c1-search-native-sibling-boundary-v2",
    "native_search_boundary": "non-last: min_i + epsilon < radius < next_sibling_min - epsilon; final: radius > min_last + epsilon",
    "required_parent_shape": "all 10 physical siblings non-empty, common pivot, finite strictly increasing min_dis",
    "max_distance": "remains in frozen hash only; never used as a direct-routing upper bound",
    "on_failure": "fail closed to exact global delta",
}
TIE_DOMAIN = BOUNDED_TIE_DOMAIN


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_selection_contract(bundle: Path, manifest: dict[str, Any], errors: list[str]) -> dict[str, Any] | None:
    selection = manifest.get("candidate_selection")
    if not isinstance(selection, dict):
        fail(errors, "missing_candidate_selection_contract")
        return None
    if selection.get("status") != "PENDING_V2_CERTIFICATE_SELECTION":
        fail(errors, "certificate_selection_not_pending")
    if selection.get("required_schema") != "safe-c1-g1-v2-candidate-selection":
        fail(errors, "certificate_selection_wrong_required_schema")
    if selection.get("v1_evidence_used") is not False:
        fail(errors, "certificate_selection_v1_evidence_not_explicitly_rejected")
    artifact = selection.get("selection_contract")
    artifact_sha = selection.get("selection_contract_sha256")
    if not isinstance(artifact, str) or not isinstance(artifact_sha, str):
        fail(errors, "certificate_selection_missing_contract_binding")
        return None
    path = (bundle / artifact).resolve()
    try:
        path.relative_to(bundle)
    except ValueError:
        fail(errors, "certificate_selection_contract_outside_bundle")
        return None
    if not path.is_file() or sha256(path) != artifact_sha:
        fail(errors, "certificate_selection_contract_sha_mismatch")
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        fail(errors, f"bad_certificate_selection_contract:{exc}")
        return None
    if raw.get("schema") != "safe-c1-g1-v2-certificate-selection-contract":
        fail(errors, "certificate_selection_contract_wrong_schema")
    if raw.get("engine_schema") != "safe-c1-g1-topk-v2-search-native":
        fail(errors, "certificate_selection_contract_wrong_engine_schema")
    if raw.get("certificate_schema") != CERTIFICATE["schema"]:
        fail(errors, "certificate_selection_contract_wrong_certificate_schema")
    if raw.get("tie_domain") != TIE_DOMAIN:
        fail(errors, "certificate_selection_contract_wrong_tie_domain")
    if raw.get("static_base_oracle_policy") != STATIC_BASE_POLICY:
        fail(errors, "certificate_selection_contract_wrong_static_base_policy")
    if raw.get("dynamic_k_boundary_policy") != DYNAMIC_K_BOUNDARY_POLICY:
        fail(errors, "certificate_selection_contract_wrong_dynamic_boundary_policy")
    if raw.get("v1_evidence_used") is not False:
        fail(errors, "certificate_selection_contract_v1_evidence_not_rejected")
    if raw.get("stable_id_layout") != CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE:
        fail(errors, "certificate_selection_contract_wrong_stable_id_layout")
    if raw.get("leaf_capacity") != 1:
        fail(errors, "certificate_selection_contract_leaf_capacity_not_one")
    if not isinstance(raw.get("min_direct_candidates"), int) or raw["min_direct_candidates"] < 3:
        fail(errors, "certificate_selection_contract_bad_min_direct")
    if not isinstance(raw.get("min_same_sidecar_leaf_direct_candidates"), int) or \
            raw["min_same_sidecar_leaf_direct_candidates"] < 3:
        fail(errors, "certificate_selection_contract_bad_min_same_leaf_direct")
    elif raw["min_same_sidecar_leaf_direct_candidates"] > raw["min_direct_candidates"]:
        fail(errors, "certificate_selection_contract_same_leaf_gate_exceeds_direct_gate")
    if raw.get("certificate_delta_rejections_policy") != "record_only_not_selection_gate":
        fail(errors, "certificate_selection_contract_bad_certificate_delta_policy")
    followup = raw.get("g1b_followup")
    if not isinstance(followup, dict) or followup.get("leaf_capacity") != 2 or \
            followup.get("same_leaf_group_size") != 3:
        fail(errors, "certificate_selection_contract_bad_g1b_followup")
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        fail(errors, "certificate_selection_contract_missing_candidates")
    else:
        seen: set[int] = set()
        for index, row in enumerate(candidates):
            if not isinstance(row, dict):
                fail(errors, f"certificate_selection_bad_candidate:{index}")
                continue
            expected = {"stable_id", "query_id", "insert_op_index", "query_op_index", "delete_op_index"}
            if set(row) != expected or not all(isinstance(row[key], int) for key in expected):
                fail(errors, f"certificate_selection_bad_candidate_fields:{index}")
                continue
            if row["stable_id"] in seen:
                fail(errors, f"certificate_selection_duplicate_stable_id:{row['stable_id']}")
            seen.add(row["stable_id"])
            if not (row["insert_op_index"] + 1 == row["query_op_index"] and
                    row["query_op_index"] + 1 == row["delete_op_index"]):
                fail(errors, f"certificate_selection_nonlocal_event_triplet:{index}")
    return raw if not errors else None


def validate_exact_isolated_selection_trace(
    header: Any, events: list[Any], contract: dict[str, Any], errors: list[str]
) -> bool:
    """Ensure every selection delta is attributable to the certificate, not occupancy.

    Leaf capacity is one, so every declared candidate must be the sole dynamic
    object in a strict insert/query/delete triplet. Any extra trace event could
    create an occupancy-induced delta and is rejected before a GPU run.
    """
    before = len(errors)
    candidates = contract.get("candidates")
    if not isinstance(candidates, list):
        fail(errors, "certificate_selection_trace_missing_candidates")
        return False
    expected_event_count = 3 * len(candidates)
    if header.event_count != expected_event_count or len(events) != expected_event_count:
        fail(errors, "certificate_selection_trace_not_exactly_isolated")
    if header.query_n != len(candidates):
        fail(errors, "certificate_selection_trace_query_count_mismatch")
    for index, row in enumerate(candidates):
        if not isinstance(row, dict) or not all(
            isinstance(row.get(key), int)
            for key in ("stable_id", "query_id", "insert_op_index", "query_op_index", "delete_op_index")
        ):
            fail(errors, f"certificate_selection_trace_bad_contract_row:{index}")
            continue
        first = 3 * index
        if (row["query_id"] != index or row["insert_op_index"] != first or
                row["query_op_index"] != first + 1 or row["delete_op_index"] != first + 2):
            fail(errors, f"certificate_selection_trace_contract_not_sequential:{index}")
            continue
        if first + 2 >= len(events):
            fail(errors, f"certificate_selection_trace_triplet_missing:{index}")
            continue
        insert, query, delete = events[first:first + 3]
        if insert.op != OP_INSERT or insert.argument != row["stable_id"] or \
                query.op != OP_KNN or query.argument != row["query_id"] or \
                delete.op != OP_DELETE or delete.argument != row["stable_id"]:
            fail(errors, f"certificate_selection_trace_triplet_mismatch:{index}")
    return len(errors) == before


def validate_ready_selection(bundle: Path, manifest: dict[str, Any], errors: list[str]) -> bool:
    selection = manifest.get("candidate_selection")
    if not isinstance(selection, dict):
        fail(errors, "missing_candidate_selection_contract")
        return False
    if selection.get("status") != "READY_FOR_V2_WITNESS":
        fail(errors, "candidate_selection_pending_not_execution_ready")
        return False
    if selection.get("tie_domain") != TIE_DOMAIN or \
            selection.get("static_base_oracle_policy") != STATIC_BASE_POLICY or \
            selection.get("dynamic_k_boundary_policy") != DYNAMIC_K_BOUNDARY_POLICY or \
            selection.get("on_tie_violation") != BOUNDED_TIE_ABORT or \
            selection.get("stable_id_layout") != CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE or \
            selection.get("selection_trace_exactly_isolated") is not True:
        fail(errors, "ready_selection_manifest_tie_or_isolation_binding_mismatch")
    artifact = selection.get("artifact")
    artifact_sha = selection.get("artifact_sha256")
    if not isinstance(artifact, str) or not isinstance(artifact_sha, str):
        fail(errors, "ready_selection_missing_artifact_binding")
        return False
    artifact_path = (bundle / artifact).resolve()
    try:
        artifact_path.relative_to(bundle)
    except ValueError:
        fail(errors, "ready_selection_artifact_outside_bundle")
        return False
    if not artifact_path.is_file() or sha256(artifact_path) != artifact_sha:
        fail(errors, "ready_selection_artifact_sha_mismatch")
        return False
    files = manifest.get("files_sha256")
    if not isinstance(files, dict) or files.get(artifact) != artifact_sha:
        fail(errors, "ready_selection_artifact_not_in_manifest_hashes")
    try:
        selection_doc = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        fail(errors, f"bad_ready_selection_artifact:{exc}")
        return False
    if selection_doc.get("schema") != "safe-c1-g1-v2-candidate-selection":
        fail(errors, "ready_selection_wrong_schema")
    if selection_doc.get("status") != "READY_FOR_V2_WITNESS":
        fail(errors, "ready_selection_not_ready")
    if selection_doc.get("engine_schema") != "safe-c1-g1-topk-v2-search-native":
        fail(errors, "ready_selection_wrong_engine_schema")
    if selection_doc.get("certificate_schema") != CERTIFICATE["schema"]:
        fail(errors, "ready_selection_wrong_certificate_schema")
    if selection_doc.get("tie_domain") != TIE_DOMAIN:
        fail(errors, "ready_selection_wrong_tie_domain")
    if selection_doc.get("static_base_oracle_policy") != STATIC_BASE_POLICY:
        fail(errors, "ready_selection_wrong_static_base_policy")
    if selection_doc.get("dynamic_k_boundary_policy") != DYNAMIC_K_BOUNDARY_POLICY:
        fail(errors, "ready_selection_wrong_dynamic_boundary_policy")
    if selection_doc.get("selection_trace_exactly_isolated") is not True:
        fail(errors, "ready_selection_trace_not_exactly_isolated")
    if selection_doc.get("on_tie_violation") != BOUNDED_TIE_ABORT:
        fail(errors, "ready_selection_wrong_tie_abort")
    if selection_doc.get("v1_evidence_used") is not False:
        fail(errors, "ready_selection_v1_evidence_not_explicitly_rejected")
    if selection_doc.get("stable_id_layout") != CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE:
        fail(errors, "ready_selection_wrong_stable_id_layout")
    direct_candidates = selection_doc.get("direct_candidates")
    if not isinstance(direct_candidates, list) or len(direct_candidates) < 3:
        fail(errors, "ready_selection_has_insufficient_direct_candidates")
    required_same_leaf = selection_doc.get("min_same_sidecar_leaf_direct_candidates")
    group = selection_doc.get("g1b_same_leaf_direct_group")
    if not isinstance(required_same_leaf, int) or required_same_leaf < 3:
        fail(errors, "ready_selection_bad_same_leaf_requirement")
    if not isinstance(group, dict):
        fail(errors, "ready_selection_missing_same_leaf_group")
    else:
        leaf = group.get("sidecar_leaf_id")
        stable_ids = group.get("stable_ids")
        sequence = group.get("g1b_capacity_two_sequence")
        if not isinstance(leaf, int) or leaf < 0 or not isinstance(stable_ids, list) or \
                len(stable_ids) < required_same_leaf or len(set(stable_ids)) != len(stable_ids) or \
                not all(isinstance(stable_id, int) for stable_id in stable_ids):
            fail(errors, "ready_selection_bad_same_leaf_group")
        else:
            by_id = {row.get("stable_id"): row for row in direct_candidates if isinstance(row, dict)}
            for stable_id in stable_ids[:required_same_leaf]:
                row = by_id.get(stable_id)
                if not isinstance(row, dict) or row.get("sidecar_leaf_id") != leaf:
                    fail(errors, "ready_selection_same_leaf_group_receipt_mismatch")
        expected_first_two = stable_ids[:2] if isinstance(stable_ids, list) else []
        expected_third = stable_ids[2] if isinstance(stable_ids, list) and len(stable_ids) >= 3 else None
        if not isinstance(sequence, dict) or sequence.get("leaf_capacity") != 2 or \
                sequence.get("insert_stable_ids_first") != expected_first_two or \
                sequence.get("third_candidate_stable_id") != expected_third or \
                sequence.get("result_claimed_here") is not False:
            fail(errors, "ready_selection_bad_g1b_capacity_two_sequence")
    oracle = selection_doc.get("independent_full_active_exact_topk_oracle")
    if not isinstance(oracle, dict) or oracle.get("status") != "PASS" or \
            oracle.get("selected_trace_only") is not True or \
            oracle.get("global_canonical_distance_stable_id_proof") is not False or \
            not isinstance(oracle.get("queries_checked"), int) or oracle["queries_checked"] <= 0 or \
            oracle.get("topk_id_order_mismatch_count") != 0 or \
            oracle.get("topk_distance_mismatch_count") != 0 or \
            oracle.get("unique_candidate_top1_mismatch_count") != 0 or \
            oracle.get("query_payload_mismatch_count") != 0:
        fail(errors, "ready_selection_full_active_exact_topk_oracle_missing_or_failed")
    layout = oracle.get("stable_id_layout") if isinstance(oracle, dict) else None
    if not isinstance(layout, dict) or \
            layout.get("mode") != CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE or \
            layout.get("mapping_identity_verified") is not True or \
            layout.get("initial_base_identity_verified") is not True:
        fail(errors, "ready_selection_oracle_runner_identity_layout_missing_or_failed")
    if selection_doc.get("certificate_delta_rejections_policy") != "record_only_not_selection_gate":
        fail(errors, "ready_selection_bad_certificate_delta_policy")
    if not isinstance(selection_doc.get("certificate_delta_rejection_count"), int) or \
            selection_doc["certificate_delta_rejection_count"] < 0:
        fail(errors, "ready_selection_bad_certificate_delta_count")
    input_files = selection_doc.get("bundle_input_files_sha256")
    if not isinstance(input_files, dict):
        fail(errors, "ready_selection_missing_input_file_binding")
    elif isinstance(files, dict):
        expected_input = {name: value for name, value in files.items() if name != artifact}
        if input_files != expected_input:
            fail(errors, "ready_selection_input_file_binding_mismatch")
        oracle_inputs = oracle.get("input_sha256") if isinstance(oracle, dict) else None
        required_oracle_inputs = {
            name: files.get(name)
            for name in (
                "pool.i16", "queries.i16", "trace.e1gtrc",
                "stable_id_to_pool_row.i32", "initial_base_stable_ids.i32",
            )
        }
        if not isinstance(oracle_inputs, dict) or any(
            not isinstance(value, str) or oracle_inputs.get(name) != value
            for name, value in required_oracle_inputs.items()
        ):
            fail(errors, "ready_selection_oracle_input_hash_binding_mismatch")
    return not errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=("witness", "certificate-selection"), default="witness")
    parser.add_argument(
        "--allow-pending-selection",
        action="store_true",
        help="CPU test only in witness mode: validate a generic pending bundle but never authorize execution",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    canonical_root = Path(__file__).resolve().parents[1]
    bundle = args.bundle.resolve()
    out = args.out.resolve()
    errors: list[str] = []
    if root != canonical_root:
        fail(errors, "root_must_match_this_v2_preflight_source_root")
    try:
        bundle.relative_to(root / "bundles")
    except ValueError:
        fail(errors, "bundle_outside_v2_bundles_root")

    manifest_path = bundle / "manifest.json"
    manifest: dict[str, Any] = {}
    if not manifest_path.is_file():
        fail(errors, "missing_manifest")
    else:
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("not_object")
            manifest = raw
        except (json.JSONDecodeError, ValueError) as exc:
            fail(errors, f"bad_manifest_json:{exc}")

    if manifest.get("schema") != "safe-c1-g1-v2-tie-free-bundle-manifest":
        fail(errors, "wrong_or_v1_bundle_schema")
    if manifest.get("engine_schema_required") != "safe-c1-g1-topk-v2-search-native":
        fail(errors, "wrong_engine_schema_required")
    if manifest.get("gpu_used") is not False or manifest.get("cuda_binary_executed") is not False:
        fail(errors, "bundle_manifest_claims_gpu_or_cuda_execution")
    if manifest.get("stable_id_layout") != CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE:
        fail(errors, "bundle_manifest_wrong_stable_id_layout")
    if manifest.get("certificate_contract") != CERTIFICATE:
        fail(errors, "certificate_contract_mismatch")

    files = manifest.get("files_sha256")
    if not isinstance(files, dict):
        fail(errors, "bad_files_sha256")
    else:
        required = {
            "pool.i16", "queries.i16", "trace.e1gtrc", "bundle_spec.json", "tie_free_preflight.json",
            "stable_id_to_pool_row.i32", "initial_base_stable_ids.i32",
        }
        if not required <= set(files):
            fail(errors, "missing_required_bundle_hashes")
        for name, expected in files.items():
            path = bundle / name
            if not isinstance(name, str) or not isinstance(expected, str) or not path.is_file():
                fail(errors, f"missing_or_invalid_hashed_file:{name}")
            elif sha256(path) != expected:
                fail(errors, f"bundle_sha_mismatch:{name}")

    recomputed: dict[str, Any] | None = None
    header = None
    events = None
    current_runner_identity_stable_id_layout_validated = False
    try:
        header, events = parse_trace(bundle / "trace.e1gtrc")
        pool = load_i16(bundle / "pool.i16", header.pool_n * header.dimension)
        queries = load_i16(bundle / "queries.i16", header.query_n * header.dimension)
        load_current_runner_identity_stable_id_layout(bundle, header)
        current_runner_identity_stable_id_layout_validated = True
        recomputed = validate_tie_free_witness(pool, queries, header, events)
        expected_header = {
            "magic": "E1GTRC01", "version": 1, "dimension": header.dimension,
            "base_n": header.base_n, "reservoir_n": header.reservoir_n,
            "pool_n": header.pool_n, "query_n": header.query_n, "k": header.k,
            "event_count": header.event_count, "radius": header.radius,
        }
        if manifest.get("header") != expected_header:
            fail(errors, "manifest_trace_header_mismatch")
        declared_preflight = json.loads((bundle / "tie_free_preflight.json").read_text(encoding="utf-8"))
        for key in (
            "schema", "status", "gpu_used", "cuda_binary_executed",
            "static_base_queries_checked", "dynamic_k_boundary_queries_checked",
            "static_base_policy", "dynamic_k_boundary_policy", "on_violation",
            "global_canonical_distance_stable_id_proof",
        ):
            if declared_preflight.get(key) != recomputed.get(key):
                fail(errors, f"tie_preflight_mismatch:{key}")
        domain = manifest.get("tie_free_witness_domain")
        if not isinstance(domain, dict):
            fail(errors, "missing_tie_free_witness_domain")
        else:
            for key, value in recomputed.items():
                if domain.get(key) != value:
                    fail(errors, f"manifest_tie_domain_mismatch:{key}")
    except (BundleContractError, OSError, json.JSONDecodeError, ValueError) as exc:
        fail(errors, f"bundle_cpu_revalidation_failed:{exc}")

    selection = manifest.get("candidate_selection")
    ready = False
    selection_ready = False
    if args.mode == "certificate-selection":
        if manifest.get("bundle_kind") != "certificate_selection":
            fail(errors, "wrong_bundle_kind_for_certificate_selection")
        if manifest.get("status") != "PREPARED_CPU_ONLY_READY_FOR_V2_CERTIFICATE_SELECTION_RUN":
            fail(errors, "certificate_selection_bundle_not_fresh_pending")
        contract = validate_selection_contract(bundle, manifest, errors)
        if contract is not None and header is not None and events is not None:
            validate_exact_isolated_selection_trace(header, events, contract, errors)
            selection_ready = not errors
    else:
        if isinstance(selection, dict) and selection.get("status") == "READY_FOR_V2_WITNESS":
            ready = validate_ready_selection(bundle, manifest, errors)
        elif isinstance(selection, dict) and selection.get("status") == "PENDING_V2_CERTIFICATE_SELECTION":
            if not args.allow_pending_selection:
                fail(errors, "candidate_selection_pending_not_execution_ready")
        else:
            fail(errors, "unknown_candidate_selection_status")

    if args.mode == "certificate-selection":
        status = (
            "PASS_CPU_ONLY_CERTIFICATE_SELECTION_BUNDLE_PREFLIGHT"
            if not errors and selection_ready
            else "FAIL_CPU_ONLY_CERTIFICATE_SELECTION_BUNDLE_PREFLIGHT"
        )
        return_ok = not errors and selection_ready
    else:
        status = (
            "PASS_CPU_ONLY_BUNDLE_PREFLIGHT_READY_FOR_V2_GUARDED_RUN"
            if not errors and ready
            else "PASS_CPU_ONLY_BUNDLE_PREFLIGHT_PENDING_SELECTION"
            if not errors and args.allow_pending_selection
            else "FAIL_CPU_ONLY_BUNDLE_PREFLIGHT"
        )
        return_ok = not errors and (ready or args.allow_pending_selection)

    result = {
        "schema": "safe-c1-g1-v2-bundle-preflight",
        "status": status,
        "mode": args.mode,
        "gpu_used": False,
        "cuda_binary_executed": False,
        "execution_ready": bool(ready),
        "certificate_selection_ready": bool(selection_ready),
        "root": str(root),
        "bundle": str(bundle),
        "certificate_contract": CERTIFICATE,
        "tie_free_witness_domain": {
            "domain": TIE_DOMAIN,
            "status": recomputed.get("status") if recomputed else "UNVERIFIED",
            "static_base_policy": STATIC_BASE_POLICY,
            "dynamic_k_boundary_policy": DYNAMIC_K_BOUNDARY_POLICY,
            "on_violation": BOUNDED_TIE_ABORT,
            "global_canonical_distance_stable_id_proof": False,
        },
        "candidate_selection_status": selection.get("status") if isinstance(selection, dict) else None,
        "current_runner_identity_stable_id_layout_validated": current_runner_identity_stable_id_layout_validated,
        "errors": errors,
    }
    write_json(out, result)
    print(json.dumps({"status": status, "errors": len(errors), "gpu_used": False}, sort_keys=True))
    return 0 if return_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
