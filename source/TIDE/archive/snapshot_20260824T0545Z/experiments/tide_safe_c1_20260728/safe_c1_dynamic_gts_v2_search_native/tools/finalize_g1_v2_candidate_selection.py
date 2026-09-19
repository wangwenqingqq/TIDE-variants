#!/usr/bin/env python3
"""CPU-only finalizer for one new-v2 Safe-C1 certificate-selection run.

It accepts only the new v2 binary/static-audit/bundle schemas.  On a validated
selection run it writes candidate_selection_v2.json and atomically changes that
fresh bundle to READY_FOR_V2_WITNESS only after it finds a same-sidecar-leaf
group of three direct receipts. Certificate-delta rejections are diagnostic
statistics, not a selection hard gate. It never reports timing/throughput and
never accepts v1 selection evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from safe_c1_v2_bundle_contract import (
    BOUNDED_TIE_ABORT,
    BOUNDED_TIE_DOMAIN,
    BOUNDED_TIE_PREFLIGHT_STATUS,
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
)

CERTIFICATE_SCHEMA = "safe-c1-search-native-sibling-boundary-v2"
ENGINE_SCHEMA = "safe-c1-g1-topk-v2-search-native"
STATIC_SCHEMA = "safe-c1-g1-static-contract-audit-v2-search-native-tie-free"
ENGINE_TIE_STATUS = "BOUNDED_TIE_WITNESS_PRECONDITION_SATISFIED"
ENGINE_STATIC_PROBE_STATUS = "BOUNDED_TIE_WITNESS_VALIDATED"
ENGINE_SUMMARY_STATUS = "PASS_G1_BOUNDED_TIE_WITNESS_PENDING_INDEPENDENT_VALIDATOR"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def load_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("not_object")
        return value
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"bad_{label}:{exc}")
        return {}


def load_jsonl(path: Path, errors: list[str]) -> tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]:
    meta: dict[str, Any] | None = None
    by_op: dict[int, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"cannot_read_engine_jsonl:{exc}")
        return meta, by_op
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"bad_engine_jsonl_line:{line_number}:{exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"nonobject_engine_jsonl_line:{line_number}")
            continue
        if row.get("record") == "meta":
            if meta is not None:
                errors.append("duplicate_engine_meta")
            meta = row
            continue
        op_index = row.get("op_index")
        if not isinstance(op_index, int) or op_index in by_op:
            errors.append(f"bad_or_duplicate_engine_op_index:{line_number}")
            continue
        by_op[op_index] = row
    if meta is None:
        errors.append("missing_engine_meta")
    return meta, by_op


def in_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_exact_isolated_selection_trace(
    bundle: Path, candidates: object, errors: list[str]
) -> bool:
    """Require exactly one insert/query/delete triplet per candidate.

    This is what makes a delta receipt in the leaf-capacity-one selection run a
    certificate fallback statistic rather than a capacity fallback. The later
    G1B capacity-two run has a separate contract for capacity evidence.
    """
    before = len(errors)
    if not isinstance(candidates, list):
        errors.append("selection_trace_missing_candidates")
        return False
    try:
        header, events = parse_trace(bundle / "trace.e1gtrc")
    except (BundleContractError, OSError) as exc:
        errors.append(f"selection_trace_parse_failed:{exc}")
        return False
    expected_event_count = 3 * len(candidates)
    if header.event_count != expected_event_count or len(events) != expected_event_count:
        errors.append("selection_trace_not_exactly_isolated")
    if header.query_n != len(candidates):
        errors.append("selection_trace_query_count_mismatch")
    for index, row in enumerate(candidates):
        if not isinstance(row, dict) or not all(
            isinstance(row.get(key), int)
            for key in ("stable_id", "query_id", "insert_op_index", "query_op_index", "delete_op_index")
        ):
            errors.append(f"selection_trace_bad_contract_row:{index}")
            continue
        first = 3 * index
        if (row["query_id"] != index or row["insert_op_index"] != first or
                row["query_op_index"] != first + 1 or row["delete_op_index"] != first + 2):
            errors.append(f"selection_trace_contract_not_sequential:{index}")
            continue
        if first + 2 >= len(events):
            errors.append(f"selection_trace_triplet_missing:{index}")
            continue
        insert, query, delete = events[first:first + 3]
        if insert.op != OP_INSERT or insert.argument != row["stable_id"] or \
                query.op != OP_KNN or query.argument != row["query_id"] or \
                delete.op != OP_DELETE or delete.argument != row["stable_id"]:
            errors.append(f"selection_trace_triplet_mismatch:{index}")
    return len(errors) == before


def _oracle_add(oracle: dict[str, Any], errors: list[str], message: str) -> None:
    oracle["errors"].append(message)
    errors.append(message)


def _record_oracle_example(
    examples: list[dict[str, Any]], value: dict[str, Any], limit: int = 5
) -> None:
    if len(examples) < limit:
        examples.append(value)


def _distance_matches_exact_squared_l2(observed: float, squared_l2: int) -> bool:
    """Accept only documented JSON precision loss, not a different metric value."""
    expected = math.sqrt(squared_l2)
    return math.isclose(
        observed,
        expected,
        rel_tol=1.0e-8,
        abs_tol=2.0e-6,
    )


def validate_independent_full_active_exact_topk(
    bundle: Path, by_op: dict[int, dict[str, Any]], errors: list[str]
) -> dict[str, Any]:
    """Independently replay every selected trace query over its full active set.

    The check is CPU-only: Python integers calculate exact quantized squared-L2
    for all active IDs, while an explicit identity stable-ID layout is enforced
    because the current v2 executable itself addresses ``pool[stable_id]``.
    It validates the *selected trace* only. In particular, deterministic
    ``(squared_l2, stable_id)`` ordering here is not claimed as an all-input
    canonical tie proof.
    """
    oracle: dict[str, Any] = {
        "schema": "safe-c1-g1-v2-independent-full-active-exact-topk-oracle",
        "method": (
            "CPU Python-integer replay of every selected trace kNN active set; exact quantized "
            "squared-L2 top-k IDs/order and reported-L2 checks under the explicit identity-only "
            "stable-ID layout required by the current v2 runner"
        ),
        "selected_trace_only": True,
        "global_canonical_distance_stable_id_proof": False,
        "queries_checked": 0,
        "topk_id_order_mismatch_count": 0,
        "topk_membership_mismatch_count": 0,
        "topk_distance_mismatch_count": 0,
        "malformed_engine_result_count": 0,
        "dynamic_k_boundary_tie_count": 0,
        "query_payload_mismatch_count": 0,
        "unique_candidate_top1_mismatch_count": 0,
        "trace_replay_error_count": 0,
        "input_sha256": {},
        "stable_id_layout": None,
        "mismatch_examples": [],
        "errors": [],
    }
    try:
        header, events = parse_trace(bundle / "trace.e1gtrc")
        pool = load_i16(bundle / "pool.i16", header.pool_n * header.dimension)
        queries = load_i16(bundle / "queries.i16", header.query_n * header.dimension)
        stable_to_row, initial_base_ids = load_current_runner_identity_stable_id_layout(bundle, header)
    except (BundleContractError, OSError) as exc:
        _oracle_add(oracle, errors, f"oracle_bundle_or_identity_layout_parse_failed:{exc}")
        oracle["status"] = "FAIL"
        return oracle

    mapping_path = bundle / "stable_id_to_pool_row.i32"
    base_ids_path = bundle / "initial_base_stable_ids.i32"
    oracle["input_sha256"] = {
        "pool.i16": sha256(bundle / "pool.i16"),
        "queries.i16": sha256(bundle / "queries.i16"),
        "trace.e1gtrc": sha256(bundle / "trace.e1gtrc"),
        "stable_id_to_pool_row.i32": sha256(mapping_path),
        "initial_base_stable_ids.i32": sha256(base_ids_path),
    }
    oracle["stable_id_layout"] = {
        "mode": CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
        "mapping_identity_verified": True,
        "initial_base_identity_verified": True,
        "stable_id_to_pool_row_entries": len(stable_to_row),
        "initial_base_stable_id_entries": len(initial_base_ids),
    }

    base_ids = set(initial_base_ids)
    active: set[int] = set(initial_base_ids)
    mismatch_examples: list[dict[str, Any]] = []
    id_order_mismatches = 0
    membership_mismatches = 0
    distance_mismatches = 0
    malformed_results = 0
    boundary_ties = 0
    query_payload_mismatches = 0
    unique_candidate_top1_mismatches = 0
    replay_errors = 0

    def exact_squared_l2_for(stable_id: int, query_id: int) -> int:
        row = stable_to_row[stable_id]
        point_offset = row * header.dimension
        query_offset = query_id * header.dimension
        return sum(
            (int(pool[point_offset + axis]) - int(queries[query_offset + axis])) ** 2
            for axis in range(header.dimension)
        )

    def query_matches_candidate_payload(stable_id: int, query_id: int) -> bool:
        row = stable_to_row[stable_id]
        point_offset = row * header.dimension
        query_offset = query_id * header.dimension
        return all(
            pool[point_offset + axis] == queries[query_offset + axis]
            for axis in range(header.dimension)
        )

    for event in events:
        argument = event.argument
        if event.op == OP_INSERT:
            if argument < header.base_n or argument >= header.pool_n or argument in active:
                replay_errors += 1
                _record_oracle_example(mismatch_examples, {
                    "op_index": event.op_index, "issue": "invalid_insert", "stable_id": argument,
                })
            else:
                active.add(argument)
            continue
        if event.op == OP_DELETE:
            if argument < header.base_n or argument >= header.pool_n or argument not in active:
                replay_errors += 1
                _record_oracle_example(mismatch_examples, {
                    "op_index": event.op_index, "issue": "invalid_delete", "stable_id": argument,
                })
            else:
                active.remove(argument)
            continue
        if event.op != OP_KNN:
            replay_errors += 1
            _record_oracle_example(mismatch_examples, {
                "op_index": event.op_index, "issue": "non_knn_trace_event", "op": event.op,
            })
            continue

        query_id = argument
        if query_id < 0 or query_id >= header.query_n or len(active) < header.k:
            replay_errors += 1
            _record_oracle_example(mismatch_examples, {
                "op_index": event.op_index,
                "issue": "invalid_query_or_active_count",
                "query_id": query_id,
                "active_count": len(active),
            })
            continue

        dynamic_active = sorted(active - base_ids)
        if len(dynamic_active) != 1:
            replay_errors += 1
            _record_oracle_example(mismatch_examples, {
                "op_index": event.op_index,
                "issue": "selection_query_not_single_active_reservoir_object",
                "dynamic_active_ids": dynamic_active[:8],
                "dynamic_active_count": len(dynamic_active),
            })
        else:
            candidate_id = dynamic_active[0]
            if not query_matches_candidate_payload(candidate_id, query_id):
                query_payload_mismatches += 1
                _record_oracle_example(mismatch_examples, {
                    "op_index": event.op_index,
                    "query_id": query_id,
                    "issue": "query_payload_not_equal_to_inserted_candidate_mapped_pool_row",
                    "candidate_stable_id": candidate_id,
                })

        ranked = sorted(
            (exact_squared_l2_for(candidate_id, query_id), candidate_id)
            for candidate_id in active
        )
        oracle["queries_checked"] += 1
        if len(ranked) > header.k and ranked[header.k - 1][0] == ranked[header.k][0]:
            boundary_ties += 1
            _record_oracle_example(mismatch_examples, {
                "op_index": event.op_index,
                "query_id": query_id,
                "issue": "dynamic_k_boundary_tie",
                "squared_l2": ranked[header.k - 1][0],
                "ids": [ranked[header.k - 1][1], ranked[header.k][1]],
            })
        if len(dynamic_active) == 1:
            candidate_id = dynamic_active[0]
            if ranked[0][1] != candidate_id or (
                len(ranked) > 1 and ranked[0][0] == ranked[1][0]
            ):
                unique_candidate_top1_mismatches += 1
                _record_oracle_example(mismatch_examples, {
                    "op_index": event.op_index,
                    "query_id": query_id,
                    "issue": "inserted_candidate_not_unique_exact_top1",
                    "candidate_stable_id": candidate_id,
                    "exact_top1": [ranked[0][1], ranked[0][0]],
                    "exact_top2": [ranked[1][1], ranked[1][0]] if len(ranked) > 1 else None,
                })

        expected_ids = [candidate_id for _, candidate_id in ranked[:header.k]]
        receipt = by_op.get(event.op_index)
        observed_ids: list[int] | None = None
        observed_distances: list[float] | None = None
        if not isinstance(receipt, dict) or receipt.get("record") != "query" or \
                receipt.get("kind") != "knn" or receipt.get("query_id") != query_id:
            malformed_results += 1
            _record_oracle_example(mismatch_examples, {
                "op_index": event.op_index, "issue": "missing_or_malformed_query_receipt",
            })
        else:
            results = receipt.get("results")
            if not isinstance(results, list) or len(results) != header.k:
                malformed_results += 1
                _record_oracle_example(mismatch_examples, {
                    "op_index": event.op_index, "issue": "wrong_result_count",
                    "expected_k": header.k,
                    "actual": len(results) if isinstance(results, list) else None,
                })
            else:
                parsed_ids: list[int] = []
                parsed_distances: list[float] = []
                valid_shape = True
                for row in results:
                    if not isinstance(row, list) or len(row) != 2 or type(row[0]) is not int or \
                            isinstance(row[1], bool) or not isinstance(row[1], (int, float)) or \
                            not math.isfinite(float(row[1])):
                        valid_shape = False
                        break
                    parsed_ids.append(row[0])
                    parsed_distances.append(float(row[1]))
                if not valid_shape or len(set(parsed_ids)) != len(parsed_ids) or \
                        any(stable_id not in active for stable_id in parsed_ids):
                    malformed_results += 1
                    _record_oracle_example(mismatch_examples, {
                        "op_index": event.op_index, "issue": "invalid_duplicate_or_inactive_result_id",
                    })
                else:
                    observed_ids = parsed_ids
                    observed_distances = parsed_distances

        if observed_ids is not None and observed_distances is not None:
            if observed_ids != expected_ids:
                id_order_mismatches += 1
                if set(observed_ids) != set(expected_ids):
                    membership_mismatches += 1
                _record_oracle_example(mismatch_examples, {
                    "op_index": event.op_index,
                    "query_id": query_id,
                    "issue": "full_active_exact_topk_id_order_mismatch",
                    "expected_ids": expected_ids,
                    "observed_ids": observed_ids,
                })
            for stable_id, observed_distance in zip(observed_ids, observed_distances):
                if not _distance_matches_exact_squared_l2(
                    observed_distance, exact_squared_l2_for(stable_id, query_id)
                ):
                    distance_mismatches += 1
                    _record_oracle_example(mismatch_examples, {
                        "op_index": event.op_index,
                        "query_id": query_id,
                        "issue": "reported_distance_not_exact_l2_for_reported_stable_id",
                        "stable_id": stable_id,
                        "observed_distance": observed_distance,
                        "expected_distance": math.sqrt(exact_squared_l2_for(stable_id, query_id)),
                    })

    if active != base_ids:
        replay_errors += 1
        _record_oracle_example(mismatch_examples, {
            "issue": "final_active_set_mismatch",
            "expected_base_active_count": len(base_ids),
            "actual_active_count": len(active),
        })

    oracle["topk_id_order_mismatch_count"] = id_order_mismatches
    oracle["topk_membership_mismatch_count"] = membership_mismatches
    oracle["topk_distance_mismatch_count"] = distance_mismatches
    oracle["malformed_engine_result_count"] = malformed_results
    oracle["dynamic_k_boundary_tie_count"] = boundary_ties
    oracle["query_payload_mismatch_count"] = query_payload_mismatches
    oracle["unique_candidate_top1_mismatch_count"] = unique_candidate_top1_mismatches
    oracle["trace_replay_error_count"] = replay_errors
    oracle["mismatch_examples"] = mismatch_examples
    if boundary_ties:
        _oracle_add(oracle, errors, f"full_active_exact_topk_dynamic_k_boundary_tie:count={boundary_ties}")
    if malformed_results:
        _oracle_add(oracle, errors, f"full_active_exact_topk_malformed_engine_results:count={malformed_results}")
    if id_order_mismatches:
        _oracle_add(oracle, errors, f"full_active_exact_topk_id_order_mismatch:count={id_order_mismatches}")
    if membership_mismatches:
        _oracle_add(oracle, errors, f"full_active_exact_topk_membership_mismatch:count={membership_mismatches}")
    if distance_mismatches:
        _oracle_add(oracle, errors, f"full_active_exact_topk_distance_mismatch:count={distance_mismatches}")
    if query_payload_mismatches:
        _oracle_add(oracle, errors, f"full_active_exact_topk_query_payload_mismatch:count={query_payload_mismatches}")
    if unique_candidate_top1_mismatches:
        _oracle_add(oracle, errors, f"full_active_exact_topk_unique_candidate_top1_mismatch:count={unique_candidate_top1_mismatches}")
    if replay_errors:
        _oracle_add(oracle, errors, f"full_active_exact_topk_trace_replay_error:count={replay_errors}")
    oracle["status"] = "PASS" if not oracle["errors"] else "FAIL"
    return oracle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--engine-jsonl", required=True, type=Path)
    parser.add_argument("--engine-summary", required=True, type=Path)
    parser.add_argument("--static-contract", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    canonical_root = Path(__file__).resolve().parents[1]
    bundle = args.bundle.resolve()
    run_root = args.run_root.resolve()
    engine_jsonl = args.engine_jsonl.resolve()
    engine_summary = args.engine_summary.resolve()
    static_contract = args.static_contract.resolve()
    binary = args.binary.resolve()
    out = args.out.resolve()
    errors: list[str] = []

    if root != canonical_root:
        errors.append("root_must_match_this_v2_finalizer_source_root")
    if not in_root(bundle, root / "bundles"):
        errors.append("bundle_outside_v2_bundles_root")
    if not in_root(run_root, root.parent / "runs"):
        errors.append("run_root_outside_new_v2_runs_root")
    if out.parent != bundle or out.name != "candidate_selection_v2.json":
        errors.append("candidate_selection_output_must_be_new_bundle_local_file")
    if out.exists():
        errors.append("candidate_selection_output_already_exists")
    for label, path in (("engine_jsonl", engine_jsonl), ("engine_summary", engine_summary),
                        ("static_contract", static_contract), ("binary", binary)):
        if not path.is_file():
            errors.append(f"missing_{label}")

    manifest_path = bundle / "manifest.json"
    manifest = load_json(manifest_path, errors, "bundle_manifest")
    manifest_before_sha = sha256(manifest_path) if manifest_path.is_file() else None
    if manifest.get("schema") != "safe-c1-g1-v2-tie-free-bundle-manifest":
        errors.append("wrong_or_v1_bundle_schema")
    if manifest.get("bundle_kind") != "certificate_selection":
        errors.append("wrong_bundle_kind")
    if manifest.get("status") != "PREPARED_CPU_ONLY_READY_FOR_V2_CERTIFICATE_SELECTION_RUN":
        errors.append("bundle_not_fresh_for_certificate_selection")
    selection_contract_binding = manifest.get("candidate_selection")
    if not isinstance(selection_contract_binding, dict) or \
            selection_contract_binding.get("status") != "PENDING_V2_CERTIFICATE_SELECTION":
        errors.append("bundle_candidate_selection_not_pending")
    if manifest.get("engine_schema_required") != ENGINE_SCHEMA:
        errors.append("bundle_wrong_engine_schema")
    if manifest.get("certificate_contract", {}).get("schema") != CERTIFICATE_SCHEMA:
        errors.append("bundle_wrong_certificate_schema")
    manifest_tie = manifest.get("tie_free_witness_domain")
    if not isinstance(manifest_tie, dict) or \
            manifest_tie.get("status") != BOUNDED_TIE_PREFLIGHT_STATUS or \
            manifest_tie.get("static_base_policy") != STATIC_BASE_POLICY or \
            manifest_tie.get("dynamic_k_boundary_policy") != DYNAMIC_K_BOUNDARY_POLICY or \
            manifest_tie.get("on_violation") != BOUNDED_TIE_ABORT:
        errors.append("bundle_bounded_tie_preflight_mismatch")
    input_files = manifest.get("files_sha256")
    if not isinstance(input_files, dict) or not input_files:
        errors.append("bundle_missing_input_file_hashes")
        input_files = {}
    else:
        for name, expected_sha in input_files.items():
            candidate_path = (bundle / name).resolve() if isinstance(name, str) else bundle
            if not isinstance(name, str) or Path(name).name != name or not in_root(candidate_path, bundle) or \
                    not candidate_path.is_file() or not isinstance(expected_sha, str) or sha256(candidate_path) != expected_sha:
                errors.append(f"bundle_input_file_sha_mismatch:{name}")

    selection_contract: dict[str, Any] = {}
    if isinstance(selection_contract_binding, dict):
        contract_rel = selection_contract_binding.get("selection_contract")
        contract_sha = selection_contract_binding.get("selection_contract_sha256")
        if isinstance(contract_rel, str) and isinstance(contract_sha, str):
            contract_path = (bundle / contract_rel).resolve()
            if not in_root(contract_path, bundle) or not contract_path.is_file() or sha256(contract_path) != contract_sha:
                errors.append("selection_contract_binding_mismatch")
            else:
                selection_contract = load_json(contract_path, errors, "selection_contract")
        else:
            errors.append("selection_contract_binding_missing")
    if selection_contract.get("schema") != "safe-c1-g1-v2-certificate-selection-contract":
        errors.append("selection_contract_wrong_schema")
    if selection_contract.get("engine_schema") != ENGINE_SCHEMA:
        errors.append("selection_contract_wrong_engine_schema")
    if selection_contract.get("certificate_schema") != CERTIFICATE_SCHEMA:
        errors.append("selection_contract_wrong_certificate_schema")
    if selection_contract.get("tie_domain") != BOUNDED_TIE_DOMAIN:
        errors.append("selection_contract_wrong_tie_domain")
    if selection_contract.get("static_base_oracle_policy") != STATIC_BASE_POLICY:
        errors.append("selection_contract_wrong_static_base_policy")
    if selection_contract.get("dynamic_k_boundary_policy") != DYNAMIC_K_BOUNDARY_POLICY:
        errors.append("selection_contract_wrong_dynamic_boundary_policy")
    if selection_contract.get("v1_evidence_used") is not False:
        errors.append("selection_contract_v1_evidence_not_rejected")
    if selection_contract.get("stable_id_layout") != CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE:
        errors.append("selection_contract_wrong_stable_id_layout")
    if selection_contract.get("leaf_capacity") != 1:
        errors.append("selection_contract_leaf_capacity_not_one")
    min_direct = selection_contract.get("min_direct_candidates")
    if not isinstance(min_direct, int) or min_direct < 3:
        errors.append("selection_contract_bad_min_direct")
        min_direct = 3
    min_same_leaf = selection_contract.get("min_same_sidecar_leaf_direct_candidates")
    if not isinstance(min_same_leaf, int) or min_same_leaf < 3:
        errors.append("selection_contract_bad_min_same_leaf_direct")
        min_same_leaf = 3
    if min_same_leaf > min_direct:
        errors.append("selection_contract_same_leaf_gate_exceeds_direct_gate")
    if selection_contract.get("certificate_delta_rejections_policy") != "record_only_not_selection_gate":
        errors.append("selection_contract_bad_certificate_delta_policy")
    followup = selection_contract.get("g1b_followup")
    if not isinstance(followup, dict) or followup.get("leaf_capacity") != 2 or \
            followup.get("same_leaf_group_size") != 3:
        errors.append("selection_contract_bad_g1b_followup")
    expected_candidates = selection_contract.get("candidates")
    if not isinstance(expected_candidates, list) or not expected_candidates:
        errors.append("selection_contract_missing_candidates")
        expected_candidates = []
    selection_trace_isolated = validate_exact_isolated_selection_trace(bundle, expected_candidates, errors)

    static = load_json(static_contract, errors, "static_contract")
    binary_sha = sha256(binary) if binary.is_file() else None
    if static.get("schema") != STATIC_SCHEMA or static.get("status") != "PASS_CPU_ONLY_STATIC_CONTRACT":
        errors.append("static_contract_not_valid_v2")
    if static.get("gpu_used") is not False or static.get("cuda_binary_executed") is not False:
        errors.append("static_contract_not_cpu_only")
    static_files = static.get("files")
    if not isinstance(static_files, dict) or binary_sha is None or \
            static_files.get("bin/GTS_safe_c1_g1_v2_search_native") != binary_sha:
        errors.append("static_contract_binary_sha_mismatch")

    summary = load_json(engine_summary, errors, "engine_summary")
    if summary.get("schema") != ENGINE_SCHEMA:
        errors.append("engine_summary_wrong_schema")
    if summary.get("status") != ENGINE_SUMMARY_STATUS:
        errors.append("engine_summary_not_bounded_tie_pass")
    summary_tie = summary.get("tie_semantics")
    if not isinstance(summary_tie, dict) or \
            summary_tie.get("domain") != BOUNDED_TIE_DOMAIN or \
            summary_tie.get("status") != ENGINE_TIE_STATUS or \
            summary_tie.get("static_base_oracle_policy") != STATIC_BASE_POLICY or \
            summary_tie.get("dynamic_k_boundary_policy") != DYNAMIC_K_BOUNDARY_POLICY or \
            summary_tie.get("on_violation") != BOUNDED_TIE_ABORT or \
            summary_tie.get("global_canonical_distance_stable_id_proof") is not False:
        errors.append("engine_summary_tie_contract_mismatch")
    summary_layout = summary.get("stable_id_layout")
    if not isinstance(summary_layout, dict) or \
            summary_layout.get("mode") != CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE or \
            summary_layout.get("validated") is not True:
        errors.append("engine_summary_stable_id_layout_mismatch")

    meta, by_op = load_jsonl(engine_jsonl, errors)
    if meta is not None:
        if meta.get("schema") != ENGINE_SCHEMA or meta.get("gts_static_probe") != ENGINE_STATIC_PROBE_STATUS:
            errors.append("engine_meta_wrong_v2_schema_or_probe")
        meta_tie = meta.get("tie_semantics")
        if not isinstance(meta_tie, dict) or \
                meta_tie.get("domain") != BOUNDED_TIE_DOMAIN or \
                meta_tie.get("status") != ENGINE_TIE_STATUS or \
                meta_tie.get("static_base_oracle_policy") != STATIC_BASE_POLICY or \
                meta_tie.get("dynamic_k_boundary_policy") != DYNAMIC_K_BOUNDARY_POLICY or \
                meta_tie.get("on_violation") != BOUNDED_TIE_ABORT or \
                meta_tie.get("global_canonical_distance_stable_id_proof") is not False:
            errors.append("engine_meta_tie_contract_mismatch")
        meta_layout = meta.get("stable_id_layout")
        if not isinstance(meta_layout, dict) or \
                meta_layout.get("mode") != CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE or \
                meta_layout.get("validated") is not True:
            errors.append("engine_meta_stable_id_layout_mismatch")

    full_active_exact_topk_oracle = validate_independent_full_active_exact_topk(bundle, by_op, errors)

    direct_candidates: list[dict[str, Any]] = []
    delta_rejections: list[dict[str, Any]] = []
    seen_expected_ids: set[int] = set()
    for index, expected in enumerate(expected_candidates):
        fields = ("stable_id", "query_id", "insert_op_index", "query_op_index", "delete_op_index")
        if not isinstance(expected, dict) or not all(isinstance(expected.get(key), int) for key in fields):
            errors.append(f"bad_expected_candidate:{index}")
            continue
        stable_id = expected["stable_id"]
        if stable_id in seen_expected_ids:
            errors.append(f"duplicate_expected_candidate_stable_id:{index}")
            continue
        seen_expected_ids.add(stable_id)
        insert = by_op.get(expected["insert_op_index"])
        query = by_op.get(expected["query_op_index"])
        delete = by_op.get(expected["delete_op_index"])
        if not isinstance(insert, dict) or insert.get("record") != "update" or insert.get("op") != "insert" or \
                insert.get("stable_id") != stable_id or insert.get("placement") not in {"direct", "delta"} or \
                not isinstance(insert.get("sidecar_leaf_id"), int):
            errors.append(f"bad_insert_receipt:{index}")
            continue
        if not isinstance(query, dict) or query.get("record") != "query" or query.get("kind") != "knn" or \
                query.get("query_id") != expected["query_id"]:
            errors.append(f"bad_query_receipt:{index}")
            continue
        results = query.get("results")
        if not isinstance(results, list) or not results or not isinstance(results[0], list) or \
                len(results[0]) != 2 or results[0][0] != stable_id:
            errors.append(f"candidate_not_top1:{index}")
            continue
        if not isinstance(delete, dict) or delete.get("record") != "update" or delete.get("op") != "delete" or \
                delete.get("stable_id") != stable_id or delete.get("placement") != "deleted":
            errors.append(f"bad_delete_receipt:{index}")
            continue
        placement = insert["placement"]
        leaf = insert["sidecar_leaf_id"]
        if placement == "direct":
            visited = query.get("gts_visited_leaf_ids")
            if leaf < 0 or not isinstance(visited, list) or leaf not in visited or \
                    not isinstance(query.get("sidecar_candidate_count"), int) or query["sidecar_candidate_count"] < 1:
                errors.append(f"direct_candidate_visibility_not_proven:{index}")
                continue
            direct_candidates.append({
                "stable_id": stable_id,
                "sidecar_leaf_id": leaf,
                "insert_op_index": expected["insert_op_index"],
                "query_op_index": expected["query_op_index"],
                "delete_op_index": expected["delete_op_index"],
                "gts_visited_leaf_ids": visited,
            })
        else:
            if leaf != -1 or not isinstance(query.get("delta_candidate_count"), int) or query["delta_candidate_count"] < 1:
                errors.append(f"delta_candidate_fallback_not_proven:{index}")
                continue
            delta_rejections.append({
                "stable_id": stable_id,
                "reason": "certificate_reject_to_exact_delta_under_isolated_capacity_one",
            })

    if len(direct_candidates) < min_direct:
        errors.append("insufficient_new_v2_direct_candidates")

    same_leaf_groups: dict[int, list[dict[str, Any]]] = {}
    for candidate in direct_candidates:
        same_leaf_groups.setdefault(int(candidate["sidecar_leaf_id"]), []).append(candidate)
    qualifying_groups = [
        (leaf, sorted(rows, key=lambda row: int(row["stable_id"])))
        for leaf, rows in same_leaf_groups.items()
        if len(rows) >= min_same_leaf
    ]
    qualifying_groups.sort(key=lambda item: (item[0], [int(row["stable_id"]) for row in item[1]]))
    g1b_same_leaf_direct_group: dict[str, Any] | None = None
    if not qualifying_groups:
        errors.append("insufficient_new_v2_same_leaf_direct_group")
    else:
        group_leaf, group_rows = qualifying_groups[0]
        selected_rows = group_rows[:min_same_leaf]
        selected_ids = [int(row["stable_id"]) for row in selected_rows]
        g1b_same_leaf_direct_group = {
            "sidecar_leaf_id": group_leaf,
            "required_group_size": min_same_leaf,
            "stable_ids": selected_ids,
            "direct_receipts": selected_rows,
            "g1b_capacity_two_sequence": {
                "leaf_capacity": 2,
                "insert_stable_ids_first": selected_ids[:2],
                "third_candidate_stable_id": selected_ids[2],
                "validation_goal": (
                    "future G1B must independently validate the third insert as an exact global-delta "
                    "capacity fallback under leaf capacity two; this selection artifact makes no capacity result claim"
                ),
                "result_claimed_here": False,
            },
        }

    result = {
        "schema": "safe-c1-g1-v2-candidate-selection",
        "status": "READY_FOR_V2_WITNESS" if not errors else "FAIL",
        "scope": (
            "new-v2 certificate selection only: direct placement/visibility receipts with exactly isolated leaf-capacity-one triplets plus "
            "an independent CPU full-active-set exact squared-L2 top-k trace replay; no timing, throughput, capacity result, "
            "all-input tie proof, or v1-selection claim"
        ),
        "gpu_used": True if not errors else None,
        "v1_evidence_used": False,
        "engine_schema": ENGINE_SCHEMA,
        "certificate_schema": CERTIFICATE_SCHEMA,
        "tie_domain": BOUNDED_TIE_DOMAIN,
        "static_base_oracle_policy": STATIC_BASE_POLICY,
        "dynamic_k_boundary_policy": DYNAMIC_K_BOUNDARY_POLICY,
        "on_tie_violation": BOUNDED_TIE_ABORT,
        "global_canonical_distance_stable_id_proof": False,
        "stable_id_layout": CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
        "bundle_input_manifest_sha256": manifest_before_sha,
        "bundle_input_files_sha256": input_files,
        "selection_trace_exactly_isolated": selection_trace_isolated,
        "independent_full_active_exact_topk_oracle": full_active_exact_topk_oracle,
        "run_root": str(run_root),
        "binary_sha256": binary_sha,
        "static_contract_schema": static.get("schema"),
        "static_contract_sha256": sha256(static_contract) if static_contract.is_file() else None,
        "engine_results_sha256": sha256(engine_jsonl) if engine_jsonl.is_file() else None,
        "engine_summary_sha256": sha256(engine_summary) if engine_summary.is_file() else None,
        "min_direct_candidates": min_direct,
        "min_same_sidecar_leaf_direct_candidates": min_same_leaf,
        "certificate_delta_rejections_policy": "record_only_not_selection_gate",
        "direct_candidates": direct_candidates,
        "certificate_delta_rejection_count": len(delta_rejections),
        "delta_rejections": delta_rejections,
        "g1b_same_leaf_direct_group": g1b_same_leaf_direct_group,
        "error_count": len(errors),
        "errors": errors,
    }
    write_json(out, result)

    if not errors:
        updated = dict(manifest)
        updated["status"] = "READY_FOR_V2_WITNESS"
        updated["candidate_selection"] = {
            "status": "READY_FOR_V2_WITNESS",
            "artifact": out.name,
            "artifact_sha256": sha256(out),
            "tie_domain": BOUNDED_TIE_DOMAIN,
            "static_base_oracle_policy": STATIC_BASE_POLICY,
            "dynamic_k_boundary_policy": DYNAMIC_K_BOUNDARY_POLICY,
            "on_tie_violation": BOUNDED_TIE_ABORT,
            "stable_id_layout": CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
            "selection_trace_exactly_isolated": True,
            "v1_evidence_used": False,
            "required_schema": "safe-c1-g1-v2-candidate-selection",
            "selection_contract": selection_contract_binding.get("selection_contract"),
            "selection_contract_sha256": selection_contract_binding.get("selection_contract_sha256"),
            "g1b_same_leaf_direct_group": {
                "sidecar_leaf_id": g1b_same_leaf_direct_group["sidecar_leaf_id"],
                "stable_ids": g1b_same_leaf_direct_group["stable_ids"],
            },
        }
        updated_files = dict(input_files)
        updated_files[out.name] = sha256(out)
        updated["files_sha256"] = updated_files
        atomic_write_json(manifest_path, updated)
    print(json.dumps({"status": result["status"], "errors": len(errors), "gpu_used": result["gpu_used"]}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
