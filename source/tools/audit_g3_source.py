#!/usr/bin/env python3
"""Static P0 audit for the isolated Safe-C1 G3 source closure.

This tool only reads source text and writes an audit JSON.  It does not invoke
CUDA, a compiler, a subprocess, GPU telemetry, a binary, or a benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from g3_common import sha256_file, write_json


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict")


def require(source: str, tokens: list[str], errors: list[str], prefix: str) -> None:
    for token in tokens:
        if token not in source:
            errors.append(f"{prefix}:{token}")


def body(source: str, start: str, end: str) -> str:
    first = source.find(start)
    if first < 0:
        return ""
    last = source.find(end, first)
    return source[first:last] if last > first else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    production = [root / "src/g3_safe_c1_native_matrix.cu", root / "src/g3_safe_search_v2.cuh"]
    closure = production + [root / "reference/include" / name for name in (
        "tree.cuh", "config.cuh", "file.cuh", "mlp_constant.cuh", "residual_pruning.cuh")]
    for path in closure:
        if not path.is_file():
            errors.append(f"missing_closure_file:{path.relative_to(root)}")
    source = text(production[0]) if production[0].is_file() else ""
    search = text(production[1]) if production[1].is_file() else ""

    require(source, [
        "G3_NATIVE_TOPK_RECEIPT_REUSED_FROM_G1 = true",
        "G3_NATIVE_RANGE_RECEIPT_IMPLEMENTED = false",
        "G3_RANGE_BRANCH_ALIGNED_VECTOR_PREDICATE_MIRROR_IMPLEMENTED = true",
        "G3_NATIVE_FRESH_STABLE_ID_REBUILD_IMPLEMENTED = true",
        "G3_DIRECT_SIDECAR_RUNTIME_GUARD_OPEN = false",
        "G3_DIRECT_CERTIFICATE_KNN_VISIBILITY_ONLY = true",
        "G3_KNN_DISK_MONOTONIC_ASSUMPTION_RUNTIME_UNVERIFIED = true",
        "G3_NATIVE_ENGINE_EXECUTABLE = false",
        "G3_NATIVE_ENGINE_EXECUTED = false",
        "run_gts_base_topk_with_receipt(",
        "run_gts_base_range_with_receipt(",
        "materialize_receipt_leaf_rows(",
        "receipt_leaf_spans(",
        "ReceiptLeafSpan",
        "base_receipt_rows",
        "native_final_res_ids_cross_checked",
        "receipt_before_leaf_materialization",
        "id_list_capacity",
        "LEAF_PAD_SLOTS",
        "immutable_base_stable_ids_sha256",
        "immutable_base_payload_sha256",
        "native_tree_bytes_sha256",
        "assert_partition(",
        "swap_noexcept(",
        "kRebuildDestructive",
        "kFailedStop",
        "kAwaitingPostRebuildOracle",
        "commit_generation_noexcept",
        "enter_failed_stop_noexcept",
        "cudaMemcpyFromSymbol(&device_mode, c_rp_mode",
        "post_rebuild_gate_.mark_query_issued",
        "direct_sidecars_exact_range_filtered = is_range",
        "gts_vector_range_branch_aligned_predicate_mirror_receipt",
        "production_full_live_scan = false",
        "kResidualPruningMode = 0",
    ], errors, "missing_p0_implementation_token")

    # The copied KNN traversal must retain actual receipt placement and the
    # exact sibling predicate / all-include branch it is audited against.
    require(search, [
        "safe_c1_last_visited_leaf_pairs.clear()",
        "mergeLNodeKnn<<<",
        "Safe-C1 instrumentation: materialize only the (query, leaf) pairs",
        "if (nid % TREE_ORDER != 0)",
        "node_list[nid + 1].min_dis",
        "if (update_disk == false && (pnum_level < k || cur_level <= 2))",
        "labelCNode<<<",
    ], errors, "missing_pinned_gts_knn_token")

    certificate = body(source, "[[nodiscard]] StrictCertificate certify_leaf(", "\n};\n\nenum class PlacementKind")
    require(certificate, [
        "all_include_branch_not_relied_on = true",
        "const bool last_child = (child % fanout) == 0",
        "upper = nodes[next_sibling].min_dis",
        "distance > nodes[child].min_dis + kStrictEpsilon",
        "(last_child || distance < upper - kStrictEpsilon)",
        "native_sibling_gap_or_boundary",
        "native_sibling_overlap_or_ambiguous",
        "leaf_is_not_at_final_native_knn_payload_depth",
    ], errors, "missing_knn_certificate_branch_token")
    if "max_distance[child]" in certificate:
        errors.append("certificate_uses_max_distance_as_knn_acceptance_fence")

    snapshot = body(source, "struct FrozenTreeSnapshot {", "\nenum class PlacementKind")
    require(snapshot, [
        "validate_pruning_topology()",
        "frozen GTS parent lacks a full contiguous sibling family",
        "frozen GTS native next-sibling predicate input is unavailable",
        "frozen GTS sibling min_dis bridge is nonmonotone",
        "immutable GTS leaf lid/size exceeds physical padded id_list capacity",
        "immutable native GTS tree/base payload changed outside explicit destructive rebuild",
    ], errors, "missing_snapshot_invariant_token")

    topk = body(source, "inline TraversalReceipt run_gts_base_topk_with_receipt(", "\ninline TraversalReceipt run_gts_base_range_with_receipt(")
    require(topk, [
        "output.receipt_before_leaf_materialization = true",
        "output.id_list_capacity = runtime.id_list_capacity",
        "output.receipt_leaf_spans =",
        "receipt_leaf_spans(runtime, snapshot, output.visited_leaf_ids)",
        "materialize_receipt_leaf_rows(runtime, snapshot, output.visited_leaf_ids)",
        "output.base_results = scan_local_rows_exact(runtime, query_device, receipt_locals)",
        "native top-k res_ids is not attributable to a receipt leaf payload",
        "output.native_final_res_ids_cross_checked = true",
    ], errors, "missing_topk_full_receipt_materialization_token")
    if topk.find("output.receipt_before_leaf_materialization = true") > topk.find("materialize_receipt_leaf_rows("):
        errors.append("topk_receipt_not_emitted_before_leaf_materialization")
    if "scan_local_rows_exact(runtime, query_device, local_result_ids" in topk:
        errors.append("topk_res_ids_still_used_as_candidate_source")

    range_body = body(source, "__global__ void g3_mark_range_level(", "\n__global__ void g3_scan_local_candidates(")
    require(range_body, [
        "const bool native_last_child = (node_id % fanout) == 0",
        "if (!native_last_child)",
        "distance - nodes[next].min_dis",
    ], errors, "missing_range_native_branch_mirror_token")
    if "empty_list[next]" in range_body:
        errors.append("range_mirror_has_non_native_next_empty_exception")
    range_adapter = body(source, "inline TraversalReceipt run_gts_base_range_with_receipt(", "\nstruct RebuildReceipt")
    require(range_adapter, [
        "output.receipt_before_leaf_materialization = true",
        "output.range_predicate_mirror_branch_aligned = true",
        "output.id_list_capacity = runtime.id_list_capacity",
        "output.receipt_leaf_spans =",
        "receipt_leaf_spans(runtime, snapshot, output.visited_leaf_ids)",
        "materialize_receipt_leaf_rows(runtime, snapshot, output.visited_leaf_ids)",
        "gts_vector_range_branch_aligned_predicate_mirror_receipt",
    ], errors, "missing_range_receipt_materialization_token")
    if range_adapter.find("output.receipt_before_leaf_materialization = true") > range_adapter.find("materialize_receipt_leaf_rows("):
        errors.append("range_receipt_not_emitted_before_leaf_materialization")
    if "real_gts_vector_range" in range_adapter or "searchIndexRnnV2(" in source:
        errors.append("range_overclaims_legacy_native_vector_execution")

    serializer = body(source, "inline std::string serialize_query_jsonl_record(", "\ninline std::string serialize_rebuild_jsonl_record(")
    require(serializer, [
        "receipt_before_leaf_materialization",
        "id_list_capacity",
        "receipt_leaf_spans",
        "json_receipt_leaf_spans(&out, row.receipt_leaf_spans)",
        "json_receipt_base_rows(&out, row.base_receipt_rows)",
    ], errors, "missing_receipt_trace_schema_token")

    merge = body(source, "QueryExport merge_query_partitions(", "\n  void require_ready_for_mutation()")
    require(merge, [
        "output.receipt_before_leaf_materialization = base.receipt_before_leaf_materialization",
        "output.id_list_capacity = base.id_list_capacity",
        "output.receipt_leaf_spans = base.receipt_leaf_spans",
        "output.range_receipt_before_leaf_materialization = is_range && base.receipt_before_leaf_materialization",
    ], errors, "missing_query_export_receipt_binding")

    lifecycle = body(source, "class NativeSafeC1Matrix {", "\nclass FreshStableIdRebuild {")
    require(lifecycle, [
        "frozen_.assert_unchanged(runtime_)",
        "state_.assert_partition(frozen_, tree_version_)",
        "SafeC1State draft = state_",
        "state_.swap_noexcept(draft)",
        "release_runtime(&runtime_)",
        "mode_ = EngineMode::kRebuildDestructive",
        "enter_failed_stop_noexcept()",
        "mode_ = EngineMode::kAwaitingPostRebuildOracle",
        "post_rebuild_gate_.verify_first_after_rebuild",
    ], errors, "missing_state_or_failstop_token")
    if "assert_unchanged(runtime_, state_.live_ids())" in source:
        errors.append("frozen_assertion_still_compares_dynamic_live_ids")
    if "dynamic_tiers_cleared_after_atomic_switch" in source or "atomic" in source or "rollback" in source:
        errors.append("obsolete_atomic_or_rollback_rebuild_claim_present")

    forbidden = [
        '#include "incremental_insert.cuh"', '#include "update.cuh"', '#include "update_optimized.cuh"',
        "updateIndexRnn(", "deleteIncrementalInsert(", "exact_live_l2", "active_pool_scan",
        "searchIndexRnnV2(", "legacy direct insert",
    ]
    for token in forbidden:
        if token in source:
            errors.append(f"forbidden_production_token:{token}")
    for path in production:
        if not path.is_file():
            continue
        for line_no, line in enumerate(text(path).splitlines(), 1):
            if "#include" in line and ("../" in line or "/home/" in line or "v2/" in line or "v3/" in line or "v5/" in line):
                errors.append(f"nonisolated_production_include:{path.relative_to(root)}:{line_no}")
    if '#include "tree.cuh"' not in source or '#include "g3_safe_search_v2.cuh"' not in source:
        errors.append("production_target_missing_isolated_GTS_includes")

    closure_hash = hashlib.sha256(
        json.dumps({str(p.relative_to(root)): sha256_file(p) for p in closure if p.is_file()},
                   sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    result = {
        "schema": "safe-c1-g3-source-audit-v4-p0",
        "status": "PASS_STATIC_P0_REPAIR" if not errors else "FAIL_STATIC_P0_REPAIR",
        "gpu_used": False,
        "scope": "static isolated source closure only; no CUDA build/run/performance/correctness result",
        "root": str(root),
        "closure_files": {str(p.relative_to(root)): sha256_file(p) for p in closure if p.is_file()},
        "source_closure_sha256": closure_hash,
        "errors": errors,
        "implemented_source": {
            "topk_real_gts_receipt_all_leaf_payload_rows": True,
            "res_ids_diagnostic_crosscheck_only": True,
            "padded_id_list_capacity_contract": True,
            "physical_receipt_span_trace_schema": True,
            "frozen_base_not_dynamic_live_assertion": True,
            "state_partition_and_draft_commit": True,
            "knn_sibling_min_certificate_fail_close": True,
            "direct_runtime_gate_closed": True,
            "direct_certificate_knn_visibility_only": True,
            "range_branch_aligned_vector_predicate_mirror": True,
            "range_direct_sidecar_exact_filter_source": True,
            "destructive_rebuild_fail_stop": True,
            "post_rebuild_oracle_gate_source": True,
        },
        "not_established": [
            "native runtime correctness", "native KNN exactness", "native range correctness",
            "direct-sidecar range correctness", "runtime tree height", "device residual-mode readback",
            "KNN disk monotonicity", "rebuild runtime behavior", "latency", "throughput",
            "GPU utilization", "performance benefit",
        ],
        "forbidden_claims": [
            "native runtime correctness", "native KNN/range exactness", "direct range correctness",
            "tree height observed at runtime", "latency", "throughput", "GPU utilization",
            "performance benefit",
        ],
    }
    write_json(args.out, result)
    print(json.dumps({"status": result["status"], "gpu_used": False, "errors": len(errors)}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
