#!/usr/bin/env python3
"""Static P0 audit for the isolated Safe-C1 G3 source closure.

This tool only reads source text and writes an audit JSON.  It does not invoke
CUDA, a compiler, a subprocess, GPU telemetry, a binary, or a benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

# Keep this static audit self-contained: g3_common imports NumPy for fixture
# operations, but the source-closure audit needs only deterministic hashing and
# JSON output and must work in a minimal CPU Python environment.
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def braced_scope_at(source: str, start: int) -> str:
    """Return the lexical C++ block beginning at an already-validated offset."""
    if start < 0:
        return ""
    opening = source.find("{", start)
    if opening < 0:
        return ""
    depth = 0
    for index in range(opening, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    return ""


def braced_scope(source: str, marker: str) -> str:
    """Return the lexical C++ block beginning at the first exact marker.

    This deliberately stays a static text helper: it is used only to make
    policy checks local to a class/function rather than accidentally accepting
    a matching token elsewhere in the translation unit.  Callers that need a
    particular overloaded function must locate a unique signature with regex
    and then call braced_scope_at(), rather than using this first-match helper.
    """
    return braced_scope_at(source, source.find(marker))


def production_vector_res_ids_knn_scope(source: str) -> tuple[str, dict[str, object]]:
    """Find the unique G3 production KNN overload, not a legacy overload.

    The copied header carries four overloads.  Safe-C1 invokes only the vector
    + res_ids form: `float *query_data, int *res_ids, ...`.  This complete
    regex intentionally binds every parameter through size_s so a future
    accidental legacy-first match fails the audit rather than passing it.
    """
    signature = re.compile(
        r"void\s+searchIndexKnnV2\s*\(\s*"
        r"short\s*\*\s*data_d\s*,\s*"
        r"TN\s*\*\s*node_list\s*,\s*"
        r"int\s*\*\s*id_list\s*,\s*"
        r"int\s*\*\s*max_node_num\s*,\s*"
        r"float\s*\*\s*query_data\s*,\s*"
        r"int\s*\*\s*res_ids\s*,\s*"
        r"int\s+qnum\s*,\s*int\s+k\s*,\s*int\s+tree_h\s*,\s*"
        r"int\s*\*\s*data_info\s*,\s*int\s*\*\s*empty_list\s*,\s*"
        r"char\s*\*\s*data_s\s*,\s*int\s*\*\s*size_s\s*\)",
        re.DOTALL,
    )
    matches = list(signature.finditer(source))
    metadata: dict[str, object] = {
        "required_signature": (
            "searchIndexKnnV2(short* data_d, TN* node_list, int* id_list, "
            "int* max_node_num, float* query_data, int* res_ids, int qnum, "
            "int k, int tree_h, int* data_info, int* empty_list, "
            "char* data_s, int* size_s)"
        ),
        "signature_match_count": len(matches),
        "start_line": None,
        "end_line": None,
    }
    if len(matches) != 1:
        return "", metadata
    scope = braced_scope_at(source, matches[0].start())
    metadata["start_line"] = source.count("\n", 0, matches[0].start()) + 1
    metadata["end_line"] = (source.count("\n", 0, matches[0].start() + len(scope)) + 1
                            if scope else None)
    return scope, metadata


def strip_cpp_comments(source: str) -> str:
    """Remove comments before checking for an executable legacy call."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)


def require_absent(source: str, tokens: list[str], errors: list[str], prefix: str) -> None:
    for token in tokens:
        if token in source:
            errors.append(f"{prefix}:{token}")


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
        "G3_REBUILD_FROM_CURRENT_LIVE_ONLY_IMPLEMENTED = true",
        "G3_RANGE_CLAIM_SCOPE",
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
    executable_source = strip_cpp_comments(source)
    if "real_gts_vector_range" in range_adapter or re.search(r"\bsearchIndexRnnV2\s*\(", executable_source):
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

    # v2 API/invariant audit.  These checks intentionally distinguish public
    # entrypoints from private helpers so a token elsewhere cannot mask a
    # missing per-instance serialization guard or an accidentally reopened
    # raw-rebuild API.
    lifecycle = braced_scope(source, "class NativeSafeC1Matrix {")
    public_at = lifecycle.find("\n public:")
    private_at = lifecycle.find("\n private:")
    public_lifecycle = lifecycle[public_at:private_at] if public_at >= 0 and private_at > public_at else ""
    private_lifecycle = lifecycle[private_at:] if private_at >= 0 else ""
    require(lifecycle, [
        "frozen_.assert_unchanged(runtime_)",
        "state_.assert_partition(frozen_, tree_version_)",
        "SafeC1State draft = state_",
        "state_.swap_noexcept(draft)",
        "release_runtime(&runtime_)",
        "mode_ = EngineMode::kRebuildDestructive",
        "enter_failed_stop_noexcept()",
        "mode_ = EngineMode::kAwaitingPostRebuildOracle",
        "post_rebuild_gate_.verify_issued_query",
        "rebuild_from_validated_ids_private",
    ], errors, "missing_state_or_failstop_token")
    if not public_lifecycle or not private_lifecycle:
        errors.append("native_matrix_public_private_scope_not_found")
    if "assert_unchanged(runtime_, state_.live_ids())" in source:
        errors.append("frozen_assertion_still_compares_dynamic_live_ids")
    if "dynamic_tiers_cleared_after_atomic_switch" in source or "atomic" in source or "rollback" in source:
        errors.append("obsolete_atomic_or_rollback_rebuild_claim_present")

    # Process-global GTS state (max_dis_d and the copied traversal-receipt
    # vector) has an RAII process lease.  It must be the first engine member,
    # ahead of the nonrecursive per-instance API mutex.
    lease = braced_scope(source, "class SameProcessExclusiveGtsLease final")
    require(lease, [
        "std::try_to_lock",
        "std::unique_lock<std::mutex> lock_",
        "static std::mutex& mutex()",
        "SameProcessExclusiveGtsLease(const SameProcessExclusiveGtsLease&) = delete",
        "SameProcessExclusiveGtsLease(SameProcessExclusiveGtsLease&&) = delete",
    ], errors, "missing_same_process_lease_token")
    require(source, [
        "#include <mutex>",
        "max_dis_d",
        "safe_c1_last_visited_leaf_pairs",
        "SameProcessExclusiveGtsLease process_lease_;",
        "mutable std::mutex api_mutex_;",
    ], errors, "missing_same_process_global_or_mutex_token")
    lease_member = private_lifecycle.find("SameProcessExclusiveGtsLease process_lease_;")
    mutex_member = private_lifecycle.find("mutable std::mutex api_mutex_;")
    if lease_member < 0 or mutex_member < 0 or lease_member > mutex_member:
        errors.append("process_lease_not_declared_before_api_mutex")
    if "std::recursive_mutex" in source or "recursive_mutex" in source:
        errors.append("recursive_mutex_forbidden_for_native_matrix")

    # Every public lifecycle/mutation/query/verification/read accessor uses
    # the same nonrecursive guard exactly once.  Private helpers must not take
    # it, which would create a hidden nested-lock path.
    public_api_markers = [
        "~NativeSafeC1Matrix()",
        "void initialize_immutable_pool(",
        "RebuildReceipt build_initial_base(",
        "Placement insert(",
        "Placement erase_mutable(",
        "RebuildReceipt rebuild_from_current_live()",
        "IssuedQuery query_knn(",
        "IssuedQuery query_range(",
        "void verify_first_post_rebuild_query(",
        "[[nodiscard]] std::uint64_t tree_version() const",
        "[[nodiscard]] EngineMode mode() const",
        "[[nodiscard]] bool post_rebuild_oracle_pending() const",
    ]
    api_guard = "std::lock_guard<std::mutex> api_guard(api_mutex_);"
    public_api_bodies: dict[str, str] = {}
    for marker in public_api_markers:
        method = braced_scope(public_lifecycle, marker)
        public_api_bodies[marker] = method
        if not method:
            errors.append(f"missing_public_native_matrix_entrypoint:{marker}")
        elif method.count(api_guard) != 1:
            errors.append(f"public_native_matrix_entrypoint_not_singly_serialized:{marker}")
    if api_guard in private_lifecycle:
        errors.append("private_native_matrix_helper_takes_api_mutex_nested_lock_risk")
    for internal_type in ("SafeC1State", "FrozenTreeSnapshot", "PostRebuildOracleGate", "BaseTreeRuntime"):
        if re.search(rf"\b(?:const\s+)?{internal_type}\s*&", public_lifecycle):
            errors.append(f"public_native_matrix_returns_or_exposes_internal_reference:{internal_type}")

    # The old facade could rebuild arbitrary IDs while Ready.  v2 has exactly
    # one initial raw-base input and a ready-state rebuild that snapshots only
    # state_.live_ids() before calling a private helper.
    require_absent(source, [
        "FreshStableIdRebuild",
        "mark_query_issued",
        "verify_first_after_rebuild",
    ], errors, "obsolete_or_unsafe_v1_api_present")
    build_initial = public_api_bodies.get("RebuildReceipt build_initial_base(", "")
    rebuild_current = public_api_bodies.get("RebuildReceipt rebuild_from_current_live()", "")
    require(build_initial, [
        "require_device_residual_mode_zero_or_fail_stop()",
        "return rebuild_from_validated_ids_private(base_ids);",
    ], errors, "missing_initial_base_rebuild_contract")
    require(rebuild_current, [
        "require_ready_for_mutation()",
        "return rebuild_from_validated_ids_private(state_.live_ids());",
    ], errors, "missing_current_live_rebuild_contract")
    public_raw_rebuilds = re.findall(
        r"RebuildReceipt\s+([A-Za-z_]\w*)\s*\(\s*const\s+std::vector<StableId>\s*&",
        public_lifecycle)
    if public_raw_rebuilds != ["build_initial_base"]:
        errors.append("public_rebuild_accepts_raw_live_ids_beyond_initial_base")
    if "RebuildReceipt rebuild_from_validated_ids_private(" not in private_lifecycle:
        errors.append("raw_id_rebuild_helper_not_private")

    # Opaque, move-only ticket contract.  Verification accepts an opaque
    # ticket plus an independently computed expectation, never a caller-owned
    # QueryExport that could spoof the native actual/binding/result.
    ticket_secret = braced_scope(source, "class PostRebuildTicketSecret final")
    ticket = braced_scope(source, "class PostRebuildVerificationTicket final")
    issued_query = braced_scope(source, "struct IssuedQuery final")
    oracle_gate = braced_scope(source, "class PostRebuildOracleGate {")
    require(ticket_secret, [
        "private:",
        "PostRebuildTicketSecret(std::uint64_t owner_nonce",
        "std::uint64_t serial_",
        "std::string kind_",
        "std::string binding_sha256_",
    ], errors, "missing_opaque_ticket_secret_token")
    require(ticket, [
        "PostRebuildVerificationTicket(const PostRebuildVerificationTicket&) = delete",
        "PostRebuildVerificationTicket(PostRebuildVerificationTicket&&) noexcept = default",
        "explicit PostRebuildVerificationTicket(std::shared_ptr<const PostRebuildTicketSecret> secret)",
        "std::shared_ptr<const PostRebuildTicketSecret> secret_",
        "void consume() noexcept",
    ], errors, "missing_move_only_ticket_token")
    require(issued_query, [
        "QueryExport export_data;",
        "std::optional<PostRebuildVerificationTicket> post_rebuild_ticket;",
        "IssuedQuery(const IssuedQuery&) = delete",
        "IssuedQuery(IssuedQuery&&) noexcept = default",
    ], errors, "missing_issued_query_ticket_token")
    require(oracle_gate, [
        "struct IssuedRecord",
        "std::shared_ptr<const PostRebuildTicketSecret> ticket_secret",
        "std::vector<StableDistance> engine_results",
        "std::uint64_t issuance_nonce",
        "next.ticket_secret = secret",
        "current.ticket_secret.get() != ticket.secret_.get()",
        "current.engine_results != independent_expected",
        "ticket.consume();",
    ], errors, "missing_private_issued_record_or_ticket_binding")
    verify_public = public_api_bodies.get("void verify_first_post_rebuild_query(", "")
    verify_gate = braced_scope(oracle_gate, "void verify_issued_query(")
    require(verify_public, [
        "PostRebuildVerificationTicket&& ticket",
        "const std::vector<StableDistance>& independent_expected",
        "post_rebuild_gate_.verify_issued_query(std::move(ticket), independent_expected)",
    ], errors, "missing_public_opaque_ticket_verifier")
    require(verify_gate, [
        "PostRebuildVerificationTicket&& ticket",
        "const std::vector<StableDistance>& independent_expected",
        "validate_independent_expected(current, independent_expected)",
    ], errors, "missing_gate_opaque_ticket_verifier")
    if "QueryExport" in verify_public or "QueryExport" in verify_gate:
        errors.append("post_rebuild_verifier_accepts_caller_owned_query_export")

    binding = braced_scope(source, "inline std::string post_rebuild_binding_sha256(")
    require(binding, [
        "kind=", "query_id=", "dimension=", "query_vector=", "requested_k=",
        "radius_sq=", "tree_version=", "tree_payload=", "state_epoch=", "active_ids=",
        "issuance_nonce=", "receipt=", "results=",
    ], errors, "missing_post_rebuild_binding_component")
    require(source, [
        "struct QueryInputSnapshot",
        "snapshot.values.assign(query_values, query_values + snapshot.dimension)",
        "query_input_snapshot_sha256(snapshot.values, snapshot.dimension)",
        "upload_query_snapshot(input_snapshot, &query_device)",
        "QueryExport staged = *actual",
        "next.engine_results = staged.results",
        "current = std::move(next)",
    ], errors, "missing_query_snapshot_or_private_actual_binding")
    expected_validation = braced_scope(oracle_gate, "static void validate_independent_expected(")
    require(expected_validation, [
        "require_canonical_stable_distances(independent_expected",
        "active.find(row.stable_id) == active.end()",
        "record.kind == \"range\" && row.distance_sq > record.radius_sq",
        "const std::size_t expected_count = std::min(",
        "independent_expected.size() != expected_count",
    ], errors, "missing_independent_expected_validation")
    for marker in ("IssuedQuery query_knn(", "IssuedQuery query_range("):
        query_body = public_api_bodies.get(marker, "")
        if (query_body.find("require_canonical_stable_distances(output.results") >
                query_body.find("post_rebuild_gate_.issue_successful_query")):
            errors.append(f"post_rebuild_ticket_issued_before_canonical_result:{marker}")

    # Trace JSON may record a ticket issuance/binding diagnostic, but source
    # must not serialize an oracle comparison outcome or claim CPU-oracle
    # provenance/correctness.  Those are later runner responsibilities.
    rebuild_serializer = braced_scope(source, "inline std::string serialize_rebuild_jsonl_record(")
    serialized = serializer + rebuild_serializer
    require_absent(serialized, [
        "post_rebuild_oracle_match",
        "post_rebuild_oracle_verified",
        "post_rebuild_oracle_comparison",
        "independent_oracle_passed",
        "oracle_comparison_passed",
    ], errors, "serialized_oracle_match_claim_forbidden")
    require_absent(source, [
        "post_rebuild_oracle_match",
        "post_rebuild_oracle_verified",
        "oracle_comparison_passed",
    ], errors, "oracle_match_field_forbidden")

    # c_rp_mode is process-global, so initialization readback is not enough.
    # Public mutators/query paths use shared readiness helpers, while rebuild
    # initialization and ticket verification read it directly before use.
    mode_guard = braced_scope(lifecycle, "void require_device_residual_mode_zero_or_fail_stop()")
    mutation_ready = braced_scope(lifecycle, "void require_ready_for_mutation()")
    query_ready = braced_scope(lifecycle, "void require_ready_for_query(")
    require(mode_guard, [
        "cudaMemcpyFromSymbol(&device_mode, c_rp_mode",
        "enter_failed_stop_noexcept()",
        "device residual pruning mode drifted from zero",
    ], errors, "missing_per_operation_residual_mode_readback")
    require(mutation_ready, ["require_device_residual_mode_zero_or_fail_stop()"],
            errors, "mutation_path_missing_residual_mode_readback")
    require(query_ready, ["require_device_residual_mode_zero_or_fail_stop()"],
            errors, "query_path_missing_residual_mode_readback")
    for marker, required_call in (
        ("Placement insert(", "require_ready_for_mutation()"),
        ("Placement erase_mutable(", "require_ready_for_mutation()"),
        ("RebuildReceipt rebuild_from_current_live()", "require_ready_for_mutation()"),
        ("IssuedQuery query_knn(", "require_ready_for_query(\"knn\")"),
        ("IssuedQuery query_range(", "require_ready_for_query(\"range\")"),
        ("void verify_first_post_rebuild_query(", "require_device_residual_mode_zero_or_fail_stop()"),
    ):
        require(public_api_bodies.get(marker, ""), [required_call], errors,
                f"public_operation_missing_residual_mode_guard:{marker}")
    if "G3_DIRECT_SIDECAR_RUNTIME_GUARD_OPEN = true" in source:
        errors.append("future_direct_gate_opened_without_runtime_p0")

    # The copied header has several legacy KNN overloads.  Audit only the
    # actual Safe-C1 production vector+res_ids overload, not the first legacy
    # ID-query overload that happens to share the function name.
    knn_header, knn_production_scope = production_vector_res_ids_knn_scope(search)
    require(search, [
        "inline void safe_c1_traversal_require_cuda_success",
        "[[noreturn]] inline void safe_c1_traversal_fail_stop",
        "throw std::runtime_error",
    ], errors, "missing_knn_failstop_helper")
    if not knn_header:
        errors.append("production_vector_res_ids_searchIndexKnnV2_scope_not_unique_or_missing")
    else:
        require(knn_header, [
            "safe_c1_last_visited_leaf_pairs.clear()",
            "safe_c1_traversal_fail_stop(\"invalid GTS KNN leaf receipt\")",
            "mergeResKnnIds<<<",
        ], errors, "missing_production_vector_res_ids_knn_receipt_token")
        knn_executable = strip_cpp_comments(knn_header)
        if ("fprintf" in knn_executable or "exit(" in knn_executable or
                "if (cudaStatus != cudaSuccess)" in knn_executable):
            errors.append("production_vector_res_ids_knn_cuda_error_path_can_print_or_exit_or_continue")
        status_reads = len(re.findall(r"cudaStatus\s*=\s*cudaGetLastError\(\);\s*"
                                      r"safe_c1_traversal_require_cuda_success\(cudaStatus,", knn_executable))
        all_status_reads = knn_executable.count("cudaGetLastError()")
        knn_production_scope["cuda_get_last_error_count"] = all_status_reads
        knn_production_scope["immediately_failstop_checked_count"] = status_reads
        # The header may also fail-stop cudaDeviceSynchronize/cudaFree/etc.;
        # only require every cudaGetLastError readback itself to be immediately
        # consumed by the helper, not that helper calls equal readback count.
        if all_status_reads == 0 or all_status_reads != status_reads:
            errors.append("production_vector_res_ids_knn_cuda_error_readback_not_immediately_failstop_checked")

    # The transitive archive CHECK macro intentionally retains exit(1).  It is
    # not a recoverable result path: document/enforce it only as a nonzero
    # process fail-stop, while the copied G3 header itself has no local
    # print-and-continue status path.
    tree_header = text(root / "reference/include/tree.cuh") if (root / "reference/include/tree.cuh").is_file() else ""
    require(tree_header, [
        "#define CHECK(call)",
        "exit(1);",
    ], errors, "missing_transitive_tree_check_nonzero_failstop")

    # Range is intentionally only a v2 branch-aligned predicate mirror with
    # exact sidecar filtering.  It must not be re-described as archive-native
    # equivalence or as a direct-range correctness result.
    require(source, [
        "v2 branch-aligned predicate mirror plus exact sidecar filter; not native archive equivalence",
        "output.direct_sidecars_exact_range_filtered = is_range",
        "not represented as archive-native equivalence or direct-range proof",
    ], errors, "missing_v2_range_claim_scope")
    if re.search(r"G3_NATIVE_RANGE_RECEIPT_IMPLEMENTED\s*=\s*true", source) or \
            re.search(r"G3_DIRECT_SIDECAR_RUNTIME_GUARD_OPEN\s*=\s*true", source) or \
            re.search(r"G3_DIRECT_CERTIFICATE_KNN_VISIBILITY_ONLY\s*=\s*false", source):
        errors.append("range_or_direct_overclaim_flag_present")
    require_absent(source, [
        "archive_all_include_native_equivalence",
        "archive_native_range_equivalence = true",
        "direct_range_correctness = true",
        "direct_range_proven = true",
    ], errors, "range_overclaim_token_present")

    forbidden = [
        '#include "incremental_insert.cuh"', '#include "update.cuh"', '#include "update_optimized.cuh"',
        "updateIndexRnn(", "deleteIncrementalInsert(", "exact_live_l2", "active_pool_scan",
        "legacy direct insert",
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
        "schema": "safe-c1-g3-source-audit-v5-api-invariants-p0",
        "status": "PASS_STATIC_P0_REPAIR" if not errors else "FAIL_STATIC_P0_REPAIR",
        "gpu_used": False,
        "scope": "static isolated source closure only; no CUDA build/run/performance/correctness result",
        "root": str(root),
        "closure_files": {str(p.relative_to(root)): sha256_file(p) for p in closure if p.is_file()},
        "source_closure_sha256": closure_hash,
        "audited_production_knn_vector_res_ids_scope": knn_production_scope,
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
            "same_process_exclusive_raii_lease_source": True,
            "same_instance_nonrecursive_api_serialization_source": True,
            "public_rebuild_closes_raw_live_id_facade_source": True,
            "opaque_post_rebuild_ticket_binding_source": True,
            "per_operation_residual_mode_readback_source": True,
            "knn_header_cuda_error_failstop_source": True,
            "transitive_tree_check_nonzero_process_failstop": True,
            "range_claim_scope_fail_closed_source": True,
            "oracle_match_not_serialized_source": True,
        },
        "not_established": [
            "native runtime correctness", "native KNN exactness", "native range correctness",
            "direct-sidecar range correctness", "runtime tree height", "device residual-mode readback",
            "KNN disk monotonicity", "rebuild runtime behavior", "same-process lease runtime behavior",
            "opaque ticket runtime behavior", "independent CPU oracle provenance/correctness",
            "latency", "throughput", "GPU utilization", "performance benefit",
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
