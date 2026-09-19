#!/usr/bin/env python3
"""CPU-only static contract audit for isolated Safe-C1 G1 v2.

This checks source/artifact binding only. It does not run a CUDA binary, query a
GPU, establish performance, or turn the bounded static-prefix/dynamic-boundary
witness restriction into an all-input canonical (distance,stable_id) claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from safe_c1_v2_bundle_contract import (
    BOUNDED_TIE_ABORT,
    BOUNDED_TIE_DOMAIN,
    CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
    DYNAMIC_K_BOUNDARY_POLICY,
    STATIC_BASE_POLICY,
)

CANONICAL_V2_ROOT = Path("/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v2_search_native")
V1_ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/safe_c1_dynamic_gts_v1")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(text: str, token: str, errors: list[str], label: str) -> None:
    if token not in text:
        errors.append(f"missing:{label}:{token}")


def forbid(text: str, token: str, errors: list[str], label: str) -> None:
    if token in text:
        errors.append(f"forbidden:{label}:{token}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def require_manifest_binding(
    manifest: dict[str, Any], root: Path, paths: dict[str, Path], errors: list[str]
) -> None:
    if manifest.get("schema") != "safe-c1-v2-cpu-test-build-manifest":
        errors.append("bad_cpu_test_manifest_schema")
    if manifest.get("gpu_used") is not False or manifest.get("cuda_binary_executed") is not False:
        errors.append("cpu_test_manifest_not_cpu_only")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("bad_cpu_test_manifest_artifacts")
        return
    for label, path in paths.items():
        expected = artifacts.get(label)
        if not isinstance(expected, str):
            errors.append(f"cpu_test_manifest_missing:{label}")
        elif path.is_file() and expected != sha256(path):
            errors.append(f"cpu_test_manifest_sha_mismatch:{label}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    boundary_errors: list[str] = []
    if root != CANONICAL_V2_ROOT:
        boundary_errors.append("audit_root_must_equal_canonical_v2_root")
    if root == V1_ROOT or V1_ROOT in root.parents:
        boundary_errors.append("audit_v1_root_forbidden")

    paths = {
        "source": root / "src/safe_c1_dynamic_gts.cu",
        "search_receipt_header": root / "include/safe_search_v2.cuh",
        "routing_certificate_header": root / "include/safe_c1_search_native_routing_certificate.hpp",
        "routing_certificate_test_source": root / "tools/routing_certificate_test.cpp",
        "routing_certificate_test_binary": root / "build/routing_certificate_test",
        "routing_certificate_test_log": root / "logs/routing_certificate_test_cpu.log",
        "tie_free_witness_header": root / "include/safe_c1_tie_free_witness.hpp",
        "tie_free_witness_test_source": root / "tools/tie_free_witness_test.cpp",
        "tie_free_witness_test_binary": root / "build/tie_free_witness_test",
        "tie_free_witness_test_log": root / "logs/tie_free_witness_test_cpu.log",
        "cuda_compile_binary": root / "bin/GTS_safe_c1_g1_v2_search_native",
        "cpu_test_build_manifest": root / "provenance/cpu_test_build_manifest.json",
        "bundle_contract_module": root / "tools/safe_c1_v2_bundle_contract.py",
        "bundle_generator": root / "tools/make_g1_v2_bundle.py",
        "bundle_preflight": root / "tools/preflight_g1_v2_bundle.py",
        "v2_cpu_guard": root / "tools/prepare_g1_v2_guarded_run.sh",
        "v2_cpu_guard_help_log": root / "logs/prepare_g1_v2_guarded_run_help_cpu.log",
        "bundle_preflight_test_source": root / "tools/test_v2_bundle_preflight.py",
        "bundle_preflight_test_log": root / "logs/v2_bundle_preflight_test_cpu.log",
        "bundle_guard_test_manifest": root / "provenance/v2_bundle_guard_cpu_test_manifest.json",
        "cuda_compile_manifest": root / "provenance/cuda_compile_v2_search_native_manifest.json",
        "selection_bundle_generator": root / "tools/make_g1_v2_certificate_selection_bundle.py",
        "selection_finalizer": root / "tools/finalize_g1_v2_candidate_selection.py",
        "selection_guard": root / "tools/run_g1_v2_certificate_selection_guarded.sh",
        "selection_guard_help_log": root / "logs/run_g1_v2_certificate_selection_guarded_help_cpu.log",
        "selection_pipeline_test_source": root / "tools/test_v2_certificate_selection_pipeline.py",
        "selection_pipeline_test_log": root / "logs/v2_certificate_selection_pipeline_cpu.log",
        "selection_pipeline_test_manifest": root / "provenance/v2_selection_pipeline_cpu_test_manifest.json",
    }
    errors: list[str] = list(boundary_errors)
    for label, path in paths.items():
        if not path.is_file():
            errors.append(f"missing_file:{label}:{path}")

    src = read_text(paths["source"])
    receipt = read_text(paths["search_receipt_header"])
    routing_header = read_text(paths["routing_certificate_header"])
    routing_test = read_text(paths["routing_certificate_test_source"])
    routing_log = read_text(paths["routing_certificate_test_log"])
    tie_header = read_text(paths["tie_free_witness_header"])
    tie_test = read_text(paths["tie_free_witness_test_source"])
    tie_log = read_text(paths["tie_free_witness_test_log"])
    bundle_contract = read_text(paths["bundle_contract_module"])
    bundle_generator = read_text(paths["bundle_generator"])
    bundle_preflight = read_text(paths["bundle_preflight"])
    v2_cpu_guard = read_text(paths["v2_cpu_guard"])
    v2_cpu_guard_help = read_text(paths["v2_cpu_guard_help_log"])
    bundle_preflight_test = read_text(paths["bundle_preflight_test_source"])
    bundle_preflight_log = read_text(paths["bundle_preflight_test_log"])
    selection_bundle_generator = read_text(paths["selection_bundle_generator"])
    selection_finalizer = read_text(paths["selection_finalizer"])
    selection_guard = read_text(paths["selection_guard"])
    selection_guard_help = read_text(paths["selection_guard_help_log"])
    selection_pipeline_test = read_text(paths["selection_pipeline_test_source"])
    selection_pipeline_log = read_text(paths["selection_pipeline_test_log"])

    # G1 receipt/merge mechanism remains bound to the real copied GTS traversal.
    for token in (
        '#include "safe_search_v2.cuh"',
        'run_gts_base_topk_with_receipt',
        'state.sidecar_candidates_for(gts_answer.visited_leaf_ids)',
        'exact_candidate_l2',
        'G1 refuses base deletion',
        'G1 deliberately rejects range events',
    ):
        need(src, token, errors, "source")
    forbid(src, "exact_live_l2", errors, "source")
    for token in (
        '#include "incremental_insert.cuh"',
        '#include "update.cuh"',
        'updateIndexRnn(',
        'deleteIncrementalInsert(',
    ):
        forbid(src, token, errors, "source")

    # Search-native certificate: no old max-distance routing. The snapshot still
    # hashes max_distance, which is expressly allowed and required for freeze
    # mutation detection.
    for token in (
        'common_pivot_or_reject',
        'only_strict_search_native_child',
        'fanout != TREE_ORDER',
        'node.min_dis > previous_min + kStrictEpsilon',
        'distance < nodes[next].min_dis - kStrictEpsilon',
        'remains fingerprinted in compute_hash',
        'not a routing bound',
        'caller puts the object in exact global delta (fail closed)',
    ):
        need(src, token, errors, "source")
    for token in (
        'const float upper = max_distance[child];',
        'only_strict_child(int parent, float distance)',
    ):
        forbid(src, token, errors, "source")

    # Bounded tie witness domain: only the static top-(k+1) prefix and every
    # dynamic k/k+1 membership boundary must be unambiguous. Ties strictly
    # beyond the static prefix are allowed and must never be reported as a
    # global canonical (distance,stable_id) result.
    for token in (
        '#include "safe_c1_tie_free_witness.hpp"',
        'validate_tie_free_witness_trace',
        'require_static_base_top_k_plus_one_tie_free',
        'require_dynamic_k_boundary_tie_free',
        'BOUNDED_TIE_WITNESS_VALIDATED',
        'BOUNDED_TIE_WITNESS_PRECONDITION_SATISFIED',
        'ABORT_UNPROVED_BOUNDED_TIE_WITNESS_DOMAIN',
        'gts_probe_same_distance_group_set_mismatch',
        'modeled_gts_float_l2_key_for',
        'global_canonical_distance_stable_id_proof',
        'all-input tie claim',
        'require_identity_stable_id_layout',
        'read_little_endian_i32_identity_payload',
    ):
        need(src, token, errors, "source")
    for token in (
        'static_base_top_k_plus_one_exact_squared_l2_tie',
        'static_base_top_k_plus_one_modeled_gts_float_key_tie',
        'static_base_top_k_order_mismatch_exact_vs_modeled_gts_float',
        'dynamic_k_boundary_distance_tie',
        'modeled_gts_float_l2_sort_key',
        'ABORT_UNPROVED_BOUNDED_TIE_WITNESS_DOMAIN',
    ):
        need(tie_header, token, errors, "tie_header")
    forbid(tie_header, 'require_all_static_base_oracles_tie_free', errors, "tie_header")
    if src.count(STATIC_BASE_POLICY) != 2:
        errors.append("source_static_base_policy_literal_drift")
    if src.count(DYNAMIC_K_BOUNDARY_POLICY) != 2:
        errors.append("source_dynamic_boundary_policy_literal_drift")
    if src.count(CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE) != 2:
        errors.append("source_stable_id_layout_literal_drift")

    identity_layout_guard = src.find('require_identity_stable_id_layout(args.bundle, header)')
    preflight_call = src.find('validate_tie_free_witness_trace(header, events, tie_witness_checker)')
    first_tree_build = src.find('indexConstru(runtime.data_d')
    first_main_cuda = src.find('GI_CUDA(cudaMallocManaged', identity_layout_guard)
    first_output = src.find('std::ofstream output(args.output)')
    if identity_layout_guard < 0 or first_tree_build < 0 or first_main_cuda < 0 or not (
        identity_layout_guard < first_tree_build and identity_layout_guard < first_main_cuda
    ):
        errors.append('stable_id_layout_guard_not_before_tree_build_or_main_cuda')
    if preflight_call < 0 or first_tree_build < 0 or first_output < 0 or not (
        preflight_call < first_tree_build < first_output
    ):
        errors.append('tie_preflight_not_before_tree_build_and_output')

    # Receipt must be emitted before the native leaf scan invocation.
    for token in (
        'safe_c1_last_visited_leaf_pairs.clear()',
        'Safe-C1 instrumentation: materialize only the (query, leaf) pairs',
        'mergeLNodeKnn error',
        'dataProcessKnnVec<<<',
    ):
        need(receipt, token, errors, "search_receipt_header")
    receipt_marker = receipt.find('Safe-C1 instrumentation: materialize only the (query, leaf) pairs')
    leaf_scan = receipt.find('dataProcessKnnVec<<<', receipt_marker if receipt_marker >= 0 else 0)
    if receipt_marker < 0 or leaf_scan < 0 or receipt_marker > leaf_scan:
        errors.append('receipt_not_before_native_leaf_scan')

    # Bind CPU-only semantic tests to the runtime contract.
    for token in (
        'search_v2 does not use it in',
        'nodeProcessKnn/nodeProcessRnn pruning',
        'm_(i+1) - eps',
        'last child',
        'common_pivot_or_reject',
    ):
        need(routing_header, token, errors, "routing_certificate_header")
    for token in (
        'stale_overlap_old_accepts_new_rejects',
        'full_ten_sibling_contract_or_delta',
        'PASS safe_c1_search_native_routing_certificate CPU-only',
    ):
        need(routing_test, token, errors, "routing_certificate_test_source")
    need(routing_log, 'PASS safe_c1_search_native_routing_certificate CPU-only',
         errors, "routing_certificate_test_log")
    for token in (
        'tie_free_witness_allows_static_exact_tie_after_top_k_plus_one',
        'tie_free_witness_aborts_static_exact_tie_in_top_k_plus_one',
        'tie_free_witness_allows_static_modeled_float_tie_after_top_k_plus_one',
        'tie_free_witness_aborts_static_modeled_float_tie_in_top_k_plus_one',
        'tie_free_witness_allows_dynamic_internal_tie_away_from_boundary',
        'tie_free_witness_aborts_dynamic_k_boundary_tie',
        'PASS safe_c1_tie_free_witness CPU-only',
    ):
        need(tie_test, token, errors, "tie_free_witness_test_source")
    need(tie_log, 'PASS safe_c1_tie_free_witness CPU-only', errors,
         "tie_free_witness_test_log")
    for label, text in (
        ("routing_certificate_test_source", routing_test),
        ("tie_free_witness_test_source", tie_test),
    ):
        lowered = text.lower()
        for forbidden in ('cuda', 'nvidia-smi', 'nvml', 'nsys'):
            if forbidden in lowered:
                errors.append(f"forbidden_cpu_test_token:{label}:{forbidden}")

    # The new bundle chain is v2-only and intentionally cannot repurpose any
    # v1 bundle/run output as authorization. Its normal guard stops after CPU
    # preflight; this preparation turn has no GPU launch path.
    for token in (
        'safe-c1-g1-v2-tie-free-witness-preflight',
        'BOUNDED_TIE_DOMAIN',
        'STATIC_BASE_POLICY',
        'DYNAMIC_K_BOUNDARY_POLICY',
        'ABORT_UNPROVED_BOUNDED_TIE_WITNESS_DOMAIN',
        'static_base_top_k_plus_one_exact_squared_l2_tie',
        'modeled_gts_float_l2_key',
        'dynamic_k_boundary_distance_tie',
        'forbidden_g1_range_event',
        'load_current_runner_identity_stable_id_layout',
        'stable_id_layout_not_identity_for_current_runner',
        'CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE',
    ):
        need(bundle_contract, token, errors, "bundle_contract_module")
    for token in (
        'safe-c1-g1-v2-tie-free-bundle-manifest',
        'PREPARED_CPU_ONLY_PENDING_V2_CERTIFICATE_SELECTION',
        'safe-c1-search-native-sibling-boundary-v2',
        'immutable input bytes only; not v1 execution evidence',
        'PENDING_V2_CERTIFICATE_SELECTION',
        'v2 bundle output must be under this v2 root',
        'load_current_runner_identity_stable_id_layout',
        'CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE',
        'stable_id_layout',
    ):
        need(bundle_generator, token, errors, "bundle_generator")
    for token in (
        'wrong_or_v1_bundle_schema',
        'candidate_selection_pending_not_execution_ready',
        'READY_FOR_V2_WITNESS',
        'safe-c1-g1-v2-candidate-selection',
        'validate_tie_free_witness',
        'root_must_match_this_v2_preflight_source_root',
        'ready_selection_manifest_tie_or_isolation_binding_mismatch',
        'load_current_runner_identity_stable_id_layout',
        'ready_selection_oracle_runner_identity_layout_missing_or_failed',
        'ready_selection_oracle_input_hash_binding_mismatch',
    ):
        need(bundle_preflight, token, errors, "bundle_preflight")
    for token in (
        'safe_c1_dynamic_gts_v2_search_native',
        'tide_safe_c1_20260728/runs',
        'V1_ROOT=',
        'v1 root is forbidden',
        'v1 run boundary is forbidden',
        'BUNDLE_PREFLIGHT',
        'CPU_PREFLIGHT_COMPLETE_AWAITING_EXPLICIT_GPU_AUTHORIZATION',
        'never launches CUDA',
        'v1 bundle schema',
    ):
        need(v2_cpu_guard, token, errors, "v2_cpu_guard")
    for forbidden in ('CUDA_VISIBLE_DEVICES=', 'nvidia-smi -i', 'timeout 900', '"$BIN" --bundle'):
        forbid(v2_cpu_guard, forbidden, errors, "v2_cpu_guard")
    for token in (
        'v2_bundle_generator_manifest_certificate_and_tie_domain',
        'v2_bundle_preflight_allows_cpu_pending_but_refuses_execution',
        'v2_bundle_root_boundary_rejects_foreign_writer_and_preflight_root',
        'PASS safe_c1_v2_bundle_preflight CPU-only',
    ):
        need(bundle_preflight_test, token, errors, "bundle_preflight_test_source")
        need(bundle_preflight_log, token, errors, "bundle_preflight_test_log")

    # Certificate-selection is implemented only as a separately guarded future
    # run path. It can accept a fresh v2 bounded-tie bundle, but source checks keep
    # it GPU0-only/no-kill/new-root and require a CPU finalizer before READY.
    for token in (
        'PREPARED_CPU_ONLY_READY_FOR_V2_CERTIFICATE_SELECTION_RUN',
        'safe-c1-g1-v2-certificate-selection-contract',
        'CPU bounded-tie input screen only; no certificate placement result',
        'BOUNDED_TIE_DOMAIN',
        'STATIC_BASE_POLICY',
        'DYNAMIC_K_BOUNDARY_POLICY',
        'leaf_capacity": 1',
        'v1_evidence_used": False',
        'min_same_sidecar_leaf_direct_candidates',
        'record_only_not_selection_gate',
        'g1b_followup',
        'at least three direct receipts sharing one sidecar leaf',
        'inserted/query/deleted',
        'load_current_runner_identity_stable_id_layout',
        'unique_self_top1_screen',
        'stable_id_layout',
    ):
        need(selection_bundle_generator, token, errors, "selection_bundle_generator")
    for token in (
        'safe-c1-g1-v2-candidate-selection',
        'READY_FOR_V2_WITNESS',
        'candidate_selection_v2.json',
        'static_contract_binary_sha_mismatch',
        'direct_candidate_visibility_not_proven',
        'selection_trace_not_exactly_isolated',
        'selection_trace_exactly_isolated',
        'ENGINE_SUMMARY_STATUS',
        'ENGINE_STATIC_PROBE_STATUS',
        'STATIC_BASE_POLICY',
        'DYNAMIC_K_BOUNDARY_POLICY',
        'on_tie_violation',
        'root_must_match_this_v2_finalizer_source_root',
        'independent_full_active_exact_topk_oracle',
        'full_active_exact_topk_id_order_mismatch',
        'full_active_exact_topk_membership_mismatch',
        'full_active_exact_topk_distance_mismatch',
        'load_current_runner_identity_stable_id_layout',
        'CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE',
        'engine_summary_stable_id_layout_mismatch',
        'engine_meta_stable_id_layout_mismatch',
        'certificate_reject_to_exact_delta_under_isolated_capacity_one',
        'insufficient_new_v2_direct_candidates',
        'insufficient_new_v2_same_leaf_direct_group',
        'g1b_same_leaf_direct_group',
        'certificate_delta_rejection_count',
        'record_only_not_selection_gate',
        'all-input tie proof, or v1-selection claim',
        'atomic_write_json',
    ):
        need(selection_finalizer, token, errors, "selection_finalizer")
    for token in (
        'GPU0',
        'EXPECTED_UUID',
        'MAX_IDLE_MEMORY_MIB',
        'no_process_control',
        'new v2 binary only',
        '--leaf-capacity 1',
        'candidate_selection_v2.json',
        'same-leaf direct group of three',
        'not a selection gate',
        'CPU-only gates precede every GPU query or CUDA binary invocation',
        'V1_ROOT=',
        'v1 root is forbidden',
        'v1 run boundary is forbidden',
        'PASS_G1_V2_CERTIFICATE_SELECTION_NO_PERFORMANCE_CLAIM',
        'no timing/throughput/capacity-result/all-input-tie claim',
    ):
        need(selection_guard, token, errors, "selection_guard")
    for token in (
        'certificate_selection_trace_not_exactly_isolated',
        'strict insert/query/delete triplet',
    ):
        need(bundle_preflight, token, errors, "bundle_preflight")
    for forbidden in ('kill ', 'pkill ', 'killall ', 'gpu-reset', 'nvidia-smi --gpu-reset'):
        forbid(selection_guard, forbidden, errors, "selection_guard")
    first_cpu_gate = selection_guard.find('"$AUDITOR"')
    first_gpu_query = selection_guard.find('nvidia-smi -i 0')
    first_binary_run = selection_guard.find('timeout 900 "$BIN"')
    if first_cpu_gate < 0 or first_gpu_query < 0 or first_binary_run < 0 or not (
        first_cpu_gate < first_gpu_query < first_binary_run
    ):
        errors.append('selection_guard_gpu_gate_order_invalid')
    for token in (
        'Do not execute until explicit GPU0 authorization',
        'bounded-tie selection-bundle preflight',
        'same-leaf direct group of three',
        'not a selection gate',
    ):
        need(selection_guard_help, token, errors, "selection_guard_help_log")
    for token in (
        'CPU-only authorization preparation',
        'never invokes nvidia-smi, CUDA, Nsight, or the GTS binary',
    ):
        need(v2_cpu_guard_help, token, errors, "v2_cpu_guard_help_log")
    for token in (
        'v2_certificate_selection_bundle_preflight',
        'v2_certificate_selection_rejects_nonisolated_trace',
        'v2_candidate_selection_rejects_split_direct_leaves',
        'v2_candidate_selection_same_leaf_direct_triple_hard_gate',
        'v2_candidate_selection_all_direct_without_delta_gate',
        'v2_candidate_selection_records_delta_statistic_only',
        'v2_engine_tie_policy_literal_contract',
        'v2_identity_stable_id_layout_generator_and_preflight_guard',
        'v2_candidate_selection_rejects_nonidentity_layout_after_preflight',
        'v2_candidate_selection_rejects_dynamic_policy_drift',
        'v2_candidate_selection_rejects_top1_correct_but_topk_wrong',
        'PASS safe_c1_v2_certificate_selection_pipeline CPU-only',
    ):
        need(selection_pipeline_test, token, errors, "selection_pipeline_test_source")
        need(selection_pipeline_log, token, errors, "selection_pipeline_test_log")
    for label, text in (
        ("selection_bundle_generator", selection_bundle_generator),
        ("selection_finalizer", selection_finalizer),
        ("bundle_preflight", bundle_preflight),
        ("selection_guard", selection_guard),
        ("selection_pipeline_test_source", selection_pipeline_test),
    ):
        for forbidden in ('min_delta_rejections', 'insufficient_new_v2_delta_rejections'):
            forbid(text, forbidden, errors, label)

    manifest: dict[str, Any] = {}
    manifest_path = paths["cpu_test_build_manifest"]
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"bad_cpu_test_manifest_json:{exc}")
    require_manifest_binding(
        manifest,
        root,
        {
            "routing_certificate_header": paths["routing_certificate_header"],
            "routing_certificate_test_source": paths["routing_certificate_test_source"],
            "routing_certificate_test_binary": paths["routing_certificate_test_binary"],
            "routing_certificate_test_log": paths["routing_certificate_test_log"],
            "tie_free_witness_header": paths["tie_free_witness_header"],
            "tie_free_witness_test_source": paths["tie_free_witness_test_source"],
            "tie_free_witness_test_binary": paths["tie_free_witness_test_binary"],
            "tie_free_witness_test_log": paths["tie_free_witness_test_log"],
        },
        errors,
    )

    compile_manifest: dict[str, Any] = {}
    compile_manifest_path = paths["cuda_compile_manifest"]
    if compile_manifest_path.is_file():
        try:
            compile_manifest = json.loads(compile_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"bad_cuda_compile_manifest_json:{exc}")
    if compile_manifest.get("schema") != "safe-c1-v2-search-native-cuda-compile-manifest":
        errors.append("bad_cuda_compile_manifest_schema")
    if compile_manifest.get("gpu_used") is not False or compile_manifest.get("cuda_binary_executed") is not False:
        errors.append("cuda_compile_manifest_not_compile_only")
    compile_artifacts = compile_manifest.get("artifacts")
    expected_compile_paths = {
        "source": paths["source"],
        "search_receipt_header": paths["search_receipt_header"],
        "routing_certificate_header": paths["routing_certificate_header"],
        "tie_free_witness_header": paths["tie_free_witness_header"],
        "binary": paths["cuda_compile_binary"],
    }
    build_log_rel = compile_manifest.get("build_log_relative_path")
    if not isinstance(build_log_rel, str) or Path(build_log_rel).is_absolute() or \
            Path(build_log_rel).name != build_log_rel:
        errors.append("bad_cuda_compile_manifest_build_log_relative_path")
    else:
        expected_compile_paths["build_log"] = root / "logs" / build_log_rel
    if not isinstance(compile_artifacts, dict):
        errors.append("bad_cuda_compile_manifest_artifacts")
    else:
        for label, path in expected_compile_paths.items():
            if not path.is_file() or compile_artifacts.get(label) != sha256(path):
                errors.append(f"cuda_compile_manifest_sha_mismatch:{label}")

    guard_manifest_path = paths["bundle_guard_test_manifest"]
    guard_manifest: dict[str, Any] = {}
    if guard_manifest_path.is_file():
        try:
            guard_manifest = json.loads(guard_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"bad_bundle_guard_manifest_json:{exc}")
    if guard_manifest.get("schema") != "safe-c1-v2-bundle-guard-cpu-test-manifest":
        errors.append("bad_bundle_guard_manifest_schema")
    if guard_manifest.get("gpu_used") is not False or guard_manifest.get("cuda_binary_executed") is not False:
        errors.append("bundle_guard_manifest_not_cpu_only")
    guard_artifacts = guard_manifest.get("artifacts")
    if not isinstance(guard_artifacts, dict):
        errors.append("bad_bundle_guard_manifest_artifacts")
    else:
        for label in (
            "bundle_contract_module", "bundle_generator", "bundle_preflight",
            "v2_cpu_guard", "v2_cpu_guard_help_log", "bundle_preflight_test_source", "bundle_preflight_test_log",
        ):
            path = paths[label]
            if guard_artifacts.get(label) != sha256(path):
                errors.append(f"bundle_guard_manifest_sha_mismatch:{label}")

    selection_test_manifest_path = paths["selection_pipeline_test_manifest"]
    selection_test_manifest: dict[str, Any] = {}
    if selection_test_manifest_path.is_file():
        try:
            selection_test_manifest = json.loads(selection_test_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"bad_selection_pipeline_manifest_json:{exc}")
    if selection_test_manifest.get("schema") != "safe-c1-v2-selection-pipeline-cpu-test-manifest":
        errors.append("bad_selection_pipeline_manifest_schema")
    if selection_test_manifest.get("gpu_used") is not False or selection_test_manifest.get("cuda_binary_executed") is not False:
        errors.append("selection_pipeline_manifest_not_cpu_only")
    selection_artifacts = selection_test_manifest.get("artifacts")
    if not isinstance(selection_artifacts, dict):
        errors.append("bad_selection_pipeline_manifest_artifacts")
    else:
        for label in (
            "selection_bundle_generator", "selection_finalizer", "selection_guard",
            "selection_guard_help_log", "selection_pipeline_test_source", "selection_pipeline_test_log",
            "bundle_preflight",
        ):
            path = paths[label]
            if selection_artifacts.get(label) != sha256(path):
                errors.append(f"selection_pipeline_manifest_sha_mismatch:{label}")

    result = {
        "schema": "safe-c1-g1-static-contract-audit-v2-search-native-tie-free",
        "gpu_used": False,
        "cuda_binary_executed": False,
        "root": str(root),
        "status": "PASS_CPU_ONLY_STATIC_CONTRACT" if not errors else "FAIL_CPU_ONLY_STATIC_CONTRACT",
        "formal_claim_eligible": False,
        "scope": (
            "static source/binary/test-artifact contract only; no CUDA execution, no "
            "performance result, and no all-input canonical (distance,stable_id) claim"
        ),
        "routing_contract": {
            "source_native_boundary": "min_i < radius < next_sibling_min for non-last; final sibling has no max-distance upper bound",
            "required_parent_shape": "all TREE_ORDER physical siblings non-empty, common pivot, finite strictly increasing min_dis",
            "on_failure": "direct placement rejected to exact global delta",
            "frozen_hash": "max_distance remains fingerprinted but is not used as a routing bound",
        },
        "bundle_guard": {
            "bundle_schema": "safe-c1-g1-v2-tie-free-bundle-manifest",
            "witness_preflight_requires": "new v2 candidate-selection artifact; pending/v1 bundle is refused",
            "selection_runner": "implemented but not executed; GPU0-only idle/no-kill/new-root gate; hard gate is a same-leaf direct triple, certificate-delta is record-only",
            "output_root": "/workspace/experiments/tide_safe_c1_20260728/runs",
            "performance_claim": False,
        },
        "tie_semantics": {
            "domain": BOUNDED_TIE_DOMAIN,
            "static_base_oracle": STATIC_BASE_POLICY + "; ties after that prefix are permitted",
            "dynamic_k_boundary": DYNAMIC_K_BOUNDARY_POLICY,
            "on_violation": BOUNDED_TIE_ABORT,
            "global_canonical_distance_stable_id_proof": False,
        },
        "files": {
            str(path.relative_to(root)): sha256(path)
            for path in paths.values()
            if path.is_file()
        },
        "errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": len(errors), "gpu_used": False}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
