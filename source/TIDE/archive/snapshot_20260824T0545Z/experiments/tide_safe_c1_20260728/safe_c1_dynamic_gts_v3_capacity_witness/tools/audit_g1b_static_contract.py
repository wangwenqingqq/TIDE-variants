#!/usr/bin/env python3
"""CPU-only static audit for the isolated Safe-C1 G1B capacity witness.

It establishes only source/artifact binding and scope boundaries.  It never
runs a CUDA binary or GPU query and cannot establish a performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v3_capacity_witness')
V1_ROOT = Path('/workspace/experiments/tide_safe_c1_20260727/safe_c1_dynamic_gts_v1')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(text: str, token: str, errors: list[str], label: str) -> None:
    if token not in text:
        errors.append(f'missing:{label}:{token}')


def forbid(text: str, token: str, errors: list[str], label: str) -> None:
    if token in text:
        errors.append(f'forbidden:{label}:{token}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    if root != ROOT.resolve():
        errors.append('audit_root_must_equal_canonical_v3_root')
    if root == V1_ROOT or V1_ROOT in root.parents:
        errors.append('audit_v1_root_forbidden')
    # v3 may read the frozen v2 selection artifact through an explicit external
    # path, but it must not retain copied v2 runner/tool artifacts locally.
    for stale in (
        root / 'bin/GTS_safe_c1_g1_v2_search_native',
        root / 'tools/safe_c1_v2_bundle_contract.py',
        root / 'tools/run_g1_v2_certificate_selection_guarded.sh',
        root / 'tools/finalize_g1_v2_candidate_selection.py',
    ):
        if stale.exists():
            errors.append(f'copied_v2_artifact_forbidden:{stale.name}')
    paths = {
        'source': root / 'src/safe_c1_dynamic_gts.cu',
        'binary': root / 'bin/GTS_safe_c1_g1b_v3_capacity_witness',
        'search_receipt_header': root / 'include/safe_search_v2.cuh',
        'routing_certificate_header': root / 'include/safe_c1_search_native_routing_certificate.hpp',
        'tie_witness_header': root / 'include/safe_c1_tie_free_witness.hpp',
        'contract_module': root / 'tools/safe_c1_v3_bundle_contract.py',
        'generator': root / 'tools/make_g1b_capacity_witness_bundle.py',
        'preflight': root / 'tools/preflight_g1b_capacity_witness.py',
        'finalizer': root / 'tools/finalize_g1b_capacity_witness.py',
        'guard': root / 'tools/run_g1b_capacity_witness_guarded.sh',
        'pipeline_test': root / 'tools/test_g1b_capacity_witness_pipeline.py',
    }
    for label, path in paths.items():
        if not path.is_file():
            errors.append(f'missing_file:{label}:{path}')
    text = {label: path.read_text(encoding='utf-8', errors='replace')
            if path.suffix != '' and path.is_file() and label != 'binary' else ''
            for label, path in paths.items()}
    src = text['source']
    # The query merge remains a real GTS traversal receipt + selected sidecars
    # + exact global delta.  No host all-active scan is permitted in this runner.
    for token in (
        '#include "safe_search_v2.cuh"', 'run_gts_base_topk_with_receipt',
        'state.sidecar_candidates_for(gts_answer.visited_leaf_ids)', 'exact_candidate_l2',
        'FrozenTreeSnapshot', 'frozen.assert_unchanged(runtime)',
        'require_identity_stable_id_layout', 'validate_tie_free_witness_trace',
        'only_strict_search_native_child', 'distance < nodes[next].min_dis - kStrictEpsilon',
        'tree mutation, but it is not a routing bound',
    ):
        need(src, token, errors, 'source')
    for token in ('#include "incremental_insert.cuh"', '#include "update.cuh"',
                  'updateIndexRnn(', 'deleteIncrementalInsert(', 'exact_live_l2'):
        forbid(src, token, errors, 'source')
    # Specific capacity evidence receipt: distinguish certificate failure from
    # certificate success followed by full-sidecar fallback.
    for token in (
        'struct InsertReceipt', 'certified_leaf_id', 'InsertFallbackReason',
        'kCertificateReject', 'kCapacityFull', 'fallback_reason',
        'G1BCapacityWitnessContract', 'validate_g1b_capacity_witness_trace',
        'G1B requires c to pass the same-leaf certificate then fall back only for capacity',
        'expected_sidecar_ids', 'expected_delta_ids',
        'safe-c1-g1b-capacity-witness-v3-search-native',
        'PASS_G1B_CAPACITY_WITNESS_PENDING_INDEPENDENT_VALIDATOR',
        'not_established', 'performance benefit',
    ):
        need(src, token, errors, 'source')
    # Contract/tie CPU tools must stay CPU-only and retain the external-v2
    # read-only boundary rather than importing v2 output as a v3 result.
    for label, token in (
        ('generator', 'copied_into_v3_bundle\': False'),
        ('generator', 'external read-only'),
        ('preflight', 'cuda_binary_executed\': False'),
        ('preflight', 'external_v2_selection_must_not_be_copied_into_v3_bundle'),
        ('finalizer', 'independent_full_active_exact_topk'),
        ('finalizer', 'capacity_full'),
        ('guard', 'DO NOT execute without explicit approval'),
        ('guard', 'no_process_control'),
        ('pipeline_test', 'negative'),
    ):
        need(text[label], token, errors, label)
    # Receipt instrumentation must precede legacy native leaf scan.
    receipt = text['search_receipt_header']
    marker = receipt.find('Safe-C1 instrumentation: materialize only the (query, leaf) pairs')
    leaf_scan = receipt.find('dataProcessKnnVec<<<', marker if marker >= 0 else 0)
    if marker < 0 or leaf_scan < 0 or marker > leaf_scan:
        errors.append('receipt_not_before_native_leaf_scan')
    # Static binding: identity/tie/capacity trace contract must be parsed before
    # the base tree construction and before any runner output is emitted.
    identity = src.find('require_identity_stable_id_layout(args.bundle, header)')
    g1b_contract = src.find('validate_g1b_capacity_witness_trace(header, events, capacity_contract')
    tie = src.find('validate_tie_free_witness_trace(header, events, tie_witness_checker)')
    tree = src.find('indexConstru(runtime.data_d')
    output = src.find('std::ofstream output(args.output)')
    if min(identity, g1b_contract, tie, tree, output) < 0 or not (identity < tree and g1b_contract < tree and tie < tree < output):
        errors.append('identity_tie_or_capacity_contract_not_before_tree_and_output')
    report = {
        'schema': 'safe-c1-g1b-static-contract-audit-v3-search-native',
        'status': 'PASS_CPU_ONLY_STATIC_CONTRACT' if not errors else 'FAIL_CPU_ONLY_STATIC_CONTRACT',
        'gpu_used': False,
        'cuda_binary_executed': False,
        'scope': ('static source/artifact binding only; strict G1B capacity-two witness; no timing, throughput, '
                  'rebuild, range, base-delete, complete-C3, or all-input tie claim'),
        'root': str(root),
        'artifacts_sha256': {label: sha256(path) for label, path in paths.items() if path.is_file()},
        'errors': errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(report['status'])
    return 0 if not errors else 2


if __name__ == '__main__':
    raise SystemExit(main())
