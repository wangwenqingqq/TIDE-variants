#!/usr/bin/env python3
"""CPU-only preflight for the strict v5 stable-ID seeded G2 rebuild witness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safe_c1_v5_rebuild_bundle_contract import (
    BOUNDED_TIE_ABORT,
    CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
    DYNAMIC_POLICY,
    SCHEMA,
    STATIC_POLICY,
    BundleContractError,
    cint,
    load_i16,
    load_identity_layout,
    parse_trace,
    read_contract,
    replay_g2_trace,
    require_dynamic_k_boundary,
    require_static_epoch_tie_free,
    sha256,
    topk_exact,
    validate_strict_g2_trace,
)

ROOT = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v5_g2_artifact_pinned')
V3_ROOT = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v3_capacity_witness')
V3_BUNDLE = V3_ROOT / 'bundles/g1b_capacity_witness_20260728T114307Z_cpu'
V3_RESULT = V3_BUNDLE / 'g1b_capacity_witness_v3.json'
V2_ROOT = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v2_search_native')
V2_SELECTION = V2_ROOT / 'bundles/g1a_v2_identity_oracle_selection_20260728T110122Z/candidate_selection_v2.json'


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise BundleContractError(f'not_json_object:{path}')
    return value


def emit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def validate_bundle(root: Path, bundle: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    static_checks: list[dict[str, Any]] = []
    dynamic_checks: list[dict[str, Any]] = []
    replay: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    try:
        if root.resolve() != ROOT.resolve():
            errors.append('root_must_equal_canonical_v5_root')
        try:
            bundle.resolve().relative_to((ROOT / 'bundles').resolve())
        except ValueError:
            errors.append('bundle_must_be_under_v5_bundles_root')
        if not bundle.is_dir():
            errors.append('bundle_missing')
            raise BundleContractError('bundle_missing')
        for forbidden in ('g1b_capacity_witness_v3.json', 'candidate_selection_v2.json', 'engine_results.jsonl'):
            if (bundle / forbidden).exists():
                errors.append(f'external_result_copied_into_v5_bundle_forbidden:{forbidden}')
        manifest = read_json(bundle / 'manifest.json')
        if manifest.get('schema') != 'safe-c1-g2-rebuild-witness-v5-bundle-manifest':
            errors.append('wrong_bundle_manifest_schema')
        if manifest.get('status') != 'PREPARED_CPU_ONLY_G2_REBUILD_WITNESS':
            errors.append('bundle_not_fresh_cpu_prepared_status')
        if manifest.get('gpu_used') is not False or manifest.get('cuda_binary_executed') is not False:
            errors.append('bundle_claims_gpu_or_cuda_execution')
        if manifest.get('stable_id_layout') != CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE:
            errors.append('wrong_v5_stable_id_layout')
        files = manifest.get('files_sha256')
        if not isinstance(files, dict):
            errors.append('bundle_files_sha256_missing')
        else:
            for name, expected in files.items():
                path = bundle / str(name)
                if not path.is_file() or sha256(path) != expected:
                    errors.append(f'bundle_file_hash_mismatch:{name}')
        header, events = parse_trace(bundle / 'trace.g2trc')
        mapping, initial = load_identity_layout(bundle, header)
        if mapping != list(range(header.pool_n)) or initial != list(range(header.base_n)):
            errors.append('identity_layout_not_verified')
        contract = read_contract(bundle / 'g2_rebuild_contract.txt')
        validate_strict_g2_trace(header, events, contract)
        g1b = read_json(V3_RESULT)
        selection = read_json(V2_SELECTION)
        ext_g1b = manifest.get('external_g1b_pass_artifact')
        ext_selection = manifest.get('external_v2_selection')
        if not isinstance(ext_g1b, dict) or ext_g1b.get('path') != str(V3_RESULT) or \
                ext_g1b.get('sha256') != sha256(V3_RESULT) or ext_g1b.get('copied_into_v5_bundle') is not False:
            errors.append('external_g1b_binding_invalid')
        if not isinstance(ext_selection, dict) or ext_selection.get('path') != str(V2_SELECTION) or \
                ext_selection.get('sha256') != sha256(V2_SELECTION) or ext_selection.get('copied_into_v5_bundle') is not False:
            errors.append('external_v2_selection_binding_invalid')
        if g1b.get('status') != 'PASS_G1B_CAPACITY_WITNESS' or g1b.get('error_count') != 0:
            errors.append('external_g1b_artifact_not_pass')
        group = selection.get('g1b_same_leaf_direct_group')
        if not isinstance(group, dict):
            errors.append('external_v2_selection_missing_group')
        else:
            group_ids = group.get('stable_ids'); group_leaf = group.get('sidecar_leaf_id')
            expected_ids = [cint(contract, 'stable_id_a'), cint(contract, 'stable_id_b'), cint(contract, 'stable_id_c')]
            if group_ids != expected_ids or group_leaf != cint(contract, 'sidecar_leaf_id'):
                errors.append('contract_group_does_not_match_external_v2_selection')
        if contract.get('v3_g1b_result_sha256') != sha256(V3_RESULT) or \
                contract.get('v2_selection_sha256') != sha256(V2_SELECTION):
            errors.append('contract_external_sha_binding_invalid')
        v3_manifest = read_json(V3_BUNDLE / 'manifest.json')
        ext_payload = manifest.get('external_v3_immutable_payload_source')
        v3_hashes = {name: sha256(V3_BUNDLE / name) for name in ('pool.i16','queries.i16','stable_id_to_pool_row.i32','initial_base_stable_ids.i32')}
        if not isinstance(ext_payload, dict) or ext_payload.get('path') != str(V3_BUNDLE) or ext_payload.get('sha256') != v3_hashes:
            errors.append('external_v3_immutable_payload_binding_invalid')
        v3_files = v3_manifest.get('files_sha256')
        if not isinstance(v3_files, dict) or any(v3_files.get(name) != digest for name, digest in v3_hashes.items()):
            errors.append('external_v3_immutable_payload_hash_invalid')
        if any(sha256(bundle / name) != digest for name, digest in v3_hashes.items()):
            errors.append('local_v5_immutable_payload_not_exact_v3_copy')
        replay = replay_g2_trace(header, events, contract)
        pool = load_i16(bundle / 'pool.i16', header.pool_n * header.dimension)
        queries = load_i16(bundle / 'queries.i16', header.query_n * header.dimension)
        static_checks.extend(require_static_epoch_tie_free(pool, queries, replay['initial_base_ids'], header,
                                                           range(header.query_n), 'E0'))
        static_checks.extend(require_static_epoch_tie_free(pool, queries, replay['active_after_rebuild'], header,
                                                           range(header.query_n), 'E1'))
        for text_op, active_ids in replay['query_active_ids'].items():
            op = int(text_op)
            event = events[op]
            dynamic_checks.append(require_dynamic_k_boundary(pool, queries, active_ids, header,
                                                              event.argument, op))
        a = cint(contract, 'stable_id_a'); c = cint(contract, 'stable_id_c')
        if topk_exact(pool, queries, replay['query_active_ids']['2'], header,
                      cint(contract, 'query_id_before'))[0][1] != a:
            errors.append('a_not_unique_self_top1_before_rebuild')
        if topk_exact(pool, queries, replay['query_active_ids']['5'], header,
                      cint(contract, 'query_id_middle'))[0][1] != c:
            errors.append('c_not_unique_self_top1_before_rebuild')
        if topk_exact(pool, queries, replay['query_active_ids']['8'], header,
                      cint(contract, 'query_id_after'))[0][1] != c:
            errors.append('c_not_unique_self_top1_after_rebuild')
    except (BundleContractError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        errors.append(f'preflight_exception:{exc}')
    report = {
        'schema': 'safe-c1-g2-v5-rebuild-witness-preflight',
        'status': 'PASS_CPU_ONLY_G2_REBUILD_WITNESS_PREFLIGHT' if not errors else 'FAIL_CPU_ONLY_G2_REBUILD_WITNESS_PREFLIGHT',
        'gpu_used': False,
        'cuda_binary_executed': False,
        'scope': ('CPU contract only: one strict seeded stable-ID rebuild trace; no CUDA execution, no performance, '
                  'no general rebuild/range/C2/C3/all-input tie claim'),
        'root': str(root.resolve()), 'bundle': str(bundle.resolve()),
        'stable_id_layout': CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
        'static_base_policy': STATIC_POLICY, 'dynamic_k_boundary_policy': DYNAMIC_POLICY,
        'bounded_tie_abort': BOUNDED_TIE_ABORT,
        'static_epoch_checks': static_checks, 'dynamic_checks': dynamic_checks,
        'rebuild_replay': replay, 'errors': errors,
    }
    return report, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report, errors = validate_bundle(args.root.resolve(), args.bundle.resolve())
    emit(args.out, report)
    print(report['status'])
    return 0 if not errors else 2


if __name__ == '__main__':
    raise SystemExit(main())
