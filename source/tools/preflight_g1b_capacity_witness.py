#!/usr/bin/env python3
"""CPU-only preflight for a fresh v3 G1B capacity-two witness bundle.

This checks immutable input bindings and exact trace semantics before a guarded
runner is even allowed to inspect GPU0.  It does not execute CUDA, nvidia-smi,
or the GTS binary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safe_c1_v3_bundle_contract import (
    BOUNDED_TIE_PREFLIGHT_STATUS,
    CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
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

ROOT = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v3_capacity_witness')
EXPECTED_EXTERNAL = Path(
    '/workspace/experiments/tide_safe_c1_20260728/'
    'safe_c1_dynamic_gts_v2_search_native/bundles/'
    'g1a_v2_identity_oracle_selection_20260728T110122Z/candidate_selection_v2.json'
)
EXPECTED_IDS = (4330, 4357, 4399)
EXPECTED_LEAF = 133
ENGINE_SCHEMA = 'safe-c1-g1b-capacity-witness-v3-search-native'
MANIFEST_SCHEMA = 'safe-c1-g1b-v3-capacity-witness-bundle-manifest'
CONTRACT_SCHEMA = 'safe-c1-g1b-capacity-witness-contract-v3'
EXTERNAL_SELECTION_PAYLOAD_NAMES = (
    'pool.i16', 'queries.i16', 'trace.e1gtrc',
    'initial_base_stable_ids.i32', 'stable_id_to_pool_row.i32',
)
COPIED_IMMUTABLE_PAYLOAD_NAMES = ('pool.i16', 'initial_base_stable_ids.i32', 'stable_id_to_pool_row.i32')


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def in_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def json_object(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f'{label}_parse_failed:{exc}')
        return {}
    if not isinstance(value, dict):
        errors.append(f'{label}_not_object')
        return {}
    return value


def parse_contract(path: Path, errors: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError as exc:
        errors.append(f'contract_read_failed:{exc}')
        return output
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.count('=') != 1:
            errors.append(f'contract_malformed_line:{number}')
            continue
        key, value = (part.strip() for part in line.split('=', 1))
        if not key or not value or key in output:
            errors.append(f'contract_empty_or_duplicate_field:{number}')
            continue
        output[key] = value
    return output


def _as_int(contract: dict[str, str], key: str, errors: list[str]) -> int | None:
    raw = contract.get(key)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        errors.append(f'contract_noninteger:{key}')
        return None


def validate_external_selection(manifest: dict[str, Any], errors: list[str]) -> None:
    provenance = manifest.get('external_v2_selection_provenance')
    if not isinstance(provenance, dict):
        errors.append('missing_external_v2_selection_provenance')
        return
    path = provenance.get('path')
    expected_sha = provenance.get('sha256')
    if not isinstance(path, str) or Path(path).resolve() != EXPECTED_EXTERNAL.resolve():
        errors.append('external_v2_selection_path_not_expected_read_only_artifact')
        return
    if provenance.get('copied_into_v3_bundle') is not False:
        errors.append('external_v2_selection_must_not_be_copied_into_v3_bundle')
    if provenance.get('required_same_leaf_id') != EXPECTED_LEAF or \
            provenance.get('required_stable_ids') != list(EXPECTED_IDS):
        errors.append('external_v2_selection_expected_group_mismatch')
    if not isinstance(expected_sha, str) or not EXPECTED_EXTERNAL.is_file() or \
            sha256(EXPECTED_EXTERNAL) != expected_sha:
        errors.append('external_v2_selection_sha_or_presence_mismatch')
        return
    external = json_object(EXPECTED_EXTERNAL, errors, 'external_v2_selection')
    if external.get('schema') != 'safe-c1-g1-v2-candidate-selection' or \
            external.get('status') != 'READY_FOR_V2_WITNESS' or \
            external.get('v1_evidence_used') is not False:
        errors.append('external_v2_selection_not_ready_or_wrong_provenance')
    recorded_hashes = external.get('bundle_input_files_sha256')
    source_info = manifest.get('external_v2_payload_source')
    if not isinstance(recorded_hashes, dict) or not isinstance(source_info, dict) or \
            source_info.get('path') != str(EXPECTED_EXTERNAL.parent) or not isinstance(source_info.get('sha256'), dict):
        errors.append('external_v2_selection_payload_provenance_missing')
    else:
        source_hashes = source_info['sha256']
        for name in EXTERNAL_SELECTION_PAYLOAD_NAMES:
            expected = recorded_hashes.get(name)
            source = EXPECTED_EXTERNAL.parent / name
            if not isinstance(expected, str) or not source.is_file() or sha256(source) != expected or \
                    source_hashes.get(name) != expected:
                errors.append(f'external_v2_selection_payload_sha_mismatch:{name}')
        # These v3 files are byte-for-byte copies rather than derived query rows.
        for name in COPIED_IMMUTABLE_PAYLOAD_NAMES:
            local = manifest.get('files_sha256', {}).get(name) if isinstance(manifest.get('files_sha256'), dict) else None
            expected = recorded_hashes.get(name)
            if local != expected:
                errors.append(f'v3_copied_immutable_payload_not_bound_to_external_selection:{name}')
    group = external.get('g1b_same_leaf_direct_group')
    if not isinstance(group, dict) or group.get('sidecar_leaf_id') != EXPECTED_LEAF or \
            group.get('stable_ids') != list(EXPECTED_IDS):
        errors.append('external_v2_selection_group_changed')


def validate_bundle(root: Path, bundle: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    root = root.resolve()
    bundle = bundle.resolve()
    report: dict[str, Any] = {
        'schema': 'safe-c1-g1b-v3-capacity-witness-preflight',
        'gpu_used': False,
        'cuda_binary_executed': False,
        'root': str(root),
        'bundle': str(bundle),
        'engine_schema_required': ENGINE_SCHEMA,
        'checks': {},
        'errors': errors,
    }
    if root != ROOT.resolve():
        errors.append('wrong_v3_root')
    if not in_root(bundle, root / 'bundles'):
        errors.append('bundle_outside_v3_bundles_root')
    if not bundle.is_dir():
        errors.append('bundle_not_directory')
        report['status'] = 'FAIL'
        return report, errors
    manifest = json_object(bundle / 'manifest.json', errors, 'manifest')
    if manifest.get('schema') != MANIFEST_SCHEMA:
        errors.append('wrong_bundle_manifest_schema')
    if manifest.get('status') != 'PREPARED_CPU_ONLY_READY_FOR_G1B_CAPACITY_WITNESS':
        errors.append('bundle_not_fresh_or_not_pending_g1b')
    if manifest.get('bundle_kind') != 'g1b_capacity_witness' or \
            manifest.get('gpu_used') is not False or manifest.get('cuda_binary_executed') is not False:
        errors.append('bundle_kind_or_cpu_only_state_mismatch')
    if manifest.get('engine_schema_required') != ENGINE_SCHEMA or \
            manifest.get('stable_id_layout') != CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE:
        errors.append('bundle_engine_or_identity_layout_mismatch')
    validate_external_selection(manifest, errors)

    listed = manifest.get('files_sha256')
    if not isinstance(listed, dict):
        errors.append('missing_files_sha256')
    else:
        for name, expected in listed.items():
            path = bundle / name
            if not isinstance(name, str) or Path(name).name != name or not path.is_file() or \
                    not isinstance(expected, str) or sha256(path) != expected:
                errors.append(f'bundle_input_sha_mismatch:{name}')
        forbidden = {'candidate_selection_v2.json', 'candidate_selection_v3.json'}
        if any(name in forbidden for name in listed):
            errors.append('v2_or_v3_result_artifact_must_not_be_bundle_input')

    contract = parse_contract(bundle / 'g1b_capacity_contract.txt', errors)
    required_contract = {
        'schema', 'leaf_capacity', 'sidecar_leaf_id', 'stable_id_a', 'stable_id_b',
        'stable_id_c', 'query_id_a', 'query_id_b', 'query_id_c', 'v2_selection_sha256',
    }
    if set(contract) != required_contract:
        errors.append('contract_field_set_mismatch')
    if contract.get('schema') != CONTRACT_SCHEMA:
        errors.append('contract_wrong_schema')
    if _as_int(contract, 'leaf_capacity', errors) != 2 or \
            _as_int(contract, 'sidecar_leaf_id', errors) != EXPECTED_LEAF or \
            tuple(_as_int(contract, f'stable_id_{name}', errors) for name in ('a', 'b', 'c')) != EXPECTED_IDS or \
            tuple(_as_int(contract, f'query_id_{name}', errors) for name in ('a', 'b', 'c')) != (0, 1, 2):
        errors.append('contract_capacity_leaf_ids_or_query_ids_mismatch')
    provenance = manifest.get('external_v2_selection_provenance')
    if isinstance(provenance, dict) and contract.get('v2_selection_sha256') != provenance.get('sha256'):
        errors.append('contract_external_selection_sha_binding_mismatch')

    try:
        header, events = parse_trace(bundle / 'trace.e1gtrc')
        pool = load_i16(bundle / 'pool.i16', header.pool_n * header.dimension)
        queries = load_i16(bundle / 'queries.i16', header.query_n * header.dimension)
        stable_to_row, initial_base = load_current_runner_identity_stable_id_layout(bundle, header)
    except (BundleContractError, OSError) as exc:
        errors.append(f'bundle_payload_parse_or_identity_failed:{exc}')
        header = events = pool = queries = None  # type: ignore[assignment]
        stable_to_row = initial_base = None  # type: ignore[assignment]
    if header is not None:
        if header.query_n != 3 or header.event_count != 9:
            errors.append('trace_header_not_strict_g1b_shape')
        expected_events = [
            (OP_INSERT, EXPECTED_IDS[0]), (OP_KNN, 0),
            (OP_INSERT, EXPECTED_IDS[1]), (OP_KNN, 1),
            (OP_INSERT, EXPECTED_IDS[2]), (OP_KNN, 2),
            (OP_DELETE, EXPECTED_IDS[2]), (OP_DELETE, EXPECTED_IDS[1]),
            (OP_DELETE, EXPECTED_IDS[0]),
        ]
        if len(events) != len(expected_events) or any(
                event.op_index != index or (event.op, event.argument) != expected_events[index]
                for index, event in enumerate(events)):
            errors.append('trace_not_exact_IaQaIbQbIcQcDcDbDa')
        if stable_to_row != list(range(header.pool_n)) or initial_base != list(range(header.base_n)):
            errors.append('identity_layout_recheck_failed')
        for query_id, stable_id in enumerate(EXPECTED_IDS):
            if not all(pool[stable_id * header.dimension + axis] == queries[query_id * header.dimension + axis]
                       for axis in range(header.dimension)):
                errors.append(f'query_payload_not_self_vector:{query_id}')
        try:
            audit = validate_tie_free_witness(pool, queries, header, events)
            report['checks']['tie_free_witness'] = audit
            if audit.get('status') != BOUNDED_TIE_PREFLIGHT_STATUS:
                errors.append('tie_free_preflight_wrong_status')
        except BundleContractError as exc:
            errors.append(f'tie_free_witness_failed:{exc}')
    report['checks']['frozen_base_runtime_requirement'] = {
        'required': True,
        'meaning': 'v3 runner must snapshot and assert immutable TN/id_list/interval payload after every trace event; this preflight does not execute CUDA',
    }
    report['status'] = 'PASS_CPU_ONLY_READY_FOR_G1B_CAPACITY_WITNESS' if not errors else 'FAIL'
    return report, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--mode', choices=['capacity-witness'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report, errors = validate_bundle(args.root, args.bundle)
    report['mode'] = args.mode
    write_json(args.out, report)
    print(report['status'])
    return 0 if not errors else 2


if __name__ == '__main__':
    raise SystemExit(main())
