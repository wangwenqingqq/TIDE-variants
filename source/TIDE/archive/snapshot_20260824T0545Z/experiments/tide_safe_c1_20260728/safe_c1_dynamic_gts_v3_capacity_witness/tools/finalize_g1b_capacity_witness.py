#!/usr/bin/env python3
"""CPU finalizer for one strict Safe-C1 G1B capacity-two witness run.

The finalizer accepts only a v3 bundle generated from the externally frozen v2
selection receipt.  It verifies the source receipts and independently replays
all three active sets with exact quantized squared-L2.  It makes no performance,
rebuild, range, base-delete, complete-C3, or all-input tie claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from safe_c1_v3_bundle_contract import (
    BundleContractError,
    CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
    OP_DELETE,
    OP_INSERT,
    OP_KNN,
    load_current_runner_identity_stable_id_layout,
    load_i16,
    parse_trace,
)
from preflight_g1b_capacity_witness import ROOT, validate_bundle

RUNS_ROOT = Path('/workspace/experiments/tide_safe_c1_20260728/runs')
ENGINE_SCHEMA = 'safe-c1-g1b-capacity-witness-v3-search-native'
STATIC_SCHEMA = 'safe-c1-g1b-static-contract-audit-v3-search-native'
RESULT_SCHEMA = 'safe-c1-g1b-v3-capacity-witness-result'
EXPECTED_IDS = (4330, 4357, 4399)
EXPECTED_LEAF = 133


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + '.tmp')
    write_json(temporary, value)
    os.replace(temporary, path)


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


def load_jsonl(path: Path, errors: list[str]) -> tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]:
    meta: dict[str, Any] | None = None
    by_op: dict[int, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError as exc:
        errors.append(f'cannot_read_engine_jsonl:{exc}')
        return meta, by_op
    for line_no, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f'bad_engine_jsonl_line:{line_no}:{exc}')
            continue
        if not isinstance(row, dict):
            errors.append(f'nonobject_engine_jsonl_line:{line_no}')
            continue
        if row.get('record') == 'meta':
            if meta is not None:
                errors.append('duplicate_engine_meta')
            meta = row
            continue
        op = row.get('op_index')
        if type(op) is not int or op in by_op:
            errors.append(f'bad_or_duplicate_engine_op_index:{line_no}')
            continue
        by_op[op] = row
    if meta is None:
        errors.append('missing_engine_meta')
    return meta, by_op


def record(errors: list[str], examples: list[dict[str, Any]], message: str, **detail: Any) -> None:
    errors.append(message)
    if len(examples) < 8:
        examples.append({'issue': message, **detail})


def close_distance(observed: float, squared: int) -> bool:
    return math.isclose(observed, math.sqrt(squared), rel_tol=1e-8, abs_tol=2e-6)


def validate_full_active_exact_oracle(bundle: Path, by_op: dict[int, dict[str, Any],], errors: list[str]) -> dict[str, Any]:
    oracle: dict[str, Any] = {
        'schema': 'safe-c1-g1b-v3-independent-full-active-exact-topk-oracle',
        'method': ('CPU Python-integer replay of every strict G1B query active set; exact quantized squared-L2 '
                   'top-k IDs/order and reported L2 checks under explicit identity-only v3 runner layout'),
        'selected_trace_only': True,
        'global_canonical_distance_stable_id_proof': False,
        'queries_checked': 0,
        'errors': [],
        'mismatch_examples': [],
    }
    local_errors: list[str] = []
    examples: list[dict[str, Any]] = []
    try:
        header, events = parse_trace(bundle / 'trace.e1gtrc')
        pool = load_i16(bundle / 'pool.i16', header.pool_n * header.dimension)
        queries = load_i16(bundle / 'queries.i16', header.query_n * header.dimension)
        mapping, base = load_current_runner_identity_stable_id_layout(bundle, header)
    except (BundleContractError, OSError) as exc:
        record(local_errors, examples, 'oracle_bundle_parse_or_identity_failed', error=str(exc))
        oracle['errors'] = local_errors; oracle['mismatch_examples'] = examples; oracle['status'] = 'FAIL'
        errors.extend(local_errors)
        return oracle
    oracle['input_sha256'] = {name: sha256(bundle / name) for name in (
        'pool.i16', 'queries.i16', 'trace.e1gtrc', 'initial_base_stable_ids.i32', 'stable_id_to_pool_row.i32')}
    oracle['stable_id_layout'] = {
        'mode': CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
        'mapping_identity_verified': mapping == list(range(header.pool_n)),
        'initial_base_identity_verified': base == list(range(header.base_n)),
    }
    active: set[int] = set(base)
    base_set = set(base)
    counts = {'id_order_mismatch': 0, 'membership_mismatch': 0, 'distance_mismatch': 0,
              'malformed_result': 0, 'dynamic_boundary_tie': 0, 'self_top1_mismatch': 0,
              'trace_replay_error': 0}

    def d2(stable_id: int, query_id: int) -> int:
        row = mapping[stable_id]
        return sum((int(pool[row * header.dimension + axis]) - int(queries[query_id * header.dimension + axis])) ** 2
                   for axis in range(header.dimension))

    for event in events:
        if event.op == OP_INSERT:
            if event.argument in active or not header.base_n <= event.argument < header.pool_n:
                counts['trace_replay_error'] += 1
                record(local_errors, examples, 'oracle_invalid_insert', op_index=event.op_index, stable_id=event.argument)
            else:
                active.add(event.argument)
            continue
        if event.op == OP_DELETE:
            if event.argument not in active or event.argument < header.base_n:
                counts['trace_replay_error'] += 1
                record(local_errors, examples, 'oracle_invalid_delete', op_index=event.op_index, stable_id=event.argument)
            else:
                active.remove(event.argument)
            continue
        if event.op != OP_KNN or not 0 <= event.argument < header.query_n:
            counts['trace_replay_error'] += 1
            record(local_errors, examples, 'oracle_invalid_nonknn_event', op_index=event.op_index)
            continue
        query_id = event.argument
        ranked = sorted((d2(stable_id, query_id), stable_id) for stable_id in active)
        oracle['queries_checked'] += 1
        if len(ranked) <= header.k or ranked[header.k - 1][0] == ranked[header.k][0]:
            counts['dynamic_boundary_tie'] += 1
            record(local_errors, examples, 'oracle_dynamic_k_boundary_tie_or_short_active',
                   op_index=event.op_index, active_count=len(ranked))
        expected_ids = [stable_id for _, stable_id in ranked[:header.k]]
        expected_self = EXPECTED_IDS[query_id]
        if not ranked or ranked[0][1] != expected_self or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
            counts['self_top1_mismatch'] += 1
            record(local_errors, examples, 'oracle_inserted_self_not_unique_top1',
                   op_index=event.op_index, expected_self=expected_self,
                   top2=ranked[:2])
        receipt = by_op.get(event.op_index)
        if not isinstance(receipt, dict) or receipt.get('record') != 'query' or \
                receipt.get('kind') != 'knn' or receipt.get('query_id') != query_id:
            counts['malformed_result'] += 1
            record(local_errors, examples, 'oracle_missing_or_malformed_query_receipt', op_index=event.op_index)
            continue
        result_rows = receipt.get('results')
        if not isinstance(result_rows, list) or len(result_rows) != header.k:
            counts['malformed_result'] += 1
            record(local_errors, examples, 'oracle_wrong_result_count', op_index=event.op_index)
            continue
        observed_ids: list[int] = []
        observed_d: list[float] = []
        valid = True
        for row in result_rows:
            if not isinstance(row, list) or len(row) != 2 or type(row[0]) is not int or \
                    isinstance(row[1], bool) or not isinstance(row[1], (int, float)) or not math.isfinite(float(row[1])):
                valid = False; break
            observed_ids.append(row[0]); observed_d.append(float(row[1]))
        if not valid or len(set(observed_ids)) != len(observed_ids) or any(item not in active for item in observed_ids):
            counts['malformed_result'] += 1
            record(local_errors, examples, 'oracle_invalid_result_shape_or_ids', op_index=event.op_index)
            continue
        if observed_ids != expected_ids:
            counts['id_order_mismatch'] += 1
            if set(observed_ids) != set(expected_ids):
                counts['membership_mismatch'] += 1
            record(local_errors, examples, 'oracle_full_active_exact_topk_id_order_mismatch',
                   op_index=event.op_index, expected_ids=expected_ids, observed_ids=observed_ids)
        for stable_id, distance in zip(observed_ids, observed_d):
            if not close_distance(distance, d2(stable_id, query_id)):
                counts['distance_mismatch'] += 1
                record(local_errors, examples, 'oracle_reported_distance_mismatch',
                       op_index=event.op_index, stable_id=stable_id, observed=distance,
                       expected=math.sqrt(d2(stable_id, query_id)))
    if active != base_set:
        counts['trace_replay_error'] += 1
        record(local_errors, examples, 'oracle_final_active_set_not_base_only', active_count=len(active))
    oracle.update({f'{key}_count': value for key, value in counts.items()})
    oracle['errors'] = local_errors
    oracle['mismatch_examples'] = examples
    oracle['status'] = 'PASS' if not local_errors else 'FAIL'
    errors.extend(local_errors)
    return oracle


def check_update(row: object, *, op: int, stable_id: int, placement: str,
                 sidecar_leaf: int, certified_leaf: int, fallback: str,
                 errors: list[str]) -> None:
    if not isinstance(row, dict) or row.get('record') != 'update' or row.get('op_index') != op or \
            row.get('stable_id') != stable_id or row.get('placement') != placement or \
            row.get('sidecar_leaf_id') != sidecar_leaf or row.get('certified_leaf_id') != certified_leaf or \
            row.get('fallback_reason') != fallback:
        errors.append(f'update_receipt_mismatch:op={op}')


def validate_capacity_receipts(meta: dict[str, Any] | None, by_op: dict[int, dict[str, Any]],
                               summary: dict[str, Any], errors: list[str],
                               expected_v2_selection_sha256: str | None = None) -> None:
    if not isinstance(meta, dict) or meta.get('schema') != ENGINE_SCHEMA:
        errors.append('engine_meta_wrong_schema')
        return
    contract = meta.get('capacity_witness_contract')
    expected_contract = {'leaf_capacity': 2, 'sidecar_leaf_id': EXPECTED_LEAF,
                         'stable_ids': list(EXPECTED_IDS), 'query_ids': [0, 1, 2]}
    if not isinstance(contract, dict) or any(contract.get(key) != value for key, value in expected_contract.items()) or \
            not isinstance(contract.get('v2_selection_sha256'), str):
        errors.append('engine_meta_capacity_contract_mismatch')
    elif expected_v2_selection_sha256 is not None and \
            contract.get('v2_selection_sha256') != expected_v2_selection_sha256:
        errors.append('engine_meta_external_v2_selection_sha_binding_mismatch')
    check_update(by_op.get(0), op=0, stable_id=EXPECTED_IDS[0], placement='direct',
                 sidecar_leaf=EXPECTED_LEAF, certified_leaf=EXPECTED_LEAF, fallback='none', errors=errors)
    check_update(by_op.get(2), op=2, stable_id=EXPECTED_IDS[1], placement='direct',
                 sidecar_leaf=EXPECTED_LEAF, certified_leaf=EXPECTED_LEAF, fallback='none', errors=errors)
    check_update(by_op.get(4), op=4, stable_id=EXPECTED_IDS[2], placement='delta',
                 sidecar_leaf=-1, certified_leaf=EXPECTED_LEAF, fallback='capacity_full', errors=errors)
    check_update(by_op.get(6), op=6, stable_id=EXPECTED_IDS[2], placement='deleted',
                 sidecar_leaf=-1, certified_leaf=EXPECTED_LEAF, fallback='capacity_full', errors=errors)
    check_update(by_op.get(7), op=7, stable_id=EXPECTED_IDS[1], placement='deleted',
                 sidecar_leaf=EXPECTED_LEAF, certified_leaf=EXPECTED_LEAF, fallback='none', errors=errors)
    check_update(by_op.get(8), op=8, stable_id=EXPECTED_IDS[0], placement='deleted',
                 sidecar_leaf=EXPECTED_LEAF, certified_leaf=EXPECTED_LEAF, fallback='none', errors=errors)
    expected_query_tiers = {1: (0, 1, 0), 3: (1, 2, 0), 5: (2, 2, 1)}
    for op, (query_id, sidecar_count, delta_count) in expected_query_tiers.items():
        row = by_op.get(op)
        if not isinstance(row, dict) or row.get('record') != 'query' or row.get('query_id') != query_id or \
                row.get('sidecar_candidate_count') != sidecar_count or row.get('delta_candidate_count') != delta_count or \
                not isinstance(row.get('gts_visited_leaf_ids'), list) or EXPECTED_LEAF not in row['gts_visited_leaf_ids']:
            errors.append(f'query_visibility_or_tier_receipt_mismatch:op={op}')
    updates = summary.get('updates')
    if not isinstance(updates, dict) or updates != {
        'insert': 3, 'delete': 3, 'direct_insert': 2, 'delta_insert': 1,
        'certificate_reject': 0, 'capacity_reject': 1, 'direct_live_final': 0, 'delta_live_final': 0,
    }:
        errors.append('summary_update_counts_not_strict_capacity_witness')
    if summary.get('status') != 'PASS_G1B_CAPACITY_WITNESS_PENDING_INDEPENDENT_VALIDATOR' or \
            summary.get('schema') != ENGINE_SCHEMA or summary.get('archived_incremental_updater_used') is not False:
        errors.append('summary_schema_status_or_legacy_updater_mismatch')
    summary_contract = summary.get('capacity_witness_contract')
    if not isinstance(contract, dict) or not isinstance(summary_contract, dict) or \
            summary_contract != contract:
        errors.append('summary_capacity_contract_does_not_match_engine_meta')
    base = summary.get('base_tree')
    if not isinstance(base, dict) or not isinstance(base.get('frozen_hash'), int) or \
            not isinstance(base.get('frozen_logical_leaf_id_hash'), int):
        errors.append('summary_missing_frozen_base_receipt')
    if not isinstance(base, dict) or summary.get('final_active_count') != base.get('base_n'):
        errors.append('summary_final_active_count_not_base_n')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--engine-jsonl', type=Path, required=True)
    parser.add_argument('--engine-summary', type=Path, required=True)
    parser.add_argument('--static-contract', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, bundle, run_root, out = (args.root.resolve(), args.bundle.resolve(),
                                   args.run_root.resolve(), args.out.resolve())
    errors: list[str] = []
    if root != ROOT.resolve() or not in_root(bundle, root / 'bundles'):
        errors.append('wrong_v3_root_or_bundle_boundary')
    if not in_root(run_root, RUNS_ROOT) or not in_root(args.engine_jsonl.resolve(), run_root) or \
            not in_root(args.engine_summary.resolve(), run_root):
        errors.append('run_or_engine_outputs_outside_new_runs_root')
    if out.parent != bundle or out.name != 'g1b_capacity_witness_v3.json' or out.exists():
        errors.append('result_output_must_be_new_bundle_local_g1b_capacity_witness_v3_json')
    preflight, preflight_errors = validate_bundle(root, bundle)
    if preflight_errors:
        errors.extend(f'bundle_preflight:{value}' for value in preflight_errors)
    static = json_object(args.static_contract, errors, 'static_contract')
    if static.get('schema') != STATIC_SCHEMA or static.get('status') != 'PASS_CPU_ONLY_STATIC_CONTRACT' or \
            static.get('gpu_used') is not False or static.get('cuda_binary_executed') is not False:
        errors.append('static_contract_schema_or_status_mismatch')
    artifact_hashes = static.get('artifacts_sha256')
    if not isinstance(artifact_hashes, dict) or artifact_hashes.get('binary') != sha256(args.binary):
        errors.append('static_contract_binary_sha_binding_mismatch')
    manifest = json_object(bundle / 'manifest.json', errors, 'manifest')
    if manifest.get('status') != 'PREPARED_CPU_ONLY_READY_FOR_G1B_CAPACITY_WITNESS':
        errors.append('bundle_not_pending_for_first_g1b_finalization')
    meta, by_op = load_jsonl(args.engine_jsonl, errors)
    summary = json_object(args.engine_summary, errors, 'engine_summary')
    if set(by_op) != set(range(9)):
        errors.append('engine_jsonl_not_exactly_one_receipt_per_nine_trace_ops')
    provenance = manifest.get('external_v2_selection_provenance')
    expected_v2_selection_sha256 = provenance.get('sha256') if isinstance(provenance, dict) and \
        isinstance(provenance.get('sha256'), str) else None
    if expected_v2_selection_sha256 is None:
        errors.append('manifest_missing_external_v2_selection_sha_for_engine_binding')
    validate_capacity_receipts(meta, by_op, summary, errors, expected_v2_selection_sha256)
    oracle = validate_full_active_exact_oracle(bundle, by_op, errors)
    result = {
        'schema': RESULT_SCHEMA,
        'status': 'PASS_G1B_CAPACITY_WITNESS' if not errors else 'FAIL_G1B_CAPACITY_WITNESS',
        'scope': ('strict capacity-two receipt only: c had certified_leaf_id=133 then exact delta due to '
                  'capacity_full; no performance, rebuild, range, base-delete, complete-C3, or all-input tie claim'),
        'gpu_used': True,
        'binary_sha256': sha256(args.binary) if args.binary.is_file() else None,
        'bundle_input_manifest_sha256': sha256(bundle / 'manifest.json') if (bundle / 'manifest.json').is_file() else None,
        'static_contract_sha256': sha256(args.static_contract) if args.static_contract.is_file() else None,
        'engine_results_sha256': sha256(args.engine_jsonl) if args.engine_jsonl.is_file() else None,
        'engine_summary_sha256': sha256(args.engine_summary) if args.engine_summary.is_file() else None,
        'run_root': str(run_root),
        'external_v2_selection_provenance': manifest.get('external_v2_selection_provenance'),
        'independent_full_active_exact_topk_oracle': oracle,
        'error_count': len(errors), 'errors': errors,
    }
    if errors:
        print('FAIL_G1B_CAPACITY_WITNESS')
        return 2
    atomic_write_json(out, result)
    updated = dict(manifest)
    updated['status'] = 'FINALIZED_G1B_CAPACITY_WITNESS'
    updated['g1b_capacity_witness_result'] = {
        'artifact': out.name, 'artifact_sha256': sha256(out), 'status': result['status'],
        'run_root': str(run_root), 'scope': result['scope'],
    }
    atomic_write_json(bundle / 'manifest.json', updated)
    print('PASS_G1B_CAPACITY_WITNESS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
