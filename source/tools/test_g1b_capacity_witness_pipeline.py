#!/usr/bin/env python3
"""CPU-only positive/negative fixtures for the strict v3 G1B pipeline."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path

from finalize_g1b_capacity_witness import validate_capacity_receipts, validate_full_active_exact_oracle
from preflight_g1b_capacity_witness import ROOT, validate_bundle
from safe_c1_v3_bundle_contract import OP_DELETE, OP_INSERT, OP_KNN, TraceEvent, load_current_runner_identity_stable_id_layout, load_i16, parse_trace, write_trace


def _receipt_fixture() -> tuple[dict, dict[int, dict], dict]:
    contract = {'leaf_capacity': 2, 'sidecar_leaf_id': 133, 'stable_ids': [4330, 4357, 4399],
                'query_ids': [0, 1, 2], 'v2_selection_sha256': '0' * 64}
    meta = {'record': 'meta', 'schema': 'safe-c1-g1b-capacity-witness-v3-search-native',
            'capacity_witness_contract': contract}
    by_op = {
        0: {'record': 'update', 'op_index': 0, 'stable_id': 4330, 'placement': 'direct',
            'sidecar_leaf_id': 133, 'certified_leaf_id': 133, 'fallback_reason': 'none'},
        1: {'record': 'query', 'kind': 'knn', 'op_index': 1, 'query_id': 0, 'sidecar_candidate_count': 1,
            'delta_candidate_count': 0, 'gts_visited_leaf_ids': [133]},
        2: {'record': 'update', 'op_index': 2, 'stable_id': 4357, 'placement': 'direct',
            'sidecar_leaf_id': 133, 'certified_leaf_id': 133, 'fallback_reason': 'none'},
        3: {'record': 'query', 'kind': 'knn', 'op_index': 3, 'query_id': 1, 'sidecar_candidate_count': 2,
            'delta_candidate_count': 0, 'gts_visited_leaf_ids': [133]},
        4: {'record': 'update', 'op_index': 4, 'stable_id': 4399, 'placement': 'delta',
            'sidecar_leaf_id': -1, 'certified_leaf_id': 133, 'fallback_reason': 'capacity_full'},
        5: {'record': 'query', 'kind': 'knn', 'op_index': 5, 'query_id': 2, 'sidecar_candidate_count': 2,
            'delta_candidate_count': 1, 'gts_visited_leaf_ids': [133]},
        6: {'record': 'update', 'op_index': 6, 'stable_id': 4399, 'placement': 'deleted',
            'sidecar_leaf_id': -1, 'certified_leaf_id': 133, 'fallback_reason': 'capacity_full'},
        7: {'record': 'update', 'op_index': 7, 'stable_id': 4357, 'placement': 'deleted',
            'sidecar_leaf_id': 133, 'certified_leaf_id': 133, 'fallback_reason': 'none'},
        8: {'record': 'update', 'op_index': 8, 'stable_id': 4330, 'placement': 'deleted',
            'sidecar_leaf_id': 133, 'certified_leaf_id': 133, 'fallback_reason': 'none'},
    }
    summary = {
        'schema': 'safe-c1-g1b-capacity-witness-v3-search-native',
        'status': 'PASS_G1B_CAPACITY_WITNESS_PENDING_INDEPENDENT_VALIDATOR',
        'archived_incremental_updater_used': False,
        'capacity_witness_contract': contract,
        'updates': {'insert': 3, 'delete': 3, 'direct_insert': 2, 'delta_insert': 1,
                    'certificate_reject': 0, 'capacity_reject': 1,
                    'direct_live_final': 0, 'delta_live_final': 0},
        'base_tree': {'base_n': 4096, 'frozen_hash': 1, 'frozen_logical_leaf_id_hash': 2},
        'final_active_count': 4096,
    }
    return meta, by_op, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, bundle = args.root.resolve(), args.bundle.resolve()
    outcomes: dict[str, object] = {'gpu_used': False, 'cuda_binary_executed': False, 'negative': {}}
    report, errors = validate_bundle(root, bundle)
    if errors:
        raise SystemExit(f'positive_preflight_failed:{errors}')
    outcomes['positive_preflight'] = report['status']
    with tempfile.TemporaryDirectory(prefix='g1b_v3_fixture_', dir=str(root / 'bundles')) as tmp:
        tmp_root = Path(tmp)
        trace_copy = tmp_root / 'trace_tamper'
        shutil.copytree(bundle, trace_copy)
        header, events = parse_trace(trace_copy / 'trace.e1gtrc')
        bad = list(events)
        bad[4] = TraceEvent(4, OP_INSERT, 4357)  # duplicate b: violates exact c insertion.
        write_trace(trace_copy / 'trace.e1gtrc', header, bad)
        _report, trace_errors = validate_bundle(root, trace_copy)
        if not trace_errors or not any('trace_not_exact_IaQaIbQbIcQcDcDbDa' in item for item in trace_errors):
            raise SystemExit(f'negative_trace_fixture_did_not_fail_strictly:{trace_errors}')
        outcomes['negative']['tampered_trace_rejected'] = True

        contract_copy = tmp_root / 'contract_tamper'
        shutil.copytree(bundle, contract_copy)
        path = contract_copy / 'g1b_capacity_contract.txt'
        path.write_text(path.read_text(encoding='utf-8').replace('sidecar_leaf_id=133', 'sidecar_leaf_id=134'), encoding='utf-8')
        _report, contract_errors = validate_bundle(root, contract_copy)
        if not contract_errors or not any('contract_capacity_leaf_ids_or_query_ids_mismatch' in item for item in contract_errors):
            raise SystemExit(f'negative_contract_fixture_did_not_fail_strictly:{contract_errors}')
        outcomes['negative']['tampered_contract_rejected'] = True

    meta, by_op, summary = _receipt_fixture()
    receipt_errors: list[str] = []
    validate_capacity_receipts(meta, by_op, summary, receipt_errors)
    if receipt_errors:
        raise SystemExit(f'positive_receipt_fixture_failed:{receipt_errors}')
    bad_meta, bad_by_op, bad_summary = _receipt_fixture()
    bad_by_op[4]['fallback_reason'] = 'certificate_reject'
    fallback_errors: list[str] = []
    validate_capacity_receipts(bad_meta, bad_by_op, bad_summary, fallback_errors)
    if not any('update_receipt_mismatch:op=4' == item for item in fallback_errors):
        raise SystemExit(f'negative_capacity_fallback_fixture_did_not_fail:{fallback_errors}')
    outcomes['negative']['certificate_vs_capacity_fallback_confusion_rejected'] = True

    # Build a pure-CPU synthetic engine answer from the exact active sets. This
    # tests the independent finalizer oracle itself without manufacturing any
    # GPU run artifact or finalizing the real v3 bundle.
    header, events = parse_trace(bundle / 'trace.e1gtrc')
    pool = load_i16(bundle / 'pool.i16', header.pool_n * header.dimension)
    queries = load_i16(bundle / 'queries.i16', header.query_n * header.dimension)
    mapping, initial = load_current_runner_identity_stable_id_layout(bundle, header)
    _meta, oracle_rows, _summary = _receipt_fixture()
    active = set(initial)
    for event in events:
        if event.op == OP_INSERT:
            active.add(event.argument)
        elif event.op == OP_DELETE:
            active.remove(event.argument)
        elif event.op == OP_KNN:
            def d2(stable_id: int) -> int:
                row = mapping[stable_id]
                return sum((int(pool[row * header.dimension + axis]) - int(queries[event.argument * header.dimension + axis])) ** 2
                           for axis in range(header.dimension))
            ranked = sorted((d2(stable_id), stable_id) for stable_id in active)[:header.k]
            oracle_rows[event.op_index]['results'] = [[stable_id, float(squared ** 0.5)] for squared, stable_id in ranked]
    oracle_errors: list[str] = []
    oracle = validate_full_active_exact_oracle(bundle, oracle_rows, oracle_errors)
    if oracle_errors or oracle.get('status') != 'PASS':
        raise SystemExit(f'positive_independent_oracle_fixture_failed:{oracle_errors}:{oracle}')
    outcomes['positive_independent_full_active_oracle'] = 'PASS'
    wrong_rows = copy.deepcopy(oracle_rows)
    wrong_rows[5]['results'][0][0] = 0  # active but wrong top-1 and wrong distance.
    wrong_errors: list[str] = []
    wrong_oracle = validate_full_active_exact_oracle(bundle, wrong_rows, wrong_errors)
    if wrong_oracle.get('status') != 'FAIL' or not wrong_errors:
        raise SystemExit(f'negative_independent_oracle_fixture_did_not_fail:{wrong_errors}:{wrong_oracle}')
    outcomes['negative']['full_active_exact_oracle_rejects_wrong_result'] = True
    outcomes['status'] = 'PASS_CPU_ONLY_G1B_CAPACITY_PIPELINE_FIXTURES'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(outcomes['status'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
