#!/usr/bin/env python3
"""CPU-only positive/negative G2 v5 artifact-pin and finalizer test.

It never invokes CUDA, a CUDA binary, nvidia-smi, or a GPU query.  The positive
case uses a synthetic engine receipt; the negative case alters only a copied
CPU verification receipt and proves the finalizer rejects hash disagreement.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from artifact_pinning_v5 import BINARY, PIN_NAME, VERIFY_PASS, verify_execution_artifact_pin, verify_pin_payload
from finalize_g2_rebuild_witness_v5 import active_hash, float_pool_hash, validate_engine
from preflight_g2_rebuild_witness_v5 import ROOT, validate_bundle
from safe_c1_v5_rebuild_bundle_contract import cint, fnv1a_i32, fnv1a_u64_bytes, load_i16, parse_trace, read_contract, replay_g2_trace, topk_exact


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows), encoding='utf-8')


def make_run_card(path: Path, bundle: Path, pin: Path, verification: Path, receipt: dict) -> None:
    write_json(path, {
        'schema': 'safe-c1-g2-v5-rebuild-witness-run-card',
        'status': 'CPU_SYNTHETIC_TEST_ONLY',
        'inputs': {'binary': str(BINARY.resolve()), 'bundle': str(bundle.resolve())},
        'artifact_pinning': {
            'status': receipt.get('status'), 'pin_path': str(pin.resolve()),
            'verification_path': str(verification.resolve()), 'pin_sha256': receipt.get('pin_sha256'),
            'actual_hashes': receipt.get('actual_hashes'),
        },
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    report: dict = {'schema': 'safe-c1-g2-v5-cpu-artifact-pin-positive-negative-test', 'gpu_used': False, 'cuda_binary_executed': False, 'nvidia_smi_called': False, 'bundle': str(bundle), 'checks': {}, 'errors': []}
    try:
        preflight, preflight_errors = validate_bundle(ROOT, bundle)
        if preflight_errors:
            raise RuntimeError('preflight_fixture_unexpectedly_fails:' + repr(preflight_errors))
        pin = bundle / PIN_NAME
        receipt = verify_execution_artifact_pin(ROOT, bundle, BINARY, pin)
        if receipt.get('status') != VERIFY_PASS:
            raise RuntimeError('artifact_pin_positive_verification_failed:' + repr(receipt.get('errors')))
        report['checks']['positive_artifact_pin_verification'] = True
        # Direct CPU-only negative: an in-memory altered pinned binary SHA must
        # be rejected without touching a pinned file or any external artifact.
        pin_data = json.loads(pin.read_text(encoding='utf-8'))
        bad_pin = copy.deepcopy(pin_data)
        bad_pin['pinned_hashes']['binary']['sha256'] = '0' * 64
        _actual, pin_errors = verify_pin_payload(ROOT, bundle, BINARY, pin, bad_pin)
        if not any(x == 'hash_mismatch:pinned_hashes.binary.sha256' for x in pin_errors):
            raise RuntimeError('in_memory_bad_pin_not_rejected:' + repr(pin_errors))
        report['checks']['negative_pinned_binary_hash_rejected'] = True

        header, events = parse_trace(bundle / 'trace.g2trc')
        contract = read_contract(bundle / 'g2_rebuild_contract.txt')
        replay = replay_g2_trace(header, events, contract)
        a, b, c = (cint(contract, key) for key in ('stable_id_a', 'stable_id_b', 'stable_id_c'))
        leaf = cint(contract, 'sidecar_leaf_id'); base = cint(contract, 'base_delete_id')
        initial = replay['initial_base_ids']; live = replay['active_after_rebuild']; ah = active_hash(header.pool_n, live)
        e0 = {'record':'epoch','epoch':0,'phase':'E0','tree_hash':101,'leaf_stable_id_hash':fnv1a_i32(initial),'pivot_stable_id_hash':fnv1a_i32([]),'live_stable_id_hash':fnv1a_i32(initial),'live_count':len(initial),'leaf_set_exact':True,'pivot_set_subset_of_live':True,'pivot_stable_ids':[]}
        e1 = {'record':'epoch','epoch':1,'phase':'E1','tree_hash':202,'leaf_stable_id_hash':fnv1a_i32(live),'pivot_stable_id_hash':fnv1a_i32([]),'live_stable_id_hash':fnv1a_i32(live),'live_count':len(live),'leaf_set_exact':True,'pivot_set_subset_of_live':True,'pivot_stable_ids':[]}
        pool_i16 = load_i16(bundle/'pool.i16', header.pool_n*header.dimension)
        queries_i16 = load_i16(bundle/'queries.i16', header.query_n*header.dimension)
        pool_hash = float_pool_hash(pool_i16)
        raw_i16_hash = fnv1a_u64_bytes((bundle/'pool.i16').read_bytes())
        if pool_hash == raw_i16_hash:
            raise RuntimeError('float32_gts_scalar_hash_regressed_to_raw_i16_bytes')
        query_ids = {2:cint(contract,'query_id_before'), 5:cint(contract,'query_id_middle'), 8:cint(contract,'query_id_after')}
        def results(op: int) -> list[list[float|int]]:
            return [[ident, math.sqrt(square)] for square, ident in topk_exact(pool_i16, queries_i16, replay['query_active_ids'][str(op)], header, query_ids[op])]
        rows = [
            {'record':'meta','schema':'safe-c1-g2-rebuild-witness-v5','scope':'no timing and no performance claim','legacy_incremental_updater_used':False,'stable_id_layout':'identity_full_immutable_pool_seeded_stable_ids_v5','host_full_pool_hash':pool_hash}, e0,
            {'record':'update','op_index':0,'op':'insert','stable_id':a,'placement':'direct','sidecar_leaf_id':leaf,'certified_leaf_id':leaf,'fallback_reason':'none'},
            {'record':'update','op_index':1,'op':'insert','stable_id':b,'placement':'delta','sidecar_leaf_id':-1,'certified_leaf_id':leaf,'fallback_reason':'capacity_full'},
            {'record':'query','op_index':2,'query_id':query_ids[2],'gts_visited_leaf_ids':[leaf],'sidecar_ids':[a],'delta_ids':[b],'oracle_full_active_set_checked':True,'results':results(2)},
            {'record':'update','op_index':3,'op':'delete','stable_id':a,'placement':'deleted','sidecar_leaf_id':leaf,'certified_leaf_id':leaf,'fallback_reason':'none'},
            {'record':'update','op_index':4,'op':'insert','stable_id':c,'placement':'direct','sidecar_leaf_id':leaf,'certified_leaf_id':leaf,'fallback_reason':'none'},
            {'record':'query','op_index':5,'query_id':query_ids[5],'gts_visited_leaf_ids':[leaf],'sidecar_ids':[c],'delta_ids':[b],'oracle_full_active_set_checked':True,'results':results(5)},
            {'record':'update','op_index':6,'op':'delete','stable_id':base,'placement':'deleted','sidecar_leaf_id':-1,'certified_leaf_id':-1,'fallback_reason':'none'},
            {'record':'rebuild','op_index':7,'trigger':'base_delete_immediate','metadata_only_release':True,'legacy_incremental_updater_used':False,'seeded_base_stable_ids':live,'pre_rebuild_active_hash':ah,'post_rebuild_active_hash':ah,'immutable_pool_hash_before':pool_hash,'immutable_pool_hash_after':pool_hash,'transient_direct_before':[c],'transient_delta_before':[b],'transient_direct_after':[],'transient_delta_after':[],'E1_tree_hash':202,'E1_leaf_stable_id_hash':fnv1a_i32(live),'E1_pivot_stable_id_hash':fnv1a_i32([]),'E1_leaf_set_exact':True,'E1_pivot_set_subset_of_live':True}, e1,
            {'record':'query','op_index':8,'query_id':query_ids[8],'gts_visited_leaf_ids':[],'sidecar_ids':[],'delta_ids':[],'oracle_full_active_set_checked':True,'results':results(8)},
        ]
        summary = {'schema':'safe-c1-g2-rebuild-witness-v5','status':'PASS_G2_REBUILD_WITNESS_PENDING_INDEPENDENT_VALIDATOR','scope':'strict witness no performance claim','archived_incremental_updater_used':False,'rebuild':{'metadata_only_release':True,'full_pool_preserved':True,'active_set_preserved':True,'transient_tiers_cleared':True},'E0':{'leaf_set_exact':True,'pivot_set_subset_of_live':True},'E1':{'leaf_set_exact':True,'pivot_set_subset_of_live':True},'active_hash':{'before_rebuild':ah,'after_rebuild':ah},'immutable_full_pool_hash':{'host':pool_hash,'device_after_rebuild':pool_hash}}
        if validate_engine(bundle, preflight, rows, summary):
            raise RuntimeError('synthetic_engine_receipt_rejected:' + repr(validate_engine(bundle, preflight, rows, summary)))
        report['checks']['synthetic_engine_contract_positive'] = True

        guard = (ROOT/'tools/run_g2_rebuild_witness_guarded_v5.sh').read_text(encoding='utf-8')
        pin_marker = guard.find('"$PYTHON" "$PIN_TOOL" verify')
        nvidia_marker = guard.find('command -v nvidia-smi')
        if not (pin_marker >= 0 and nvidia_marker >= 0 and pin_marker < nvidia_marker):
            raise RuntimeError('guard_artifact_pin_not_before_nvidia_smi')
        report['checks']['guard_fail_closed_pin_before_nvidia_smi'] = True

        with tempfile.TemporaryDirectory(prefix='g2_v5_cpu_pin_test_') as temporary:
            d = Path(temporary)
            pre = d/'preflight.json'; engine = d/'engine.jsonl'; engine_summary = d/'summary.json'; verify = d/'pin_verified.json'; card = d/'run_card.json'; final = d/'final_positive.json'; bad_verify = d/'pin_verified_bad.json'; bad_final = d/'final_negative.json'
            write_json(pre, preflight); write_jsonl(engine, rows); write_json(engine_summary, summary); write_json(verify, receipt); make_run_card(card, bundle, pin, verify, receipt)
            command = [sys.executable, str(ROOT/'tools/finalize_g2_rebuild_witness_v5.py'), '--root', str(ROOT), '--bundle', str(bundle), '--preflight', str(pre), '--engine-results', str(engine), '--engine-summary', str(engine_summary), '--artifact-pin', str(pin), '--pin-verification', str(verify), '--binary', str(BINARY), '--run-card', str(card), '--out', str(final)]
            good = subprocess.run(command, text=True, capture_output=True, check=False)
            if good.returncode != 0:
                raise RuntimeError('finalizer_positive_failed:' + good.stdout + good.stderr)
            final_data = json.loads(final.read_text(encoding='utf-8'))
            if final_data.get('artifact_pin_verified') is not True:
                raise RuntimeError('finalizer_positive_missing_pin_verified')
            report['checks']['positive_finalizer_with_actual_hashes'] = True
            altered = copy.deepcopy(receipt)
            altered['actual_hashes']['binary']['sha256'] = '0' * 64
            write_json(bad_verify, altered)
            bad_command = command.copy(); bad_command[bad_command.index('--pin-verification') + 1] = str(bad_verify); bad_command[bad_command.index('--out') + 1] = str(bad_final)
            bad = subprocess.run(bad_command, text=True, capture_output=True, check=False)
            bad_data = json.loads(bad_final.read_text(encoding='utf-8'))
            if bad.returncode == 0 or 'artifact_pin_mismatch' not in bad_data.get('errors', []) or 'artifact_pin_verification_actual_hash_mismatch' not in bad_data.get('errors', []):
                raise RuntimeError('finalizer_hash_mismatch_negative_not_rejected:' + repr(bad_data))
            report['checks']['negative_finalizer_hash_mismatch_rejected'] = True
        report['status'] = 'PASS_CPU_ONLY_G2_V5_ARTIFACT_PIN_POSITIVE_NEGATIVE'
    except Exception as exc:
        report['status'] = 'FAIL_CPU_ONLY_G2_V5_ARTIFACT_PIN_POSITIVE_NEGATIVE'
        report['errors'].append(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(report['status'])
    return 0 if not report['errors'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
