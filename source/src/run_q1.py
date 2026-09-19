#!/usr/bin/env python3
"""One real private delta admission; fixed queried population and complete IDs."""
import sys
sys.dont_write_bytecode = True
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import resource
import threading
import time
import numpy as np
from svc_q0_data import datasets, packed, dump_json
from q0_engine_reference import Tide, EpochStore, sha, input_audit

ROOT = Path(__file__).resolve().parents[1]


def cell(arm, threshold, repetition, count, output):
    queries = datasets(ROOT / 'data')['real'][2]
    order = np.random.default_rng(91800 + repetition).permutation(list(queries)).tolist()
    qs = {i: packed(i, queries[i]) for i in order}
    num, den = (7,10) if threshold == '7/10' else (4,5)
    population = 'base' if arm[0] == 'B' else 'union'
    epoch = 0 if population == 'base' else 1
    expected_all = json.loads((ROOT / 'prepared/ORACLE.json').read_text())['real']
    expected = {i: expected_all[f'{epoch}:{i}:{threshold}'] for i in order}
    engine = Tide('real', population)
    store = EpochStore(engine)
    original_handle = engine.handle
    lib = engine.lib
    lib.svcq_shadow_create.argtypes = [ct.c_char_p]
    lib.svcq_shadow_create.restype = ct.c_void_p
    lib.svcq_shadow_destroy.argtypes = [ct.c_void_p]
    lib.svcq_shadow_destroy.restype = None
    trigger = threading.Event()
    state = {}
    rows = []
    shadow = [None]

    def updater():
        try:
            if not trigger.wait(90):
                raise RuntimeError('query trigger timeout')
            state['start_ns'] = time.perf_counter_ns()
            if arm[1] == '1':
                shadow[0] = lib.svcq_shadow_create(str(ROOT / 'tide_prepared/real_delta').encode())
                if not shadow[0]:
                    raise RuntimeError(lib.svcq_error().decode())
            state['end_ns'] = time.perf_counter_ns()
            state['admissions'] = int(arm[1] == '1')
        except BaseException as exc:
            state['error'] = repr(exc)

    worker = threading.Thread(target=updater)
    wall_begin = wall_end = None
    try:
        for j in range(144):
            qid = order[j % 36]
            ids = store.acquire().search(qs[qid], num, den)
            assert ids.tolist() == expected[qid]
        worker.start()
        wall_begin = time.perf_counter_ns()
        for j in range(count):
            qid = order[j % 36]
            if j == 128:
                state['trigger_ns'] = time.perf_counter_ns()
                trigger.set()
            query = qs[qid]
            begin = time.perf_counter_ns()
            captured = store.acquire()
            ids = captured.search(query, num, den)
            end = time.perf_counter_ns()
            rows.append({'index': j, 'query': int(qid), 'begin_ns': begin,
                         'end_ns': end, 'ids_array': ids})
            if j % 36 == 0 and (ROOT / 'raw/CONTAMINATION.json').exists():
                raise RuntimeError('foreign GPU process observed')
        wall_end = time.perf_counter_ns()
        worker.join(90)
        assert not worker.is_alive() and 'error' not in state, state
        assert engine.handle == original_handle
        assert state['admissions'] == int(arm[1] == '1')
        assert state['trigger_ns'] <= state['start_ns'] <= state['end_ns']
        for row in rows:
            ids = row.pop('ids_array')
            row['ids'] = ids.tolist()
            assert ids.flags.owndata and ids.dtype == np.int64
            assert row['ids'] == expected[row['query']], row
        overlap = [r['index'] for r in rows if r['begin_ns'] < state['end_ns'] and r['end_ns'] > state['start_ns']]
        result = {'arm': arm, 'threshold': threshold, 'repetition': repetition,
            'status': 'PASS', 'pid': os.getpid(), 'queries': order, 'warmup_verified': 144,
            'measured_requests': count, 'admission': state, 'overlap_indices': overlap,
            'snapshot_handle_unchanged': True, 'private_shadow_retained_until_reader_complete': True,
            'reader_wall_begin_ns': wall_begin, 'reader_wall_end_ns': wall_end,
            'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'binary_sha256': sha(ROOT / 'build/libsvcq.so'), 'runner_sha256': sha(__file__),
            'scope': 'one reader, one native-ready private admission, no publication'}
        assert result['maxrss_kib'] < 2*1024*1024
        dump_json(output / 'RECORDS.json', rows)
        dump_json(output / 'SUMMARY.json', result)
        return result
    finally:
        trigger.set()
        if worker.ident is not None:
            worker.join(90)
        if shadow[0]:
            lib.svcq_shadow_destroy(shadow[0])
        engine.close()
        if not (output / 'SUMMARY.json').exists():
            for row in rows:
                if 'ids_array' in row:
                    row['ids'] = row.pop('ids_array').tolist()
            dump_json(output / 'PARTIAL_RECORDS.json', rows)
            dump_json(output / 'FAILED_STATE.json', state)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['audit','validate','measure'], required=True)
    p.add_argument('--arm', choices=['B0','B1','U0','U1'], default='B0')
    p.add_argument('--threshold', choices=['7/10','4/5'], default='7/10')
    p.add_argument('--repetition', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if a.mode == 'audit':
        result = input_audit()
        dump_json(a.output / 'SUMMARY.json', result)
    elif a.mode == 'validate':
        assert os.environ.get('SVC_Q0_GPU_ADMITTED') == '1'
        result = []
        for threshold in ('7/10','4/5'):
            for arm in ('B0','B1','U0','U1'):
                out = a.output / (arm + '_' + threshold.replace('/','-'))
                out.mkdir()
                result.append(cell(arm, threshold, 0, 288, out))
        dump_json(a.output / 'SUMMARY.json', {'status':'PASS','cells':result})
    else:
        assert os.environ.get('SVC_Q0_GPU_ADMITTED') == '1' and 0 <= a.repetition < 4
        result = cell(a.arm, a.threshold, a.repetition, 1152, a.output)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
