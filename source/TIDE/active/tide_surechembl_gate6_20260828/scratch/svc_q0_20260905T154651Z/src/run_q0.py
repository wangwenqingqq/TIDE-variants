#!/usr/bin/env python3
"""Same-output, process-paired Q1 runner. Startup/oracle work is never timed."""
import sys
sys.dont_write_bytecode = True
import argparse
import atexit
import ctypes as ct
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import threading
import time
import numpy as np
from svc_q0_data import datasets, arrays, packed, THRESHOLDS, dump_json

ROOT = Path(__file__).resolve().parents[1]
MODULE_SHA = '383e372ee1ccc0fc3afd82367ce3621b8a7cf662497f970219fb9fe5c4a6356d'
U64P = ct.POINTER(ct.c_uint64)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EpochStore:
    def __init__(self, engine):
        self.current = engine
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            return self.current


class Tide:
    def __init__(self, dataset, layout):
        self.lib = ct.CDLL(str(ROOT / 'build/libsvcq.so'))
        self.lib.svcq_error.restype = ct.c_char_p
        self.lib.svcq_create.argtypes = [ct.c_char_p] * 3
        self.lib.svcq_create.restype = ct.c_void_p
        self.lib.svcq_destroy.argtypes = [ct.c_void_p]
        self.lib.svcq_destroy.restype = None
        self.lib.svcq_search.argtypes = [ct.c_void_p, U64P, ct.c_uint, ct.c_uint, U64P, ct.c_uint64]
        self.lib.svcq_search.restype = ct.c_int64
        self.handle = self.lib.svcq_create(str(ROOT / 'tide_prepared').encode(), dataset.encode(), layout.encode())
        if not self.handle:
            raise RuntimeError(self.lib.svcq_error().decode())
        self.output = np.empty(65536, dtype=np.uint64)
        self.output_ptr = self.output.ctypes.data_as(U64P)

    def search(self, query, num, den):
        count = self.lib.svcq_search(self.handle, query.ctypes.data_as(U64P), num, den,
                                     self.output_ptr, len(self.output))
        if count < 0:
            raise RuntimeError(self.lib.svcq_error().decode())
        result = self.output[:count].view(np.int64).copy()
        result.sort()
        return result

    def close(self):
        if self.handle:
            self.lib.svcq_destroy(self.handle)
            self.handle = None


class Fpsim2:
    def __init__(self, dataset, layout):
        import cupy as cp
        from FPSim2 import FPSim2CudaEngine
        assert importlib.metadata.version('FPSim2') == '0.7.4'
        assert cp.__version__ == '14.2.0'
        self.module = inspect.getfile(sys.modules[FPSim2CudaEngine.__module__])
        assert sha(self.module) == MODULE_SHA
        self.cp = cp
        cp.cuda.Device(0).use()
        cp.get_default_memory_pool().set_limit(size=1 << 30)
        self.stream = cp.cuda.Stream(non_blocking=True)
        parts = ['base', 'delta'] if layout == 'sharded' else [layout]
        self.engines = tuple(FPSim2CudaEngine(str(ROOT / 'prepared' / f'{dataset}_{p}.h5')) for p in parts)
        cp.cuda.runtime.deviceSynchronize()

    def search(self, query, num, den):
        with self.stream:
            parts = []
            for engine in self.engines:
                ids, _scores = engine._raw_kernel_search(query, num / den)
                parts.append(np.asarray(ids, dtype=np.uint64).view(np.int64))
            self.stream.synchronize()
        result = np.concatenate(parts)
        result.sort()
        return result

    def close(self):
        self.stream.synchronize()
        assert sha(self.module) == MODULE_SHA
        self.engines = ()
        gc.collect()
        self.cp.get_default_memory_pool().free_all_blocks()


def input_audit():
    import tables
    manifest = json.loads((ROOT / 'prepared/PREPARATION.json').read_text())
    tide_manifest = json.loads((ROOT / 'tide_prepared/MANIFEST.json').read_text())
    count = 0
    for name, (base, delta, _) in datasets(ROOT / 'data').items():
        union = dict(base); union.update(delta)
        for part, source in [('base', base), ('delta', delta), ('union', union)]:
            p = ROOT / 'prepared' / f'{name}_{part}.h5'
            assert sha(p) == manifest[name]['artifacts'][part]['sha256']
            ids, bits, pc = arrays(source)
            with tables.open_file(str(p), mode='r') as f:
                rows = f.root.fps.read()
            assert np.array_equal(rows['fp_id'], ids)
            assert np.array_equal(rows['popcnt'], pc)
            words = np.ascontiguousarray(bits).view('<u8').reshape(-1, 4)
            for w in range(4):
                assert np.array_equal(rows[f'f{w + 1}'], words[:, w])
            folder = ROOT / 'tide_prepared' / f'{name}_{part}'
            for fname, meta in tide_manifest[f'{name}_{part}']['files'].items():
                assert sha(folder / fname) == meta['sha256']
            assert np.array_equal(np.fromfile(folder / 'id_i64.bin', dtype='<i8'), ids)
            assert np.array_equal(np.fromfile(folder / 'popcnt_u16.bin', dtype='<u2'), pc)
            assert np.array_equal(np.fromfile(folder / 'fp_u64x4.bin', dtype=np.uint8).reshape(-1, 32), bits)
            count += len(ids)
    return {'status': 'PASS', 'physical_rows_per_layout_family': count,
            'all_H5_TIDE_FPS_bytes_equal': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['audit', 'validate', 'measure'], required=True)
    parser.add_argument('--engine', choices=['tide', 'fpsim2'], default='tide')
    parser.add_argument('--pair', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == 'audit':
        result = input_audit()
        dump_json(args.output / 'SUMMARY.json', result)
        print(json.dumps(result))
        return
    # The shell owns admission and continuous foreign-process telemetry.
    assert os.environ.get('SVC_Q0_GPU_ADMITTED') == '1'
    assert 0 <= args.pair < 4
    cls = Tide if args.engine == 'tide' else Fpsim2
    oracle = json.loads((ROOT / 'prepared/ORACLE.json').read_text())
    data = datasets(ROOT / 'data')
    records = []
    def save_partial():
        if not (args.output / 'SUMMARY.json').exists():
            dump_json(args.output / 'PARTIAL_RECORDS.json', records)
    atexit.register(save_partial)
    cells = []
    layouts = ['base', 'union', 'sharded'] if args.mode == 'validate' else ['union', 'sharded']
    if args.mode == 'measure' and args.pair % 2:
        layouts.reverse()
    thresholds = list(THRESHOLDS)
    if args.mode == 'measure' and args.pair >= 2:
        thresholds.reverse()
    names = ['real', 'fixture'] if args.mode == 'validate' else ['real']
    for name in names:
        queries = data[name][2]
        qids = list(queries)
        if args.mode == 'measure':
            qids = np.random.default_rng(91700 + args.pair).permutation(qids).tolist()
        packed_queries = {i: packed(i, queries[i]) for i in qids}
        for layout in layouts:
            start = time.perf_counter_ns()
            engine = cls(name, layout)
            load_ns = time.perf_counter_ns() - start
            store = EpochStore(engine)
            epoch = 0 if layout == 'base' else 1
            try:
                for num, den in thresholds:
                    expected = {i: oracle[name][f'{epoch}:{i}:{num}/{den}'] for i in qids}
                    cycles = 1 if args.mode == 'validate' else 36
                    warmup_cycles = 0 if args.mode == 'validate' else 4
                    timings = []
                    for cycle in range(cycles):
                        for qid in qids:
                            query = packed_queries[qid]
                            begin = time.perf_counter_ns()
                            captured = store.acquire()
                            ids = captured.search(query, num, den)
                            elapsed = time.perf_counter_ns() - begin
                            # No oracle, serialization, or recording inside the service timer.
                            result = ids.tolist()
                            assert result == expected[qid], (name, layout, qid, num, den, result, expected[qid])
                            assert ids.dtype == np.int64 and ids.flags.owndata
                            warmup = cycle < warmup_cycles
                            records.append({'dataset': name, 'layout': layout, 'threshold': f'{num}/{den}',
                                'query': int(qid), 'cycle': cycle, 'warmup': warmup,
                                'elapsed_ns': int(elapsed), 'ids': result, 'match': True})
                            if not warmup:
                                timings.append(elapsed)
                        if (ROOT / 'raw/CONTAMINATION.json').exists():
                            raise RuntimeError('foreign GPU process detected; no timing admitted')
                    cells.append({'dataset': name, 'layout': layout, 'threshold': f'{num}/{den}',
                        'query_order': qids, 'warmup_count': warmup_cycles * len(qids),
                        'sample_count': len(timings), 'load_diagnostic_ns': load_ns,
                        'percentiles_ns': dict(zip(['p10', 'p50', 'p90', 'p95', 'p99'],
                            np.percentile(timings, [10, 50, 90, 95, 99]).tolist()))})
            finally:
                engine.close()
                del store, engine
                gc.collect()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert rss < 2 * 1024 * 1024
    dump_json(args.output / 'RECORDS.json', records)
    summary = {'experiment': 'SVC-Q0', 'mode': args.mode, 'engine': args.engine,
        'pair': args.pair, 'pid': os.getpid(), 'python': sys.version, 'numpy': np.__version__,
        'records': len(records), 'cells': cells, 'maxrss_kib': rss,
        'native_fpsim2_sha256': MODULE_SHA, 'bridge_sha256': sha(ROOT / 'build/libsvcq.so'),
        'runner_sha256': sha(__file__), 'status': 'PASS',
        'scope': 'static warmed complete owning sorted signed-int64 host IDs; Q1; not an update experiment'}
    dump_json(args.output / 'SUMMARY.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cells'}), flush=True)


if __name__ == '__main__':
    main()
