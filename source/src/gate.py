#!/usr/bin/env python3
"""Small, complete-ID sharding/epoch gate over unmodified FPSim2Cuda 0.7.4."""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
import resource
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

MASK = (1 << 64) - 1
THRESHOLDS = ((7, 10), (4, 5))
MODULE_SHA = '383e372ee1ccc0fc3afd82367ce3621b8a7cf662497f970219fb9fe5c4a6356d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def read_fps(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        fingerprint, identifier = line.split('\t')
        fp, identifier = bytes.fromhex(fingerprint), int(identifier)
        assert len(fp) == 32 and identifier not in result
        result[identifier] = int.from_bytes(fp, 'little')
    return result


def inputs(directory):
    base = read_fps(directory / 'base_a.fps')
    base_b = read_fps(directory / 'base_b.fps')
    delta = read_fps(directory / 'delta.fps')
    assert len(base) == len(base_b) == 32768 and len(delta) == 4096
    assert not (base.keys() & base_b.keys() or base.keys() & delta.keys() or base_b.keys() & delta.keys())
    base.update(base_b)
    queries = read_fps(directory / 'queries.fps')
    assert len(queries) == 4
    assert all(q not in base and delta[q] == fp for q, fp in queries.items())
    for source in (base, delta):
        selected = [i for i in sorted(source) if source[i] != 0][:16]
        for i in selected:
            queries[i] = source[i]
    assert all(queries.values())
    fixture_base = read_fps(directory / 'synthetic_targets.fps')
    fixture_delta = {(1 << 40) + 1: 127, (1 << 40) + 2: 127,
                     (1 << 40) + 3: 0, (1 << 40) + 4: (1 << 256) - 1,
                     (1 << 40) + 5: 1}
    fixture_queries = read_fps(directory / 'synthetic_queries.fps')
    fixture_queries.update({(1 << 40) + 101: 1, (1 << 40) + 102: (1 << 256) - 1})
    return {'real': (base, delta, queries),
            'fixture': (fixture_base, fixture_delta, fixture_queries)}


def integer_oracle(targets, query, num, den):
    assert query > 0
    return sorted(i for i, t in targets.items()
                  if den * (query & t).bit_count() >= num * (query | t).bit_count())


def native_record(identifier, fingerprint):
    return (identifier, *((fingerprint >> (64 * j)) & MASK for j in range(4)), fingerprint.bit_count())


def make_h5(path, targets):
    import numpy as np
    import tables as tb
    import rdkit
    from FPSim2.io.backends.pytables import create_schema
    ordered = sorted(targets.items(), key=lambda item: (item[1].bit_count(), item[0]))
    rows = [native_record(i, fp) for i, fp in ordered]
    counts, start = [], 0
    for pc in range(257):
        n = sum(fp.bit_count() == pc for _, fp in ordered)
        if n:
            counts.append((pc, (start, start + n)))
            start += n
    assert start == len(targets)
    with tb.open_file(str(path), mode='w') as handle:
        table = handle.create_table('/', 'fps', create_schema(256), filters=tb.Filters(complevel=0))
        table.append(rows)
        config = handle.create_vlarray('/', 'config', atom=tb.ObjectAtom())
        for value in ('Morgan', {'radius': 2, 'fpSize': 256}, rdkit.__version__, '0.7.4', counts):
            config.append(value)
        table.flush()
    with tb.open_file(str(path), mode='r') as handle:
        data = handle.root.fps.read()
        got = {}
        for row in data:
            fp = sum(int(row[f'f{j+1}']) << (64 * j) for j in range(4))
            assert fp.bit_count() == int(row['popcnt'])
            got[int(row['fp_id'])] = fp
        assert got == targets
    return {'rows': len(targets), 'bytes': path.stat().st_size, 'sha256': sha(path),
            'round_trip_all_records_exact': True, 'bins': len(counts)}


def prepare(args):
    assert importlib.metadata.version('FPSim2') == '0.7.4'
    args.output.mkdir(parents=True, exist_ok=False)
    datasets, manifest, oracle = inputs(args.data), {}, {}
    for name, (base, delta, queries) in datasets.items():
        union = dict(base); union.update(delta)
        manifest[name] = {'queries': list(queries), 'artifacts': {}}
        for geometry, targets in [('base', base), ('delta', delta), ('union', union)]:
            path = args.output / f'{name}_{geometry}.h5'
            manifest[name]['artifacts'][geometry] = make_h5(path, targets)
        oracle[name] = {}
        for epoch, targets in ((0, base), (1, union)):
            for qid, query in queries.items():
                for n, d in THRESHOLDS:
                    oracle[name][f'{epoch}:{qid}:{n}/{d}'] = integer_oracle(targets, query, n, d)
    write_json(args.output / 'PREPARATION.json', manifest)
    write_json(args.output / 'ORACLE.json', oracle)
    print(json.dumps({'stage': 'prepared', 'datasets': {k: {'queries': len(v['queries']),
          'artifacts': v['artifacts']} for k, v in manifest.items()}}, indent=2))


@dataclass(frozen=True)
class Epoch:
    number: int
    engines: tuple


class EpochStore:
    def __init__(self, epoch):
        self.current = epoch
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            return self.current

    def publish(self, epoch):
        with self.lock:
            self.current = epoch


def validate(args):
    import numpy as np
    import cupy as cp
    import FPSim2
    from FPSim2 import FPSim2CudaEngine
    assert FPSim2.__version__ == '0.7.4'
    module = sys.modules[FPSim2CudaEngine.__module__]
    assert sha(inspect.getfile(module)) == MODULE_SHA
    args.output.mkdir(parents=True, exist_ok=False)
    cp.cuda.Device(0).use()
    datasets = inputs(args.data)
    preparation = json.loads((args.prepared / 'PREPARATION.json').read_text())
    oracle = json.loads((args.prepared / 'ORACLE.json').read_text())
    records, deterministic, concurrency = [], [], []
    before_hashes = {}
    for name in datasets:
        for geometry in ('base', 'delta', 'union'):
            p = args.prepared / f'{name}_{geometry}.h5'
            before_hashes[str(p)] = sha(p)
            assert before_hashes[str(p)] == preparation[name]['artifacts'][geometry]['sha256']
    original_pool_limit = cp.get_default_memory_pool().get_limit()
    cp.get_default_memory_pool().set_limit(size=1 << 30)
    print(json.dumps({'stage': 'start', 'pid': os.getpid(), 'runtime': {
        'python': sys.version, 'fpsim2': FPSim2.__version__, 'cupy': cp.__version__,
        'numpy': np.__version__, 'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode()},
        'pool_limit': 1 << 30}), flush=True)

    def execute(epoch, qid, fp, n, d):
        query = np.asarray([int(x) & MASK for x in native_record(qid, fp)], dtype=np.uint64)
        result = []
        for engine in epoch.engines:
            ids, _scores = engine._raw_kernel_search(query, n / d)
            result.extend(int(i) for i in np.asarray(ids, dtype=np.uint64).view(np.int64))
        cp.cuda.get_current_stream().synchronize()
        assert len(result) == len(set(result)), 'duplicate result ID'
        return sorted(result)

    def record_result(name, phase, reader, epoch, qid, fp, n, d):
        result = execute(epoch, qid, fp, n, d)
        expected = oracle[name][f'{epoch.number}:{qid}:{n}/{d}']
        assert result == expected, (name, phase, reader, epoch.number, qid, n, d, result, expected)
        return dict(dataset=name, phase=phase, reader=reader, epoch=epoch.number,
                    query=qid, threshold=f'{n}/{d}', ids=result, match=True)

    for name, (base_rows, delta_rows, queries) in datasets.items():
        engines = {g: FPSim2CudaEngine(str(args.prepared / f'{name}_{g}.h5'))
                   for g in ('base', 'delta', 'union')}
        cp.cuda.get_current_stream().synchronize()
        base, sharded, union = Epoch(0, (engines['base'],)), Epoch(1, (engines['base'], engines['delta'])), Epoch(1, (engines['union'],))
        # Run every real/fixture query once per geometry, compiling native kernels before sharing.
        for label, epoch in [('base', base), ('sharded', sharded), ('union', union)]:
            for qid, fp in queries.items():
                for n, d in THRESHOLDS:
                    records.append(record_result(name, label, -1, epoch, qid, fp, n, d))
        store = EpochStore(base)
        acquire_barrier = threading.Barrier(5, timeout=90)
        published = threading.Event()

        def reader(worker):
            cp.cuda.Device(0).use()
            stream = cp.cuda.Stream(non_blocking=True)
            held_old = store.acquire()
            acquire_barrier.wait()
            assert published.wait(90)
            local = []
            with stream:
                # Every old snapshot is used after publication, without reacquisition.
                witness_id = next(iter(queries))
                for n, d in THRESHOLDS:
                    local.append(record_result(name, 'held_old_after_publish', worker, held_old,
                                               witness_id, queries[witness_id], n, d))
                for cycle in range(2):
                    for qid, fp in queries.items():
                        for n, d in THRESHOLDS:
                            captured = store.acquire()
                            assert captured.number == 1
                            local.append(record_result(name, 'concurrent_new', worker, captured, qid, fp, n, d))
            stream.synchronize()
            return local

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(reader, i) for i in range(4)]
            acquire_barrier.wait()
            store.publish(sharded)
            published.set()
            for future in futures:
                records.extend(future.result())
        assert store.acquire().engines[0] is engines['base']
        # Retained tuple still owns the unchanged base after new visibility.
        deterministic.append({'dataset': name, 'old_retained': True, 'base_object_reused': True})
        concurrency.append({'dataset': name, 'readers': 4, 'new_epoch_cycles_per_reader': 2})
        del engines, base, sharded, union, store
        cp.get_default_memory_pool().free_all_blocks()

    for path, digest in before_hashes.items():
        assert sha(path) == digest, path
    summary = dict(experiment='SVC-F0', status='CAPABILITY_PASS_NOT_PERFORMANCE_OR_NOVELTY',
        records=len(records), full_id_vector_matches=sum(r['match'] for r in records),
        deterministic=deterministic, concurrency=concurrency, all_input_h5_hashes_unchanged=True,
        process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        gpu_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
        api='released _raw_kernel_search; signed int64 ID reinterpretation; no scoring edits',
        timing_scope='none: validation only', source_sha256=sha(__file__),
        native_module_sha256=MODULE_SHA, original_pool_limit=original_pool_limit)
    assert summary['process_maxrss_kib'] < 2 * 1024 * 1024, 'exceeded small-gate RSS budget'
    write_json(args.output / 'RECORDS.json', records)
    write_json(args.output / 'SUMMARY.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'validate'])
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--prepared', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare(args)
    else:
        assert args.prepared is not None
        validate(args)


if __name__ == '__main__':
    main()
