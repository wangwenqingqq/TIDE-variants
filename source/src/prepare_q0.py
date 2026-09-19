#!/usr/bin/env python3
"""Create TIDE layouts and independently validate the retained integer oracle."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import numpy as np
from svc_q0_data import datasets, arrays, LUT, THRESHOLDS, dump_json

root = Path(__file__).resolve().parents[1]
out = root / 'tide_prepared'
out.mkdir(exist_ok=False)
oracle = json.loads((root / 'prepared/ORACLE.json').read_text())
manifest = {}
checks = pairs = 0
for name, (base, delta, queries) in datasets(root / 'data').items():
    union = dict(base); union.update(delta)
    for layout, rows in (('base', base), ('delta', delta), ('union', union)):
        ids, bits, pc = arrays(rows)
        folder = out / f'{name}_{layout}'
        folder.mkdir()
        ids.tofile(folder / 'id_i64.bin')
        bits.tofile(folder / 'fp_u64x4.bin')
        pc.tofile(folder / 'popcnt_u16.bin')
        manifest[f'{name}_{layout}'] = {'rows': len(rows), 'files': {}}
        for p in sorted(folder.iterdir()):
            manifest[f'{name}_{layout}']['files'][p.name] = {
                'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        if layout == 'delta':
            continue
        epoch = 0 if layout == 'base' else 1
        for qid, fp in queries.items():
            q = np.frombuffer(fp.to_bytes(32, 'little'), dtype=np.uint8)
            intersections = LUT[np.bitwise_and(bits, q)].sum(axis=1).astype(np.int32)
            unions = pc.astype(np.int32) + int(LUT[q].sum()) - intersections
            for n, d in THRESHOLDS:
                actual = np.sort(ids[d * intersections >= n * unions]).tolist()
                assert actual == oracle[name][f'{epoch}:{qid}:{n}/{d}']
                checks += 1; pairs += len(rows)
dump_json(out / 'MANIFEST.json', manifest)
dump_json(root / 'analysis/PREPARATION_AUDIT.json', {
    'status': 'PASS', 'oracle_vectors': checks, 'independent_pair_evaluations': pairs,
    'method': 'NumPy byte-LUT popcount; exact rational comparison; full ID vectors',
    'gpu_execution': False, 'source': 'SVC-F0 FPS bytes; no old Python imports'})
print(json.dumps({'status': 'PASS', 'oracle_vectors': checks, 'pair_evaluations': pairs}))
