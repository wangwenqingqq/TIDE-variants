#!/usr/bin/env python3
"""One verified row-oriented nvMolKit kernel after sixteen warmups."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from run_backend import NvMolKit, validate_inputs

ap = argparse.ArgumentParser(); ap.add_argument('--batch', type=int, choices=[1, 64], required=True)
args = ap.parse_args()
if os.environ.get('P1_GUARDED_UUID') != 'GPU-CONFIGURE-ARCHIVE-DEVICE':
    raise RuntimeError('admission guard required')
print(json.dumps({'pid': os.getpid(), 'ppid': os.getppid(), 'sid': os.getsid(0),
                  'batch': args.batch, 'state': 'diagnostic-only'}), flush=True)
manifest, queries, oracle = validate_inputs(ROOT / 'data/real')
indices = np.arange(args.batch)
b = NvMolKit('nv_row', ROOT, ROOT / 'data/real', queries)
verified = 0
try:
    for iteration in range(17):
        result, _, _ = b.run(queries, indices, 7, 10)
        for qi, values in zip(indices, result):
            if not np.array_equal(np.sort(values), oracle[int(qi), 7, 10]):
                raise AssertionError('complete oracle mismatch')
            verified += 1
finally:
    b.close()
print(json.dumps({'complete_vectors_verified': verified, 'pid': os.getpid(),
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}), flush=True)
