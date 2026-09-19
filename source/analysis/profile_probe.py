#!/usr/bin/env python3
"""Diagnostic-only NVTX attribution; never replace ordinary service timings."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from run_backend import NvMolKit, validate_inputs


def instrumented(backend, indices, p, d):
    torch = backend.torch
    begin = int(indices[0]); end = begin + len(indices)
    cutoff = float(np.nextafter(np.float32(p / d), np.float32(-np.inf)))
    nvtx = torch.cuda.nvtx.range
    with torch.cuda.stream(backend.stream):
        with nvtx('query_h2d'):
            q = backend.query_host[begin:end].to('cuda', non_blocking=True)
            qid = backend.qid_host[begin:end].to('cuda', non_blocking=True)
        with nvtx('dense_score_api'):
            scores = (backend.cross(q, backend.database, stream=backend.stream).torch()
                      if backend.name == 'nv_row' else
                      backend.cross(backend.database, q, stream=backend.stream).torch())
        with nvtx('threshold_and_self_mask'):
            eligible = ((scores >= cutoff) & (backend.ids[None, :] != qid[:, None])
                        if backend.name == 'nv_row' else
                        (scores >= cutoff) & (backend.ids[:, None] != qid[None, :]))
        with nvtx('nonzero_selection'):
            coords = torch.nonzero(eligible, as_tuple=False)
        with nvtx('stable_id_gather'):
            pairs = (torch.stack((coords[:, 0], backend.ids[coords[:, 1]]), dim=1)
                     if backend.name == 'nv_row' else
                     torch.stack((coords[:, 1], backend.ids[coords[:, 0]]), dim=1))
        with nvtx('complete_result_d2h'):
            host = pairs.to('cpu', non_blocking=False).numpy()
    with nvtx('host_result_split'):
        result = [np.array(host[host[:, 0] == i, 1], dtype=np.uint64) for i in range(len(indices))]
    return result


def main():
    if os.environ.get('P1_GUARDED_UUID') != 'GPU-CONFIGURE-ARCHIVE-DEVICE':
        raise RuntimeError('GPU admission guard required')
    manifest, queries, oracle = validate_inputs(ROOT / 'data/real')
    backends = {name: NvMolKit(name, ROOT, ROOT / 'data/real', queries)
                for name in ['nv_row', 'nv_column']}
    cells = [(name, start, size, p, d) for name in backends for start in [0, 512]
             for size in [1, 64] for p, d in [(7, 10), (4, 5)]]
    verified = 0
    def check(indices, p, d, results):
        nonlocal verified
        if len(results) != len(indices):
            raise ValueError('missing result vector')
        for qi, values in zip(indices, results):
            if not np.array_equal(np.sort(values), oracle[int(qi), p, d]):
                raise AssertionError('complete oracle mismatch')
            verified += 1
    # Pair instrumented and original paths before collecting diagnostics.
    for name, start, size, p, d in cells:
        indices = np.arange(start, start + size)
        backend = backends[name]
        for _ in range(16):
            original, _, _ = backend.run(queries, indices, p, d)
            check(indices, p, d, original)
        check(indices, p, d, instrumented(backend, indices, p, d))
    torch = next(iter(backends.values())).torch
    torch.cuda.profiler.start()
    try:
        for repetition in range(3):
            for name, start, size, p, d in cells:
                indices = np.arange(start, start + size)
                label = f'P1/{name}/cohort{start}/Q{size}/t{p}_{d}/r{repetition}'
                with torch.cuda.nvtx.range(label):
                    result = instrumented(backends[name], indices, p, d)
                check(indices, p, d, result)
    finally:
        torch.cuda.profiler.stop()
        for backend in backends.values():
            backend.close()
    print(json.dumps({'status': 'diagnostic-only', 'complete_vectors_verified': verified,
                      'profiled_ranges': 48, 'rows': manifest['rows'],
                      'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'scope': 'NVTX host API ranges require CUDA correlation; not pure device durations'}))


if __name__ == '__main__':
    main()
