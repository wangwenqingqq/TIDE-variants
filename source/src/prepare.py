#!/usr/bin/env python3
"""Create deterministic CPU-only P1 inputs without altering canonical data."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

SEED = 20260905


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def hash_ids(ids, seed=SEED):
    x = np.asarray(ids, dtype=np.uint64) ^ np.uint64(seed)
    with np.errstate(over='ignore'):
        x = x + np.uint64(0x9e3779b97f4a7c15)
        x = (x ^ (x >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> 27)) * np.uint64(0x94d049bb133111eb)
    return x ^ (x >> 31)


def quotas(counts, total):
    counts = np.asarray(counts, dtype=np.int64)
    if not 0 < total <= int(counts.sum()):
        raise ValueError('sample size outside source')
    products = counts * total
    result = products // counts.sum()
    remaining = total - int(result.sum())
    order = np.argsort(-(products % counts.sum()), kind='stable')
    result[order[:remaining]] += 1
    assert result.sum() == total and np.all(result <= counts)
    return result


def pc(fp):
    return np.bitwise_count(fp).sum(axis=1).astype('<u2')


def records(ids, fp):
    result = np.zeros((len(ids), 6), dtype='<u8')
    result[:, 0], result[:, 1:5], result[:, 5] = ids, fp, pc(fp)
    return result


def save_dataset(path, ids, fp, query_records, cohort_names, extra=None):
    path.mkdir(parents=True, exist_ok=False)
    counts = pc(fp)
    order = np.argsort(counts, kind='stable')
    np.asarray(fp[order], dtype='<u8').tofile(path / 'fp_u64x4.bin')
    np.asarray(ids[order], dtype='<u8').tofile(path / 'id_i64.bin')
    counts[order].tofile(path / 'popcnt_u16.bin')
    np.asarray(query_records, dtype='<u8').tofile(path / 'queries_u64x6.bin')
    assert len(np.unique(ids)) == len(ids)
    assert len(np.unique(query_records[:, 0])) == len(query_records)
    assert np.all(query_records[:, 5] > 0)
    info = {'rows': len(ids), 'queries': len(query_records), 'bits': 256,
            'seed': SEED, 'cohorts': cohort_names, 'self_id_excluded': True,
            'zero_query_admitted': False, 'popcount_histogram': np.bincount(counts, minlength=257).tolist(),
            'extra': extra or {}}
    info['files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
                     for p in sorted(path.glob('*.bin'))}
    (path / 'MANIFEST.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


def fixture(path):
    rng = np.random.default_rng(SEED)
    fp = rng.integers(0, np.iinfo(np.uint64).max, size=(4096, 4), dtype=np.uint64)
    for count in range(257):
        bits = (1 << count) - 1
        fp[count] = [(bits >> (64*w)) & ((1 << 64)-1) for w in range(4)]
    fp[300:316] = fp[100:116]
    ids = np.arange(10000, 10000+len(fp), dtype=np.uint64)
    qfp = np.concatenate([fp[1:257], fp[512:768]])
    qids = np.arange(1000000, 1000000+512, dtype=np.uint64)
    qids[-16:] = ids[752:768]
    return save_dataset(path, ids, fp, records(qids, qfp), ['fixture']*512,
                        {'purpose': 'prefix-ratio, high-entropy, zero-target, self-exclusion and duplicate-fingerprint checks'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--rows', type=int, default=1048576)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    canonical = args.canonical
    manifest = json.loads((canonical/'data/prepared/FLAT_INPUT_MANIFEST.json').read_text())
    verified = []
    sources = {}
    for version in ('2026-08-25', '2026-08-18'):
        rec = manifest['snapshots'][version]
        paths = {k: canonical/f'data/prepared/snapshots/{version}'/name
                 for k,name in [('fp','fp_u64x4.bin'),('ids','id_i64.bin'),('popcnt','popcnt_u16.bin')]}
        for key, path in paths.items():
            actual = digest(path)
            if actual != rec[key+'_sha256']:
                raise ValueError(f'canonical hash mismatch: {path}')
            verified.append({'path':str(path), 'sha256':actual})
        n = rec['rows']
        sources[version] = (np.memmap(paths['ids'], mode='r', dtype='<u8', shape=(n,)),
                            np.memmap(paths['fp'], mode='r', dtype='<u8', shape=(n,4)),
                            np.memmap(paths['popcnt'], mode='r', dtype='<u2', shape=(n,)))
    ids, fp, counts = sources['2026-08-25']
    if np.any(counts[1:] < counts[:-1]) or np.any(counts > 256):
        raise ValueError('invalid canonical population-count order')
    hist = np.bincount(counts, minlength=257)
    allocation = quotas(hist, args.rows)
    cuts = np.concatenate(([0], np.cumsum(hist)))
    selected = []
    for bitcount, take in enumerate(allocation):
        if take == 0:
            continue
        begin,end = int(cuts[bitcount]),int(cuts[bitcount+1])
        hashes = hash_ids(ids[begin:end])
        local = np.arange(end-begin) if take == end-begin else np.argpartition(hashes, int(take)-1)[:take]
        selected.append(np.sort(local) + begin)
    selected = np.concatenate(selected).astype('<u8')
    querypath = canonical/'data/prepared/transition_queries/2026-08-18_to_2026-08-25/queries_u64x6.bin'
    future = np.fromfile(querypath, dtype='<u8').reshape(-1,6)
    assert future.shape == (512,6)
    assert np.array_equal(future[:,5], pc(future[:,1:5]))
    verified.append({'path':str(querypath),'sha256':digest(querypath)})
    baseids,basefp,basecounts = sources['2026-08-18']
    hashes = hash_ids(baseids, SEED+1)
    excluded = np.isin(baseids, future[:,0]) | (basecounts == 0)
    hashes[excluded] = np.iinfo(np.uint64).max
    picked = np.argpartition(hashes,511)[:512]
    picked = picked[np.argsort(hashes[picked], kind='stable')]
    base_queries = records(baseids[picked], basefp[picked])
    queries = np.concatenate([future, base_queries])
    args.output.mkdir(parents=True)
    (args.output/'SOURCE_MANIFEST.json').write_text(json.dumps(verified,indent=2)+'\n')
    real = save_dataset(args.output/'real', ids[selected], fp[selected], queries,
                        ['future']*512+['base_hash']*512,
                        {'source_rows': len(ids), 'sampling':'proportional popcount strata, lowest stable-ID hash',
                         'stratum_quotas':allocation.tolist()})
    selected.tofile(args.output/'real/selected_source_rows_u64.bin')
    base_queries[:,0].tofile(args.output/'real/base_query_ids_u64.bin')
    fixture(args.output/'fixture')
    print(json.dumps({'real_rows':real['rows'],'queries':real['queries'],
                      'snapshot_hashes_verified_against_existing_manifest':6,
                      'source_hashes_recorded':len(verified), 'output':str(args.output)}))


if __name__ == '__main__':
    main()
