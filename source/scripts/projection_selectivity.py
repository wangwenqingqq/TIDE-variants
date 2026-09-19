#!/usr/bin/env python3
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


def read_vecs(path: Path, value_dtype):
    header = np.memmap(path, dtype=np.int32, mode='r')
    dim = int(header[0])
    item_dtype = np.dtype(value_dtype)
    if item_dtype.itemsize != 4:
        raise ValueError('only 4-byte fvecs/ivecs are supported')
    raw = np.memmap(path, dtype=item_dtype, mode='r')
    if raw.size % (dim + 1):
        raise ValueError('bad vecs file size: ' + str(path))
    matrix = raw.reshape(-1, dim + 1)
    dims = matrix[:, 0].view(np.int32).reshape(-1)
    if not np.all(dims == dim):
        raise ValueError('inconsistent dimensions in ' + str(path))
    return matrix[:, 1:]


def sqdist_block(x, y):
    # x: nxd, y:mxd; float64 accumulation avoids negative round-off.
    xd = np.asarray(x, dtype=np.float64)
    yd = np.asarray(y, dtype=np.float64)
    out = (np.sum(xd * xd, axis=1)[:, None]
           + np.sum(yd * yd, axis=1)[None, :])
    out -= 2.0 * (xd @ yd.T)
    np.maximum(out, 0.0, out=out)
    return out


def farthest_landmarks(sample, count):
    sample = np.asarray(sample, dtype=np.float32)
    chosen = [0]
    min_d2 = sqdist_block(sample, sample[[0]])[:, 0]
    for _ in range(1, count):
        idx = int(np.argmax(min_d2))
        chosen.append(idx)
        d2 = sqdist_block(sample, sample[[idx]])[:, 0]
        np.minimum(min_d2, d2, out=min_d2)
    return sample[np.asarray(chosen)]


def landmark_distances(x, landmarks, block=10000):
    ans = np.empty((x.shape[0], landmarks.shape[0]), dtype=np.float32)
    for begin in range(0, x.shape[0], block):
        end = min(begin + block, x.shape[0])
        ans[begin:end] = np.sqrt(sqdist_block(x[begin:end], landmarks)).astype(np.float32)
    return ans


def save_rt_case(out_dir, name, points, query_coords, taus):
    case = out_dir / name
    case.mkdir(parents=True, exist_ok=True)
    p = np.asarray(points, dtype=np.float32)
    q = np.asarray(query_coords, dtype=np.float32)
    t = np.asarray(taus, dtype=np.float32)
    # Conservative numeric padding for float projection and RT AABBs.
    pad = np.maximum(np.float32(1e-3), np.abs(t) * np.float32(1e-5))
    tr = np.nextafter(t + pad, np.float32(np.inf))
    boxes = np.stack((q[:, 0] - tr, q[:, 1] - tr,
                      q[:, 0] + tr, q[:, 1] + tr), axis=1).astype(np.float32)
    p.tofile(case / 'points.f32')
    boxes.tofile(case / 'query_boxes.f32')
    np.stack((t, tr), axis=1).astype(np.float32).tofile(case / 'taus.f32')
    meta = {
        'case': name,
        'n_points': int(p.shape[0]),
        'n_queries': int(q.shape[0]),
        'dims': 2,
        'point_file': 'points.f32',
        'query_box_file': 'query_boxes.f32',
        'tau_file': 'taus.f32',
        'point_semantics': 'degenerate 2D envelope at projected/signature coordinate',
        'query_semantics': 'inclusive square with padded base kth L2 threshold'
    }
    (case / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base-count', type=int, default=900000)
    ap.add_argument('--delta-count', type=int, default=100000)
    ap.add_argument('--queries', type=int, default=32)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--pca-sample', type=int, default=50000)
    ap.add_argument('--landmark-sample', type=int, default=10000)
    ap.add_argument('--landmarks', type=int, default=32)
    args = ap.parse_args()

    started = time.time()
    ds = Path(args.dataset_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base_all = read_vecs(ds / 'sift_base.fvecs', np.float32)
    queries_all = read_vecs(ds / 'sift_query.fvecs', np.float32)
    gt_all = read_vecs(ds / 'sift_groundtruth.ivecs', np.int32)
    n_total, dim = base_all.shape
    if args.base_count + args.delta_count > n_total:
        raise ValueError('split exceeds dataset')
    base_n = args.base_count
    delta_begin = base_n
    delta_end = base_n + args.delta_count
    delta = np.asarray(base_all[delta_begin:delta_end], dtype=np.float32)
    queries = np.asarray(queries_all[:args.queries], dtype=np.float32)
    gt = np.asarray(gt_all[:args.queries], dtype=np.int32)

    taus = np.empty(args.queries, dtype=np.float32)
    base_seed_ids = []
    for qi in range(args.queries):
        valid = gt[qi][gt[qi] < base_n]
        if valid.size < args.k:
            raise RuntimeError('GT depth insufficient for base threshold')
        ids = valid[:args.k].astype(np.int64)
        base_seed_ids.append(ids.tolist())
        d2 = np.sum((np.asarray(base_all[ids], dtype=np.float64) - queries[qi]) ** 2, axis=1)
        taus[qi] = np.float32(math.sqrt(float(np.max(d2))))

    # Exhaustive delta distances: empirical oracle for the conservative filter.
    delta_true_masks = []
    delta_true_counts = []
    for qi in range(args.queries):
        d2 = np.sum((delta.astype(np.float64) - queries[qi].astype(np.float64)) ** 2, axis=1)
        mask = d2 <= float(taus[qi]) ** 2 + 1e-8
        delta_true_masks.append(mask)
        delta_true_counts.append(int(mask.sum()))

    # PCA basis has orthonormal columns, hence every prefix is non-expansive.
    train = np.asarray(base_all[:min(args.pca_sample, base_n)], dtype=np.float64)
    mean = train.mean(axis=0)
    centered = train - mean
    cov = centered.T @ centered
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    basis = evecs[:, order].astype(np.float32)
    delta_pca = (delta - mean.astype(np.float32)) @ basis
    query_pca = (queries - mean.astype(np.float32)) @ basis

    # Farthest-point landmarks on a deterministic prefix sample.
    lm_sample_n = min(args.landmark_sample, base_n)
    landmarks = farthest_landmarks(np.asarray(base_all[:lm_sample_n], dtype=np.float32), args.landmarks)
    delta_lm = landmark_distances(delta, landmarks)
    query_lm = landmark_distances(queries, landmarks)

    rows = []

    def record(method, components, qi, mask):
        true_mask = delta_true_masks[qi]
        misses = int(np.count_nonzero(true_mask & ~mask))
        rows.append({
            'method': method,
            'components': components,
            'query': qi,
            'tau_base': float(taus[qi]),
            'candidates': int(mask.sum()),
            'candidate_ratio': float(mask.mean()),
            'true_delta_within_tau': int(true_mask.sum()),
            'false_negatives': misses,
        })

    pca_dims = [2, 4, 8, 16, 32, 64, 96, 128]
    for qi in range(args.queries):
        tau2 = float(taus[qi]) ** 2
        diff = delta_pca - query_pca[qi]
        cumulative = np.cumsum(diff.astype(np.float64) ** 2, axis=1)
        for m in pca_dims:
            mask = cumulative[:, m - 1] <= tau2 + 1e-8
            record('pca_l2_lower_bound', m, qi, mask)
        box2 = (np.abs(diff[:, 0]) <= taus[qi]) & (np.abs(diff[:, 1]) <= taus[qi])
        record('pca2_rt_box', 2, qi, box2)

        lm_diff = np.abs(delta_lm - query_lm[qi])
        for m in [2, 4, 8, 16, 32]:
            mask = np.max(lm_diff[:, :m], axis=1) <= taus[qi]
            record('landmark_max_lower_bound', m, qi, mask)

    csv_path = out / 'selectivity_per_query.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method in sorted(set(r['method'] for r in rows)):
        for comp in sorted(set(r['components'] for r in rows if r['method'] == method)):
            subset = [r for r in rows if r['method'] == method and r['components'] == comp]
            key = method + ':' + str(comp)
            ratios = np.asarray([r['candidate_ratio'] for r in subset])
            summary[key] = {
                'mean_candidate_ratio': float(ratios.mean()),
                'p50_candidate_ratio': float(np.quantile(ratios, 0.50)),
                'p95_candidate_ratio': float(np.quantile(ratios, 0.95)),
                'max_false_negatives': int(max(r['false_negatives'] for r in subset)),
                'total_true_delta_within_tau': int(sum(r['true_delta_within_tau'] for r in subset)),
            }

    meta = {
        'dataset': str(ds),
        'total_vectors': int(n_total),
        'dimension': int(dim),
        'base_count': int(base_n),
        'delta_begin': int(delta_begin),
        'delta_count': int(args.delta_count),
        'queries': int(args.queries),
        'k': int(args.k),
        'threshold_source': 'exact SIFT1M GT filtered to frozen base, distance recomputed in 128D',
        'numeric_note': 'selectivity oracle uses float64 distance accumulation; RT boxes receive an explicit float padding',
        'elapsed_s': time.time() - started,
        'summary': summary,
        'base_seed_ids': base_seed_ids,
        'tau_base': taus.tolist(),
        'delta_true_counts': delta_true_counts,
    }
    (out / 'summary.json').write_text(json.dumps(meta, indent=2) + '\n')

    save_rt_case(out, 'pca2_rt_case', delta_pca[:, :2], query_pca[:, :2], taus)
    save_rt_case(out, 'landmark2_rt_case', delta_lm[:, :2], query_lm[:, :2], taus)

    print(json.dumps({
        'elapsed_s': meta['elapsed_s'],
        'summary': summary,
        'outputs': str(out),
    }, indent=2))


if __name__ == '__main__':
    main()
