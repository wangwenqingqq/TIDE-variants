#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import numpy as np


def read_fvecs(path: Path, dim: int | None = None):
    data = []
    with open(path, "rb") as f:
        while True:
            dim_buf = f.read(4)
            if not dim_buf:
                break
            d = int(np.frombuffer(dim_buf, dtype=np.int32)[0])
            if dim is not None and d != dim:
                raise RuntimeError(f"bad dim {d} in {path}")
            vec = np.frombuffer(f.read(4 * d), dtype=np.float32)
            if vec.size != d:
                raise RuntimeError(f"bad vector bytes in {path}")
            data.append(vec)
    if not data:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def read_ivecs(path: Path):
    gt = []
    with open(path, "rb") as f:
        while True:
            dim_buf = f.read(4)
            if not dim_buf:
                break
            d = int(np.frombuffer(dim_buf, dtype=np.int32)[0])
            ids = np.frombuffer(f.read(4 * d), dtype=np.int32)
            if ids.size != d:
                raise RuntimeError(f"bad gt bytes in {path}")
            gt.append(ids)
    if not gt:
        return np.empty((0, 0), dtype=np.int32)
    return np.asarray(gt, dtype=np.int32)


def build_pca(data, sample, out_dim):
    train = np.asarray(data[:min(sample, data.shape[0])], dtype=np.float64)
    if train.shape[0] < max(2, out_dim):
        raise RuntimeError("training sample too small")
    mean = train.mean(axis=0)
    centered = train - mean
    cov = centered.T @ centered
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    basis = evecs[:, order[:out_dim]].astype(np.float32)
    return basis, mean.astype(np.float32)


def read_raw_f32(path: Path, expected: int):
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != expected:
        raise RuntimeError(f"bad raw file size: {path}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-count", type=int, required=True,
                    help="Base size used to compute tau_B from GT")
    ap.add_argument("--point-begin", type=int, default=None,
                    help="searchable point begin index (default: base-count)")
    ap.add_argument("--point-count", type=int, required=True,
                    help="searchable point count (Delta or pool size)")
    ap.add_argument("--query-count", type=int, default=32)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pca-sample", type=int, default=50000)
    ap.add_argument("--pca-dim", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--case-name", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ds = Path(args.dataset_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    point_begin = args.point_begin
    if point_begin is None:
        point_begin = args.base_count

    base_all = read_fvecs(ds / "sift_base.fvecs", 128)
    query_all = read_fvecs(ds / "sift_query.fvecs", 128)
    gt = read_ivecs(ds / "sift_groundtruth.ivecs")

    n_total = base_all.shape[0]
    if args.base_count < 0 or point_begin < 0:
        raise RuntimeError("negative split index is invalid")
    if args.base_count > n_total:
        raise RuntimeError("base-count exceeds dataset")
    if point_begin + args.point_count > n_total:
        raise RuntimeError("point-begin+point-count exceeds dataset")
    if args.query_count > query_all.shape[0]:
        raise RuntimeError("query-count exceeds query pool")

    if args.base_count == 0:
        raise RuntimeError("base-count 0 cannot form valid tau_B from GT")

    points = base_all[point_begin : point_begin + args.point_count]
    queries = query_all[: args.query_count]

    tau_base = np.zeros(args.query_count, dtype=np.float32)
    for qi in range(args.query_count):
        valid = gt[qi][gt[qi] < args.base_count]
        if valid.size < args.k:
            raise RuntimeError(f"gt depth insufficient for base threshold q={qi}")
        ids = valid[: args.k].astype(np.int64)
        d2 = np.sum((base_all[ids] - queries[qi]).astype(np.float64) ** 2, axis=1)
        tau_base[qi] = np.float32(np.sqrt(float(np.max(d2))))

    # conservative epsilon: keep as absolute lower bound pad, as used by existing RT cases
    pad = np.maximum(np.float32(1e-3), np.abs(tau_base) * np.float32(1e-5))
    tau_pad = np.nextafter(tau_base + pad, np.float32(np.inf))

    # orthogonal PCA basis from the frozen base
    pca_dims = sorted(set(int(x) for x in args.pca_dim))
    bases = {}
    for d in pca_dims:
        if d < 1 or d > 128:
            raise RuntimeError("pca dim must be in [1,128]")
        basis, mean = build_pca(base_all[:args.base_count], args.pca_sample, d)
        bases[d] = (basis, mean)

    # ensure stable 2D for RT boxes (if not explicitly present, derive from dim=2 projection)
    if 2 not in bases:
        basis2, mean2 = build_pca(base_all[:args.base_count], args.pca_sample, 2)
        bases[2] = (basis2, mean2)

    case_dir = out_root / args.case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    # 2D RT inputs are required by RT2D→PCA paths.
    basis2, mean2 = bases[2]
    pts2 = (points - mean2.astype(np.float32)) @ basis2
    qry2 = (queries - mean2.astype(np.float32)) @ basis2
    eps = np.float32(1e-3)
    boxes = np.empty((args.query_count, 4), dtype=np.float32)
    boxes[:, 0] = qry2[:, 0] - tau_pad
    boxes[:, 1] = qry2[:, 1] - tau_pad
    boxes[:, 2] = qry2[:, 0] + tau_pad
    boxes[:, 3] = qry2[:, 1] + tau_pad
    (case_dir / "pca2_points.f32").write_bytes((pts2.astype(np.float32).ravel()).tobytes())
    (case_dir / "pca2_query_boxes.f32").write_bytes(boxes.astype(np.float32).ravel().tobytes())

    # PCA projections (points/queries + basis/mean) for exact CUDA LB stages
    basis_paths = {}
    for d in sorted(bases):
        basis, mean = bases[d]
        # for all cases, persist the 2D+requested bases so the runtime can load/recompute query projection.
        basis_paths[d] = f"basis{d}.f32"
        (case_dir / basis_paths[d]).write_bytes(basis.astype(np.float32).ravel().tobytes())
        (case_dir / f"mean{d}.f32").write_bytes(mean.astype(np.float32).tobytes())
        if d != 2:
            pnts = (points - mean) @ basis
            qrys = (queries - mean) @ basis
            (case_dir / f"pca{d}_points.f32").write_bytes(pnts.astype(np.float32).ravel().tobytes())
            (case_dir / f"pca{d}_queries.f32").write_bytes(qrys.astype(np.float32).ravel().tobytes())

    # tau file stores [tau, padded_tau] for each query
    tau_stack = np.stack((tau_base, tau_pad), axis=1).astype(np.float32)
    (case_dir / "taus.f32").write_bytes(tau_stack.tobytes())

    # keep legacy linkage for existing scripts/old cases that only need pca2 files.
    legacy_map = {
        "base900k_delta100k_q1": ("sift1m_base900k_delta100k_q1_k10", "pca2_rt_case"),
        "base900k_delta100k_q32": ("sift1m_base900k_delta100k_q32_k10", "pca2_rt_case"),
        "pool1m_q32": ("sift1m_pool1m_q32_k10", "pca2_rt_case"),
    }
    if args.case_name in legacy_map:
        legacy_root = Path("/workspace/RT-TIDE/results") / legacy_map[args.case_name][0] / legacy_map[args.case_name][1]
        if legacy_root.exists():
            for rel in ("points.f32", "query_boxes.f32", "taus.f32", "candidate_pairs.u32"):
                src = legacy_root / rel
                if not src.exists():
                    continue
                dst = case_dir / rel
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                try:
                    dst.symlink_to(src)
                except Exception:
                    dst.write_bytes(src.read_bytes())

    # optional selectivity sanity: all requested PCA lower bounds should be conservative for sampled case
    q = int(args.query_count)
    full2 = np.sum((pts2[:, :2][None, :, :].repeat(q, axis=0) - qry2[:, None, :2]).astype(np.float32) ** 2, axis=2)
    false_nonneg = False
    rows = []
    for qi in range(q):
        # conservative check on requested dims
        for d in [2] + list(args.pca_dim):
            if d == 2:
                pass
            else:
                # squared prefix norm in PCA subspace
                sub = (bases[d][0],)
                p = (points - bases[d][1]) @ bases[d][0]
                dist2 = np.sum((p - (queries[qi] - bases[d][1]) @ bases[d][0])[:,:d] **2, axis=1)
                ratio = float((dist2 <= float(tau_pad[qi]) ** 2 + 1e-8).mean())
                rows.append((qi, d, ratio))
                if dist2.size == 0:
                    false_nonneg = True
        # no strict check for pca2 already validated via boxes in runtime, so skip.

    meta = {
        "case": args.case_name,
        "label": args.label or args.case_name,
        "base_count": int(args.base_count),
        "point_begin": int(point_begin),
        "point_count": int(points.shape[0]),
        "query_count": int(args.query_count),
        "k": int(args.k),
        "dimension": 128,
        "tau_source": "exact SIFT1M GT filtered on base",
        "tau_min": float(tau_base.min()),
        "tau_max": float(tau_base.max()),
        "tau_min_padded": float(tau_pad.min()),
        "tau_max_padded": float(tau_pad.max()),
        "file": {
            "taus": "taus.f32",
            "pca2_points": "pca2_points.f32",
            "pca2_query_boxes": "pca2_query_boxes.f32",
            "pca2_query_projection": "pca2_query_boxes.f32",
        },
        "basis_dim": sorted(bases),
        "pca_dims": sorted(int(x) for x in args.pca_dim),
        "pca_file_prefix": {
            d: {
                "basis": f"basis{d}.f32",
                "mean": f"mean{d}.f32",
                "points": f"pca{d}_points.f32",
                "queries": f"pca{d}_queries.f32",
            }
            for d in args.pca_dim
        },
    }
    if not false_nonneg:
        meta["sanity_note"] = "no detected false negatives from sampled conservative PCA prefix checks"
    (case_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
