#!/usr/bin/env python3
"""Prepare the frozen SureChEMBL snapshot-delta Gate-0 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import hdf5plugin  # noqa: F401  # Registers the HDF5 BLOSC2 filter.
import h5py
import numpy as np


FIELDS = ("f1", "f2", "f3", "f4")


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_fps(path: Path) -> tuple[np.ndarray, dict]:
    with h5py.File(path, "r") as f:
        ds = f["fps"]
        schema = {
            "shape": list(ds.shape),
            "dtype": str(ds.dtype),
            "chunks": list(ds.chunks or ()),
            "compression": str(ds.compression),
        }
        arr = ds[:]
    return arr, schema


def unique_sorted(ids: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    dup = np.flatnonzero(sorted_ids[1:] == sorted_ids[:-1])
    if dup.size:
        raise RuntimeError(f"{label} contains {dup.size} duplicate fp_id values")
    return order, sorted_ids


def write_flat(arr: np.ndarray, out_dir: Path, prefix: str) -> dict:
    n = len(arr)
    fp_path = out_dir / f"{prefix}_fp_u64x4.bin"
    id_path = out_dir / f"{prefix}_id_i64.bin"
    pc_path = out_dir / f"{prefix}_popcnt_u16.bin"

    fp = np.memmap(fp_path, mode="w+", dtype="<u8", shape=(n, 4))
    for j, field in enumerate(FIELDS):
        fp[:, j] = arr[field]
    fp.flush()
    np.asarray(arr["fp_id"], dtype="<i8").tofile(id_path)
    np.asarray(arr["popcnt"], dtype="<u2").tofile(pc_path)
    return {
        "count": n,
        "fp": str(fp_path),
        "ids": str(id_path),
        "popcnt": str(pc_path),
        "fp_sha256": sha256_file(fp_path),
        "ids_sha256": sha256_file(id_path),
        "popcnt_sha256": sha256_file(pc_path),
    }


def fingerprint_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    same = np.ones(len(a), dtype=bool)
    for field in FIELDS:
        same &= a[field] == b[field]
    same &= a["popcnt"] == b["popcnt"]
    return same


def bitcount_u64(x: np.ndarray) -> np.ndarray:
    # NumPy 2.x exposes bitwise_count; retain a byte-table fallback.
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).sum(axis=1)
    table = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    return table[x.view(np.uint8).reshape(len(x), -1)].sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--union", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", default="20260827")
    ap.add_argument("--calibration", type=int, default=512)
    ap.add_argument("--test", type=int, default=4096)
    args = ap.parse_args()

    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    print("reading base", flush=True)
    base, base_schema = read_fps(args.base)
    print("reading union", flush=True)
    union, union_schema = read_fps(args.union)
    if base.dtype != union.dtype:
        raise RuntimeError(f"dtype mismatch: {base.dtype} vs {union.dtype}")

    print("sorting IDs", flush=True)
    b_order, b_ids = unique_sorted(base["fp_id"], "base")
    u_order, u_ids = unique_sorted(union["fp_id"], "union")

    bpos_for_u = np.searchsorted(b_ids, u_ids)
    u_added = bpos_for_u == len(b_ids)
    in_range = ~u_added
    u_added[in_range] = b_ids[bpos_for_u[in_range]] != u_ids[in_range]

    upos_for_b = np.searchsorted(u_ids, b_ids)
    b_removed = upos_for_b == len(u_ids)
    in_range = ~b_removed
    b_removed[in_range] = u_ids[upos_for_b[in_range]] != b_ids[in_range]

    common_u_sorted = np.flatnonzero(~u_added)
    common_b_sorted = bpos_for_u[common_u_sorted]
    changed_ids: list[int] = []
    chunk = 1_000_000
    print("checking common fingerprints", flush=True)
    for start in range(0, len(common_u_sorted), chunk):
        us = common_u_sorted[start : start + chunk]
        bs = common_b_sorted[start : start + chunk]
        u_rows = union[u_order[us]]
        b_rows = base[b_order[bs]]
        neq = ~fingerprint_equal(u_rows, b_rows)
        if np.any(neq):
            changed_ids.extend(int(x) for x in u_ids[us[neq]])

    added_union_rows = u_order[np.flatnonzero(u_added)]
    removed_base_rows = b_order[np.flatnonzero(b_removed)]
    added_ids = union["fp_id"][added_union_rows]
    ranked = sorted(
        ((hashlib.sha256(f"{args.seed}:{int(fp_id)}".encode()).digest(), int(row))
         for fp_id, row in zip(added_ids, added_union_rows, strict=True)),
        key=lambda x: x[0],
    )
    required = args.calibration + args.test
    if len(ranked) < required:
        raise RuntimeError(f"only {len(ranked)} added IDs; need {required}")
    selected_rows = np.array([row for _, row in ranked[:required]], dtype=np.int64)
    query_rows = union[selected_rows]

    # Verify stored population counts on all queries and a deterministic base sample.
    sample_rows = np.unique(
        np.concatenate(
            [selected_rows, np.linspace(0, len(union) - 1, 10000, dtype=np.int64)]
        )
    )
    sample_fp = np.stack([union[f][sample_rows] for f in FIELDS], axis=1)
    recomputed = bitcount_u64(sample_fp)
    stored = union["popcnt"][sample_rows]
    popcnt_mismatch = int(np.count_nonzero(recomputed != stored))
    if popcnt_mismatch:
        raise RuntimeError(f"population-count mismatch in {popcnt_mismatch} sampled rows")

    print("writing flat arrays", flush=True)
    base_files = write_flat(base, args.out, "base")
    union_files = write_flat(union, args.out, "union")
    query_files = write_flat(query_rows, args.out, "queries")
    np.asarray(selected_rows, dtype="<i8").tofile(args.out / "query_union_row_i64.bin")

    query_manifest = []
    for rank, row in enumerate(selected_rows):
        record = union[int(row)]
        query_manifest.append(
            {
                "split": "calibration" if rank < args.calibration else "test",
                "rank": rank,
                "union_row": int(row),
                "fp_id": int(record["fp_id"]),
                "popcnt": int(record["popcnt"]),
                "selection_hash": hashlib.sha256(
                    f"{args.seed}:{int(record['fp_id'])}".encode()
                ).hexdigest(),
            }
        )

    manifest = {
        "experiment_id": "tide_20260827_surechembl_delta_exact_top10",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "source": {
            "base": str(args.base),
            "union": str(args.union),
            "base_sha256": sha256_file(args.base),
            "union_sha256": sha256_file(args.union),
            "base_schema": base_schema,
            "union_schema": union_schema,
        },
        "counts": {
            "base": int(len(base)),
            "union": int(len(union)),
            "added": int(len(added_union_rows)),
            "removed": int(len(removed_base_rows)),
            "changed_common": int(len(changed_ids)),
            "calibration_queries": args.calibration,
            "test_queries": args.test,
        },
        "removed_ids_sample": [int(base["fp_id"][r]) for r in removed_base_rows[:100]],
        "changed_ids_sample": changed_ids[:100],
        "popcnt_sample_rows": int(len(sample_rows)),
        "popcnt_sample_mismatches": popcnt_mismatch,
        "files": {"base": base_files, "union": union_files, "queries": query_files},
        "elapsed_seconds": time.time() - t0,
    }
    (args.out / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "query_manifest.json").write_text(json.dumps(query_manifest, indent=2) + "\n")
    with (args.out / "added_ids_i64.bin").open("wb") as f:
        np.asarray(added_ids, dtype="<i8").tofile(f)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"elapsed_seconds={time.time() - t0:.3f}")


if __name__ == "__main__":
    main()
