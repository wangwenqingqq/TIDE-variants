#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate0-prepared", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    p = args.gate0_prepared
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    manifest0 = json.loads((p / "data_manifest.json").read_text())
    n_union = int(manifest0["counts"]["union"])
    n_queries = int(manifest0["counts"]["calibration_queries"] +
                    manifest0["counts"]["test_queries"])

    union_ids = np.memmap(p / "union_id_i64.bin", dtype="<i8", mode="r",
                          shape=(n_union,))
    union_fp = np.memmap(p / "union_fp_u64x4.bin", dtype="<u8", mode="r",
                         shape=(n_union, 4))
    union_pc = np.memmap(p / "union_popcnt_u16.bin", dtype="<u2", mode="r",
                         shape=(n_union,))
    added_ids = np.fromfile(p / "added_ids_i64.bin", dtype="<i8")
    if len(added_ids) != int(manifest0["counts"]["added"]):
        raise RuntimeError("added ID count mismatch")

    mask = np.isin(union_ids, added_ids, assume_unique=True)
    idx = np.flatnonzero(mask)
    if len(idx) != len(added_ids):
        raise RuntimeError(f"delta rows {len(idx)} != added IDs {len(added_ids)}")
    if not np.array_equal(np.sort(union_ids[idx]), np.sort(added_ids)):
        raise RuntimeError("delta ID set mismatch")

    delta = np.empty((len(idx), 6), dtype="<u8")
    delta[:, 0] = union_ids[idx].view("<u8")
    delta[:, 1:5] = union_fp[idx]
    delta[:, 5] = union_pc[idx]
    order = np.lexsort((delta[:, 0], delta[:, 5]))
    delta = np.ascontiguousarray(delta[order])
    delta_path = out / "delta_u64x6.bin"
    delta.tofile(delta_path)

    qids = np.memmap(p / "queries_id_i64.bin", dtype="<i8", mode="r",
                     shape=(n_queries,))
    qfp = np.memmap(p / "queries_fp_u64x4.bin", dtype="<u8", mode="r",
                    shape=(n_queries, 4))
    qpc = np.memmap(p / "queries_popcnt_u16.bin", dtype="<u2", mode="r",
                    shape=(n_queries,))
    queries = np.empty((n_queries, 6), dtype="<u8")
    queries[:, 0] = qids.view("<u8")
    queries[:, 1:5] = qfp
    queries[:, 5] = qpc
    query_path = out / "queries_u64x6.bin"
    queries.tofile(query_path)

    if np.any(delta[1:, 5] < delta[:-1, 5]):
        raise RuntimeError("delta not sorted by population count")
    recomputed = np.bitwise_count(delta[:, 1:5]).sum(axis=1, dtype=np.uint16)
    if not np.array_equal(recomputed, delta[:, 5].astype(np.uint16)):
        raise RuntimeError("delta population count mismatch")
    q_recomputed = np.bitwise_count(queries[:, 1:5]).sum(axis=1, dtype=np.uint16)
    if not np.array_equal(q_recomputed, queries[:, 5].astype(np.uint16)):
        raise RuntimeError("query population count mismatch")

    result = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gate0_manifest_sha256": sha256(p / "data_manifest.json"),
        "delta_rows": int(delta.shape[0]),
        "delta_columns": int(delta.shape[1]),
        "delta_bytes": delta_path.stat().st_size,
        "delta_sha256": sha256(delta_path),
        "query_rows": int(queries.shape[0]),
        "query_columns": int(queries.shape[1]),
        "query_bytes": query_path.stat().st_size,
        "query_sha256": sha256(query_path),
        "sort": "population_count ascending, fp_id ascending",
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (out / "stage_a_data_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
