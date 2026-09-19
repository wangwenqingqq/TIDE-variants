#!/usr/bin/env python3
"""Create a small, exact, width-preserving Gate-6 sanitizer fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


WORDS = 4
ROW_WORDS = WORDS + 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def stratified_indices(rows: int, count: int) -> np.ndarray:
    if rows < count:
        raise RuntimeError(f"cannot select {count} rows from {rows}")
    return (np.arange(count, dtype=np.uint64) * rows // count).astype(np.int64)


def atomic_tofile(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    array.tofile(part)
    os.replace(part, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-rows", type=int, default=65_536)
    parser.add_argument("--delta-rows", type=int, default=4_096)
    parser.add_argument("--queries", type=int, default=4)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to replace existing fixture: {output}")

    base_fp_path = source / "data/gate0_prepared/base_fp_u64x4.bin"
    base_id_path = source / "data/gate0_prepared/base_id_i64.bin"
    base_pc_path = source / "data/gate0_prepared/base_popcnt_u16.bin"
    delta_path = source / "data/stage_a/delta_u64x6.bin"
    base_ids_all = np.memmap(base_id_path, mode="r", dtype="<i8")
    base_fp_all = np.memmap(
        base_fp_path, mode="r", dtype="<u8", shape=(len(base_ids_all), WORDS)
    )
    base_pc_all = np.memmap(base_pc_path, mode="r", dtype="<u2")
    packed_delta = np.memmap(delta_path, mode="r", dtype="<u8")
    if packed_delta.size % ROW_WORDS:
        raise RuntimeError("invalid delta row width")
    packed_delta = packed_delta.reshape(-1, ROW_WORDS)

    base_index = stratified_indices(len(base_ids_all), args.base_rows)
    delta_index = stratified_indices(len(packed_delta), args.delta_rows)
    base_fp = np.asarray(base_fp_all[base_index], dtype="<u8")
    base_ids = np.asarray(base_ids_all[base_index], dtype="<i8")
    base_pc = np.asarray(base_pc_all[base_index], dtype="<u2")
    delta = np.asarray(packed_delta[delta_index], dtype="<u8")
    delta_ids = delta[:, 0].view("<i8")
    delta_fp = delta[:, 1 : 1 + WORDS]
    delta_pc = delta[:, -1].astype("<u2")
    if np.intersect1d(base_ids, delta_ids).size:
        raise RuntimeError("fixture base and delta IDs overlap")
    if np.any(base_pc[1:] < base_pc[:-1]) or np.any(delta_pc[1:] < delta_pc[:-1]):
        raise RuntimeError("fixture input order is not population-count sorted")

    union_fp = np.concatenate([base_fp, delta_fp], axis=0)
    union_ids = np.concatenate([base_ids, delta_ids], axis=0)
    union_pc = np.concatenate([base_pc, delta_pc], axis=0)
    order = np.lexsort((union_ids, union_pc))
    union_fp = union_fp[order]
    union_ids = union_ids[order]
    union_pc = union_pc[order]
    queries = delta[stratified_indices(len(delta), args.queries)]

    paths = {
        "base_fp": output / "data/gate0_prepared/base_fp_u64x4.bin",
        "base_ids": output / "data/gate0_prepared/base_id_i64.bin",
        "base_pc": output / "data/gate0_prepared/base_popcnt_u16.bin",
        "union_fp": output / "data/gate0_prepared/union_fp_u64x4.bin",
        "union_ids": output / "data/gate0_prepared/union_id_i64.bin",
        "union_pc": output / "data/gate0_prepared/union_popcnt_u16.bin",
        "delta": output / "data/stage_a/delta_u64x6.bin",
        "queries": output / "data/stage_a/queries_u64x6.bin",
    }
    for array, key in [
        (base_fp, "base_fp"),
        (base_ids, "base_ids"),
        (base_pc, "base_pc"),
        (union_fp, "union_fp"),
        (union_ids, "union_ids"),
        (union_pc, "union_pc"),
        (delta, "delta"),
        (queries, "queries"),
    ]:
        atomic_tofile(np.ascontiguousarray(array), paths[key])

    result = {
        "experiment_id": "tide_20260829_gate6_sanitizer_fixture",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_root": str(source),
        "selection": "deterministic evenly spaced source-row ranks",
        "base_rows": len(base_ids),
        "delta_rows": len(delta),
        "union_rows": len(union_ids),
        "query_rows": len(queries),
        "word_count": WORDS,
        "population_count_sorted": {
            "base": bool(np.all(base_pc[1:] >= base_pc[:-1])),
            "delta": bool(np.all(delta_pc[1:] >= delta_pc[:-1])),
            "union": bool(np.all(union_pc[1:] >= union_pc[:-1])),
        },
        "files": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for key, path in paths.items()
        },
    }
    manifest = output / "FIXTURE_MANIFEST.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
