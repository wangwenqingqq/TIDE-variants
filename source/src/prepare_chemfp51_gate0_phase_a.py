#!/usr/bin/env python3
"""Prepare a license-admissible real-data chemfp shardsearch Gate-0 fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import numpy as np


WORDS = 4
ROW_WORDS = 6
THRESHOLDS = ((7, 10), (4, 5))


def stratified_indices(rows: int, count: int) -> np.ndarray:
    if rows < count:
        raise RuntimeError(f"cannot select {count} rows from {rows}")
    return (np.arange(count, dtype=np.uint64) * rows // count).astype(np.int64)


def fp_hex(words: np.ndarray) -> str:
    return struct.pack("<4Q", *(int(value) for value in words)).hex()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def id_hash(ids: list[int]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(ids)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def write_fps(path: Path, ids: np.ndarray, fps: np.ndarray) -> None:
    part = path.with_name(path.name + ".part")
    with part.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("#FPS1\n#num_bits=256\n")
        stream.write("#software=TIDE-chemfp-Gate0/1\n")
        for record_id, words in zip(ids, fps, strict=True):
            stream.write(f"{fp_hex(words)}\t{int(record_id)}\n")
    os.replace(part, path)


def exact_hits(
    query: np.ndarray, target_ids: np.ndarray, target_fps: np.ndarray, num: int, den: int
) -> list[int]:
    q_words = [int(value) for value in query]
    q_popcount = sum(value.bit_count() for value in q_words)
    hits: list[int] = []
    for record_id, row in zip(target_ids, target_fps, strict=True):
        words = [int(value) for value in row]
        intersection = sum((left & right).bit_count() for left, right in zip(q_words, words))
        union = q_popcount + sum(value.bit_count() for value in words) - intersection
        if union == 0 or den * intersection >= num * union:
            hits.append(int(record_id))
    return hits


def oracle_records(
    query_ids: np.ndarray,
    query_fps: np.ndarray,
    target_ids: np.ndarray,
    target_fps: np.ndarray,
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for num, den in THRESHOLDS:
        threshold = f"{num}/{den}"
        for query_id, query in zip(query_ids, query_fps, strict=True):
            hits = exact_hits(query, target_ids, target_fps, num, den)
            records[f"{int(query_id)}@{threshold}"] = {
                "query_id": int(query_id),
                "threshold": threshold,
                "hit_count": len(hits),
                "sorted_id_sha256": id_hash(hits),
                "query_id_present": int(query_id) in hits,
            }
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-rows", type=int, default=65_536)
    parser.add_argument("--delta-rows", type=int, default=4_096)
    parser.add_argument("--queries", type=int, default=4)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to replace existing output: {output}")
    output.mkdir(parents=True)

    gate0 = source / "data/gate0_prepared"
    stage_a = source / "data/stage_a"
    paths = {
        "base_fp": gate0 / "base_fp_u64x4.bin",
        "base_ids": gate0 / "base_id_i64.bin",
        "base_popcnt": gate0 / "base_popcnt_u16.bin",
        "delta": stage_a / "delta_u64x6.bin",
        "queries": stage_a / "queries_u64x6.bin",
    }
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f"missing source: {path}")

    base_ids_all = np.memmap(paths["base_ids"], mode="r", dtype="<i8")
    base_fps_all = np.memmap(
        paths["base_fp"], mode="r", dtype="<u8", shape=(len(base_ids_all), WORDS)
    )
    base_popcounts_all = np.memmap(paths["base_popcnt"], mode="r", dtype="<u2")
    delta_all = np.memmap(paths["delta"], mode="r", dtype="<u8")
    queries_all = np.memmap(paths["queries"], mode="r", dtype="<u8")
    if delta_all.size % ROW_WORDS or queries_all.size % ROW_WORDS:
        raise RuntimeError("invalid six-word source record")
    delta_all = delta_all.reshape(-1, ROW_WORDS)
    queries_all = queries_all.reshape(-1, ROW_WORDS)
    if len(base_popcounts_all) != len(base_ids_all):
        raise RuntimeError("base popcount length mismatch")

    base_index = stratified_indices(len(base_ids_all), args.base_rows)
    delta_index = stratified_indices(len(delta_all), args.delta_rows)
    query_index = stratified_indices(len(queries_all), args.queries)
    base_ids = np.asarray(base_ids_all[base_index], dtype="<i8")
    base_fps = np.asarray(base_fps_all[base_index], dtype="<u8")
    base_popcounts = np.asarray(base_popcounts_all[base_index], dtype="<u2")
    delta = np.asarray(delta_all[delta_index], dtype="<u8")
    queries = np.asarray(queries_all[query_index], dtype="<u8")
    delta_ids = delta[:, 0].view("<i8")
    delta_fps = delta[:, 1:5]
    delta_popcounts = delta[:, 5]
    query_ids = queries[:, 0].view("<i8")
    query_fps = queries[:, 1:5]
    query_popcounts = queries[:, 5]

    recomputed_base = np.fromiter(
        (sum(int(value).bit_count() for value in row) for row in base_fps),
        dtype=np.uint16,
        count=len(base_fps),
    )
    recomputed_delta = np.fromiter(
        (sum(int(value).bit_count() for value in row) for row in delta_fps),
        dtype=np.uint16,
        count=len(delta_fps),
    )
    recomputed_queries = np.fromiter(
        (sum(int(value).bit_count() for value in row) for row in query_fps),
        dtype=np.uint16,
        count=len(query_fps),
    )
    if not np.array_equal(base_popcounts, recomputed_base):
        raise RuntimeError("base population-count mismatch")
    if not np.array_equal(delta_popcounts, recomputed_delta.astype("<u8")):
        raise RuntimeError("delta population-count mismatch")
    if not np.array_equal(query_popcounts, recomputed_queries.astype("<u8")):
        raise RuntimeError("query population-count mismatch")
    if len(np.unique(base_ids)) != len(base_ids) or len(np.unique(delta_ids)) != len(delta_ids):
        raise RuntimeError("duplicate ID within a selected shard")
    if np.intersect1d(base_ids, delta_ids).size:
        raise RuntimeError("base and delta ID overlap")

    split = len(base_ids) // 2
    generated = {
        "base_a": output / "base_a.fps",
        "base_b": output / "base_b.fps",
        "delta": output / "delta.fps",
        "queries": output / "queries.fps",
        "synthetic_targets": output / "synthetic_targets.fps",
        "synthetic_queries": output / "synthetic_queries.fps",
    }
    write_fps(generated["base_a"], base_ids[:split], base_fps[:split])
    write_fps(generated["base_b"], base_ids[split:], base_fps[split:])
    write_fps(generated["delta"], delta_ids, delta_fps)
    write_fps(generated["queries"], query_ids, query_fps)

    # Q70/T70 has intersection/union 7/10. Q80/T80 has 4/5.
    synthetic_query_ids = np.array([-7001, -8001], dtype="<i8")
    synthetic_query_fps = np.zeros((2, WORDS), dtype="<u8")
    synthetic_target_ids = np.array([-7002, -8002], dtype="<i8")
    synthetic_target_fps = np.zeros((2, WORDS), dtype="<u8")
    synthetic_query_fps[0, 0] = (1 << 7) - 1
    synthetic_target_fps[0, 0] = (1 << 10) - 1
    synthetic_query_fps[1, 0] = (1 << 4) - 1
    synthetic_target_fps[1, 0] = (1 << 5) - 1
    write_fps(generated["synthetic_targets"], synthetic_target_ids, synthetic_target_fps)
    write_fps(generated["synthetic_queries"], synthetic_query_ids, synthetic_query_fps)

    union_ids = np.concatenate([base_ids, delta_ids])
    union_fps = np.concatenate([base_fps, delta_fps])
    result = {
        "experiment_id": "tide_20260903_chemfp51_shardsearch_gate0_phase_a",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_root": str(source),
        "selection": "deterministic evenly spaced source-row ranks",
        "source_rows": {
            "base": len(base_ids_all),
            "delta": len(delta_all),
            "queries": len(queries_all),
        },
        "selected_rows": {
            "base_a": split,
            "base_b": len(base_ids) - split,
            "delta": len(delta_ids),
            "queries": len(query_ids),
        },
        "selected_query_ids": [int(value) for value in query_ids],
        "selected_indices": {
            "base_sha256": hashlib.sha256(base_index.tobytes()).hexdigest(),
            "delta_sha256": hashlib.sha256(delta_index.tobytes()).hexdigest(),
            "queries_sha256": hashlib.sha256(query_index.tobytes()).hexdigest(),
        },
        "population_counts_verified": True,
        "unique_ids_verified": True,
        "generated": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in generated.items()
        },
        "oracle": {
            "base_only": oracle_records(query_ids, query_fps, base_ids, base_fps),
            "base_plus_delta": oracle_records(query_ids, query_fps, union_ids, union_fps),
            "synthetic": oracle_records(
                synthetic_query_ids,
                synthetic_query_fps,
                synthetic_target_ids,
                synthetic_target_fps,
            ),
        },
    }
    manifest = output / "PHASE_A_MANIFEST.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

