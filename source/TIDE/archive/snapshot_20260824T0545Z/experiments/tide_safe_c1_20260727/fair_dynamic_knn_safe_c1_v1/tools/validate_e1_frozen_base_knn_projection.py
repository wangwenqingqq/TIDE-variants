#!/usr/bin/env python3
"""Independent CPU validator for an E1 frozen-base KNN projection bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
MAGIC = b"E1GTRC01"
INSERT, KNN = 1, 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def active_hash(active: set[int]) -> str:
    return hashlib.sha256("".join(f"{sid}\n" for sid in sorted(active)).encode("ascii")).hexdigest()


def parse_trace(path: Path):
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise AssertionError("truncated trace")
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, radius, n = HEADER.unpack_from(raw)
    if magic != MAGIC or version != 1:
        raise AssertionError("bad magic/version")
    if len(raw) != HEADER.size + n * EVENT.size:
        raise AssertionError("bad trace byte length")
    out = []
    for position in range(n):
        op_index, op, arg = EVENT.unpack_from(raw, HEADER.size + position * EVENT.size)
        if op_index != position or op not in (INSERT, KNN):
            raise AssertionError(f"illegal projection event at {position}")
        out.append((op_index, op, arg))
    return {
        "dimension": dim, "base_n": base_n, "reservoir_n": reservoir_n,
        "pool_n": pool_n, "query_n": query_n, "k": k, "radius": radius,
        "event_count": n,
    }, out, raw


def canonical_topk(pool: np.ndarray, mapping: np.ndarray, query: np.ndarray,
                   active: set[int], k: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.fromiter(sorted(active), dtype=np.int64, count=len(active))
    rows = mapping[ids]
    values = pool[rows].astype(np.int64, copy=False)
    diff = values - query.astype(np.int64, copy=False)
    sq = np.sum(diff * diff, axis=1, dtype=np.int64)
    order = np.lexsort((ids, sq))[:k]
    return ids[order], sq[order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    metadata = json.loads((bundle / "metadata.json").read_text())
    assert manifest["schema"] == "e1-frozen-base-knn-projection-manifest-v1"
    assert metadata["schema"] == "e1-frozen-base-knn-projection-bundle-v1"
    assert metadata["projection"]["allowed_ops"] == ["insert", "knn"]
    assert metadata["projection"]["excluded_ops"] == ["delete", "range"]
    for name, expected in manifest["files_sha256"].items():
        got = sha256_file(bundle / name)
        if got != expected:
            raise AssertionError(f"hash mismatch {name}: {got} != {expected}")
    header, events, raw = parse_trace(bundle / "trace.e1gtrc")
    expected_header = metadata["header"]
    for name in ("dimension", "base_n", "reservoir_n", "pool_n", "query_n", "k", "event_count"):
        if header[name] != expected_header[name]:
            raise AssertionError(f"header mismatch {name}")
    if sha256_file(bundle / "trace.e1gtrc") != metadata["projection"]["projection_trace_sha256"]:
        raise AssertionError("projection trace hash mismatch")
    if hashlib.sha256(raw[HEADER.size:]).hexdigest() != metadata["projection"]["projection_event_stream_sha256"]:
        raise AssertionError("projection event stream hash mismatch")

    dim, pool_n = header["dimension"], header["pool_n"]
    pool = np.fromfile(bundle / "pool.i16", dtype="<i2")
    if pool.size % dim:
        raise AssertionError("pool shape")
    pool = pool.reshape((-1, dim))
    mapping = np.fromfile(bundle / "stable_id_to_pool_row.i32", dtype="<i4")
    base_ids = np.fromfile(bundle / "initial_base_stable_ids.i32", dtype="<i4")
    queries = np.fromfile(bundle / "queries.i16", dtype="<i2")
    if mapping.size != pool_n or base_ids.size != header["base_n"] or queries.size != header["query_n"] * dim:
        raise AssertionError("binary input shape")
    queries = queries.reshape((header["query_n"], dim))
    if len(np.unique(mapping)) != mapping.size or mapping.min() < 0 or mapping.max() >= pool.shape[0]:
        raise AssertionError("invalid stable-to-pool mapping")
    if len(np.unique(base_ids)) != base_ids.size or base_ids.min() < 0 or base_ids.max() >= pool_n:
        raise AssertionError("invalid initial base ids")

    active = set(map(int, base_ids.tolist()))
    seen_inserts: set[int] = set()
    counts: Counter = Counter()
    sequence = hashlib.sha256()
    for op_index, op, argument in events:
        if op == INSERT:
            if not (header["base_n"] <= argument < pool_n):
                raise AssertionError(f"insert {op_index} outside declared reservoir")
            if argument in active or argument in seen_inserts:
                raise AssertionError(f"insert {op_index} duplicates stable ID")
            active.add(argument)
            seen_inserts.add(argument)
            counts["insert"] += 1
        else:
            if not (0 <= argument < header["query_n"]):
                raise AssertionError(f"KNN {op_index} invalid query id")
            ids, sq = canonical_topk(pool, mapping, queries[argument], active, header["k"])
            if ids.size != header["k"] or len(set(map(int, ids))) != header["k"]:
                raise AssertionError(f"KNN {op_index} invalid canonical top-k")
            sequence.update(f"{op_index}:{argument}:{active_hash(active)}:".encode("ascii"))
            for sid, distance in zip(ids.tolist(), sq.tolist()):
                sequence.update(f"{sid}:{distance},".encode("ascii"))
            sequence.update(b"\n")
            counts["knn"] += 1
    if dict(counts) != metadata["trace_counts"]["projection"]:
        raise AssertionError(f"count mismatch {dict(counts)}")
    final_hash = active_hash(active)
    if len(active) != metadata["projection"]["final_active_count"] or final_hash != metadata["projection"]["final_active_set_sha256"]:
        raise AssertionError("final active-set mismatch")
    result = {
        "schema": "e1-frozen-base-knn-projection-validator-v1",
        "status": "PASS",
        "bundle": str(bundle),
        "trace_sha256": sha256_file(bundle / "trace.e1gtrc"),
        "validated_events": len(events),
        "insert": counts["insert"],
        "knn": counts["knn"],
        "final_active_count": len(active),
        "final_active_set_sha256": final_hash,
        "canonical_int64_knn_sequence_sha256": sequence.hexdigest(),
        "oracle": "int64 squared L2; canonical order=(distance_sq,stable_id); source oracle file not used",
    }
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        out = args.out.resolve()
        if out.exists():
            raise SystemExit(f"refusing to overwrite existing output: {out}")
        out.write_text(text, encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
