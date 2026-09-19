#!/usr/bin/env python3
"""CPU-only provenance/preflight for submitted-VLDB C2 gamma evaluation.

This never opens CUDA and never writes any source archive.  It checks standard
SIFT1M fvecs/ivecs structure, creates a deterministic disjoint query split,
and records raw-input provenance.  It does *not* claim that the public ground
truth provides a theorem for gamma pruning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

SCHEMA = "c2-gamma-sift1m-preflight-v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fvec_meta_and_audit(path: Path, expected_n: int | None = None, expected_dim: int | None = None) -> dict:
    """Scan the fvecs records without materializing them; validates finite values.

    The SIFT1M files use `<int32 dim><dim x float32>` records.  This uses
    memoryview casts for a CPU-only full-file scan and tracks whether every
    coordinate is exactly integral/representable in signed int16.  The runner
    itself keeps raw float values: these are provenance facts, not a request to
    quantize the corpus.
    """
    import numpy as np  # pinned bench_env is explicitly checked by launcher

    size = path.stat().st_size
    if size < 8:
        raise ValueError(f"too small for fvecs: {path}")
    with path.open("rb") as f:
        first_dim = int.from_bytes(f.read(4), "little", signed=True)
    if first_dim <= 0:
        raise ValueError(f"invalid first dimension {first_dim} in {path}")
    record_bytes = (first_dim + 1) * 4
    if size % record_bytes:
        raise ValueError(f"file size {size} is not a multiple of record size {record_bytes}: {path}")
    n = size // record_bytes
    if expected_n is not None and n != expected_n:
        raise ValueError(f"{path}: expected n={expected_n}, got {n}")
    if expected_dim is not None and first_dim != expected_dim:
        raise ValueError(f"{path}: expected dim={expected_dim}, got {first_dim}")
    words = np.memmap(path, mode="r", dtype="<i4", shape=(n, first_dim + 1))
    if not bool(np.all(words[:, 0] == first_dim)):
        mismatch = int(np.flatnonzero(words[:, 0] != first_dim)[0])
        raise ValueError(f"{path}: inconsistent dimension header at record {mismatch}")
    values = words[:, 1:].view("<f4")
    finite = bool(np.all(np.isfinite(values)))
    if not finite:
        raise ValueError(f"{path}: non-finite coordinate present")
    # Chunked reductions avoid accidental large temporary allocations.
    chunk_rows = 16384
    min_value = math.inf
    max_value = -math.inf
    all_integral = True
    all_i16 = True
    for begin in range(0, n, chunk_rows):
        x = values[begin: min(n, begin + chunk_rows)]
        min_value = min(min_value, float(x.min()))
        max_value = max(max_value, float(x.max()))
        if all_integral and not bool(np.all(x == np.rint(x))):
            all_integral = False
        if all_i16 and not bool(np.all((x >= -32768.0) & (x <= 32767.0))):
            all_i16 = False
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": size,
        "records": int(n),
        "dimension": int(first_dim),
        "record_bytes": int(record_bytes),
        "coordinate_min": min_value,
        "coordinate_max": max_value,
        "all_finite": finite,
        "all_integral": all_integral,
        "all_within_int16": all_i16,
        "representation_note": "runner retains raw float32 fvec coordinates; audit fields do not authorize quantization",
    }


def ivecs_meta(path: Path, expected_n: int | None = None, expected_dim: int | None = None) -> dict:
    import numpy as np
    size = path.stat().st_size
    if size < 8:
        raise ValueError(f"too small for ivecs: {path}")
    with path.open("rb") as f:
        dim = int.from_bytes(f.read(4), "little", signed=True)
    if dim <= 0:
        raise ValueError(f"invalid ivecs dim {dim}: {path}")
    record_bytes = (dim + 1) * 4
    if size % record_bytes:
        raise ValueError(f"invalid ivecs record layout: {path}")
    n = size // record_bytes
    if expected_n is not None and n != expected_n:
        raise ValueError(f"{path}: expected n={expected_n}, got {n}")
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(f"{path}: expected dim={expected_dim}, got {dim}")
    words = np.memmap(path, mode="r", dtype="<i4", shape=(n, dim + 1))
    if not bool(np.all(words[:, 0] == dim)):
        mismatch = int(np.flatnonzero(words[:, 0] != dim)[0])
        raise ValueError(f"{path}: inconsistent ivecs header at record {mismatch}")
    ids = words[:, 1:]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": size,
        "records": int(n),
        "dimension": int(dim),
        "record_bytes": int(record_bytes),
        "id_min": int(ids.min()),
        "id_max": int(ids.max()),
        "rows_with_duplicate_ids": int(sum(len(set(map(int, row))) != len(row) for row in ids)),
        "provenance_note": "external public SIFT1M top-100 reference used only for empirical evaluation; it is not a theorem-backed gamma safety certificate",
    }


def deterministic_order(n: int, seed: str) -> list[int]:
    # Hash sorting has a fully specified cross-version definition; no RNG state
    # or NumPy/Python shuffle version is involved.
    def key(qid: int) -> tuple[bytes, int]:
        return hashlib.sha256(f"{seed}:query:{qid}".encode("ascii")).digest(), qid
    return sorted(range(n), key=key)


def write_ids(path: Path, ids: Iterable[int]) -> dict:
    ids = list(ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{qid}\n" for qid in ids)
    path.write_text(payload, encoding="ascii")
    return {"path": str(path), "sha256": sha256_file(path), "count": len(ids), "first_ids": ids[:8]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-fvecs", type=Path, required=True)
    ap.add_argument("--query-fvecs", type=Path, required=True)
    ap.add_argument("--groundtruth-ivecs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", default="submitted-vldb2027-c2-sift1m-v1-20260727")
    ap.add_argument("--calibration-n", type=int, default=2000)
    ap.add_argument("--validation-n", type=int, default=2000)
    ap.add_argument("--test-n", type=int, default=6000)
    args = ap.parse_args()
    if min(args.calibration_n, args.validation_n, args.test_n) <= 0:
        raise SystemExit("all split sizes must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    base = fvec_meta_and_audit(args.base_fvecs, expected_n=1_000_000, expected_dim=128)
    query = fvec_meta_and_audit(args.query_fvecs, expected_n=10_000, expected_dim=128)
    gt = ivecs_meta(args.groundtruth_ivecs, expected_n=10_000, expected_dim=100)
    if args.calibration_n + args.validation_n + args.test_n != query["records"]:
        raise SystemExit("split sizes must sum exactly to query count")
    order = deterministic_order(query["records"], args.seed)
    c = args.calibration_n
    v = c + args.validation_n
    splits = {
        "calibration": write_ids(args.out / "calibration.ids", order[:c]),
        "validation": write_ids(args.out / "validation.ids", order[c:v]),
        "test": write_ids(args.out / "test.ids", order[v:]),
    }
    all_ids = [qid for s in (order[:c], order[c:v], order[v:]) for qid in s]
    if len(set(all_ids)) != len(all_ids) or sorted(all_ids) != list(range(query["records"])):
        raise RuntimeError("split construction is not a partition of all query IDs")
    meta = {
        "schema": SCHEMA,
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "cuda_used": False,
        "source_archives_read_only": [
            "/workspace/legacy_workspace/GTS",
            "/workspace/project/GTS",
        ],
        "inputs": {"base": base, "query": query, "groundtruth": gt},
        "split": {
            "method": "sort query IDs by SHA256(seed + :query: + decimal_id), then take contiguous partitions",
            "seed": args.seed,
            "partition": {"calibration": [0, c], "validation": [c, v], "test": [v, query["records"]]},
            "files": splits,
            "disjoint_and_exhaustive": True,
        },
        "scope": "CPU-only data/provenance preflight for submitted-VLDB C2 gamma-pruning evaluation",
        "non_claims": [
            "No GPU execution occurred.",
            "The public ground truth is an empirical evaluation reference only, not a proof that gamma pruning is safe.",
            "No test query was used to select gamma.",
        ],
    }
    (args.out / "preflight.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_CPU_PREFLIGHT", "out": str(args.out), "base_n": base["records"], "query_n": query["records"], "split": {k: v["count"] for k, v in splits.items()}}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
