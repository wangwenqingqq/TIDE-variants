#!/usr/bin/env python3
"""Validate actual FPSim2Cuda 0.7.4 result sets against exact ID oracles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import cupy as cp
import numpy as np

import FPSim2
from FPSim2 import FPSim2CudaEngine


WORDS = 4
ROW_WORDS = WORDS + 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def hash_ids(ids: np.ndarray) -> int:
    value = 1469598103934665603
    for byte in np.asarray(ids, dtype="<u8").tobytes(order="C"):
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def load_oracle(path: Path) -> dict[tuple[int, int, int], dict[str, int]]:
    result = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["query"]), int(row["threshold_num"]), int(row["threshold_den"]))
            result[key] = {
                "candidate_rows": int(row["candidate_rows"]),
                "hits": int(row["hits"]),
                "id_hash": int(row["id_hash"]),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--query-limit", type=int, default=512)
    args = parser.parse_args()
    if getattr(FPSim2, "__version__", "unknown") != "0.7.4":
        raise RuntimeError(f"FPSim2 version mismatch: {getattr(FPSim2, '__version__', 'unknown')}")

    packed = np.memmap(args.queries, mode="r", dtype="<u8")
    if packed.size % ROW_WORDS:
        raise RuntimeError("invalid query row width")
    packed = packed.reshape(-1, ROW_WORDS)
    query_count = min(len(packed), args.query_limit)
    oracle = load_oracle(args.oracle)
    expected_requests = query_count * 2
    if len([key for key in oracle if key[0] < query_count]) != expected_requests:
        raise RuntimeError("incomplete exact oracle denominator")

    cp.cuda.Device(args.gpu).use()
    setup_begin = time.perf_counter()
    engine = FPSim2CudaEngine(str(args.h5))
    cp.cuda.Stream.null.synchronize()
    setup_s = time.perf_counter() - setup_begin
    records = []
    mismatches = 0
    duplicate_requests = 0
    for query_index in range(query_count):
        query = np.asarray(packed[query_index], dtype=np.uint64)
        for numerator, denominator in ((7, 10), (4, 5)):
            threshold = numerator / denominator
            begin = time.perf_counter()
            ids, similarities = engine._raw_kernel_search(query, threshold)
            cp.cuda.Stream.null.synchronize()
            service_s = time.perf_counter() - begin
            ids = np.sort(np.asarray(ids, dtype="<u8"))
            duplicate = bool(len(ids) > 1 and np.any(ids[1:] == ids[:-1]))
            duplicate_requests += duplicate
            actual_hash = hash_ids(ids)
            reference = oracle[(query_index, numerator, denominator)]
            match = (
                not duplicate
                and len(ids) == reference["hits"]
                and actual_hash == reference["id_hash"]
            )
            mismatches += not match
            records.append(
                {
                    "query": query_index,
                    "threshold_num": numerator,
                    "threshold_den": denominator,
                    "hits": len(ids),
                    "id_hash": actual_hash,
                    "oracle_hits": reference["hits"],
                    "oracle_id_hash": reference["id_hash"],
                    "duplicate_id": duplicate,
                    "match": match,
                    "diagnostic_service_ms": service_s * 1e3,
                    "minimum_similarity": float(np.min(similarities)) if len(similarities) else None,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample_path = Path(str(args.output) + ".samples.csv")
    with sample_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = {
        "experiment_id": "tide_20260829_gate6_actual_fpsim2cuda_exact",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": (
            "actual FPSim2CudaEngine 0.7.4 resident database and released "
            "_raw_kernel_search path with precomputed frozen fingerprints; "
            "SMILES parsing and fingerprint generation excluded"
        ),
        "runtime": {
            "python": platform.python_version(),
            "fpsim2": getattr(FPSim2, "__version__", "unknown"),
            "cupy": cp.__version__,
            "gpu_ordinal": args.gpu,
        },
        "inputs": {
            "h5": str(args.h5),
            "h5_sha256": sha256(args.h5),
            "queries": str(args.queries),
            "queries_sha256": sha256(args.queries),
            "oracle": str(args.oracle),
            "oracle_sha256": sha256(args.oracle),
        },
        "setup_s_excluded_from_request": setup_s,
        "query_count": query_count,
        "requests": len(records),
        "result_set_mismatches": mismatches,
        "duplicate_id_requests": duplicate_requests,
        "actual_fpsim2cuda_exact_component_pass": bool(
            len(records) == expected_requests
            and mismatches == 0
            and duplicate_requests == 0
        ),
        "float_threshold_caveat": (
            "FPSim2Cuda 0.7.4 uses float32 threshold/coefficients; equality is "
            "judged only by exact-rational oracle result-set IDs."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["actual_fpsim2cuda_exact_component_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
