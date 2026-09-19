#!/usr/bin/env python3
"""Measure complete threshold-result cardinality within exact popcount bounds."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
from FPSim2 import FPSim2Engine
from FPSim2.FPSim2lib import GenericSearch


THRESHOLDS = ((7, 10), (4, 5))
SPLITS = (("calibration", 0, 512), ("sealed_test", 512, 4608))


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def popcount_interval(query_popcount: int, num: int, den: int) -> tuple[int, int]:
    lower = ceil_div(num * query_popcount, den)
    upper = (den * query_popcount) // num
    return max(0, lower), min(256, upper)


def dense_counts(engine: FPSim2Engine) -> np.ndarray:
    counts = np.zeros(257, dtype=np.int64)
    for popcount, (begin, end) in engine.popcnt_bins:
        counts[int(popcount)] = int(end) - int(begin)
    if int(counts.sum()) != int(engine.fps.shape[0]):
        raise RuntimeError("population-count directory does not cover the database")
    return counts


def percentile_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(values.size),
        "min": int(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(np.max(values)),
        "mean": float(np.mean(values)),
    }


def normalized_result_hash(results: np.ndarray) -> str:
    coeff_bits = results["coeff"].view(np.uint32)
    rows = sorted(
        zip(results["mol_id"].tolist(), coeff_bits.tolist(), strict=True),
        key=lambda row: (-row[1], row[0]),
    )
    digest = hashlib.sha256()
    for mol_id, coeff in rows:
        digest.update(struct.pack("<II", int(mol_id), int(coeff)))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--union-h5", type=Path, required=True)
    parser.add_argument("--queries-u64x6", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # Materialize ordinary arrays: this pinned pybind build is unsafe with the
    # NumPy memmap subclass even when the view appears contiguous.
    queries = np.fromfile(args.queries_u64x6, dtype="<u8").reshape(-1, 6)
    if queries.shape != (4608, 6):
        raise RuntimeError(f"unexpected query shape: {queries.shape}")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    engine = FPSim2Engine(str(args.union_h5))
    database = engine.fps
    counts = dense_counts(engine)
    cumulative = np.concatenate(([0], np.cumsum(counts)))

    tasks = [
        (query_index, num, den)
        for query_index in range(queries.shape[0])
        for num, den in THRESHOLDS
    ]

    def search(task: tuple[int, int, int]) -> dict:
        query_index, num, den = task
        query = np.array(queries[query_index], dtype=np.uint64, copy=True)
        query_popcount = int(query[5])
        lower, upper = popcount_interval(query_popcount, num, den)
        begin = int(cumulative[lower])
        end = int(cumulative[upper + 1])
        started = time.perf_counter_ns()
        results = GenericSearch(query, database, num / den, 0, 0, begin, end)
        elapsed_ns = time.perf_counter_ns() - started

        if results.size:
            result_popcounts = database[results["idx"].astype(np.int64), 5]
            outside = int(
                np.count_nonzero(
                    (result_popcounts < lower) | (result_popcounts > upper)
                )
            )
            self_count = int(np.count_nonzero(results["mol_id"] == query[0]))
        else:
            outside = 0
            self_count = 0

        return {
            "query_index": query_index,
            "query_mol_id": int(query[0]),
            "query_popcount": query_popcount,
            "threshold": f"{num / den:.2f}",
            "threshold_rational": [num, den],
            "lower_popcount": lower,
            "upper_popcount": upper,
            "eligible_begin": begin,
            "eligible_end": end,
            "eligible_rows": end - begin,
            "hit_count_including_self": int(results.size),
            "self_hit_count": self_count,
            "hit_count_excluding_self": int(results.size) - self_count,
            "outside_bound_count": outside,
            "normalized_result_sha256": normalized_result_hash(results),
            "diagnostic_elapsed_ns": elapsed_ns,
        }

    started = time.perf_counter()
    records: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for completed, record in enumerate(executor.map(search, tasks), start=1):
            records.append(record)
            if completed % 512 == 0:
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(tasks),
                            "wall_s": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )

    records.sort(key=lambda row: (row["query_index"], row["threshold"]))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    split_summary = {}
    for split, first, last in SPLITS:
        split_summary[split] = {}
        for num, den in THRESHOLDS:
            threshold = f"{num / den:.2f}"
            selected = [
                row
                for row in records
                if first <= row["query_index"] < last
                and row["threshold"] == threshold
            ]
            including = np.array(
                [row["hit_count_including_self"] for row in selected],
                dtype=np.int64,
            )
            excluding = np.array(
                [row["hit_count_excluding_self"] for row in selected],
                dtype=np.int64,
            )
            split_summary[split][threshold] = {
                "query_count": len(selected),
                "hit_count_including_self": percentile_summary(including),
                "hit_count_excluding_self": percentile_summary(excluding),
                "zero_hit_queries_including_self": int(np.count_nonzero(including == 0)),
                "zero_hit_queries_excluding_self": int(np.count_nonzero(excluding == 0)),
                "self_hit_total": int(
                    sum(row["self_hit_count"] for row in selected)
                ),
                "outside_bound_total": int(
                    sum(row["outside_bound_count"] for row in selected)
                ),
            }

    outside_total = int(sum(row["outside_bound_count"] for row in records))
    cardinality_complete = len(records) == len(tasks) and all(
        row["hit_count_including_self"] >= row["self_hit_count"] >= 0
        and len(row["normalized_result_sha256"]) == 64
        for row in records
    )
    summary = {
        "experiment_id": "tide_20260827_exact_threshold_cardinality",
        "database_rows": int(database.shape[0]),
        "query_count": int(queries.shape[0]),
        "thresholds": [num / den for num, den in THRESHOLDS],
        "workers": args.workers,
        "search_record_count": len(records),
        "wall_s": time.perf_counter() - started,
        "splits": split_summary,
        "gates": {
            "G4_A_BOUND": outside_total == 0,
            "G4_A_CARDINALITY": cardinality_complete,
        },
        "notes": [
            "Each search used k=0, so FPSim2 materialized every qualifying result in a dynamically sized vector.",
            "The elapsed fields are diagnostics from a concurrent CPU run and are not Gate-4 performance evidence.",
        ],
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
