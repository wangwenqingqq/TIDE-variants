#!/usr/bin/env python3
"""Measure nvMolKit 0.6.0 under a complete threshold-result contract."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import resource
import struct
import sys
import time
from typing import Any

import numpy as np
import torch
from nvmolkit import __version__ as nvmolkit_version
from nvmolkit.similarity import crossTanimotoSimilarity


MASK64 = (1 << 64) - 1


def parse_thresholds(value: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for token in value.split(","):
        left, right = token.split("/", 1)
        numerator, denominator = int(left), int(right)
        if not (0 < numerator <= denominator):
            raise ValueError(f"invalid threshold {token}")
        result.append((numerator, denominator))
    if len(result) != 2 or result[0] == result[1]:
        raise ValueError("exactly two distinct thresholds are required")
    return result


def load_oracle(path: pathlib.Path) -> dict[tuple[int, int, int], tuple[int, int]]:
    result: dict[tuple[int, int, int], tuple[int, int]] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (
                int(row["query"]),
                int(row["threshold_num"]),
                int(row["threshold_den"]),
            )
            if key in result:
                raise ValueError(f"duplicate oracle key {key}")
            result[key] = (int(row["hits"]), int(row["id_hash"]))
    return result


def hash_ids(values: np.ndarray) -> int:
    value = 1469598103934665603
    for identifier in values:
        for byte in struct.pack("<Q", int(identifier) & MASK64):
            value ^= byte
            value = (value * 1099511628211) & MASK64
    return value


def percentile(values: list[float], probability: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 100 * probability))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprints", type=pathlib.Path, required=True)
    parser.add_argument("--ids", type=pathlib.Path, required=True)
    parser.add_argument("--queries", type=pathlib.Path, required=True)
    parser.add_argument("--oracle", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--metadata-output", type=pathlib.Path, required=True)
    parser.add_argument("--threshold-order", default="7/10,4/5")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--query-count", type=int, default=0)
    parser.add_argument("--expected-device-name", default="")
    parser.add_argument("--max-device-bytes", type=int, default=8 * 1024**3)
    args = parser.parse_args()

    for target in (args.output, args.metadata_output):
        if target.exists():
            raise FileExistsError(f"refusing to replace {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
    if sys.byteorder != "little":
        raise RuntimeError("the frozen uint64-to-uint32 view requires little endian")
    if args.warmup < 0 or args.query_count < 0:
        raise ValueError("warmup and query-count must be nonnegative")

    thresholds = parse_thresholds(args.threshold_order)
    oracle = load_oracle(args.oracle)
    if nvmolkit_version != "0.6.0":
        raise RuntimeError(f"expected nvMolKit 0.6.0, got {nvmolkit_version}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if args.expected_device_name and args.expected_device_name not in properties.name:
        raise RuntimeError(
            f"unexpected device {properties.name!r}; expected substring {args.expected_device_name!r}"
        )

    fp_bytes = args.fingerprints.stat().st_size
    id_bytes = args.ids.stat().st_size
    query_bytes = args.queries.stat().st_size
    if fp_bytes % 32 or id_bytes % 8 or query_bytes % 48:
        raise ValueError("frozen binary file size is not record aligned")
    rows = fp_bytes // 32
    if id_bytes // 8 != rows:
        raise ValueError("fingerprint and stable-ID row counts differ")
    query_rows = query_bytes // 48
    retained_queries = query_rows if args.query_count == 0 else args.query_count
    if retained_queries <= 0 or retained_queries > query_rows:
        raise ValueError("query-count is outside the available query range")

    setup_start = time.perf_counter()
    fp_map = np.memmap(args.fingerprints, mode="c", dtype="<u8", shape=(rows, 4))
    id_map = np.memmap(args.ids, mode="c", dtype="<i8", shape=(rows,))
    query_map = np.memmap(args.queries, mode="r", dtype="<u8", shape=(query_rows, 6))
    fp_u32 = fp_map.view("<u4").reshape(rows, 8)
    query_u32 = np.ascontiguousarray(query_map[:, 1:5]).view("<u4").reshape(query_rows, 8)

    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        database = torch.from_numpy(fp_u32).to(device=device, dtype=torch.uint32)
        stable_ids = torch.from_numpy(id_map).to(device=device, dtype=torch.int64)
    stream.synchronize()
    query_host = torch.from_numpy(query_u32).pin_memory()
    torch.cuda.reset_peak_memory_stats(device)
    setup_seconds = time.perf_counter() - setup_start

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def complete_request(query_index: int, numerator: int, denominator: int):
        effective_threshold = float(
            np.nextafter(
                np.float32(numerator / denominator), np.float32(-np.inf)
            )
        )
        wall_start = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            start_event.record(stream)
            query_device = query_host[query_index : query_index + 1].to(
                device=device, non_blocking=True
            )
            scores = crossTanimotoSimilarity(
                query_device, database, stream=stream
            ).torch()
            if scores.dtype != torch.float64 or tuple(scores.shape) != (1, rows):
                raise RuntimeError(
                    f"unexpected nvMolKit result dtype/shape {scores.dtype} {tuple(scores.shape)}"
                )
            selected = torch.nonzero(
                scores[0] >= effective_threshold, as_tuple=False
            ).flatten()
            result_device = stable_ids[selected]
            result_host = result_device.to(device="cpu", non_blocking=False)
            end_event.record(stream)
        end_event.synchronize()
        wall_seconds = (time.perf_counter_ns() - wall_start) / 1e9
        gpu_seconds = start_event.elapsed_time(end_event) / 1e3
        return result_host.numpy(), wall_seconds, gpu_seconds, effective_threshold

    for numerator, denominator in thresholds:
        for _ in range(args.warmup):
            complete_request(0, numerator, denominator)

    fields = (
        "request_order",
        "query",
        "threshold_num",
        "threshold_den",
        "effective_threshold",
        "effective_threshold_hex",
        "service_seconds",
        "gpu_seconds",
        "validation_seconds",
        "returned_count",
        "duplicate_id",
        "id_hash",
    )
    service_samples: dict[str, list[float]] = {
        f"{n}_{d}": [] for n, d in thresholds
    }
    request_order = 0
    run_start = time.perf_counter()
    with args.output.open("x", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        target.flush()
        for numerator, denominator in thresholds:
            for query_index in range(retained_queries):
                result_ids, service_seconds, gpu_seconds, effective_threshold = complete_request(
                    query_index, numerator, denominator
                )
                validation_start = time.perf_counter_ns()
                sorted_ids = np.sort(result_ids.astype(np.int64, copy=False))
                duplicate = bool(
                    sorted_ids.size > 1 and np.any(sorted_ids[1:] == sorted_ids[:-1])
                )
                identifier_hash = hash_ids(sorted_ids)
                validation_seconds = (time.perf_counter_ns() - validation_start) / 1e9
                observed = (int(sorted_ids.size), identifier_hash)
                key = (query_index, numerator, denominator)
                expected = oracle.get(key)
                row: dict[str, Any] = {
                    "request_order": request_order,
                    "query": query_index,
                    "threshold_num": numerator,
                    "threshold_den": denominator,
                    "effective_threshold": format(effective_threshold, ".17g"),
                    "effective_threshold_hex": effective_threshold.hex(),
                    "service_seconds": format(service_seconds, ".17g"),
                    "gpu_seconds": format(gpu_seconds, ".17g"),
                    "validation_seconds": format(validation_seconds, ".17g"),
                    "returned_count": sorted_ids.size,
                    "duplicate_id": int(duplicate),
                    "id_hash": identifier_hash,
                }
                writer.writerow(row)
                target.flush()
                if expected is None:
                    raise RuntimeError(f"oracle missing request {key}")
                if duplicate or observed != expected:
                    raise RuntimeError(
                        f"exact oracle mismatch {key}: observed={observed}, expected={expected}, duplicate={duplicate}"
                    )
                service_samples[f"{numerator}_{denominator}"].append(service_seconds)
                request_order += 1

    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    if peak_allocated > args.max_device_bytes:
        raise RuntimeError(
            f"device allocation limit exceeded: {peak_allocated} > {args.max_device_bytes}"
        )
    summary = {}
    for key, values in service_samples.items():
        summary[key] = {
            "count": len(values),
            "p10_ms": 1000 * percentile(values, 0.10),
            "median_ms": 1000 * percentile(values, 0.50),
            "p90_ms": 1000 * percentile(values, 0.90),
            "p95_ms": 1000 * percentile(values, 0.95),
            "min_ms": 1000 * min(values),
            "max_ms": 1000 * max(values),
        }
    metadata = {
        "format": "nvmolkit060_exact_complete_result_process_v1",
        "nvmolkit": nvmolkit_version,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "device_name": properties.name,
        "device_capability": [properties.major, properties.minor],
        "device_total_bytes": properties.total_memory,
        "rows": rows,
        "fingerprint_shape": [rows, 8],
        "fingerprint_dtype": "uint32",
        "score_dtype": "float64",
        "score_effective_precision": "float32 on the released tensor-op path",
        "threshold_rule": "one float32 nextafter step below the inclusive rational threshold, exhaustively validated over all nonzero 256-bit count ratios",
        "queries_available": query_rows,
        "queries_retained": retained_queries,
        "threshold_order": [[n, d] for n, d in thresholds],
        "warmup_per_threshold": args.warmup,
        "requests": request_order,
        "setup_seconds": setup_seconds,
        "retained_loop_seconds": time.perf_counter() - run_start,
        "peak_device_allocated_bytes": peak_allocated,
        "peak_device_reserved_bytes": peak_reserved,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "timing_scope": (
            "complete service wall: query H2D, released full double similarity matrix, "
            "GPU threshold/nonzero/stable-ID gather, result D2H, synchronization; "
            "setup and host sort/hash/oracle validation excluded"
        ),
        "exact_oracle_pass": True,
        "thresholds": summary,
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
