#!/usr/bin/env python3
"""Inspect production candidates around one rational nvMolKit boundary."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
from nvmolkit.similarity import crossTanimotoSimilarity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprints", type=pathlib.Path, required=True)
    parser.add_argument("--ids", type=pathlib.Path, required=True)
    parser.add_argument("--queries", type=pathlib.Path, required=True)
    parser.add_argument("--query", type=int, required=True)
    parser.add_argument("--numerator", type=int, required=True)
    parser.add_argument("--denominator", type=int, required=True)
    parser.add_argument("--band", type=float, default=0.0002)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")
    threshold = args.numerator / args.denominator
    threshold_f32 = np.float32(threshold)
    threshold_f32_down = np.nextafter(threshold_f32, np.float32(-np.inf))

    rows = args.fingerprints.stat().st_size // 32
    fp_map = np.memmap(args.fingerprints, mode="c", dtype="<u8", shape=(rows, 4))
    id_map = np.memmap(args.ids, mode="r", dtype="<i8", shape=(rows,))
    query_rows = args.queries.stat().st_size // 48
    query_map = np.memmap(args.queries, mode="r", dtype="<u8", shape=(query_rows, 6))
    query_words = np.asarray(query_map[args.query, 1:5], dtype="<u8")
    query_u32 = query_words.view("<u4").reshape(1, 8).copy()
    database_u32 = fp_map.view("<u4").reshape(rows, 8)

    setup_start = time.perf_counter()
    database = torch.from_numpy(database_u32).to("cuda", dtype=torch.uint32)
    query = torch.from_numpy(query_u32).to("cuda", dtype=torch.uint32)
    torch.cuda.synchronize()
    scores = crossTanimotoSimilarity(query, database).torch()[0]
    torch.cuda.synchronize()
    counts = {
        "python_double": int(torch.count_nonzero(scores >= threshold).item()),
        "float32_canonical": int(torch.count_nonzero(scores >= float(threshold_f32)).item()),
        "float32_nextafter_down": int(
            torch.count_nonzero(scores >= float(threshold_f32_down)).item()
        ),
    }
    near_indices = torch.nonzero(
        torch.abs(scores - float(threshold_f32)) <= args.band, as_tuple=False
    ).flatten()
    near_scores = scores[near_indices].cpu().numpy()
    near_rows = near_indices.cpu().numpy().astype(np.int64, copy=False)
    torch.cuda.synchronize()

    query_tuple = tuple(int(x) for x in query_words)
    query_count = sum(x.bit_count() for x in query_tuple)
    records = []
    for row, score in zip(near_rows, near_scores):
        candidate = tuple(int(x) for x in fp_map[int(row)])
        intersection = sum(
            (left & right).bit_count() for left, right in zip(query_tuple, candidate)
        )
        candidate_count = sum(x.bit_count() for x in candidate)
        union = query_count + candidate_count - intersection
        exact_pass = union != 0 and args.denominator * intersection >= args.numerator * union
        records.append(
            {
                "row": int(row),
                "id": int(id_map[int(row)]),
                "intersection": intersection,
                "union": union,
                "exact_ratio": None if union == 0 else intersection / union,
                "exact_pass": exact_pass,
                "score": float(score),
                "score_hex": float(score).hex(),
                "pass_python_double": bool(score >= threshold),
                "pass_float32_canonical": bool(score >= threshold_f32),
                "pass_float32_nextafter_down": bool(score >= threshold_f32_down),
            }
        )
    result = {
        "format": "nvmolkit060_full_boundary_diagnostic_v1",
        "rows": rows,
        "query": args.query,
        "numerator": args.numerator,
        "denominator": args.denominator,
        "threshold_python": threshold,
        "threshold_python_hex": float(threshold).hex(),
        "threshold_f32": float(threshold_f32),
        "threshold_f32_hex": float(threshold_f32).hex(),
        "threshold_f32_down": float(threshold_f32_down),
        "threshold_f32_down_hex": float(threshold_f32_down).hex(),
        "band": args.band,
        "counts": counts,
        "near_count": len(records),
        "near_records": records,
        "setup_and_diagnostic_seconds": time.perf_counter() - setup_start,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
