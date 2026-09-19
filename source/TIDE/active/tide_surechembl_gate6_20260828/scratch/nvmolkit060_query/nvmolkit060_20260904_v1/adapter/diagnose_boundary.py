#!/usr/bin/env python3
"""Localize nvMolKit rational-boundary differences on the frozen fixture."""

from __future__ import annotations

import argparse
import json
import pathlib
import struct

import numpy as np
import torch
from nvmolkit.similarity import crossTanimotoSimilarity


def popcount(words: tuple[int, ...]) -> int:
    return sum(word.bit_count() for word in words)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")

    fp_raw = args.fixture.joinpath("fp_u64x4.bin").read_bytes()
    id_raw = args.fixture.joinpath("id_i64.bin").read_bytes()
    query_raw = args.fixture.joinpath("queries_u64x6.bin").read_bytes()
    fingerprints = [
        struct.unpack_from("<4Q", fp_raw, offset)
        for offset in range(0, len(fp_raw), 32)
    ]
    identifiers = [
        struct.unpack_from("<q", id_raw, offset)[0]
        for offset in range(0, len(id_raw), 8)
    ]
    query_records = [
        struct.unpack_from("<6Q", query_raw, offset)
        for offset in range(0, len(query_raw), 48)
    ]
    database_np = np.frombuffer(fp_raw, dtype="<u4").reshape(-1, 8).copy()
    database = torch.from_numpy(database_np).to("cuda", dtype=torch.uint32)

    records = []
    for query_index, query_record in enumerate(query_records):
        query_words = tuple(query_record[1:5])
        query_np = (
            np.asarray(query_words, dtype="<u8").view("<u4").reshape(1, 8).copy()
        )
        query = torch.from_numpy(query_np).to("cuda", dtype=torch.uint32)
        scores = crossTanimotoSimilarity(query, database).torch()
        torch.cuda.synchronize()
        values = scores.cpu().numpy()[0]
        for row, (identifier, candidate, score) in enumerate(
            zip(identifiers, fingerprints, values)
        ):
            intersection = sum(
                (left & right).bit_count()
                for left, right in zip(query_words, candidate)
            )
            union = popcount(query_words) + popcount(candidate) - intersection
            exact_ratio = None if union == 0 else intersection / union
            records.append(
                {
                    "query": query_index,
                    "row": row,
                    "id": identifier,
                    "intersection": intersection,
                    "union": union,
                    "exact_ratio_python": exact_ratio,
                    "nvmolkit_score": float(score),
                    "nvmolkit_score_hex": float(score).hex(),
                    "error": None if exact_ratio is None else float(score) - exact_ratio,
                    "ge_7_10": bool(score >= (7 / 10)),
                    "ge_4_5": bool(score >= (4 / 5)),
                }
            )
    result = {
        "format": "nvmolkit060_boundary_diagnostic_v1",
        "score_dtype": str(scores.dtype),
        "threshold_7_10": {"value": 7 / 10, "hex": float(7 / 10).hex()},
        "threshold_4_5": {"value": 4 / 5, "hex": float(4 / 5).hex()},
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
