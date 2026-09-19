#!/usr/bin/env python3
"""Create a deterministic GPUSimilarity exact-result fixture."""

from __future__ import annotations

import argparse
import csv
import pathlib
import struct


THRESHOLDS = ((7, 10), (4, 5))


def fnv_ids(ids: list[int]) -> int:
    value = 1469598103934665603
    for identifier in sorted(ids):
        for byte in struct.pack("<Q", identifier):
            value ^= byte
            value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def popcount(words: tuple[int, int, int, int]) -> int:
    return sum(bin(word).count("1") for word in words)


def exact_match(
    query: tuple[int, int, int, int],
    candidate: tuple[int, int, int, int],
    numerator: int,
    denominator: int,
) -> bool:
    intersection = sum(
        bin(left & right).count("1") for left, right in zip(query, candidate)
    )
    union = popcount(query) + popcount(candidate) - intersection
    return union != 0 and denominator * intersection >= numerator * union


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    fingerprints = [
        (0x3FF, 0, 0, 0),       # query 0, ten bits
        (0x07F, 0, 0, 0),       # exact 0.7 boundary
        (0x0FF, 0, 0, 0),       # exact 0.8 boundary
        (0x7FF, 0, 0, 0),       # 10/11
        (0x003, 0, 0, 0),       # below both thresholds
        (0, 0xFFFF, 0, 0),      # disjoint
        (0, 0x0FFF, 0, 0),      # query 1, twelve bits
        (0, 0x00FF, 0, 0),      # 8/12, below both
        (0, 0x03FF, 0, 0),      # 10/12, above both
    ]
    ids = [101, 55, 7003, 42, 999, 1234567, 88, 89, 90]
    queries = [(101, fingerprints[0]), (88, fingerprints[6])]

    with (output / "fp_u64x4.bin").open("wb") as target:
        for fingerprint in fingerprints:
            target.write(struct.pack("<4Q", *fingerprint))
    with (output / "id_i64.bin").open("wb") as target:
        for identifier in ids:
            target.write(struct.pack("<Q", identifier))
    with (output / "queries_u64x6.bin").open("wb") as target:
        for identifier, fingerprint in queries:
            target.write(struct.pack("<6Q", identifier, *fingerprint, popcount(fingerprint)))

    with (output / "oracle.csv").open("w", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "query",
                "threshold_num",
                "threshold_den",
                "hits",
                "id_hash",
                "duplicate_id",
            ),
        )
        writer.writeheader()
        for query_index, (_, query) in enumerate(queries):
            for numerator, denominator in THRESHOLDS:
                hits = [
                    identifier
                    for identifier, fingerprint in zip(ids, fingerprints)
                    if exact_match(query, fingerprint, numerator, denominator)
                ]
                writer.writerow(
                    {
                        "query": query_index,
                        "threshold_num": numerator,
                        "threshold_den": denominator,
                        "hits": len(hits),
                        "id_hash": fnv_ids(hits),
                        "duplicate_id": 0,
                    }
                )
    print(f"rows={len(fingerprints)} queries={len(queries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
