#!/usr/bin/env python3
"""Verify Gate-4 GPU threshold outputs against the frozen CPU lifecycle oracle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np


VARIANTS = ("released", "flat", "segmented")


def digest(rows: list[dict]) -> str:
    output = hashlib.sha256()
    for row in rows:
        output.update(
            struct.pack("<II", int(row["mol_id"]), int(row["coeff_bits"]))
        )
    return output.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--cpu-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    oracle = {}
    for line in args.cpu_jsonl.read_text().splitlines():
        row = json.loads(line)
        threshold = tuple(row["threshold_rational"])
        oracle[(row["query_index"], threshold)] = {
            "hash": row["hashes"]["union_oracle"],
            "count": row["counts"]["union_oracle"],
            "self": row["self_hits"]["union_oracle"],
        }

    grouped: dict[tuple[str, int, tuple[int, int]], list[dict]] = defaultdict(list)
    with args.gpu_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["variant"],
                int(row["query_index"]),
                (int(row["threshold_num"]), int(row["threshold_den"])),
            )
            grouped[key].append(row)

    expected_keys = {
        (variant, query_index, threshold)
        for variant in VARIANTS
        for query_index, threshold in oracle
    }
    missing = sorted(expected_keys - set(grouped))
    unexpected = sorted(set(grouped) - expected_keys)
    mismatches = []
    custom_contract_errors = []
    cross_variant_errors = []
    for key in sorted(expected_keys & set(grouped)):
        variant, query_index, threshold = key
        rows = grouped[key]
        ranks = [int(row["rank"]) for row in rows]
        ids = [int(row["mol_id"]) for row in rows]
        observed_hash = digest(rows)
        expected = oracle[(query_index, threshold)]
        reasons = []
        if ranks != list(range(len(rows))):
            reasons.append("noncontiguous_rank")
        if len(ids) != len(set(ids)):
            reasons.append("duplicate_id")
        if len(rows) != expected["count"]:
            reasons.append(f"count:{len(rows)}!={expected['count']}")
        if observed_hash != expected["hash"]:
            reasons.append("hash")
        if reasons:
            mismatches.append(
                {
                    "variant": variant,
                    "query_index": query_index,
                    "threshold": threshold,
                    "reasons": reasons,
                    "observed_hash": observed_hash,
                    "expected_hash": expected["hash"],
                }
            )

        if variant in ("flat", "segmented"):
            threshold_num, threshold_den = threshold
            for rank, row in enumerate(rows):
                num = int(row["num"])
                den = int(row["den"])
                if not (0 <= num <= 256 and 1 <= den <= 512):
                    custom_contract_errors.append((key, rank, "rational_range"))
                    continue
                if num * threshold_den < den * threshold_num:
                    custom_contract_errors.append((key, rank, "threshold"))
                expected_bits = struct.unpack(
                    "<I", struct.pack("<f", np.float32(num / den))
                )[0]
                if expected_bits != int(row["coeff_bits"]):
                    custom_contract_errors.append((key, rank, "coeff_bits"))

    for query_index, threshold in sorted(oracle):
        hashes = {
            variant: digest(grouped[(variant, query_index, threshold)])
            for variant in VARIANTS
            if (variant, query_index, threshold) in grouped
        }
        if len(set(hashes.values())) > 1:
            cross_variant_errors.append(
                {
                    "query_index": query_index,
                    "threshold": threshold,
                    "hashes": hashes,
                }
            )

    summary = {
        "experiment_id": "tide_20260827_gate4_gpu_exactness",
        "expected_request_count": len(expected_keys),
        "observed_request_count": len(grouped),
        "output_row_count": sum(len(rows) for rows in grouped.values()),
        "missing_request_count": len(missing),
        "unexpected_request_count": len(unexpected),
        "oracle_mismatch_count": len(mismatches),
        "custom_contract_error_count": len(custom_contract_errors),
        "cross_variant_mismatch_count": len(cross_variant_errors),
        "gates": {
            "G4_B_RELEASED": not (
                missing
                or unexpected
                or mismatches
                or custom_contract_errors
                or cross_variant_errors
            )
        },
        "first_failures": {
            "missing": missing[:3],
            "unexpected": unexpected[:3],
            "oracle": mismatches[:3],
            "custom_contract": custom_contract_errors[:3],
            "cross_variant": cross_variant_errors[:3],
        },
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["gates"]["G4_B_RELEASED"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
