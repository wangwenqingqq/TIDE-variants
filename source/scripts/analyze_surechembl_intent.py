#!/usr/bin/env python3
"""Measure Page2Tile routing opportunity on the frozen SureChEMBL proxy trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from page2tile.tide_intent import summarize_windows  # noqa: E402

EXPECTED_SHA256: Dict[str, str] = {
    "data_manifest": "bc8bf98e30e7b4d43e2a8d3a91d42917a9b3a0af34e53a1e38860d0e9fa9abaa",
    "query_manifest": "7b19a46a5b558bb944acbb0269fb5576ffc676a315ef890c5a4f0705583423fd",
    "union_popcnt": "943e243f30426f5b79048605ab56dbf58d9b98880414fefc31b01577844433b1",
    "queries_popcnt": "59f35ef054ba629f15b8a69411423167e260ea7b873c3c0b2b8ba51119a24918",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--union-popcnt", type=Path, required=True)
    parser.add_argument("--queries-popcnt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "data_manifest": args.data_manifest,
        "query_manifest": args.query_manifest,
        "union_popcnt": args.union_popcnt,
        "queries_popcnt": args.queries_popcnt,
    }
    receipts = {name: sha256(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": EXPECTED_SHA256[name], "observed": receipts[name]}
        for name in paths
        if receipts[name] != EXPECTED_SHA256[name]
    }
    if mismatches:
        raise RuntimeError(
            "input checksum mismatch: %s" % json.dumps(mismatches, sort_keys=True)
        )

    data_manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    query_manifest = json.loads(args.query_manifest.read_text(encoding="utf-8"))
    expected_rows = int(data_manifest["counts"]["union"])
    expected_queries = int(data_manifest["counts"]["calibration_queries"]) + int(
        data_manifest["counts"]["test_queries"]
    )

    union_popcounts = np.memmap(args.union_popcnt, mode="r", dtype="<u2")
    query_popcounts = np.fromfile(args.queries_popcnt, dtype="<u2")
    if len(union_popcounts) != expected_rows:
        raise RuntimeError("union row count mismatch")
    if len(query_manifest) != expected_queries or len(query_popcounts) != expected_queries:
        raise RuntimeError("query count mismatch")
    manifest_popcounts = np.asarray(
        [int(row["popcnt"]) for row in query_manifest], dtype=np.uint16
    )
    if not np.array_equal(query_popcounts, manifest_popcounts):
        raise RuntimeError("binary and JSON query popcounts differ")
    if np.any(union_popcounts[1:] < union_popcounts[:-1]):
        raise RuntimeError("union population counts are not monotone")

    bin_rows = np.bincount(union_popcounts, minlength=257).astype(np.int64).tolist()
    test_popcounts = [
        int(row["popcnt"]) for row in query_manifest if row.get("split") == "test"
    ]
    if len(test_popcounts) != int(data_manifest["counts"]["test_queries"]):
        raise RuntimeError("test split count mismatch")

    window_sizes = [1, 8, 16, 32, 64, 128, 256, 512, 1024]
    thresholds = {"0.70": (7, 10), "0.80": (4, 5)}
    threshold_results = {
        label: {
            "threshold_rational": [numerator, denominator],
            "window_results": [
                summarize_windows(
                    test_popcounts,
                    bin_rows,
                    numerator,
                    denominator,
                    window_size,
                )
                for window_size in window_sizes
            ],
        }
        for label, (numerator, denominator) in thresholds.items()
    }

    screen = {}
    for label, result in threshold_results.items():
        admitted = []
        for window in result["window_results"]:
            if window["window_size"] <= 128:
                fraction = window["aggregate"]["pair_fraction_by_min_fanin"]["8"]
                if fraction >= 0.30:
                    admitted.append(
                        {
                            "window_size": window["window_size"],
                            "pair_fraction_fanin_ge_8": fraction,
                        }
                    )
        screen[label] = {
            "passed": bool(admitted),
            "first_passing_window": admitted[0] if admitted else None,
        }

    payload = {
        "schema_version": "page2tile-surechembl-intent-v1",
        "experiment_id": "page2tile_20260829_surechembl_popcount_intent_gate0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "inputs": {
            name: {"path": str(path), "sha256": receipts[name]}
            for name, path in paths.items()
        },
        "dataset": {
            "union_rows": expected_rows,
            "fingerprint_bits": 256,
            "test_queries": len(test_popcounts),
            "nonempty_popcount_bins": sum(1 for count in bin_rows if count),
            "fingerprint_payload_bytes": expected_rows * 32,
            "resident_fp_id_popcnt_bytes": expected_rows * 42,
            "query_order": "query_manifest test rows in original manifest order",
            "query_provenance": (
                "deterministic future-arrival proxy; not production traffic"
            ),
        },
        "thresholds": threshold_results,
        "opportunity_screen": {
            "criterion": (
                "both thresholds have >=0.30 candidate-pair fraction at "
                "fan-in >=8 for some window <=128"
            ),
            "per_threshold": screen,
            "passed": all(item["passed"] for item in screen.values()),
            "claim_boundary": (
                "admits only a same-data B1 batching experiment; "
                "not Page2Tile M1/M2/M3"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
