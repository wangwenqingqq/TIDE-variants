#!/usr/bin/env python3
"""Verify chemfp shardsearch TSV output against the frozen integer oracle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def id_hash(ids: list[int]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(ids)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def read_tsv(path: Path) -> tuple[dict[int, list[int]], list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader((line for line in stream if not line.startswith("#")), delimiter="\t")
        rows = list(reader)
    hits: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        query_id = int(row["query_id"])
        if row["target_id"] != "*":
            hits[query_id].append(int(row["target_id"]))
    return hits, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", choices=("base_only", "base_plus_delta", "synthetic"), required=True)
    parser.add_argument("--threshold", choices=("7/10", "4/5"), required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    expected_all = manifest["oracle"][args.case]
    expected = {
        int(record["query_id"]): record
        for record in expected_all.values()
        if record["threshold"] == args.threshold
    }
    actual, raw_rows = read_tsv(args.result)
    records = []
    mismatches = []
    for query_id, oracle in sorted(expected.items()):
        ids = actual.get(query_id, [])
        duplicate_count = len(ids) - len(set(ids))
        record = {
            "query_id": query_id,
            "expected_hit_count": oracle["hit_count"],
            "actual_hit_count": len(ids),
            "expected_sorted_id_sha256": oracle["sorted_id_sha256"],
            "actual_sorted_id_sha256": id_hash(ids),
            "duplicate_count": duplicate_count,
            "query_id_present": query_id in ids,
        }
        record["exact"] = (
            record["expected_hit_count"] == record["actual_hit_count"]
            and record["expected_sorted_id_sha256"] == record["actual_sorted_id_sha256"]
            and duplicate_count == 0
        )
        records.append(record)
        if not record["exact"]:
            mismatches.append(record)
    unexpected_queries = sorted(set(actual) - set(expected))
    result = {
        "case": args.case,
        "threshold": args.threshold,
        "result_path": str(args.result),
        "raw_rows": len(raw_rows),
        "queries": len(expected),
        "all_exact": not mismatches and not unexpected_queries,
        "mismatches": mismatches,
        "unexpected_queries": unexpected_queries,
        "records": records,
    }
    if args.output.exists():
        raise RuntimeError(f"refusing to replace output: {args.output}")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["all_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
