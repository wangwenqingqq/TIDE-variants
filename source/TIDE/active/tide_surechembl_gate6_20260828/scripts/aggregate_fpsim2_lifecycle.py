#!/usr/bin/env python3
"""Aggregate the frozen six-transition correctness-restoring lifecycle trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


TRANSITIONS = [
    "2026-06-01_to_2026-06-15",
    "2026-06-15_to_2026-07-01",
    "2026-07-01_to_2026-07-17",
    "2026-07-17_to_2026-08-04",
    "2026-08-04_to_2026-08-18",
    "2026-08-18_to_2026-08-25",
]
STAGES = [
    "append_precomputed_fingerprints_s",
    "complete_popcount_csi_creation_s",
    "global_sort_and_bin_repair_s",
    "actual_engine_reload_s",
    "lifecycle_total_s",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    input_hashes = {}
    for transition in TRANSITIONS:
        path = args.trace_dir / f"{transition}.json"
        if not path.is_file():
            raise RuntimeError(f"missing formal transition result: {path}")
        input_hashes[str(path)] = sha256(path)
        record = json.loads(path.read_text())
        if record.get("transition") != transition:
            raise RuntimeError(f"transition label mismatch in {path}")
        if not record.get("query_ready_component_pass"):
            raise RuntimeError(f"query-ready component failed in {path}")
        if not record.get("population_count_sorted_after"):
            raise RuntimeError(f"population-count order failed in {path}")
        if not record.get("pre_sort_index", {}).get("is_csi"):
            raise RuntimeError(f"pre-sort index was not CSI in {path}")
        if int(record["rows_after"]) != int(record["union_rows"]):
            raise RuntimeError(f"row balance failed in {path}")
        if int(record["last_bin_end"]) != int(record["union_rows"]):
            raise RuntimeError(f"bin coverage failed in {path}")
        for stage in STAGES:
            if stage not in record["timed"]:
                raise RuntimeError(f"missing stage {stage} in {path}")
        records.append(record)

    stage_summary = {}
    for stage in STAGES:
        values = [float(record["timed"][stage]) for record in records]
        stage_summary[stage] = {
            "values_by_transition": dict(zip(TRANSITIONS, values)),
            "minimum": min(values),
            "median": statistics.median(values),
            "maximum": max(values),
        }
    median_total = stage_summary["lifecycle_total_s"]["median"]
    result = {
        "experiment_id": "tide_20260828_gate6_fpsim2_lifecycle_trace02",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": (
            "actual FPSim2 0.7.4 sort/bin repair and engine reload after append "
            "and required complete PyTables popcount CSI; post-fingerprint local "
            "artifact scope; source copy excluded"
        ),
        "transition_count": len(records),
        "all_query_ready_component_pass": True,
        "stage_summary_s": stage_summary,
        "median_lifecycle_s": median_total,
        "G6_PROBLEM_LIFECYCLE_COMPONENT": bool(median_total >= 10.0),
        "complete_result_equality_gate": "pending exact query campaign",
        "input_sha256": input_hashes,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
