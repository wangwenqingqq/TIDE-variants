#!/usr/bin/env python3
"""Aggregate the frozen six-transition Gate-6 update campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/update/formal_surechembl_v1"
RAW = ROOT / "results/raw/20260829T050000Z_gate6_update_formal_v1"
LIFECYCLE = ROOT / "results/FPSIM2_LIFECYCLE_SUMMARY.json"
TRANSITIONS = (
    "2026-06-01_to_2026-06-15",
    "2026-06-15_to_2026-07-01",
    "2026-07-01_to_2026-07-17",
    "2026-07-17_to_2026-08-04",
    "2026-08-04_to_2026-08-18",
    "2026-08-18_to_2026-08-25",
)
STAGES = (
    "artifact_map_ms",
    "host_prepare_ms",
    "device_run_ms",
    "snapshot_publish_ms",
    "first_query_visible_ms",
    "update_to_query_visible_ms",
    "kernel_ms",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Iterable[float], q: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("empty percentile input")
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def stats(values: list[float]) -> dict[str, float]:
    return {
        "p10": percentile(values, 0.10),
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "maximum": max(values),
        "mean": sum(values) / len(values),
    }


def main() -> None:
    lifecycle = json.loads(LIFECYCLE.read_text())
    lifecycle_by_transition = lifecycle["stage_summary_s"]["lifecycle_total_s"]["values_by_transition"]
    records: dict[str, object] = {}
    total_retained = 0
    total_mismatches = 0
    all_processes_clean = True
    all_visibility_exact = True
    ratios = []
    p95_values = []

    for transition in TRANSITIONS:
        csv_path = OUT / f"{transition}.csv"
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        setup = json.loads(Path(f"{csv_path}.setup.json").read_text())
        validation = json.loads(Path(f"{csv_path}.validation.json").read_text())
        sample_ids = [int(row["sample"]) for row in rows]
        exact = bool(
            len(rows) == 512
            and sample_ids == list(range(512))
            and all(
                row["overflow"] == "0"
                and row["contains_added_id"] == "1"
                and row["candidate_match"] == "1"
                and row["match"] == "1"
                for row in rows
            )
            and validation["pass"]
            and validation["retained_mismatches"] == 0
            and validation["post_4_5_match"]
            and not validation["base_contains_added_id"]
        )
        process_clean = bool(
            int((RAW / f"{transition}.exit_code.txt").read_text().strip()) == 0
            and (RAW / f"{transition}.stderr.log").stat().st_size == 0
            and (RAW / f"{transition}.compute_before.csv").stat().st_size == 0
            and (RAW / f"{transition}.compute_after.csv").stat().st_size == 0
        )
        stage_stats = {
            stage: stats([float(row[stage]) for row in rows]) for stage in STAGES
        }
        stage_sum_gaps = []
        for row in rows:
            component_sum = sum(
                float(row[stage])
                for stage in (
                    "artifact_map_ms",
                    "host_prepare_ms",
                    "device_run_ms",
                    "snapshot_publish_ms",
                    "first_query_visible_ms",
                )
            )
            stage_sum_gaps.append(float(row["update_to_query_visible_ms"]) - component_sum)
        fpsim_s = float(lifecycle_by_transition[transition])
        update_p95_ms = stage_stats["update_to_query_visible_ms"]["p95"]
        ratio = update_p95_ms / 1000.0 / fpsim_s
        gate_pass = bool(exact and process_clean and ratio <= 0.01)
        ratios.append(ratio)
        p95_values.append(update_p95_ms)
        total_retained += len(rows)
        total_mismatches += sum(row["match"] != "1" for row in rows)
        all_processes_clean &= process_clean
        all_visibility_exact &= exact
        records[transition] = {
            "retained_requests": len(rows),
            "process_clean": process_clean,
            "visibility_and_results_exact": exact,
            "setup": setup,
            "validation": validation,
            "constant_observables": {
                "candidate_rows": int(rows[0]["candidate_rows"]),
                "hits": int(rows[0]["hits"]),
                "result_hash": int(rows[0]["result_hash"]),
            },
            "stage_distributions_ms": stage_stats,
            "stage_sum_gap_ms": stats(stage_sum_gaps),
            "fpsim2_lifecycle_s": fpsim_s,
            "update_p95_over_fpsim2_lifecycle": ratio,
            "update_p95_percent_of_fpsim2_lifecycle": ratio * 100.0,
            "gate_ratio_limit": 0.01,
            "pass": gate_pass,
            "csv_sha256": sha256(csv_path),
        }

    all_pass = bool(all(record["pass"] for record in records.values()))
    max_ratio_transition = max(TRANSITIONS, key=lambda t: records[t]["update_p95_over_fpsim2_lifecycle"])
    max_p95_transition = max(TRANSITIONS, key=lambda t: records[t]["stage_distributions_ms"]["update_to_query_visible_ms"]["p95"])
    summary = {
        "experiment_id": "tide_20260829_gate6_update_formal_surechembl_v1",
        "evidence_state": "measured",
        "scope": "six eligible insert-only SureChEMBL transitions; persistent base; warm local precomputed delta artifact map/decode/H2D/publish plus first exact complete 7/10 query; physical GPU2",
        "excluded_scope": "process startup, persistent base mapping/validation/H2D, network, HDF5 parsing, fingerprint generation, page-cache flush, and harness reset/destruction",
        "frozen_contract": "UPDATE_EXECUTION_CARD_20260829.md",
        "source_sha256": "cbdd124c669b2b8c9b680b7357095e2611bb4418aeaea41bb5cd9256789a8a3a",
        "binary_sha256": "99864430bbd97fa9f38eea1b67f5f0f5087e7475a18a668bd695be57641f30b8",
        "selected_words4_instruction_body_sha256": "92027514dfbd14260b084e919d1d16a2237f9169b0afab97b0ec144cfd8ff264",
        "selected_instruction_body_identical_to_keeper": True,
        "selected_resources": {"registers": 34, "stack_bytes": 0, "shared_bytes": 0, "local_bytes": 0, "constant_bytes": 952},
        "fixture_validation": {
            "functional_exact": True,
            "memcheck_errors": 0,
            "memcheck_leaks": 0,
            "synccheck_errors": 0,
            "raw": "results/raw/20260829T044200Z_gate6_update_fixture_validation_v1",
        },
        "warmups_per_transition": 16,
        "retained_requests_per_transition": 512,
        "total_retained_requests": total_retained,
        "total_retained_mismatches": total_mismatches,
        "all_processes_clean": all_processes_clean,
        "all_visibility_and_results_exact": all_visibility_exact,
        "records": records,
        "aggregate": {
            "update_p95_ms_range": {"minimum": min(p95_values), "maximum": max(p95_values)},
            "median_transition_p95_ms": percentile(p95_values, 0.50),
            "maximum_ratio": max(ratios),
            "maximum_ratio_percent": max(ratios) * 100.0,
            "maximum_ratio_transition": max_ratio_transition,
            "maximum_absolute_p95_transition": max_p95_transition,
            "all_six_ratio_at_most_0_01": all(ratio <= 0.01 for ratio in ratios),
        },
        "G6_UPDATE": all_pass,
        "decision": "G6_UPDATE_PASS" if all_pass else "G6_UPDATE_FAIL",
        "wording_guard": "The comparator ratio is a cross-system post-fingerprint local-artifact-to-query-visible ratio, not whole-pipeline speedup.",
    }
    summary_path = OUT / "FORMAL_UPDATE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    raw_pointer = {
        "primary_summary": str(summary_path.relative_to(ROOT)),
        "primary_summary_sha256": sha256(summary_path),
        "decision": summary["decision"],
    }
    (RAW / "FORMAL_AGGREGATE.json").write_text(json.dumps(raw_pointer, indent=2) + "\n")
    print(json.dumps({
        "all_processes_clean": all_processes_clean,
        "all_visibility_and_results_exact": all_visibility_exact,
        "total_retained_requests": total_retained,
        "p95_ms_range": summary["aggregate"]["update_p95_ms_range"],
        "maximum_ratio": summary["aggregate"]["maximum_ratio"],
        "maximum_ratio_percent": summary["aggregate"]["maximum_ratio_percent"],
        "maximum_ratio_transition": max_ratio_transition,
        "G6_UPDATE": all_pass,
        "summary": str(summary_path),
    }, indent=2))


if __name__ == "__main__":
    main()
