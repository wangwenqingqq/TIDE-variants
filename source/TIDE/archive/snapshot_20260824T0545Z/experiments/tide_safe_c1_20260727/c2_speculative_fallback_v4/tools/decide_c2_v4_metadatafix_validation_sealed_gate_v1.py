#!/usr/bin/env python3
"""One-shot CPU-only sealed-test decision from a frozen validation summary."""
from __future__ import annotations
import hashlib
import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
VALIDATION_PROTOCOL = ROOT / "protocols/c2_v4_metadatafix_validation_execution_protocol_v1.json"
DECISION_PROTOCOL = ROOT / "protocols/c2_v4_metadatafix_validation_sealed_decision_protocol_v1.json"
OUT = ROOT / "formal_runs_v2/c2_v4_validation_v1"
RECEIPT = ROOT / "provenance/c2_v4_metadatafix_validation_sealed_test_decision_v1.json"

EXPECTED_FORMULA = {
    "numerator": "paired_same_split_baseline.guarded_timing.query_pipeline_wall_ms_sum",
    "denominator": "timing.stage.guarded_timing.query_pipeline_wall_ms_sum",
    "aggregation": "aggregate sum over identical validation queries and timed ABBA repetitions",
    "raw_speculative_timing_forbidden": True,
    "requires": {
        "query_count": 2463,
        "timed_reps": 7,
        "shared_warmup_pairs": 1,
        "final_guarded_gate_pass": True,
        "denominator_positive_finite": True,
    },
    "pass_condition": "speedup >= 1.05",
    "on_failure": "sealed test remains locked; no retry",
}


def fail(message: str) -> None:
    raise RuntimeError("SEALED_TEST_DECISION_FAIL: " + message)


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path: pathlib.Path) -> dict:
    if not path.is_file() or path.is_symlink():
        fail("not direct file " + str(path))
    value = json.load(path.open(encoding="utf-8"))
    if not isinstance(value, dict):
        fail("not JSON object " + str(path))
    return value


def positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail("non-numeric " + label)
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        fail("non-positive/nonfinite " + label)
    return result


def evaluate(summary: dict) -> tuple[float, float, float]:
    if (
        summary.get("schema") != "safe-c2-speculative-fallback-v4-run-v1"
        or summary.get("status") != "PASS_V4_HELDOUT_GUARDED"
        or summary.get("mode") != "evaluate"
        or summary.get("stage") != "validation"
    ):
        fail("summary identity")
    expected_order = [
        "baseline_then_speculative_guarded" if i % 2 == 0
        else "speculative_guarded_then_baseline"
        for i in range(7)
    ]
    comparison = summary.get("timing_comparison", {})
    if (
        comparison.get("eligible_for_speed_comparison") is not True
        or comparison.get("shared_warmup_pairs") != 1
        or comparison.get("timed_pair_order") != expected_order
    ):
        fail("ABBA contract")
    gate = summary.get("final_guarded_per_query_no_regression_gate", {})
    if (
        gate.get("passed") is not True
        or gate.get("query_count") != 2463
        or gate.get("repetitions_checked") != 7
        or gate.get("max_violating_queries") != 0
    ):
        fail("final guarded gate")
    baseline = summary.get("paired_same_split_baseline", {})
    guarded = summary.get("timing", {}).get("stage", {})
    if (
        baseline.get("query_count") != 2463
        or baseline.get("timed_reps") != 7
        or guarded.get("query_count") != 2463
        or guarded.get("timed_reps") != 7
    ):
        fail("paired sample sizes")
    numerator = positive(
        baseline.get("guarded_timing", {}).get("query_pipeline_wall_ms_sum"),
        "baseline final-guarded pipeline sum",
    )
    denominator = positive(
        guarded.get("guarded_timing", {}).get("query_pipeline_wall_ms_sum"),
        "guarded final-guarded pipeline sum",
    )
    return numerator, denominator, numerator / denominator


def self_test() -> None:
    summary = {
        "schema": "safe-c2-speculative-fallback-v4-run-v1",
        "status": "PASS_V4_HELDOUT_GUARDED",
        "mode": "evaluate",
        "stage": "validation",
        "timing_comparison": {
            "eligible_for_speed_comparison": True,
            "shared_warmup_pairs": 1,
            "timed_pair_order": [
                "baseline_then_speculative_guarded",
                "speculative_guarded_then_baseline",
                "baseline_then_speculative_guarded",
                "speculative_guarded_then_baseline",
                "baseline_then_speculative_guarded",
                "speculative_guarded_then_baseline",
                "baseline_then_speculative_guarded",
            ],
        },
        "final_guarded_per_query_no_regression_gate": {
            "passed": True,
            "query_count": 2463,
            "repetitions_checked": 7,
            "max_violating_queries": 0,
        },
        "paired_same_split_baseline": {
            "query_count": 2463,
            "timed_reps": 7,
            "guarded_timing": {"query_pipeline_wall_ms_sum": 105.0},
        },
        "timing": {
            "stage": {
                "query_count": 2463,
                "timed_reps": 7,
                "guarded_timing": {"query_pipeline_wall_ms_sum": 100.0},
            }
        },
    }
    numerator, denominator, speedup = evaluate(summary)
    if (numerator, denominator, round(speedup, 2)) != (105.0, 100.0, 1.05):
        fail("self test")
    print(json.dumps({"status": "PASS_SEALED_DECISION_TOOL_SELF_TEST", "raw_speculative_metric_used": False}, sort_keys=True))


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    if len(sys.argv) != 1:
        fail("no arguments accepted")
    if RECEIPT.exists() or RECEIPT.is_symlink():
        fail("decision already exists; no retry")

    validation_protocol = load(VALIDATION_PROTOCOL)
    decision_protocol = load(DECISION_PROTOCOL)
    if validation_protocol.get("sealed_test_decision_protocol") != {
        "path": str(DECISION_PROTOCOL),
        "sha256": sha(DECISION_PROTOCOL),
    }:
        fail("validation protocol decision binding")
    if (
        decision_protocol.get("schema")
        != "safe-c2-v4-metadatafix-validation-sealed-decision-protocol-v1"
        or decision_protocol.get("status")
        != "FROZEN_BEFORE_METADATAFIX_VALIDATION_GPU_EXECUTION"
    ):
        fail("decision protocol identity")
    if decision_protocol.get("tool") != {
        "path": str(pathlib.Path(__file__)),
        "sha256": sha(pathlib.Path(__file__)),
    }:
        fail("tool binding")
    if (
        decision_protocol.get("validation_output_root") != str(OUT)
        or decision_protocol.get("receipt_path") != str(RECEIPT)
        or decision_protocol.get("formula") != EXPECTED_FORMULA
    ):
        fail("decision protocol content")

    state = load(OUT / "run_state.json")
    if (
        state.get("status")
        != "COMPLETE_METADATAFIX_VALIDATION_PASS_PENDING_SEALED_DECISION"
        or state.get("sealed_test_unlocked") is not False
    ):
        fail("validation state")
    summary_path = OUT / "summary.json"
    summary = load(summary_path)
    numerator, denominator, speedup = evaluate(summary)

    artifacts = OUT / "output_artifacts.sha256"
    if not artifacts.is_file() or artifacts.is_symlink():
        fail("artifact list")
    found_summary = False
    for line in artifacts.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        artifact = OUT / name
        if not artifact.is_file() or artifact.is_symlink() or sha(artifact) != digest:
            fail("artifact mismatch " + name)
        if name == "summary.json":
            found_summary = True
    if not found_summary:
        fail("summary missing from artifact list")

    passed = speedup >= 1.05
    receipt = {
        "schema": "safe-c2-v4-metadatafix-validation-sealed-test-decision-v1",
        "status": "PASS_SEALED_TEST_UNLOCKED" if passed else "FAIL_SEALED_TEST_REMAINS_LOCKED",
        "scope": "one-shot CPU-only decision from precommitted final-guarded validation aggregate only",
        "validation_protocol": {"path": str(VALIDATION_PROTOCOL), "sha256": sha(VALIDATION_PROTOCOL)},
        "decision_protocol": {"path": str(DECISION_PROTOCOL), "sha256": sha(DECISION_PROTOCOL)},
        "validation_summary": {"path": str(summary_path), "sha256": sha(summary_path)},
        "validation_artifacts": {"path": str(artifacts), "sha256": sha(artifacts)},
        "formula": EXPECTED_FORMULA,
        "baseline_final_guarded_pipeline_wall_ms_sum": numerator,
        "guarded_final_guarded_pipeline_wall_ms_sum": denominator,
        "speedup": speedup,
        "threshold": 1.05,
        "raw_speculative_metric_used": False,
        "sealed_test_unlocked": passed,
        "retry_permitted": False,
    }
    fd = os.open(RECEIPT, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(receipt, f, sort_keys=True, indent=2)
        f.write("\n")
    print(json.dumps(receipt, sort_keys=True))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
