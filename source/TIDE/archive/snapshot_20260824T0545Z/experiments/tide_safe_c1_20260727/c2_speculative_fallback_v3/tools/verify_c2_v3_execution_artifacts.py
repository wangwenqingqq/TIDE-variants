#!/usr/bin/env python3
"""CPU-only external artifact verifier for the Safe-C2 v3 execution trust root.

The C++ runner's FNV fields are cross-checked, but SHA-256 and full JSONL
parsing are the trust evidence.  This program never launches a binary or GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from c2_v3_execution_common import (
    ContractError,
    MANIFEST_SCHEMA,
    ROOT_LITERAL,
    RUN_RE,
    STAGE_ORDER,
    STAGE_RUNNER_NAME,
    STAGE_TIMED_REPS,
    V2_SEALED_TEST_SHA256,
    WORKLOAD_SHA256,
    fnv1a64,
    load_json,
    load_workload,
    require_direct_dir,
    require_regular_file,
    root_from_module,
    sha256_file,
)
from verify_c2_v3_execution_pins import verify_pins

ROOT = root_from_module(__file__)
PINS_PATH = ROOT / "provenance/c2_v3_execution_pins_v1.json"
PLAN_PATH = ROOT / "protocols/c2_v3_execution_plan_v1.json"
BINARY = ROOT / "bin/GTS_safe_c2_speculative_fallback_v3_sift1m"
GAMMA_NAME = "final_v3_speculative_gamma_vector.txt"
PER_QUERY_RE = re.compile(r"^per_query_([A-Za-z0-9_]+)_rep_([0-9]+)\.jsonl$")


def fail(message: str) -> None:
    raise ContractError(message)


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} must be finite numeric")
    return float(value)


def direct_run(raw: str) -> Path:
    run = Path(raw)
    runs = ROOT / "runs"
    if not run.is_absolute() or run.parent != runs or not RUN_RE.fullmatch(run.name):
        fail(f"invalid run path: {run}")
    require_direct_dir(runs, "runs directory", inside_root=True)
    require_direct_dir(run, "run directory", inside_root=True)
    return run


def load_execution_manifest(run: Path) -> dict[str, Any]:
    path = run / "execution_manifest.json"
    require_regular_file(path, "execution manifest", inside_root=True)
    manifest = load_json(path, "execution manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("run_path") != str(run):
        fail("execution manifest schema/path mismatch")
    if manifest.get("v2_sealed_test_sha256_forbidden") != V2_SEALED_TEST_SHA256:
        fail("execution manifest v2 sealed-test guard mismatch")
    return manifest


def descriptor(path: Path, label: str) -> dict[str, Any]:
    require_regular_file(path, label, inside_root=True)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def verify_manifest_bindings(run: Path, manifest: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    pins = PINS_PATH
    plan = PLAN_PATH
    expected_pins = descriptor(pins, "execution PINS")
    expected_plan = descriptor(plan, "execution plan")
    if manifest.get("pins") != expected_pins or manifest.get("plan") != expected_plan:
        fail("execution manifest PINS/plan descriptor mismatch")
    if manifest.get("workload") != {"path": workload["path"], "sha256": WORKLOAD_SHA256}:
        fail("execution manifest workload binding mismatch")
    guard = manifest.get("guard")
    launcher = manifest.get("launcher")
    if not isinstance(guard, dict) or guard.get("path") != str(ROOT / "tools/run_c2_v3_execution_guard.sh"):
        fail("execution manifest guard descriptor missing/mismatch")
    if not isinstance(launcher, dict) or launcher.get("path") != str(ROOT / "tools/launch_c2_v3_execution.sh"):
        fail("execution manifest launcher descriptor missing/mismatch")
    if descriptor(Path(guard["path"]), "outer guard") != guard or descriptor(Path(launcher["path"]), "launcher") != launcher:
        fail("execution manifest guard/launcher hash mismatch")
    if manifest.get("host") != "CONFIGURE_ARCHIVE_HOST":
        fail("execution manifest host mismatch")
    return verify_pins(pins, expected_pins["sha256"])


def expected_argv(run: Path, stage: str, workload: dict[str, Any]) -> list[str]:
    binding = workload["stages"][stage]
    inputs = workload["bindings"]
    out = run / stage
    result = [
        str(BINARY), "--mode", "calibrate" if stage == "calibration" else "evaluate",
        "--base-fvecs", inputs["base_fvecs"]["path"],
        "--query-fvecs", inputs["query_fvecs"]["path"],
        "--groundtruth-ivecs", inputs["groundtruth_ivecs"]["path"],
        "--query-ids", binding["path"],
        "--out", str(out),
        "--stage", STAGE_RUNNER_NAME[stage],
        "--workload-manifest", workload["path"],
    ]
    if stage != "calibration":
        result += ["--gamma-vector-file", str(run / "calibration" / GAMMA_NAME)]
    result += ["--warmup-reps", "1", "--timed-reps", str(STAGE_TIMED_REPS[stage])]
    return result


def expected_summary_artifacts(stage: str) -> dict[str, list[tuple[str, int]]]:
    reps = STAGE_TIMED_REPS[stage]
    if stage == "calibration":
        return {
            "stage": [("calibration_final_guarded", 0)],
            "baseline": [("calibration_baseline", 0)],
        }
    return {
        "stage": [("speculative_guarded", rep) for rep in range(reps)],
        "baseline": [("baseline", rep) for rep in range(reps)],
    }


def validate_reference(ref: Any, query_count: int, label: str) -> None:
    if not isinstance(ref, dict):
        fail(f"{label} is not an object")
    for key in ("correct", "total", "invalid_output_ids", "duplicate_output_ids", "observed_distance_mismatches", "reference_inconsistencies", "boundary_ties"):
        if not isinstance(ref.get(key), int):
            fail(f"{label}.{key} is not an integer")
    if ref["total"] != query_count * 10:
        fail(f"{label}.total mismatch")
    if not 0 <= ref["correct"] <= ref["total"]:
        fail(f"{label}.correct outside range")
    if any(ref[key] != 0 for key in ("invalid_output_ids", "duplicate_output_ids", "observed_distance_mismatches", "reference_inconsistencies", "boundary_ties")):
        fail(f"{label} reports invalid/ambiguous reference output")
    finite_number(ref.get("recall_at_10"), f"{label}.recall_at_10")
    if not isinstance(ref.get("first_error"), str) or ref["first_error"]:
        fail(f"{label}.first_error must be empty on PASS")


def validate_stage_object(stage_obj: Any, stage: str, query_count: int, label: str) -> None:
    if not isinstance(stage_obj, dict):
        fail(f"{label} is not a stage object")
    reps = STAGE_TIMED_REPS[stage]
    if stage_obj.get("query_count") != query_count or stage_obj.get("timed_reps") != reps:
        fail(f"{label} query-count/timed-rep mismatch")
    # Calibration's reported final stage has no warmup; its paired baseline has
    # the registered warmup. Held-out ABBA reports one shared warmup for both.
    expected_warmup = 0 if stage == "calibration" and label.endswith("stage") else 1
    if stage_obj.get("warmup_reps") != expected_warmup:
        fail(f"{label} warmup-rep mismatch")
    if stage_obj.get("final_output_policy") != "gamma=1 fallback overwrites every gamma-only-pruned query" or stage_obj.get("raw_speculative_is_diagnostic_only") is not True:
        fail(f"{label} output policy mismatch")
    for name in ("speculative_static_api", "copies_and_fallback", "gamma_only_fallback", "guarded_timing"):
        if not isinstance(stage_obj.get(name), dict):
            fail(f"{label}.{name} missing")
    for key, value in stage_obj["guarded_timing"].items():
        finite_number(value, f"{label}.guarded_timing.{key}")
    if stage_obj["guarded_timing"].get("query_pipeline_wall_ms_sum", -1) < 0:
        fail(f"{label} has negative guarded pipeline time")
    final_reps = stage_obj.get("final_guarded_per_repetition")
    raw_reps = stage_obj.get("raw_speculative_per_repetition")
    if not isinstance(final_reps, list) or not isinstance(raw_reps, list) or len(final_reps) != reps or len(raw_reps) != reps:
        fail(f"{label} per-repetition count mismatch")
    for rep, value in enumerate(final_reps):
        validate_reference(value, query_count, f"{label}.final_rep[{rep}]")
    # Raw diagnostic can be weaker in overlap but must still be a complete finite
    # result representation for a successful guarded experiment.
    for rep, value in enumerate(raw_reps):
        validate_reference(value, query_count, f"{label}.raw_rep[{rep}]")
    final_sum = stage_obj.get("final_guarded_reference_sum")
    raw_sum = stage_obj.get("raw_speculative_reference_sum")
    if not isinstance(final_sum, dict) or not isinstance(raw_sum, dict):
        fail(f"{label} reference sums missing")
    for sum_name, repeated in (("final_guarded_reference_sum", final_reps), ("raw_speculative_reference_sum", raw_reps)):
        total = sum(item["total"] for item in repeated)
        correct = sum(item["correct"] for item in repeated)
        summary = stage_obj[sum_name]
        if summary.get("total") != total or summary.get("correct") != correct:
            fail(f"{label}.{sum_name} does not equal repetitions")
        for key in ("invalid_output_ids", "duplicate_output_ids", "observed_distance_mismatches", "reference_inconsistencies", "boundary_ties"):
            if summary.get(key) != 0:
                fail(f"{label}.{sum_name}.{key} is nonzero")


def parse_gamma(path: Path) -> list[float]:
    require_regular_file(path, "gamma vector", inside_root=True)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 8:
        fail("gamma vector must have exactly 8 lines")
    values: list[float] = []
    for expected_level, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 2:
            fail("malformed gamma vector line")
        if fields[0] != str(expected_level):
            fail("gamma vector levels are not canonical 0..7")
        try:
            value = float(fields[1])
        except ValueError as error:
            raise ContractError("gamma vector has nonnumeric value") from error
        if not math.isfinite(value) or value < 1.0:
            fail("gamma vector value is not finite >= 1")
        values.append(value)
    if values[:3] != [1.0, 1.0, 1.0]:
        fail("gamma vector violates frozen shallow levels 0..2")
    return values


def validate_query_h2d(path: Path, expected_count: int) -> None:
    data = load_json(path, "query_h2d")
    if data.get("query_count") != expected_count:
        fail("query_h2d query-count mismatch")
    finite_number(data.get("query_h2d_wall_ms"), "query_h2d.wall")
    finite_number(data.get("query_h2d_gpu_ms"), "query_h2d.gpu")


def validate_vector(value: Any, label: str, *, integer: bool) -> None:
    if not isinstance(value, list) or len(value) != 10:
        fail(f"{label} must contain exactly 10 elements")
    for index, item in enumerate(value):
        if integer:
            if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 1_000_000:
                fail(f"{label}[{index}] invalid ID")
        else:
            finite_number(item, f"{label}[{index}]")
    if integer and len(set(value)) != 10:
        fail(f"{label} contains duplicate IDs")


def validate_per_query_file(path: Path, expected_role: str, expected_rep: int, ids: list[int]) -> tuple[list[int], list[int]]:
    require_regular_file(path, "per-query JSONL", inside_root=True)
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    if len(raw_lines) != len(ids):
        fail(f"per-query row count mismatch: {path.name}")
    raw_overlaps: list[int] = []
    final_overlaps: list[int] = []
    for local, (line, global_qid) in enumerate(zip(raw_lines, ids)):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"invalid JSONL row {path.name}:{local}: {error}") from error
        if not isinstance(row, dict) or row.get("schema") != "safe-c2-v3-per-query-v1" or row.get("role") != expected_role or row.get("timed_rep") != expected_rep or row.get("local_index") != local or row.get("global_qid") != global_qid:
            fail(f"per-query identity mismatch: {path.name}:{local}")
        if not isinstance(row.get("gamma_only_fallback_applied"), bool):
            fail(f"per-query fallback boolean missing: {path.name}:{local}")
        validate_vector(row.get("gt_top10"), f"{path.name}:{local}.gt_top10", integer=True)
        for which in ("raw_speculative", "final_guarded"):
            block = row.get(which)
            if not isinstance(block, dict) or not isinstance(block.get("overlap"), int) or not 0 <= block["overlap"] <= 10:
                fail(f"per-query {which} overlap invalid: {path.name}:{local}")
            validate_vector(block.get("ids"), f"{path.name}:{local}.{which}.ids", integer=True)
            validate_vector(block.get("distances"), f"{path.name}:{local}.{which}.distances", integer=False)
        if row["gamma_only_fallback_applied"]:
            trace = row.get("first_gamma_only_prune")
            if not isinstance(trace, dict) or not isinstance(trace.get("nid"), int) or not isinstance(trace.get("level"), int):
                fail(f"per-query fallback trace missing: {path.name}:{local}")
            for field in ("lb_tri", "lb_eff", "disk_at_decision"):
                finite_number(trace.get(field), f"per-query fallback trace {field}")
        elif "first_gamma_only_prune" in row:
            fail(f"unexpected fallback trace when false: {path.name}:{local}")
        raw_overlaps.append(row["raw_speculative"]["overlap"])
        final_overlaps.append(row["final_guarded"]["overlap"])
    return raw_overlaps, final_overlaps


def artifact_descriptor_list(stage_obj: dict[str, Any], output: Path, roles: list[tuple[str, int]], label: str) -> dict[tuple[str, int], Path]:
    entries = stage_obj.get("per_query_artifacts")
    if not isinstance(entries, list) or len(entries) != len(roles):
        fail(f"{label} per-query artifact count mismatch")
    found: dict[tuple[str, int], Path] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            fail(f"{label} malformed artifact descriptor")
        role, rep, raw_path, fnv = entry.get("role"), entry.get("timed_rep"), entry.get("path"), entry.get("fnv1a64")
        if not isinstance(role, str) or not isinstance(rep, int) or not isinstance(raw_path, str) or not isinstance(fnv, str):
            fail(f"{label} malformed artifact fields")
        key = (role, rep)
        if key in found or key not in roles:
            fail(f"{label} unexpected/duplicate per-query role-rep")
        path = Path(raw_path)
        if path.parent != output or path.name != f"per_query_{role}_rep_{rep}.jsonl":
            fail(f"{label} per-query artifact path mismatch")
        require_regular_file(path, f"{label} per-query artifact", inside_root=True)
        if fnv1a64(path.read_bytes()) != fnv:
            fail(f"{label} runner FNV mismatch for {path.name}")
        found[key] = path
    if set(found) != set(roles):
        fail(f"{label} per-query artifact role-rep set mismatch")
    return found


def check_summary(stage: str, output: Path, workload: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    summary_path = output / "summary.json"
    require_regular_file(summary_path, "runner summary", inside_root=True)
    summary = load_json(summary_path, "runner summary")
    expected_status = "PASS_V3_CALIBRATION_GUARDED" if stage == "calibration" else "PASS_V3_HELDOUT_GUARDED"
    if summary.get("schema") != "safe-c2-speculative-fallback-v3-run-v1" or summary.get("status") != expected_status or summary.get("v3_speculative_fallback") is not True:
        fail(f"summary schema/status mismatch: {stage}")
    if summary.get("mode") != ("calibrate" if stage == "calibration" else "evaluate") or summary.get("stage") != STAGE_RUNNER_NAME[stage]:
        fail(f"summary mode/stage mismatch: {stage}")
    if summary.get("archived_incremental_updater_used") is not False or summary.get("dynamic_exact_smoke_used") is not False:
        fail("summary claims a forbidden archived/dynamic path")
    workload_binding = summary.get("workload_binding")
    if not isinstance(workload_binding, dict) or workload_binding.get("manifest") != workload["path"] or workload_binding.get("schema") != "gts-v3-compact-learn-workload-v1" or workload_binding.get("sha256_verified_in_runner") is not True:
        fail(f"summary workload binding mismatch: {stage}")
    expected_inputs = {
        "base_fvecs": workload["bindings"]["base_fvecs"]["path"],
        "query_fvecs": workload["bindings"]["query_fvecs"]["path"],
        "groundtruth_ivecs": workload["bindings"]["groundtruth_ivecs"]["path"],
        "stage_query_ids": workload["stages"][stage]["path"],
    }
    if summary.get("input_paths") != expected_inputs:
        fail(f"summary input-path binding mismatch: {stage}")
    q_count = workload["stages"][stage]["count"]
    gamma = summary.get("gamma_vector")
    if not isinstance(gamma, list) or len(gamma) != 8 or any(not math.isfinite(float(value)) or float(value) < 1.0 for value in gamma) or [float(value) for value in gamma[:3]] != [1.0, 1.0, 1.0]:
        fail(f"summary gamma vector invalid: {stage}")
    certificate = summary.get("interval_certificate")
    if not isinstance(certificate, dict) or certificate.get("status") != "PASS_OUTWARD_SAFE_REPAIRED_AND_REVERIFIED" or certificate.get("post_reverify_passed") is not True:
        fail(f"summary interval certificate invalid: {stage}")
    gate = summary.get("final_guarded_per_query_no_regression_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True or gate.get("query_count") != q_count or gate.get("repetitions_checked") != STAGE_TIMED_REPS[stage] or gate.get("violating_queries_sum") != 0 or gate.get("max_violating_queries") != 0 or not isinstance(gate.get("min_per_query_delta"), int) or gate["min_per_query_delta"] < 0:
        fail(f"summary final guarded gate invalid: {stage}")
    timing = summary.get("timing")
    if not isinstance(timing, dict) or not isinstance(timing.get("stage"), dict):
        fail(f"summary timing stage missing: {stage}")
    stage_obj = timing["stage"]
    validate_stage_object(stage_obj, stage, q_count, f"{stage}.stage")
    baseline = summary.get("paired_same_split_baseline")
    if not isinstance(baseline, dict):
        fail(f"summary paired baseline missing: {stage}")
    validate_stage_object(baseline, stage, q_count, f"{stage}.baseline")
    comparison = summary.get("timing_comparison")
    if not isinstance(comparison, dict):
        fail(f"summary timing comparison missing: {stage}")
    if comparison.get("metric_if_eligible") != "guarded_timing.query_pipeline_wall_ms; never speculative_static_api alone":
        fail(f"summary speed metric boundary mismatch: {stage}")
    if stage == "calibration":
        if comparison.get("eligible_for_speed_comparison") is not False or comparison.get("shared_warmup_pairs") != 0 or comparison.get("timed_pair_order") != []:
            fail("calibration must not be speed eligible or ABBA")
    else:
        expected_order = ["baseline_then_speculative_guarded" if rep % 2 == 0 else "speculative_guarded_then_baseline" for rep in range(STAGE_TIMED_REPS[stage])]
        if comparison.get("eligible_for_speed_comparison") is not True or comparison.get("shared_warmup_pairs") != 1 or comparison.get("timed_pair_order") != expected_order:
            fail(f"held-out ABBA contract mismatch: {stage}")
    expected = expected_summary_artifacts(stage)
    stage_artifacts = artifact_descriptor_list(stage_obj, output, expected["stage"], f"{stage}.stage")
    baseline_artifacts = artifact_descriptor_list(baseline, output, expected["baseline"], f"{stage}.baseline")
    return {"summary": summary, "q_count": q_count, "stage_artifacts": stage_artifacts, "baseline_artifacts": baseline_artifacts, "gamma": [float(value) for value in gamma]}


def verify_stage(run: Path, manifest: dict[str, Any], workload: dict[str, Any], stage: str, *, require_manifest_pass: bool) -> dict[str, Any]:
    state = manifest.get("stages", {}).get(stage)
    if not isinstance(state, dict):
        fail(f"run manifest missing stage state: {stage}")
    allowed = {"PASS"} if require_manifest_pass else {"RUNNING"}
    if state.get("state") not in allowed:
        fail(f"run manifest stage status mismatch: {stage}")
    output = run / stage
    require_direct_dir(output, f"{stage} output", inside_root=True)
    if state.get("output_dir") != str(output):
        fail(f"run manifest output path mismatch: {stage}")
    if state.get("command_argv") != expected_argv(run, stage, workload):
        fail(f"run manifest fixed argv mismatch: {stage}")
    session = state.get("session")
    if not isinstance(session, dict) or session.get("physical_gpu_index") != 0 or not isinstance(session.get("uuid"), str) or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", session["uuid"]) or not isinstance(session.get("pci_bus_id"), str) or not session["pci_bus_id"]:
        fail(f"run manifest UUID/PCI session binding invalid: {stage}")
    result = check_summary(stage, output, workload, manifest)
    ids = workload["stages"][stage]["ids"]
    stage_paths = result["stage_artifacts"]
    baseline_paths = result["baseline_artifacts"]
    for (role, rep), path in {**stage_paths, **baseline_paths}.items():
        raw, final = validate_per_query_file(path, role, rep, ids)
        target_obj = result["summary"]["timing"]["stage"] if (role, rep) in stage_paths else result["summary"]["paired_same_split_baseline"]
        if sum(raw) != target_obj["raw_speculative_per_repetition"][rep]["correct"] or sum(final) != target_obj["final_guarded_per_repetition"][rep]["correct"]:
            fail(f"per-query overlap sum does not match summary: {path.name}")
    # Ensure every on-disk JSONL belongs to exactly the current stage and is
    # covered by summary or is an allowed calibration selection diagnostic.
    files = [path for path in sorted(output.iterdir()) if path.is_file() and not path.is_symlink()]
    expected_regular = {"summary.json", "query_h2d.json"}
    if stage == "calibration":
        expected_regular.add(GAMMA_NAME)
    observed_jsonl: set[str] = set()
    known_paths = {path.name for path in stage_paths.values()} | {path.name for path in baseline_paths.values()}
    for path in files:
        if path.name in expected_regular:
            continue
        match = PER_QUERY_RE.fullmatch(path.name)
        if match is None:
            fail(f"unexpected regular stage artifact: {stage}/{path.name}")
        role, rep = match.group(1), int(match.group(2))
        if stage == "calibration":
            if role not in {"calibration_baseline", "calibration_final_guarded"} and not re.fullmatch(r"calibration_(candidate_l[0-9]+_i[0-9]+|confirm_l[0-9]+)", role):
                fail(f"unexpected calibration diagnostic role: {role}")
            validate_per_query_file(path, role, rep, ids)
        else:
            if role not in {"baseline", "speculative_guarded"} or rep not in range(STAGE_TIMED_REPS[stage]):
                fail(f"unexpected held-out JSONL role/rep: {path.name}")
        observed_jsonl.add(path.name)
    if stage != "calibration" and observed_jsonl != known_paths:
        fail(f"held-out JSONL set differs from exact ABBA summary: {stage}")
    validate_query_h2d(output / "query_h2d.json", result["q_count"])
    gamma_path = run / "calibration" / GAMMA_NAME
    gamma_values = parse_gamma(gamma_path)
    if any(not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6) for left, right in zip(gamma_values, result["gamma"])):
        fail(f"summary gamma vector differs from calibration gamma file: {stage}")
    if stage != "calibration":
        calibration_state = manifest["stages"].get("calibration")
        if not isinstance(calibration_state, dict) or calibration_state.get("state") != "PASS" or calibration_state.get("produced_gamma") != descriptor(gamma_path, "calibration gamma"):
            fail("held-out stage lacks exact recorded calibration gamma SHA provenance")
    if require_manifest_pass:
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, list):
            fail(f"completed stage lacks artifact SHA list: {stage}")
        actual: list[dict[str, Any]] = []
        for path in sorted(output.rglob("*")):
            if path.is_symlink() or (not path.is_dir() and not path.is_file()):
                fail(f"bad completed artifact entry: {path}")
            if path.is_file():
                actual.append({"relative_path": str(path.relative_to(run)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
        if artifacts != actual:
            fail(f"completed stage artifact SHA list mismatch: {stage}")
        verifier = state.get("artifact_verifier")
        if not isinstance(verifier, dict):
            fail(f"completed stage verifier descriptor missing: {stage}")
        if descriptor(Path(verifier.get("path", "")), f"{stage} verifier") != verifier:
            fail(f"completed stage verifier SHA mismatch: {stage}")
        child = state.get("child")
        if not isinstance(child, dict) or not isinstance(child.get("pid"), int) or child.get("pid") <= 0 or child.get("returncode") != 0:
            fail(f"completed stage direct child PID/zero-exit provenance missing: {stage}")
        child_logs = state.get("child_logs")
        if not isinstance(child_logs, dict) or set(child_logs) != {"stdout", "stderr"}:
            fail(f"completed stage child-log descriptors missing: {stage}")
        for role, value in child_logs.items():
            if not isinstance(value, dict) or descriptor(Path(value.get("path", "")), f"{stage} {role} log") != value:
                fail(f"completed stage child-log SHA mismatch: {stage}/{role}")
    return {"stage": stage, "summary_sha256": sha256_file(output / "summary.json"), "jsonl_count": len(observed_jsonl), "gamma_sha256": sha256_file(gamma_path)}


def validate_runtime_binding_and_telemetry(run: Path, manifest: dict[str, Any]) -> None:
    runtime = manifest.get("runtime_binding")
    if not isinstance(runtime, dict) or runtime.get("physical_gpu_index") != 0:
        fail("runtime UUID/PCI/environment binding missing")
    uuid, pci = runtime.get("physical_gpu_uuid"), runtime.get("pci_bus_id")
    if not isinstance(uuid, str) or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid) or not isinstance(pci, str) or not pci:
        fail("runtime UUID/PCI binding malformed")
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/root",
        "LANG": "C",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": uuid,
        "NVIDIA_VISIBLE_DEVICES": uuid,
        "LD_LIBRARY_PATH": "/usr/local/cuda-13.1/lib64",
    }
    if runtime.get("environment") != env:
        fail("runtime env -i whitelist mismatch")
    env_desc = runtime.get("environment_file")
    if not isinstance(env_desc, dict) or descriptor(Path(env_desc.get("path", "")), "runtime environment file") != env_desc:
        fail("runtime environment SHA descriptor mismatch")
    env_json = load_json(Path(env_desc["path"]), "runtime environment file")
    if env_json.get("schema") != "safe-c2-v3-child-environment-v1" or env_json.get("environment") != env:
        fail("runtime environment JSON mismatch")
    for stage in STAGE_ORDER:
        session = manifest.get("stages", {}).get(stage, {}).get("session")
        if not isinstance(session, dict) or session.get("physical_gpu_index") != 0 or session.get("uuid") != uuid or session.get("pci_bus_id") != pci:
            fail(f"stage UUID/PCI does not equal runtime binding: {stage}")
    entries = manifest.get("telemetry")
    if not isinstance(entries, list):
        fail("complete manifest telemetry list missing")
    strict_labels = {"outer_pre_strict_idle", "calibration_pre_strict_idle", "validation_pre_strict_idle", "sealed_test_pre_strict_idle"}
    nonstrict_labels = {"calibration_post", "validation_post", "sealed_test_post", "outer_exit"}
    required_labels = strict_labels | nonstrict_labels
    seen: set[str] = set()
    hardware_identity: tuple[str, str] | None = None
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("label"), str):
            fail("malformed telemetry descriptor")
        label = entry["label"]
        if label in seen or label not in required_labels:
            fail(f"telemetry labels must be exact/unique; saw {label}")
        path = Path(entry.get("path", ""))
        if path.parent != run / "telemetry":
            fail("telemetry path leaves canonical telemetry directory")
        expected_desc = descriptor(path, "telemetry")
        if {key: entry.get(key) for key in ("path", "sha256", "bytes")} != expected_desc:
            fail("telemetry SHA descriptor mismatch")
        data = load_json(path, "telemetry JSON")
        if data.get("schema") != "safe-c2-v3-gpu0-telemetry-v1" or data.get("label") != label:
            fail("telemetry schema/label mismatch")
        if data.get("physical_gpu_index") != 0 or data.get("physical_gpu_uuid") != uuid or data.get("pci_bus_id") != pci:
            fail("telemetry UUID/PCI does not equal runtime binding")
        if data.get("compute_processes") != "none":
            fail("telemetry records active/unknown compute process")
        total = data.get("memory_total_mib"); used = data.get("memory_used_mib"); util = data.get("utilization_percent")
        if not isinstance(total, int) or not isinstance(used, int) or not isinstance(util, int) or total <= 0 or used < 0 or util < 0:
            fail("telemetry numeric state malformed")
        gpu_name, driver, pstate = data.get("gpu_name"), data.get("driver_version"), data.get("pstate")
        if not isinstance(gpu_name, str) or not gpu_name or not isinstance(driver, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", driver) or not isinstance(pstate, str) or not re.fullmatch(r"P[0-9]+", pstate):
            fail("telemetry hardware identity/pstate malformed")
        for key in ("temperature_c", "sm_clock_mhz", "mem_clock_mhz", "power_draw_w", "power_limit_w"):
            value = finite_number(data.get(key), f"telemetry.{key}")
            if value < 0.0 or (key == "power_limit_w" and value <= 0.0):
                fail(f"telemetry {key} is outside physical range")
        if hardware_identity is None:
            hardware_identity = (gpu_name, driver)
        elif hardware_identity != (gpu_name, driver):
            fail("telemetry GPU name/driver changed during one guarded workflow")
        strict = data.get("strict_idle_required")
        if label in strict_labels:
            if strict is not True or used > 256 or util != 0:
                fail(f"strict-idle telemetry fails threshold: {label}")
        else:
            if strict is not False:
                fail(f"post/exit telemetry must not masquerade as strict idle: {label}")
        for raw_path_key, raw_hash_key in (("status_raw_path", "status_raw_sha256"), ("compute_apps_raw_path", "compute_apps_raw_sha256")):
            raw_path = Path(data.get(raw_path_key, ""))
            if raw_path.parent != run / "telemetry":
                fail(f"telemetry raw path leaves canonical directory: {label}/{raw_path_key}")
            require_regular_file(raw_path, "telemetry raw provenance", inside_root=True)
            if not isinstance(data.get(raw_hash_key), str) or sha256_file(raw_path) != data[raw_hash_key]:
                fail(f"telemetry raw provenance mismatch: {label}/{raw_path_key}")
        seen.add(label)
    if seen != required_labels:
        fail(f"complete telemetry label set mismatch; missing={sorted(required_labels - seen)}, extra={sorted(seen - required_labels)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", choices=STAGE_ORDER)
    group.add_argument("--all-stages", action="store_true")
    group.add_argument("--run", dest="full_run", action="store_true")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = direct_run(args.run_dir)
    manifest = load_execution_manifest(run)
    workload = load_workload(ROOT, hash_inputs=True)
    pins_result = verify_manifest_bindings(run, manifest, workload)
    if args.stage:
        if manifest.get("status") != "RUNNING":
            fail("stage verification requires a RUNNING manifest")
        result = verify_stage(run, manifest, workload, args.stage, require_manifest_pass=False)
        status = "PASS_STAGE"
    elif args.all_stages:
        if manifest.get("status") != "RUNNING":
            fail("all-stages verification requires RUNNING pre-finalization manifest")
        result = {stage: verify_stage(run, manifest, workload, stage, require_manifest_pass=True) for stage in STAGE_ORDER}
        # This is the verifier consumed by finalize, so UUID/PCI/env and exact
        # pre/post telemetry must be valid *before* COMPLETE can be written.
        validate_runtime_binding_and_telemetry(run, manifest)
        status = "PASS_ALL_STAGES"
    else:
        if manifest.get("status") != "COMPLETE":
            fail("full-run verification requires COMPLETE manifest")
        result = {stage: verify_stage(run, manifest, workload, stage, require_manifest_pass=True) for stage in STAGE_ORDER}
        final = manifest.get("final_artifact_verifier")
        if not isinstance(final, dict) or descriptor(Path(final.get("path", "")), "final artifact verifier") != final:
            fail("complete manifest final verifier descriptor mismatch")
        final_json = load_json(Path(final["path"]), "final artifact verifier")
        if final_json.get("status") != "PASS_ALL_STAGES":
            fail("complete manifest does not bind PASS_ALL_STAGES final verifier")
        validate_runtime_binding_and_telemetry(run, manifest)
        status = "PASS_RUN"
    print(json.dumps({
        "schema": "safe-c2-v3-execution-artifact-verification-v1",
        "status": status,
        "run": str(run),
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
        "pins_sha256": pins_result["pins_sha256"],
        "result": result,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"SAFE-C2-V3 EXECUTION ARTIFACT VERIFY FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)
