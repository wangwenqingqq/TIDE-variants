#!/usr/bin/env python3
"""
Independent CPU-only, read-only verifier for a completed Safe-C2 static-AABB
held-out performance batch with v3 per-invocation benchmark timelines.

The canonical v3 run contract is deliberately explicit:
  * five invocation_XX children, each with result.json and
    benchmark_timeline.json;
  * every timeline has one run token/PID and exactly 5 warmups + 30 paired
    repetitions (60 timed operations);
  * result.json timing arrays, timing_pairs.json, and timeline CUDA timings
    agree exactly up to a small JSON-float tolerance;
  * telemetry/process samples cover each timeline's timed envelope without a
    foreign compute process and record the clock fields.

The normal audit only reads existing artifacts.  It imports no GPU library and
does not execute external commands.  --selftest creates a disposable synthetic
artifact tree solely to exercise the parser and invariants.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any, Iterable

AUDIT_SCHEMA = "safe-c2-static-aabb-heldout-perf-result-audit-v3"
DEFAULT_COMPLETE_STATUS = "COMPLETE_CONDITIONAL_DVFS_UNCONTROLLED"
INVOCATIONS = 5
WARMUPS = 5
REPS = 30
OPS_PER_INVOCATION = REPS * 2
DEFAULT_MAX_GAP_MS = 500.0
REQUIRED_CSV_COLUMNS = (
    "wall_utc_ns",
    "monotonic_ns",
    "sample_seq",
    "phase",
    "gpu_index",
    "gpu_uuid",
    "pstate",
    "clock_sm_mhz",
    "clock_graphics_mhz",
    "clock_memory_mhz",
    "power_draw_w",
    "power_limit_w",
    "temperature_gpu_c",
    "utilization_gpu_pct",
    "utilization_memory_pct",
    "memory_used_mib",
)
NUMERIC_CSV_COLUMNS = (
    "clock_sm_mhz",
    "clock_graphics_mhz",
    "clock_memory_mhz",
    "power_draw_w",
    "power_limit_w",
    "temperature_gpu_c",
    "utilization_gpu_pct",
    "utilization_memory_pct",
    "memory_used_mib",
)
CLOCK_COLUMNS = (
    "clock_sm_mhz",
    "clock_graphics_mhz",
    "clock_memory_mhz",
)
CHILD_GATE_COUNTERS = (
    "invalid",
    "duplicate",
    "gt_set_mismatch",
    "distance_mismatch",
)


class AuditFailure(RuntimeError):
    """A fail-closed artifact-contract violation."""


def fail(message: str) -> None:
    raise AuditFailure(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value.lower())
    )


def as_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label}: expected integer")
    if minimum is not None and value < minimum:
        fail(f"{label}: expected >= {minimum}, got {value}")
    return value


def as_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        fail(f"{label}: expected finite number")
    try:
        if isinstance(value, str):
            require(value.strip() != "", f"{label}: expected finite number")
            result = float(value.strip())
        elif isinstance(value, (int, float)):
            result = float(value)
        else:
            fail(f"{label}: expected finite number")
    except (TypeError, ValueError):
        fail(f"{label}: expected finite number")
    if not math.isfinite(result):
        fail(f"{label}: non-finite number")
    if positive and result <= 0.0:
        fail(f"{label}: expected positive number")
    return result


def same_number(a: Any, b: Any, label: str) -> None:
    x = as_number(a, label + " lhs")
    y = as_number(b, label + " rhs")
    # CUDA event times originate from the same serialized float.  The small
    # allowance only covers JSON decimal round-tripping, not a real mismatch.
    if not math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6):
        fail(f"{label}: {x} != {y}")


def sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        fail(f"not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_root(path: Path) -> Path:
    try:
        root = path.resolve(strict=True)
    except FileNotFoundError:
        fail(f"run directory does not exist: {path}")
    require(root.is_dir(), f"run path is not a directory: {root}")
    return root


def resolve_inside(root: Path, raw: Any, label: str) -> Path:
    require(isinstance(raw, str) and raw, f"{label}: missing artifact path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        fail(f"{label}: cannot resolve artifact path: {exc}")
    if resolved != root and root not in resolved.parents:
        fail(f"{label}: artifact escapes run directory: {raw}")
    return resolved


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        fail(f"{label}: missing regular JSON file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"{label}: invalid JSON: {exc}")
    require(isinstance(payload, dict), f"{label}: JSON top level must be an object")
    return payload


def artifact_record(
    root: Path,
    record: Any,
    label: str,
    *,
    expected_path: Path | None = None,
) -> tuple[Path, str, int]:
    require(isinstance(record, dict), f"{label}: artifact record must be an object")
    for key in ("path", "bytes", "sha256"):
        require(key in record, f"{label}: artifact record missing {key}")
    path = resolve_inside(root, record["path"], label)
    if expected_path is not None:
        expected = expected_path.resolve(strict=False)
        require(path == expected, f"{label}: path mismatch {path} != {expected}")
    require(path.is_file() and not path.is_symlink(), f"{label}: artifact missing or symlink: {path}")
    expected_bytes = as_int(record["bytes"], label + ".bytes", minimum=0)
    actual_bytes = path.stat().st_size
    require(actual_bytes == expected_bytes, f"{label}: byte size mismatch {actual_bytes} != {expected_bytes}")
    expected_sha = record["sha256"]
    require(is_sha256(expected_sha), f"{label}: invalid SHA-256")
    actual_sha = sha256_file(path)
    require(actual_sha == expected_sha.lower(), f"{label}: SHA-256 mismatch")
    return path, actual_sha, actual_bytes


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def art_for(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def nested_first(mapping: Any, names: Iterable[str]) -> Any:
    if not isinstance(mapping, dict):
        return None
    for name in names:
        if name in mapping:
            return mapping[name]
    meta = mapping.get("metadata")
    if isinstance(meta, dict):
        for name in names:
            if name in meta:
                return meta[name]
    return None


def require_v3_schema(document: dict[str, Any], label: str) -> None:
    schema = document.get("schema")
    require(isinstance(schema, str) and "v3" in schema.lower(), f"{label}: expected v3 schema")


def build_manifest_index(root: Path, manifest: dict[str, Any]) -> dict[str, tuple[Path, str, int]]:
    raw = manifest.get("artifacts")
    require(isinstance(raw, dict) and raw, "manifest.artifacts: expected nonempty object")
    index: dict[str, tuple[Path, str, int]] = {}
    seen_paths: set[Path] = set()
    for key, record in raw.items():
        require(isinstance(key, str) and key and not Path(key).is_absolute(), "manifest artifact key invalid")
        expected_path = resolve_inside(root, key, "manifest artifact key")
        path, digest, byte_count = artifact_record(
            root, record, f"manifest.artifacts[{key!r}]", expected_path=expected_path
        )
        require(path not in seen_paths, f"manifest contains duplicate artifact path: {path}")
        seen_paths.add(path)
        index[key] = (path, digest, byte_count)
    return index


def require_manifest_path(
    root: Path,
    manifest_index: dict[str, tuple[Path, str, int]],
    relative: str,
) -> tuple[Path, str, int]:
    expected = resolve_inside(root, relative, "required manifest path")
    for _, value in manifest_index.items():
        if value[0] == expected:
            return value
    fail(f"manifest omits required artifact: {relative}")


def normalize_children(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    raw = result.get("children")
    require(isinstance(raw, list) and len(raw) == INVOCATIONS, "result.children: expected five entries")
    children: dict[int, dict[str, Any]] = {}
    for entry in raw:
        require(isinstance(entry, dict), "result.children entry must be object")
        invocation = as_int(entry.get("invocation"), "result.children.invocation", minimum=1)
        require(invocation <= INVOCATIONS, f"result.children invocation out of range: {invocation}")
        require(invocation not in children, f"result.children duplicate invocation {invocation}")
        children[invocation] = entry
    require(set(children) == set(range(1, INVOCATIONS + 1)), "result.children invocation set must be 1..5")
    return children


def top_level_timeline_binding(result: dict[str, Any], invocation: int) -> Any:
    for key in ("benchmark_timelines", "timeline_bindings", "timelines"):
        candidate = result.get(key)
        if isinstance(candidate, dict):
            if str(invocation) in candidate:
                return candidate[str(invocation)]
            if invocation in candidate:
                return candidate[invocation]
        if isinstance(candidate, list):
            for entry in candidate:
                if isinstance(entry, dict) and entry.get("invocation") == invocation:
                    return entry
    return None


def timeline_binding_for(result: dict[str, Any], child: dict[str, Any], invocation: int) -> dict[str, Any]:
    candidate = nested_first(
        child,
        ("timeline", "benchmark_timeline", "timeline_binding", "benchmark_timeline_binding"),
    )
    if candidate is None:
        candidate = top_level_timeline_binding(result, invocation)
    require(isinstance(candidate, dict), f"invocation {invocation}: aggregate timeline binding missing")
    return candidate


def artifact_from_binding(binding: dict[str, Any], label: str) -> dict[str, Any]:
    if all(key in binding for key in ("path", "bytes", "sha256")):
        return binding
    candidate = nested_first(
        binding,
        ("artifact", "timeline_artifact", "file", "timeline_file", "benchmark_timeline_file"),
    )
    require(isinstance(candidate, dict), f"{label}: timeline artifact record missing")
    return candidate


def result_artifact_from_child(child: dict[str, Any], label: str) -> dict[str, Any]:
    candidate = nested_first(child, ("result", "result_artifact", "child_result"))
    require(isinstance(candidate, dict), f"{label}: child result artifact record missing")
    return candidate


def binding_value(binding: dict[str, Any], names: Iterable[str], label: str) -> Any:
    value = nested_first(binding, names)
    require(value is not None, f"{label}: binding missing one of {', '.join(names)}")
    return value


def parse_launches(launch: dict[str, Any]) -> dict[int, dict[str, Any]]:
    raw = launch.get("launches")
    require(isinstance(raw, list) and len(raw) == INVOCATIONS, "launch.launches: expected five entries")
    launches: dict[int, dict[str, Any]] = {}
    for entry in raw:
        require(isinstance(entry, dict), "launch.launches entry must be object")
        invocation = as_int(entry.get("invocation"), "launch invocation", minimum=1)
        require(invocation <= INVOCATIONS and invocation not in launches, "launch invocation set invalid")
        as_int(entry.get("pid"), f"launch {invocation}.pid", minimum=1)
        token = nested_first(entry, ("run_token", "timeline_token", "benchmark_token"))
        require(isinstance(token, str) and token, f"launch {invocation}: run token missing")
        launches[invocation] = entry
    require(set(launches) == set(range(1, INVOCATIONS + 1)), "launch invocation set must be 1..5")
    return launches


def parse_timeline(path: Path, invocation: int) -> dict[str, Any]:
    timeline = load_json(path, f"invocation {invocation} benchmark_timeline")
    require_v3_schema(timeline, f"timeline {invocation}")
    status = timeline.get("status")
    require(isinstance(status, str) and status.startswith("PASS"), f"timeline {invocation}: non-pass status")
    token = timeline.get("run_token")
    require(isinstance(token, str) and token, f"timeline {invocation}: run_token missing")
    pid = as_int(timeline.get("pid"), f"timeline {invocation}.pid", minimum=1)
    clock_id = timeline.get("clock_id")
    require(
        isinstance(clock_id, str) and "MONOTONIC" in clock_id.upper(),
        f"timeline {invocation}: clock_id must identify monotonic clock",
    )
    start = as_int(
        timeline.get("timed_envelope_start_ns"),
        f"timeline {invocation}.timed_envelope_start_ns",
        minimum=0,
    )
    end = as_int(
        timeline.get("timed_envelope_end_ns"),
        f"timeline {invocation}.timed_envelope_end_ns",
        minimum=1,
    )
    require(end > start, f"timeline {invocation}: invalid timed envelope")
    require(as_int(timeline.get("warmups"), f"timeline {invocation}.warmups", minimum=0) == WARMUPS,
            f"timeline {invocation}: warmups must be {WARMUPS}")
    require(as_int(timeline.get("reps"), f"timeline {invocation}.reps", minimum=0) == REPS,
            f"timeline {invocation}: reps must be {REPS}")
    operations = timeline.get("operations")
    require(isinstance(operations, list) and len(operations) == OPS_PER_INVOCATION,
            f"timeline {invocation}: expected {OPS_PER_INVOCATION} operations")

    by_index: dict[int, dict[str, Any]] = {}
    by_method_rep: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in operations:
        require(isinstance(raw, dict), f"timeline {invocation}: operation must be object")
        op_index = as_int(raw.get("op_index"), f"timeline {invocation}.op_index", minimum=0)
        require(op_index < OPS_PER_INVOCATION and op_index not in by_index,
                f"timeline {invocation}: invalid/duplicate op_index {op_index}")
        rep = as_int(raw.get("rep"), f"timeline {invocation}.rep", minimum=0)
        require(rep < REPS, f"timeline {invocation}: rep out of range")
        method = raw.get("method")
        require(method in ("baseline", "static_aabb"),
                f"timeline {invocation}: invalid method {method!r}")
        order = as_int(raw.get("order"), f"timeline {invocation}.order", minimum=0)
        require(order in (0, 1), f"timeline {invocation}: order must be 0 or 1")
        require(op_index == 2 * rep + order,
                f"timeline {invocation}: op_index must equal 2*rep+order")
        key = (method, rep)
        require(key not in by_method_rep,
                f"timeline {invocation}: duplicate method/rep {method}/{rep}")
        host_start = as_int(raw.get("host_start_ns"), f"timeline {invocation}.host_start_ns", minimum=0)
        host_end = as_int(raw.get("host_end_ns"), f"timeline {invocation}.host_end_ns", minimum=1)
        require(host_end > host_start, f"timeline {invocation}: operation with nonpositive host duration")
        require(start <= host_start and host_end <= end,
                f"timeline {invocation}: operation outside timed envelope")
        as_number(raw.get("cuda_ms"), f"timeline {invocation}.cuda_ms", positive=True)
        by_index[op_index] = raw
        by_method_rep[key] = raw

    require(set(by_index) == set(range(OPS_PER_INVOCATION)), f"timeline {invocation}: op index set invalid")
    require(
        set(by_method_rep) == {
            (method, rep)
            for method in ("baseline", "static_aabb")
            for rep in range(REPS)
        },
        f"timeline {invocation}: methods/repetitions are incomplete",
    )
    previous_end = start
    orders: dict[int, tuple[str, str]] = {}
    for op_index in range(OPS_PER_INVOCATION):
        op = by_index[op_index]
        require(as_int(op["host_start_ns"], "timeline host_start", minimum=0) >= previous_end,
                f"timeline {invocation}: operations overlap or are non-monotonic")
        previous_end = as_int(op["host_end_ns"], "timeline host_end", minimum=1)
    for rep in range(REPS):
        first = by_index[2 * rep]
        second = by_index[2 * rep + 1]
        require(first["rep"] == rep and second["rep"] == rep, f"timeline {invocation}: pair ordering invalid")
        require(first["method"] != second["method"], f"timeline {invocation}: pair uses one method twice")
        # Paired order must alternate to avoid a systematic first-operation bias.
        expected = ("baseline", "static_aabb") if rep % 2 == 0 else ("static_aabb", "baseline")
        require((first["method"], second["method"]) == expected,
                f"timeline {invocation}: alternating order violation at rep {rep}")
        orders[rep] = expected

    return {
        "document": timeline,
        "token": token,
        "pid": pid,
        "clock_id": clock_id,
        "start_ns": start,
        "end_ns": end,
        "operations": by_index,
        "by_method_rep": by_method_rep,
        "orders": orders,
    }


def child_timing_arrays(child_result: dict[str, Any], invocation: int) -> dict[str, list[float]]:
    status = child_result.get("status")
    require(isinstance(status, str) and status.startswith("PASS"), f"child {invocation}: non-pass status")
    gate = nested_first(child_result, ("unmeasured_gate", "correctness_gate"))
    require(isinstance(gate, dict), f"child {invocation}: correctness gate missing")
    for method in ("baseline", "static_aabb"):
        method_gate = gate.get(method)
        require(isinstance(method_gate, dict), f"child {invocation}: {method} gate missing")
        for key in CHILD_GATE_COUNTERS:
            require(method_gate.get(key) == 0, f"child {invocation}: {method}.{key} must be zero")
    require(gate.get("id_sets_equal") is True, f"child {invocation}: ID sets differ")
    require(child_result.get("snapshot_pre_post_byte_equal") is True,
            f"child {invocation}: static snapshot changed")
    timing = nested_first(child_result, ("timing_gpu_ms", "cuda_timing_ms"))
    require(isinstance(timing, dict), f"child {invocation}: GPU timing arrays missing")
    output: dict[str, list[float]] = {}
    for method in ("baseline", "static_aabb"):
        values = timing.get(method)
        require(isinstance(values, list) and len(values) == REPS,
                f"child {invocation}: {method} timing array must have {REPS} entries")
        output[method] = [
            as_number(value, f"child {invocation}.{method}[{rep}]", positive=True)
            for rep, value in enumerate(values)
        ]
    return output


def compare_timeline_to_child(
    timeline: dict[str, Any],
    child_arrays: dict[str, list[float]],
    invocation: int,
) -> None:
    for method in ("baseline", "static_aabb"):
        for rep in range(REPS):
            operation = timeline["by_method_rep"][(method, rep)]
            same_number(
                operation["cuda_ms"],
                child_arrays[method][rep],
                f"invocation {invocation}: timeline vs child {method}[{rep}]",
            )


def parse_timing_pairs(
    root: Path,
    timing_pairs: dict[str, Any],
    child_data: dict[int, dict[str, Any]],
    manifest_index: dict[str, tuple[Path, str, int]],
) -> None:
    require_v3_schema(timing_pairs, "timing_pairs")
    invocations = timing_pairs.get("invocations")
    require(isinstance(invocations, list) and len(invocations) == INVOCATIONS,
            "timing_pairs.invocations: expected five entries")
    entries: dict[int, dict[str, Any]] = {}
    for entry in invocations:
        require(isinstance(entry, dict), "timing_pairs invocation must be object")
        invocation = as_int(entry.get("invocation"), "timing_pairs.invocation", minimum=1)
        require(invocation <= INVOCATIONS and invocation not in entries, "timing_pairs invocation set invalid")
        entries[invocation] = entry
    require(set(entries) == set(range(1, INVOCATIONS + 1)), "timing_pairs invocation set must be 1..5")

    invocation_medians: list[float] = []
    for invocation in range(1, INVOCATIONS + 1):
        entry = entries[invocation]
        child = child_data[invocation]
        child_result_path = child["result_path"]
        actual_sha = sha256_file(child_result_path)
        require(entry.get("source_result_sha256") == actual_sha,
                f"timing_pairs {invocation}: source result SHA mismatch")
        raw_record = entry.get("raw_result")
        require(isinstance(raw_record, dict), f"timing_pairs {invocation}: raw_result record missing")
        artifact_record(
            root,
            raw_record,
            f"timing_pairs {invocation}.raw_result",
            expected_path=child_result_path,
        )
        pairs = entry.get("pairs")
        require(isinstance(pairs, list) and len(pairs) == REPS,
                f"timing_pairs {invocation}: expected {REPS} pairs")
        seen_reps: set[int] = set()
        ratios: list[float] = []
        for pair in pairs:
            require(isinstance(pair, dict), f"timing_pairs {invocation}: pair must be object")
            rep = as_int(pair.get("rep"), f"timing_pairs {invocation}.rep", minimum=0)
            require(rep < REPS and rep not in seen_reps, f"timing_pairs {invocation}: bad/duplicate rep")
            seen_reps.add(rep)
            baseline = as_number(pair.get("baseline_gpu_ms"), "pair baseline", positive=True)
            static = as_number(pair.get("static_aabb_gpu_ms"), "pair static", positive=True)
            same_number(baseline, child["arrays"]["baseline"][rep],
                        f"timing_pairs {invocation} baseline[{rep}]")
            same_number(static, child["arrays"]["static_aabb"][rep],
                        f"timing_pairs {invocation} static_aabb[{rep}]")
            expected_order = (
                "baseline_then_static_aabb"
                if child["timeline"]["orders"][rep][0] == "baseline"
                else "static_aabb_then_baseline"
            )
            require(pair.get("order") == expected_order,
                    f"timing_pairs {invocation}: order mismatch at rep {rep}")
            ratio = static / baseline
            same_number(pair.get("ratio_static_over_baseline"), ratio,
                        f"timing_pairs {invocation} ratio[{rep}]")
            if "static_over_baseline" in pair:
                same_number(pair["static_over_baseline"], ratio,
                            f"timing_pairs {invocation} static_over_baseline[{rep}]")
            ratios.append(ratio)
        require(seen_reps == set(range(REPS)), f"timing_pairs {invocation}: incomplete reps")
        median = statistics.median(ratios)
        same_number(entry.get("median_ratio_static_over_baseline"), median,
                    f"timing_pairs {invocation}: median")
        invocation_medians.append(median)

    same_number(
        timing_pairs.get("aggregate_median_of_invocation_medians"),
        statistics.median(invocation_medians),
        "timing_pairs aggregate median",
    )
    range_value = timing_pairs.get("range_of_invocation_medians")
    require(isinstance(range_value, list) and len(range_value) == 2,
            "timing_pairs range_of_invocation_medians invalid")
    same_number(range_value[0], min(invocation_medians), "timing_pairs range min")
    same_number(range_value[1], max(invocation_medians), "timing_pairs range max")

    # The manifest must include the timing source records already checked.
    for invocation in range(1, INVOCATIONS + 1):
        expected_relative = f"invocation_{invocation:02d}/result.json"
        require_manifest_path(root, manifest_index, expected_relative)


def parse_csv_telemetry(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        fail(f"telemetry CSV missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "telemetry CSV lacks header")
        require(set(REQUIRED_CSV_COLUMNS).issubset(set(reader.fieldnames)),
                "telemetry CSV missing required columns")
        rows = list(reader)
    require(rows, "telemetry CSV has no samples")
    parsed: list[dict[str, Any]] = []
    previous_mono: int | None = None
    for expected_seq, row in enumerate(rows):
        require(isinstance(row, dict), "telemetry CSV row invalid")
        sequence = as_int(int(row.get("sample_seq", "")), "telemetry sample_seq", minimum=0)
        require(sequence == expected_seq, "telemetry sample_seq must be contiguous from zero")
        mono = as_int(int(row.get("monotonic_ns", "")), "telemetry monotonic_ns", minimum=0)
        if previous_mono is not None:
            require(mono > previous_mono, "telemetry monotonic_ns must be strictly increasing")
        previous_mono = mono
        for key in NUMERIC_CSV_COLUMNS:
            as_number(row.get(key), f"telemetry.{key}", positive=(key in CLOCK_COLUMNS))
        require(str(row.get("pstate", "")).strip(), "telemetry pstate missing")
        require(str(row.get("phase", "")).strip(), "telemetry phase missing")
        parsed.append({"raw": row, "sample_seq": sequence, "monotonic_ns": mono})
    return parsed


def parse_process_samples(
    path: Path,
    rows: list[dict[str, Any]],
    allowed_pids: set[int],
) -> tuple[dict[int, list[int]], list[dict[str, Any]]]:
    """Parse process snapshots and return both PID times and aligned full events."""
    if not path.is_file() or path.is_symlink():
        fail(f"process sample JSONL missing: {path}")
    parsed: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"process sample line {line_number}: invalid JSON: {exc}")
            require(isinstance(entry, dict), f"process sample line {line_number}: expected object")
            parsed.append(entry)
    require(len(parsed) == len(rows), "process sample count must equal telemetry sample count")
    target_times: dict[int, list[int]] = {pid: [] for pid in allowed_pids}
    events: list[dict[str, Any]] = []
    for row, entry in zip(rows, parsed):
        seq = as_int(entry.get("sample_seq"), "process sample_seq", minimum=0)
        mono = as_int(entry.get("monotonic_ns"), "process monotonic_ns", minimum=0)
        require(
            seq == row["sample_seq"] and mono == row["monotonic_ns"],
            "process sample does not align with telemetry row",
        )
        phase = entry.get("phase")
        require(isinstance(phase, str) and phase, "process sample phase missing")
        require(phase == row["raw"].get("phase"), "process sample phase differs from telemetry CSV")
        processes = entry.get("compute_processes")
        require(isinstance(processes, list), "process sample compute_processes missing")
        pids: list[int] = []
        for process in processes:
            require(isinstance(process, dict), "process sample process entry invalid")
            pids.append(as_int(process.get("pid"), "compute process pid", minimum=1))
        require(len(set(pids)) == len(pids), "process sample has duplicate PID")
        foreign = set(pids) - allowed_pids
        require(not foreign, f"foreign compute PID(s) observed: {sorted(foreign)}")
        declared = entry.get("target_pid")
        target_pid: int | None
        if declared is not None:
            target_pid = as_int(declared, "process target_pid", minimum=1)
            require(target_pid in allowed_pids, f"process target_pid unknown: {target_pid}")
            require(
                pids == [target_pid],
                "process target_pid must be the sole observed compute PID",
            )
            target_times[target_pid].append(mono)
        else:
            target_pid = None
            require(not pids, "compute PID observed while target_pid is null")
        events.append(
            {
                "sample_seq": seq,
                "monotonic_ns": mono,
                "phase": phase,
                "target_pid": target_pid,
                "pids": pids,
            }
        )
    return target_times, events


def stats_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in NUMERIC_CSV_COLUMNS:
        values = [as_number(row["raw"][key], "telemetry stat") for row in rows]
        result[key] = {
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
        }
    return result


def compare_telemetry_summary(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    target_times: dict[int, list[int]],
    allowed_pids: set[int],
    max_gap_ms: float,
) -> None:
    sample_times = [row["monotonic_ns"] for row in rows]
    gaps_ms = [(right - left) / 1_000_000.0 for left, right in zip(sample_times, sample_times[1:])]
    actual_max_gap = max(gaps_ms) if gaps_ms else 0.0
    require(summary.get("sample_count") == len(rows), "telemetry summary sample_count mismatch")
    require(summary.get("first_monotonic_ns") == sample_times[0], "telemetry summary first timestamp mismatch")
    require(summary.get("last_monotonic_ns") == sample_times[-1], "telemetry summary last timestamp mismatch")
    same_number(summary.get("max_inter_sample_gap_ms"), actual_max_gap,
                "telemetry summary max inter-sample gap")
    require(actual_max_gap <= max_gap_ms,
            f"telemetry max inter-sample gap {actual_max_gap:.3f} ms exceeds {max_gap_ms:.3f} ms")
    require(summary.get("foreign_compute_pids") == [], "telemetry summary reports foreign PID")
    require(summary.get("query_errors") == [], "telemetry summary reports query errors")
    require(summary.get("numeric_field_complete") is True, "telemetry numeric fields incomplete")
    observed = summary.get("observed_target_pids")
    require(isinstance(observed, list) and sorted(observed) == sorted(allowed_pids),
            "telemetry observed target PID set mismatch")
    summary_target_times = summary.get("target_sample_times")
    require(isinstance(summary_target_times, dict), "telemetry target_sample_times missing")
    for pid in sorted(allowed_pids):
        recorded = summary_target_times.get(str(pid))
        require(recorded == target_times[pid], f"telemetry target samples mismatch for PID {pid}")
    stats = summary.get("stats")
    require(isinstance(stats, dict), "telemetry summary stats missing")
    actual_stats = stats_from_rows(rows)
    for key, expected in actual_stats.items():
        reported = stats.get(key)
        require(isinstance(reported, dict), f"telemetry stats missing {key}")
        for stat_name, stat_value in expected.items():
            same_number(reported.get(stat_name), stat_value, f"telemetry stats {key}.{stat_name}")
    pstates = sorted({str(row["raw"]["pstate"]) for row in rows if str(row["raw"]["pstate"])})
    require(summary.get("pstates") == pstates, "telemetry pstates mismatch")


def check_timeline_telemetry_coverage(
    child_data: dict[int, dict[str, Any]],
    target_times: dict[int, list[int]],
    events: list[dict[str, Any]],
    edge_gap_ms: float,
) -> None:
    """Require continuous, PID-exact process telemetry throughout each envelope."""
    edge_ns = int(edge_gap_ms * 1_000_000.0)
    for invocation, child in child_data.items():
        timeline = child["timeline"]
        pid = timeline["pid"]
        start = timeline["start_ns"]
        end = timeline["end_ns"]
        envelope_events = [
            event
            for event in events
            if start <= event["monotonic_ns"] <= end
        ]
        require(
            envelope_events,
            f"invocation {invocation}: no process telemetry sample inside timed envelope",
        )
        for event in envelope_events:
            require(
                event["target_pid"] == pid and event["pids"] == [pid],
                (
                    f"invocation {invocation}: envelope sample {event['sample_seq']} "
                    "is not exclusively bound to its benchmark PID"
                ),
            )
        visible_times = [event["monotonic_ns"] for event in envelope_events]
        require(
            visible_times
            == [item for item in target_times.get(pid, []) if start <= item <= end],
            f"invocation {invocation}: target-time accounting mismatch",
        )
        # Global telemetry cadence is audited separately.  This checks the
        # stronger condition requested for target-visible samples specifically.
        target_gaps = [
            (right - left) / 1_000_000.0
            for left, right in zip(visible_times, visible_times[1:])
        ]
        require(
            all(gap <= edge_gap_ms for gap in target_gaps),
            f"invocation {invocation}: target-visible sample gap exceeds {edge_gap_ms:.3f} ms",
        )
        require(
            visible_times[0] <= start + edge_ns,
            f"invocation {invocation}: telemetry begins too late for timed envelope",
        )
        require(
            visible_times[-1] >= end - edge_ns,
            f"invocation {invocation}: telemetry ends too early for timed envelope",
        )


def launch_boundary(entry: dict[str, Any], names: tuple[str, ...], label: str) -> int:
    value = nested_first(entry, names)
    require(value is not None, f"{label}: launch boundary missing ({', '.join(names)})")
    return as_int(value, label, minimum=0)


def check_process_lifecycle_guards(
    launches: dict[int, dict[str, Any]],
    child_data: dict[int, dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    """Prove a clean sampler before launch and after each child exits."""
    boundaries: dict[int, tuple[int, int]] = {}
    for invocation in range(1, INVOCATIONS + 1):
        entry = launches[invocation]
        start = launch_boundary(
            entry,
            ("target_start_monotonic_ns", "process_start_monotonic_ns", "child_start_monotonic_ns"),
            f"launch {invocation} start",
        )
        end = launch_boundary(
            entry,
            ("target_exit_monotonic_ns", "process_exit_monotonic_ns", "child_exit_monotonic_ns"),
            f"launch {invocation} exit",
        )
        require(end > start, f"launch {invocation}: exit must be after start")
        timeline = child_data[invocation]["timeline"]
        require(start <= timeline["start_ns"] <= timeline["end_ns"] <= end,
                f"launch {invocation}: timeline envelope is outside process lifetime")
        boundaries[invocation] = (start, end)

    first_start = boundaries[1][0]
    before_first = [event for event in events if event["monotonic_ns"] < first_start]
    require(
        len(before_first) >= 3,
        "fewer than three telemetry samples before first target launch",
    )
    require(
        all(event["target_pid"] is None and event["pids"] == [] for event in before_first),
        "a target/compute PID appears before first target launch",
    )
    prelaunch_empty = [
        event
        for event in before_first
        if event["phase"] == "prelaunch"
        and event["target_pid"] is None
        and event["pids"] == []
    ]
    require(
        len(prelaunch_empty) >= 3,
        "need at least three empty phase=prelaunch telemetry samples",
    )

    final_sample = events[-1]["monotonic_ns"]
    for invocation in range(1, INVOCATIONS + 1):
        exit_ns = boundaries[invocation][1]
        next_start = (
            boundaries[invocation + 1][0]
            if invocation < INVOCATIONS
            else final_sample + 1
        )
        require(next_start > exit_ns, f"launch {invocation}: no post-exit sampling interval")
        postexit = [
            event
            for event in events
            if exit_ns <= event["monotonic_ns"] < next_start
        ]
        require(
            len(postexit) >= 3,
            f"launch {invocation}: fewer than three samples after child exit",
        )
        require(
            all(event["target_pid"] is None and event["pids"] == [] for event in postexit),
            f"launch {invocation}: nonempty process sample after child exit",
        )


def verify_environment_and_gpu(
    environment: dict[str, Any],
    launch: dict[str, Any],
    telemetry_rows: list[dict[str, Any]],
) -> tuple[str, int]:
    require_v3_schema(environment, "environment")
    target = environment.get("target_gpu")
    require(isinstance(target, dict), "environment.target_gpu missing")
    gpu_uuid = target.get("uuid")
    require(isinstance(gpu_uuid, str) and gpu_uuid, "environment target UUID missing")
    gpu_index = as_int(int(str(target.get("index"))), "environment target index", minimum=0)
    require(environment.get("cuda_visible_devices") == gpu_uuid, "environment CUDA_VISIBLE_DEVICES mismatch")
    controls = environment.get("clock_control")
    require(isinstance(controls, dict), "environment clock_control missing")
    for key in (
        "runner_attempted_clock_lock",
        "runner_attempted_application_clocks",
        "runner_attempted_power_limit_change",
        "runner_attempted_compute_mode_change",
    ):
        require(controls.get(key) is False, f"environment {key} must be false")
    launch_gpu = launch.get("target_gpu")
    if isinstance(launch_gpu, dict):
        require(launch_gpu.get("uuid") == gpu_uuid, "launch target GPU UUID mismatch")
    require(launch.get("cuda_visible_devices") == gpu_uuid, "launch CUDA_VISIBLE_DEVICES mismatch")
    for row in telemetry_rows:
        require(row["raw"].get("gpu_uuid") == gpu_uuid, "telemetry GPU UUID mismatch")
        require(int(row["raw"].get("gpu_index", "")) == gpu_index, "telemetry GPU index mismatch")
    return gpu_uuid, gpu_index


def equivalent_json(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def collect_record_hashes(document: Any, prefix: str = "") -> dict[str, str]:
    """Collect SHA values in nested artifact records without dereferencing them."""
    output: dict[str, str] = {}
    if isinstance(document, dict):
        if all(key in document for key in ("path", "bytes", "sha256")) and is_sha256(document["sha256"]):
            output[prefix.rstrip(".")] = document["sha256"].lower()
        for key, value in document.items():
            next_prefix = f"{prefix}{key}."
            output.update(collect_record_hashes(value, next_prefix))
    elif isinstance(document, list):
        for index, value in enumerate(document):
            output.update(collect_record_hashes(value, f"{prefix}{index}."))
    return output


def verify_expected_hashes(
    root: Path,
    specs: list[str],
    manifest_index: dict[str, tuple[Path, str, int]],
    terminal: dict[str, Any],
    manifest: dict[str, Any],
    result: dict[str, Any],
    launch: dict[str, Any],
) -> list[dict[str, str]]:
    available: dict[str, str] = {}
    aliases = {
        "terminal": root / "terminal.json",
        "manifest": root / "manifest.json",
        "result": root / "result.json",
        "launch": root / "launch.json",
        "timing_pairs": root / "timing_pairs.json",
        "telemetry_summary": root / "telemetry_summary.json",
        "telemetry": root / "gpu_telemetry.csv",
        "process_samples": root / "gpu_process_samples.jsonl",
        "environment": root / "environment.json",
    }
    for alias, path in aliases.items():
        if path.is_file() and not path.is_symlink():
            available[alias] = sha256_file(path)
    for relative, (_, digest, _) in manifest_index.items():
        available[relative] = digest
    records: dict[str, str] = {}
    for prefix, document in (
        ("terminal.", terminal),
        ("manifest.", manifest),
        ("result.", result),
        ("launch.", launch),
    ):
        records.update(collect_record_hashes(document, prefix))
    # Helpful shorthand for common provenance objects declared by launch.
    for short_name in ("binary", "source", "protocol", "heldout_ids", "ids", "runner", "runner_freeze"):
        if f"launch.{short_name}" in records:
            available[short_name] = records[f"launch.{short_name}"]
    available.update(records)

    checked: list[dict[str, str]] = []
    for spec in specs:
        require("=" in spec, f"--expected-hash must be NAME=SHA256, got {spec!r}")
        name, expected = spec.split("=", 1)
        require(name and is_sha256(expected), f"invalid expected hash specification: {spec!r}")
        actual = available.get(name)
        require(actual is not None, f"expected hash target unavailable: {name}")
        require(actual == expected.lower(), f"expected hash mismatch for {name}")
        checked.append({"name": name, "sha256": actual})
    return checked


def audit_run(
    run_dir: Path,
    *,
    expected_hashes: list[str],
    allowed_statuses: set[str],
    max_gap_ms: float,
    max_window_edge_gap_ms: float,
) -> dict[str, Any]:
    root = canonical_root(run_dir)
    checks: list[str] = []
    try:
        require(max_gap_ms > 0.0 and max_window_edge_gap_ms > 0.0, "gap thresholds must be positive")
        terminal_path = root / "terminal.json"
        manifest_path = root / "manifest.json"
        result_path = root / "result.json"
        launch_path = root / "launch.json"
        environment_path = root / "environment.json"
        timing_pairs_path = root / "timing_pairs.json"
        telemetry_summary_path = root / "telemetry_summary.json"
        telemetry_csv_path = root / "gpu_telemetry.csv"
        process_jsonl_path = root / "gpu_process_samples.jsonl"

        terminal = load_json(terminal_path, "terminal")
        manifest = load_json(manifest_path, "manifest")
        result = load_json(result_path, "result")
        launch = load_json(launch_path, "launch")
        environment = load_json(environment_path, "environment")
        timing_pairs = load_json(timing_pairs_path, "timing_pairs")
        telemetry_summary = load_json(telemetry_summary_path, "telemetry_summary")

        for label, document in (("terminal", terminal), ("manifest", manifest), ("result", result)):
            require_v3_schema(document, label)
            require(document.get("status") in allowed_statuses,
                    f"{label}: non-complete or unexpected status {document.get('status')!r}")
            require(document.get("formal_claim_eligible") is False,
                    f"{label}: formal_claim_eligible must remain false")
        require(terminal.get("exit_code") == 0, "terminal: exit_code must be zero")
        require(isinstance(terminal.get("batch_id"), str) and terminal["batch_id"], "terminal batch_id missing")
        if "batch_id" in result:
            require(result["batch_id"] == terminal["batch_id"], "result batch_id mismatch")
        artifact_record(root, terminal.get("manifest"), "terminal.manifest", expected_path=manifest_path)
        artifact_record(root, terminal.get("result"), "terminal.result", expected_path=result_path)
        checks.append("complete v3 terminal/result/manifest state")

        manifest_index = build_manifest_index(root, manifest)
        for relative in (
            "result.json",
            "launch.json",
            "environment.json",
            "timing_pairs.json",
            "telemetry_summary.json",
            "gpu_telemetry.csv",
            "gpu_process_samples.jsonl",
        ):
            require_manifest_path(root, manifest_index, relative)
        checks.append("manifest artifact hashes and byte counts")

        launches = parse_launches(launch)
        children = normalize_children(result)
        child_data: dict[int, dict[str, Any]] = {}
        all_tokens: set[str] = set()
        all_pids: set[int] = set()
        for invocation in range(1, INVOCATIONS + 1):
            child_entry = children[invocation]
            expected_result_path = root / f"invocation_{invocation:02d}" / "result.json"
            result_record = result_artifact_from_child(child_entry, f"invocation {invocation}")
            child_result_path, child_result_sha, _ = artifact_record(
                root,
                result_record,
                f"result child {invocation}",
                expected_path=expected_result_path,
            )
            require_manifest_path(root, manifest_index, f"invocation_{invocation:02d}/result.json")
            child_result = load_json(child_result_path, f"child result {invocation}")
            arrays = child_timing_arrays(child_result, invocation)

            binding = timeline_binding_for(result, child_entry, invocation)
            timeline_record = artifact_from_binding(binding, f"invocation {invocation}")
            expected_timeline_path = root / f"invocation_{invocation:02d}" / "benchmark_timeline.json"
            timeline_path, _, _ = artifact_record(
                root,
                timeline_record,
                f"result timeline {invocation}",
                expected_path=expected_timeline_path,
            )
            require_manifest_path(root, manifest_index, f"invocation_{invocation:02d}/benchmark_timeline.json")
            timeline = parse_timeline(timeline_path, invocation)
            bound_token = binding_value(binding, ("run_token", "timeline_token", "benchmark_token"),
                                        f"invocation {invocation}")
            bound_pid = binding_value(binding, ("pid", "target_pid"), f"invocation {invocation}")
            require(bound_token == timeline["token"], f"invocation {invocation}: aggregate token mismatch")
            require(as_int(bound_pid, f"invocation {invocation} aggregate PID", minimum=1) == timeline["pid"],
                    f"invocation {invocation}: aggregate PID mismatch")
            launch_token = nested_first(launches[invocation], ("run_token", "timeline_token", "benchmark_token"))
            require(launch_token == timeline["token"], f"invocation {invocation}: launch token mismatch")
            require(as_int(launches[invocation]["pid"], "launch PID", minimum=1) == timeline["pid"],
                    f"invocation {invocation}: launch PID mismatch")
            launch_start_ns = launch_boundary(
                launches[invocation],
                ("target_start_monotonic_ns", "process_start_monotonic_ns", "child_start_monotonic_ns"),
                f"launch {invocation} start",
            )
            launch_exit_ns = launch_boundary(
                launches[invocation],
                ("target_exit_monotonic_ns", "process_exit_monotonic_ns", "child_exit_monotonic_ns"),
                f"launch {invocation} exit",
            )
            require(
                launch_start_ns <= timeline["start_ns"] <= timeline["end_ns"] <= launch_exit_ns,
                f"invocation {invocation}: timeline envelope outside launch process lifetime",
            )
            require(timeline["token"] not in all_tokens, "timeline run tokens must be unique")
            require(timeline["pid"] not in all_pids, "timeline PIDs must be unique")
            all_tokens.add(timeline["token"])
            all_pids.add(timeline["pid"])
            compare_timeline_to_child(timeline, arrays, invocation)
            child_data[invocation] = {
                "result_path": child_result_path,
                "result_sha": child_result_sha,
                "result": child_result,
                "arrays": arrays,
                "timeline": timeline,
                "launch_start_ns": launch_start_ns,
                "launch_exit_ns": launch_exit_ns,
            }
        checks.append("five child gates, timeline tokens/PIDs, and 5x30 CUDA timings")

        parse_timing_pairs(root, timing_pairs, child_data, manifest_index)
        aggregate_pairs = result.get("timing_pairs")
        require(aggregate_pairs is not None, "aggregate result timing_pairs binding missing")
        if isinstance(aggregate_pairs, dict) and all(
            key in aggregate_pairs for key in ("path", "bytes", "sha256")
        ):
            artifact_record(root, aggregate_pairs, "aggregate timing_pairs artifact", expected_path=timing_pairs_path)
        else:
            require(equivalent_json(aggregate_pairs, timing_pairs),
                    "aggregate result timing_pairs differs from timing_pairs.json")
        checks.append("five-by-thirty paired timing aggregation")

        telemetry_rows = parse_csv_telemetry(telemetry_csv_path)
        verify_environment_and_gpu(environment, launch, telemetry_rows)
        target_times, process_events = parse_process_samples(
            process_jsonl_path,
            telemetry_rows,
            all_pids,
        )
        compare_telemetry_summary(
            telemetry_summary,
            telemetry_rows,
            target_times,
            all_pids,
            max_gap_ms,
        )
        check_timeline_telemetry_coverage(
            child_data,
            target_times,
            process_events,
            max_window_edge_gap_ms,
        )
        check_process_lifecycle_guards(launches, child_data, process_events)
        checks.append(
            "telemetry clock fields, strict envelope PID coverage, and clean prelaunch/postexit sampling"
        )

        expected_checked = verify_expected_hashes(
            root,
            expected_hashes,
            manifest_index,
            terminal,
            manifest,
            result,
            launch,
        )
        if expected_checked:
            checks.append("caller-supplied expected hashes")

        return {
            "schema": AUDIT_SCHEMA,
            "status": "PASS",
            "scope": (
                "CPU-only read-only completed-run verification; no external command, "
                "GPU runtime, or GPU control operation is used"
            ),
            "run_dir": str(root),
            "formal_claim_eligible": False,
            "gpu_binary_executed": False,
            "nvidia_smi_called": False,
            "checks": checks,
            "expected_hashes_checked": expected_checked,
            "measurement_contract": {
                "independent_invocations": INVOCATIONS,
                "warmups_per_invocation": WARMUPS,
                "paired_repetitions_per_invocation": REPS,
                "timed_operations_per_invocation": OPS_PER_INVOCATION,
                "max_inter_sample_gap_ms": max_gap_ms,
                "max_timeline_edge_gap_ms": max_window_edge_gap_ms,
            },
        }
    except AuditFailure as exc:
        return {
            "schema": AUDIT_SCHEMA,
            "status": "FAIL",
            "scope": "CPU-only read-only completed-run verification",
            "run_dir": str(root),
            "formal_claim_eligible": False,
            "gpu_binary_executed": False,
            "nvidia_smi_called": False,
            "checks_completed": checks,
            "failure": str(exc),
        }
    except Exception as exc:  # Defensive fail closed for malformed external artifacts.
        return {
            "schema": AUDIT_SCHEMA,
            "status": "FAIL",
            "scope": "CPU-only read-only completed-run verification",
            "run_dir": str(root),
            "formal_claim_eligible": False,
            "gpu_binary_executed": False,
            "nvidia_smi_called": False,
            "checks_completed": checks,
            "failure": f"unexpected verifier exception: {type(exc).__name__}: {exc}",
        }


def build_synthetic_run(root: Path) -> None:
    """Create a disposable positive fixture for --selftest only."""
    gpu_uuid = "GPU-synthetic-v3-0000"
    gpu_index = 0
    base_time = 1_000_000_000_000
    pids = {invocation: 41000 + invocation for invocation in range(1, INVOCATIONS + 1)}
    tokens = {invocation: f"synthetic-run-token-{invocation}" for invocation in range(1, INVOCATIONS + 1)}
    child_arrays: dict[int, dict[str, list[float]]] = {}
    timelines: dict[int, dict[str, Any]] = {}
    process_bounds: dict[int, tuple[int, int]] = {}

    for invocation in range(1, INVOCATIONS + 1):
        # Two seconds leave an explicit clean post-exit sampling segment.
        start = base_time + (invocation - 1) * 2_000_000_000
        end = start + 600_000_000
        process_bounds[invocation] = (start - 300_000_000, end + 100_000_000)
        operations: list[dict[str, Any]] = []
        arrays = {"baseline": [], "static_aabb": []}
        for rep in range(REPS):
            methods = ("baseline", "static_aabb") if rep % 2 == 0 else ("static_aabb", "baseline")
            for order, method in enumerate(methods):
                op_index = 2 * rep + order
                host_start = start + 10_000_000 + op_index * 9_000_000
                host_end = host_start + 8_000_000
                cuda_ms = (1.0 if method == "baseline" else 0.75) + invocation * 0.01 + rep * 0.0001
                arrays[method].append(cuda_ms)
                operations.append(
                    {
                        "op_index": op_index,
                        "rep": rep,
                        "method": method,
                        "order": order,
                        "host_start_ns": host_start,
                        "host_end_ns": host_end,
                        "cuda_ms": cuda_ms,
                    }
                )
        timeline = {
            "schema": "safe-c2-static-aabb-benchmark-timeline-v3",
            "status": "PASS_SYNTHETIC_TIMELINE_V3",
            "run_token": tokens[invocation],
            "pid": pids[invocation],
            "clock_id": "CLOCK_MONOTONIC",
            "timed_envelope_start_ns": start,
            "timed_envelope_end_ns": end,
            "warmups": WARMUPS,
            "reps": REPS,
            "operations": operations,
        }
        timeline_path = root / f"invocation_{invocation:02d}" / "benchmark_timeline.json"
        write_json(timeline_path, timeline)
        child_result = {
            "schema": "safe-c2-static-aabb-perf-child-v3",
            "status": "PASS_SYNTHETIC_CHILD_V3",
            "unmeasured_gate": {
                "baseline": {key: 0 for key in CHILD_GATE_COUNTERS},
                "static_aabb": {key: 0 for key in CHILD_GATE_COUNTERS},
                "id_sets_equal": True,
            },
            "snapshot_pre_post_byte_equal": True,
            "timing_gpu_ms": arrays,
        }
        write_json(root / f"invocation_{invocation:02d}" / "result.json", child_result)
        child_arrays[invocation] = arrays
        timelines[invocation] = timeline

    pair_entries: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    for invocation in range(1, INVOCATIONS + 1):
        pairs: list[dict[str, Any]] = []
        for rep in range(REPS):
            baseline = child_arrays[invocation]["baseline"][rep]
            static = child_arrays[invocation]["static_aabb"][rep]
            ratio = static / baseline
            pairs.append(
                {
                    "rep": rep,
                    "baseline_gpu_ms": baseline,
                    "static_aabb_gpu_ms": static,
                    "ratio_static_over_baseline": ratio,
                    "static_over_baseline": ratio,
                    "order": (
                        "baseline_then_static_aabb"
                        if rep % 2 == 0
                        else "static_aabb_then_baseline"
                    ),
                }
            )
        child_result_path = root / f"invocation_{invocation:02d}" / "result.json"
        timeline_path = root / f"invocation_{invocation:02d}" / "benchmark_timeline.json"
        pair_entries.append(
            {
                "invocation": invocation,
                "source_result_sha256": sha256_file(child_result_path),
                "raw_result": art_for(root, child_result_path),
                "pairs": pairs,
                "median_ratio_static_over_baseline": statistics.median(
                    [entry["ratio_static_over_baseline"] for entry in pairs]
                ),
            }
        )
        children.append(
            {
                "invocation": invocation,
                "result": art_for(root, child_result_path),
                "timeline": {
                    **art_for(root, timeline_path),
                    "run_token": tokens[invocation],
                    "pid": pids[invocation],
                    "timed_envelope_start_ns": timelines[invocation]["timed_envelope_start_ns"],
                    "timed_envelope_end_ns": timelines[invocation]["timed_envelope_end_ns"],
                },
            }
        )
    medians = [entry["median_ratio_static_over_baseline"] for entry in pair_entries]
    timing_pairs = {
        "schema": "safe-c2-static-aabb-heldout-perf-timing-pairs-v3",
        "status": DEFAULT_COMPLETE_STATUS,
        "formal_claim_eligible": False,
        "invocations": pair_entries,
        "aggregate_median_of_invocation_medians": statistics.median(medians),
        "range_of_invocation_medians": [min(medians), max(medians)],
    }
    write_json(root / "timing_pairs.json", timing_pairs)

    launches = [
        {
            "invocation": invocation,
            "pid": pids[invocation],
            "run_token": tokens[invocation],
            "target_start_monotonic_ns": process_bounds[invocation][0],
            "target_exit_monotonic_ns": process_bounds[invocation][1],
        }
        for invocation in range(1, INVOCATIONS + 1)
    ]
    launch = {
        "schema": "safe-c2-static-aabb-heldout-perf-launch-v3",
        "cuda_visible_devices": gpu_uuid,
        "target_gpu": {"uuid": gpu_uuid, "index": str(gpu_index)},
        "launches": launches,
    }
    write_json(root / "launch.json", launch)
    environment = {
        "schema": "safe-c2-static-aabb-heldout-perf-environment-v3",
        "cuda_visible_devices": gpu_uuid,
        "target_gpu": {"uuid": gpu_uuid, "index": str(gpu_index)},
        "clock_control": {
            "runner_attempted_clock_lock": False,
            "runner_attempted_application_clocks": False,
            "runner_attempted_power_limit_change": False,
            "runner_attempted_compute_mode_change": False,
        },
    }
    write_json(root / "environment.json", environment)
    write_json(root / "preflight.json", {"schema": "synthetic-v3", "status": "PASS"})

    csv_path = root / "gpu_telemetry.csv"
    process_path = root / "gpu_process_samples.jsonl"
    sample_times: list[int] = []
    first_sample = base_time - 600_000_000
    final_sample = timelines[INVOCATIONS]["timed_envelope_end_ns"] + 500_000_000
    with csv_path.open("w", encoding="utf-8", newline="") as csv_handle, process_path.open(
        "w", encoding="utf-8"
    ) as process_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=list(REQUIRED_CSV_COLUMNS))
        writer.writeheader()
        sample_seq = 0
        current = first_sample
        while current <= final_sample:
            active = None
            for invocation in range(1, INVOCATIONS + 1):
                process_start, process_exit = process_bounds[invocation]
                if process_start <= current < process_exit:
                    active = invocation
                    break
            phase = (
                "target"
                if active is not None
                else ("prelaunch" if current < process_bounds[1][0] else "between_invocations")
            )
            writer.writerow(
                {
                    "wall_utc_ns": str(current + 5_000_000),
                    "monotonic_ns": str(current),
                    "sample_seq": str(sample_seq),
                    "phase": phase,
                    "gpu_index": str(gpu_index),
                    "gpu_uuid": gpu_uuid,
                    "pstate": "P0",
                    "clock_sm_mhz": "2000.0",
                    "clock_graphics_mhz": "2000.0",
                    "clock_memory_mhz": "9000.0",
                    "power_draw_w": "250.0",
                    "power_limit_w": "600.0",
                    "temperature_gpu_c": "50.0",
                    "utilization_gpu_pct": "90.0",
                    "utilization_memory_pct": "40.0",
                    "memory_used_mib": "1000.0",
                }
            )
            processes = [] if active is None else [{"pid": pids[active], "process_name": "synthetic"}]
            process_handle.write(
                json.dumps(
                    {
                        "sample_seq": sample_seq,
                        "monotonic_ns": current,
                        "phase": phase,
                        "target_pid": None if active is None else pids[active],
                        "compute_processes": processes,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            sample_times.append(current)
            sample_seq += 1
            current += 100_000_000

    rows = parse_csv_telemetry(csv_path)
    target_times: dict[int, list[int]] = {pid: [] for pid in pids.values()}
    with process_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if entry["target_pid"] is not None:
                target_times[entry["target_pid"]].append(entry["monotonic_ns"])
    stats = stats_from_rows(rows)
    gaps = [(b - a) / 1_000_000.0 for a, b in zip(sample_times, sample_times[1:])]
    telemetry_summary = {
        "schema": "safe-c2-static-aabb-heldout-perf-telemetry-summary-v3",
        "status": DEFAULT_COMPLETE_STATUS,
        "formal_claim_eligible": False,
        "sample_count": len(rows),
        "first_monotonic_ns": sample_times[0],
        "last_monotonic_ns": sample_times[-1],
        "max_inter_sample_gap_ms": max(gaps),
        "pstates": ["P0"],
        "observed_target_pids": sorted(pids.values()),
        "target_sample_times": {str(pid): times for pid, times in target_times.items()},
        "foreign_compute_pids": [],
        "query_errors": [],
        "numeric_field_complete": True,
        "stats": stats,
    }
    write_json(root / "telemetry_summary.json", telemetry_summary)
    result = {
        "schema": "safe-c2-static-aabb-heldout-perf-evidence-result-v3",
        "status": DEFAULT_COMPLETE_STATUS,
        "formal_claim_eligible": False,
        "batch_id": "synthetic-v3-batch",
        "children": children,
        "timing_pairs": timing_pairs,
    }
    write_json(root / "result.json", result)
    write_json(root / "stdout.log", {"synthetic": True})
    write_json(root / "stderr.log", {"synthetic": False})

    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in ("manifest.json", "terminal.json"):
            artifacts[str(path.relative_to(root))] = art_for(root, path)
    manifest = {
        "schema": "safe-c2-static-aabb-heldout-perf-manifest-v3",
        "status": DEFAULT_COMPLETE_STATUS,
        "formal_claim_eligible": False,
        "artifacts": artifacts,
    }
    write_json(root / "manifest.json", manifest)
    terminal = {
        "schema": "safe-c2-static-aabb-heldout-perf-terminal-v3",
        "status": DEFAULT_COMPLETE_STATUS,
        "formal_claim_eligible": False,
        "batch_id": "synthetic-v3-batch",
        "exit_code": 0,
        "manifest": art_for(root, root / "manifest.json"),
        "result": art_for(root, root / "result.json"),
    }
    write_json(root / "terminal.json", terminal)


def selftest() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="safe_c2_perf_audit_v3_") as directory:
        root = Path(directory)
        build_synthetic_run(root)
        report = audit_run(
            root,
            expected_hashes=[f"manifest={sha256_file(root / 'manifest.json')}"],
            allowed_statuses={DEFAULT_COMPLETE_STATUS},
            max_gap_ms=DEFAULT_MAX_GAP_MS,
            max_window_edge_gap_ms=DEFAULT_MAX_GAP_MS,
        )
        require(report.get("status") == "PASS", "synthetic positive fixture failed: " + str(report))
        # A one-byte semantic corruption must fail closed.  It is inside the
        # disposable fixture, never in a real run directory.
        timeline_path = root / "invocation_01" / "benchmark_timeline.json"
        broken = load_json(timeline_path, "synthetic timeline")
        broken["operations"][0]["cuda_ms"] = broken["operations"][0]["cuda_ms"] + 1.0
        write_json(timeline_path, broken)
        negative = audit_run(
            root,
            expected_hashes=[],
            allowed_statuses={DEFAULT_COMPLETE_STATUS},
            max_gap_ms=DEFAULT_MAX_GAP_MS,
            max_window_edge_gap_ms=DEFAULT_MAX_GAP_MS,
        )
        require(negative.get("status") == "FAIL", "synthetic corruption did not fail closed")
        return {
            "schema": AUDIT_SCHEMA,
            "status": "PASS_SYNTHETIC_CPU_SELFTEST",
            "scope": "temporary CPU-only fixture; no real run artifact was modified",
            "positive_fixture": "PASS",
            "corruption_fixture": "FAIL_CLOSED",
            "gpu_binary_executed": False,
            "nvidia_smi_called": False,
        }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only read-only v3 held-out paired-performance result verifier"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path, help="completed v3 run directory to audit")
    group.add_argument("--selftest", action="store_true", help="run disposable CPU-only synthetic fixture")
    parser.add_argument(
        "--expected-hash",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help=(
            "optional SHA binding; NAME may be a run-relative manifest path, "
            "an alias (manifest/result/terminal/launch/timing_pairs), or a "
            "record path such as launch.binary"
        ),
    )
    parser.add_argument(
        "--allow-status",
        action="append",
        default=None,
        metavar="STATUS",
        help="allowed complete terminal status; default is conditional unlocked-DVFS completion",
    )
    parser.add_argument(
        "--max-inter-sample-gap-ms",
        type=float,
        default=DEFAULT_MAX_GAP_MS,
        help=f"fail if global telemetry gap exceeds this (default {DEFAULT_MAX_GAP_MS:g})",
    )
    parser.add_argument(
        "--max-window-edge-gap-ms",
        type=float,
        default=DEFAULT_MAX_GAP_MS,
        help=f"fail if process telemetry misses a timeline edge by this much (default {DEFAULT_MAX_GAP_MS:g})",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.selftest:
        try:
            report = selftest()
        except AuditFailure as exc:
            report = {
                "schema": AUDIT_SCHEMA,
                "status": "FAIL_SYNTHETIC_CPU_SELFTEST",
                "failure": str(exc),
                "gpu_binary_executed": False,
                "nvidia_smi_called": False,
            }
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0 if report["status"] == "PASS_SYNTHETIC_CPU_SELFTEST" else 2
    allowed = set(args.allow_status or [DEFAULT_COMPLETE_STATUS])
    report = audit_run(
        args.run_dir,
        expected_hashes=args.expected_hash,
        allowed_statuses=allowed,
        max_gap_ms=args.max_inter_sample_gap_ms,
        max_window_edge_gap_ms=args.max_window_edge_gap_ms,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
