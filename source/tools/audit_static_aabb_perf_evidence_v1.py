#!/usr/bin/env python3
"""CPU-only audit of static-AABB paired-performance evidence.

It never invokes CUDA, nvidia-smi, or a target binary.  It can classify the
legacy v2.1 exploratory runs and verify the artifact contract prescribed by
static_aabb_heldout_perf_evidence_protocol_v1.json for a later successor run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
DEFAULT_PROTOCOL = ROOT / "provenance/static_aabb_heldout_perf_evidence_protocol_v1.json"
EXPECTED_BINARY = "e587ad822de007880e1caae8ea89072c07a5b786dcb5d15357ecde61ce577cde"
EXPECTED_SOURCE = "5dc92a9660bc168e1325edae7a1c9d41f59a93e036e83f57f2ccd9d55fa6819f"
EXPECTED_RESULT_STATUS = "PASS_EXPLORATORY_PAIRED_PROBE_V2_DISKGUARD"
LEGACY_PRIMARY = (
    "preflight.json",
    "gpu_before.csv",
    "gpu_after.csv",
    "gpu_processes_before.txt",
    "stdout.log",
    "stderr.log",
    "result.json",
    "snapshot_sha256.txt",
)
FUTURE_MANIFEST_SCHEMA = "safe-c2-static-aabb-heldout-perf-manifest-v1"
FUTURE_RUNNER_FREEZE_SCHEMA = "safe-c2-static-aabb-heldout-perf-runner-freeze-v1"


def sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid JSON file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def finite_positive(value: Any, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric") from error
    if not math.isfinite(out) or out <= 0:
        raise RuntimeError(f"{label} is not positive finite")
    return out


def median(values: list[float]) -> float:
    if not values:
        raise RuntimeError("median of empty vector")
    return float(statistics.median(values))


def compact_stats(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "max": max(values),
        "mean": float(statistics.fmean(values)),
    }


def atomic_write_new(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing pre-existing output: {path}")
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def verify_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    if protocol.get("schema") != "safe-c2-static-aabb-heldout-perf-evidence-protocol-v1":
        raise RuntimeError("unexpected performance evidence protocol schema")
    if protocol.get("status") != "FROZEN_CPU_ONLY_PREPARED":
        raise RuntimeError("performance evidence protocol not frozen CPU-only")
    if protocol.get("hard_boundaries", {}).get("gpu_executed_for_this_protocol") is not False:
        raise RuntimeError("protocol claims GPU execution")
    if protocol.get("hard_boundaries", {}).get("nvidia_smi_called_for_this_protocol") is not False:
        raise RuntimeError("protocol claims nvidia-smi use")
    binding = protocol.get("algorithm_binding", {})
    if binding.get("perf_binary", {}).get("sha256") != EXPECTED_BINARY:
        raise RuntimeError("unexpected v2.1 binary freeze")
    if binding.get("perf_source", {}).get("sha256") != EXPECTED_SOURCE:
        raise RuntimeError("unexpected v2.1 source freeze")
    binary = Path(binding["perf_binary"]["path"])
    source = Path(binding["perf_source"]["path"])
    if sha256(binary) != EXPECTED_BINARY or sha256(source) != EXPECTED_SOURCE:
        raise RuntimeError("frozen v2.1 artifact changed")
    parent = Path(protocol["parent_heldout_protocol"]["path"])
    if sha256(parent) != protocol["parent_heldout_protocol"]["sha256"]:
        raise RuntimeError("parent heldout protocol changed")
    return protocol


def verify_legacy_manifest(run: Path, manifest: dict[str, Any], full_hash: bool) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("legacy manifest missing artifacts")
    checked: list[str] = []
    names = sorted(artifacts) if full_hash else list(LEGACY_PRIMARY)
    for name in names:
        if name not in artifacts:
            raise RuntimeError(f"legacy manifest misses primary artifact: {name}")
        path = run / name
        info = artifacts[name]
        if not isinstance(info, dict) or info.get("sha256") != sha256(path) or info.get("bytes") != path.stat().st_size:
            raise RuntimeError(f"legacy manifest hash mismatch: {name}")
        checked.append(name)
    return {
        "verified_artifact_count": len(checked),
        "verified_artifacts": checked,
        "full_manifest_rehashed": full_hash,
        "manifest_artifact_count": len(artifacts),
    }


def audit_legacy_run(run: Path, protocol: dict[str, Any], full_hash: bool) -> dict[str, Any]:
    if not run.is_dir() or run.is_symlink():
        raise RuntimeError(f"invalid legacy run directory: {run}")
    manifest = load_json(run / "manifest.json")
    result = load_json(run / "result.json")
    if manifest.get("schema") != "safe-c2-static-aabb-diskguard-v2.1-manifest":
        raise RuntimeError("unexpected legacy manifest schema")
    if manifest.get("status") != "COMPLETE_EXPLORATORY_PAIRED_PROBE":
        raise RuntimeError("legacy run not a complete exploratory probe")
    binary = manifest.get("binary", {})
    if binary.get("sha256") != EXPECTED_BINARY:
        raise RuntimeError("legacy run used unexpected binary")
    if protocol["algorithm_binding"]["perf_binary"]["sha256"] != binary["sha256"]:
        raise RuntimeError("legacy binary does not match performance protocol")
    manifest_check = verify_legacy_manifest(run, manifest, full_hash)
    if result.get("status") != EXPECTED_RESULT_STATUS:
        raise RuntimeError("legacy performance result did not pass")
    if result.get("formal_claim_eligible") is not False:
        raise RuntimeError("legacy result is incorrectly formal")
    if result.get("projection", {}).get("dimensions") != 128:
        raise RuntimeError("legacy result is not full-coordinate")
    gate = result.get("unmeasured_gate", {})
    for variant in ("baseline", "static_aabb"):
        evidence = gate.get(variant, {})
        if any(evidence.get(key) != 0 for key in ("invalid", "duplicate", "gt_set_mismatch", "distance_mismatch")):
            raise RuntimeError(f"legacy GT gate failed: {variant}")
    if gate.get("id_sets_equal") is not True or result.get("snapshot_pre_post_byte_equal") is not True:
        raise RuntimeError("legacy equality/snapshot gate failed")
    timing = result.get("timing_gpu_ms", {})
    baseline = [finite_positive(x, "baseline timing") for x in timing.get("baseline", [])]
    static_aabb = [finite_positive(x, "static AABB timing") for x in timing.get("static_aabb", [])]
    if len(baseline) != 7 or len(static_aabb) != 7:
        raise RuntimeError("legacy paired-repetition count is not seven")
    ratios = [candidate / base for base, candidate in zip(baseline, static_aabb)]
    source_order = [
        "baseline_then_static_aabb" if rep % 2 == 0 else "static_aabb_then_baseline"
        for rep in range(len(ratios))
    ]
    before = (run / "gpu_before.csv").read_text().strip()
    after = (run / "gpu_after.csv").read_text().strip()
    proc_before = (run / "gpu_processes_before.txt").read_text().strip()
    # The old script logged only identity/driver/memory: no dynamic telemetry fields.
    before_fields = [part.strip() for part in before.split(",")] if before else []
    after_fields = [part.strip() for part in after.split(",")] if after else []
    return {
        "path": str(run),
        "manifest_sha256": sha256(run / "manifest.json"),
        "result_sha256": sha256(run / "result.json"),
        "binary_sha256": binary["sha256"],
        "manifest_check": manifest_check,
        "correctness_gate_passed": True,
        "paired_gpu_event_ms": {
            "baseline": baseline,
            "static_aabb": static_aabb,
            "pairwise_static_over_baseline": ratios,
            "pair_orders_reconstructed_from_frozen_v2_1_source": source_order,
            "pairwise_ratio_stats": compact_stats(ratios),
            "reported_ratio_of_method_medians": finite_positive(
                timing.get("median_ratio_static_over_baseline"), "reported median ratio"
            ),
        },
        "legacy_environment_observed": {
            "gpu_before_raw": before,
            "gpu_after_raw": after,
            "gpu_before_field_count": len(before_fields),
            "gpu_after_field_count": len(after_fields),
            "prelaunch_compute_processes_raw": proc_before,
            "prelaunch_compute_process_guard_clean": proc_before == "",
            "postlaunch_process_guard_present": False,
            "continuous_clock_power_temperature_telemetry_present": False,
            "target_pid_recorded": False,
            "cuda_visible_devices_recorded": False,
            "physical_gpu_binding_by_uuid_recorded": False,
        },
        "evidence_grade": "EXPLORATORY_TELEMETRY_INSUFFICIENT",
        "formal_claim_eligible": False,
        "limitations": [
            "Only a single pre-launch sampled compute-process query was retained; no post-run or continuous process observation exists.",
            "The before/after CSV records GPU identity, driver, total memory, and used memory only; it lacks P-state, clocks, power draw, temperature, and telemetry timestamps.",
            "The three legacy invocations have materially different absolute GPU-event medians, so their differences cannot be attributed from the archived files.",
            "The stored ordering is recoverable from the frozen source but individual timing intervals cannot be aligned with environment state.",
        ],
    }


def require_hash_manifest(run: Path, manifest: dict[str, Any], names: list[str]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("future manifest missing artifacts")
    for name in names:
        path = run / name
        if name not in artifacts or not path.is_file() or path.is_symlink():
            raise RuntimeError(f"future artifact missing: {name}")
        info = artifacts[name]
        if not isinstance(info, dict) or info.get("bytes") != path.stat().st_size or info.get("sha256") != sha256(path):
            raise RuntimeError(f"future artifact hash mismatch: {name}")


def numeric_cell(row: dict[str, str], name: str, nonnegative: bool = True) -> float:
    value = row.get(name)
    if value is None or value.strip() in ("", "N/A", "[N/A]"):
        raise RuntimeError(f"missing telemetry {name}")
    try:
        parsed = float(value)
    except ValueError as error:
        raise RuntimeError(f"invalid telemetry {name}") from error
    if not math.isfinite(parsed) or (nonnegative and parsed < 0):
        raise RuntimeError(f"invalid telemetry range {name}")
    return parsed


def audit_future_run(run: Path, protocol: dict[str, Any], protocol_path: Path) -> dict[str, Any]:
    if not run.is_dir() or run.is_symlink():
        raise RuntimeError(f"invalid future run directory: {run}")
    required = list(protocol["required_run_artifacts"])
    manifest = load_json(run / "manifest.json")
    if manifest.get("schema") != FUTURE_MANIFEST_SCHEMA:
        raise RuntimeError("unexpected future manifest schema")
    if manifest.get("protocol", {}).get("sha256") != sha256(protocol_path):
        raise RuntimeError("future run protocol hash mismatch")
    require_hash_manifest(run, manifest, [name for name in required if name != "manifest.json"])

    freeze = load_json(run / "runner_freeze_reference.json")
    if freeze.get("schema") != FUTURE_RUNNER_FREEZE_SCHEMA or freeze.get("status") != "FROZEN_CPU_ONLY":
        raise RuntimeError("future runner is not independently frozen CPU-only")
    if freeze.get("protocol_sha256") != sha256(protocol_path):
        raise RuntimeError("future runner freeze protocol mismatch")
    runner = Path(freeze["runner"]["path"])
    if sha256(runner) != freeze["runner"]["sha256"]:
        raise RuntimeError("future runner file mismatch")

    preflight = load_json(run / "preflight.json")
    if preflight.get("status") != "PASS_CPU_ONLY_FROZEN_EXECUTION_PENDING":
        raise RuntimeError("future heldout CPU-only preflight did not pass")
    launch = load_json(run / "launch.json")
    target_pid = launch.get("target_pid")
    if not isinstance(target_pid, int) or target_pid <= 0:
        raise RuntimeError("future launch lacks target PID")
    try:
        target_start_ns = int(launch["target_start_monotonic_ns"])
        target_exit_ns = int(launch["target_exit_monotonic_ns"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("future launch lacks target monotonic lifetime") from error
    if target_start_ns >= target_exit_ns:
        raise RuntimeError("future target lifetime is invalid")
    gpu_uuid = launch.get("gpu", {}).get("uuid")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise RuntimeError("future launch lacks physical GPU UUID")
    if launch.get("cuda_visible_devices") != gpu_uuid:
        raise RuntimeError("future runner did not bind CUDA_VISIBLE_DEVICES to GPU UUID")
    env = load_json(run / "environment.json")
    if env.get("gpu", {}).get("uuid") != gpu_uuid or env.get("cuda_visible_devices") != gpu_uuid:
        raise RuntimeError("future environment/launch GPU binding mismatch")
    control = env.get("clock_control", {})
    for field in (
        "runner_attempted_clock_lock",
        "runner_attempted_application_clocks",
        "runner_attempted_power_limit_change",
        "runner_attempted_compute_mode_change",
    ):
        if control.get(field) is not False:
            raise RuntimeError(f"future clock-control protocol violated: {field}")
    if control.get("evidence_label") != "DVFS_UNCONTROLLED_RECORDED":
        raise RuntimeError("future clock-control evidence label")

    telemetry_path = run / "gpu_telemetry.csv"
    with telemetry_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required_columns = protocol["telemetry_contract"]["gpu_telemetry_csv_required_columns"]
    if not rows or any(column not in rows[0] for column in required_columns):
        raise RuntimeError("future telemetry CSV schema")
    target_rows = [row for row in rows if row.get("phase") == "target"]
    if not target_rows:
        raise RuntimeError("no target-phase telemetry")
    monotonic = []
    sm_clock = []
    graphics_clock = []
    memory_clock = []
    power = []
    temp = []
    pstates: set[str] = set()
    for row in target_rows:
        if row.get("gpu_uuid") != gpu_uuid:
            raise RuntimeError("telemetry GPU UUID mismatch")
        try:
            monotonic.append(int(row["monotonic_ns"]))
        except (TypeError, ValueError) as error:
            raise RuntimeError("invalid telemetry monotonic timestamp") from error
        pstates.add(row.get("pstate", ""))
        sm_clock.append(numeric_cell(row, "clock_sm_mhz"))
        graphics_clock.append(numeric_cell(row, "clock_graphics_mhz"))
        memory_clock.append(numeric_cell(row, "clock_memory_mhz"))
        power.append(numeric_cell(row, "power_draw_w"))
        temp.append(numeric_cell(row, "temperature_gpu_c"))
        numeric_cell(row, "power_limit_w")
        numeric_cell(row, "utilization_gpu_pct")
        numeric_cell(row, "utilization_memory_pct")
        numeric_cell(row, "memory_used_mib")
    if monotonic != sorted(monotonic) or len(set(monotonic)) != len(monotonic):
        raise RuntimeError("telemetry monotonic timestamps are not strictly increasing")
    max_gap_ms = max((right - left) / 1e6 for left, right in zip(monotonic, monotonic[1:])) if len(monotonic) > 1 else math.inf
    accepted_gap_ms = protocol["telemetry_contract"]["max_accepted_inter_sample_gap_ms"]
    if max_gap_ms > accepted_gap_ms:
        raise RuntimeError("telemetry sampling gap exceeds frozen maximum")
    if monotonic[0] > target_start_ns + accepted_gap_ms * 1_000_000 or monotonic[-1] < target_exit_ns - accepted_gap_ms * 1_000_000:
        raise RuntimeError("telemetry does not cover target lifetime")
    if "" in pstates or "N/A" in pstates:
        raise RuntimeError("missing telemetry P-state")

    foreign_pids: set[int] = set()
    process_rows = 0
    process_target_monotonic: list[int] = []
    with (run / "gpu_process_samples.jsonl").open() as stream:
        for raw in stream:
            if not raw.strip():
                continue
            sample = json.loads(raw)
            process_rows += 1
            if sample.get("phase") != "target":
                continue
            if sample.get("gpu_uuid") != gpu_uuid:
                raise RuntimeError("process telemetry GPU UUID mismatch")
            if not isinstance(sample.get("raw_query_output"), str):
                raise RuntimeError("process telemetry lacks raw query output")
            try:
                process_target_monotonic.append(int(sample["monotonic_ns"]))
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("process telemetry lacks monotonic timestamp") from error
            procs = sample.get("compute_processes")
            if not isinstance(procs, list):
                raise RuntimeError("compute process list missing")
            for process in procs:
                pid = process.get("pid")
                if not isinstance(pid, int) or pid <= 0:
                    raise RuntimeError("invalid sampled compute PID")
                if pid != target_pid:
                    foreign_pids.add(pid)
    if process_rows == 0 or not process_target_monotonic:
        raise RuntimeError("no target-phase process samples")
    if process_target_monotonic != sorted(process_target_monotonic) or len(set(process_target_monotonic)) != len(process_target_monotonic):
        raise RuntimeError("process telemetry timestamps are not strictly increasing")
    process_gap_ms = max((right - left) / 1e6 for left, right in zip(process_target_monotonic, process_target_monotonic[1:])) if len(process_target_monotonic) > 1 else math.inf
    if process_gap_ms > accepted_gap_ms:
        raise RuntimeError("process telemetry sampling gap exceeds frozen maximum")
    if process_target_monotonic[0] > target_start_ns + accepted_gap_ms * 1_000_000 or process_target_monotonic[-1] < target_exit_ns - accepted_gap_ms * 1_000_000:
        raise RuntimeError("process telemetry does not cover target lifetime")
    if foreign_pids:
        raise RuntimeError(f"foreign compute PID sampled: {sorted(foreign_pids)}")

    timing = load_json(run / "timing_pairs.json")
    if timing.get("source_result_sha256") != sha256(run / "result.json"):
        raise RuntimeError("future timing pairs are not bound to result.json")
    pairs = timing.get("pairs")
    expected_reps = protocol["successor_runner_requirements"]["measurement"]["paired_repetitions_per_invocation"]
    if not isinstance(pairs, list) or len(pairs) != expected_reps:
        raise RuntimeError("future timing pair count")
    ratios: list[float] = []
    for rep, pair in enumerate(pairs):
        if pair.get("rep") != rep:
            raise RuntimeError("future timing rep order")
        expected_order = "baseline_then_static_aabb" if rep % 2 == 0 else "static_aabb_then_baseline"
        if pair.get("order") != expected_order:
            raise RuntimeError("future timing order")
        base = finite_positive(pair.get("baseline_gpu_ms"), "future baseline timing")
        candidate = finite_positive(pair.get("static_aabb_gpu_ms"), "future static timing")
        ratio = finite_positive(pair.get("static_over_baseline"), "future timing ratio")
        if not math.isclose(ratio, candidate / base, rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError("future timing ratio mismatch")
        ratios.append(ratio)
    result = load_json(run / "result.json")
    gate = result.get("unmeasured_gate", {})
    if result.get("status") != EXPECTED_RESULT_STATUS or result.get("snapshot_pre_post_byte_equal") is not True:
        raise RuntimeError("future v2.1 result failed")
    for variant in ("baseline", "static_aabb"):
        if any(gate.get(variant, {}).get(key) != 0 for key in ("invalid", "duplicate", "gt_set_mismatch", "distance_mismatch")):
            raise RuntimeError(f"future correctness gate failed: {variant}")
    if gate.get("id_sets_equal") is not True:
        raise RuntimeError("future baseline/AABB sets differ")
    terminal = load_json(run / "terminal.json")
    terminal_contract = protocol["terminal_contract"]
    if (terminal.get("status") != terminal_contract["success_status"] or
            terminal.get("exit_code") != terminal_contract["success_exit_code"] or
            terminal.get("formal_claim_eligible") is not terminal_contract["formal_claim_eligible"]):
        raise RuntimeError("future terminal contract")
    return {
        "path": str(run),
        "evidence_grade": "CONDITIONAL_PAIRED_DVFS_UNCONTROLLED",
        "formal_claim_eligible": False,
        "target_gpu_uuid": gpu_uuid,
        "target_pid": target_pid,
        "telemetry_target_samples": len(target_rows),
        "telemetry_max_gap_ms": max_gap_ms,
        "process_telemetry_max_gap_ms": process_gap_ms,
        "pstates_observed": sorted(pstates),
        "sm_clock_mhz": compact_stats(sm_clock),
        "graphics_clock_mhz": compact_stats(graphics_clock),
        "memory_clock_mhz": compact_stats(memory_clock),
        "power_draw_w": compact_stats(power),
        "temperature_gpu_c": compact_stats(temp),
        "pairwise_ratio": compact_stats(ratios),
        "interpretation": (
            "all prescribed run-level telemetry and sampled-process evidence is present, but clocks "
            "were intentionally not controlled; only a conditional paired relative result under the "
            "recorded DVFS trace is eligible"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--legacy-run", type=Path, action="append", default=[])
    parser.add_argument("--future-run", type=Path, action="append", default=[])
    parser.add_argument("--full-hash-legacy-manifests", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.legacy_run and not args.future_run:
        parser.error("at least one --legacy-run or --future-run is required")
    protocol = verify_protocol(args.protocol)
    legacy = [audit_legacy_run(path, protocol, args.full_hash_legacy_manifests) for path in args.legacy_run]
    future = [audit_future_run(path, protocol, args.protocol) for path in args.future_run]
    legacy_run_medians = [item["paired_gpu_event_ms"]["pairwise_ratio_stats"]["median"] for item in legacy]
    future_run_medians = [item["pairwise_ratio"]["median"] for item in future]
    output: dict[str, Any] = {
        "schema": "safe-c2-static-aabb-perf-evidence-audit-v1",
        "status": "PASS_CPU_ONLY_AUDIT",
        "scope": "CPU-only evidence audit; no CUDA execution, nvidia-smi invocation, or clock/power/compute-mode change",
        "protocol": {"path": str(args.protocol), "sha256": sha256(args.protocol)},
        "auditor": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "legacy_v2_1_runs": legacy,
        "future_heldout_runs": future,
        "legacy_campaign": {
            "run_count": len(legacy),
            "per_run_median_pairwise_ratios": legacy_run_medians,
            "median_of_run_medians": median(legacy_run_medians) if legacy_run_medians else None,
            "evidence_grade": "EXPLORATORY_TELEMETRY_INSUFFICIENT" if legacy else None,
        },
        "future_campaign": {
            "run_count": len(future),
            "required_independent_invocations": protocol["successor_runner_requirements"]["measurement"]["independent_process_invocations"],
            "per_run_median_pairwise_ratios": future_run_medians,
            "campaign_complete": len(future) == protocol["successor_runner_requirements"]["measurement"]["independent_process_invocations"],
            "evidence_grade": (
                "CONDITIONAL_PAIRED_DVFS_UNCONTROLLED"
                if len(future) == protocol["successor_runner_requirements"]["measurement"]["independent_process_invocations"]
                else "INCOMPLETE_CAMPAIGN"
            ) if future else None,
        },
        "formal_claim_eligible": False,
    }
    text = json.dumps(output, sort_keys=True, indent=2) + "\n"
    if args.out:
        atomic_write_new(args.out, text)
        print(args.out)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL_STATIC_AABB_PERF_EVIDENCE_AUDIT_V1: {error}", file=sys.stderr)
        raise SystemExit(2)
