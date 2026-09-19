#!/usr/bin/env python3
"""
Fail-closed runner for the frozen Safe-C2 static-AABB v3 timeline protocol.

CPU-only modes never invoke nvidia-smi or the CUDA binary.  A real batch needs
both a runner freeze and a separately recorded one-batch authorization, then an
explicit --execute-authorized flag.  The executor uses only read-only
nvidia-smi queries and never changes clocks, power, persistence, compute mode,
or another process.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
RUNS = ROOT / "runs"
BINARY = ROOT / "bin/static_aabb_perf_probe_v3_timeline"
SOURCE = ROOT / "src/static_aabb_perf_probe_v3_timeline.cu"
IDS = ROOT / "inputs/standard_sift_static_aabb_perfheldout256_v2.ids"
SELECTION = ROOT / "provenance/static_aabb_perfheldout_selection_v2.json"
PROTOCOL = ROOT / "provenance/static_aabb_timeline_perf_protocol_v3.json"
SOURCE_AUDIT_RESULT = ROOT / "provenance/static_aabb_perf_probe_v3_timeline_source_audit_v1.json"
BUILD = ROOT / "provenance/static_aabb_perf_probe_v3_timeline_build_v1.json"
SOURCE_AUDITOR = ROOT / "tools/audit_static_aabb_perf_probe_v3_timeline.py"
POSTRUN_AUDITOR = ROOT / "tools/audit_static_aabb_heldout_perf_result_v3.py"
RUNNER_FREEZE = ROOT / "provenance/static_aabb_timeline_perf_runner_freeze_v1.json"
AUTHORIZATION = ROOT / "provenance/static_aabb_timeline_perf_runner_authorization_v1.json"
BASE = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs")
QUERY = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs")
GT = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs")

GPU_INDEX = 0
GPU_UUID = "GPU-CONFIGURE-ARCHIVE-DEVICE"
WARMUPS = 5
REPS = 30
INVOCATIONS = 5
SAMPLE_PERIOD_S = 0.100
MAX_GAP_MS = 500.0
POSTEXIT_EMPTY_SAMPLES = 3
PRELAUNCH_EMPTY_SAMPLES = 3
CHILD_TIMEOUT_S = 1800

EXPECTED = {
    "binary": "3b5ce855ba28085caf3424643d69b90d9590ec8b38738519c7e239f2e4bc8910",
    "source": "3f8396566500ea22dcfd02fb9437565d5f7df614a693426301de84bcb8424059",
    "ids": "990c423807d9b79eafee1491056c4b3a75243e70c1705439972ba2faecdda547",
    "selection": "b519557866646ba0085e1999c6861a1ed7c54c50de5257fe4b729bd642adc806",
    "protocol": "93cdcee0bc4c854dfddea27a55ccc4d6840acd7eef9b4cbeda42ba99b76dce1e",
    "source_audit_result": "f133828fc81dfd80dd0895ea184fa9dd1b7ffb44f6d8bb82bb7edcbdaa95c5ce",
    "build": "ccd3446686cb68e950b27ef853b3e8849271288ae9ef770737f90befd64579aa",
    "source_auditor": "b7db883912e61af23dd79ac5fd10f66e5a676b70a6e1d65b99dac6aea06a23bf",
    "postrun_auditor": "ceeece94951a96ae6f928f014ba8b4c50f1a4cec0c462ec2e70d81080a357be4",
    "base": "21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816",
    "query": "f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc",
    "gt": "2b71de0a8d3a6cd62f070ab65ea65f4f",
}
# The GT digest above is intentionally filled in immediately below to keep the
# input dictionary visually compact and to make an accidental truncation obvious.
EXPECTED["gt"] = "2b71de0a8d5a83e6a84eec3e23fb8b611d8801dd9b3a6cd62f070ab65ea65f4f"

GPU_FIELDS = [
    "index",
    "uuid",
    "pci.bus_id",
    "name",
    "driver_version",
    "compute_mode",
    "persistence_mode",
    "power.limit",
    "pstate",
    "clocks.sm",
    "clocks.current.graphics",
    "clocks.current.memory",
    "power.draw",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
]
CSV_HEADER = [
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
]
NUMERIC_COLUMNS = [
    "clock_sm_mhz",
    "clock_graphics_mhz",
    "clock_memory_mhz",
    "power_draw_w",
    "power_limit_w",
    "temperature_gpu_c",
    "utilization_gpu_pct",
    "utilization_memory_pct",
    "memory_used_mib",
]
SNAPSHOT_SUFFIXES = (
    "nodes",
    "empty",
    "maxd",
    "ids",
    "aabb_lo",
    "aabb_hi",
    "aabb_dims",
)


class RunnerError(RuntimeError):
    pass


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RunnerError("not a regular non-symlink file: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def artifact_relative(root: Path, path: Path) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise RunnerError("artifact is outside run directory: " + str(path))
    return {
        "path": str(resolved_path.relative_to(resolved_root)),
        "bytes": resolved_path.stat().st_size,
        "sha256": sha256_file(resolved_path),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except FileNotFoundError:
            pass


def parse_json_file(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RunnerError(label + " missing or not regular")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RunnerError(label + " malformed JSON: " + str(exc)) from exc
    if not isinstance(value, dict):
        raise RunnerError(label + " top level is not an object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def same_number(left: Any, right: Any, label: str) -> None:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError) as exc:
        raise RunnerError(label + " is not numeric") from exc
    if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6):
        raise RunnerError(label + " mismatch: " + repr(left) + " != " + repr(right))


def number(value: Any, label: str, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RunnerError(label + " is not numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise RunnerError(label + " has invalid value")
    return result


def file_expected(name: str, path: Path) -> dict[str, Any]:
    record = artifact(path)
    expected = EXPECTED[name]
    require(record["sha256"] == expected, "SHA mismatch for " + name)
    return record


def run_json_cpu(command: list[str], label: str) -> dict[str, Any]:
    # This helper is only used for the documented CPU-only auditors.
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RunnerError(label + " failed: " + completed.stderr.strip())
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RunnerError(label + " did not emit JSON") from exc
    if not isinstance(value, dict):
        raise RunnerError(label + " JSON top level is not an object")
    return value


def expected_record(binding: Any, actual: dict[str, Any], label: str) -> None:
    require(isinstance(binding, dict), label + " binding missing")
    require(binding.get("sha256") == actual["sha256"], label + " SHA binding mismatch")
    require(binding.get("bytes") == actual["bytes"], label + " byte binding mismatch")


def verify_fresh_ids() -> list[int]:
    try:
        values = [int(line) for line in IDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    except ValueError as exc:
        raise RunnerError("held-out IDs contain non-integer value") from exc
    require(len(values) == 256, "held-out ID count is not 256")
    require(values == sorted(values) and len(set(values)) == len(values), "held-out IDs not strictly increasing")
    require(values[0] == 5260 and values[-1] == 5520, "held-out endpoint mismatch")
    require(not any(5000 <= item <= 5259 for item in values), "held-out IDs overlap failed predecessor interval")
    return values


def runner_path() -> Path:
    return Path(__file__).resolve()


def verify_protocol_contract(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    protocol = parse_json_file(PROTOCOL, "timeline protocol")
    require(protocol.get("schema") == "safe-c2-static-aabb-timeline-perf-protocol-v3", "timeline protocol schema")
    require(protocol.get("status") == "FROZEN_CPU_ONLY_PREPARED", "timeline protocol status")
    boundary = protocol.get("execution_boundary")
    require(isinstance(boundary, dict) and boundary.get("does_not_authorize_gpu_execution") is True, "protocol execution boundary")
    algorithm = protocol.get("algorithm_binding")
    fresh = protocol.get("fresh_heldout_binding")
    dataset = protocol.get("dataset_binding")
    require(isinstance(algorithm, dict) and isinstance(fresh, dict) and isinstance(dataset, dict), "protocol binding sections")
    expected_record(algorithm.get("timeline_binary"), artifacts["binary"], "protocol binary")
    expected_record(algorithm.get("timeline_source"), artifacts["source"], "protocol source")
    expected_record(algorithm.get("source_audit"), artifacts["source_audit_result"], "protocol source audit")
    expected_record(algorithm.get("build"), artifacts["build"], "protocol build")
    expected_record(fresh.get("ids"), artifacts["ids"], "protocol IDs")
    expected_record(fresh.get("selection"), artifacts["selection"], "protocol selection")
    expected_record(dataset.get("base"), artifacts["base"], "protocol base")
    expected_record(dataset.get("query"), artifacts["query"], "protocol query")
    expected_record(dataset.get("groundtruth"), artifacts["gt"], "protocol groundtruth")
    expected_record(protocol.get("matching_postrun_auditor"), artifacts["postrun_auditor"], "protocol postrun auditor")
    telemetry = protocol.get("telemetry_contract")
    timeline = protocol.get("timeline_contract")
    require(isinstance(telemetry, dict) and telemetry.get("max_gap_ms") == MAX_GAP_MS, "protocol telemetry gap")
    require(isinstance(timeline, dict), "protocol timeline contract")
    return protocol


def bound_artifact(document: dict[str, Any], name: str, actual: dict[str, Any], label: str) -> None:
    value = document.get(name)
    expected_record(value, actual, label + "." + name)


def verify_execution_gate(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not RUNNER_FREEZE.is_file() or RUNNER_FREEZE.is_symlink():
        raise RunnerError("runner freeze absent")
    if not AUTHORIZATION.is_file() or AUTHORIZATION.is_symlink():
        raise RunnerError("one-batch authorization absent")
    freeze = parse_json_file(RUNNER_FREEZE, "runner freeze")
    authorization = parse_json_file(AUTHORIZATION, "runner authorization")
    require(freeze.get("status") == "FROZEN_CPU_ONLY", "runner freeze status")
    require(authorization.get("status") == "CPU_AUDITED_READY_FOR_ONE_GPU_BATCH", "authorization status")
    current_runner = artifact(runner_path())
    freeze_artifacts = {
        "runner": current_runner,
        "protocol": artifacts["protocol"],
        "binary": artifacts["binary"],
        "ids": artifacts["ids"],
        "source": artifacts["source"],
        "postrun_auditor": artifacts["postrun_auditor"],
    }
    authorization_artifacts = {
        "runner": current_runner,
        "protocol": artifacts["protocol"],
        "binary": artifacts["binary"],
        "ids": artifacts["ids"],
    }
    for key, value in freeze_artifacts.items():
        bound_artifact(freeze, key, value, "runner freeze")
    for key, value in authorization_artifacts.items():
        bound_artifact(authorization, key, value, "runner authorization")
    permission = authorization.get("authorization")
    if isinstance(permission, dict):
        require(permission.get("one_gpu_batch") is True, "authorization does not permit one GPU batch")
    else:
        require(authorization.get("one_gpu_batch") is True, "authorization does not permit one GPU batch")
    return {
        "runner_freeze": artifact(RUNNER_FREEZE),
        "authorization": artifact(AUTHORIZATION),
        "runner": current_runner,
    }


def cpu_preflight(require_execution_gate: bool) -> dict[str, Any]:
    paths = {
        "binary": BINARY,
        "source": SOURCE,
        "ids": IDS,
        "selection": SELECTION,
        "protocol": PROTOCOL,
        "source_audit_result": SOURCE_AUDIT_RESULT,
        "build": BUILD,
        "source_auditor": SOURCE_AUDITOR,
        "postrun_auditor": POSTRUN_AUDITOR,
        "base": BASE,
        "query": QUERY,
        "gt": GT,
    }
    artifacts = {name: file_expected(name, path) for name, path in paths.items()}
    ids = verify_fresh_ids()
    selection = parse_json_file(SELECTION, "held-out selection")
    source_audit_result = parse_json_file(SOURCE_AUDIT_RESULT, "source audit result")
    build = parse_json_file(BUILD, "build provenance")
    require(selection.get("status") == "FROZEN_CPU_ONLY", "held-out selection state")
    require(source_audit_result.get("status") == "PASS_SOURCE_ONLY_AUDIT", "source audit result state")
    require(build.get("status") == "COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION", "build provenance state")
    require(build.get("binary", {}).get("sha256") == artifacts["binary"]["sha256"], "build binary binding")
    require(build.get("source", {}).get("sha256") == artifacts["source"]["sha256"], "build source binding")
    protocol = verify_protocol_contract(artifacts)
    source_audit_now = run_json_cpu([sys.executable, str(SOURCE_AUDITOR)], "source-only timeline auditor")
    require(source_audit_now.get("status") == "PASS_SOURCE_ONLY_AUDIT", "fresh source-only audit status")
    require(source_audit_now.get("source_v3", {}).get("sha256") == artifacts["source"]["sha256"], "fresh source audit source binding")
    postrun_selftest = run_json_cpu([sys.executable, str(POSTRUN_AUDITOR), "--selftest"], "postrun auditor selftest")
    require(postrun_selftest.get("status") == "PASS_SYNTHETIC_CPU_SELFTEST", "postrun auditor synthetic selftest")
    gate = verify_execution_gate(artifacts) if require_execution_gate else None
    return {
        "schema": "safe-c2-static-aabb-timeline-perf-runner-preflight-v3",
        "status": "PASS_CPU_ONLY_READY_FOR_ONE_GPU_BATCH" if require_execution_gate else "PASS_CPU_ONLY_UNFROZEN_RUNNER_AUDIT",
        "scope": "artifact, source-audit, and synthetic-audit checks only; no nvidia-smi, CUDA binary, or GPU execution",
        "formal_claim_eligible": False,
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
        "target_gpu": {"index": GPU_INDEX, "uuid": GPU_UUID},
        "measurement": {
            "warmups_per_invocation": WARMUPS,
            "paired_repetitions_per_invocation": REPS,
            "independent_process_invocations": INVOCATIONS,
            "timed_operations_per_invocation": REPS * 2,
        },
        "artifacts": artifacts,
        "runner": artifact(runner_path()),
        "protocol_status": protocol.get("status"),
        "heldout_ids": {"count": len(ids), "first": ids[0], "last": ids[-1], "sha256": artifacts["ids"]["sha256"]},
        "source_audit_now": source_audit_now,
        "postrun_auditor_selftest": postrun_selftest,
        "execution_gate": gate,
    }


def numeric(value: str) -> float | None:
    try:
        result = float(value.strip().replace("W", "").replace("MiB", "").replace("MHz", ""))
    except (AttributeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def nvidia_smi() -> str:
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise RunnerError("nvidia-smi not found")
    return executable


def physical_gpu_info(nv: str) -> dict[str, Any]:
    command = [
        nv,
        "--query-gpu=index,uuid,pci.bus_id,name,driver_version,compute_mode,persistence_mode,power.limit",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RunnerError("read-only physical GPU query failed")
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in next(csv.reader([line]))]
        if values and values[0] == str(GPU_INDEX):
            require(len(values) == 8, "physical GPU query column count")
            require(values[1] == GPU_UUID, "physical GPU UUID mismatch")
            return {
                "index": values[0],
                "uuid": values[1],
                "pci_bus_id": values[2],
                "name": values[3],
                "driver_version": values[4],
                "compute_mode": values[5],
                "persistence_mode": values[6],
                "power_limit_w": values[7],
            }
    raise RunnerError("target physical GPU absent")


def read_only_capture(command: list[str], output: Path) -> None:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RunnerError("read-only command failed: " + " ".join(command))


class TelemetryCollector:
    """A serialized query/state collector preventing PID attribution races."""

    def __init__(self, nv: str, output: Path) -> None:
        self.nv = nv
        self.output = output
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.phase = "prelaunch"
        self.active_pid: int | None = None
        self.known_pids: set[int] = set()
        self.sequence = 0
        self.samples: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.foreign: set[int] = set()
        self.thread = threading.Thread(target=self._loop, name="safe-c2-v3-telemetry", daemon=True)
        self.csv_path = output / "gpu_telemetry.csv"
        self.process_path = output / "gpu_process_samples.jsonl"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=20)
        if self.thread.is_alive():
            raise RunnerError("telemetry collector did not stop")

    def _set_state_locked(self, pid: int | None, phase: str) -> None:
        self.active_pid = pid
        self.phase = phase

    def launch_locked(self, command: list[str], cwd: Path, env: dict[str, str], stdout: Any, stderr: Any, invocation: int) -> tuple[subprocess.Popen[str], int]:
        # Caller gets the lock through this method.  No sampler query can
        # observe the spawned PID while it is still unregistered.
        with self.lock:
            self._set_state_locked(None, "launching_{:02d}".format(invocation))
            started = time.monotonic_ns()
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, text=True)
            self.known_pids.add(process.pid)
            self._set_state_locked(process.pid, "target_{:02d}".format(invocation))
            return process, started

    def mark_postexit(self, invocation: int) -> int:
        with self.lock:
            exited = time.monotonic_ns()
            self._set_state_locked(None, "postexit_{:02d}".format(invocation))
            self.condition.notify_all()
            return exited

    def _query_gpu_locked(self) -> tuple[list[str], str, list[dict[str, Any]], str]:
        gpu_command = [self.nv, "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"]
        gpu = subprocess.run(gpu_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        row: list[str] | None = None
        for line in gpu.stdout.splitlines():
            if not line.strip():
                continue
            values = [item.strip() for item in next(csv.reader([line]))]
            if values and values[0] == str(GPU_INDEX):
                row = values
                break
        if gpu.returncode != 0 or row is None or len(row) != len(GPU_FIELDS) or row[1] != GPU_UUID:
            self.errors.append("invalid GPU query at sample " + str(self.sequence))
            row = [""] * len(GPU_FIELDS)
            row[0] = str(GPU_INDEX)
            row[1] = GPU_UUID
        app_command = [
            self.nv,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
        apps = subprocess.run(app_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        processes: list[dict[str, Any]] = []
        if apps.returncode != 0:
            self.errors.append("invalid compute-process query at sample " + str(self.sequence))
        for line in apps.stdout.splitlines():
            if not line.strip():
                continue
            values = [item.strip() for item in next(csv.reader([line]))]
            if len(values) >= 2 and values[0] == GPU_UUID:
                try:
                    processes.append(
                        {
                            "pid": int(values[1]),
                            "process_name": values[2] if len(values) > 2 else "",
                            "used_memory": values[3] if len(values) > 3 else "",
                        }
                    )
                except ValueError:
                    self.errors.append("bad PID in compute-process query at sample " + str(self.sequence))
        return row, gpu.stdout, processes, apps.stdout

    def _sample(self, writer: csv.writer, process_handle: Any) -> None:
        # State and both queries are serialized with process launch registration.
        with self.lock:
            row, raw_gpu, processes, raw_processes = self._query_gpu_locked()
            mono = time.monotonic_ns()
            wall = time.time_ns()
            pids = [entry["pid"] for entry in processes]
            active = self.active_pid
            sole_target = active if active is not None and pids == [active] else None
            if sole_target is None:
                self.foreign.update(pids)
            else:
                # This is the only accepted compute-process state in a target sample.
                pass
            values = {
                "wall_utc_ns": wall,
                "monotonic_ns": mono,
                "sample_seq": self.sequence,
                "phase": self.phase,
                "gpu_index": row[0],
                "gpu_uuid": row[1],
                "pstate": row[8],
                "clock_sm_mhz": row[9],
                "clock_graphics_mhz": row[10],
                "clock_memory_mhz": row[11],
                "power_draw_w": row[12],
                "power_limit_w": row[7],
                "temperature_gpu_c": row[13],
                "utilization_gpu_pct": row[14],
                "utilization_memory_pct": row[15],
                "memory_used_mib": row[16],
            }
            event = {
                "wall_utc_ns": wall,
                "monotonic_ns": mono,
                "sample_seq": self.sequence,
                "phase": self.phase,
                "gpu_uuid": GPU_UUID,
                "target_pid": sole_target,
                "known_pids": sorted(self.known_pids),
                "compute_processes": processes,
                "raw_gpu_query_output": raw_gpu,
                "raw_compute_query_output": raw_processes,
            }
            writer.writerow([values[key] for key in CSV_HEADER])
            process_handle.write(json.dumps(event, sort_keys=True) + "\n")
            process_handle.flush()
            self.samples.append(values)
            self.events.append(event)
            self.sequence += 1
            self.condition.notify_all()

    def _loop(self) -> None:
        try:
            with self.csv_path.open("w", encoding="utf-8", newline="") as csv_file, self.process_path.open("w", encoding="utf-8") as process_file:
                writer = csv.writer(csv_file)
                writer.writerow(CSV_HEADER)
                csv_file.flush()
                next_time = time.monotonic()
                while not self.stop_event.is_set():
                    self._sample(writer, process_file)
                    csv_file.flush()
                    next_time += SAMPLE_PERIOD_S
                    self.stop_event.wait(max(0.0, next_time - time.monotonic()))
        except Exception as exc:
            with self.lock:
                self.errors.append("collector exception: " + type(exc).__name__ + ": " + str(exc))
                self.condition.notify_all()

    def wait_for_prelaunch_empty(self, count: int, timeout_s: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while True:
                empty = [
                    event
                    for event in self.events
                    if event["phase"] == "prelaunch" and event["target_pid"] is None and event["compute_processes"] == []
                ]
                if len(empty) >= count:
                    return
                if self.foreign:
                    raise RunnerError("foreign compute PID before launch: " + str(sorted(self.foreign)))
                if self.errors:
                    raise RunnerError("telemetry error before launch: " + "; ".join(self.errors))
                remain = deadline - time.monotonic()
                if remain <= 0:
                    raise RunnerError("timeout waiting for prelaunch empty telemetry")
                self.condition.wait(timeout=min(0.25, remain))

    def wait_for_postexit_empty(self, exit_ns: int, count: int, timeout_s: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while True:
                empty = [
                    event
                    for event in self.events
                    if event["monotonic_ns"] >= exit_ns and event["target_pid"] is None and event["compute_processes"] == []
                ]
                if len(empty) >= count:
                    return
                if self.foreign:
                    raise RunnerError("foreign/nonempty compute process after child exit: " + str(sorted(self.foreign)))
                if self.errors:
                    raise RunnerError("telemetry error after child exit: " + "; ".join(self.errors))
                remain = deadline - time.monotonic()
                if remain <= 0:
                    raise RunnerError("timeout waiting for postexit empty telemetry")
                self.condition.wait(timeout=min(0.25, remain))

    def summary(self) -> dict[str, Any]:
        with self.lock:
            require(bool(self.samples), "no telemetry samples")
            times = [item["monotonic_ns"] for item in self.samples]
            gaps = [(right - left) / 1_000_000.0 for left, right in zip(times, times[1:])]
            statistics_by_field: dict[str, dict[str, float | None]] = {}
            complete = True
            for key in NUMERIC_COLUMNS:
                values = [numeric(str(item[key])) for item in self.samples]
                if any(value is None for value in values):
                    complete = False
                usable = [value for value in values if value is not None]
                statistics_by_field[key] = {
                    "min": min(usable) if usable else None,
                    "max": max(usable) if usable else None,
                    "median": statistics.median(usable) if usable else None,
                }
            target_times: dict[str, list[int]] = {}
            for event in self.events:
                target = event["target_pid"]
                if target is not None:
                    target_times.setdefault(str(target), []).append(event["monotonic_ns"])
            return {
                "schema": "safe-c2-static-aabb-timeline-perf-telemetry-summary-v3",
                "sample_count": len(self.samples),
                "first_monotonic_ns": times[0],
                "last_monotonic_ns": times[-1],
                "max_inter_sample_gap_ms": max(gaps) if gaps else 0.0,
                "pstates": sorted({str(item["pstate"]) for item in self.samples if str(item["pstate"])}),
                "observed_target_pids": sorted(int(item) for item in target_times),
                "target_sample_times": target_times,
                "foreign_compute_pids": sorted(self.foreign),
                "query_errors": list(self.errors),
                "numeric_field_complete": complete,
                "stats": statistics_by_field,
            }

    def event_snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.events)


def parse_child_result(child: Path, expected_token: str, expected_pid: int, launch_start: int, launch_end: int) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    result_path = child / "result.json"
    timeline_path = child / "benchmark_timeline.json"
    result = parse_json_file(result_path, "child result")
    timeline = parse_json_file(timeline_path, "benchmark timeline")
    require(result.get("status") == "PASS_EXPLORATORY_PAIRED_PROBE_V2_DISKGUARD", "child result status")
    require(result.get("snapshot_pre_post_byte_equal") is True, "child snapshot equality flag")
    gate = result.get("unmeasured_gate")
    require(isinstance(gate, dict), "child unmeasured gate missing")
    for method in ("baseline", "static_aabb"):
        method_gate = gate.get(method)
        require(isinstance(method_gate, dict), "child gate missing for " + method)
        for key in ("invalid", "duplicate", "gt_set_mismatch", "distance_mismatch"):
            require(method_gate.get(key) == 0, "child gate " + method + "." + key)
    require(gate.get("id_sets_equal") is True, "child ID sets differ")
    timing = result.get("timing_gpu_ms")
    require(isinstance(timing, dict), "child timing object missing")
    baseline = timing.get("baseline")
    static = timing.get("static_aabb")
    require(isinstance(baseline, list) and isinstance(static, list), "child timing arrays missing")
    require(len(baseline) == REPS and len(static) == REPS, "child timing array length")
    for value in baseline + static:
        number(value, "child CUDA timing", positive=True)
    for suffix in SNAPSHOT_SUFFIXES:
        before = child / ("snapshot_before_" + suffix + ".bin")
        after = child / ("snapshot_after_" + suffix + ".bin")
        require(sha256_file(before) == sha256_file(after), "child snapshot mismatch for " + suffix)

    require(isinstance(timeline.get("schema"), str) and "v3" in timeline["schema"].lower(), "timeline schema")
    require(isinstance(timeline.get("status"), str) and timeline["status"].startswith("PASS"), "timeline status")
    require(timeline.get("run_token") == expected_token, "timeline run token mismatch")
    require(timeline.get("pid") == expected_pid, "timeline PID mismatch")
    require(isinstance(timeline.get("clock_id"), str) and "MONOTONIC" in timeline["clock_id"].upper(), "timeline clock")
    start = timeline.get("timed_envelope_start_ns")
    end = timeline.get("timed_envelope_end_ns")
    require(isinstance(start, int) and isinstance(end, int) and launch_start <= start < end <= launch_end, "timeline envelope/process lifetime")
    require(timeline.get("warmups") == WARMUPS and timeline.get("reps") == REPS, "timeline warmup/repetition contract")
    operations = timeline.get("operations")
    require(isinstance(operations, list) and len(operations) == REPS * 2, "timeline operation count")
    pairs: list[dict[str, Any]] = []
    values: dict[tuple[str, int], float] = {}
    previous_end = start
    for index, operation in enumerate(operations):
        require(isinstance(operation, dict), "timeline operation not object")
        rep = index // 2
        order = index % 2
        expected_method = ("baseline", "static_aabb")[order] if rep % 2 == 0 else ("static_aabb", "baseline")[order]
        require(operation.get("op_index") == index and operation.get("rep") == rep and operation.get("order") == order, "timeline operation indexing")
        require(operation.get("method") == expected_method, "timeline alternating order")
        host_start = operation.get("host_start_ns")
        host_end = operation.get("host_end_ns")
        require(isinstance(host_start, int) and isinstance(host_end, int) and previous_end <= host_start < host_end <= end, "timeline operation window")
        previous_end = host_end
        cuda_ms = number(operation.get("cuda_ms"), "timeline CUDA ms", positive=True)
        values[(expected_method, rep)] = cuda_ms
    require(set(values) == {(method, rep) for method in ("baseline", "static_aabb") for rep in range(REPS)}, "timeline method/rep coverage")
    for rep in range(REPS):
        same_number(values[("baseline", rep)], baseline[rep], "timeline/baseline timing " + str(rep))
        same_number(values[("static_aabb", rep)], static[rep], "timeline/static timing " + str(rep))
        ratio = float(static[rep]) / float(baseline[rep])
        pairs.append(
            {
                "rep": rep,
                "baseline_gpu_ms": baseline[rep],
                "static_aabb_gpu_ms": static[rep],
                "ratio_static_over_baseline": ratio,
                "static_over_baseline": ratio,
                "order": "baseline_then_static_aabb" if rep % 2 == 0 else "static_aabb_then_baseline",
            }
        )
    return result, timeline, pairs


def validate_telemetry_attribution(events: list[dict[str, Any]], launches: list[dict[str, Any]], children: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(launches) == INVOCATIONS and len(children) == INVOCATIONS, "telemetry attribution invocation count")
    by_invocation = {item["invocation"]: item for item in children}
    event_times = [event["monotonic_ns"] for event in events]
    require(bool(event_times) and event_times == sorted(event_times), "telemetry events are not monotonic")
    reports: list[dict[str, Any]] = []
    for launch in launches:
        invocation = launch["invocation"]
        timeline = by_invocation[invocation]["timeline_document"]
        pid = launch["pid"]
        start = timeline["timed_envelope_start_ns"]
        end = timeline["timed_envelope_end_ns"]
        enclosed = [event for event in events if start <= event["monotonic_ns"] <= end]
        require(bool(enclosed), "no telemetry inside timeline envelope " + str(invocation))
        for event in enclosed:
            require(event["target_pid"] == pid, "timeline envelope target mismatch " + str(invocation))
            require([item["pid"] for item in event["compute_processes"]] == [pid], "timeline envelope nonexclusive process " + str(invocation))
        target_times = [event["monotonic_ns"] for event in enclosed]
        gaps = [(right - left) / 1_000_000.0 for left, right in zip(target_times, target_times[1:])]
        require(target_times[0] <= start + int(MAX_GAP_MS * 1_000_000), "timeline start telemetry gap " + str(invocation))
        require(target_times[-1] >= end - int(MAX_GAP_MS * 1_000_000), "timeline end telemetry gap " + str(invocation))
        require(all(gap <= MAX_GAP_MS for gap in gaps), "timeline internal telemetry gap " + str(invocation))
        reports.append(
            {
                "invocation": invocation,
                "pid": pid,
                "run_token": launch["run_token"],
                "timeline_envelope_start_ns": start,
                "timeline_envelope_end_ns": end,
                "target_sample_count": len(enclosed),
                "first_target_sample_ns": target_times[0],
                "last_target_sample_ns": target_times[-1],
                "max_target_gap_ms": max(gaps) if gaps else 0.0,
                "status": "PASS_PID_EXCLUSIVE_TIMELINE_COVERAGE",
            }
        )
    first_start = launches[0]["target_start_monotonic_ns"]
    pre = [event for event in events if event["monotonic_ns"] < first_start]
    require(len(pre) >= PRELAUNCH_EMPTY_SAMPLES, "fewer than three prelaunch samples")
    require(all(event["target_pid"] is None and event["compute_processes"] == [] for event in pre), "nonempty prelaunch sample")
    post_reports: list[dict[str, Any]] = []
    for index, launch in enumerate(launches):
        next_start = launches[index + 1]["target_start_monotonic_ns"] if index + 1 < len(launches) else event_times[-1] + 1
        post = [
            event for event in events
            if launch["target_exit_monotonic_ns"] <= event["monotonic_ns"] < next_start
        ]
        require(len(post) >= POSTEXIT_EMPTY_SAMPLES, "fewer than three postexit samples " + str(launch["invocation"]))
        require(all(event["target_pid"] is None and event["compute_processes"] == [] for event in post), "nonempty postexit sample " + str(launch["invocation"]))
        post_reports.append({"invocation": launch["invocation"], "empty_postexit_samples": len(post), "status": "PASS"})
    return {
        "schema": "safe-c2-static-aabb-timeline-perf-telemetry-attribution-v3",
        "status": "PASS_STRICT_TIMELINE_PID_ATTRIBUTION",
        "formal_claim_eligible": False,
        "max_gap_ms": MAX_GAP_MS,
        "timeline_coverage": reports,
        "prelaunch_empty_samples": len(pre),
        "postexit_empty_samples": post_reports,
    }


def build_manifest(output: Path, aggregate: dict[str, Any]) -> dict[str, Any]:
    excluded = {"terminal.json", "manifest.json", "postrun_audit_verify.json"}
    artifacts: dict[str, Any] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative = str(path.relative_to(output))
            if relative not in excluded:
                artifacts[relative] = artifact_relative(output, path)
    return {
        "schema": "safe-c2-static-aabb-timeline-perf-manifest-v3",
        "status": "COMPLETE_CONDITIONAL_DVFS_UNCONTROLLED",
        "scope": aggregate["scope"],
        "formal_claim_eligible": False,
        "artifacts": artifacts,
    }


def write_terminal(output: Path, batch_id: str, status: str, exit_code: int, **extra: Any) -> None:
    value: dict[str, Any] = {
        "schema": "safe-c2-static-aabb-timeline-perf-terminal-v3",
        "status": status,
        "batch_id": batch_id,
        "finished_utc": utc(),
        "exit_code": exit_code,
        "formal_claim_eligible": False,
    }
    value.update(extra)
    atomic_json(output / "terminal.json", value)


def assert_authorization_is_unconsumed() -> None:
    """Fail closed if a previous or incomplete v3 prefix might have used this authorization."""
    current = artifact(AUTHORIZATION)
    if not RUNS.is_dir() or RUNS.is_symlink():
        raise RunnerError("runs directory missing or unsafe for authorization-consumption scan")
    for candidate in sorted(RUNS.glob("static_aabb_timeline_perf_v3_*")):
        if candidate.is_symlink() or not candidate.is_dir():
            raise RunnerError("ambiguous existing v3 run prefix: " + str(candidate))
        reference = candidate / "runner_authorization_reference.json"
        if not reference.exists() or reference.is_symlink() or not reference.is_file():
            raise RunnerError("ambiguous existing v3 run has no regular authorization reference: " + str(candidate))
        if sha256_file(reference) == current["sha256"]:
            raise RunnerError("one-batch authorization already consumed or ambiguous: " + str(candidate))


def verify_execution_gate_unchanged(preflight: dict[str, Any]) -> None:
    """Close the short CPU preflight-to-copy TOCTOU window before execution."""
    gate = preflight.get("execution_gate")
    require(isinstance(gate, dict), "preflight execution gate missing")
    current = {
        "runner_freeze": artifact(RUNNER_FREEZE),
        "authorization": artifact(AUTHORIZATION),
        "runner": artifact(runner_path()),
    }
    for key, record in current.items():
        require(gate.get(key) == record, "preflight execution gate changed for " + key)


def execute(preflight: dict[str, Any]) -> int:
    assert_authorization_is_unconsumed()
    batch_id = "static_aabb_timeline_perf_v3_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + secrets.token_hex(3)
    output = RUNS / batch_id
    if output.exists() or output.is_symlink():
        raise RunnerError("batch directory collision")
    output.mkdir(mode=0o700, parents=False)
    terminal = output / "terminal.json"
    atomic_json(
        terminal,
        {
            "schema": "safe-c2-static-aabb-timeline-perf-terminal-v3",
            "status": "PREPARING",
            "batch_id": batch_id,
            "started_utc": utc(),
            "formal_claim_eligible": False,
        },
    )
    collector: TelemetryCollector | None = None
    try:
        atomic_json(output / "preflight.json", preflight)
        verify_execution_gate_unchanged(preflight)
        shutil.copyfile(RUNNER_FREEZE, output / "runner_freeze_reference.json")
        shutil.copyfile(AUTHORIZATION, output / "runner_authorization_reference.json")

        nv = nvidia_smi()
        gpu_info = physical_gpu_info(nv)
        environment = {
            "schema": "safe-c2-static-aabb-timeline-perf-environment-v3",
            "hostname": os.uname().nodename,
            "uname": list(os.uname()),
            "target_gpu": gpu_info,
            "cuda_visible_devices": GPU_UUID,
            "clock_control": {
                "evidence_label": "DVFS_UNCONTROLLED_RECORDED",
                "runner_attempted_clock_lock": False,
                "runner_attempted_application_clocks": False,
                "runner_attempted_power_limit_change": False,
                "runner_attempted_compute_mode_change": False,
                "runner_attempted_persistence_mode_change": False,
            },
        }
        atomic_json(output / "environment.json", environment)
        read_only_capture([nv, "-q", "-x"], output / "gpu_state_before.xml")

        collector = TelemetryCollector(nv, output)
        collector.start()
        collector.wait_for_prelaunch_empty(PRELAUNCH_EMPTY_SAMPLES)

        environment_for_child = os.environ.copy()
        environment_for_child["CUDA_VISIBLE_DEVICES"] = GPU_UUID
        launches: list[dict[str, Any]] = []
        result_children: list[dict[str, Any]] = []
        timing_entries: list[dict[str, Any]] = []
        snapshot_lines: list[str] = []
        internal_children: list[dict[str, Any]] = []

        for invocation in range(1, INVOCATIONS + 1):
            child = output / ("invocation_{:02d}".format(invocation))
            child.mkdir(mode=0o700)
            token = secrets.token_hex(24)
            command = [
                shutil.which("nice") or "nice",
                "-n",
                "10",
                str(BINARY),
                "--base",
                str(BASE),
                "--queries",
                str(QUERY),
                "--groundtruth",
                str(GT),
                "--ids",
                str(IDS),
                "--outdir",
                str(child),
                "--run-token",
                token,
                "--projection-dims",
                "128",
                "--warmups",
                str(WARMUPS),
                "--reps",
                str(REPS),
            ]
            with (child / "stdout.log").open("w", encoding="utf-8") as stdout_file, (child / "stderr.log").open("w", encoding="utf-8") as stderr_file:
                process, started = collector.launch_locked(command, ROOT, environment_for_child, stdout_file, stderr_file, invocation)
                return_code = process.wait(timeout=CHILD_TIMEOUT_S)
            exited = collector.mark_postexit(invocation)
            launch = {
                "invocation": invocation,
                "pid": process.pid,
                "run_token": token,
                "target_start_monotonic_ns": started,
                "target_exit_monotonic_ns": exited,
                "returncode": return_code,
                "command": command,
            }
            launches.append(launch)
            if return_code != 0:
                raise RunnerError("child " + str(invocation) + " returned " + str(return_code))
            child_result, timeline, pairs = parse_child_result(child, token, process.pid, started, exited)
            collector.wait_for_postexit_empty(exited, POSTEXIT_EMPTY_SAMPLES)
            child_result_record = artifact_relative(output, child / "result.json")
            timeline_record = artifact_relative(output, child / "benchmark_timeline.json")
            result_children.append(
                {
                    "invocation": invocation,
                    "result": child_result_record,
                    "timeline": {"artifact": timeline_record, "run_token": token, "pid": process.pid},
                }
            )
            ratios = [item["ratio_static_over_baseline"] for item in pairs]
            timing_entries.append(
                {
                    "invocation": invocation,
                    "source_result_sha256": child_result_record["sha256"],
                    "raw_result": child_result_record,
                    "pairs": pairs,
                    "median_ratio_static_over_baseline": statistics.median(ratios),
                }
            )
            internal_children.append({"invocation": invocation, "timeline_document": timeline})
            for suffix in SNAPSHOT_SUFFIXES:
                snap = child / ("snapshot_after_" + suffix + ".bin")
                snapshot_lines.append("{:02d} {} {}\n".format(invocation, sha256_file(snap), snap.name))

        collector.stop()
        summary = collector.summary()
        events = collector.event_snapshot()
        collector = None
        require(summary["foreign_compute_pids"] == [], "foreign compute PID observed")
        require(summary["query_errors"] == [], "telemetry query error")
        require(summary["numeric_field_complete"] is True, "telemetry numeric field missing")
        require(summary["max_inter_sample_gap_ms"] <= MAX_GAP_MS, "telemetry sample gap exceeds threshold")
        require(summary["observed_target_pids"] == sorted(item["pid"] for item in launches), "observed target PID mismatch")
        attribution = validate_telemetry_attribution(events, launches, internal_children)

        read_only_capture([nv, "-q", "-x"], output / "gpu_state_after.xml")
        launch_document = {
            "schema": "safe-c2-static-aabb-timeline-perf-launch-v3",
            "target_gpu": gpu_info,
            "cuda_visible_devices": GPU_UUID,
            "binary": artifact(BINARY),
            "source": artifact(SOURCE),
            "heldout_ids": artifact(IDS),
            "base": artifact(BASE),
            "query": artifact(QUERY),
            "groundtruth": artifact(GT),
            "protocol": artifact(PROTOCOL),
            "runner": artifact(runner_path()),
            "runner_freeze": artifact(RUNNER_FREEZE),
            "authorization": artifact(AUTHORIZATION),
            "runner_attempted_clock_lock": False,
            "runner_attempted_application_clocks": False,
            "runner_attempted_power_limit_change": False,
            "runner_attempted_compute_mode_change": False,
            "launches": launches,
        }
        atomic_json(output / "launch.json", launch_document)
        (output / "snapshot_sha256.txt").write_text("".join(snapshot_lines), encoding="utf-8")
        medians = [entry["median_ratio_static_over_baseline"] for entry in timing_entries]
        timing_document = {
            "schema": "safe-c2-static-aabb-timeline-perf-timing-pairs-v3",
            "status": "COMPLETE_CONDITIONAL_DVFS_UNCONTROLLED",
            "formal_claim_eligible": False,
            "invocations": timing_entries,
            "aggregate_median_of_invocation_medians": statistics.median(medians),
            "range_of_invocation_medians": [min(medians), max(medians)],
        }
        atomic_json(output / "timing_pairs.json", timing_document)
        atomic_json(output / "telemetry_summary.json", summary)
        atomic_json(output / "telemetry_attribution.json", attribution)
        aggregate = {
            "schema": "safe-c2-static-aabb-timeline-perf-result-v3",
            "status": "COMPLETE_CONDITIONAL_DVFS_UNCONTROLLED",
            "batch_id": batch_id,
            "scope": "paired relative timing under recorded unlocked DVFS trace; not stable absolute latency or hardware-controlled evidence",
            "formal_claim_eligible": False,
            "evidence_grade": "CONDITIONAL_PAIRED_DVFS_UNCONTROLLED",
            "children": result_children,
            "timing_pairs": timing_document,
            "telemetry_summary": summary,
            "telemetry_attribution": artifact_relative(output, output / "telemetry_attribution.json"),
        }
        atomic_json(output / "result.json", aggregate)
        (output / "stdout.log").write_text("\n".join(json.dumps(item, sort_keys=True) for item in launches) + "\n", encoding="utf-8")
        (output / "stderr.log").write_text("", encoding="utf-8")

        manifest = build_manifest(output, aggregate)
        atomic_json(output / "manifest.json", manifest)
        write_terminal(
            output,
            batch_id,
            "COMPLETE_CONDITIONAL_DVFS_UNCONTROLLED",
            0,
            manifest=artifact_relative(output, output / "manifest.json"),
            result=artifact_relative(output, output / "result.json"),
        )
        expected_manifest_a = sha256_file(output / "manifest.json")
        first = subprocess.run(
            [
                sys.executable,
                str(POSTRUN_AUDITOR),
                "--run-dir",
                str(output),
                "--expected-hash",
                "manifest=" + expected_manifest_a,
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (output / "postrun_audit.json").write_text(first.stdout, encoding="utf-8")
        if first.returncode != 0:
            raise RunnerError("first postrun audit failed: " + first.stderr.strip())
        first_report = parse_json_file(output / "postrun_audit.json", "first postrun audit")
        require(first_report.get("status") == "PASS", "first postrun audit status")

        manifest = build_manifest(output, aggregate)
        atomic_json(output / "manifest.json", manifest)
        write_terminal(
            output,
            batch_id,
            "COMPLETE_CONDITIONAL_DVFS_UNCONTROLLED",
            0,
            manifest=artifact_relative(output, output / "manifest.json"),
            result=artifact_relative(output, output / "result.json"),
            postrun_audit=artifact_relative(output, output / "postrun_audit.json"),
        )
        expected_manifest_b = sha256_file(output / "manifest.json")
        second = subprocess.run(
            [
                sys.executable,
                str(POSTRUN_AUDITOR),
                "--run-dir",
                str(output),
                "--expected-hash",
                "manifest=" + expected_manifest_b,
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (output / "postrun_audit_verify.json").write_text(second.stdout, encoding="utf-8")
        if second.returncode != 0:
            raise RunnerError("verification postrun audit failed: " + second.stderr.strip())
        second_report = parse_json_file(output / "postrun_audit_verify.json", "verification postrun audit")
        require(second_report.get("status") == "PASS", "verification postrun audit status")
        print(json.dumps({"status": "COMPLETE_CONDITIONAL_DVFS_UNCONTROLLED", "run_dir": str(output)}, sort_keys=True))
        return 0
    except Exception as exc:
        if collector is not None:
            try:
                collector.stop()
            except Exception:
                pass
        try:
            write_terminal(
                output,
                batch_id,
                "FAILED",
                2,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        except Exception:
            pass
        print(json.dumps({"status": "FAILED", "run_dir": str(output), "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe-C2 v3 timeline performance runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true", help="CPU-only artifact and auditor checks")
    mode.add_argument("--selftest", action="store_true", help="CPU-only synthetic auditor check")
    mode.add_argument("--execute-authorized", action="store_true", help="execute exactly one explicitly authorized GPU batch")
    parser.add_argument(
        "--allow-unfrozen-audit",
        action="store_true",
        help="permit only --preflight-only to audit this runner before freeze and authorization exist",
    )
    args = parser.parse_args()
    if args.allow_unfrozen_audit and not args.preflight_only:
        raise RunnerError("--allow-unfrozen-audit is only valid with --preflight-only")
    if args.selftest:
        report = cpu_preflight(require_execution_gate=False)
        print(json.dumps({"schema": "safe-c2-static-aabb-timeline-perf-runner-selftest-v3", "status": "PASS_CPU_ONLY_SYNTHETIC_TIMELINE_VALIDATION", "formal_claim_eligible": False, "gpu_binary_executed": False, "nvidia_smi_called": False, "preflight_status": report["status"], "postrun_auditor_selftest": report["postrun_auditor_selftest"]}, sort_keys=True, indent=2))
        return 0
    if args.preflight_only:
        report = cpu_preflight(require_execution_gate=not args.allow_unfrozen_audit)
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    report = cpu_preflight(require_execution_gate=True)
    return execute(report)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL_STATIC_AABB_TIMELINE_PERF_RUNNER_V3: " + str(error), file=sys.stderr)
        raise SystemExit(2)
