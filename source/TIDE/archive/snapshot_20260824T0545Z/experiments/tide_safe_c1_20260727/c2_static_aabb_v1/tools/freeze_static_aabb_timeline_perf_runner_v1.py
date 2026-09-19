#!/usr/bin/env python3
"""
CPU-only freezer and one-batch authorizer for the v3 SIFT static-AABB timeline runner.

This program deliberately has no subprocess, CUDA, or GPU-management dependency. It
only validates frozen regular files and a CPU-only preflight JSON, then atomically
creates two provenance records. It does not execute the runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
PROVENANCE = ROOT / "provenance"
RUNNER_CANONICAL = ROOT / "tools/run_static_aabb_timeline_perf_v3.py"
PREFLIGHT_CANONICAL = PROVENANCE / "static_aabb_timeline_perf_runner_preflight_v1.json"
FREEZE = PROVENANCE / "static_aabb_timeline_perf_runner_freeze_v1.json"
AUTHORIZATION = PROVENANCE / "static_aabb_timeline_perf_runner_authorization_v1.json"

PROTOCOL = PROVENANCE / "static_aabb_timeline_perf_protocol_v3.json"
BINARY = ROOT / "bin/static_aabb_perf_probe_v3_timeline"
SOURCE = ROOT / "src/static_aabb_perf_probe_v3_timeline.cu"
SOURCE_AUDIT_RESULT = PROVENANCE / "static_aabb_perf_probe_v3_timeline_source_audit_v1.json"
SOURCE_AUDITOR = ROOT / "tools/audit_static_aabb_perf_probe_v3_timeline.py"
BUILD = PROVENANCE / "static_aabb_perf_probe_v3_timeline_build_v1.json"
IDS = ROOT / "inputs/standard_sift_static_aabb_perfheldout256_v2.ids"
SELECTION = PROVENANCE / "static_aabb_perfheldout_selection_v2.json"
DISK_CONTRACT = PROVENANCE / "sift_integer_disk_contract_v1.json"
POSTRUN_AUDITOR = ROOT / "tools/audit_static_aabb_heldout_perf_result_v3.py"

BASE = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs")
QUERY = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs")
GROUNDTRUTH = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs")

GPU_INDEX = 0
GPU_UUID = "GPU-CONFIGURE-ARCHIVE-DEVICE"
WARMUPS = 5
REPS = 30
INVOCATIONS = 5

EXPECTED = {
    "protocol": "93cdcee0bc4c854dfddea27a55ccc4d6840acd7eef9b4cbeda42ba99b76dce1e",
    "binary": "3b5ce855ba28085caf3424643d69b90d9590ec8b38738519c7e239f2e4bc8910",
    "source": "3f8396566500ea22dcfd02fb9437565d5f7df614a693426301de84bcb8424059",
    "source_audit_result": "f133828fc81dfd80dd0895ea184fa9dd1b7ffb44f6d8bb82bb7edcbdaa95c5ce",
    "source_auditor": "b7db883912e61af23dd79ac5fd10f66e5a676b70a6e1d65b99dac6aea06a23bf",
    "build": "ccd3446686cb68e950b27ef853b3e8849271288ae9ef770737f90befd64579aa",
    "ids": "990c423807d9b79eafee1491056c4b3a75243e70c1705439972ba2faecdda547",
    "selection": "b519557866646ba0085e1999c6861a1ed7c54c50de5257fe4b729bd642adc806",
    "disk_contract": "5c2f6b392d3fabffd323057033fbbcffd74604829aed2450094ba3f18757e435",
    "postrun_auditor": "ceeece94951a96ae6f928f014ba8b4c50f1a4cec0c462ec2e70d81080a357be4",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeError(message)


def regular(path: Path, label: str) -> Path:
    require(not path.is_symlink() and path.is_file(), label + " is not a regular non-symlink file")
    return path


def sha256_file(path: Path) -> str:
    regular(path, str(path))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    regular(path, str(path))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def artifact_for_payload(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def load_object(path: Path, label: str) -> dict[str, Any]:
    regular(path, label)
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise FreezeError(label + " is not valid JSON: " + str(exc)) from exc
    require(isinstance(value, dict), label + " is not a JSON object")
    return value


def require_bool_false(obj: dict[str, Any], key: str, label: str) -> None:
    require(obj.get(key) is False, label + "." + key + " must be false")


def require_record(
    record: Any,
    expected_path: Path,
    expected_sha: str,
    label: str,
    check_current: bool = True,
) -> dict[str, object]:
    require(isinstance(record, dict), label + " record absent")
    regular(expected_path, label + " expected file")
    require(record.get("path") == str(expected_path), label + " path binding")
    require(record.get("bytes") == expected_path.stat().st_size, label + " byte binding")
    require(record.get("sha256") == expected_sha, label + " declared SHA binding")
    if check_current:
        require(sha256_file(expected_path) == expected_sha, label + " current SHA binding")
    return {
        "path": str(expected_path),
        "bytes": expected_path.stat().st_size,
        "sha256": expected_sha,
    }


def require_path_sha(record: Any, expected: dict[str, object], label: str) -> None:
    require(isinstance(record, dict), label + " record absent")
    require(record.get("path") == expected.get("path"), label + " path mismatch")
    require(record.get("sha256") == expected.get("sha256"), label + " SHA mismatch")


def require_matching_record(record: Any, expected: dict[str, object], label: str) -> None:
    require(isinstance(record, dict), label + " record absent")
    require(record.get("path") == expected.get("path"), label + " path mismatch")
    require(record.get("bytes") == expected.get("bytes"), label + " byte mismatch")
    require(record.get("sha256") == expected.get("sha256"), label + " SHA mismatch")


def one_record(
    records: Any,
    keys: tuple[str, ...],
    expected_path: Path,
    expected_sha: str,
    label: str,
) -> dict[str, object]:
    require(isinstance(records, dict), "preflight artifacts absent")
    present = [key for key in keys if key in records]
    require(len(present) == 1, label + " artifact must appear exactly once")
    return require_record(records[present[0]], expected_path, expected_sha, label)


def canonical_path(raw: str, label: str) -> Path:
    candidate = Path(raw)
    require(candidate.is_absolute(), label + " must be an absolute path")
    try:
        return candidate.resolve(strict=True)
    except Exception as exc:
        raise FreezeError(label + " cannot be resolved: " + str(exc)) from exc


def require_canonical_argument(raw: str, expected: Path, label: str) -> Path:
    actual = canonical_path(raw, label)
    canonical_expected = expected.resolve(strict=True)
    require(actual == canonical_expected, label + " must be " + str(expected))
    require(not Path(raw).is_symlink(), label + " must not be a symlink")
    return expected


def checked_ids() -> list[int]:
    regular(IDS, "IDs")
    try:
        ids = [int(text) for text in IDS.read_text().splitlines() if text.strip()]
    except ValueError as exc:
        raise FreezeError("IDs are not integers") from exc
    require(len(ids) == 256, "IDs count")
    require(ids == sorted(ids), "IDs sort order")
    require(len(set(ids)) == 256, "IDs uniqueness")
    require(ids[0] == 5260 and ids[-1] == 5520, "IDs frozen endpoints")
    require(not any(5000 <= value <= 5259 for value in ids), "IDs predecessor overlap")
    return ids


def validate_protocol() -> dict[str, dict[str, object]]:
    require(sha256_file(PROTOCOL) == EXPECTED["protocol"], "protocol current SHA")
    protocol = load_object(PROTOCOL, "protocol")
    require(protocol.get("schema") == "safe-c2-static-aabb-timeline-perf-protocol-v3", "protocol schema")
    require(protocol.get("status") == "FROZEN_CPU_ONLY_PREPARED", "protocol status")
    require_bool_false(protocol, "formal_claim_eligible", "protocol")
    execution = protocol.get("execution_boundary")
    require(isinstance(execution, dict), "protocol execution boundary")
    require(execution.get("does_not_authorize_gpu_execution") is True, "protocol GPU authorization boundary")
    require(execution.get("gpu_executed_for_protocol_freeze") is False, "protocol GPU execution boundary")
    require(execution.get("nvidia_smi_called_for_protocol_freeze") is False, "protocol telemetry boundary")
    require(execution.get("old_c2_validation_or_sealed_accessed") is False, "protocol sealed boundary")
    require(execution.get("successor_output_prefix") == "static_aabb_timeline_perf_v3_", "protocol output prefix")

    algorithm = protocol.get("algorithm_binding")
    require(isinstance(algorithm, dict), "protocol algorithm binding")
    binary = require_record(algorithm.get("timeline_binary"), BINARY, EXPECTED["binary"], "protocol timeline binary")
    source = require_record(algorithm.get("timeline_source"), SOURCE, EXPECTED["source"], "protocol timeline source")
    source_audit_result = require_record(
        algorithm.get("source_audit"),
        SOURCE_AUDIT_RESULT,
        EXPECTED["source_audit_result"],
        "protocol timeline source audit result",
    )
    build = require_record(algorithm.get("build"), BUILD, EXPECTED["build"], "protocol build")
    disk_contract = require_record(
        algorithm.get("disk_contract"), DISK_CONTRACT, EXPECTED["disk_contract"], "protocol disk contract"
    )
    postrun = require_record(
        protocol.get("matching_postrun_auditor"),
        POSTRUN_AUDITOR,
        EXPECTED["postrun_auditor"],
        "protocol postrun auditor",
    )

    heldout = protocol.get("fresh_heldout_binding")
    require(isinstance(heldout, dict), "protocol fresh heldout binding")
    ids = require_record(heldout.get("ids"), IDS, EXPECTED["ids"], "protocol IDs")
    selection = require_record(heldout.get("selection"), SELECTION, EXPECTED["selection"], "protocol selection")
    require(heldout.get("count") == 256, "protocol ID count")
    require(heldout.get("first_id") == 5260 and heldout.get("last_id") == 5520, "protocol ID endpoints")

    datasets = protocol.get("dataset_binding")
    require(isinstance(datasets, dict), "protocol dataset binding")
    base_record = datasets.get("base")
    query_record = datasets.get("query")
    groundtruth_record = datasets.get("groundtruth")
    for label, record, path in (
        ("protocol base", base_record, BASE),
        ("protocol query", query_record, QUERY),
        ("protocol groundtruth", groundtruth_record, GROUNDTRUTH),
    ):
        require(isinstance(record, dict), label + " record absent")
        require(record.get("path") == str(path), label + " path")
        require(isinstance(record.get("bytes"), int) and record.get("bytes") > 0, label + " bytes")
        require(isinstance(record.get("sha256"), str) and SHA256_RE.fullmatch(record["sha256"]) is not None, label + " SHA")

    return {
        "protocol": artifact(PROTOCOL),
        "binary": binary,
        "source": source,
        "source_audit_result": source_audit_result,
        "source_auditor": artifact(SOURCE_AUDITOR),
        "build": build,
        "ids": ids,
        "selection": selection,
        "disk_contract": disk_contract,
        "postrun_auditor": postrun,
        "base": dict(base_record),
        "query": dict(query_record),
        "groundtruth": dict(groundtruth_record),
    }


def validate_upstream_cpu_only(bindings: dict[str, dict[str, object]]) -> None:
    require(bindings["source_auditor"]["sha256"] == EXPECTED["source_auditor"], "source auditor current SHA")

    audit = load_object(SOURCE_AUDIT_RESULT, "timeline source audit")
    require(audit.get("schema") == "safe-c2-static-aabb-perf-timeline-v3-source-audit-v1", "timeline source audit schema")
    require(audit.get("status") == "PASS_SOURCE_ONLY_AUDIT", "timeline source audit status")
    require_bool_false(audit, "gpu_executed", "timeline source audit")
    require_bool_false(audit, "nvidia_smi_called", "timeline source audit")
    require(audit.get("checks_passed") == 25, "timeline source audit check count")
    checks = audit.get("checks")
    require(isinstance(checks, dict) and len(checks) == 25 and all(value is True for value in checks.values()), "timeline source audit checks")
    require_path_sha(audit.get("source_v3"), bindings["source"], "timeline source audit source")

    build = load_object(BUILD, "timeline build")
    require(build.get("schema") == "safe-c2-static-aabb-perf-timeline-v3-build-v1", "timeline build schema")
    require(build.get("status") == "COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION", "timeline build status")
    require_bool_false(build, "gpu_binary_executed", "timeline build")
    require_bool_false(build, "nvidia_smi_called", "timeline build")
    require_bool_false(build, "formal_claim_eligible", "timeline build")
    require_matching_record(build.get("binary"), bindings["binary"], "timeline build binary")
    require_matching_record(build.get("source"), bindings["source"], "timeline build source")
    require_matching_record(build.get("audit_tool"), bindings["source_auditor"], "timeline build source auditor")
    audit_ref = build.get("source_audit")
    require(isinstance(audit_ref, dict), "timeline build source audit reference")
    require(audit_ref.get("path") == str(SOURCE_AUDIT_RESULT), "timeline build source audit path")
    require(audit_ref.get("result", {}).get("status") == "PASS_SOURCE_ONLY_AUDIT", "timeline build source audit status")

    selection = load_object(SELECTION, "fresh heldout selection")
    require(selection.get("schema") == "safe-c2-static-aabb-fresh-performance-heldout-selection-v2", "selection schema")
    require(selection.get("status") == "FROZEN_CPU_ONLY", "selection status")
    require_bool_false(selection, "gpu_executed", "selection")
    require_bool_false(selection, "nvidia_smi_called", "selection")
    require_matching_record(selection.get("output_ids"), bindings["ids"], "selection IDs")
    selected = selection.get("selected_ids")
    require(isinstance(selected, list) and selected == checked_ids(), "selection frozen IDs")


def validate_preflight(
    preflight_path: Path,
    runner_record: dict[str, object],
    bindings: dict[str, dict[str, object]],
) -> dict[str, Any]:
    preflight = load_object(preflight_path, "runner preflight")
    require(preflight.get("schema") == "safe-c2-static-aabb-timeline-perf-runner-preflight-v3", "preflight schema")
    require(preflight.get("status") == "PASS_CPU_ONLY_UNFROZEN_RUNNER_AUDIT", "preflight status")
    require_bool_false(preflight, "gpu_binary_executed", "preflight")
    require_bool_false(preflight, "nvidia_smi_called", "preflight")
    require_bool_false(preflight, "formal_claim_eligible", "preflight")
    require_matching_record(preflight.get("runner"), runner_record, "preflight runner")

    target = preflight.get("target_gpu")
    require(target == {"index": GPU_INDEX, "uuid": GPU_UUID}, "preflight target GPU")
    measurement = preflight.get("measurement")
    require(isinstance(measurement, dict), "preflight measurement")
    require(measurement.get("warmups_per_invocation") == WARMUPS, "preflight warmups")
    require(measurement.get("paired_repetitions_per_invocation") == REPS, "preflight paired repetitions")
    require(measurement.get("independent_process_invocations") == INVOCATIONS, "preflight invocations")

    records = preflight.get("artifacts")
    one_record(records, ("protocol",), PROTOCOL, EXPECTED["protocol"], "preflight protocol")
    one_record(records, ("binary",), BINARY, EXPECTED["binary"], "preflight binary")
    one_record(records, ("source",), SOURCE, EXPECTED["source"], "preflight source")
    one_record(records, ("ids",), IDS, EXPECTED["ids"], "preflight IDs")
    one_record(
        records,
        ("source_audit_result",),
        SOURCE_AUDIT_RESULT,
        EXPECTED["source_audit_result"],
        "preflight source audit result",
    )
    one_record(records, ("source_auditor",), SOURCE_AUDITOR, EXPECTED["source_auditor"], "preflight source auditor")
    one_record(records, ("build",), BUILD, EXPECTED["build"], "preflight build")
    one_record(records, ("selection",), SELECTION, EXPECTED["selection"], "preflight selection")
    one_record(records, ("postrun_auditor",), POSTRUN_AUDITOR, EXPECTED["postrun_auditor"], "preflight postrun auditor")

    require(isinstance(records, dict), "preflight artifacts")
    require_matching_record(records.get("base"), bindings["base"], "preflight base")
    require_matching_record(records.get("query"), bindings["query"], "preflight query")
    groundtruth_records = [records[key] for key in ("groundtruth", "gt") if key in records]
    require(len(groundtruth_records) == 1, "preflight groundtruth artifact must appear exactly once")
    require_matching_record(groundtruth_records[0], bindings["groundtruth"], "preflight groundtruth")

    source_audit = preflight.get("source_audit_now")
    require(isinstance(source_audit, dict), "preflight source audit result object")
    require(source_audit.get("status") == "PASS_SOURCE_ONLY_AUDIT", "preflight source audit result status")
    require_bool_false(source_audit, "gpu_executed", "preflight source audit result")
    require_bool_false(source_audit, "nvidia_smi_called", "preflight source audit result")
    require(source_audit.get("checks_passed") == 25, "preflight source audit result count")
    require_path_sha(source_audit.get("source_v3"), bindings["source"], "preflight source audit source")

    selftest = preflight.get("postrun_auditor_selftest")
    require(isinstance(selftest, dict), "preflight postrun auditor selftest")
    require(selftest.get("status") == "PASS_SYNTHETIC_CPU_SELFTEST", "preflight postrun auditor selftest status")
    require_bool_false(selftest, "gpu_binary_executed", "preflight postrun auditor selftest")
    require_bool_false(selftest, "nvidia_smi_called", "preflight postrun auditor selftest")
    require(selftest.get("positive_fixture") == "PASS", "preflight postrun auditor positive fixture")
    require(selftest.get("corruption_fixture") == "FAIL_CLOSED", "preflight postrun auditor corruption fixture")

    heldout = preflight.get("heldout_ids")
    require(isinstance(heldout, dict), "preflight heldout IDs")
    require(heldout.get("count") == 256 and heldout.get("sha256") == EXPECTED["ids"], "preflight heldout IDs binding")
    return preflight


def freeze_object(
    runner_record: dict[str, object],
    preflight_record: dict[str, object],
    bindings: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "schema": "safe-c2-static-aabb-timeline-perf-runner-freeze-v1",
        "status": "FROZEN_CPU_ONLY",
        "scope": "CPU-only freeze of the v3 runner and its audited fresh held-out timeline performance batch; no runner, CUDA binary, nvidia-smi, or GPU execution",
        "formal_claim_eligible": False,
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
        "runner": runner_record,
        "protocol": bindings["protocol"],
        "protocol_sha256": EXPECTED["protocol"],
        "binary": bindings["binary"],
        "ids": bindings["ids"],
        "source": bindings["source"],
        "postrun_auditor": bindings["postrun_auditor"],
        "algorithm_binding": {
            "timeline_binary": bindings["binary"],
            "timeline_source": bindings["source"],
            "timeline_source_audit_result": bindings["source_audit_result"],
            "timeline_source_auditor": bindings["source_auditor"],
            "build": bindings["build"],
            "disk_contract": bindings["disk_contract"],
            "postrun_auditor": bindings["postrun_auditor"],
        },
        "fresh_heldout_binding": {
            "ids": bindings["ids"],
            "selection": bindings["selection"],
            "count": 256,
            "first_id": 5260,
            "last_id": 5520,
            "disjoint_predecessor_observed_interval": [5000, 5259],
        },
        "unfrozen_cpu_preflight": preflight_record,
        "measurement": {
            "warmups": WARMUPS,
            "paired_repetitions": REPS,
            "independent_process_invocations": INVOCATIONS,
            "sample_period_target_ms": 100,
            "max_accepted_gap_ms": 500,
        },
        "target_gpu": {"index": GPU_INDEX, "uuid": GPU_UUID},
        "clock_control": {
            "evidence_label": "CONDITIONAL_PAIRED_DVFS_UNCONTROLLED",
            "runner_attempted_application_clocks": False,
            "runner_attempted_clock_lock": False,
            "runner_attempted_compute_mode_change": False,
            "runner_attempted_power_limit_change": False,
        },
        "execution_boundary": {
            "does_not_authorize_gpu_execution_by_itself": True,
            "requires_matching_authorization": True,
            "single_batch_only": True,
            "no_retry_under_this_authorization": True,
            "does_not_access_old_c2_validation_or_sealed": True,
        },
        "freezer": artifact(Path(__file__).resolve()),
    }


def authorization_object(
    runner_record: dict[str, object],
    preflight_record: dict[str, object],
    bindings: dict[str, dict[str, object]],
    freeze_record: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": "safe-c2-static-aabb-timeline-perf-runner-authorization-v1",
        "status": "CPU_AUDITED_READY_FOR_ONE_GPU_BATCH",
        "scope": "authorizes exactly one serial five-invocation fresh standard-SIFT1M static raw-float32 L2 AABB v3 timeline batch with read-only telemetry and unlocked-DVFS conditional evidence only",
        "formal_claim_eligible": False,
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
        "runner": runner_record,
        "protocol": bindings["protocol"],
        "protocol_sha256": EXPECTED["protocol"],
        "binary": bindings["binary"],
        "ids": bindings["ids"],
        "authorization": {
            "one_gpu_batch": True,
            "retry_permitted": False,
            "requires_matching_runner_freeze": True,
        },
        "algorithm_binding": {
            "timeline_binary": bindings["binary"],
            "timeline_source": bindings["source"],
            "timeline_source_audit_result": bindings["source_audit_result"],
            "timeline_source_auditor": bindings["source_auditor"],
            "build": bindings["build"],
            "disk_contract": bindings["disk_contract"],
            "postrun_auditor": bindings["postrun_auditor"],
        },
        "fresh_heldout_binding": {
            "ids": bindings["ids"],
            "selection": bindings["selection"],
            "count": 256,
            "first_id": 5260,
            "last_id": 5520,
        },
        "runner_preflight": preflight_record,
        "runner_freeze": freeze_record,
        "target_gpu": {"index": GPU_INDEX, "uuid": GPU_UUID},
        "measurement": {
            "warmups": WARMUPS,
            "paired_repetitions": REPS,
            "independent_process_invocations": INVOCATIONS,
            "telemetry_sample_target_ms": 100,
            "max_accepted_gap_ms": 500,
        },
        "evidence_label": "CONDITIONAL_PAIRED_DVFS_UNCONTROLLED",
        "single_batch_rule": "The next authorized execution must create exactly one fresh static_aabb_timeline_perf_v3_ batch. A failure remains FAILED; no retry is authorized by this record.",
        "forbidden_actions": [
            "application-clock lock or change",
            "power-limit change",
            "compute-mode change",
            "persistence-mode change",
            "process kill",
            "C3 insertion or update",
            "old C2 validation or sealed access",
            "reuse of predecessor failed timing values",
        ],
        "authorization_preconditions": [
            "runner source hash equals the frozen runner record",
            "runner verifies this authorization and matching freeze before any GPU operation",
            "all telemetry is read-only and bound to source-defined CLOCK_MONOTONIC timeline envelopes",
            "postrun verifier must pass before any result is interpreted",
        ],
        "freezer": artifact(Path(__file__).resolve()),
    }


def encode_json(obj: dict[str, object]) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode("utf-8")


def atomic_new(path: Path, payload: bytes) -> None:
    require(not path.exists() and not path.is_symlink(), "refusing pre-existing output: " + str(path))
    temp = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def validate_existing(path: Path, expected: dict[str, object], label: str) -> dict[str, object]:
    existing = load_object(path, label)
    require(existing == expected, label + " content differs from deterministic expected binding")
    return artifact(path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only freezer/authorizer for an already-audited v3 timeline performance runner"
    )
    parser.add_argument("--runner", required=True, help="canonical v3 runner path")
    parser.add_argument("--runner-sha256", required=True, help="expected lower-case SHA256 of runner source")
    parser.add_argument("--preflight", required=True, help="canonical CPU-only unfrozen preflight JSON")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate existing deterministic freeze and authorization without creating anything",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    require(SHA256_RE.fullmatch(args.runner_sha256) is not None, "runner SHA format")
    runner_path = require_canonical_argument(args.runner, RUNNER_CANONICAL, "runner")
    preflight_path = require_canonical_argument(args.preflight, PREFLIGHT_CANONICAL, "preflight")
    require(sha256_file(runner_path) == args.runner_sha256, "runner current SHA differs from supplied SHA")
    runner_record = artifact(runner_path)

    bindings = validate_protocol()
    validate_upstream_cpu_only(bindings)
    validate_preflight(preflight_path, runner_record, bindings)
    preflight_record = artifact(preflight_path)
    frozen = freeze_object(runner_record, preflight_record, bindings)
    freeze_payload = encode_json(frozen)
    freeze_record = artifact_for_payload(FREEZE, freeze_payload)
    authorized = authorization_object(runner_record, preflight_record, bindings, freeze_record)
    authorization_payload = encode_json(authorized)

    if args.verify_only:
        actual_freeze = validate_existing(FREEZE, frozen, "runner freeze")
        require_matching_record(actual_freeze, freeze_record, "runner freeze artifact")
        actual_authorization = validate_existing(AUTHORIZATION, authorized, "runner authorization")
        print(json.dumps({
            "status": "PASS_CPU_ONLY_FREEZE_AUTHORIZATION_VERIFY",
            "runner_freeze": actual_freeze,
            "runner_authorization": actual_authorization,
            "gpu_binary_executed": False,
            "nvidia_smi_called": False,
        }, sort_keys=True))
        return 0

    require(not FREEZE.exists() and not FREEZE.is_symlink(), "runner freeze output already exists")
    require(not AUTHORIZATION.exists() and not AUTHORIZATION.is_symlink(), "runner authorization output already exists")
    atomic_new(FREEZE, freeze_payload)
    atomic_new(AUTHORIZATION, authorization_payload)
    actual_freeze = artifact(FREEZE)
    actual_authorization = artifact(AUTHORIZATION)
    require_matching_record(actual_freeze, freeze_record, "written runner freeze")
    require_matching_record(actual_authorization, artifact_for_payload(AUTHORIZATION, authorization_payload), "written runner authorization")
    print(json.dumps({
        "status": "FROZEN_CPU_ONLY_AUTHORIZED_ONE_BATCH",
        "runner_freeze": actual_freeze,
        "runner_authorization": actual_authorization,
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except FreezeError as exc:
        print("FAIL_STATIC_AABB_TIMELINE_PERF_RUNNER_FREEZE_V1: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
