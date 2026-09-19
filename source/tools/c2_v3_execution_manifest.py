#!/usr/bin/env python3
"""CPU-only state machine for the Safe-C2 v3 future execution trust root.

This tool never invokes a CUDA binary, CUDA runtime/management tool, or GPU.  Its
ledger is deliberately irreversible: it has no reset/delete subcommand.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from c2_v3_execution_common import (
    ContractError,
    LEDGER_SCHEMA,
    MANIFEST_SCHEMA,
    ROOT_LITERAL,
    STAGE_ORDER,
    V2_SEALED_TEST_SHA256,
    WORKLOAD_SHA256,
    atomic_write_json,
    ensure_direct_child,
    expected_workflow_id,
    load_json,
    require_direct_dir,
    require_regular_file,
    root_from_module,
    safe_run_dir,
    sha256_file,
)

ROOT = root_from_module(__file__)
LEDGER = ROOT / "provenance/c2_v3_execution_consumption_ledger_v1.json"
LEDGER_LOCK = ROOT / ".locks/c2_v3_execution_consumption_ledger_v1.lock"
MANIFEST_NAME = "execution_manifest.json"
GAMMA_NAME = "final_v3_speculative_gamma_vector.txt"


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fail(message: str) -> None:
    raise ContractError(message)


def direct_run_existing(raw: str) -> Path:
    run = Path(raw)
    runs = ROOT / "runs"
    if not run.is_absolute() or run.parent != runs or not run.name.startswith("c2_v3_exec_"):
        fail(f"invalid v3 run path: {run}")
    require_direct_dir(runs, "runs directory", inside_root=True)
    require_direct_dir(run, "run directory", inside_root=True)
    return run


def manifest_path(run: Path) -> Path:
    path = run / MANIFEST_NAME
    require_regular_file(path, "execution manifest", inside_root=True)
    return path


def load_manifest(run: Path) -> dict[str, Any]:
    data = load_json(manifest_path(run), "execution manifest")
    if data.get("schema") != MANIFEST_SCHEMA:
        fail("unexpected execution manifest schema")
    if data.get("run_path") != str(run):
        fail("execution manifest run-path mismatch")
    return data


def file_descriptor(path: Path, label: str, *, inside_root: bool = True) -> dict[str, Any]:
    require_regular_file(path, label, inside_root=inside_root)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def descriptor_matches(value: Any, path: Path, label: str) -> None:
    if not isinstance(value, dict):
        fail(f"missing descriptor: {label}")
    expected = file_descriptor(path, label)
    if value != expected:
        fail(f"descriptor mismatch: {label}")


def plan_pins_descriptors(plan_raw: str, pins_raw: str) -> tuple[Path, dict[str, Any], Path, dict[str, Any], str]:
    plan = Path(plan_raw)
    pins = Path(pins_raw)
    plan_desc = file_descriptor(plan, "execution plan")
    pins_desc = file_descriptor(pins, "execution PINS")
    workflow_id = expected_workflow_id(plan_desc["sha256"], pins_desc["sha256"], WORKLOAD_SHA256)
    return plan, plan_desc, pins, pins_desc, workflow_id


@contextmanager
def locked_ledger() -> Iterator[None]:
    require_direct_dir(LEDGER_LOCK.parent, "ledger-lock parent", inside_root=True)
    if LEDGER_LOCK.is_symlink():
        fail("ledger lock must not be a symlink")
    handle = LEDGER_LOCK.open("a+", encoding="utf-8")
    try:
        os.chmod(LEDGER_LOCK, 0o640)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def load_ledger_required() -> dict[str, Any]:
    require_regular_file(LEDGER, "irrevocable consumption ledger", inside_root=True)
    ledger = load_json(LEDGER, "irrevocable consumption ledger")
    if ledger.get("schema") != LEDGER_SCHEMA:
        fail("unexpected consumption ledger schema")
    if set(ledger.get("stages", {})) != set(STAGE_ORDER):
        fail("ledger stage set mismatch")
    return ledger


def require_ledger_binding(ledger: dict[str, Any], run: Path, plan_desc: dict[str, Any], pins_desc: dict[str, Any], workflow_id: str) -> None:
    if ledger.get("workflow_id") != workflow_id:
        fail("ledger workflow-ID mismatch")
    if ledger.get("run_path") != str(run):
        fail("ledger run-path mismatch; cannot create a new workflow")
    if ledger.get("plan") != plan_desc or ledger.get("pins") != pins_desc:
        fail("ledger plan/PINS binding mismatch")
    workload = ledger.get("workload")
    if not isinstance(workload, dict) or workload.get("sha256") != WORKLOAD_SHA256:
        fail("ledger workload binding mismatch")


def recursive_stage_artifacts(run: Path, stage_dir: Path) -> list[dict[str, Any]]:
    require_direct_dir(stage_dir, "stage output directory", inside_root=True)
    out: list[dict[str, Any]] = []
    for path in sorted(stage_dir.rglob("*")):
        if path.is_symlink():
            fail(f"stage output contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            fail(f"stage output contains non-regular entry: {path}")
        ensure_direct_child(path, stage_dir, "stage artifact")
        out.append({
            "relative_path": str(path.relative_to(run)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    if not out:
        fail("stage output is empty")
    return out


def command_claim_workflow(args: argparse.Namespace) -> int:
    run = safe_run_dir(ROOT, args.run)
    _, plan_desc, _, pins_desc, workflow_id = plan_pins_descriptors(args.plan, args.pins)
    workload = Path(args.workload_manifest)
    require_regular_file(workload, "active workload manifest", inside_root=True)
    if sha256_file(workload) != WORKLOAD_SHA256:
        fail("active workload manifest SHA-256 mismatch at workflow claim")
    if LEDGER.exists() or LEDGER.is_symlink():
        fail("irrevocable workflow ledger already exists; no new workflow/retry is permitted")
    with locked_ledger():
        if LEDGER.exists() or LEDGER.is_symlink():
            fail("irrevocable workflow ledger already exists after lock acquisition")
        ledger: dict[str, Any] = {
            "schema": LEDGER_SCHEMA,
            "status": "WORKFLOW_CLAIMED",
            "workflow_id": workflow_id,
            "created_utc": now_utc(),
            "run_path": str(run),
            "plan": plan_desc,
            "pins": pins_desc,
            "workload": {"path": str(workload), "sha256": WORKLOAD_SHA256},
            "v2_sealed_test_sha256_forbidden": V2_SEALED_TEST_SHA256,
            "stages": {stage: {"state": "UNCLAIMED"} for stage in STAGE_ORDER},
            "events": [{"event": "WORKFLOW_CLAIMED", "utc": now_utc()}],
            "no_reset_or_retry": True,
        }
        atomic_write_json(LEDGER, ledger)
    print(json.dumps({"status": "PASS_WORKFLOW_CLAIMED", "workflow_id": workflow_id, "ledger": str(LEDGER)}, sort_keys=True))
    return 0


def command_claim_stage(args: argparse.Namespace) -> int:
    run = Path(args.run)
    runs = ROOT / "runs"
    if args.stage not in STAGE_ORDER:
        fail("unknown execution stage")
    # Calibration is consumed before *any* run/output mkdir.  Held-out stages
    # are consumed before their own output mkdir/telemetry, while the already
    # claimed calibration run directory necessarily exists.
    if args.stage == "calibration":
        if not run.is_absolute() or run.parent != runs or not run.name.startswith("c2_v3_exec_") or run.exists() or run.is_symlink():
            fail("calibration claim must happen before output/run mkdir on a fresh direct run path")
    else:
        run = direct_run_existing(str(run))
        heldout_manifest = load_manifest(run)
        pending = heldout_manifest.get("stages", {}).get(args.stage)
        if heldout_manifest.get("status") != "RUNNING" or not isinstance(pending, dict) or pending.get("state") != "PENDING":
            fail("held-out stage must be pending in a running manifest before its claim")
        stage_output = run / args.stage
        if stage_output.exists() or stage_output.is_symlink():
            fail("held-out stage claim must happen before its output mkdir")
    _, plan_desc, _, pins_desc, workflow_id = plan_pins_descriptors(args.plan, args.pins)
    gamma_desc: dict[str, Any] | None = None
    if args.stage == "calibration":
        if args.gamma_file:
            fail("calibration must not have a gamma input")
    else:
        if not args.gamma_file:
            fail("held-out stage requires calibration gamma input")
        gamma = Path(args.gamma_file)
        expected_gamma = run / "calibration" / GAMMA_NAME
        if gamma != expected_gamma:
            fail("held-out stage gamma must be exactly the calibration output in the same fresh run")
        require_regular_file(gamma, "calibration gamma", inside_root=True)
        gamma_desc = file_descriptor(gamma, "calibration gamma")
    with locked_ledger():
        ledger = load_ledger_required()
        require_ledger_binding(ledger, run, plan_desc, pins_desc, workflow_id)
        stages = ledger["stages"]
        current = stages[args.stage]
        if current.get("state") != "UNCLAIMED":
            fail(f"stage already claimed/consumed: {args.stage}")
        prior = {"calibration": None, "validation": "calibration", "sealed_test": "validation"}[args.stage]
        if prior is not None:
            prior_value = stages[prior]
            if prior_value.get("state") != "PASS":
                fail(f"cannot claim {args.stage}; prior stage is not PASS")
            prior_gamma = stages["calibration"].get("produced_gamma")
            if not isinstance(prior_gamma, dict) or gamma_desc != prior_gamma:
                fail("held-out stage gamma does not exactly match calibration ledger descriptor")
        stages[args.stage] = {
            "state": "ATTEMPTED",
            "claimed_utc": now_utc(),
            "gamma_input": gamma_desc,
            "attempt_is_irrevocable": True,
        }
        ledger["status"] = f"{args.stage.upper()}_ATTEMPTED"
        ledger["events"].append({"event": "STAGE_ATTEMPTED", "stage": args.stage, "utc": now_utc(), "gamma_input": gamma_desc})
        atomic_write_json(LEDGER, ledger)
    print(json.dumps({"status": "PASS_STAGE_CLAIMED", "stage": args.stage, "workflow_id": workflow_id}, sort_keys=True))
    return 0


def command_init_run(args: argparse.Namespace) -> int:
    run = safe_run_dir(ROOT, args.run)
    plan, plan_desc, pins, pins_desc, workflow_id = plan_pins_descriptors(args.plan, args.pins)
    guard = Path(args.guard)
    launcher = Path(args.launcher)
    guard_desc = file_descriptor(guard, "outer guard")
    launcher_desc = file_descriptor(launcher, "launcher")
    with locked_ledger():
        ledger = load_ledger_required()
        require_ledger_binding(ledger, run, plan_desc, pins_desc, workflow_id)
        if ledger["stages"]["calibration"].get("state") != "ATTEMPTED":
            fail("run initialization requires already-claimed calibration")
    os.mkdir(run, 0o750)
    require_direct_dir(run, "new run directory", inside_root=True)
    data: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "RUNNING",
        "created_utc": now_utc(),
        "run_path": str(run),
        "workflow_id": workflow_id,
        "host": args.host,
        "plan": plan_desc,
        "pins": pins_desc,
        "guard": guard_desc,
        "launcher": launcher_desc,
        "workload": {"path": args.workload_manifest, "sha256": WORKLOAD_SHA256},
        "v2_sealed_test_sha256_forbidden": V2_SEALED_TEST_SHA256,
        "stages": {stage: {"state": "PENDING"} for stage in STAGE_ORDER},
        "telemetry": [],
        "events": [{"event": "RUN_INITIALIZED_AFTER_CALIBRATION_CLAIM", "utc": now_utc()}],
        "no_retry": True,
    }
    atomic_write_json(run / MANIFEST_NAME, data)
    print(json.dumps({"status": "PASS_RUN_INITIALIZED", "run": str(run), "workflow_id": workflow_id}, sort_keys=True))
    return 0


def command_stage_start(args: argparse.Namespace) -> int:
    run = direct_run_existing(args.run)
    if args.stage not in STAGE_ORDER:
        fail("unknown execution stage")
    manifest = load_manifest(run)
    if manifest.get("status") != "RUNNING":
        fail("cannot start a stage unless manifest is RUNNING")
    stage_state = manifest.get("stages", {}).get(args.stage)
    if not isinstance(stage_state, dict) or stage_state.get("state") != "PENDING":
        fail("stage is not pending in run manifest")
    expected_output = run / args.stage
    output = Path(args.output)
    if output != expected_output or output.exists() or output.is_symlink():
        fail("stage output must be a fresh direct canonical stage directory")
    if not args.argv or not Path(args.argv[0]).is_absolute():
        fail("stage command argv must begin with an absolute direct binary path")
    with locked_ledger():
        ledger = load_ledger_required()
        if ledger.get("workflow_id") != manifest.get("workflow_id") or ledger.get("run_path") != str(run):
            fail("ledger/run manifest workflow binding mismatch")
        if ledger["stages"][args.stage].get("state") != "ATTEMPTED":
            fail("stage was not irrevocably claimed before output mkdir")
    os.mkdir(output, 0o750)
    require_direct_dir(output, "new stage output directory", inside_root=True)
    manifest["stages"][args.stage] = {
        "state": "RUNNING",
        "output_dir": str(output),
        "command_argv": list(args.argv),
        "session": {"physical_gpu_index": 0, "uuid": args.gpu_uuid, "pci_bus_id": args.pci_bus_id},
        "started_utc": now_utc(),
    }
    manifest["events"].append({"event": "STAGE_STARTED_AFTER_IRREVOCABLE_CLAIM", "stage": args.stage, "utc": now_utc()})
    atomic_write_json(run / MANIFEST_NAME, manifest)
    print(json.dumps({"status": "PASS_STAGE_STARTED", "stage": args.stage, "output": str(output)}, sort_keys=True))
    return 0


def command_record_runtime_binding(args: argparse.Namespace) -> int:
    run = direct_run_existing(args.run)
    manifest = load_manifest(run)
    if manifest.get("status") != "RUNNING":
        fail("runtime binding requires a RUNNING manifest")
    env_file = Path(args.environment_file)
    ensure_direct_child(env_file, run, "runtime environment file")
    env_value = load_json(env_file, "runtime environment file")
    if env_value.get("schema") != "safe-c2-v3-child-environment-v1":
        fail("runtime environment schema mismatch")
    environment = env_value.get("environment")
    required = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/root",
        "LANG": "C",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
        "NVIDIA_VISIBLE_DEVICES": args.gpu_uuid,
        "LD_LIBRARY_PATH": "/usr/local/cuda-13.1/lib64",
    }
    if environment != required:
        fail("runtime environment whitelist mismatch")
    if not args.gpu_uuid.startswith("GPU-") or not args.pci_bus_id:
        fail("runtime UUID/PCI binding malformed")
    manifest["runtime_binding"] = {
        "physical_gpu_index": 0,
        "physical_gpu_uuid": args.gpu_uuid,
        "pci_bus_id": args.pci_bus_id,
        "environment_file": file_descriptor(env_file, "runtime environment file"),
        "environment": environment,
        "recorded_utc": now_utc(),
    }
    manifest["events"].append({"event": "RUNTIME_UUID_PCI_ENVIRONMENT_BOUND", "utc": now_utc()})
    atomic_write_json(run / MANIFEST_NAME, manifest)
    print(json.dumps({"status": "PASS_RUNTIME_BINDING_RECORDED"}, sort_keys=True))
    return 0


def command_record_telemetry(args: argparse.Namespace) -> int:
    run = direct_run_existing(args.run)
    manifest = load_manifest(run)
    if manifest.get("status") not in {"RUNNING", "FAILED", "COMPLETE"}:
        fail("unexpected manifest status for telemetry")
    telemetry = Path(args.file)
    ensure_direct_child(telemetry, run, "telemetry file")
    descriptor = file_descriptor(telemetry, "telemetry file")
    manifest["telemetry"].append({"label": args.label, **descriptor, "recorded_utc": now_utc()})
    atomic_write_json(run / MANIFEST_NAME, manifest)
    print(json.dumps({"status": "PASS_TELEMETRY_RECORDED", "label": args.label}, sort_keys=True))
    return 0


def _read_verifier_pass(path: Path) -> dict[str, Any]:
    require_regular_file(path, "artifact verifier output", inside_root=True)
    value = load_json(path, "artifact verifier output")
    status = value.get("status")
    if not isinstance(status, str) or not status.startswith("PASS"):
        fail("artifact verifier did not report PASS")
    return value


def command_record_child(args: argparse.Namespace) -> int:
    run = direct_run_existing(args.run)
    if args.stage not in STAGE_ORDER:
        fail("unknown execution stage")
    manifest = load_manifest(run)
    state = manifest.get("stages", {}).get(args.stage)
    if not isinstance(state, dict) or state.get("state") != "RUNNING":
        fail("child record requires a RUNNING stage")
    if args.pid <= 0 or args.returncode < 0:
        fail("child PID/return code malformed")
    state["child"] = {"pid": args.pid, "returncode": args.returncode, "recorded_utc": now_utc()}
    manifest["events"].append({"event": "DIRECT_PINNED_BINARY_CHILD_EXIT", "stage": args.stage, "pid": args.pid, "returncode": args.returncode, "utc": now_utc()})
    atomic_write_json(run / MANIFEST_NAME, manifest)
    print(json.dumps({"status": "PASS_CHILD_RECORDED", "stage": args.stage, "pid": args.pid, "returncode": args.returncode}, sort_keys=True))
    return 0


def command_stage_complete(args: argparse.Namespace) -> int:
    run = direct_run_existing(args.run)
    if args.stage not in STAGE_ORDER:
        fail("unknown execution stage")
    manifest = load_manifest(run)
    stage = manifest.get("stages", {}).get(args.stage)
    if not isinstance(stage, dict) or stage.get("state") != "RUNNING":
        fail("only a running stage can be completed")
    output = Path(stage.get("output_dir", ""))
    if output != run / args.stage:
        fail("stage output path mismatch")
    verifier = Path(args.verification)
    ensure_direct_child(verifier, run, "stage verifier output")
    verifier_value = _read_verifier_pass(verifier)
    stdout_log = Path(args.stdout_log)
    stderr_log = Path(args.stderr_log)
    ensure_direct_child(stdout_log, run, "child stdout log")
    ensure_direct_child(stderr_log, run, "child stderr log")
    stdout_desc = file_descriptor(stdout_log, "child stdout log")
    stderr_desc = file_descriptor(stderr_log, "child stderr log")
    artifacts = recursive_stage_artifacts(run, output)
    gamma_desc: dict[str, Any] | None = None
    if args.stage == "calibration":
        gamma = output / GAMMA_NAME
        gamma_desc = file_descriptor(gamma, "produced calibration gamma")
    with locked_ledger():
        ledger = load_ledger_required()
        if ledger.get("workflow_id") != manifest.get("workflow_id") or ledger.get("run_path") != str(run):
            fail("ledger/run manifest workflow mismatch at stage completion")
        entry = ledger["stages"][args.stage]
        if entry.get("state") != "ATTEMPTED":
            fail("ledger stage state does not permit completion")
        if args.stage != "calibration":
            if entry.get("gamma_input") != ledger["stages"]["calibration"].get("produced_gamma"):
                fail("held-out ledger gamma provenance mismatch")
        entry["state"] = "PASS"
        entry["completed_utc"] = now_utc()
        entry["verification"] = file_descriptor(verifier, "stage verifier output")
        if gamma_desc is not None:
            entry["produced_gamma"] = gamma_desc
        ledger["status"] = f"{args.stage.upper()}_PASS"
        ledger["events"].append({"event": "STAGE_PASS", "stage": args.stage, "utc": now_utc()})
        atomic_write_json(LEDGER, ledger)
    stage.update({
        "state": "PASS",
        "completed_utc": now_utc(),
        "artifact_verifier": file_descriptor(verifier, "stage verifier output"),
        "child_logs": {"stdout": stdout_desc, "stderr": stderr_desc},
        "artifacts": artifacts,
    })
    if gamma_desc is not None:
        stage["produced_gamma"] = gamma_desc
    manifest["events"].append({"event": "STAGE_PASS", "stage": args.stage, "utc": now_utc(), "verifier_schema": verifier_value.get("schema")})
    atomic_write_json(run / MANIFEST_NAME, manifest)
    print(json.dumps({"status": "PASS_STAGE_COMPLETED", "stage": args.stage, "artifact_count": len(artifacts)}, sort_keys=True))
    return 0


def command_stage_fail(args: argparse.Namespace) -> int:
    run = Path(args.run)
    if args.stage not in STAGE_ORDER:
        fail("unknown execution stage")
    _, plan_desc, _, pins_desc, workflow_id = plan_pins_descriptors(args.plan, args.pins)
    with locked_ledger():
        ledger = load_ledger_required()
        # The caller may fail before run initialization; validate only fixed bindings then consume failure.
        if ledger.get("workflow_id") != workflow_id or ledger.get("run_path") != str(run) or ledger.get("plan") != plan_desc or ledger.get("pins") != pins_desc:
            fail("cannot fail unrelated ledger/workflow")
        entry = ledger["stages"][args.stage]
        if entry.get("state") == "PASS":
            fail("cannot convert PASS stage to failure")
        if entry.get("state") != "ATTEMPTED":
            fail("cannot mark an unclaimed stage as failed")
        entry["state"] = "FAILED"
        entry["failed_utc"] = now_utc()
        entry["failure_reason"] = args.reason
        ledger["status"] = f"{args.stage.upper()}_FAILED_CONSUMED"
        ledger["events"].append({"event": "STAGE_FAILED_CONSUMED", "stage": args.stage, "utc": now_utc(), "reason": args.reason})
        atomic_write_json(LEDGER, ledger)
    if run.exists() and not run.is_symlink():
        manifest = load_manifest(run)
        state = manifest.get("stages", {}).get(args.stage)
        if isinstance(state, dict):
            state["state"] = "FAILED"
            state["failure_reason"] = args.reason
            state["failed_utc"] = now_utc()
        manifest["status"] = "FAILED"
        manifest["events"].append({"event": "STAGE_FAILED_CONSUMED", "stage": args.stage, "utc": now_utc(), "reason": args.reason})
        atomic_write_json(run / MANIFEST_NAME, manifest)
    print(json.dumps({"status": "PASS_STAGE_FAILURE_RECORDED_AS_CONSUMED", "stage": args.stage}, sort_keys=True))
    return 0


def command_record_cleanup(args: argparse.Namespace) -> int:
    run = direct_run_existing(args.run)
    manifest = load_manifest(run)
    if manifest.get("status") not in {"RUNNING", "FAILED"}:
        fail("cleanup provenance may be recorded only for a running/failed manifest")
    cleanup = Path(args.file)
    ensure_direct_child(cleanup, run, "cleanup provenance file")
    value = load_json(cleanup, "cleanup provenance file")
    if value.get("schema") != "safe-c2-v3-own-child-cleanup-v1":
        fail("cleanup provenance schema mismatch")
    descriptor_value = file_descriptor(cleanup, "cleanup provenance file")
    manifest.setdefault("cleanup_provenance", []).append({"stage": args.stage, **descriptor_value, "recorded_utc": now_utc()})
    manifest["events"].append({"event": "OWN_CHILD_CLEANUP_PROVENANCE_RECORDED", "stage": args.stage, "utc": now_utc()})
    atomic_write_json(run / MANIFEST_NAME, manifest)
    print(json.dumps({"status": "PASS_CLEANUP_PROVENANCE_RECORDED", "stage": args.stage}, sort_keys=True))
    return 0


def command_finalize(args: argparse.Namespace) -> int:
    run = direct_run_existing(args.run)
    manifest = load_manifest(run)
    if manifest.get("status") != "RUNNING":
        fail("only a RUNNING manifest can finalize")
    if any(manifest.get("stages", {}).get(stage, {}).get("state") != "PASS" for stage in STAGE_ORDER):
        fail("cannot finalize without all stage PASS states")
    verifier = Path(args.verification)
    ensure_direct_child(verifier, run, "final verifier output")
    _read_verifier_pass(verifier)
    with locked_ledger():
        ledger = load_ledger_required()
        if ledger.get("workflow_id") != manifest.get("workflow_id") or ledger.get("run_path") != str(run):
            fail("ledger/run manifest workflow mismatch at finalization")
        if any(ledger["stages"][stage].get("state") != "PASS" for stage in STAGE_ORDER):
            fail("ledger lacks all PASS stage states")
        ledger["status"] = "COMPLETE"
        ledger["completed_utc"] = now_utc()
        ledger["events"].append({"event": "WORKFLOW_COMPLETE", "utc": now_utc()})
        atomic_write_json(LEDGER, ledger)
    manifest["status"] = "COMPLETE"
    manifest["completed_utc"] = now_utc()
    manifest["final_artifact_verifier"] = file_descriptor(verifier, "final verifier output")
    manifest["events"].append({"event": "RUN_COMPLETE", "utc": now_utc()})
    atomic_write_json(run / MANIFEST_NAME, manifest)
    print(json.dumps({"status": "PASS_RUN_FINALIZED", "run": str(run)}, sort_keys=True))
    return 0


def command_workflow_id(args: argparse.Namespace) -> int:
    _, plan_desc, _, pins_desc, workflow_id = plan_pins_descriptors(args.plan, args.pins)
    print(json.dumps({"status": "PASS_WORKFLOW_ID_DERIVED", "workflow_id": workflow_id, "plan_sha256": plan_desc["sha256"], "pins_sha256": pins_desc["sha256"], "workload_sha256": WORKLOAD_SHA256}, sort_keys=True))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    def shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run", required=True)
        p.add_argument("--plan", required=True)
        p.add_argument("--pins", required=True)
    p = sub.add_parser("claim-workflow", help="atomically create the single irrevocable root workflow ledger")
    shared(p); p.add_argument("--workload-manifest", required=True); p.set_defaults(func=command_claim_workflow)
    p = sub.add_parser("claim-stage", help="atomically consume a stage before output mkdir or GPU telemetry")
    shared(p); p.add_argument("--stage", required=True, choices=STAGE_ORDER); p.add_argument("--gamma-file", default=""); p.set_defaults(func=command_claim_stage)
    p = sub.add_parser("init-run", help="create a run manifest only after calibration was claimed")
    shared(p); p.add_argument("--guard", required=True); p.add_argument("--launcher", required=True); p.add_argument("--host", required=True); p.add_argument("--workload-manifest", required=True); p.set_defaults(func=command_init_run)
    p = sub.add_parser("stage-start", help="create canonical empty stage output only after an irrevocable claim")
    p.add_argument("--run", required=True); p.add_argument("--stage", required=True, choices=STAGE_ORDER); p.add_argument("--output", required=True); p.add_argument("--gpu-uuid", required=True); p.add_argument("--pci-bus-id", required=True); p.add_argument("--argv", nargs=argparse.REMAINDER, required=True); p.set_defaults(func=command_stage_start)
    p = sub.add_parser("record-runtime-binding", help="bind the exact UUID/PCI and env -i child whitelist")
    p.add_argument("--run", required=True); p.add_argument("--gpu-uuid", required=True); p.add_argument("--pci-bus-id", required=True); p.add_argument("--environment-file", required=True); p.set_defaults(func=command_record_runtime_binding)
    p = sub.add_parser("record-telemetry", help="hash and bind a CPU-written telemetry file")
    p.add_argument("--run", required=True); p.add_argument("--label", required=True); p.add_argument("--file", required=True); p.set_defaults(func=command_record_telemetry)
    p = sub.add_parser("record-child", help="record direct pinned-binary PID and exit code before output verification")
    p.add_argument("--run", required=True); p.add_argument("--stage", required=True, choices=STAGE_ORDER); p.add_argument("--pid", type=int, required=True); p.add_argument("--returncode", type=int, required=True); p.set_defaults(func=command_record_child)
    p = sub.add_parser("stage-complete", help="record an externally verified PASS stage and its SHA-256 artifacts")
    p.add_argument("--run", required=True); p.add_argument("--stage", required=True, choices=STAGE_ORDER); p.add_argument("--verification", required=True); p.add_argument("--stdout-log", required=True); p.add_argument("--stderr-log", required=True); p.set_defaults(func=command_stage_complete)
    p = sub.add_parser("stage-fail", help="record an already-claimed failure as irrevocably consumed")
    shared(p); p.add_argument("--stage", required=True, choices=STAGE_ORDER); p.add_argument("--reason", required=True); p.set_defaults(func=command_stage_fail)
    p = sub.add_parser("record-cleanup", help="bind fail-path own-child cleanup provenance")
    p.add_argument("--run", required=True); p.add_argument("--stage", required=True); p.add_argument("--file", required=True); p.set_defaults(func=command_record_cleanup)
    p = sub.add_parser("finalize", help="mark a complete all-PASS run after final external verification")
    p.add_argument("--run", required=True); p.add_argument("--verification", required=True); p.set_defaults(func=command_finalize)
    p = sub.add_parser("workflow-id", help="derive, without writing, the fixed workflow ID")
    p.add_argument("--plan", required=True); p.add_argument("--pins", required=True); p.set_defaults(func=command_workflow_id)
    return parser



if __name__ == "__main__":
    try:
        # Parse once; keeping the handler dispatch outside the parser makes static review straightforward.
        parser = make_parser()
        namespace = parser.parse_args()
        raise SystemExit(int(namespace.func(namespace)))
    except ContractError as error:
        print(f"SAFE-C2-V3 EXECUTION MANIFEST FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)
