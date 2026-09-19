#!/usr/bin/env python3
"""External one-time approval validator/consumer for C1 v8.

Production mode reads one external approval from one non-symlink file descriptor,
checks its exact v8 release binding, then creates a sidecar claim with O_EXCL.
It never invokes CUDA, nvidia-smi, Nsight, or a benchmark binary and never
creates directories.  A claim is deliberately permanent even if the later GPU
preflight fails.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c1_execution_release_v8_external_one_time_approval")
CONTROL = ROOT / "control"
PINS = CONTROL / "release_pins_v8.json"
PLAN = CONTROL / "measured_execution_plan_v8.json"
BUILD_MANIFEST = ROOT / "build_stage" / "build_stage_manifest_v8.json"
OUTER_GUARD = CONTROL / "run_c1_execution_guard_v8.sh"
GATE = CONTROL / "approval_gate_v8.py"
VERIFIER = CONTROL / "verify_release_v8.py"
LAUNCHER = CONTROL / "launch_c1_execution_v8.sh"
PYTHON = pathlib.Path("/usr/bin/python3")
GIT = pathlib.Path("/usr/bin/git")
MODE = "MEASURED_5X4_C1_V8"
APPROVAL_SCHEMA = "gtspp-c1-v8-external-one-time-approval-v1"
FIXTURE_SCHEMA = "gtspp-c1-v8-approval-claim-fixture-v1"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$")
MAX_APPROVAL_BYTES = 1 << 20


def fail(message: str) -> None:
    print("C1-V8-APPROVAL BLOCKED: " + message, file=sys.stderr)
    raise SystemExit(69)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_regular(path: pathlib.Path, label: str) -> pathlib.Path:
    if not path.is_absolute():
        fail(f"{label}: absolute path required")
    try:
        info = path.lstat()
    except FileNotFoundError:
        fail(f"{label}: missing")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        fail(f"{label}: regular non-symlink file required")
    if path.resolve(strict=True) != path:
        fail(f"{label}: noncanonical path")
    return path


def safe_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    if not path.is_absolute():
        fail(f"{label}: absolute path required")
    try:
        info = path.lstat()
    except FileNotFoundError:
        fail(f"{label}: missing")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        fail(f"{label}: directory non-symlink required")
    if path.resolve(strict=True) != path:
        fail(f"{label}: noncanonical path")
    return path


def json_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except Exception as exc:
        fail(f"{label}: invalid JSON: {exc}")
    if not isinstance(value, dict):
        fail(f"{label}: JSON object required")
    return value


def strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    got = set(value)
    if got != expected:
        fail(f"{label}: exact keys required; missing={sorted(expected - got)} extra={sorted(got - expected)}")


def entry(path: pathlib.Path) -> dict[str, Any]:
    safe_regular(path, "bound release file")
    return {"path": str(path), "sha256": sha256_path(path)}


def require_entry(value: Any, path: pathlib.Path, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"} or value != entry(path):
        fail(f"{label}: exact path/hash binding mismatch")


def parse_time(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str):
        fail(f"{label}: UTC timestamp required")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        fail(f"{label}: invalid UTC timestamp")
    if parsed.tzinfo is None:
        fail(f"{label}: timezone required")
    return parsed.astimezone(dt.timezone.utc)


def read_external(path_text: str) -> tuple[pathlib.Path, int, os.stat_result, bytes]:
    path = pathlib.Path(path_text)
    if not path.is_absolute() or path.name in {"", ".", ".."} or ROOT in path.parents or path == ROOT:
        fail("approval must be an external absolute path")
    parent = path.parent
    safe_directory(parent, "approval parent")
    parent_mode = parent.stat().st_mode
    if parent_mode & 0o022:
        fail("approval parent is group/other writable")
    if parent.stat().st_uid != os.geteuid():
        fail("approval parent owner differs from executing account")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(parent, flags)
    except OSError as exc:
        fail(f"approval parent open: {exc}")
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        approval_fd = os.open(path.name, flags, dir_fd=parent_fd)
        info = os.fstat(approval_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or (info.st_mode & 0o022):
            os.close(approval_fd)
            fail("approval file ownership/mode/type")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(approval_fd, 65536)
            if not block:
                break
            total += len(block)
            if total > MAX_APPROVAL_BYTES:
                os.close(approval_fd)
                fail("approval file too large")
            chunks.append(block)
        after = os.fstat(approval_fd)
        if (info.st_dev, info.st_ino, info.st_size) != (after.st_dev, after.st_ino, after.st_size):
            os.close(approval_fd)
            fail("approval FD changed while read")
        return path, parent_fd, info, b"".join(chunks)
    except BaseException:
        os.close(parent_fd)
        raise


def claim(parent_fd: int, approval_path: pathlib.Path, approval_info: os.stat_result, payload_sha256: str, record: dict[str, Any]) -> pathlib.Path:
    name = approval_path.name + ".c1-v8-consumed.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
    except FileExistsError:
        fail("approval already consumed")
    except OSError as exc:
        fail(f"approval claim O_EXCL failed: {exc}")
    try:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        claim_payload = {
            "schema": "gtspp-c1-v8-external-approval-claim-v1",
            "consumed_utc": now,
            "approval_path": str(approval_path),
            "approval_sha256": payload_sha256,
            "approval_device": approval_info.st_dev,
            "approval_inode": approval_info.st_ino,
            **record,
        }
        data = (json.dumps(claim_payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(parent_fd)
    return approval_path.with_name(name)


def clean_git_commit() -> str:
    safe_regular(GIT, "git executable")
    try:
        commit = subprocess.check_output([str(GIT), "-C", str(ROOT), "rev-parse", "HEAD"], text=True, env={"PATH": "/usr/bin:/bin", "LANG": "C"}).strip()
        dirty = subprocess.check_output([str(GIT), "-C", str(ROOT), "status", "--porcelain"], text=True, env={"PATH": "/usr/bin:/bin", "LANG": "C"})
    except Exception as exc:
        fail(f"release git state unavailable: {exc}")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or dirty:
        fail("release git must be clean at a full commit")
    return commit


def release_objects() -> tuple[dict[str, Any], dict[str, Any]]:
    safe_directory(ROOT, "release root")
    pins_data = json_object(safe_regular(PINS, "release PINS").read_bytes(), "release PINS")
    plan_data = json_object(safe_regular(PLAN, "release plan").read_bytes(), "release plan")
    if pins_data.get("schema") != "gtspp-c1-v8-release-pins-v1" or plan_data.get("schema") != "gtspp-c1-v8-measured-execution-plan-v1" or plan_data.get("mode") != MODE:
        fail("release PINS/plan schema or mode")
    return pins_data, plan_data


def check_production(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    strict_keys(payload, {"schema", "approval_id", "one_time", "issued_utc", "expires_utc", "mode", "release_root", "commit", "pins", "guards", "launchers", "buildmanifest", "binaries", "inputs", "plan", "host", "gpu", "output", "policy"}, "approval")
    if payload.get("schema") != APPROVAL_SCHEMA or payload.get("one_time") is not True or payload.get("mode") != MODE or payload.get("release_root") != str(ROOT):
        fail("approval schema/one-time/mode/root")
    approval_id = payload.get("approval_id")
    if not isinstance(approval_id, str) or not ID_RE.fullmatch(approval_id):
        fail("approval id")
    issued, expires = parse_time(payload.get("issued_utc"), "issued_utc"), parse_time(payload.get("expires_utc"), "expires_utc")
    now = dt.datetime.now(dt.timezone.utc)
    if not issued <= now <= expires or expires - issued > dt.timedelta(days=7):
        fail("approval time window")
    pins, plan = release_objects()
    require_entry(payload.get("pins"), PINS, "approval PINS")
    require_entry(payload.get("buildmanifest"), BUILD_MANIFEST, "approval build manifest")
    require_entry(payload.get("plan"), PLAN, "approval plan")
    expected_guards = {"outer_guard": entry(OUTER_GUARD), "approval_gate": entry(GATE), "release_verifier": entry(VERIFIER)}
    expected_launchers = {"measured_launcher": entry(LAUNCHER)}
    if payload.get("guards") != expected_guards or payload.get("launchers") != expected_launchers:
        fail("approval guard/launcher binding")
    if payload.get("binaries") != pins.get("binaries") or payload.get("inputs") != pins.get("inputs"):
        fail("approval binary/input binding")
    commit = payload.get("commit")
    if not isinstance(commit, dict) or set(commit) != {"v7_source_commit", "release_git_commit"}:
        fail("approval commit binding schema")
    if commit.get("v7_source_commit") != pins.get("origin", {}).get("v7_git_commit") or commit.get("release_git_commit") != clean_git_commit():
        fail("approval commit binding")
    if payload.get("host") != plan.get("host") or payload.get("host") != os.uname().nodename:
        fail("approval host binding")
    expected_gpu = {"physical_index": plan.get("gpu", {}).get("physical_index"), "expected_uuid": plan.get("gpu", {}).get("expected_uuid")}
    if payload.get("gpu") != expected_gpu:
        fail("approval GPU binding")
    expected_policy = {"allow_gpu_telemetry": True, "allow_cuda_binary": True, "allow_nsys": False, "kill_preexisting_processes": False}
    if payload.get("policy") != expected_policy:
        fail("approval policy binding")
    output = payload.get("output")
    expected_output = ROOT / "runs" / ("c1_v8_measured_" + approval_id)
    if not isinstance(output, dict) or set(output) != {"absolute_run_path"} or output.get("absolute_run_path") != str(expected_output):
        fail("approval output binding")
    if expected_output.exists() or expected_output.is_symlink():
        fail("approval output already exists")
    # The static verifier is itself approval-bound immediately above and checks
    # source/build separation plus all pinned inputs/binaries before claim.
    completed = subprocess.run([str(PYTHON), "-B", "-I", str(VERIFIER), "--root", str(ROOT)], text=True, capture_output=True, env={"PATH": "/usr/bin:/bin", "LANG": "C"}, check=False)
    if completed.returncode != 0:
        fail("release static verifier failed before claim: " + completed.stderr.strip())
    try:
        verifier_result = json.loads(completed.stdout)
    except Exception as exc:
        fail(f"release verifier output malformed: {exc}")
    if verifier_result.get("pass") is not True or verifier_result.get("gpu_tools_invoked") is not False or verifier_result.get("cuda_binary_executed") is not False:
        fail("release verifier did not remain CPU-only")
    return pins, plan, str(expected_output)


def consume_production(approval_file: str) -> int:
    path, parent_fd, info, raw = read_external(approval_file)
    try:
        payload = json_object(raw, "approval")
        pins, plan, output_path = check_production(payload)
        record = {
            "approval_id": payload["approval_id"],
            "mode": payload["mode"],
            "release_root": str(ROOT),
            "release_pins_sha256": sha256_path(PINS),
            "build_manifest_sha256": sha256_path(BUILD_MANIFEST),
            "plan_sha256": sha256_path(PLAN),
            "release_git_commit": payload["commit"]["release_git_commit"],
            "v7_source_commit": payload["commit"]["v7_source_commit"],
            "host": payload["host"],
            "gpu": payload["gpu"],
            "output_path": output_path,
            "binaries": payload["binaries"],
            "inputs": payload["inputs"],
            "guards": payload["guards"],
            "launchers": payload["launchers"],
            "cpu_only_preclaim_verifier": True,
        }
        claim_path = claim(parent_fd, path, info, sha256_bytes(raw), record)
        print(json.dumps({"pass": True, "claim_path": str(claim_path), "approval_id": payload["approval_id"], "mode": payload["mode"], "output_path": output_path, "approval_sha256": sha256_bytes(raw), "gpu_tools_invoked": False, "cuda_binary_executed": False}, sort_keys=True))
        return 0
    finally:
        os.close(parent_fd)


def consume_fixture(approval_file: str) -> int:
    """Exercise the same O_EXCL primitive with an unexecutable fixture schema.

    This branch cannot authorize C1: it accepts only CPU_ONLY_CLAIM_FIXTURE and
    never invokes the release verifier, the guard, GPU telemetry, or children.
    """
    path, parent_fd, info, raw = read_external(approval_file)
    try:
        payload = json_object(raw, "fixture approval")
        strict_keys(payload, {"schema", "fixture_id", "fixture_only", "mode", "not_valid_for_execution"}, "fixture approval")
        if payload.get("schema") != FIXTURE_SCHEMA or payload.get("fixture_only") is not True or payload.get("mode") != "CPU_ONLY_CLAIM_FIXTURE" or payload.get("not_valid_for_execution") is not True:
            fail("fixture approval is not an unexecutable CPU-only fixture")
        fixture_id = payload.get("fixture_id")
        if not isinstance(fixture_id, str) or not ID_RE.fullmatch(fixture_id):
            fail("fixture id")
        claim_path = claim(parent_fd, path, info, sha256_bytes(raw), {"fixture_id": fixture_id, "mode": "CPU_ONLY_CLAIM_FIXTURE", "fixture_only": True, "not_valid_for_execution": True, "gpu_tools_invoked": False, "cuda_binary_executed": False})
        print(json.dumps({"pass": True, "fixture_only": True, "claim_path": str(claim_path), "fixture_id": fixture_id, "gpu_tools_invoked": False, "cuda_binary_executed": False}, sort_keys=True))
        return 0
    finally:
        os.close(parent_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--consume-production", action="store_true")
    mode.add_argument("--fixture-claim", action="store_true")
    parser.add_argument("--approval-file", required=True)
    args = parser.parse_args()
    if args.consume_production:
        return consume_production(args.approval_file)
    return consume_fixture(args.approval_file)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
