#!/usr/bin/python3
"""Single-GPU, NVML-guarded launch for the Safe-C1 delta-only correctness gate.

This launcher is intentionally not a performance harness. It uses only the
root-private read-only NVML helper to admit physical GPU 1, runs one binary
under a short-lived approval token, and then invokes an independent CPU oracle
validator. It never calls nvidia-smi, kills processes, or touches shared code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/fair_dynamic_knn_safe_c1_v1")
BUNDLE = ROOT / "inputs/e1_frozen_base_knn_projection_v1"
RUNS = ROOT / "runs"
BIN = ROOT / "bin/fair_safe_c1_delta_knn_e1_runner.sm80"
RUNNER = ROOT / "runner/fair_safe_c1_delta_knn_e1_runner.cu"
BUILD_RECEIPT = ROOT / "build/fair_safe_c1_delta_knn_e1_runner.sm80.build.env"
OBJECT = ROOT / "build/fair_safe_c1_delta_knn_e1_runner.sm_80.o"
COMPILE_LOG = ROOT / "build/compile_sm80_v4_20260729T1150Z.log"
LINK_LOG = ROOT / "build/link_sm80_v2_20260729T1150Z.log"
HELP_LOG = ROOT / "build/help_sm80_v2_20260729T1150Z.log"
COMPILE_HELPER = ROOT / "tools/compile_only.sh"
SOURCE_CLOSURE = ROOT / "provenance/v24_unmodified_source_closure.sha256"
PREFLIGHT = ROOT / "tools/preflight_frozen_base_knn_bundle.py"
VALIDATOR = ROOT / "tools/validate_fair_safe_c1_runner_output.py"
NVML_HELPER = ROOT / "tools/.nvml_idle_snapshot.py"
SYSTEM_PYTHON = Path("/usr/bin/python3.12")
NVCC = Path("/usr/local/cuda-13.1/bin/nvcc")
PRELAUNCH_SAMPLES = 3
SAMPLE_INTERVAL_SECONDS = 2
MAX_GPU_RUNTIME_SECONDS = 120
ALLOWED_GPU_ORDINAL = 1
# One root-private, single-use scope derived from the user's current controlled-run authorization.
APPROVED_GRANT = (
    "safe-c1-delta-e1-exactfallback-v4-20260729",
    "archive_user-20260729-safe-c1-delta-e1-04",
    120,
)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
APPROVAL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CLOSURE_LINE_RE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9._/-]+)$")
UUID_RE = re.compile(r"^GPU-[0-9a-fA-F-]{36}$")
PCI_RE = re.compile(r"^[0-9a-fA-F]{8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")


class GateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_private_dir(path: Path) -> None:
    st = path.lstat()
    if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode) or st.st_uid != 0 or st.st_gid != 0 or stat.S_IMODE(st.st_mode) != 0o700:
        raise GateError(f"unsafe root-private directory: {path}")


def require_private_regular(path: Path, executable: bool = False) -> None:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode) or st.st_uid != 0 or st.st_gid != 0 or stat.S_IMODE(st.st_mode) & 0o077:
        raise GateError(f"unsafe root-private file: {path}")
    if executable and not (stat.S_IMODE(st.st_mode) & 0o100):
        raise GateError(f"required executable bit absent: {path}")


def require_trusted_root_regular(path: Path, executable: bool = False) -> None:
    st = path.lstat()
    if (not stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode) or st.st_uid != 0 or
            st.st_gid != 0 or stat.S_IMODE(st.st_mode) & 0o022):
        raise GateError(f"unsafe trusted system file: {path}")
    if executable and not (stat.S_IMODE(st.st_mode) & 0o100):
        raise GateError(f"trusted system executable bit absent: {path}")


def boot_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def read_kv_receipt(path: Path) -> dict[str, str]:
    require_private_regular(path)
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="ascii").splitlines():
        if not raw or raw.count("=") != 1:
            raise GateError("malformed build receipt")
        key, value = raw.split("=", 1)
        if not key or not value or key in result or any(ch.isspace() for ch in value):
            raise GateError("unsafe build receipt field")
        result[key] = value
    return result


def verify_v24_source_closure() -> None:
    require_private_regular(SOURCE_CLOSURE)
    lines = SOURCE_CLOSURE.read_text(encoding="ascii").splitlines()
    if len(lines) != 7:
        raise GateError("unexpected v24 source closure cardinality")
    seen: set[str] = set()
    for line in lines:
        match = CLOSURE_LINE_RE.fullmatch(line)
        if match is None:
            raise GateError("malformed v24 source closure")
        expected, rel_text = match.groups()
        rel = Path(rel_text)
        if rel.is_absolute() or not rel.parts or any(part in ("", ".", "..") for part in rel.parts):
            raise GateError("unsafe v24 source-closure path")
        if rel_text in seen:
            raise GateError("duplicate v24 source-closure path")
        seen.add(rel_text)
        candidate = ROOT / rel
        try:
            candidate.relative_to(ROOT)
        except ValueError as exc:
            raise GateError("v24 source-closure escapes root") from exc
        require_private_regular(candidate)
        if sha256_file(candidate) != expected:
            raise GateError(f"v24 source-closure drift: {rel_text}")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    tmp = path.with_name("." + path.name + ".tmp")
    if tmp.exists():
        raise GateError(f"stale temporary output exists: {tmp}")
    with tmp.open("x", encoding="utf-8") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def read_snapshot(gpu: int) -> tuple[dict[str, str], str]:
    env = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}
    result = subprocess.run(
        [str(SYSTEM_PYTHON), "-I", "-S", str(NVML_HELPER), "--gpu-ordinal", str(gpu)],
        cwd=str(ROOT), env=env, check=False, text=True, capture_output=True,
    )
    if result.returncode != 0:
        raise GateError(f"NVML helper refused GPU {gpu}: {result.stderr.strip() or result.stdout.strip()}")
    raw = result.stdout
    rows = raw.splitlines()
    expected = (
        "schema", "gpu_ordinal", "gpu_uuid", "gpu_pci_bus_id", "compute_process_count",
        "graphics_process_count", "python_realpath", "python_sha256", "nvml_library_realpath",
        "nvml_library_sha256", "nvml_driver_version",
    )
    if len(rows) != len(expected):
        raise GateError("NVML snapshot field count is noncanonical")
    parsed: dict[str, str] = {}
    for key, row in zip(expected, rows):
        if row.count("=") != 1:
            raise GateError("NVML snapshot malformed")
        actual_key, value = row.split("=", 1)
        if actual_key != key or not value or any(ch.isspace() for ch in value):
            raise GateError("NVML snapshot key/order/value mismatch")
        parsed[key] = value
    if parsed["schema"] != "safe-c1-g3-nvml-snapshot-v2":
        raise GateError("unexpected NVML snapshot schema")
    if (parsed["gpu_ordinal"] != str(gpu) or parsed["compute_process_count"] != "0" or
            parsed["graphics_process_count"] != "0"):
        raise GateError("GPU ordinal or idle-process contract failed")
    if not UUID_RE.fullmatch(parsed["gpu_uuid"]) or not PCI_RE.fullmatch(parsed["gpu_pci_bus_id"]):
        raise GateError("NVML GPU identity format invalid")
    return parsed, raw


def preflight_static(gpu: int, run_name: str, approval_id: str, ttl: int) -> dict[str, str]:
    if gpu != ALLOWED_GPU_ORDINAL:
        raise GateError(f"this admitted pilot is pinned to GPU={ALLOWED_GPU_ORDINAL}")
    if not NAME_RE.fullmatch(run_name):
        raise GateError("invalid run name")
    if not APPROVAL_RE.fullmatch(approval_id):
        raise GateError("invalid approval ID")
    if not 1 <= ttl <= 120:
        raise GateError("TTL must be in [1,120]")
    if (run_name, approval_id, ttl) != APPROVED_GRANT:
        raise GateError("run is not the one scoped approval grant")
    for directory in (ROOT, RUNS, ROOT / "tools", ROOT / "runner", ROOT / "bin", ROOT / "build",
                      ROOT / "provenance", ROOT / "src", ROOT / "reference", ROOT / "reference/include", BUNDLE):
        require_private_dir(directory)
    for file, executable in (
        (BIN, True), (RUNNER, False), (BUILD_RECEIPT, False), (OBJECT, False),
        (COMPILE_LOG, False), (LINK_LOG, False), (HELP_LOG, False),
        (COMPILE_HELPER, True), (SOURCE_CLOSURE, False), (PREFLIGHT, True),
        (VALIDATOR, True), (NVML_HELPER, False),
    ):
        require_private_regular(file, executable=executable)
    require_trusted_root_regular(SYSTEM_PYTHON, executable=True)
    require_trusted_root_regular(NVCC, executable=True)
    receipt = read_kv_receipt(BUILD_RECEIPT)
    expected_receipt = {
        "schema": "fair-safe-c1-delta-build-v2",
        "arch": "sm_80",
        "runner_sha256": sha256_file(RUNNER),
        "object_sha256": sha256_file(OBJECT),
        "binary_sha256": sha256_file(BIN),
        "compile_log_sha256": sha256_file(COMPILE_LOG),
        "link_log_sha256": sha256_file(LINK_LOG),
        "help_log_sha256": sha256_file(HELP_LOG),
        "compile_helper_sha256": sha256_file(COMPILE_HELPER),
        "source_closure_sha256": sha256_file(SOURCE_CLOSURE),
        "preflight_sha256": sha256_file(PREFLIGHT),
        "validator_sha256": sha256_file(VALIDATOR),
        "nvml_helper_sha256": sha256_file(NVML_HELPER),
        "system_python_realpath": str(SYSTEM_PYTHON),
        "system_python_sha256": sha256_file(SYSTEM_PYTHON),
        "nvcc_realpath": str(NVCC),
        "nvcc_sha256": sha256_file(NVCC),
    }
    if set(receipt) != set(expected_receipt):
        raise GateError("build receipt key set drift")
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            raise GateError(f"build receipt binding mismatch: {key}")
    verify_v24_source_closure()
    if not BUNDLE.is_dir() or BUNDLE.is_symlink():
        raise GateError("unsafe bundle directory")
    return {
        "binary_sha256": expected_receipt["binary_sha256"],
        "runner_sha256": expected_receipt["runner_sha256"],
        "object_sha256": expected_receipt["object_sha256"],
        "build_receipt_sha256": sha256_file(BUILD_RECEIPT),
        "preflight_sha256": expected_receipt["preflight_sha256"],
        "validator_sha256": expected_receipt["validator_sha256"],
        "nvml_helper_sha256": expected_receipt["nvml_helper_sha256"],
        "source_closure_sha256": expected_receipt["source_closure_sha256"],
        "system_python_sha256": expected_receipt["system_python_sha256"],
        "nvcc_sha256": expected_receipt["nvcc_sha256"],
        "approval_grant_sha256": hashlib.sha256(
            (run_name + "\n" + approval_id + "\n" + str(ttl)).encode("ascii")
        ).hexdigest(),
    }

def run_preflight(admission: Path) -> None:
    result = subprocess.run(
        [str(SYSTEM_PYTHON), "-I", "-S", str(PREFLIGHT), "--bundle", str(BUNDLE), "--out", str(admission)],
        cwd=str(ROOT), env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        check=False, text=True, capture_output=True,
    )
    (admission.parent / "preflight.stdout.log").write_text(result.stdout, encoding="utf-8")
    (admission.parent / "preflight.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise GateError(f"CPU projection preflight failed: {result.stderr.strip() or result.stdout.strip()}")
    require_private_regular(admission)


def run_binary(gpu_uuid: str, run_dir: Path, admission: Path, expiry_boottime_ns: int) -> tuple[int, str | None, float]:
    remaining_ns = expiry_boottime_ns - boot_ns()
    if remaining_ns <= 0:
        return 124, "approval TTL expired before CUDA subprocess", 0.0
    timeout_seconds = min(float(MAX_GPU_RUNTIME_SECONDS), remaining_ns / 1_000_000_000.0)
    if timeout_seconds <= 0.0:
        return 124, "approval TTL expired before CUDA subprocess", 0.0
    stdout = run_dir / "logs/runner.stdout.log"
    stderr = run_dir / "logs/runner.stderr.log"
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "NVIDIA_VISIBLE_DEVICES": gpu_uuid,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    command = [
        str(BIN), "--bundle", str(BUNDLE), "--preflight", str(admission),
        "--out", str(run_dir / "engine.jsonl"), "--summary", str(run_dir / "summary.json"),
        "--mode", "correctness",
    ]
    with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
        try:
            completed = subprocess.run(command, cwd=str(ROOT), env=env, stdout=out, stderr=err,
                                       timeout=timeout_seconds, check=False)
            return completed.returncode, None, timeout_seconds
        except subprocess.TimeoutExpired:
            # This terminates only the subprocess started above; no shared process is touched.
            return 124, f"CUDA subprocess exceeded remaining approval TTL ({timeout_seconds:.3f}s)", timeout_seconds

def run_validator(run_dir: Path, admission: Path) -> tuple[int, str, str]:
    command = [
        str(SYSTEM_PYTHON), "-I", "-S", str(VALIDATOR),
        "--bundle", str(BUNDLE), "--admission", str(admission),
        "--output", str(run_dir / "engine.jsonl"), "--summary", str(run_dir / "summary.json"),
        "--out", str(run_dir / "independent_validation.json"),
    ]
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    completed = subprocess.run(command, cwd=str(ROOT), env=env, text=True, capture_output=True, check=False)
    (run_dir / "logs/validator.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "logs/validator.stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed.returncode, completed.stdout, completed.stderr


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ordinal", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--ttl-seconds", type=int, required=True)
    args = parser.parse_args()

    static = preflight_static(args.gpu_ordinal, args.run_name, args.approval_id, args.ttl_seconds)
    run_dir = RUNS / args.run_name
    if run_dir.exists() or run_dir.is_symlink():
        raise GateError(f"run directory already exists: {run_dir}")

    snapshots: list[dict[str, Any]] = []
    reference: dict[str, str] | None = None
    for number in range(1, PRELAUNCH_SAMPLES + 1):
        data, raw = read_snapshot(args.gpu_ordinal)
        if reference is None:
            reference = data
        elif data["gpu_uuid"] != reference["gpu_uuid"] or data["gpu_pci_bus_id"] != reference["gpu_pci_bus_id"]:
            raise GateError("GPU identity drift across prelaunch NVML samples")
        snapshots.append({"sample": number, "boottime_ns": boot_ns(), "fields": data, "raw": raw})
        if number != PRELAUNCH_SAMPLES:
            time.sleep(SAMPLE_INTERVAL_SECONDS)
    assert reference is not None

    run_dir.mkdir(mode=0o700)
    (run_dir / "logs").mkdir(mode=0o700)
    admission = run_dir / "admission.env"
    run_preflight(admission)
    static["admission_sha256"] = sha256_file(admission)
    nonce = secrets.token_hex(32)
    issued = boot_ns()
    expires = issued + args.ttl_seconds * 1_000_000_000
    card: dict[str, Any] = {
        "schema": "fair-safe-c1-delta-correctness-guard-v1",
        "status": "ADMITTED_NOT_STARTED",
        "scope": (
            "correctness only: each projected KNN uses the typed exact full-immutable-base "
            "fallback plus exact global delta; no timing/performance/direct-sidecar/delete/" 
            "projected-range/rebuild claim"
        ),
        "host": os.uname().nodename,
        "run_name": args.run_name,
        "approval_id": args.approval_id,
        "token": {
            "sha256": hashlib.sha256(nonce.encode("ascii")).hexdigest(),
            "issued_boottime_ns": issued,
            "expires_boottime_ns": expires,
            "ttl_seconds": args.ttl_seconds,
        },
        "gpu": {
            "requested_ordinal": args.gpu_ordinal,
            "uuid": reference["gpu_uuid"],
            "pci_bus_id": reference["gpu_pci_bus_id"],
            "nvml_helper_sha256": static["nvml_helper_sha256"],
            "prelaunch_idle_samples": snapshots,
        },
        "inputs": {
            "bundle": str(BUNDLE),
            "binary": str(BIN),
            "runner_source": str(RUNNER),
            "preflight": str(PREFLIGHT),
            "validator": str(VALIDATOR),
            **static,
        },
        "outputs": {
            "engine_jsonl": str(run_dir / "engine.jsonl"),
            "summary_json": str(run_dir / "summary.json"),
            "validator_json": str(run_dir / "independent_validation.json"),
        },
        "do_not_touch": [
            "GPU ordinal other than 1",
            "any existing process",
            "/workspace/legacy_workspace/GTS",
            "/workspace/project/GTS",
            "A800-1, A800-2, A800-3",
        ],
    }
    atomic_json(run_dir / "run_card.json", card)

    last, raw = read_snapshot(args.gpu_ordinal)
    (run_dir / "logs/prelaunch_nvml_final.txt").write_text(raw, encoding="ascii")
    if last["gpu_uuid"] != reference["gpu_uuid"] or last["gpu_pci_bus_id"] != reference["gpu_pci_bus_id"]:
        card["status"] = "BLOCKED_GPU_IDENTITY_DRIFT"
        atomic_json(run_dir / "run_card.json", card)
        raise GateError("GPU identity drift immediately before launch")
    if boot_ns() > expires:
        card["status"] = "BLOCKED_TOKEN_EXPIRED_BEFORE_LAUNCH"
        atomic_json(run_dir / "run_card.json", card)
        raise GateError("GPU token expired before launch")

    card["status"] = "RUNNING"
    card["launch_cuda_visible_devices"] = reference["gpu_uuid"]
    card["started_boottime_ns"] = boot_ns()
    atomic_json(run_dir / "run_card.json", card)
    runner_code, runner_timeout, runner_timeout_seconds = run_binary(
        reference["gpu_uuid"], run_dir, admission, expires
    )
    card["runner_exit"] = runner_code
    card["runner_timeout"] = runner_timeout
    card["runner_timeout_seconds"] = runner_timeout_seconds
    card["finished_boottime_ns"] = boot_ns()
    runner_finished_within_ttl = card["finished_boottime_ns"] <= expires
    card["runner_finished_within_ttl"] = runner_finished_within_ttl

    validator_code = None
    validator_stdout = ""
    validator_stderr = ""
    if runner_code == 0 and runner_finished_within_ttl:
        validator_code, validator_stdout, validator_stderr = run_validator(run_dir, admission)
        card["validator_exit"] = validator_code
        card["validator_stdout_sha256"] = hashlib.sha256(validator_stdout.encode()).hexdigest()
        card["validator_stderr_sha256"] = hashlib.sha256(validator_stderr.encode()).hexdigest()

    try:
        post, raw = read_snapshot(args.gpu_ordinal)
        (run_dir / "logs/postrun_nvml.txt").write_text(raw, encoding="ascii")
        card["postrun_idle_snapshot"] = post
        post_ok = post["gpu_uuid"] == reference["gpu_uuid"] and post["gpu_pci_bus_id"] == reference["gpu_pci_bus_id"]
    except GateError as exc:
        card["postrun_idle_snapshot_error"] = str(exc)
        post_ok = False

    if runner_code == 0 and runner_finished_within_ttl and validator_code == 0 and post_ok:
        card["status"] = "PASS_CORRECTNESS_GATE"
    else:
        card["status"] = "FAILED_OR_BLOCKED"
    atomic_json(run_dir / "run_card.json", card)
    print(json.dumps({
        "status": card["status"],
        "run_dir": str(run_dir),
        "gpu_uuid": reference["gpu_uuid"],
        "runner_exit": runner_code,
        "validator_exit": validator_code,
        "gpu_used": [args.gpu_ordinal],
    }, sort_keys=True))
    return 0 if card["status"] == "PASS_CORRECTNESS_GATE" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"guard blocked: {exc}", file=sys.stderr)
        raise SystemExit(2)
