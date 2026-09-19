#!/usr/bin/python3.12
"""GPU1-only, NVML-gated Safe-C1 latency--quality pilot launcher.

This is a root-private launch guard, not a scheduler.  It uses the sealed
read-only NVML helper rather than a CLI monitor, never enumerates or signals
a pre-existing process, and uses only physical ordinal 1.  A TTL overrun is
contained to the CUDA subprocess created by this guard itself.

The scope is deliberately narrow:
  native query_knn candidate API versus current exact query_range(UINT64_MAX)
  full-result immutable-base fallback.  These are non-equivalent APIs, so the
  only permitted label is a latency--quality tradeoff, never fair speedup.
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
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/fair_dynamic_knn_safe_c1_v1")
RUNS = ROOT / "runs"
BUNDLE = ROOT / "inputs/e1_frozen_base_knn_projection_v1"
RUNNER = ROOT / "runner/fair_safe_c1_latency_tradeoff_e1_runner.cu"
COMPILE_HELPER = ROOT / "tools/compile_safe_c1_latency_tradeoff.sh"
OBJECT = ROOT / "build/fair_safe_c1_latency_tradeoff_e1_runner.sm_80.o"
BINARY = ROOT / "bin/fair_safe_c1_latency_tradeoff_e1_runner.sm_80"
PREFLIGHT_TOOL = ROOT / "tools/preflight_frozen_base_knn_bundle.py"
NVML_HELPER = ROOT / "tools/.nvml_idle_snapshot.py"
VALIDATOR = ROOT / "tools/validate_safe_c1_latency_tradeoff_output.py"
SOURCE_CLOSURE = ROOT / "provenance/v24_unmodified_source_closure.sha256"
SYSTEM_PYTHON = Path("/usr/bin/python3.12")

ALLOWED_GPU_ORDINAL = 1
ALLOWED_GPU_UUID = "GPU-CONFIGURE-ARCHIVE-DEVICE"
PRELAUNCH_SAMPLES = 3
SAMPLE_INTERVAL_SECONDS = 1
MAX_TTL_SECONDS = 120
PILOT_WARMUP_PASSES = 1
PILOT_MEASURED_PASSES = 1
GUARD_SCHEMA = "fair-safe-c1-latency-tradeoff-guard-v1"
SCOPE = (
    "native query_knn candidate API versus current exact query_range(UINT64_MAX) "
    "full-result immutable-base fallback; not a fair same-API speedup"
)
SNAPSHOT_SCHEMA = "safe-c1-g3-nvml-snapshot-v2"

EXPECTED_ARTIFACT_HASHES = {
    "runner_source": "3a283f322ef2675b7e6d64062fb219b228375c4d15ebda1003b8f9f63766f9fb",
    "compile_helper": "63067e078cc9349dc5060d27c848d72ed696b29c4e37dd088ea66497576df382",
    "object": "ada205321f6abde0c7be6b354ec93bc59196b25cdf6001afb5558604b6862e1f",
    "binary": "af77e4dd6881e946f128f0908bcf003ea807698f7540a2101241035b4a25cc01",
    "preflight_tool": "62258f1461055744bfc7f74bb8825acb135291873f0289845038eb3ff0037b7d",
    "nvml_helper": "7d039a4317bf0f0ebed6d7fd1cb363d536b541ea1444caf3d73ee8098c755f34",
    "source_closure": "b528800f0a476ac70d7fedf7b7234470a975149d212ab29b31bdac4e9f7c864a",
    "validator": "6c6e8468a54a82e77fa6ca0375a77490800b080ef7c447baa16c8693a7723d72",
}
EXPECTED_INPUT_HASHES = {
    "manifest.json": "68dbf15788793a828c8a9503d11304da576acbdca2f609dd0f8799ae39cd9a4c",
    "metadata.json": "2f515a5cf3bef61d6084f4c0d90075ee16cb4e365971b75102a24bdbbc588798",
    "trace.e1gtrc": "9402c609710fc9076f46653dd5bf30527013e158c63b4576dd665c27f9973ae5",
    "pool.i16": "899adaa59b265ee788841f1a48b667b7da39df166972ba9568c4e94727b31170",
    "queries.i16": "18e0ebbe8ddcdcf6e2312e1e48310111a4d96c1fcb752622d1e9ec9508c21c1e",
    "stable_id_to_pool_row.i32": "93710cce11c994b6b1934713842c93cfcec76a3563fc47574abf419137f4c5c8",
    "initial_base_stable_ids.i32": "6b0751ba5e64fc9c13ddfb44778fa7d6a1f7d7aa9d6a5e38a1f0a1502c3fb9e3",
}
EXPECTED_ADMISSION = {
    "schema": "e1-frozen-base-knn-projection-admission-v2",
    "status": "PASS",
    "bundle_realpath": str(BUNDLE),
    "projection_manifest_schema": "e1-frozen-base-knn-projection-manifest-v1",
    "projection_metadata_schema": "e1-frozen-base-knn-projection-bundle-v1",
    "ops": "insert,knn",
    "excluded_ops": "delete,range",
    "base_immutable": "true",
    "direct_sidecar_allowed": "false",
    "legacy_routing_allowed": "false",
    "dimension": "128",
    "base_n": "4096",
    "pool_n": "6144",
    "query_n": "128",
    "k": "10",
    "event_count": "305",
    "insert_count": "169",
    "knn_count": "136",
    "final_active_count": "4265",
    "source_event_count": "512",
    "final_active_set_sha256": "281d47954a4bbb2a85bafb09e150dd72e2ce4f9086568cf1a1a77d5657fa8e99",
    "base_ids_sha256": "6b0751ba5e64fc9c13ddfb44778fa7d6a1f7d7aa9d6a5e38a1f0a1502c3fb9e3",
    "manifest_sha256": "68dbf15788793a828c8a9503d11304da576acbdca2f609dd0f8799ae39cd9a4c",
    "mapping_sha256": "93710cce11c994b6b1934713842c93cfcec76a3563fc47574abf419137f4c5c8",
    "metadata_sha256": "2f515a5cf3bef61d6084f4c0d90075ee16cb4e365971b75102a24bdbbc588798",
    "pool_sha256": "899adaa59b265ee788841f1a48b667b7da39df166972ba9568c4e94727b31170",
    "projection_event_stream_sha256": "0d7267e098445723c7c065e9937206e37eca040b27025ad98b979c9a9af16b64",
    "queries_sha256": "18e0ebbe8ddcdcf6e2312e1e48310111a4d96c1fcb752622d1e9ec9508c21c1e",
    "source_event_stream_sha256": "4829e4fb83a45d5647b216d7fc8bc88cbc9614f64d36e3b99925918f512de14d",
    "source_trace_sha256": "1ce6b2e961ddf6fd54d5593958c799b61033bf9821f19d34db196d8dc18444df",
    "trace_sha256": "9402c609710fc9076f46653dd5bf30527013e158c63b4576dd665c27f9973ae5",
}
SNAPSHOT_KEYS = (
    "schema", "gpu_ordinal", "gpu_uuid", "gpu_pci_bus_id", "compute_process_count",
    "graphics_process_count", "python_realpath", "python_sha256", "nvml_library_realpath",
    "nvml_library_sha256", "nvml_driver_version",
)

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
APPROVAL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
NONCE_RE = re.compile(r"^[0-9a-f]{64}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
PCI_RE = re.compile(r"^[0-9a-fA-F]{8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
CLOSURE_RE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9._/-]+)$")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def boot_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_private_dir(path: Path, label: str) -> None:
    st = path.lstat()
    require(stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode), f"unsafe {label} directory")
    require(st.st_uid == 0 and st.st_gid == 0 and stat.S_IMODE(st.st_mode) == 0o700,
            f"{label} directory is not root-private 0700")


def require_private_regular(path: Path, label: str, executable: bool = False) -> None:
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode), f"unsafe/missing {label}: {path}")
    require(st.st_uid == 0 and st.st_gid == 0 and (stat.S_IMODE(st.st_mode) & 0o077) == 0,
            f"{label} file is not root-private")
    if executable:
        require((stat.S_IMODE(st.st_mode) & 0o100) != 0, f"{label} is not owner executable")


def require_trusted_root_regular(path: Path, label: str, executable: bool = False) -> None:
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode), f"unsafe/missing trusted {label}: {path}")
    require(st.st_uid == 0 and st.st_gid == 0 and (stat.S_IMODE(st.st_mode) & 0o022) == 0,
            f"trusted {label} is writable by a non-root principal")
    if executable:
        require((stat.S_IMODE(st.st_mode) & 0o100) != 0, f"trusted {label} is not owner executable")


def read_env(path: Path) -> dict[str, str]:
    require_private_regular(path, "admission")
    try:
        text = path.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise GateError("admission is not ASCII") from exc
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        require(line.count("=") == 1 and line.index("=") > 0, "malformed admission line")
        key, value = line.split("=", 1)
        require(key and value and key not in values and not any(ch.isspace() for ch in value),
                f"invalid admission field {key!r}")
        values[key] = value
    return values


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    require_private_dir(path.parent, "receipt parent")
    temporary = path.with_name("." + path.name + ".tmp")
    require(not temporary.exists() and not temporary.is_symlink(), f"stale receipt temporary: {temporary}")
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def write_new_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    require_private_dir(path.parent, "log parent")
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    with path.open("x", encoding=encoding) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def verify_source_closure() -> None:
    require_private_regular(SOURCE_CLOSURE, "source closure")
    require(sha256_file(SOURCE_CLOSURE) == EXPECTED_ARTIFACT_HASHES["source_closure"],
            "source closure hash drift")
    lines = SOURCE_CLOSURE.read_text(encoding="ascii").splitlines()
    require(len(lines) == 7, "source closure cardinality drift")
    seen: set[str] = set()
    for line in lines:
        match = CLOSURE_RE.fullmatch(line)
        require(match is not None, "malformed source closure entry")
        expected, relative_text = match.groups()
        relative = Path(relative_text)
        require(not relative.is_absolute() and relative.parts and
                all(part not in ("", ".", "..") for part in relative.parts) and
                relative_text not in seen, "unsafe/duplicate source closure path")
        seen.add(relative_text)
        target = ROOT / relative
        try:
            target.relative_to(ROOT)
        except ValueError as exc:
            raise GateError("source closure escapes experiment root") from exc
        require_private_regular(target, "source closure member")
        require(sha256_file(target) == expected, f"source closure drift: {relative_text}")


def verify_static_artifacts() -> dict[str, str]:
    for directory, label in (
        (ROOT, "experiment root"), (RUNS, "runs"), (ROOT / "runner", "runner"),
        (ROOT / "tools", "tools"), (ROOT / "build", "build"), (ROOT / "bin", "bin"),
        (ROOT / "src", "source"), (ROOT / "reference", "reference"),
        (ROOT / "reference/include", "reference include"), (ROOT / "provenance", "provenance"),
        (BUNDLE, "bundle"),
    ):
        require_private_dir(directory, label)
    files = {
        "runner_source": (RUNNER, False),
        "compile_helper": (COMPILE_HELPER, True),
        "object": (OBJECT, False),
        "binary": (BINARY, True),
        "preflight_tool": (PREFLIGHT_TOOL, True),
        "nvml_helper": (NVML_HELPER, False),
        "validator": (VALIDATOR, True),
        "source_closure": (SOURCE_CLOSURE, False),
    }
    result: dict[str, str] = {}
    for name, (path, executable) in files.items():
        require_private_regular(path, name, executable)
        actual = sha256_file(path)
        require(actual == EXPECTED_ARTIFACT_HASHES[name], f"immutable artifact hash drift: {name}")
        result[name + "_sha256"] = actual
    require_trusted_root_regular(SYSTEM_PYTHON, "system python", executable=True)
    verify_source_closure()
    for filename, expected in EXPECTED_INPUT_HASHES.items():
        target = BUNDLE / filename
        require_trusted_root_regular(target, "sealed input")
        actual = sha256_file(target)
        require(actual == expected, f"sealed input hash drift: {filename}")
    return result


def run_preflight(admission: Path, log_dir: Path) -> str:
    require_private_dir(log_dir, "preflight log")
    result = subprocess.run(
        [str(SYSTEM_PYTHON), "-I", "-S", str(PREFLIGHT_TOOL), "--bundle", str(BUNDLE),
         "--out", str(admission)],
        cwd=str(ROOT),
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "CUDA_VISIBLE_DEVICES": "",
             "NVIDIA_VISIBLE_DEVICES": "void"},
        text=True,
        capture_output=True,
        check=False,
    )
    write_new_text(log_dir / "preflight.stdout.log", result.stdout)
    write_new_text(log_dir / "preflight.stderr.log", result.stderr)
    require(result.returncode == 0,
            "CPU-only projection preflight failed: " + (result.stderr.strip() or result.stdout.strip()))
    values = read_env(admission)
    require(values == EXPECTED_ADMISSION, "preflight admission content drift")
    return sha256_file(admission)


def read_snapshot(gpu_ordinal: int) -> tuple[dict[str, str], str]:
    result = subprocess.run(
        [str(SYSTEM_PYTHON), "-I", "-S", str(NVML_HELPER), "--gpu-ordinal", str(gpu_ordinal)],
        cwd=str(ROOT),
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        text=True,
        capture_output=True,
        check=False,
    )
    require(result.returncode == 0,
            "NVML idle snapshot refused GPU: " + (result.stderr.strip() or result.stdout.strip()))
    raw = result.stdout
    rows = raw.splitlines()
    require(len(rows) == len(SNAPSHOT_KEYS), "noncanonical NVML snapshot line count")
    values: dict[str, str] = {}
    for expected_key, row in zip(SNAPSHOT_KEYS, rows):
        require(row.count("=") == 1, "malformed NVML snapshot row")
        key, value = row.split("=", 1)
        require(key == expected_key and value and not any(ch.isspace() for ch in value),
                "NVML snapshot key/order/value drift")
        values[key] = value
    require(values["schema"] == SNAPSHOT_SCHEMA and values["gpu_ordinal"] == str(ALLOWED_GPU_ORDINAL),
            "wrong NVML schema/GPU ordinal")
    require(values["gpu_uuid"] == ALLOWED_GPU_UUID and UUID_RE.fullmatch(values["gpu_uuid"]) is not None,
            "GPU1 UUID does not match admitted device")
    require(PCI_RE.fullmatch(values["gpu_pci_bus_id"]) is not None, "malformed GPU PCI identity")
    require(values["compute_process_count"] == "0" and values["graphics_process_count"] == "0",
            "GPU1 is not idle")
    for key in ("python_sha256", "nvml_library_sha256"):
        require(SHA_RE.fullmatch(values[key]) is not None, f"bad NVML snapshot SHA: {key}")
    return values, raw


def snapshot_entry(number: int, fields: dict[str, str], raw: str) -> dict[str, Any]:
    return {
        "sample": number,
        "boottime_ns": boot_ns(),
        "fields": fields,
        "raw_sha256": hashlib.sha256(raw.encode("ascii")).hexdigest(),
    }


def validate_normal_args(args: argparse.Namespace) -> None:
    require(args.gpu_ordinal == ALLOWED_GPU_ORDINAL, f"guard is pinned to GPU={ALLOWED_GPU_ORDINAL}")
    require(isinstance(args.run_name, str) and NAME_RE.fullmatch(args.run_name) is not None,
            "invalid exact run name")
    require(isinstance(args.approval_id, str) and APPROVAL_RE.fullmatch(args.approval_id) is not None,
            "invalid exact approval ID")
    require(isinstance(args.nonce, str) and NONCE_RE.fullmatch(args.nonce) is not None,
            "nonce must be exact 64-char lowercase hex")
    require(isinstance(args.ttl_seconds, int) and 1 <= args.ttl_seconds <= MAX_TTL_SECONDS,
            "TTL must be in [1,120] seconds")


def run_binary(run_dir: Path, admission: Path, nonce: str, expiry_ns: int) -> dict[str, Any]:
    remaining_ns = expiry_ns - boot_ns()
    require(remaining_ns > 0, "approval TTL expired before CUDA subprocess")
    timeout_seconds = min(MAX_TTL_SECONDS, remaining_ns / 1_000_000_000.0)
    require(timeout_seconds > 0.0, "nonpositive runner timeout")
    stdout_path = run_dir / "logs/runner.stdout.log"
    stderr_path = run_dir / "logs/runner.stderr.log"
    command = [
        str(BINARY), "--bundle", str(BUNDLE), "--preflight", str(admission),
        "--out", str(run_dir / "engine.jsonl"), "--summary", str(run_dir / "summary.json"),
        "--mode", "timing", "--warmup-passes", str(PILOT_WARMUP_PASSES),
        "--measured-passes", str(PILOT_MEASURED_PASSES), "--timing-guard-nonce", nonce,
    ]
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        "CUDA_VISIBLE_DEVICES": ALLOWED_GPU_UUID,
        "NVIDIA_VISIBLE_DEVICES": ALLOWED_GPU_UUID,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "SAFE_C1_LATENCY_TRADEOFF_GUARD_NONCE": nonce,
    }
    started = boot_ns()
    exit_code: int
    timed_out = False
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(command, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr,
                                       timeout=timeout_seconds, check=False)
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            # subprocess.run only terminates the child started by this guard;
            # it never enumerates, signals, or alters an unrelated process.
            exit_code = 124
            timed_out = True
    os.chmod(stdout_path, 0o600)
    os.chmod(stderr_path, 0o600)
    finished = boot_ns()
    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "started_boottime_ns": started,
        "finished_boottime_ns": finished,
        "finished_within_ttl": finished <= expiry_ns,
        "cuda_visible_devices": ALLOWED_GPU_UUID,
        "nvidia_visible_devices": ALLOWED_GPU_UUID,
        "warmup_passes": PILOT_WARMUP_PASSES,
        "measured_passes": PILOT_MEASURED_PASSES,
        "argv_redacted": [
            str(BINARY), "--bundle", str(BUNDLE), "--preflight", str(admission),
            "--out", str(run_dir / "engine.jsonl"), "--summary", str(run_dir / "summary.json"),
            "--mode", "timing", "--warmup-passes", str(PILOT_WARMUP_PASSES),
            "--measured-passes", str(PILOT_MEASURED_PASSES), "--timing-guard-nonce", "<redacted>",
        ],
    }


def run_validator(run_dir: Path, admission: Path) -> dict[str, Any]:
    stdout_path = run_dir / "logs/validator.stdout.log"
    stderr_path = run_dir / "logs/validator.stderr.log"
    command = [
        str(SYSTEM_PYTHON), "-I", "-S", str(VALIDATOR), "--bundle", str(BUNDLE),
        "--admission", str(admission), "--events", str(run_dir / "engine.jsonl"),
        "--summary", str(run_dir / "summary.json"), "--guard-receipt", str(run_dir / "guard_receipt.json"),
        "--out", str(run_dir / "independent_validation.json"),
    ]
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, check=False)
    os.chmod(stdout_path, 0o600)
    os.chmod(stderr_path, 0o600)
    output = run_dir / "independent_validation.json"
    output_sha = None
    if completed.returncode == 0:
        require_private_regular(output, "independent validation")
        output_sha = sha256_file(output)
    return {"exit_code": completed.returncode, "output_sha256": output_sha}


def self_check() -> int:
    os.umask(0o077)
    static = verify_static_artifacts()
    with tempfile.TemporaryDirectory(prefix="safe-c1-latency-guard-selfcheck.") as temporary:
        temp_dir = Path(temporary)
        os.chmod(temp_dir, 0o700)
        admission_sha = run_preflight(temp_dir / "admission.env", temp_dir)
    print(json.dumps({
        "schema": GUARD_SCHEMA,
        "status": "PASS_CPU_ONLY_STATIC_SELFCHECK",
        "gpu_workload_launched": False,
        "nvidia_smi_used": False,
        "artifacts": static,
        "admission_sha256": admission_sha,
        "scope": "static/preflight only; no CUDA engine instantiated and no NVML snapshot taken",
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--gpu-ordinal", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--approval-id")
    parser.add_argument("--nonce")
    parser.add_argument("--ttl-seconds", type=int)
    args = parser.parse_args()
    if args.self_check:
        require(all(value is None for value in (
            args.gpu_ordinal, args.run_name, args.approval_id, args.nonce, args.ttl_seconds
        )), "--self-check cannot accept launch arguments")
        return self_check()

    require(all(value is not None for value in (
        args.gpu_ordinal, args.run_name, args.approval_id, args.nonce, args.ttl_seconds
    )), "exact GPU/run-name/approval-id/nonce/TTL are all required")
    validate_normal_args(args)
    os.umask(0o077)
    static = verify_static_artifacts()
    run_dir = RUNS / args.run_name
    require(not run_dir.exists() and not run_dir.is_symlink(), f"run directory already exists: {run_dir}")
    run_dir.mkdir(mode=0o700)
    (run_dir / "logs").mkdir(mode=0o700)
    admission = run_dir / "admission.env"
    receipt_path = run_dir / "guard_receipt.json"
    card: dict[str, Any] = {
        "schema": GUARD_SCHEMA,
        "status": "ADMISSION_READY",
        "scope": SCOPE,
        "host": os.uname().nodename,
        "run_name": args.run_name,
        "approval_id": args.approval_id,
        "token": {
            "nonce_sha256": hashlib.sha256(args.nonce.encode("ascii")).hexdigest(),
            "issued_boottime_ns": 0,
            "expires_boottime_ns": 0,
            "ttl_seconds": args.ttl_seconds,
        },
        "gpu": {
            "requested_ordinal": args.gpu_ordinal,
            "expected_uuid": ALLOWED_GPU_UUID,
            "launch_cuda_visible_devices": ALLOWED_GPU_UUID,
            "prelaunch_idle_samples": [],
        },
        "inputs": {
            "bundle_path": str(BUNDLE),
            "admission_path": str(admission),
            "runner_source_path": str(RUNNER),
            "compile_helper_path": str(COMPILE_HELPER),
            "object_path": str(OBJECT),
            "binary_path": str(BINARY),
            "preflight_tool_path": str(PREFLIGHT_TOOL),
            "nvml_helper_path": str(NVML_HELPER),
            "source_closure_path": str(SOURCE_CLOSURE),
            **static,
            "input_files_sha256": EXPECTED_INPUT_HASHES,
            "guard_sha256": sha256_file(Path(__file__).resolve()),
        },
        "outputs": {
            "events": str(run_dir / "engine.jsonl"),
            "summary": str(run_dir / "summary.json"),
            "validator": str(run_dir / "independent_validation.json"),
        },
        "do_not_touch": [
            "GPU ordinal other than 1",
            "any pre-existing process",
            "/workspace/legacy_workspace/GTS",
            "/workspace/project/GTS",
            "A800-1",
            "A800-2",
            "A800-3",
        ],
    }
    try:
        card["inputs"]["admission_sha256"] = run_preflight(admission, run_dir / "logs")
    except GateError as exc:
        card["status"] = "BLOCKED_CPU_PREFLIGHT"
        card["error"] = str(exc)
        atomic_json(receipt_path, card)
        raise

    # No slow CPU work is allowed after this point: the third of the three
    # successful idle snapshots is immediately followed by the scoped launch.
    pci: str | None = None
    samples: list[dict[str, Any]] = []
    try:
        for number in range(1, PRELAUNCH_SAMPLES + 1):
            fields, raw = read_snapshot(args.gpu_ordinal)
            if pci is None:
                pci = fields["gpu_pci_bus_id"]
            else:
                require(fields["gpu_pci_bus_id"] == pci, "GPU1 PCI identity drift during prelaunch samples")
            write_new_text(run_dir / "logs" / f"prelaunch_nvml_{number}.txt", raw, encoding="ascii")
            samples.append(snapshot_entry(number, fields, raw))
            if number != PRELAUNCH_SAMPLES:
                time.sleep(SAMPLE_INTERVAL_SECONDS)
    except GateError as exc:
        card["status"] = "BLOCKED_PRELAUNCH_NVML"
        card["gpu"]["prelaunch_idle_samples"] = samples
        card["error"] = str(exc)
        atomic_json(receipt_path, card)
        raise

    issued = boot_ns()
    expiry = issued + args.ttl_seconds * 1_000_000_000
    card["token"]["issued_boottime_ns"] = issued
    card["token"]["expires_boottime_ns"] = expiry
    card["gpu"]["prelaunch_idle_samples"] = samples
    card["status"] = "RUNNING"
    atomic_json(receipt_path, card)

    runner = run_binary(run_dir, admission, args.nonce, expiry)
    card["runner"] = runner
    output_ready = False
    if runner["exit_code"] == 0 and not runner["timed_out"] and runner["finished_within_ttl"]:
        try:
            events = run_dir / "engine.jsonl"
            summary = run_dir / "summary.json"
            require_private_regular(events, "runner events")
            require_private_regular(summary, "runner summary")
            require(events.stat().st_size > 0 and summary.stat().st_size > 0, "runner emitted empty timing artifact")
            runner["events_sha256"] = sha256_file(events)
            runner["summary_sha256"] = sha256_file(summary)
            output_ready = True
        except GateError as exc:
            runner["output_error"] = str(exc)

    postrun_ok = False
    try:
        fields, raw = read_snapshot(args.gpu_ordinal)
        require(fields["gpu_pci_bus_id"] == pci, "GPU1 PCI identity drift after runner")
        write_new_text(run_dir / "logs" / "postrun_nvml.txt", raw, encoding="ascii")
        card["gpu"]["postrun_idle_snapshot"] = {
            "boottime_ns": boot_ns(),
            "fields": fields,
            "raw_sha256": hashlib.sha256(raw.encode("ascii")).hexdigest(),
        }
        postrun_ok = True
    except GateError as exc:
        card["gpu"]["postrun_snapshot_error"] = str(exc)

    validator = {"exit_code": None, "output_sha256": None}
    if output_ready and postrun_ok:
        card["status"] = "RUNNER_EXITED_PENDING_VALIDATION"
        atomic_json(receipt_path, card)
        validator = run_validator(run_dir, admission)
    card["validator"] = validator

    if output_ready and postrun_ok and validator["exit_code"] == 0:
        card["status"] = "PASS_TRADEOFF_TIMING_GUARDED"
    else:
        card["status"] = "FAILED_OR_BLOCKED"
    atomic_json(receipt_path, card)
    print(json.dumps({
        "status": card["status"],
        "run_dir": str(run_dir),
        "gpu_ordinal": args.gpu_ordinal,
        "gpu_uuid": ALLOWED_GPU_UUID,
        "runner_exit": runner["exit_code"],
        "validator_exit": validator["exit_code"],
        "claim": "latency_quality_tradeoff_only_not_same_api",
    }, sort_keys=True))
    return 0 if card["status"] == "PASS_TRADEOFF_TIMING_GUARDED" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"guard blocked: {exc}", file=sys.stderr)
        raise SystemExit(2)
