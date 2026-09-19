#!/usr/bin/python3.12
"""Root-private guard for the fresh-allocation telemetry control.

This diagnostic preserves the copied native traversal's per-query cudaMalloc /
cudaFree path. It is not a benchmark and emits no performance conclusion. Normal
mode is intentionally narrow: after static checks and CPU-only preflight, it
takes three UUID-bound idle NVML snapshots, launches one direct GPU child on
GPU1 under a short nonce/TTL, takes a postrun snapshot, and invokes an
independent CPU-only validator against the frozen v1b control. --self-check
never invokes NVML or a CUDA binary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
DIAG = ROOT / "diagnostics/fresh_alloc_telemetry_v1"
RUNNER = DIAG / "runner/fair_safe_c1_fresh_alloc_telemetry_e1_runner.cu"
HEADER = DIAG / "src/g3_safe_search_v2_fresh_alloc_telemetry.cuh"
MATRIX = DIAG / "src/g3_safe_c1_native_matrix_fresh_alloc_telemetry.cu"
COMPILE_HELPER = DIAG / "tools/compile_safe_c1_fresh_alloc_telemetry.sh"
VALIDATOR = DIAG / "tools/validate_safe_c1_fresh_alloc_telemetry_output.py"
# Resolve this script once so its reviewed source-manifest path, the static
# build/audit receipts, and the immutable run receipt name the same file.
GUARD = Path(__file__).resolve()
BINARY = DIAG / "bin/fair_safe_c1_fresh_alloc_telemetry_e1_runner.sm_80"
OBJECT = DIAG / "build/fair_safe_c1_fresh_alloc_telemetry_e1_runner.sm_80.o"
BUILD_RECEIPT = DIAG / "build/fresh_alloc_telemetry_static_build_receipt.env"
MANIFEST = DIAG / "provenance/fresh_alloc_telemetry_source_manifest.json"
REVIEWED_DIFF = DIAG / "provenance/fresh_alloc_telemetry_reviewed_diff.patch"
STATIC_AUDIT = DIAG / "provenance/fresh_alloc_telemetry_static_audit.json"
PREFLIGHT = ROOT / "tools/preflight_frozen_base_knn_bundle.py"
NVML = ROOT / "tools/.nvml_idle_snapshot.py"
SYSTEM_PYTHON = Path("/usr/bin/python3.12")
CONTROL = RUNS / "safe-c1-tradeoff-pilot-v1b-20260730"

GPU_ORDINAL = 1
GPU_UUID = "GPU-CONFIGURE-ARCHIVE-DEVICE"
MAX_TTL = 120
SAMPLES = 3
SAMPLE_INTERVAL_SECONDS = 1
WARMUP_PASSES = 1
MEASURED_PASSES = 1

SCHEMA = "fresh-allocation-telemetry-guard-v1"
VARIANT = "fresh_allocation_telemetry_control"
MANIFEST_SCHEMA = "fair-safe-c1-fresh-allocation-telemetry-source-manifest-v1"
STATIC_AUDIT_SCHEMA = "fair-safe-c1-fresh-allocation-telemetry-static-audit-v1"
BUILD_RECEIPT_SCHEMA = "fair-safe-c1-fresh-allocation-telemetry-static-build-receipt-v1"
EVENT_SCHEMA = "fresh-allocation-telemetry-e1-v1"
SUMMARY_SCHEMA = "fresh-allocation-telemetry-e1-v1"
CLAIM_SCOPE = "semantic_fresh_allocation_control_only"
EVENT_KEYS = frozenset((
    "schema", "record", "diagnostic_variant", "publication_eligible",
    "condition", "semantic_pass", "schedule_phase", "schedule_phase_pass",
    "schedule_phase_slot", "case_ordinal", "op_index", "query_id",
    "external_global_delta_live", "api_result_count_before_adapter_truncation",
    "base_candidate_count", "visited_leaf_count",
    "full_immutable_base_candidate_count", "exact_full_immutable_base_fallback",
    "base_path", "merged_overlap_at_k", "merged_exact_match",
    "merged_result_sha256", "merged",
))
SUMMARY_KEYS = frozenset((
    "schema", "mode", "status", "diagnostic_variant", "publication_eligible",
    "claim_scope", "scope", "schedule", "conditions", "quality",
    "fresh_witness", "limitations",
))
VALIDATOR_PASS = "PASS_SEMANTIC_CONTROL_VALIDATED"
VALIDATOR_SELF_CHECK_PASS = "PASS_CPU_ONLY_STATIC_SELFCHECK"
SNAPSHOT_SCHEMA = "safe-c1-g3-nvml-snapshot-v2"
SNAPSHOT_KEYS = (
    "schema", "gpu_ordinal", "gpu_uuid", "gpu_pci_bus_id",
    "compute_process_count", "graphics_process_count", "python_realpath",
    "python_sha256", "nvml_library_realpath", "nvml_library_sha256",
    "nvml_driver_version",
)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
APPROVAL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
NONCE_RE = re.compile(r"^[0-9a-f]{64}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
PCI_RE = re.compile(r"^[0-9a-fA-F]{8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
INPUT_HASHES = {
    "manifest.json": "68dbf15788793a828c8a9503d11304da576acbdca2f609dd0f8799ae39cd9a4c",
    "metadata.json": "2f515a5cf3bef61d6084f4c0d90075ee16cb4e365971b75102a24bdbbc588798",
    "trace.e1gtrc": "9402c609710fc9076f46653dd5bf30527013e158c63b4576dd665c27f9973ae5",
    "pool.i16": "899adaa59b265ee788841f1a48b667b7da39df166972ba9568c4e94727b31170",
    "queries.i16": "18e0ebbe8ddcdcf6e2312e1e48310111a4d96c1fcb752622d1e9ec9508c21c1e",
    "stable_id_to_pool_row.i32": "93710cce11c994b6b1934713842c93cfcec76a3563fc47574abf419137f4c5c8",
    "initial_base_stable_ids.i32": "6b0751ba5e64fc9c13ddfb44778fa7d6a1f7d7aa9d6a5e38a1f0a1502c3fb9e3",
}
CONTROL_HASHES = {
    "engine.jsonl": "4a32720f49ac49bed6bd6f2755532231ea8a04e5d3c797e342e56ef22cb35f11",
    "summary.json": "5d9f197cf7297cb970a69f8a5fdb7255efad9d06c37adac343377af5a185c103",
    "guard_receipt.json": "1400941c7d898c2f8daa4884c85c8fa8e1a027f1ffdcc2f88af4a574070b63f8",
    "independent_validation.json": "2719fcffa018e352d4c7b8da480c889545a57d5a39605bff424511fb8acb042e",
    "admission.env": "a979ba830a2f39bf12ddace378b1be7106835233cd2e0f8e1bd8917175c75721",
}
EXPECTED_ADMISSION_SHA = CONTROL_HASHES["admission.env"]
EXPECTED_PREFLIGHT_SHA = "62258f1461055744bfc7f74bb8825acb135291873f0289845038eb3ff0037b7d"
EXPECTED_NVML_SHA = "7d039a4317bf0f0ebed6d7fd1cb363d536b541ea1444caf3d73ee8098c755f34"

# Source provenance is one-way: the manifest pins only sources/tools and
# reviewed diff.  Build receipt/static audit bind that manifest afterwards.
# They must not be recursively listed in the manifest, which would create an
# impossible manifest<->audit hash cycle.
SOURCE_ARTIFACTS: dict[str, tuple[Path, bool]] = {
    "runner_source": (RUNNER, False),
    "header_source": (HEADER, False),
    "matrix_source": (MATRIX, False),
    "compile_helper": (COMPILE_HELPER, True),
    "validator": (VALIDATOR, True),
    "guard": (GUARD, True),
    "reviewed_diff": (REVIEWED_DIFF, False),
}
BUILD_ARTIFACTS: dict[str, tuple[Path, bool]] = {
    "object": (OBJECT, False),
    "binary": (BINARY, True),
}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


# The guard receipt records the accepted launch arguments.  Reject repeated
# spellings before argparse can silently keep the last value, including the
# --option=value form.  Long-option abbreviation is disabled below as well.
def reject_duplicate_cli_options(argv: list[str]) -> None:
    seen: set[str] = set()
    for token in argv:
        require(token != "--", "end-of-options marker is not accepted")
        if not token.startswith("-") or token == "-":
            continue
        spelling = token.split("=", 1)[0]
        logical = "--help" if spelling == "-h" else spelling
        require(logical not in seen, "duplicate CLI option: " + logical)
        seen.add(logical)


def boot_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_lstat(path: Path, label: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as error:
        raise GateError("missing or unreadable " + label + ": " + str(path)) from error


def private_dir(path: Path, label: str) -> None:
    entry = checked_lstat(path, label + " directory")
    require(stat.S_ISDIR(entry.st_mode) and not stat.S_ISLNK(entry.st_mode),
            "unsafe " + label + " directory")
    require(entry.st_uid == 0 and entry.st_gid == 0 and stat.S_IMODE(entry.st_mode) == 0o700,
            label + " must be root-private 0700")


def private_file(path: Path, label: str, executable: bool = False) -> None:
    entry = checked_lstat(path, label)
    require(stat.S_ISREG(entry.st_mode) and not stat.S_ISLNK(entry.st_mode),
            "unsafe " + label)
    require(entry.st_uid == 0 and entry.st_gid == 0 and stat.S_IMODE(entry.st_mode) & 0o077 == 0,
            label + " must be root-private")
    if executable:
        require(stat.S_IMODE(entry.st_mode) & 0o100, label + " must be owner executable")


def trusted_file(path: Path, label: str, executable: bool = False) -> None:
    entry = checked_lstat(path, "trusted " + label)
    require(stat.S_ISREG(entry.st_mode) and not stat.S_ISLNK(entry.st_mode),
            "unsafe trusted " + label)
    require(entry.st_uid == 0 and entry.st_gid == 0 and stat.S_IMODE(entry.st_mode) & 0o022 == 0,
            "trusted " + label + " writable by non-root")
    if executable:
        require(stat.S_IMODE(entry.st_mode) & 0o100, "trusted " + label + " not executable")


def tree_private(path: Path) -> None:
    private_dir(path, "diagnostic root")
    for descendant in path.rglob("*"):
        entry = descendant.lstat()
        require(not stat.S_ISLNK(entry.st_mode), "diagnostic tree symlink: " + str(descendant))
        require(entry.st_uid == 0 and entry.st_gid == 0 and stat.S_IMODE(entry.st_mode) & 0o077 == 0,
                "nonprivate diagnostic entry: " + str(descendant))
        require(stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode),
                "unexpected diagnostic entry: " + str(descendant))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    private_dir(path.parent, "receipt parent")
    temp = path.with_name("." + path.name + ".tmp")
    require(not temp.exists() and not temp.is_symlink(), "stale receipt temporary")
    with temp.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temp, 0o600)
    os.replace(temp, path)


def write_new(path: Path, text: str, encoding: str = "utf-8") -> None:
    private_dir(path.parent, "log parent")
    require(not path.exists() and not path.is_symlink(), "refuse overwrite " + str(path))
    with path.open("x", encoding=encoding) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def write_immutable_json(path: Path, value: dict[str, Any], label: str) -> str:
    """Write a root-private, no-overwrite JSON snapshot and return its digest.

    This is deliberately distinct from ``atomic_json``: the mutable final guard
    receipt may be replaced as the run progresses, while the receipt passed to
    the independent validator must remain the exact pending-stage snapshot.
    The run directory is root-private; O_EXCL via ``write_new`` makes a second
    write fail closed rather than replacing that evidence.
    """
    write_new(path, json.dumps(value, sort_keys=True, indent=2) + "\n")
    private_file(path, label)
    return sha256_file(path)


def json_private(path: Path, label: str) -> dict[str, Any]:
    private_file(path, label)
    try:
        value = json.loads(path.read_text("utf-8"))
    except Exception as error:
        raise GateError("invalid " + label + " JSON") from error
    require(isinstance(value, dict), label + " must be an object")
    return value


def parse_env(path: Path) -> dict[str, str]:
    private_file(path, "admission")
    values: dict[str, str] = {}
    for line in path.read_text("ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        require(line.count("=") == 1 and line.index("=") > 0, "admission syntax")
        key, value = line.split("=", 1)
        require(key not in values and value and not any(char.isspace() for char in value),
                "admission field")
        values[key] = value
    return values


def artifact_spec(manifest: dict[str, Any], name: str, expected_path: Path,
                  executable: bool) -> str:
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), "manifest artifacts object missing")
    spec = artifacts.get(name)
    require(isinstance(spec, dict), "manifest artifact missing: " + name)
    require(spec.get("path") == str(expected_path), "manifest artifact path drift: " + name)
    expected_hash = spec.get("sha256")
    require(isinstance(expected_hash, str) and SHA_RE.fullmatch(expected_hash) is not None,
            "manifest artifact hash malformed: " + name)
    private_file(expected_path, name, executable)
    actual_hash = sha256_file(expected_path)
    require(actual_hash == expected_hash, "artifact hash drift: " + name)
    return actual_hash


def verify_control() -> None:
    private_dir(CONTROL, "fixed v1b control")
    for name, expected_hash in CONTROL_HASHES.items():
        path = CONTROL / name
        private_file(path, "fixed v1b " + name)
        require(sha256_file(path) == expected_hash, "fixed v1b hash drift: " + name)
    validation = json_private(CONTROL / "independent_validation.json", "fixed v1b validation")
    require(validation.get("status") == "PASS", "fixed v1b validation is not PASS")


def verify_manifest() -> tuple[dict[str, Any], dict[str, str]]:
    manifest = json_private(MANIFEST, "fresh-allocation source manifest")
    require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema drift")
    require(manifest.get("diagnostic_variant") == VARIANT, "manifest variant drift")
    require(manifest.get("publication_eligible") is False, "manifest must be non-publication")
    require(manifest.get("event_schema") == EVENT_SCHEMA and
            manifest.get("summary_schema") == SUMMARY_SCHEMA,
            "manifest runner schema drift")
    require(manifest.get("claim_scope") == CLAIM_SCOPE,
            "manifest semantic claim boundary drift")
    control = manifest.get("control_artifacts")
    require(isinstance(control, dict), "manifest control binding missing")
    require(control.get("run_dir") == str(CONTROL), "manifest control run path drift")
    for name, expected_hash in CONTROL_HASHES.items():
        require(control.get(name + "_sha256") == expected_hash,
                "manifest control hash drift: " + name)

    artifact_hashes = {
        name: artifact_spec(manifest, name, path, executable)
        for name, (path, executable) in SOURCE_ARTIFACTS.items()
    }

    closure = manifest.get("source_closure")
    require(isinstance(closure, list) and closure, "manifest source closure missing")
    closure_lines: list[str] = []
    closure_paths: set[str] = set()
    for item in closure:
        require(isinstance(item, dict), "malformed source closure entry")
        item_path = item.get("path")
        item_hash = item.get("sha256")
        require(isinstance(item_path, str) and isinstance(item_hash, str) and
                SHA_RE.fullmatch(item_hash) is not None, "malformed source closure entry")
        path = Path(item_path)
        require(path.is_relative_to(ROOT), "source closure escapes experiment root")
        require(item_path not in closure_paths, "duplicate source closure path")
        closure_paths.add(item_path)
        private_file(path, "source closure member", path.suffix in (".py", ".sh"))
        require(sha256_file(path) == item_hash, "source closure hash drift: " + item_path)
        closure_lines.append(str(path.relative_to(ROOT)) + "\t" + item_hash)
    closure_digest = hashlib.sha256(
        ("\n".join(sorted(closure_lines)) + "\n").encode("ascii")
    ).hexdigest()
    require(manifest.get("full_source_closure_sha256") == closure_digest,
            "source closure digest drift")

    required_sources = {
        str(RUNNER), str(HEADER), str(MATRIX), str(COMPILE_HELPER), str(VALIDATOR), str(GUARD),
    }
    require(required_sources.issubset(closure_paths),
            "source closure omits a guarded source/tool")
    return manifest, artifact_hashes


def verify_build_receipt(manifest_hash: str,
                         source_hashes: dict[str, str]) -> dict[str, str]:
    private_file(BUILD_RECEIPT, "fresh-allocation build receipt")
    fields: dict[str, str] = {}
    for line in BUILD_RECEIPT.read_text("ascii").splitlines():
        require(line.count("=") == 1, "build receipt syntax")
        key, value = line.split("=", 1)
        require(key not in fields and value, "build receipt field")
        fields[key] = value
    require(fields.get("schema") == BUILD_RECEIPT_SCHEMA and
            fields.get("diagnostic_variant") == VARIANT and
            fields.get("publication_eligible") == "false" and
            fields.get("gpu_workload_launched") == "false" and
            fields.get("static_compile_only") == "true" and
            fields.get("compile_mode") == "all" and fields.get("arch") == "sm_80",
            "build receipt label/claim drift")
    require(fields.get("source_manifest_sha256") == manifest_hash,
            "build receipt manifest binding drift")
    source_field_map = {
        "runner_source_sha256": "runner_source",
        "header_source_sha256": "header_source",
        "matrix_source_sha256": "matrix_source",
        "compile_helper_sha256": "compile_helper",
        "validator_sha256": "validator",
        "guard_sha256": "guard",
    }
    for field, artifact in source_field_map.items():
        require(fields.get(field) == source_hashes[artifact],
                "build receipt source binding drift: " + field)
    build_hashes: dict[str, str] = {}
    for name, (path, executable) in BUILD_ARTIFACTS.items():
        private_file(path, name, executable)
        actual_hash = sha256_file(path)
        require(fields.get(name + "_sha256") == actual_hash,
                "build receipt build binding drift: " + name)
        build_hashes[name] = actual_hash
    build_hashes["build_receipt"] = sha256_file(BUILD_RECEIPT)
    return build_hashes


def verify_static_audit(manifest_hash: str, source_hashes: dict[str, str],
                        build_hashes: dict[str, str]) -> str:
    audit = json_private(STATIC_AUDIT, "fresh-allocation static audit")
    require(audit.get("schema") == STATIC_AUDIT_SCHEMA and
            audit.get("status") == "PASS_STATIC_SOURCE_AND_BUILD_AUDIT",
            "static audit status/schema drift")
    require(audit.get("diagnostic_variant") == VARIANT and
            audit.get("publication_eligible") is False and
            audit.get("claim_scope") == CLAIM_SCOPE and
            audit.get("source_manifest_sha256") == manifest_hash,
            "static audit claim/manifest drift")
    build = audit.get("static_build")
    require(isinstance(build, dict), "static audit build binding missing")
    for name in ("runner_source", "header_source", "matrix_source", "compile_helper",
                 "validator", "guard"):
        require(build.get(name + "_sha256") == source_hashes[name],
                "static audit source binding drift: " + name)
    for name in ("object", "binary"):
        require(build.get(name + "_sha256") == build_hashes[name],
                "static audit build binding drift: " + name)
    require(audit.get("reviewed_diff_sha256") == source_hashes["reviewed_diff"],
            "static audit reviewed diff binding drift")
    return sha256_file(STATIC_AUDIT)


def verify_static() -> dict[str, str]:
    for path, label in (
        (ROOT, "experiment root"), (RUNS, "runs"), (BUNDLE, "sealed bundle"),
        (DIAG, "diagnostic root"), (DIAG / "src", "diagnostic source"),
        (DIAG / "runner", "diagnostic runner"), (DIAG / "tools", "diagnostic tools"),
        (DIAG / "provenance", "diagnostic provenance"),
        (DIAG / "build", "diagnostic build"), (DIAG / "bin", "diagnostic bin"),
    ):
        private_dir(path, label)
    tree_private(DIAG)
    trusted_file(SYSTEM_PYTHON, "system Python", True)
    trusted_file(PREFLIGHT, "CPU preflight", True)
    trusted_file(NVML, "NVML snapshot helper")
    require(sha256_file(PREFLIGHT) == EXPECTED_PREFLIGHT_SHA, "CPU preflight hash drift")
    require(sha256_file(NVML) == EXPECTED_NVML_SHA, "NVML helper hash drift")
    for name, expected_hash in INPUT_HASHES.items():
        path = BUNDLE / name
        trusted_file(path, "sealed input " + name)
        require(sha256_file(path) == expected_hash, "sealed input hash drift: " + name)
    _, source_hashes = verify_manifest()
    manifest_hash = sha256_file(MANIFEST)
    build_hashes = verify_build_receipt(manifest_hash, source_hashes)
    static_audit_hash = verify_static_audit(manifest_hash, source_hashes, build_hashes)
    verify_control()
    output = {name + "_sha256": value for name, value in source_hashes.items()}
    output.update({name + "_sha256": value for name, value in build_hashes.items()})
    output["static_audit_sha256"] = static_audit_hash
    output["manifest_sha256"] = manifest_hash
    current_guard_sha = sha256_file(GUARD)
    require(source_hashes["guard"] == current_guard_sha,
            "guard source hash differs from reviewed source manifest")
    output["guard_sha256"] = current_guard_sha
    return output


def run_preflight(admission: Path, logs: Path) -> str:
    result = subprocess.run(
        [str(SYSTEM_PYTHON), "-I", "-S", str(PREFLIGHT), "--bundle", str(BUNDLE),
         "--out", str(admission)],
        cwd=str(ROOT),
        env={
            "PATH": "/usr/bin:/bin", "HOME": "/nonexistent",
            "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void",
        },
        text=True, capture_output=True, check=False,
    )
    write_new(logs / "preflight.stdout.log", result.stdout)
    write_new(logs / "preflight.stderr.log", result.stderr)
    require(result.returncode == 0,
            "CPU preflight failed: " + (result.stderr.strip() or result.stdout.strip()))
    values = parse_env(admission)
    require(values.get("schema") == "e1-frozen-base-knn-projection-admission-v2" and
            values.get("status") == "PASS" and
            values.get("bundle_realpath") == str(BUNDLE) and
            values.get("ops") == "insert,knn" and
            values.get("excluded_ops") == "delete,range" and
            values.get("base_immutable") == "true" and
            values.get("direct_sidecar_allowed") == "false",
            "admission semantics")
    admission_hash = sha256_file(admission)
    require(admission_hash == EXPECTED_ADMISSION_SHA, "admission hash drift")
    return admission_hash


def read_snapshot() -> tuple[dict[str, str], str]:
    result = subprocess.run(
        [str(SYSTEM_PYTHON), "-I", "-S", str(NVML), "--gpu-ordinal", str(GPU_ORDINAL)],
        cwd=str(ROOT), env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        text=True, capture_output=True, check=False,
    )
    require(result.returncode == 0,
            "NVML snapshot refused GPU1: " + (result.stderr.strip() or result.stdout.strip()))
    raw = result.stdout
    rows = raw.splitlines()
    require(len(rows) == len(SNAPSHOT_KEYS), "NVML snapshot line count")
    fields: dict[str, str] = {}
    for expected_key, row in zip(SNAPSHOT_KEYS, rows):
        require(row.count("=") == 1, "NVML snapshot syntax")
        key, value = row.split("=", 1)
        require(key == expected_key and value and not any(char.isspace() for char in value),
                "NVML snapshot key/order")
        fields[key] = value
    require(fields["schema"] == SNAPSHOT_SCHEMA and
            fields["gpu_ordinal"] == str(GPU_ORDINAL) and
            fields["gpu_uuid"] == GPU_UUID and
            UUID_RE.fullmatch(fields["gpu_uuid"]) is not None,
            "NVML GPU UUID binding")
    require(PCI_RE.fullmatch(fields["gpu_pci_bus_id"]) is not None and
            fields["compute_process_count"] == "0" and
            fields["graphics_process_count"] == "0",
            "NVML GPU1 is not idle")
    require(SHA_RE.fullmatch(fields["python_sha256"]) is not None and
            SHA_RE.fullmatch(fields["nvml_library_sha256"]) is not None,
            "NVML helper hash fields")
    return fields, raw


def snapshot_entry(index: int, fields: dict[str, str], raw: str) -> dict[str, Any]:
    return {
        "sample": index, "boottime_ns": boot_ns(), "fields": fields,
        "raw_sha256": hashlib.sha256(raw.encode("ascii")).hexdigest(),
    }


def validate_args(args: argparse.Namespace) -> None:
    require(args.gpu_ordinal == GPU_ORDINAL, "guard is pinned to GPU ordinal 1")
    require(isinstance(args.run_name, str) and NAME_RE.fullmatch(args.run_name) is not None,
            "invalid run name")
    require(isinstance(args.approval_id, str) and APPROVAL_RE.fullmatch(args.approval_id) is not None,
            "invalid approval identifier")
    require(isinstance(args.nonce, str) and NONCE_RE.fullmatch(args.nonce) is not None,
            "nonce must be exactly 64 lowercase hexadecimal characters")
    require(isinstance(args.ttl_seconds, int) and 1 <= args.ttl_seconds <= MAX_TTL,
            "TTL must be within 1..120 seconds")


def runner_command(run: Path, admission: Path, nonce: str) -> list[str]:
    return [
        str(BINARY), "--bundle", str(BUNDLE), "--preflight", str(admission),
        "--out", str(run / "engine.jsonl"), "--summary", str(run / "summary.json"),
        "--mode", "semantic-control", "--warmup-passes", str(WARMUP_PASSES),
        "--measured-passes", str(MEASURED_PASSES),
        "--fresh-alloc-guard-nonce", nonce,
        "--witness", str(run / "fresh_alloc_witness.jsonl"),
        "--source-manifest", str(MANIFEST), "--run-id", run.name,
    ]


def run_binary(run: Path, admission: Path, nonce: str, expiry_ns: int) -> dict[str, Any]:
    remaining_ns = expiry_ns - boot_ns()
    require(remaining_ns > 0, "approval TTL expired before CUDA child")
    timeout_seconds = min(MAX_TTL, remaining_ns / 1_000_000_000)
    require(timeout_seconds > 0, "nonpositive CUDA child timeout")
    stdout_path = run / "logs/runner.stdout.log"
    stderr_path = run / "logs/runner.stderr.log"
    started_ns = boot_ns()
    timed_out = False
    with stdout_path.open("x", encoding="utf-8") as stdout_handle, \
            stderr_path.open("x", encoding="utf-8") as stderr_handle:
        try:
            completed = subprocess.run(
                runner_command(run, admission, nonce), cwd=str(ROOT),
                env={
                    "PATH": "/usr/bin:/bin", "HOME": "/nonexistent",
                    "CUDA_VISIBLE_DEVICES": GPU_UUID, "NVIDIA_VISIBLE_DEVICES": GPU_UUID,
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "SAFE_C1_FRESH_ALLOC_TELEMETRY_GUARD_NONCE": nonce,
                },
                stdout=stdout_handle, stderr=stderr_handle, timeout=timeout_seconds,
                check=False,
            )
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
            timed_out = True
    os.chmod(stdout_path, 0o600)
    os.chmod(stderr_path, 0o600)
    finished_ns = boot_ns()
    return {
        "exit_code": exit_code, "timed_out": timed_out,
        "timeout_seconds": timeout_seconds, "started_boottime_ns": started_ns,
        "finished_boottime_ns": finished_ns, "finished_within_ttl": finished_ns <= expiry_ns,
        "cuda_visible_devices": GPU_UUID, "nvidia_visible_devices": GPU_UUID,
        "warmup_passes": WARMUP_PASSES, "measured_passes": MEASURED_PASSES,
        "argv_redacted": [*runner_command(run, admission, "<redacted>")],
    }


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        require(line, "empty JSONL line " + str(line_number))
        try:
            row = json.loads(line)
        except Exception as error:
            raise GateError("invalid JSONL line " + str(line_number)) from error
        require(isinstance(row, dict), "JSONL record must be object")
        rows.append(row)
    require(rows, "empty JSONL")
    return rows


def verify_runner_outputs(run: Path, runner: dict[str, Any]) -> None:
    events = run / "engine.jsonl"
    summary = run / "summary.json"
    witness = run / "fresh_alloc_witness.jsonl"
    stdout_path = run / "logs/runner.stdout.log"
    stderr_path = run / "logs/runner.stderr.log"
    for path, label in (
        (events, "runner events"), (summary, "runner summary"),
        (witness, "fresh-allocation witness"),
        (stdout_path, "runner stdout"), (stderr_path, "runner stderr"),
    ):
        private_file(path, label)
    require(events.stat().st_size > 0 and summary.stat().st_size > 0 and
            witness.stat().st_size > 0, "empty runner or witness output")
    records = parse_jsonl(events)
    for index, row in enumerate(records, start=1):
        require(set(row) == EVENT_KEYS, f"semantic engine event shape {index}")
        require(row.get("schema") == EVENT_SCHEMA and
                row.get("record") == "semantic_observation" and
                row.get("diagnostic_variant") == VARIANT and
                row.get("publication_eligible") is False,
                f"semantic engine event identity {index}")
    summary_value = json_private(summary, "runner summary")
    require(set(summary_value) == SUMMARY_KEYS, "semantic engine summary shape")
    require(summary_value.get("schema") == SUMMARY_SCHEMA and
            summary_value.get("mode") == "semantic-control" and
            summary_value.get("status") == "PASS_SEMANTIC_CONTROL" and
            summary_value.get("diagnostic_variant") == VARIANT and
            summary_value.get("publication_eligible") is False and
            summary_value.get("claim_scope") == CLAIM_SCOPE,
            "semantic engine summary identity")
    serialized = json.dumps({"events": records, "summary": summary_value}, sort_keys=True)
    require("api_host_ns" not in serialized and "timing_claim" not in serialized and
            "latency" not in serialized,
            "semantic output contains a prohibited duration or legacy claim field")
    require(stderr_path.read_text("utf-8") == "", "runner stderr is nonempty")
    runner["events_sha256"] = sha256_file(events)
    runner["summary_sha256"] = sha256_file(summary)
    runner["witness_sha256"] = sha256_file(witness)
    runner["stdout_sha256"] = sha256_file(stdout_path)
    runner["stderr_sha256"] = sha256_file(stderr_path)
    runner["record_count"] = len(records)
    runner["output_contract"] = "PASS_semantic_engine_schema_plus_nonempty_fresh_allocation_witness"


def cpu_env() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin", "HOME": "/nonexistent",
        "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    }


def run_cpu(command: list[str], stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    with stdout_path.open("x", encoding="utf-8") as stdout_handle, \
            stderr_path.open("x", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(
            command, cwd=str(ROOT), env=cpu_env(), stdout=stdout_handle,
            stderr=stderr_handle, check=False,
        )
    os.chmod(stdout_path, 0o600)
    os.chmod(stderr_path, 0o600)
    return {"exit_code": completed.returncode}


def run_validator(run: Path, admission: Path, guard_receipt: Path) -> dict[str, Any]:
    private_file(guard_receipt, "immutable pending-stage validator receipt")
    output = run / "independent_validation.json"
    result = run_cpu(
        [
            str(SYSTEM_PYTHON), "-I", "-S", str(VALIDATOR),
            "--bundle", str(BUNDLE), "--admission", str(admission),
            "--events", str(run / "engine.jsonl"), "--summary", str(run / "summary.json"),
            "--witness", str(run / "fresh_alloc_witness.jsonl"),
            "--control-events", str(CONTROL / "engine.jsonl"),
            "--control-summary", str(CONTROL / "summary.json"),
            "--variant-manifest", str(MANIFEST),
            "--guard-receipt", str(guard_receipt), "--out", str(output),
        ],
        run / "logs/validator.stdout.log", run / "logs/validator.stderr.log",
    )
    if result["exit_code"] == 0:
        private_file(output, "independent telemetry validation")
        validation = json_private(output, "independent telemetry validation")
        require(validation.get("status") == VALIDATOR_PASS and
                validation.get("diagnostic_variant") == VARIANT and
                validation.get("publication_eligible") is False and
                validation.get("claim_scope") == CLAIM_SCOPE and
                validation.get("fixed_reference", {}).get("run_dir") == str(CONTROL),
                "independent validator status/boundary drift")
        result["output_sha256"] = sha256_file(output)
    else:
        result["output_sha256"] = None
    return result


def self_check() -> int:
    os.umask(0o077)
    static = verify_static()
    with tempfile.TemporaryDirectory(prefix="fresh-alloc-telemetry-selfcheck.") as tempdir:
        root = Path(tempdir)
        os.chmod(root, 0o700)
        logs = root / "logs"
        logs.mkdir(mode=0o700)
        admission_hash = run_preflight(root / "admission.env", logs)
        validator = run_cpu(
            [str(SYSTEM_PYTHON), "-I", "-S", str(VALIDATOR), "--self-check"],
            logs / "validator.stdout.log", logs / "validator.stderr.log",
        )
        require(validator["exit_code"] == 0, "validator CPU self-check failed")
        try:
            validator_stdout = json.loads((logs / "validator.stdout.log").read_text("utf-8").strip())
        except Exception as error:
            raise GateError("validator self-check JSON missing") from error
        require(validator_stdout.get("status") == VALIDATOR_SELF_CHECK_PASS and
                validator_stdout.get("claim_scope") == CLAIM_SCOPE,
                "validator self-check status/boundary drift")
    print(json.dumps({
        "schema": SCHEMA, "status": "PASS_CPU_ONLY_STATIC_SELFCHECK",
        "diagnostic_variant": VARIANT, "publication_eligible": False,
        "claim_scope": CLAIM_SCOPE, "gpu_workload_launched": False,
        "nvml_used": False, "nvidia_smi_used": False,
        "guard_sha256": static["guard_sha256"], "artifacts": static,
        "admission_sha256": admission_hash, "validator_cpu_self_check": True,
        "scope": "static/source/build/preflight/validator checks only; no NVML and no CUDA binary",
    }, sort_keys=True))
    return 0


def normal(args: argparse.Namespace) -> int:
    validate_args(args)
    os.umask(0o077)
    static = verify_static()
    run = RUNS / args.run_name
    require(not run.exists() and not run.is_symlink(), "run directory already exists")
    run.mkdir(mode=0o700)
    logs = run / "logs"
    logs.mkdir(mode=0o700)
    admission = run / "admission.env"
    receipt = run / "guard_receipt.json"
    pending_receipt = run / "receipt_for_validator.json"
    card: dict[str, Any] = {
        "schema": SCHEMA, "status": "ADMISSION_READY",
        "diagnostic_variant": VARIANT, "publication_eligible": False,
        "claim_scope": CLAIM_SCOPE,
        "scope": (
            "fresh-allocation semantic control: preserve original per-query cudaMalloc/cudaFree "
            "path and emit correctness/allocator/schedule witnesses with a sealed sidecar; "
            "this guard makes no performance, current-implementation, or publication claim"
        ),
        "host": os.uname().nodename, "run_name": args.run_name,
        "approval_id": args.approval_id,
        "token": {
            "nonce_sha256": hashlib.sha256(args.nonce.encode("ascii")).hexdigest(),
            "issued_boottime_ns": 0, "expires_boottime_ns": 0,
            "ttl_seconds": args.ttl_seconds,
        },
        "gpu": {
            "requested_ordinal": GPU_ORDINAL, "expected_uuid": GPU_UUID,
            "launch_cuda_visible_devices": GPU_UUID, "prelaunch_idle_samples": [],
        },
        "inputs": {
            "bundle_path": str(BUNDLE), "admission_path": str(admission),
            "runner_source_path": str(RUNNER), "compile_helper_path": str(COMPILE_HELPER),
            "object_path": str(OBJECT), "binary_path": str(BINARY),
            "validator_path": str(VALIDATOR), "guard_path": str(GUARD),
            "variant_manifest_path": str(MANIFEST),
            "reviewed_diff_path": str(REVIEWED_DIFF), "static_audit_path": str(STATIC_AUDIT),
            "input_files_sha256": INPUT_HASHES, "control_hashes": CONTROL_HASHES,
            # ``manifest_sha256`` remains for historical guard diagnostics;
            # the immutable receipt exposes the validator's explicit name.
            "source_manifest_sha256": static["manifest_sha256"],
            **static,
        },
        "outputs": {
            "events": str(run / "engine.jsonl"), "summary": str(run / "summary.json"),
            "runner_stdout": str(logs / "runner.stdout.log"),
            "witness": str(run / "fresh_alloc_witness.jsonl"),
            "validator": str(run / "independent_validation.json"),
        },
        "do_not_touch": [
            "GPU ordinal other than 1", "any pre-existing process",
            "fixed v1b control run", "/workspace/legacy_workspace/GTS",
            "/workspace/project/GTS", "A800-1", "A800-2", "A800-3",
        ],
        "limitations": [
            "fresh_allocation_control_only", "semantic_output_contract_only",
            "no_performance_claim", "hash_bound_historical_reference_only",
        ],
    }
    try:
        card["inputs"]["admission_sha256"] = run_preflight(admission, logs)
    except GateError as error:
        card["status"] = "BLOCKED_CPU_PREFLIGHT"
        card["error"] = str(error)
        atomic_json(receipt, card)
        raise

    # After sample 3, launch direct child immediately: no compile/diff/control work.
    samples: list[dict[str, Any]] = []
    pci_bus_id: str | None = None
    try:
        for sample_index in range(1, SAMPLES + 1):
            fields, raw = read_snapshot()
            if pci_bus_id is None:
                pci_bus_id = fields["gpu_pci_bus_id"]
            else:
                require(fields["gpu_pci_bus_id"] == pci_bus_id,
                        "GPU PCI identity drift during prelaunch samples")
            write_new(logs / ("prelaunch_nvml_" + str(sample_index) + ".txt"), raw, "ascii")
            entry = snapshot_entry(sample_index, fields, raw)
            require(not samples or entry["boottime_ns"] > samples[-1]["boottime_ns"],
                    "prelaunch snapshot freshness/order drift")
            samples.append(entry)
            if sample_index < SAMPLES:
                time.sleep(SAMPLE_INTERVAL_SECONDS)
    except GateError as error:
        card["status"] = "BLOCKED_PRELAUNCH_NVML"
        card["gpu"]["prelaunch_idle_samples"] = samples
        card["error"] = str(error)
        atomic_json(receipt, card)
        raise

    issued_ns = boot_ns()
    require(issued_ns > samples[-1]["boottime_ns"],
            "TTL issue time is not after final fresh idle sample")
    expiry_ns = issued_ns + args.ttl_seconds * 1_000_000_000
    card["token"]["issued_boottime_ns"] = issued_ns
    card["token"]["expires_boottime_ns"] = expiry_ns
    card["gpu"]["prelaunch_idle_samples"] = samples
    card["status"] = "RUNNING"
    atomic_json(receipt, card)

    runner = run_binary(run, admission, args.nonce, expiry_ns)
    card["runner"] = runner
    output_ready = False
    if runner["exit_code"] == 0 and not runner["timed_out"] and runner["finished_within_ttl"]:
        try:
            verify_runner_outputs(run, runner)
            output_ready = True
        except GateError as error:
            runner["output_error"] = str(error)

    postrun_idle = False
    try:
        fields, raw = read_snapshot()
        require(fields["gpu_pci_bus_id"] == pci_bus_id,
                "GPU PCI identity drift after CUDA child")
        write_new(logs / "postrun_nvml.txt", raw, "ascii")
        postrun = {
            "boottime_ns": boot_ns(), "fields": fields,
            "raw_sha256": hashlib.sha256(raw.encode("ascii")).hexdigest(),
        }
        require(postrun["boottime_ns"] >= runner["finished_boottime_ns"] and
                postrun["boottime_ns"] > samples[-1]["boottime_ns"],
                "postrun snapshot freshness/order drift")
        card["gpu"]["postrun_idle_snapshot"] = postrun
        postrun_idle = True
    except GateError as error:
        card["gpu"]["postrun_snapshot_error"] = str(error)

    validator: dict[str, Any] = {"exit_code": None, "output_sha256": None}
    pending_receipt_sha256: str | None = None
    validator_exception: str | None = None
    if output_ready and postrun_idle:
        # First persist the mutable operational receipt, then seal an immutable
        # snapshot of this exact pending stage for the independent validator.
        # The latter is never passed to atomic_json and therefore cannot be
        # overwritten by the final guard status update below.
        card["status"] = "RUNNER_EXITED_PENDING_VALIDATION"
        atomic_json(receipt, card)
        pending_receipt_sha256 = write_immutable_json(
            pending_receipt, card, "immutable pending-stage validator receipt")
        try:
            validator = run_validator(run, admission, pending_receipt)
        except Exception as error:
            # The immutable pre-validator receipt remains write-once evidence,
            # while the mutable operational receipt must never be left pending
            # if validator launch or its output-contract check raises.
            validator_exception = type(error).__name__ + ": " + str(error)
            validator = {
                "exit_code": None,
                "output_sha256": None,
                "error": validator_exception,
            }
    if pending_receipt_sha256 is not None:
        card["validator_stage_receipt"] = {
            "path": str(pending_receipt),
            "sha256": pending_receipt_sha256,
            "immutable": True,
            "status_at_seal": "RUNNER_EXITED_PENDING_VALIDATION",
        }
    card["validator"] = validator
    if validator_exception is not None:
        card["status"] = "FAILED_VALIDATION"
        card["error"] = "validator stage exception: " + validator_exception
    elif pending_receipt_sha256 is not None and validator["exit_code"] != 0:
        card["status"] = "FAILED_VALIDATION"
        card["error"] = "validator stage nonzero exit: " + str(validator["exit_code"])
    else:
        card["status"] = (
            "PASS_FRESH_ALLOCATION_SEMANTIC_CONTROL_GUARDED"
            if output_ready and postrun_idle and validator["exit_code"] == 0
            else "FAILED_OR_BLOCKED"
        )
    atomic_json(receipt, card)
    print(json.dumps({
        "status": card["status"], "run_dir": str(run),
        "gpu_ordinal": GPU_ORDINAL, "gpu_uuid": GPU_UUID,
        "runner_exit": runner["exit_code"], "validator_exit": validator["exit_code"],
        "claim": "semantic_fresh_allocation_control_only",
    }, sort_keys=True))
    return 0 if card["status"].startswith("PASS_") else 2


def main() -> int:
    reject_duplicate_cli_options(sys.argv[1:])
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--gpu-ordinal", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--approval-id")
    parser.add_argument("--nonce")
    parser.add_argument("--ttl-seconds", type=int)
    args = parser.parse_args()
    if args.self_check:
        require(all(value is None for value in (
            args.gpu_ordinal, args.run_name, args.approval_id, args.nonce, args.ttl_seconds,
        )), "self-check cannot accept launch arguments")
        return self_check()
    require(all(value is not None for value in (
        args.gpu_ordinal, args.run_name, args.approval_id, args.nonce, args.ttl_seconds,
    )), "exact GPU/run-name/approval-id/nonce/TTL arguments are required")
    return normal(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print("guard blocked: " + str(error), file=sys.stderr)
        raise SystemExit(2)
