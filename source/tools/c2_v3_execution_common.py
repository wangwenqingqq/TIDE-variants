#!/usr/bin/env python3
"""Shared, CPU-only contract helpers for the Safe-C2 v3 execution trust root.

Nothing in this module imports CUDA, invokes a binary, or invokes nvidia-smi.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

ROOT_LITERAL = "/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v3"
V2_ROOT_LITERAL = "/workspace/experiments/tide_safe_c1_20260727/c2_interval_bound_safe_v2_tiefree"
V2_SEALED_TEST_SHA256 = "50ccb28263bf23e499b9c50e4d3e9800fca3949aa5b5ea8a454237d5987701ce"
WORKLOAD_SHA256 = "72d9b0732784d8f0f5b77d564a65d9a9294feab9781f0a098131fe74129fbf39"
WORKLOAD_SCHEMA = "gts-v3-compact-learn-workload-v1"
PLAN_SCHEMA = "safe-c2-v3-execution-plan-v1"
TRUST_SCHEMA = "safe-c2-v3-execution-trust-root-v1"
PINS_SCHEMA = "safe-c2-v3-execution-pins-v1"
MANIFEST_SCHEMA = "safe-c2-v3-execution-manifest-v1"
LEDGER_SCHEMA = "safe-c2-v3-irrevocable-consumption-ledger-v1"
RUN_RE = re.compile(r"^c2_v3_exec_[0-9]{8}T[0-9]{6}Z_[0-9]{9}$")
STAGE_ORDER = ("calibration", "validation", "sealed_test")
STAGE_RUNNER_NAME = {"calibration": "calibration", "validation": "validation", "sealed_test": "test"}
STAGE_MODE = {"calibration": "calibrate", "validation": "evaluate", "sealed_test": "evaluate"}
STAGE_TIMED_REPS = {"calibration": 1, "validation": 3, "sealed_test": 5}


class ContractError(RuntimeError):
    pass


def root_from_module(module_file: str) -> Path:
    root = Path(module_file).resolve().parents[1]
    if str(root) != ROOT_LITERAL or root.name != "c2_speculative_fallback_v3":
        raise ContractError(f"unexpected C2 v3 root: {root}")
    if root.is_symlink():
        raise ContractError(f"C2 v3 root must not be a symlink: {root}")
    return root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fnv1a64(data: bytes) -> str:
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return "0x" + format(value, "x")


def require_regular_file(path: Path, label: str, *, inside_root: bool = False) -> Path:
    if not path.is_absolute():
        raise ContractError(f"{label} is not absolute: {path}")
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} must be a direct regular file: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ContractError(f"{label} resolves differently: {path} -> {resolved}")
    if inside_root:
        root = Path(ROOT_LITERAL)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ContractError(f"{label} leaves v3 root: {path}") from error
    return path


def require_direct_dir(path: Path, label: str, *, inside_root: bool = False) -> Path:
    if not path.is_absolute():
        raise ContractError(f"{label} is not absolute: {path}")
    if path.is_symlink() or not path.is_dir():
        raise ContractError(f"{label} must be a direct directory: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ContractError(f"{label} resolves differently: {path} -> {resolved}")
    if inside_root:
        try:
            path.relative_to(Path(ROOT_LITERAL))
        except ValueError as error:
            raise ContractError(f"{label} leaves v3 root: {path}") from error
    return path


def load_json(path: Path, label: str) -> dict[str, Any]:
    require_regular_file(path, label, inside_root=str(path).startswith(ROOT_LITERAL + "/"))
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ContractError(f"cannot parse {label}: {error}") from error
    if not isinstance(loaded, dict):
        raise ContractError(f"{label} is not a JSON object")
    return loaded


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    if not path.is_absolute() or path.is_symlink():
        raise ContractError(f"refusing non-direct JSON write: {path}")
    parent = path.parent
    require_direct_dir(parent, "JSON parent", inside_root=True)
    data = canonical_json_bytes(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def ensure_direct_child(child: Path, parent: Path, label: str) -> None:
    if not child.is_absolute() or child.is_symlink():
        raise ContractError(f"{label} must be direct/absolute: {child}")
    try:
        child.relative_to(parent)
    except ValueError as error:
        raise ContractError(f"{label} leaves parent {parent}: {child}") from error


def parse_source_manifest(path: Path) -> list[tuple[str, Path]]:
    require_regular_file(path, "source manifest", inside_root=True)
    result: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise ContractError(f"malformed source-manifest row: {raw!r}")
        entry = Path(parts[1])
        require_regular_file(entry, "source-manifest entry", inside_root=True)
        if entry in seen:
            raise ContractError(f"duplicate source-manifest entry: {entry}")
        seen.add(entry)
        if sha256_file(entry) != parts[0]:
            raise ContractError(f"source-manifest hash mismatch: {entry}")
        result.append((parts[0], entry))
    if len(result) < 10:
        raise ContractError("source manifest unexpectedly incomplete")
    return result


def _expect(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ContractError(f"{label}: expected {expected!r}, saw {value!r}")


def load_workload(root: Path, *, hash_inputs: bool = True) -> dict[str, Any]:
    manifest_path = root / "inputs/sift_learn_compact10k_v1/workload_manifest.json"
    require_regular_file(manifest_path, "active workload manifest", inside_root=True)
    observed_manifest = sha256_file(manifest_path)
    _expect(observed_manifest, WORKLOAD_SHA256, "active workload manifest SHA-256")
    manifest = load_json(manifest_path, "active workload manifest")
    _expect(manifest.get("schema"), WORKLOAD_SCHEMA, "workload schema")
    _expect(manifest.get("sealed_v2_test_forbidden"), True, "v2 sealed-test prohibition")
    _expect(manifest.get("v2_sealed_test_ids_sha256_forbidden"), V2_SEALED_TEST_SHA256, "v2 sealed-test digest")
    _expect(manifest.get("status"), "READY_FOR_C2_V3_CALIBRATION_VALIDATION_AND_ONE_SEALED_TEST", "workload status")
    expected_count = {"calibration": 1969, "validation": 1965, "sealed_test": 5899}
    expected_top = {
        "base_fvecs": ("base_fvecs_path", "base_fvecs_sha256"),
        "query_fvecs": ("query_fvecs_path", "query_fvecs_sha256"),
        "groundtruth_ivecs": ("groundtruth_ivecs_path", "groundtruth_ivecs_sha256"),
        "mapping": ("mapping_path", "mapping_sha256"),
    }
    bindings: dict[str, dict[str, Any]] = {}
    for name, (path_key, hash_key) in expected_top.items():
        raw_path, digest = manifest.get(path_key), manifest.get(hash_key)
        if not isinstance(raw_path, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ContractError(f"invalid workload binding: {name}")
        path = Path(raw_path)
        require_regular_file(path, f"workload {name}")
        if V2_ROOT_LITERAL in raw_path:
            raise ContractError(f"workload {name} uses prohibited v2 root")
        if hash_inputs and sha256_file(path) != digest:
            raise ContractError(f"workload {name} SHA-256 mismatch")
        bindings[name] = {"path": str(path), "sha256": digest}
    stages = manifest.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(STAGE_ORDER):
        raise ContractError("workload stage set is not exactly calibration/validation/sealed_test")
    stage_bindings: dict[str, dict[str, Any]] = {}
    for name in STAGE_ORDER:
        stage = stages.get(name)
        if not isinstance(stage, dict):
            raise ContractError(f"invalid stage object: {name}")
        raw_path, digest = stage.get("ids_path"), stage.get("ids_sha256")
        if not isinstance(raw_path, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ContractError(f"invalid stage binding: {name}")
        path = Path(raw_path)
        require_regular_file(path, f"workload stage IDs {name}")
        if not str(path).startswith(str(root) + "/") or V2_ROOT_LITERAL in str(path):
            raise ContractError(f"stage IDs leave v3 root or use v2: {name}")
        if digest == V2_SEALED_TEST_SHA256:
            raise ContractError(f"stage IDs reuse v2 sealed-test digest: {name}")
        if hash_inputs and sha256_file(path) != digest:
            raise ContractError(f"stage IDs SHA-256 mismatch: {name}")
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        try:
            ids = [int(line) for line in lines]
        except ValueError as error:
            raise ContractError(f"non-integer stage ID: {name}") from error
        if len(ids) != expected_count[name] or stage.get("eligible_count") != expected_count[name]:
            raise ContractError(f"unexpected stage count: {name}")
        if len(set(ids)) != len(ids) or any(item < 0 or item >= 10000 for item in ids):
            raise ContractError(f"invalid stage IDs: {name}")
        stage_bindings[name] = {"path": str(path), "sha256": digest, "ids": ids, "count": len(ids)}
    all_ids = set().union(*(set(stage_bindings[name]["ids"]) for name in STAGE_ORDER))
    if len(all_ids) != sum(stage_bindings[name]["count"] for name in STAGE_ORDER):
        raise ContractError("workload stages overlap")
    # The source runner uses both top-level aliases and stage fields; bind both.
    top_alias = {
        "calibration": ("calibration_ids_path", "calibration_ids_sha256"),
        "validation": ("validation_ids_path", "validation_ids_sha256"),
        "sealed_test": ("test_ids_path", "test_ids_sha256"),
    }
    for stage, (path_key, digest_key) in top_alias.items():
        _expect(manifest.get(path_key), stage_bindings[stage]["path"], f"top-level {stage} IDs path")
        _expect(manifest.get(digest_key), stage_bindings[stage]["sha256"], f"top-level {stage} IDs digest")
    return {
        "path": str(manifest_path),
        "sha256": observed_manifest,
        "manifest": manifest,
        "bindings": bindings,
        "stages": stage_bindings,
    }


def expected_workflow_id(plan_sha256: str, pins_sha256: str, workload_sha256: str) -> str:
    if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in (plan_sha256, pins_sha256, workload_sha256)):
        raise ContractError("workflow-ID inputs must be SHA-256 digests")
    return "c2v3-" + sha256_text("safe-c2-v3-irrevocable-workflow-v1|" + plan_sha256 + "|" + pins_sha256 + "|" + workload_sha256)[:32]


def safe_run_dir(root: Path, raw: str) -> Path:
    candidate = Path(raw)
    runs = root / "runs"
    if not candidate.is_absolute() or candidate.parent != runs or not RUN_RE.fullmatch(candidate.name):
        raise ContractError(f"run path is not a fresh direct c2-v3 run path: {candidate}")
    if candidate.exists() or candidate.is_symlink():
        raise ContractError(f"run path already exists or is a symlink: {candidate}")
    require_direct_dir(runs, "runs directory", inside_root=True)
    return candidate


def relative_stage_file(root: Path, stage_dir: Path, path: Path) -> str:
    ensure_direct_child(path, stage_dir, "stage artifact")
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"stage artifact is not direct regular file: {path}")
    return str(path.relative_to(root))
