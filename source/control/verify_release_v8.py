#!/usr/bin/env python3
"""CPU-only verifier for the physically separated C1 v8 release closure.

This verifier never invokes CUDA, nvidia-smi, Nsight, a benchmark binary, or a
subprocess. It validates the copied v7 source/build provenance, release PINS,
input hashes, fixed measured plan, and physical source/build separation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import stat
import sys

ROOT_DEFAULT = pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c1_execution_release_v8_external_one_time_approval")
EXPECTED_V7_COMMIT = "4c7be18318371d970e54633afe050455b16e30ab"
EXPECTED_V7_STATIC_PINS = "a865ff0bdf4eaea7183036b927665bf47b14d0cd654ddd287a6dae02ea0f1997"
EXPECTED_V7_SOURCE_PINS = "f2ede0a614548974944075a69cccfd3a5024619573552037d06c2a9a85cf9166"
EXPECTED_V7_BUILD_MANIFEST = "a674e62b8e1b9055149973dc0f445f5cd7f523ff65bcab22a39eb6c722cd4792"
EXPECTED_V7_MEASURED_PLAN = "782c9e5199ba38d61251791d3bf94ef993319eaeaf50ec41b77e66bf0c7e45d4"


def fail(message: str) -> None:
    raise SystemExit("C1-V8-VERIFY BLOCKED: " + message)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(path: pathlib.Path, *, directory: bool, label: str) -> pathlib.Path:
    if not path.is_absolute():
        fail(f"{label}: absolute path required")
    try:
        info = path.lstat()
    except FileNotFoundError:
        fail(f"{label}: missing {path}")
    if stat.S_ISLNK(info.st_mode):
        fail(f"{label}: symlink forbidden {path}")
    wanted = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not wanted:
        kind = "directory" if directory else "regular file"
        fail(f"{label}: {kind} required {path}")
    try:
        resolved = path.resolve(strict=True)
    except RuntimeError:
        fail(f"{label}: cannot resolve {path}")
    if resolved != path:
        fail(f"{label}: noncanonical path {path} -> {resolved}")
    return path


def load_object(path: pathlib.Path, label: str) -> dict:
    safe_path(path, directory=False, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"{label}: invalid JSON: {exc}")
    if not isinstance(value, dict):
        fail(f"{label}: JSON object required")
    return value


def exact_entry(value: object, path: pathlib.Path, label: str) -> None:
    if not isinstance(value, dict):
        fail(f"{label}: entry object required")
    if value.get("path") != str(path) or value.get("sha256") != sha256(path) or value.get("bytes") != path.stat().st_size:
        fail(f"{label}: path/hash/bytes mismatch")


def list_regulars(stage: pathlib.Path) -> set[str]:
    found: set[str] = set()
    for path in stage.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            fail(f"stage symlink forbidden: {path}")
        if stat.S_ISREG(info.st_mode):
            found.add(str(path.relative_to(stage)))
        elif not stat.S_ISDIR(info.st_mode):
            fail(f"nonregular stage object forbidden: {path}")
    return found


def verify_source_stage(root: pathlib.Path, origin: dict, upstream_bm: dict, upstream_pins: dict) -> dict:
    stage = safe_path(root / "source_only_stage", directory=True, label="source-only stage")
    manifest_path = stage / "source_stage_manifest_v8.json"
    manifest = load_object(manifest_path, "source-stage manifest")
    if manifest.get("schema") != "gtspp-c1-v8-source-only-stage-v1" or manifest.get("origin") != origin or manifest.get("stage_root") != str(stage):
        fail("source-stage manifest identity/origin")
    sep = manifest.get("physical_separation")
    if not isinstance(sep, dict) or sep.get("contains_build_artifacts") is not False:
        fail("source-stage separation declaration")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 18:
        fail("source-stage file list")
    expected: set[str] = {"source_stage_manifest_v8.json"}
    upstream_sources = {"worktree/" + x["relative_path"]: x for x in upstream_bm.get("source_files", [])}
    upstream_helpers = upstream_pins.get("runtime_helpers", {})
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("relative_path"), str):
            fail("source-stage file item")
        rel = item["relative_path"]
        path = safe_path(stage / rel, directory=False, label="source-stage file")
        if path.relative_to(stage).as_posix() != rel or sha256(path) != item.get("sha256") or path.stat().st_size != item.get("bytes"):
            fail(f"source-stage copied-file mismatch: {rel}")
        if item.get("sha256") != item.get("upstream_sha256"):
            fail(f"source-stage upstream hash mismatch: {rel}")
        if rel in upstream_sources:
            upstream = upstream_sources[rel]
            if item.get("upstream_path") != upstream.get("path") or item.get("sha256") != upstream.get("sha256"):
                fail(f"source-stage build-manifest provenance: {rel}")
        elif rel == "tools/canonical_inspect_v7.cpp":
            ci = upstream_bm.get("canonical_inspector", {})
            if item.get("upstream_path") != ci.get("source") or item.get("sha256") != ci.get("source_sha256"):
                fail("canonical inspector source provenance")
        elif rel == "capacity_snapshot_contract_v8.py":
            helper = upstream_helpers.get("capacity_snapshot_contract", {})
            if item.get("upstream_path") != helper.get("path") or item.get("sha256") != helper.get("sha256"):
                fail("capacity contract provenance")
        else:
            fail(f"unexpected source-stage member: {rel}")
        expected.add(rel)
    copied_build_source = {x["relative_path"] for x in files if x["relative_path"].startswith("worktree/")}
    if set(upstream_sources) != copied_build_source:
        fail("source-stage misses/adds upstream build source")
    actual = list_regulars(stage)
    if actual != expected:
        fail("source-stage contains unexpected/missing physical files")
    for path in stage.rglob("*"):
        if path.name in {"build", "builds", "CMakeCache.txt", "C1Microbench"}:
            fail(f"source-only stage contains build artifact name: {path}")
    return manifest


def verify_build_stage(root: pathlib.Path, origin: dict, upstream_bm: dict) -> dict:
    stage = safe_path(root / "build_stage", directory=True, label="build stage")
    manifest_path = stage / "build_stage_manifest_v8.json"
    manifest = load_object(manifest_path, "build-stage manifest")
    if manifest.get("schema") != "gtspp-c1-v8-build-stage-v1" or manifest.get("origin") != origin or manifest.get("stage_root") != str(stage):
        fail("build-stage manifest identity/origin")
    sep = manifest.get("physical_separation")
    if not isinstance(sep, dict) or sep.get("contains_compiler_source") is not False:
        fail("build-stage separation declaration")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 9:
        fail("build-stage artifact list")
    variants = {x["name"]: x for x in upstream_bm.get("variants", [])}
    expected = {"build_stage_manifest_v8.json"}
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("relative_path"), str):
            fail("build-stage item")
        rel = item["relative_path"]
        path = safe_path(stage / rel, directory=False, label="build-stage artifact")
        if sha256(path) != item.get("sha256") or path.stat().st_size != item.get("bytes") or item.get("sha256") != item.get("upstream_sha256"):
            fail(f"build-stage copy/hash mismatch: {rel}")
        kind, variant = item.get("kind"), item.get("variant")
        if kind in {"binary", "cmake_cache"}:
            upstream = variants.get(variant)
            if upstream is None:
                fail(f"unknown build-stage variant {variant}")
            field = "binary" if kind == "binary" else "cmake_cache"
            hash_field = field + "_sha256"
            if item.get("upstream_path") != upstream.get(field) or item.get("sha256") != upstream.get(hash_field):
                fail(f"build-stage upstream provenance: {variant}/{kind}")
        elif kind == "canonical_inspector_binary":
            ci = upstream_bm.get("canonical_inspector", {})
            if item.get("upstream_path") != ci.get("binary") or item.get("sha256") != ci.get("binary_sha256"):
                fail("build-stage canonical inspector provenance")
        else:
            fail("unknown build-stage artifact kind")
        expected.add(rel)
    if list_regulars(stage) != expected:
        fail("build-stage contains unexpected/missing physical files")
    for path in stage.rglob("*"):
        if path.suffix in {".cu", ".cuh"} or path.name == "CMakeLists.txt":
            fail(f"build stage contains compiler source: {path}")
    return manifest


def verify_release(root: pathlib.Path) -> dict:
    root = safe_path(root, directory=True, label="release root")
    if root != ROOT_DEFAULT:
        fail("canonical v8 release root required")
    provenance = safe_path(root / "provenance", directory=True, label="provenance")
    v7_pins_path = provenance / "upstream_hardened_static_pins_v7.json"
    v7_source_pins_path = provenance / "upstream_hardened_source_pins_v7.json"
    v7_bm_path = provenance / "upstream_worktree_build_manifest_v7.json"
    v7_plan_path = provenance / "upstream_measured_execution_plan_v7.json"
    v7_pins = load_object(v7_pins_path, "upstream static PINS")
    _ = load_object(v7_source_pins_path, "upstream source PINS")
    upstream_bm = load_object(v7_bm_path, "upstream build manifest")
    upstream_plan = load_object(v7_plan_path, "upstream measured plan")
    if sha256(v7_pins_path) != EXPECTED_V7_STATIC_PINS or sha256(v7_source_pins_path) != EXPECTED_V7_SOURCE_PINS or sha256(v7_bm_path) != EXPECTED_V7_BUILD_MANIFEST or sha256(v7_plan_path) != EXPECTED_V7_MEASURED_PLAN:
        fail("copied v7 provenance hash")
    origin = {
        "v7_root": "/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v7_profile_cpu_preflight",
        "v7_git_commit": EXPECTED_V7_COMMIT,
        "v7_static_pins_sha256": EXPECTED_V7_STATIC_PINS,
        "v7_source_pins_sha256": EXPECTED_V7_SOURCE_PINS,
        "v7_build_manifest_sha256": EXPECTED_V7_BUILD_MANIFEST,
        "v7_measured_plan_sha256": EXPECTED_V7_MEASURED_PLAN,
    }
    verify_source_stage(root, origin, upstream_bm, v7_pins)
    verify_build_stage(root, origin, upstream_bm)
    control = safe_path(root / "control", directory=True, label="control")
    plan_path = control / "measured_execution_plan_v8.json"
    pins_path = control / "release_pins_v8.json"
    plan, pins = load_object(plan_path, "release plan"), load_object(pins_path, "release PINS")
    if plan.get("schema") != "gtspp-c1-v8-measured-execution-plan-v1" or plan.get("mode") != "MEASURED_5X4_C1_V8" or plan.get("origin") != origin:
        fail("release plan identity")
    if plan.get("schedule") != upstream_plan.get("schedule") or len(plan.get("schedule", [])) != 20:
        fail("release plan schedule drift")
    expected_gpu = {"physical_index": 0, "expected_uuid": "GPU-CONFIGURE-ARCHIVE-DEVICE", "strict_idle": {"memory_used_mib_max": 256, "utilization_gpu_pct": 0, "compute_apps": "empty"}}
    if plan.get("host") != "CONFIGURE_ARCHIVE_HOST" or plan.get("gpu") != expected_gpu:
        fail("release host/GPU binding")
    expected_policy = {"allow_gpu_telemetry": True, "allow_cuda_binary": True, "allow_nsys": False, "kill_preexisting_processes": False, "c2_residual_pruning_mode": 0, "c3_measured_mutations": "forbidden"}
    if plan.get("policy") != expected_policy:
        fail("release policy")
    if pins.get("schema") != "gtspp-c1-v8-release-pins-v1" or pins.get("origin") != origin:
        fail("release PINS identity")
    exact_entry(pins.get("source_stage_manifest"), root / "source_only_stage/source_stage_manifest_v8.json", "PINS source-stage manifest")
    exact_entry(pins.get("build_stage_manifest"), root / "build_stage/build_stage_manifest_v8.json", "PINS build-stage manifest")
    exact_entry(pins.get("execution_plan"), plan_path, "PINS execution plan")
    if pins.get("inputs") != plan.get("inputs") or pins.get("binaries") != plan.get("variants"):
        fail("PINS plan input/binary binding")
    for entry in pins.get("inputs", []):
        if not isinstance(entry, dict):
            fail("PINS input entry")
        path = pathlib.Path(str(entry.get("path", "")))
        safe_path(path, directory=False, label="pinned input")
        if sha256(path) != entry.get("sha256") or path.stat().st_size != entry.get("bytes"):
            fail(f"PINS input drift: {path}")
    for entry in pins.get("binaries", []):
        if not isinstance(entry, dict):
            fail("PINS binary entry")
        path = pathlib.Path(str(entry.get("path", "")))
        safe_path(path, directory=False, label="pinned binary")
        if sha256(path) != entry.get("sha256") or path.stat().st_size != entry.get("bytes"):
            fail(f"PINS binary drift: {path}")
    canonical = root / "build_stage/canonical_inspect_v7"
    exact_entry(pins.get("canonical_inspector"), canonical, "PINS canonical inspector")
    exact_entry(pins.get("cpu_fixture_runner"), control / "run_cpu_fixtures_v8.py", "PINS CPU fixture runner")
    exact_entry(pins.get("external_approval_schema"), control / "external_approval_schema_v8.json", "PINS approval schema")
    control_files = pins.get("control_files")
    expected_control_paths = {"outer_guard": control / "run_c1_execution_guard_v8.sh", "approval_gate": control / "approval_gate_v8.py", "release_verifier": control / "verify_release_v8.py", "measured_launcher": control / "launch_c1_execution_v8.sh", "cpu_fixture_runner": control / "run_cpu_fixtures_v8.py", "external_approval_schema": control / "external_approval_schema_v8.json"}
    if not isinstance(control_files, dict) or set(control_files) != set(expected_control_paths):
        fail("PINS control-file closure keys")
    for name, path in expected_control_paths.items():
        exact_entry(control_files.get(name), path, "PINS control file " + name)
    policy = pins.get("cpu_fixture_policy")
    expected_fixture_policy = {"required_receipts":["static_release_closure","direct_env_bypass_negative","bad_external_approval_negative","one_time_replay_o_excl","capacity_snapshot_contract"],"production_approval_created":False,"forbidden_tools":["nvidia-smi","nsys","nvcc","C1Microbench"]}
    if policy != expected_fixture_policy:
        fail("PINS CPU fixture policy")
    safe_path(root / "input_stage", directory=True, label="input stage")
    return {
        "pass": True,
        "mode": plan["mode"],
        "v7_commit": EXPECTED_V7_COMMIT,
        "release_pins_sha256": sha256(pins_path),
        "plan_sha256": sha256(plan_path),
        "source_stage_sha256": sha256(root / "source_only_stage/source_stage_manifest_v8.json"),
        "build_stage_sha256": sha256(root / "build_stage/build_stage_manifest_v8.json"),
        "binaries": len(pins["binaries"]),
        "inputs": len(pins["inputs"]),
        "gpu_tools_invoked": False,
        "cuda_binary_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_release(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
