#!/usr/bin/env python3
"""Create the v7 source-only integrity root without compiling or launching CUDA."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import pathlib
from datetime import datetime, timezone

ROOT_DEFAULT = pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v7_profile_cpu_preflight")
EXTERNAL_INPUTS = {
    "base": pathlib.Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt"),
    "source_update_trace": pathlib.Path("/workspace/legacy_workspace/GTS/Datasets/update_workloads/sift_1m_10k_442.txt"),
}
INPUT_PATHS = {
    "base": EXTERNAL_INPUTS["base"],
    "source_update_trace": EXTERNAL_INPUTS["source_update_trace"],
    "derived_query_only_trace": pathlib.Path("inputs/sift1m_10k_442_first1024_type2_query_only.txt"),
    "derived_trace_provenance": pathlib.Path("inputs/sift1m_10k_442_first1024_type2_provenance.json"),
}
PROTOCOL_PATHS = {
    "formal_supersession": "protocol_c1_v7_formal_supersession_v1.json",
    "frozen_protocol": "protocol_c1_workspace_onequery_sift1m_v1_v7_local.json",
    "gpu_uuid_safety": "protocol_c1_gpu_uuid_safety_v7_local.json",
    "idle_poll_safety": "protocol_c1_gpu_idle_poll_safety_v7_local.json",
    "idle_poll_logdir": "protocol_c1_gpu_idle_poll_logdir_v7_local.json",
    "implementation_correction": "implementation_correction_capacity_snapshot_v1.json",
    "execution_plan": "measured_execution_plan_v7.json",
    "profile_execution_plan": "profile_execution_plan_v7.json",
}
HELPER_PATHS = {
    "guard": "run_c1_workspace_microbenchmark_guard_v7.sh",
    "measured_engine": "run_c1_four_variants_v7.sh",
    "profile_engine": "run_c1_profile_primary_v7.sh",
    "profile_guard": "run_c1_profile_guard_v7.sh",
    "profile_manifest_utility": "c1_v7_profile_manifest.py",
    "strict_semantic_verifier": "verify_c1_semantics_v7.py",
    "measured_analyzer": "analyze_c1_microbenchmark_v7.py",
    "profile_verifier": "verify_c1_profile_artifacts_v7.py",
    "measured_finalizer": "finalize_c1_evidence_v7.py",
    "formal_aggregator": "aggregate_c1_formal_evidence_v7.py",
    "manifest_utility": "c1_v7_manifest.py",
    "runtime_pin_verifier": "verify_v7_pins.py",
    "runtime_pin_generator": "create_v7_pins.py",
    "build_script": "build_v7_variants.sh",
    "build_manifest_creator": "create_v7_build_manifest.py",
    "capacity_snapshot_contract": "capacity_snapshot_contract_v7.py",
    "capacity_snapshot_fixture": "tools/test_c1_v7_capacity_snapshot_fixtures.py",
    "source_pin_verifier": "verify_v7_source_pins.py",
    "source_pin_generator": "create_v7_source_pins.py",
    "canonical_inspector_source": "tools/canonical_inspect_v7.cpp",
    "guard_session_fixture": "tools/test_c1_v7_guard_session_fixtures.py",
}
FIXTURE_PATHS = {
    "positive": "tools/fixtures/capacity_snapshot_positive.json",
    "negative_zero": "tools/fixtures/capacity_snapshot_negative_zero.json",
    "negative_mismatch": "tools/fixtures/capacity_snapshot_negative_mismatch.json",
    "negative_stage": "tools/fixtures/capacity_snapshot_negative_stage.json",
    "negative_result_count": "tools/fixtures/capacity_snapshot_negative_result_count.json",
    "negative_overflow": "tools/fixtures/capacity_snapshot_negative_overflow.json",
    "profile_manifest_execution": "tools/test_c1_v7_profile_manifest_fixtures.py",
    "profile_terminal_execution": "tools/test_c1_v7_profile_terminal_fixture.py",
    "guard_session_execution": "tools/test_c1_v7_guard_session_fixtures.py",
    "profile_guard_cli_contract_execution": "tools/test_c1_v7_profile_guard_cli_contract.py",
}


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def regular(path: pathlib.Path, label: str) -> pathlib.Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label}: absolute regular non-symlink file required: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise SystemExit(f"{label}: noncanonical/symlink traversal: {path} -> {resolved}")
    return path


def directory(path: pathlib.Path, label: str) -> pathlib.Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise SystemExit(f"{label}: absolute non-symlink directory required: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise SystemExit(f"{label}: noncanonical/symlink traversal: {path} -> {resolved}")
    return path


def entry(path: pathlib.Path, label: str) -> dict[str, object]:
    path = regular(path, label)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def active_worktree_files(root: pathlib.Path) -> list[pathlib.Path]:
    worktree = root / "worktree"
    expected = [
        worktree / "CMakeLists.txt",
        worktree / "src" / "main.cu",
        worktree / "src" / "c1_microbench.cu",
        *[worktree / "include" / name for name in (
            "c1_alloc_counter.cuh", "config.cuh", "file.cuh", "gpu_timer.cuh",
            "incremental_insert.cuh", "mlp_constant.cuh", "residual_pruning.cuh",
            "residual_tuner.cuh", "search.cuh", "search_naive.cuh", "search_v2.cuh",
            "tree.cuh", "update.cuh",
        )],
    ]
    actual = sorted(p for p in worktree.rglob("*") if p.is_file() and not p.is_symlink())
    if set(actual) != set(expected):
        unexpected = sorted(str(p.relative_to(worktree)) for p in set(actual) - set(expected))
        missing = sorted(str(p.relative_to(worktree)) for p in set(expected) - set(actual))
        raise SystemExit(f"worktree active-source set mismatch: unexpected={unexpected}, missing={missing}")
    return expected


def forbid_runtime_artifacts(root: pathlib.Path) -> None:
    for name in ("runs", "profiles", "smoke", "launch_logs", "builds", "build_logs", ".c1_v7_gpu0.lock", "hardened_static_pins_v7.json", "worktree_build_manifest_v7.json"):
        if (root / name).exists() or (root / name).is_symlink():
            raise SystemExit(f"source-only root must not contain runtime/build artifact: {name}")
    bytecode = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if ".git" not in p.parts and (p.name == "__pycache__" or p.suffix == ".pyc")
    )
    if bytecode:
        raise SystemExit("source-only root must not contain Python bytecode: " + ", ".join(bytecode))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=ROOT_DEFAULT)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    root = directory(args.root, "root")
    if root != ROOT_DEFAULT:
        raise SystemExit("source pins must bind the canonical v7 root")
    forbid_runtime_artifacts(root)
    worktree_entries = []
    for path in active_worktree_files(root):
        item = entry(path, "worktree source")
        item["relative_path"] = str(path.relative_to(root / "worktree"))
        worktree_entries.append(item)
    inputs = {
        key: entry(path if path.is_absolute() else root / path, f"input {key}")
        for key, path in INPUT_PATHS.items()
    }
    protocols = {key: entry(root / rel, f"protocol {key}") for key, rel in PROTOCOL_PATHS.items()}
    helpers = {key: entry(root / rel, f"helper {key}") for key, rel in HELPER_PATHS.items()}
    fixtures = {key: entry(root / rel, f"fixture {key}") for key, rel in FIXTURE_PATHS.items()}
    payload = {
        "schema": "gtspp-c1-v7-source-pins-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "scope": "Source-only v7 profile-manifest-repair integrity root. It pins the executable CPU-only artifact/terminal/real-session fixtures. No NVCC build, CUDA binary, nsys, nvidia-smi, GPU query, or workload is represented or authorized by this pin set.",
        "source_only": True,
        "allowed_external_paths": [str(EXTERNAL_INPUTS["base"]), str(EXTERNAL_INPUTS["source_update_trace"])],
        "inputs": inputs,
        "protocols": protocols,
        "helpers": helpers,
        "fixtures": fixtures,
        "worktree_source_files": worktree_entries,
        "capacity_snapshot_contract": {
            "stage": "pre_ephemeral_release",
            "local_variable": "c1_capacity_snapshot_pre_release",
            "fields": ["update_result_capacity_slots", "total_result_capacity_slots"],
            "result_count_field": "result_count",
            "required_gate": "result_count is an integer >= 0 and <= both capacity fields; both capacity fields are integer > 0, exactly equal, emitted from one pre-release local snapshot, and never read from the global capacity after release",
        },
    }
    out = args.out or root / "hardened_source_pins_v7.json"
    if not out.is_absolute() or out.is_symlink() or out.resolve(strict=False) != out or out.parent != root:
        raise SystemExit("source pin output must be a direct canonical file under the v7 root")
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(json.dumps({"pass": True, "mode": "SOURCE_PIN_GENERATION_CPU_ONLY", "pins": str(out), "sha256": sha(out), "worktree_sources": len(worktree_entries), "helpers": len(helpers), "fixtures": len(fixtures)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
