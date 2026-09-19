#!/usr/bin/env python3
"""CPU-only regression fixtures for the v8 C1 capacity snapshot repair.

This never invokes CUDA, NVCC, nsys, nvidia-smi, or any benchmark binary.  It
exercises both the JSON contract and the source-policy parser with a positive
fixture plus mutations that would reintroduce the post-release global read bug.
"""
from __future__ import annotations
import sys

# This script dynamically imports root-local modules; set this before that first
# import so even a default `python3 tools/test_...py` leaves no __pycache__.
sys.dont_write_bytecode = True

import importlib.util
import json
import pathlib

ROOT = pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v8_profile_child_sessions_contract")
FIXTURES = ROOT / "tools" / "fixtures"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def expect_reject(label: str, fn) -> None:
    try:
        fn()
    except (ValueError, SystemExit):
        return
    raise AssertionError(f"negative fixture unexpectedly accepted: {label}")


def main() -> int:
    contract = load_module("capacity_snapshot_contract_v8", ROOT / "capacity_snapshot_contract_v8.py")
    source = load_module("verify_v8_source_pins", ROOT / "verify_v8_source_pins.py")

    positive = fixture("capacity_snapshot_positive.json")
    if contract.capacity_snapshot_violations(positive):
        raise AssertionError("positive JSON fixture rejected")
    for name in (
        "capacity_snapshot_negative_zero.json",
        "capacity_snapshot_negative_mismatch.json",
        "capacity_snapshot_negative_stage.json",
        "capacity_snapshot_negative_result_count.json",
        "capacity_snapshot_negative_overflow.json",
    ):
        if not contract.capacity_snapshot_violations(fixture(name)):
            raise AssertionError(f"negative JSON fixture accepted: {name}")

    cpp_path = ROOT / "worktree" / "src" / "c1_microbench.cu"
    cpp = cpp_path.read_text()
    source.assert_capacity_snapshot_source_policy(cpp)
    body = source.cpp_function(cpp, "QueryResult run_query")

    # Regression mutation 1: read the global capacity only after the release.
    snapshot = "const int c1_capacity_snapshot_pre_release = update_result_ws_cap;"
    release = "releaseC1QueryWorkspace(qresult_count, qresult_count_prefix, result_id, result_dis);"
    assert body.count(snapshot) == 1 and body.count(release) == 1
    moved = body.replace(snapshot + "\n\n#if !C1_PERSISTENT_WORKSPACE\n  " + release,
                         "#if !C1_PERSISTENT_WORKSPACE\n  " + release + "\n  " + snapshot)
    expect_reject("snapshot moved after release", lambda: source.assert_capacity_snapshot_source_policy(moved))

    # Regression mutation 2: preserve a local snapshot but export a global read.
    global_export = body.replace(
        "result.total_result_capacity_slots = c1_capacity_snapshot_pre_release;",
        "result.total_result_capacity_slots = update_result_ws_cap;",
    )
    expect_reject("post-release global export", lambda: source.assert_capacity_snapshot_source_policy(global_export))

    print(json.dumps({
        "pass": True,
        "mode": "CPU_ONLY_FIXTURES",
        "json_positive": 1,
        "json_negative": 5,
        "source_positive": 1,
        "source_negative": 2,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
