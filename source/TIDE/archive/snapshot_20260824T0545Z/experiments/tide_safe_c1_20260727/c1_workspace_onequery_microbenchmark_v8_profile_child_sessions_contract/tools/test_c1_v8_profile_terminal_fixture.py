#!/usr/bin/env python3
"""CPU-only midpoint-failure terminal-state fixture for the C1-v8 profile guard.

It drives the guard's dedicated test-only dispatch after a regular prepared
manifest exists, before any GPU session is marked ready.  Thus this fixture
exercises the real shell EXIT cleanup without nvidia-smi, nsys, NVCC, or CUDA.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
import pathlib
import subprocess
import tempfile

ROOT_DEFAULT = pathlib.Path(
    "/workspace/experiments/tide_safe_c1_20260727/"
    "c1_workspace_onequery_microbenchmark_v8_profile_child_sessions_contract"
)
GUARD_NAME = "run_c1_profile_guard_v8.sh"
MANIFEST_NAME = "profile_run_manifest_v8.json"
CARD_NAME = "profile_guard_card_v8.json"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def no_bytecode(root: pathlib.Path) -> None:
    offenders = [
        path for path in root.rglob("*")
        if ".git" not in path.parts and (path.name == "__pycache__" or path.suffix == ".pyc")
    ]
    need(not offenders, "fixture wrote bytecode: " + ", ".join(str(path) for path in offenders))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    root = args.root
    need(root.is_absolute() and root.is_dir() and not root.is_symlink() and root.resolve(strict=True) == ROOT_DEFAULT, "canonical v8 root required")
    guard = root / GUARD_NAME
    need(guard.is_file() and not guard.is_symlink(), "profile guard missing")
    with tempfile.TemporaryDirectory(prefix="c1_v8_profile_terminal_fixture_") as temporary:
        out = pathlib.Path(temporary) / "c1_v8_profile_fixture_terminal"
        out.mkdir()
        (out / MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "schema": "gtspp-c1-v8-profile-run-manifest-v1",
                    "status": "PREPARED",
                    "reason": "fixture prepared manifest",
                    "profile_artifacts": [],
                },
                sort_keys=True,
            )
            + "\n"
        )
        env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C"}
        result = subprocess.run(
            ["/usr/bin/bash", str(guard), "--fixture-terminal-failure", str(out)],
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        need(result.returncode != 0, "injected midpoint failure unexpectedly succeeded")
        manifest = json.loads((out / MANIFEST_NAME).read_text())
        card = json.loads((out / CARD_NAME).read_text())
        reason_marker = "fixture injected midpoint failure"
        need(manifest.get("status") == "FAILED", "midpoint manifest must terminalize FAILED, never PREPARING")
        need(reason_marker in str(manifest.get("reason", "")), "midpoint manifest failure reason missing")
        need(card.get("status") == "FAILED", "midpoint guard card must terminalize FAILED")
        need(reason_marker in str(card.get("reason", "")), "midpoint guard card failure reason missing")
        need(card.get("execution_mode") == "PROFILE" and card.get("physical_gpu_uuid") is None, "fixture must not establish a GPU session")
        need(not (out / "gpu_snapshots").exists(), "midpoint fixture must not call GPU telemetry")
        need(not (out / ".c1_profile_guard_session_v8.json").exists(), "midpoint fixture must not create GPU session marker")
        need("nvidia-smi" not in (result.stdout + result.stderr).lower(), "midpoint fixture output indicates forbidden GPU telemetry")
    no_bytecode(root)
    print(json.dumps({"pass": True, "mode": "CPU_ONLY_PROFILE_TERMINAL_FIXTURE", "terminal_status": "FAILED"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
