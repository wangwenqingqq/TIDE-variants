#!/usr/bin/env python3
"""Real CPU-only session/terminal-state fixtures for both C1-v7 outer guards.

Each case invokes the real guard test-only dispatch under a temporary /tmp
output root. The dispatch creates a real setsid --wait bootstrap, proves
PID=PGID=SID plus direct-command ownership, and either waits for a direct
/usr/bin/sleep or fails before GO. It never reaches nvidia-smi, nsys, NVCC,
or a CUDA binary.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import pathlib
import subprocess
import tempfile

ROOT_DEFAULT = pathlib.Path(
    "/workspace/experiments/tide_safe_c1_20260727/"
    "c1_workspace_onequery_microbenchmark_v7_profile_manifest_repair"
)
CASES = (
    {
        "name": "measured",
        "guard": "run_c1_workspace_microbenchmark_guard_v7.sh",
        "manifest": "run_manifest_v7.json",
        "card": "guard_run_card_v7.json",
        "variant": pathlib.Path("rep1") / "E_G_c1_off_reference",
        "schema": "gtspp-c1-v7-run-manifest-v2",
    },
    {
        "name": "profile",
        "guard": "run_c1_profile_guard_v7.sh",
        "manifest": "profile_run_manifest_v7.json",
        "card": "profile_guard_card_v7.json",
        "variant": pathlib.Path("E_G_c1_off_reference"),
        "schema": "gtspp-c1-v7-profile-run-manifest-v1",
    },
)
MODES = ("--fixture-owned-session", "--fixture-first-variant-midfailure")
FORBIDDEN_OUTPUT = ("nvidia-smi", "nvcc", "cuda binary", "/usr/local/bin/nsys")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def no_bytecode(root: pathlib.Path) -> None:
    offenders = [
        path for path in root.rglob("*")
        if ".git" not in path.parts and (path.name == "__pycache__" or path.suffix == ".pyc")
    ]
    need(not offenders, "fixture wrote bytecode: " + ", ".join(str(path) for path in offenders))


def run_case(root: pathlib.Path, case: dict, mode: str) -> None:
    guard = root / case["guard"]
    need(guard.is_file() and not guard.is_symlink(), f"missing regular guard: {guard}")
    with tempfile.TemporaryDirectory(prefix=f"c1_v7_{case['name']}_owned_session_", dir="/tmp") as raw:
        out = pathlib.Path(raw)
        (out / case["manifest"]).write_text(
            json.dumps(
                {
                    "schema": case["schema"],
                    "status": "PREPARED",
                    "events": [],
                    "profile_artifacts": [] if case["name"] == "profile" else None,
                },
                sort_keys=True,
            )
            + "\n"
        )
        # The exact minimal environment proves this is an explicit CPU-only
        # fixture path, not a normal GPU launcher invocation.
        result = subprocess.run(
            ["/usr/bin/bash", str(guard), mode, str(out)],
            text=True,
            capture_output=True,
            check=False,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": "/tmp",
                "LANG": "C",
                "C1_V7_CPU_ONLY_FIXTURE": "YES",
            },
        )
        combined = (result.stdout + result.stderr).lower()
        need(not any(token in combined for token in FORBIDDEN_OUTPUT), f"{case['name']} {mode}: forbidden GPU/tool output")
        manifest = load(out / case["manifest"])
        card = load(out / case["card"])
        records = list((out / case["variant"] / "child_session_provenance").glob("*.json"))
        need(len(records) == 1, f"{case['name']} {mode}: one session provenance record required")
        record = load(records[0])
        ids = [record.get(key) for key in ("wrapper_pid", "session_pid", "pgid", "sid")]
        need(all(isinstance(value, int) and value > 0 for value in ids), f"{case['name']} {mode}: integer session IDs")
        need(ids.count(ids[0]) == 4, f"{case['name']} {mode}: actual PID=PGID=SID ownership missing")
        need(record.get("verified_owned_session") is True, f"{case['name']} {mode}: ownership record not verified")
        need(record.get("still_alive_after_cleanup") is False, f"{case['name']} {mode}: child survived cleanup")
        need(not (out / "gpu_snapshots").exists(), f"{case['name']} {mode}: fixture reached GPU telemetry")
        if mode == "--fixture-owned-session":
            need(result.returncode == 0, f"{case['name']}: CPU ownership success returned {result.returncode}")
            need(manifest.get("status") == "COMPLETE" and card.get("status") == "COMPLETE", f"{case['name']}: successful terminal state")
            need(record.get("action") == "WAITED_FOR_VERIFIED_OWNED_SESSION", f"{case['name']}: success action")
        else:
            need(result.returncode != 0, f"{case['name']}: injected first-variant failure unexpectedly succeeded")
            need(manifest.get("status") == "FAILED" and card.get("status") == "FAILED", f"{case['name']}: PREPARED manifest was not terminalized FAILED")
            need("MANIFEST_READY=0" in str(manifest.get("reason", "")), f"{case['name']}: init-to-ready race witness missing")
            need(
                record.get("action") == "ABORT_PRE_ADOPTION_THEN_WAIT_OR_VERIFIED_WRAPPER_SIGNAL",
                f"{case['name']}: pre-GO abort action missing",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    root = args.root
    need(root.is_absolute() and root.is_dir() and not root.is_symlink(), "canonical v7 root required")
    need(root.resolve(strict=True) == ROOT_DEFAULT, "fixture root must be canonical v7 root")
    for case in CASES:
        for mode in MODES:
            run_case(root, case, mode)
    no_bytecode(root)
    print(
        json.dumps(
            {
                "pass": True,
                "mode": "CPU_ONLY_REAL_SESSION_OWNERSHIP_AND_MANIFEST_RACE_FIXTURE",
                "guards": [case["name"] for case in CASES],
                "modes": list(MODES),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
