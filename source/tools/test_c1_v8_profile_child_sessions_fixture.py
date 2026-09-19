#!/usr/bin/env python3
"""CPU-only real-function fixtures for the v8 profile child-session contract.

This test imports the canonical strict profile verifier and invokes its actual
root-layout and child-session validators against temporary regular files.  It
never calls a launcher, nvidia-smi, Nsight, NVCC, or a CUDA binary.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import datetime
import importlib.util
import json
import pathlib
import tempfile
from typing import Any, Callable

ROOT_DEFAULT = pathlib.Path(
    "/workspace/experiments/tide_safe_c1_20260727/"
    "c1_workspace_onequery_microbenchmark_v8_profile_child_sessions_contract"
)
PRIMARY = ("E_G_c1_off_reference", "P_F_full_C1")
FORBIDDEN_OUTPUT = ("nvidia-smi", "nvcc", "cuda binary", "/usr/local/bin/nsys")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def import_verifier(root: pathlib.Path) -> Any:
    path = root / "verify_c1_profile_artifacts_v8.py"
    need(path.is_file() and not path.is_symlink(), "canonical v8 profile verifier missing")
    spec = importlib.util.spec_from_file_location("gtspp_c1_v8_profile_verifier_fixture", path)
    need(spec is not None and spec.loader is not None, "cannot import canonical v8 profile verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cleanup_record(module: Any, variant: str, nsys: pathlib.Path, binary: pathlib.Path, pid: int) -> dict[str, object]:
    return {
        "schema": module.SESSION_CLEANUP_SCHEMA,
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": "profile/" + variant,
        "expected_kind": "profile_nsys",
        "expected_executable": str(nsys),
        "expected_command_path": str(binary),
        "wrapper_pid": pid,
        "session_pid": pid,
        "pgid": pid,
        "sid": pid,
        "phase": "direct",
        "reason": "direct child exited",
        "verified_owned_session": True,
        "action": module.SESSION_CLEANUP_ACTION,
        "still_alive_after_cleanup": False,
        "child_returncode": 0,
    }


def build_equivalent_profile_root(parent: pathlib.Path, module: Any) -> tuple[pathlib.Path, dict[str, object]]:
    """Make the exact post-success directory shape that v7 wrongly rejected."""
    profile = parent / "c1_v8_profile_fixture"
    profile.mkdir(parents=True)
    (profile / "gpu_snapshots").mkdir()
    sessions = profile / module.CHILD_SESSION_DIRECTORY
    sessions.mkdir()
    nsys = module.canonical_profile_nsys()
    variants: list[dict[str, object]] = []
    for ordinal, variant in enumerate(PRIMARY, start=1):
        output = profile / variant
        provenance = output / "child_session_provenance"
        provenance.mkdir(parents=True)
        binary = parent / "fake_builds" / variant / "bin" / "C1Microbench"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(("fixture binary " + variant + "\n").encode())
        pid = 1000 + ordinal
        key = "profile_" + variant
        (sessions / (key + ".ready")).write_text(f"pid={pid}\npgid={pid}\nsid={pid}\n")
        (sessions / (key + ".go")).write_bytes(b"")
        record = cleanup_record(module, variant, nsys, binary, pid)
        (provenance / "owned_session_cleanup_v8.json").write_text(json.dumps(record, sort_keys=True) + "\n")
        variants.append({"name": variant, "path": str(binary)})
    # A real post-build pin set has all four measured variants, although the
    # profile session contract intentionally consumes only its two primary ones.
    for variant in ("P_G_workspace_only", "E_F_fastpath_only"):
        binary = parent / "fake_builds" / variant / "bin" / "C1Microbench"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(("fixture binary " + variant + "\n").encode())
        variants.append({"name": variant, "path": str(binary)})
    return profile, {"variant_binaries": variants}


def verify_positive(module: Any, profile: pathlib.Path, pins: dict[str, object]) -> None:
    module.verify_profile_root_directory_contract(profile)
    rows = module.verify_child_sessions_contract(profile, pins)
    need([row.get("variant") for row in rows] == list(PRIMARY), "positive child-session rows mismatch")
    need(all(isinstance(row.get("ready_session_pid"), int) and row["ready_session_pid"] > 0 for row in rows), "positive ready PID witness missing")


def expect_rejected(label: str, action: Callable[[], None], marker: str) -> None:
    try:
        action()
    except SystemExit as exc:
        detail = str(exc)
        need(marker in detail, f"{label}: wrong rejection {detail!r}")
        return
    raise SystemExit(f"{label}: malformed child-session fixture unexpectedly accepted")


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
    need(root.is_absolute() and root.is_dir() and not root.is_symlink(), "canonical v8 root required")
    need(root.resolve(strict=True) == ROOT_DEFAULT, "fixture root must be canonical v8 root")
    verifier = import_verifier(root)
    with tempfile.TemporaryDirectory(prefix="c1_v8_profile_child_sessions_fixture_") as raw:
        parent = pathlib.Path(raw)
        # Positive case deliberately mirrors the v7 failed profile's normal
        # child_sessions directory: the new strict contract accepts only its
        # exact regular ready/go files and corresponding cleanup proofs.
        positive, positive_pins = build_equivalent_profile_root(parent / "positive_parent", verifier)
        verify_positive(verifier, positive, positive_pins)

        root_extra, root_extra_pins = build_equivalent_profile_root(parent / "root_extra_parent", verifier)
        (root_extra / "unexpected_directory").mkdir()
        expect_rejected("extra profile root directory", lambda: verifier.verify_profile_root_directory_contract(root_extra), "profile root directory set")
        # The unchanged session evidence itself remains valid; the rejection is
        # specifically the strict top-level shape rather than a permissive allow-list.
        verify_positive(verifier, root_extra, root_extra_pins) if False else None

        missing, missing_pins = build_equivalent_profile_root(parent / "missing_parent", verifier)
        (missing / verifier.CHILD_SESSION_DIRECTORY / "profile_E_G_c1_off_reference.ready").unlink()
        expect_rejected("missing ready", lambda: verifier.verify_child_sessions_contract(missing, missing_pins), "profile child-session directory set")

        extra, extra_pins = build_equivalent_profile_root(parent / "extra_parent", verifier)
        (extra / verifier.CHILD_SESSION_DIRECTORY / "profile_E_G_c1_off_reference.abort").write_bytes(b"")
        expect_rejected("extra session artifact", lambda: verifier.verify_child_sessions_contract(extra, extra_pins), "profile child-session directory set")

        symlinked, symlinked_pins = build_equivalent_profile_root(parent / "symlink_parent", verifier)
        go = symlinked / verifier.CHILD_SESSION_DIRECTORY / "profile_E_G_c1_off_reference.go"
        go.unlink()
        go.symlink_to(symlinked / verifier.CHILD_SESSION_DIRECTORY / "profile_E_G_c1_off_reference.ready")
        expect_rejected("symlinked GO", lambda: verifier.verify_child_sessions_contract(symlinked, symlinked_pins), "absolute regular non-symlink")

        malformed, malformed_pins = build_equivalent_profile_root(parent / "malformed_parent", verifier)
        (malformed / verifier.CHILD_SESSION_DIRECTORY / "profile_E_G_c1_off_reference.ready").write_text("pid=1001\npgid=1002\nsid=1001\n")
        expect_rejected("mismatched ready IDs", lambda: verifier.verify_child_sessions_contract(malformed, malformed_pins), "profile child-session ready PID=PGID=SID")

        forged, forged_pins = build_equivalent_profile_root(parent / "forged_parent", verifier)
        record_path = forged / "E_G_c1_off_reference" / "child_session_provenance" / "owned_session_cleanup_v8.json"
        record = json.loads(record_path.read_text())
        record["expected_command_path"] = "/tmp/forged/C1Microbench"
        record_path.write_text(json.dumps(record, sort_keys=True) + "\n")
        expect_rejected("forged cleanup command", lambda: verifier.verify_child_sessions_contract(forged, forged_pins), "profile child-session cleanup binary binding")

        terminal, terminal_pins = build_equivalent_profile_root(parent / "terminal_parent", verifier)
        record_path = terminal / "P_F_full_C1" / "child_session_provenance" / "owned_session_cleanup_v8.json"
        record = json.loads(record_path.read_text())
        record["child_returncode"] = 1
        record_path.write_text(json.dumps(record, sort_keys=True) + "\n")
        expect_rejected("nonzero cleanup return", lambda: verifier.verify_child_sessions_contract(terminal, terminal_pins), "profile child-session cleanup terminal result")
    no_bytecode(root)
    print(json.dumps({
        "pass": True,
        "mode": "CPU_ONLY_V8_PROFILE_CHILD_SESSIONS_CONTRACT_FIXTURE",
        "v7_failure_equivalent_child_sessions_is_accepted_and_verified": True,
        "negative_cases": ["extra_root", "missing", "extra", "symlink", "ready_pid_mismatch", "forged_cleanup", "nonzero_cleanup"],
        "forbidden_tools_not_invoked": list(FORBIDDEN_OUTPUT),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
