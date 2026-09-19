#!/usr/bin/env python3
"""CPU-only exact contract fixture for the v7 profile guard -> manifest CLI.

This reads the real guard and manifest utility; it never starts a guard, GPU
telemetry, Nsight, NVCC, or a CUDA binary.  The complementary manifest fixture
executes the artifact CLI on synthetic files.  Together they prevent the
previous guard/CLI drift (`--repfile` omitted and an integer treated as a path).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import pathlib

ROOT_DEFAULT = pathlib.Path(
    "/workspace/experiments/tide_safe_c1_20260727/"
    "c1_workspace_onequery_microbenchmark_v7_profile_cpu_preflight"
)
GUARD_NAME = "run_c1_profile_guard_v7.sh"
MANIFEST_NAME = "c1_v7_profile_manifest.py"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def ordered(line: str, fragments: tuple[str, ...], label: str) -> None:
    cursor = -1
    for fragment in fragments:
        pos = line.find(fragment, cursor + 1)
        need(pos >= 0, f"{label}: missing fragment {fragment!r}")
        need(pos > cursor, f"{label}: ordering failure at {fragment!r}")
        cursor = pos


def no_bytecode(root: pathlib.Path) -> None:
    offenders = [
        path
        for path in root.rglob("*")
        if ".git" not in path.parts and (path.name == "__pycache__" or path.suffix == ".pyc")
    ]
    need(not offenders, "fixture wrote bytecode: " + ", ".join(str(path) for path in offenders))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    root = args.root
    need(root.is_absolute() and root.is_dir() and not root.is_symlink(), "canonical v7 root required")
    need(root.resolve(strict=True) == ROOT_DEFAULT, "fixture root must be canonical v7 root")
    guard = root / GUARD_NAME
    manifest = root / MANIFEST_NAME
    need(guard.is_file() and not guard.is_symlink(), "profile guard missing")
    need(manifest.is_file() and not manifest.is_symlink(), "profile manifest utility missing")
    guard_text = guard.read_text(encoding="utf-8")
    manifest_text = manifest.read_text(encoding="utf-8")

    # This is the actual post-Nsight binding command in the guarded execution
    # path.  It must not become an unbound ad-hoc artifact invocation.
    artifact_lines = [
        line.strip()
        for line in guard_text.splitlines()
        if '"$PYTHON" -B -I "$MANIFEST" artifacts ' in line
    ]
    need(len(artifact_lines) == 1, "exactly one guarded manifest artifacts CLI invocation required")
    artifact = artifact_lines[0]
    ordered(
        artifact,
        (
            '"$PYTHON" -B -I "$MANIFEST" artifacts',
            '--out "$PROFILE_OUT"',
            '--variant "$variant"',
            '--rep 1',
            '--repfile "$repfile"',
            '--sqlite "$sqlite"',
            '--stats-stdout "$out/nsys/stats_stdout.log"',
            '--report "$out/nsys/reports/profile_cuda_api_sum.csv"',
            '--report "$out/nsys/reports/profile_cuda_gpu_mem_time_sum.csv"',
            '--report "$out/nsys/reports/profile_cuda_gpu_mem_size_sum.csv"',
            '--report "$out/nsys/reports/profile_um_sum.csv"',
            "|| die 'profile artifact manifest binding'",
        ),
        "guard artifact CLI",
    )
    need(artifact.count("--report") == 4, "guard must bind exactly four fixed reports")
    need(artifact.count("--repfile") == 1 and artifact.count("--stats-stdout") == 1, "guard repfile/stats witness cardinality")

    need('repfile="$out/nsys/profile.nsys-rep";sqlite="$out/nsys/profile.sqlite";report_prefix="$out/nsys/reports/profile"' in guard_text, "guard fixed rep/sqlite assignment missing")
    stats_lines = [line.strip() for line in guard_text.splitlines() if '"$NSYS" stats ' in line]
    need(len(stats_lines) == 1, "exactly one fixed Nsight stats command required")
    ordered(
        stats_lines[0],
        (
            '"$NSYS" stats',
            "--force-export=true",
            "--force-overwrite=true",
            "--format csv",
            '--output "$report_prefix"',
            "--report cuda_api_sum",
            "--report cuda_gpu_mem_time_sum",
            "--report cuda_gpu_mem_size_sum",
            "--report um_sum",
            '"$repfile" >"$out/nsys/stats_stdout.log"',
        ),
        "guard Nsight stats CLI",
    )

    # Bind the guard's argv to the real parser/handler, not merely a comment.
    for token in (
        'command.add_argument("--repfile", required=True)',
        'command.add_argument("--stats-stdout", required=True)',
        'supplied = [pathlib.Path(args.repfile), pathlib.Path(args.sqlite), *[pathlib.Path(value) for value in args.report]]',
        "_require_empty_um_witness",
    ):
        need(token in manifest_text, "manifest parser/handler token missing: " + token)
    need("pathlib.Path(args.rep)" not in manifest_text, "integer replicate Path bug remains")
    no_bytecode(root)
    print(json.dumps({
        "pass": True,
        "mode": "CPU_ONLY_PROFILE_GUARD_CLI_CONTRACT_FIXTURE",
        "guard": GUARD_NAME,
        "artifact_reports": 4,
        "repfile_bound": True,
        "stats_stdout_bound": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
