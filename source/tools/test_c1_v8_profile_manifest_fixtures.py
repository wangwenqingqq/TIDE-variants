#!/usr/bin/env python3
"""CPU-only executable fixtures for the C1-v8 profile artifact manifest.

The fixture invokes the real ``artifacts`` CLI, never ``init``.  Therefore it
cannot query GPU state, launch nsys/NVCC, or execute any CUDA binary.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import pathlib
import subprocess
import tempfile
from typing import Iterable

ROOT_DEFAULT = pathlib.Path(
    "/workspace/experiments/tide_safe_c1_20260727/"
    "c1_workspace_onequery_microbenchmark_v8_profile_child_sessions_contract"
)
MANIFEST_NAME = "profile_run_manifest_v8.json"
UTILITY_NAME = "c1_v8_profile_manifest.py"
PRIMARY = ("E_G_c1_off_reference", "P_F_full_C1")
REPORTS = ("cuda_api_sum", "cuda_gpu_mem_time_sum", "cuda_gpu_mem_size_sum", "um_sum")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def secure_root(value: pathlib.Path) -> pathlib.Path:
    need(value.is_absolute() and value.is_dir() and not value.is_symlink(), "canonical fixture root required")
    resolved = value.resolve(strict=True)
    need(resolved == value == ROOT_DEFAULT, "fixture root must be canonical v8 root")
    utility = value / UTILITY_NAME
    need(utility.is_file() and not utility.is_symlink(), "manifest utility missing")
    return value


def write(path: pathlib.Path, content: bytes) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def make_output(parent: pathlib.Path, variant: str, tag: str, *, empty_um: bool, valid_empty_um_witness: bool, empty_non_um: bool = False) -> tuple[pathlib.Path, list[pathlib.Path], pathlib.Path]:
    out = parent / ("c1_v8_profile_fixture_" + variant + "_" + tag)
    out.mkdir()
    (out / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema": "gtspp-c1-v8-profile-run-manifest-v1",
                "status": "PREPARED",
                "profile_artifacts": [],
            },
            sort_keys=True,
        )
        + "\n"
    )
    nsys = out / variant / "nsys"
    report_dir = nsys / "reports"
    repfile = write(nsys / "profile.nsys-rep", b"fixture nsys rep\n")
    sqlite = write(nsys / "profile.sqlite", b"SQLite format 3\x00fixture")
    reports: list[pathlib.Path] = []
    for report in REPORTS:
        path = report_dir / ("profile_" + report + ".csv")
        if report == "um_sum" and empty_um:
            content = b""
        elif report == "cuda_api_sum" and empty_non_um:
            content = b""
        else:
            content = b"header,value\nfixture,1\n"
        reports.append(write(path, content))
    um_report = reports[-1]
    if empty_um and valid_empty_um_witness:
        witness = (
            f"Processing [fixture] with [/opt/nvidia/nsight/reports/um_sum.py] to [{um_report}]... PROCESSED (EMPTY RESULTS)\n"
            "SKIPPED: fixture does not contain CUDA Unified Memory CPU page faults data.\n"
        )
    elif empty_um:
        witness = "Processing um_sum report without the required EMPTY RESULTS witness\n"
    else:
        witness = "Processing nonempty um_sum report... PROCESSED\n"
    stats_stdout = write(nsys / "stats_stdout.log", witness.encode())
    return out, [repfile, sqlite, *reports], stats_stdout


def command(utility: pathlib.Path, out: pathlib.Path, variant: str, artifacts: list[pathlib.Path], stats_stdout: pathlib.Path, reports: Iterable[pathlib.Path] | None = None) -> list[str]:
    ordered_reports = list(reports) if reports is not None else artifacts[2:]
    argv = [
        sys.executable,
        "-B",
        "-I",
        str(utility),
        "artifacts",
        "--out",
        str(out),
        "--variant",
        variant,
        "--rep",
        "1",
        "--repfile",
        str(artifacts[0]),
        "--sqlite",
        str(artifacts[1]),
        "--stats-stdout",
        str(stats_stdout),
    ]
    for report in ordered_reports:
        argv.extend(("--report", str(report)))
    return argv


def invoke_expect(utility: pathlib.Path, out: pathlib.Path, variant: str, artifacts: list[pathlib.Path], stats_stdout: pathlib.Path, *, reports: Iterable[pathlib.Path] | None = None, success: bool, marker: str = "") -> subprocess.CompletedProcess[str]:
    argv = command(utility, out, variant, artifacts, stats_stdout, reports)
    # Explicitly bind all input paths and prove the test sends a textual integer;
    # argparse must parse it as an integer before the real artifact handler runs.
    expected_prefix = [
        sys.executable, "-B", "-I", str(utility), "artifacts", "--out", str(out),
        "--variant", variant, "--rep", "1", "--repfile", str(artifacts[0]),
        "--sqlite", str(artifacts[1]), "--stats-stdout", str(stats_stdout),
    ]
    need(argv[: len(expected_prefix)] == expected_prefix, "fixture argv prefix/order drift")
    need(argv.count("--report") == 4, "fixture must bind exactly four reports")
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    if success:
        need(result.returncode == 0, "expected artifact manifest success: " + result.stderr)
    else:
        need(result.returncode != 0, "malformed artifact invocation unexpectedly succeeded")
        need(marker in (result.stderr + result.stdout), "negative fixture did not fail for expected reason: " + marker)
    return result


def read_record(out: pathlib.Path, variant: str, expected_paths: list[pathlib.Path], stats_stdout: pathlib.Path, expected_zero_um: bool) -> None:
    obj = json.loads((out / MANIFEST_NAME).read_text())
    records = obj.get("profile_artifacts")
    need(isinstance(records, list) and len(records) == 1, "exactly one artifact record required")
    record = records[0]
    need(record.get("replicate") == 1 and type(record.get("replicate")) is int, "replicate must remain integer identity")
    need(record.get("variant") == variant, "variant record mismatch")
    rows = record.get("artifacts")
    need(isinstance(rows, list) and [row.get("path") for row in rows] == [str(path) for path in expected_paths], "artifact output layout/order mismatch")
    need([row.get("bytes") for row in rows] == [path.stat().st_size for path in expected_paths], "artifact byte counts mismatch")
    for row, path in zip(rows, expected_paths):
        need(isinstance(row.get("sha256"), str) and len(row["sha256"]) == 64, "artifact hash missing")
        if path == expected_paths[-1]:
            need(row.get("no_um_events_observed") is expected_zero_um, "UM zero-event contract mismatch")
        else:
            need("no_um_events_observed" not in row, "non-UM artifact must not claim UM state")
    stats = record.get("nsys_stats_stdout")
    need(
        isinstance(stats, dict)
        and stats.get("path") == str(stats_stdout)
        and stats.get("bytes") == stats_stdout.stat().st_size
        and isinstance(stats.get("sha256"), str)
        and len(stats["sha256"]) == 64,
        "stats_stdout provenance missing",
    )
    contract = record.get("um_zero_byte_contract")
    need(isinstance(contract, dict) and contract.get("report") == str(expected_paths[-1]), "UM contract report binding missing")


def assert_no_bytecode(root: pathlib.Path) -> None:
    offenders = [
        path for path in root.rglob("*")
        if ".git" not in path.parts and (path.name == "__pycache__" or path.suffix == ".pyc")
    ]
    need(not offenders, "fixture wrote Python bytecode: " + ", ".join(str(path) for path in offenders))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    root = secure_root(args.root)
    utility = root / UTILITY_NAME
    with tempfile.TemporaryDirectory(prefix="c1_v8_profile_manifest_fixture_") as temporary:
        parent = pathlib.Path(temporary)
        # Positive E_G covers the observed real-world zero-byte UM report, but only
        # with the exact Nsight EMPTY RESULTS/no-page-fault witness.
        eg_out, eg_artifacts, eg_stdout = make_output(parent, PRIMARY[0], "positive_zero_um", empty_um=True, valid_empty_um_witness=True)
        invoke_expect(utility, eg_out, PRIMARY[0], eg_artifacts, eg_stdout, success=True)
        read_record(eg_out, PRIMARY[0], eg_artifacts, eg_stdout, expected_zero_um=True)
        # Positive P_F covers the normal nonempty UM-report path and the second
        # primary variant under the same int-replicate/fixed-argv contract.
        pf_out, pf_artifacts, pf_stdout = make_output(parent, PRIMARY[1], "positive_nonempty_um", empty_um=False, valid_empty_um_witness=False)
        invoke_expect(utility, pf_out, PRIMARY[1], pf_artifacts, pf_stdout, success=True)
        read_record(pf_out, PRIMARY[1], pf_artifacts, pf_stdout, expected_zero_um=False)
        # Empty cuda_api_sum must never inherit the UM exception.
        non_um_out, non_um_artifacts, non_um_stdout = make_output(parent, PRIMARY[0], "negative_empty_non_um", empty_um=False, valid_empty_um_witness=False, empty_non_um=True)
        invoke_expect(utility, non_um_out, PRIMARY[0], non_um_artifacts, non_um_stdout, success=False, marker="empty required non-UM")
        need(json.loads((non_um_out / MANIFEST_NAME).read_text()).get("profile_artifacts") == [], "failed non-UM fixture mutated manifest")
        # A zero UM file without Nsight's two-part witness is rejected.
        bad_um_out, bad_um_artifacts, bad_um_stdout = make_output(parent, PRIMARY[1], "negative_missing_um_witness", empty_um=True, valid_empty_um_witness=False)
        invoke_expect(utility, bad_um_out, PRIMARY[1], bad_um_artifacts, bad_um_stdout, success=False, marker="zero-byte profile_um_sum.csv requires")
        need(json.loads((bad_um_out / MANIFEST_NAME).read_text()).get("profile_artifacts") == [], "failed UM witness fixture mutated manifest")
        # Report reordering is a provenance failure even when files are otherwise valid.
        order_out, order_artifacts, order_stdout = make_output(parent, PRIMARY[0], "negative_order", empty_um=False, valid_empty_um_witness=False)
        invoke_expect(utility, order_out, PRIMARY[0], order_artifacts, order_stdout, reports=list(reversed(order_artifacts[2:])), success=False, marker="layout/order")
        need(json.loads((order_out / MANIFEST_NAME).read_text()).get("profile_artifacts") == [], "failed ordering fixture mutated manifest")
    assert_no_bytecode(root)
    print(json.dumps({"pass": True, "mode": "CPU_ONLY_PROFILE_MANIFEST_FIXTURES", "variants": list(PRIMARY), "zero_um_contract": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
