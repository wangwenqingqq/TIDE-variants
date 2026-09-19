#!/usr/bin/env python3
"""Atomic C1-v7 profile provenance manifest utility.

This utility is CPU-only.  It never launches a GPU workload, nsys, nvidia-smi,
or NVCC.  The ``artifacts`` subcommand is deliberately strict because its
output is consumed by the separately guarded, explicitly unmeasured profile
verifier.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
import pathlib
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any

MANIFEST_NAME = "profile_run_manifest_v7.json"
ROOT_DEFAULT = "/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v7_profile_manifest_repair"
PRIMARY = ("E_G_c1_off_reference", "P_F_full_C1")
REPORTS = ("cuda_api_sum", "cuda_gpu_mem_time_sum", "cuda_gpu_mem_size_sum", "um_sum")
NON_UM_REPORTS = REPORTS[:-1]
ENV_ALLOW = (
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
    "CUDA_HOME", "PATH", "LD_LIBRARY_PATH", "C1_EXECUTION_MODE",
)
UM_NO_PAGE_FAULT_MARKER = "does not contain CUDA Unified Memory CPU page faults data."


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def secure_file(value: str, label: str) -> pathlib.Path:
    raw = pathlib.Path(value)
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():
        raise SystemExit(f"{label} must be absolute regular non-symlink: {raw}")
    resolved = raw.resolve(strict=True)
    if raw != resolved:
        raise SystemExit(f"{label} noncanonical/symlink traversal: {raw} -> {resolved}")
    return raw


def secure_dir(value: str, label: str) -> pathlib.Path:
    raw = pathlib.Path(value)
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_dir():
        raise SystemExit(f"{label} must be absolute non-symlink directory: {raw}")
    resolved = raw.resolve(strict=True)
    if raw != resolved:
        raise SystemExit(f"{label} noncanonical/symlink traversal: {raw} -> {resolved}")
    return raw


def future_under(path: pathlib.Path, parent: pathlib.Path, label: str) -> pathlib.Path:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or path.resolve(strict=False) != path
        or not str(path).startswith(str(parent) + "/")
    ):
        raise SystemExit(f"{label} unsafe future path: {path}")
    return path


def load(path: pathlib.Path, label: str) -> dict[str, Any]:
    secure_file(str(path), label)
    try:
        obj = json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"invalid JSON {label}: {exc}")
    if not isinstance(obj, dict):
        raise SystemExit(f"JSON object required: {label}")
    return obj


def flatten(value: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        if set(("path", "sha256")).issubset(value):
            out.append(value)
        for child in value.values():
            flatten(child, out)
    elif isinstance(value, list):
        for child in value:
            flatten(child, out)


def atomic(path: pathlib.Path, obj: dict[str, Any]) -> None:
    if path.is_symlink():
        raise SystemExit("refuse symlink profile manifest output")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def manifest_path(out: str) -> pathlib.Path:
    directory = secure_dir(out, "profile output")
    path = directory / MANIFEST_NAME
    if path.is_symlink():
        raise SystemExit("profile manifest symlink")
    return path


def command_text(argv: list[str]) -> str:
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def root_and_out(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path]:
    root = secure_dir(args.root, "v7 root")
    out = secure_dir(args.out, "profile output")
    if root != pathlib.Path(ROOT_DEFAULT) or out.parent != root / "profiles" or not out.name.startswith("c1_v7_profile_"):
        raise SystemExit("profile root/output boundary")
    return root, out


def init(args: argparse.Namespace) -> None:
    root, out = root_and_out(args)
    pins = secure_file(args.pins, "pins")
    launcher = secure_file(args.launcher, "profile launcher")
    guard = secure_file(args.guard, "profile guard")
    if launcher != root / "launch_c1_profile_v7.sh" or guard != root / "run_c1_profile_guard_v7.sh":
        raise SystemExit("canonical profile trust path required")
    pin = load(pins, "pins")
    entries: list[dict[str, Any]] = []
    flatten(pin, entries)
    if pin.get("schema") != "gtspp-c1-v7-pins-v1" or not entries:
        raise SystemExit("invalid pins")
    artifacts: dict[str, dict[str, str]] = {}
    for entry in entries:
        item = secure_file(str(entry.get("path", "")), "pinned artifact")
        actual = sha(item)
        if actual != entry.get("sha256"):
            raise SystemExit(f"pinned artifact drift {item}")
        artifacts[str(item)] = {"path": str(item), "sha256": entry["sha256"], "actual_sha256": actual}
    obj = {
        "schema": "gtspp-c1-v7-profile-run-manifest-v1",
        "created_utc": utc(),
        "updated_utc": utc(),
        "status": "PREPARED",
        "execution_mode": "PROFILE",
        "root": str(root),
        "profile_root": str(out),
        "invocation_argv": args.argv,
        "host": {
            "nodename": os.uname().nodename,
            "sysname": os.uname().sysname,
            "release": os.uname().release,
            "machine": platform.machine(),
        },
        "environment": {key: os.environ[key] for key in ENV_ALLOW if key in os.environ},
        "toolchain": {
            "python": sys.version,
            "cmake": command_text(["/usr/bin/cmake", "--version"]),
            "nvcc": command_text(["/usr/local/cuda-13.1/bin/nvcc", "--version"]),
            "nsys": command_text(["/usr/local/bin/nsys", "--version"]),
        },
        "pins": {"path": str(pins), "sha256": sha(pins), "schema": pin.get("schema")},
        "trust_root_launcher": {"path": str(launcher), "sha256": sha(launcher)},
        "outer_profile_guard": {"path": str(guard), "sha256": sha(guard)},
        "pinned_artifacts": sorted(artifacts.values(), key=lambda row: row["path"]),
        "gpu": None,
        "events": [],
        "children": [],
        "profile_artifacts": [],
    }
    atomic(out / MANIFEST_NAME, obj)


def event(args: argparse.Namespace) -> None:
    path = manifest_path(args.out)
    obj = load(path, "profile manifest")
    obj.setdefault("events", []).append(
        {"utc": utc(), "kind": args.kind, "label": args.label, "variant": args.variant, "detail": args.detail}
    )
    obj["updated_utc"] = utc()
    atomic(path, obj)


def gpu(args: argparse.Namespace) -> None:
    if not args.uuid.startswith("GPU-"):
        raise SystemExit("invalid GPU UUID")
    path = manifest_path(args.out)
    obj = load(path, "profile manifest")
    snapshot = secure_file(args.snapshot, "profile GPU telemetry snapshot")
    out = secure_dir(args.out, "profile output")
    if not str(snapshot).startswith(str(out / "gpu_snapshots") + "/"):
        raise SystemExit("profile telemetry outside output")
    obj["gpu"] = {"physical_index": 0, "uuid": args.uuid, "telemetry_snapshot": str(snapshot)}
    obj["updated_utc"] = utc()
    atomic(path, obj)


def child(args: argparse.Namespace) -> None:
    path = manifest_path(args.out)
    obj = load(path, "profile manifest")
    root = secure_dir(obj["root"], "manifest root")
    out = secure_dir(args.out, "profile output")
    if args.variant not in PRIMARY or args.rep != 1 or not args.uuid.startswith("GPU-"):
        raise SystemExit("invalid profile child identity")
    binary = secure_file(args.binary, "profile binary")
    cache = secure_file(args.cmake_cache, "profile CMakeCache")
    variant_out = secure_dir(args.variant_out, "profile variant output")
    if (
        binary != root / "builds" / args.variant / "bin" / "C1Microbench"
        or cache != root / "builds" / args.variant / "CMakeCache.txt"
        or variant_out != out / args.variant
    ):
        raise SystemExit("profile child path boundary")
    if (
        args.base != "/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt"
        or args.trace != str(root / "inputs/sift1m_10k_442_first1024_type2_query_only.txt")
    ):
        raise SystemExit("profile child input boundary")
    prefix = future_under(variant_out / "nsys" / "profile", variant_out, "nsys output prefix")
    binary_argv = [
        str(binary), "--base", args.base, "--trace", args.trace, "--radius", "500",
        "--out", str(variant_out), "--replicate", "1", "--variant", args.variant, "--profile-only",
    ]
    nsys_argv = [
        "/usr/local/bin/nsys", "profile", "--trace=cuda", "--cuda-memory-usage=true",
        "--force-overwrite=true", "--output", str(prefix), "--", *binary_argv,
    ]
    obj.setdefault("children", []).append(
        {
            "replicate": 1,
            "variant": args.variant,
            "binary": {"path": str(binary), "sha256": sha(binary)},
            "cmake_cache": {"path": str(cache), "sha256": sha(cache)},
            "variant_output": str(variant_out),
            "binary_argv": binary_argv,
            "nsys_argv": nsys_argv,
            "env_i": True,
            "environment": {
                "CUDA_VISIBLE_DEVICES": args.uuid,
                "NVIDIA_VISIBLE_DEVICES": args.uuid,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "C1_EXECUTION_MODE": "PROFILE",
            },
        }
    )
    obj["updated_utc"] = utc()
    atomic(path, obj)


def _stats_stdout_entry(path: pathlib.Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def _require_empty_um_witness(stats_stdout: pathlib.Path, um_report: pathlib.Path) -> None:
    """Accept the sole zero-byte report only with Nsight's explicit no-UM witness."""
    text = stats_stdout.read_text(errors="strict")
    processed_marker = f"/um_sum.py] to [{um_report}]... PROCESSED (EMPTY RESULTS)"
    if processed_marker not in text or UM_NO_PAGE_FAULT_MARKER not in text:
        raise SystemExit(
            "zero-byte profile_um_sum.csv requires Nsight stats_stdout evidence "
            "of EMPTY RESULTS and no CUDA Unified Memory CPU page-fault data"
        )


def artifacts(args: argparse.Namespace) -> None:
    path = manifest_path(args.out)
    obj = load(path, "profile manifest")
    out = secure_dir(args.out, "profile output")
    # --rep is an identity integer; it is never converted to a filesystem path.
    if not isinstance(args.rep, int) or isinstance(args.rep, bool) or args.variant not in PRIMARY or args.rep != 1:
        raise SystemExit("invalid artifact identity")
    directory = out / args.variant
    expected = [
        directory / "nsys" / "profile.nsys-rep",
        directory / "nsys" / "profile.sqlite",
        *[directory / "nsys" / "reports" / ("profile_" + report + ".csv") for report in REPORTS],
    ]
    stats_stdout = directory / "nsys" / "stats_stdout.log"
    # The caller must bind the actual .nsys-rep, SQLite, and all four reports in
    # fixed order.  This eliminates the predecessor int-to-Path error and missing repfile
    # provenance hole rather than merely masking it.
    supplied = [pathlib.Path(args.repfile), pathlib.Path(args.sqlite), *[pathlib.Path(value) for value in args.report]]
    if supplied != expected:
        raise SystemExit("profile artifact layout/order differs from fixed plan")
    supplied_stats_stdout = pathlib.Path(args.stats_stdout)
    if supplied_stats_stdout != stats_stdout:
        raise SystemExit("profile stats_stdout layout differs from fixed plan")
    stats_stdout = secure_file(str(stats_stdout), "profile stats stdout")
    if stats_stdout.stat().st_size <= 0:
        raise SystemExit("empty profile stats stdout cannot witness Nsight report semantics")

    rows: list[dict[str, Any]] = []
    um_report = expected[-1]
    for item in expected:
        item = secure_file(str(item), "profile artifact")
        size = item.stat().st_size
        if item != um_report and size <= 0:
            raise SystemExit(f"empty required non-UM profile artifact {item}")
        row: dict[str, Any] = {"path": str(item), "sha256": sha(item), "bytes": size}
        if item == um_report:
            no_um_events_observed = size == 0
            if no_um_events_observed:
                _require_empty_um_witness(stats_stdout, um_report)
            else:
                if "EMPTY RESULTS" in stats_stdout.read_text(errors="strict"):
                    raise SystemExit("nonempty profile_um_sum.csv conflicts with EMPTY RESULTS witness")
            row["no_um_events_observed"] = no_um_events_observed
        rows.append(row)

    obj.setdefault("profile_artifacts", []).append(
        {
            "replicate": args.rep,
            "variant": args.variant,
            "artifacts": rows,
            "nsys_stats_stdout": _stats_stdout_entry(stats_stdout),
            "um_zero_byte_contract": {
                "report": str(um_report),
                "zero_byte_allowed_only_with": {
                    "processed_marker": f"/um_sum.py] to [{um_report}]... PROCESSED (EMPTY RESULTS)",
                    "no_page_fault_marker": UM_NO_PAGE_FAULT_MARKER,
                },
            },
        }
    )
    obj["updated_utc"] = utc()
    atomic(path, obj)


def final(args: argparse.Namespace) -> None:
    path = manifest_path(args.out)
    obj = load(path, "profile manifest")
    if not args.status or not args.reason:
        raise SystemExit("profile status/reason required")
    obj["status"] = args.status
    obj["reason"] = args.reason
    obj["updated_utc"] = utc()
    atomic(path, obj)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    command = sub.add_parser("init")
    command.add_argument("--out", required=True)
    command.add_argument("--root", required=True)
    command.add_argument("--pins", required=True)
    command.add_argument("--launcher", required=True)
    command.add_argument("--guard", required=True)
    command.add_argument("argv", nargs=argparse.REMAINDER)
    command = sub.add_parser("event")
    command.add_argument("--out", required=True)
    command.add_argument("--kind", required=True)
    command.add_argument("--label", default="")
    command.add_argument("--variant", default="")
    command.add_argument("--detail", default="")
    command = sub.add_parser("gpu")
    command.add_argument("--out", required=True)
    command.add_argument("--uuid", required=True)
    command.add_argument("--snapshot", required=True)
    command = sub.add_parser("child")
    command.add_argument("--out", required=True)
    command.add_argument("--rep", type=int, required=True)
    command.add_argument("--variant", required=True)
    command.add_argument("--binary", required=True)
    command.add_argument("--cmake-cache", required=True)
    command.add_argument("--variant-out", required=True)
    command.add_argument("--base", required=True)
    command.add_argument("--trace", required=True)
    command.add_argument("--uuid", required=True)
    command = sub.add_parser("artifacts")
    command.add_argument("--out", required=True)
    command.add_argument("--variant", required=True)
    command.add_argument("--rep", type=int, required=True)
    command.add_argument("--repfile", required=True)
    command.add_argument("--sqlite", required=True)
    command.add_argument("--stats-stdout", required=True)
    command.add_argument("--report", action="append", required=True)
    command = sub.add_parser("final")
    command.add_argument("--out", required=True)
    command.add_argument("--status", required=True)
    command.add_argument("--reason", required=True)
    args = parser.parse_args()
    {"init": init, "event": event, "gpu": gpu, "child": child, "artifacts": artifacts, "final": final}[args.cmd](args)


if __name__ == "__main__":
    main()
