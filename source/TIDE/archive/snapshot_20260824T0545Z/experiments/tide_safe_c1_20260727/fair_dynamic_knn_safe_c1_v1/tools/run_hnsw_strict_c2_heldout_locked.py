#!/usr/bin/python3.12
"""Run exactly one strict-C2 held-out report after a calibration lock.

The CLI intentionally has no EF/tuning/grid/selection argument.  It opens only
the lock, held-out seal, immutable input bundle, trusted HNSW runtime, and its
new output directory.  The runner and pure validator replay all inserts but
query/oracle/write only held-out KNN ordinals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


class Fail(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise Fail(message)


def root_private_dir(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise Fail(f"missing {label}: {path}") from exc
    need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_uid == 0 and
         (info.st_mode & 0o077) == 0, f"{label} must be root-owned/private directory")
    return path.resolve(strict=True)


def root_private_file(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise Fail(f"missing {label}: {path}") from exc
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_uid == 0 and
         (info.st_mode & 0o077) == 0, f"{label} must be root-owned/private file")
    return path.resolve(strict=True)


def sha_file(path: Path, label: str) -> str:
    root_private_file(path, label)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    need(not os.path.lexists(path), f"refusing to overwrite existing output: {path}")
    root_private_dir(path.parent, "output parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            need(count > 0, f"short write to {path}")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chown(path, 0, 0)
    os.chmod(path, 0o600)


def make_private_dir(path: Path) -> Path:
    need(not os.path.lexists(path), f"refusing to reuse existing held-out run directory: {path}")
    root_private_dir(path.parent, "run parent")
    os.mkdir(path, 0o700)
    os.chown(path, 0, 0)
    os.chmod(path, 0o700)
    return root_private_dir(path, "new held-out run directory")


def trusted_runtime_python(runtime: Path) -> Path:
    path = runtime / "bin/python"
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise Fail(f"missing trusted runtime Python: {path}") from exc
    need((stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)) and info.st_uid == 0,
         "trusted runtime Python must be root-owned")
    if stat.S_ISREG(info.st_mode):
        need((info.st_mode & 0o022) == 0, "trusted runtime Python regular file must be non-writable")
    need(path.resolve(strict=True) == Path("/usr/bin/python3.12").resolve(strict=True),
         "trusted runtime Python must resolve to approved system Python 3.12")
    return path


def cpu_env(runtime: Path | None = None) -> dict[str, str]:
    result = {
        "HOME": "/root", "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1", "PYTHONPATH": "",
        "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void",
        "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    if runtime is not None:
        result["VIRTUAL_ENV"] = str(runtime)
    return result


def run_logged(argv: list[str], env: dict[str, str], cwd: Path,
               stdout: Path, stderr: Path, label: str) -> None:
    with stdout.open("xb") as out, stderr.open("xb") as err:
        result = subprocess.run(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, check=False)
    for path in (stdout, stderr):
        os.chown(path, 0, 0)
        os.chmod(path, 0o600)
        root_private_file(path, f"{label} log")
    need(result.returncode == 0, f"{label} failed exit={result.returncode}; inspect saved logs")


def read_object(path: Path, label: str) -> dict[str, Any]:
    root_private_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Fail(f"invalid {label} JSON: {exc}") from exc
    need(isinstance(value, dict), f"{label} must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Locked strict-C2 held-out no-timing HNSW report")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--heldout-seal", required=True, type=Path)
    parser.add_argument("--lock-receipt", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        root = root_private_dir(args.root, "protocol root")
        bundle = root_private_dir(args.bundle, "immutable bundle")
        admission = root_private_file(args.admission, "admission")
        heldout_seal = root_private_file(args.heldout_seal, "held-out seal")
        lock = root_private_file(args.lock_receipt, "held-out lock")
        runtime = root_private_dir(args.runtime_root, "trusted HNSW runtime")
        need(bundle.is_relative_to(root) and admission.is_relative_to(root) and
             heldout_seal.is_relative_to(root) and lock.is_relative_to(root) and
             runtime.is_relative_to(root), "all held-out run inputs must lie inside protocol root")
        out_dir = make_private_dir(args.out_dir)
        runner = root_private_file(root / "runner/fair_hnsw_strict_c2_heldout_runner.py", "held-out runner")
        validator = root_private_file(root / "tools/validate_hnsw_strict_c2_heldout_output.py", "held-out validator")
        runtime_python = trusted_runtime_python(runtime)
        output = out_dir / "selected_heldout_knn.jsonl"
        summary = out_dir / "runner_summary.json"
        run_logged([
            str(runtime_python), "-I", str(runner), "--bundle", str(bundle), "--preflight", str(admission),
            "--heldout-seal", str(heldout_seal), "--lock-receipt", str(lock),
            "--runtime-root", str(runtime), "--out", str(output), "--summary", str(summary),
        ], cpu_env(runtime), root, out_dir / "runner.stdout.log", out_dir / "runner.stderr.log", "held-out runner")
        validator_json = out_dir / "validator.json"
        run_logged([
            "/usr/bin/python3.12", "-I", "-S", str(validator), "--bundle", str(bundle),
            "--admission", str(admission), "--heldout-seal", str(heldout_seal),
            "--lock-receipt", str(lock), "--output", str(output), "--summary", str(summary),
        ], cpu_env(), root, validator_json, out_dir / "validator.stderr.log", "held-out validator")
        result = read_object(validator_json, "held-out validator result")
        need(result.get("schema") == "fair-hnsw-strict-c2-partition-output-validator-v1" and
             result.get("status") == "PASS" and result.get("partition") == "held_out",
             "held-out validator result schema/status/partition mismatch")
        no_cross = result.get("no_cross_partition_metrics")
        need(no_cross == {
            "other_partition_name": "calibration", "other_partition_hnsw_query_invocations": 0,
            "other_partition_oracle_evaluations": 0, "other_partition_records_written": 0,
            "other_partition_metric_aggregates_written": 0,
            "validated_only_presealed_partition": True,
        }, "held-out validator does not prove no calibration metric access")
        receipt: dict[str, Any] = {
            "schema": "safe-c1-hnsw-strict-c2-heldout-report-receipt-v1", "status": "PASS",
            "scope": "Locked strict-C2 held-out quality-only report; no timing or calibration metric computation.",
            "lock_sha256": result.get("heldout_lock_sha256"), "partition_seal_sha256": result.get("partition_seal_sha256"),
            "runner_output_sha256": sha_file(output, "held-out runner output"),
            "runner_summary_sha256": sha_file(summary, "held-out runner summary"),
            "validator_json_sha256": sha_file(validator_json, "held-out validator JSON"),
            "hnsw_ef_search": result.get("hnsw_ef_search"),
            "heldout_metrics": result.get("partition_metrics"),
            "no_cross_partition_metrics": no_cross,
            "receipt_sha256_scope": "SHA-256 of canonical UTF-8 JSON for this object with receipt_sha256 omitted; sort_keys=true, separators=(',', ':'), trailing LF.",
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical(receipt)).hexdigest()
        write_new(out_dir / "heldout_report_receipt.json", canonical(receipt))
        print(json.dumps({"schema": receipt["schema"], "status": "PASS", "out_dir": str(out_dir),
                          "receipt_sha256": receipt["receipt_sha256"], "hnsw_ef_search": receipt["hnsw_ef_search"],
                          "no_cross_partition_metrics": no_cross},
                         sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except Fail as exc:
        print(f"strict_c2_heldout_run fail-stop: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
