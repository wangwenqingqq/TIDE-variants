#!/usr/bin/python3.12
"""Run a strictly calibration-only C2 HNSW sweep with a pre-registered gate.

This tool never opens a held-out seal, lock, output, or validator result.  Every
condition replays all trace insertions but the runner/validator evaluates only
the presealed calibration KNNs.  The only selection rule is fixed here:
smallest EF in [17..23] satisfying both calibration overlap >=651/680 and
exact-result-set count >=53/68.  No timing is measured.
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

SCHEMA = "safe-c1-hnsw-strict-c2-calibration-selection-v1"
GRID = (17, 18, 19, 20, 21, 22, 23)
POLICY = ("smallest pre-registered EF satisfying calibration merged_overlap_sum >= "
          "651/680 and merged_exact_match_count >= 53/68")
TARGET = {"merged_overlap_sum": 651, "merged_overlap_denominator": 680,
          "merged_exact_match_count": 53, "merged_exact_match_denominator": 68,
          "knn_count": 68, "k": 10}


class Fail(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise Fail(message)


def sha_file(path: Path, label: str) -> str:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise Fail(f"missing {label}: {path}") from exc
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
         f"{label} must be a real regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def root_private_dir(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise Fail(f"missing {label}: {path}") from exc
    need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_uid == 0 and
         (info.st_mode & 0o077) == 0, f"{label} must be a root-owned private directory")
    return path.resolve(strict=True)


def root_private_file(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise Fail(f"missing {label}: {path}") from exc
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_uid == 0 and
         (info.st_mode & 0o077) == 0, f"{label} must be a root-owned private regular file")
    return path.resolve(strict=True)


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
    need(not os.path.lexists(path), f"refusing to reuse existing run directory: {path}")
    root_private_dir(path.parent, "run parent")
    os.mkdir(path, 0o700)
    os.chown(path, 0, 0)
    os.chmod(path, 0o700)
    return root_private_dir(path, "new run directory")


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
    resolved = path.resolve(strict=True)
    need(resolved == Path("/usr/bin/python3.12").resolve(strict=True),
         "trusted runtime Python must resolve to approved system Python 3.12")
    return path


def cpu_env(runtime: Path | None = None) -> dict[str, str]:
    output = {
        "HOME": "/root", "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1", "PYTHONPATH": "",
        "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void",
        "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    if runtime is not None:
        output["VIRTUAL_ENV"] = str(runtime)
    return output


def run_logged(argv: list[str], env: dict[str, str], cwd: Path, stdout: Path, stderr: Path, label: str) -> None:
    need(not os.path.lexists(stdout) and not os.path.lexists(stderr),
         f"refusing to overwrite {label} logs")
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


def calibration_target(seal: Path) -> tuple[str, dict[str, int]]:
    value = read_object(seal, "calibration seal")
    target = value.get("calibration_native_gts_target")
    need(isinstance(target, dict) and set(target) ==
         {"merged_overlap_numerator", "merged_overlap_denominator",
          "merged_exact_match_numerator", "merged_exact_match_denominator", "knn_count", "k"},
         "calibration seal target shape mismatch")
    actual = {
        "merged_overlap_sum": target.get("merged_overlap_numerator"),
        "merged_overlap_denominator": target.get("merged_overlap_denominator"),
        "merged_exact_match_count": target.get("merged_exact_match_numerator"),
        "merged_exact_match_denominator": target.get("merged_exact_match_denominator"),
        "knn_count": target.get("knn_count"), "k": target.get("k"),
    }
    need(actual == TARGET, "calibration seal does not carry fixed conservative target")
    seal_sha = value.get("seal_sha256")
    need(isinstance(seal_sha, str) and len(seal_sha) == 64 and
         all(ch in "0123456789abcdef" for ch in seal_sha),
         "calibration seal self SHA missing")
    return seal_sha, TARGET


def validator_condition(result: dict[str, Any], ef: int) -> dict[str, int]:
    need(result.get("schema") == "fair-hnsw-strict-c2-partition-output-validator-v1" and
         result.get("status") == "PASS" and result.get("partition") == "calibration",
         "calibration validator result schema/status/partition mismatch")
    need(result.get("hnsw_ef_search") == ef, "validator EF does not bind its condition")
    metrics = result.get("partition_metrics")
    need(isinstance(metrics, dict) and set(metrics) ==
         {"knn_count", "k", "base_exact_match_count", "merged_exact_match_count",
          "base_overlap_sum", "merged_overlap_sum", "canonical_int64_knn_sequence_sha256"},
         "validator selected calibration metric shape mismatch")
    no_cross = result.get("no_cross_partition_metrics")
    need(no_cross == {
        "other_partition_name": "held_out", "other_partition_hnsw_query_invocations": 0,
        "other_partition_oracle_evaluations": 0, "other_partition_records_written": 0,
        "other_partition_metric_aggregates_written": 0,
        "validated_only_presealed_partition": True,
    }, "validator does not prove no cross-partition calibration metrics")
    values = {key: metrics[key] for key in
              ("knn_count", "k", "merged_exact_match_count", "merged_overlap_sum")}
    need(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values.values()),
         "validator metric value type mismatch")
    need(values["knn_count"] == 68 and values["k"] == 10 and
         values["merged_overlap_sum"] <= 680 and values["merged_exact_match_count"] <= 68,
         "validator calibration metric denominator mismatch")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict-C2 calibration-only no-timing HNSW sweep")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--calibration-seal", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        root = root_private_dir(args.root, "protocol root")
        bundle = root_private_dir(args.bundle, "immutable bundle")
        admission = root_private_file(args.admission, "admission")
        seal = root_private_file(args.calibration_seal, "calibration seal")
        runtime = root_private_dir(args.runtime_root, "trusted HNSW runtime")
        need(bundle.is_relative_to(root) and admission.is_relative_to(root) and seal.is_relative_to(root) and
             runtime.is_relative_to(root), "all strict-C2 inputs/runtime must be within protocol root")
        out_dir = make_private_dir(args.out_dir)
        runner = root_private_file(root / "runner/fair_hnsw_strict_c2_calibration_runner.py", "calibration runner")
        validator = root_private_file(root / "tools/validate_hnsw_strict_c2_calibration_output.py", "calibration validator")
        core = root_private_file(root / "runner/strict_c2_hnsw_core.py", "runner core")
        validator_core = root_private_file(root / "tools/strict_c2_pure_validator_core.py", "validator core")
        runtime_python = trusted_runtime_python(runtime)
        calibration_seal_sha, target = calibration_target(seal)
        conditions: list[dict[str, Any]] = []
        for ef in GRID:
            condition = out_dir / f"ef{ef}"
            os.mkdir(condition, 0o700)
            os.chown(condition, 0, 0)
            os.chmod(condition, 0o700)
            root_private_dir(condition, f"condition ef={ef}")
            output = condition / "selected_calibration_knn.jsonl"
            summary = condition / "runner_summary.json"
            run_logged([
                str(runtime_python), "-I", str(runner), "--bundle", str(bundle), "--preflight", str(admission),
                "--calibration-seal", str(seal), "--runtime-root", str(runtime), "--ef-search", str(ef),
                "--out", str(output), "--summary", str(summary),
            ], cpu_env(runtime), root, condition / "runner.stdout.log", condition / "runner.stderr.log",
               f"calibration runner ef={ef}")
            validator_stdout = condition / "validator.json"
            run_logged([
                "/usr/bin/python3.12", "-I", "-S", str(validator), "--bundle", str(bundle),
                "--admission", str(admission), "--calibration-seal", str(seal), "--output", str(output),
                "--summary", str(summary),
            ], cpu_env(), root, validator_stdout, condition / "validator.stderr.log",
               f"calibration validator ef={ef}")
            metrics = validator_condition(read_object(validator_stdout, f"validator ef={ef}"), ef)
            conditions.append({
                "ef_search": ef, "runner_output_sha256": sha_file(output, "runner output"),
                "runner_summary_sha256": sha_file(summary, "runner summary"),
                "validator_json_sha256": sha_file(validator_stdout, "validator JSON"),
                "calibration_metrics": metrics,
                "no_cross_partition_metrics": {
                    "other_partition_hnsw_query_invocations": 0,
                    "other_partition_oracle_evaluations": 0,
                    "other_partition_records_written": 0,
                    "other_partition_metric_aggregates_written": 0,
                },
            })
        eligible = [item for item in conditions if
                    item["calibration_metrics"]["merged_overlap_sum"] >= TARGET["merged_overlap_sum"] and
                    item["calibration_metrics"]["merged_exact_match_count"] >= TARGET["merged_exact_match_count"]]
        need(eligible, "no pre-registered EF meets both conservative calibration gates; refusing a test lock")
        selected = min(eligible, key=lambda item: int(item["ef_search"]))
        selection: dict[str, Any] = {
            "schema": SCHEMA, "status": "PASS_SELECTION",
            "scope": ("Strict C2 calibration-only no-timing selection. No held-out seal, output, metric, "
                      "or quality target was opened or used."),
            "selection_policy": POLICY, "candidate_ef_grid": list(GRID),
            "calibration_target": target, "calibration_seal_sha256": calibration_seal_sha,
            "manifest_sha256": sha_file(bundle / "manifest.json", "manifest"),
            "metadata_sha256": sha_file(bundle / "metadata.json", "metadata"),
            "trace_sha256": sha_file(bundle / "trace.e1gtrc", "trace"),
            "sources": {"sweep_sha256": sha_file(Path(__file__).resolve(), "sweep source"),
                        "runner_sha256": sha_file(runner, "runner source"),
                        "runner_core_sha256": sha_file(core, "runner core source"),
                        "validator_sha256": sha_file(validator, "validator source"),
                        "validator_core_sha256": sha_file(validator_core, "validator core source")},
            "conditions": conditions,
            "selected_ef_search": selected["ef_search"],
            "selected_calibration_metrics": selected["calibration_metrics"],
            "no_cross_partition_metrics": {
                "heldout_files_opened": 0, "heldout_hnsw_query_invocations": 0,
                "heldout_oracle_evaluations": 0, "heldout_records_written": 0,
                "heldout_metric_aggregates_written": 0, "heldout_quality_target_read": False,
            },
            "selection_sha256_scope": "SHA-256 of canonical UTF-8 JSON for this object with selection_sha256 omitted; sort_keys=true, separators=(',', ':'), trailing LF.",
        }
        selection["selection_sha256"] = hashlib.sha256(canonical(selection)).hexdigest()
        write_new(out_dir / "selection.json", canonical(selection))
        print(json.dumps({"schema": SCHEMA, "status": "PASS_SELECTION",
                          "out_dir": str(out_dir), "selection_sha256": selection["selection_sha256"],
                          "selected_ef_search": selected["ef_search"],
                          "no_cross_partition_metrics": selection["no_cross_partition_metrics"]},
                         sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except Fail as exc:
        print(f"strict_c2_calibration_sweep fail-stop: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
