#!/usr/bin/env python3
"""Run the pre-registered CPU-only HNSW calibration sweep without test selection.

Every condition runs in a fresh Python/HNSW process.  The selection routine
reads *only* the sealed calibration partition from the independent validator
result.  Full per-condition artifacts are retained for audit but held-out
metrics are neither printed nor serialized into the calibration-selection file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "safe-c1-hnsw-calibration-sweep-v1"
EFS = (10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(value: bool, message: str) -> None:
    if not value:
        fail(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def real_root_dir(path: Path, label: str) -> Path:
    st = path.lstat()
    require(stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode), f"{label} is not a real directory")
    require(st.st_uid == 0 and (st.st_mode & 0o077) == 0, f"{label} is not root-private")
    return path.resolve(strict=True)


def regular_root_file(path: Path, label: str) -> Path:
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode), f"{label} is not a regular non-symlink")
    require(st.st_uid == 0 and (st.st_mode & 0o022) == 0, f"{label} is not root-owned/non-writable")
    return path.resolve(strict=True)


def write_new(path: Path, value: Any) -> None:
    require(not os.path.lexists(path), f"refusing to overwrite {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        blob = canonical(value)
        os.write(fd, blob)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chown(path, 0, 0)
    os.chmod(path, 0o600)


def command_env() -> dict[str, str]:
    # No user Python path/site variables; runner independently requires -I.
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/root",
        "LANG": "C",
        "LC_ALL": "C",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def run_logged(argv: list[str], env: dict[str, str], stdout_path: Path, stderr_path: Path, label: str) -> None:
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        result = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, check=False)
    os.chown(stdout_path, 0, 0); os.chmod(stdout_path, 0o600)
    os.chown(stderr_path, 0, 0); os.chmod(stderr_path, 0o600)
    require(result.returncode == 0, f"{label} failed with exit={result.returncode}; inspect {stderr_path}")


def cal_counts(validator_result: dict[str, Any]) -> dict[str, int]:
    # Deliberately the only selection input from the validator result.
    require(validator_result.get("schema") == "fair-hnsw-frozen-base-recall-output-validator-v1" and
            validator_result.get("status") == "PASS", "validator did not PASS")
    split_metrics = validator_result.get("split_metrics")
    require(isinstance(split_metrics, dict), "validator lacks split metrics")
    calibration = split_metrics.get("calibration")
    require(isinstance(calibration, dict), "validator lacks calibration metrics")
    names = ("knn_count", "k", "base_exact_match_count", "merged_exact_match_count", "base_overlap_sum", "merged_overlap_sum")
    out: dict[str, int] = {}
    for name in names:
        value = calibration.get(name)
        require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"invalid calibration field {name}")
        out[name] = value
    require(out["knn_count"] == 68 and out["k"] == 10, "calibration cardinality drift")
    return out


def split_target(split: dict[str, Any]) -> tuple[int, int, str]:
    require(split.get("schema") == "safe-c1-hnsw-calibration-test-split-v1", "unexpected split schema")
    provided = split.get("split_sha256")
    require(isinstance(provided, str) and len(provided) == 64, "split hash absent")
    unchecked = dict(split); unchecked.pop("split_sha256", None)
    require(hashlib.sha256(canonical(unchecked)).hexdigest() == provided, "split self-hash mismatch")
    policy = split.get("selection_policy")
    require(isinstance(policy, dict) and policy.get("quality_optimization") == "none", "split policy drift")
    target_root = split.get("native_gts_overlap_targets")
    require(isinstance(target_root, dict), "split has no Native-GTS target")
    calibration = target_root.get("calibration")
    require(isinstance(calibration, dict), "split has no calibration target")
    overlap = calibration.get("merged_overlap")
    exact = calibration.get("merged_exact_result_sets")
    require(isinstance(overlap, dict) and isinstance(exact, dict), "split calibration target shape")
    target_overlap, denominator = overlap.get("numerator"), overlap.get("denominator")
    target_exact, exact_denominator = exact.get("numerator"), exact.get("denominator")
    require((target_overlap, denominator, target_exact, exact_denominator) == (651, 680, 53, 68),
            "unexpected sealed Native-GTS calibration target")
    return target_overlap, target_exact, provided


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="CPU-only HNSW calibration sweep; selection reads calibration only")
    parser.add_argument("--out", required=True, type=Path, help="new root-private sweep directory")
    args = parser.parse_args()
    require(os.geteuid() == 0, "must run as root")
    root = real_root_dir(root, "experiment root")
    out = args.out.resolve(strict=False)
    require(out.is_absolute() and out.parent.resolve(strict=True) == (root / "runs").resolve(strict=True),
            "--out must be a new direct child of root/runs")
    require(not os.path.lexists(out), "sweep output already exists")
    os.mkdir(out, 0o700); os.chown(out, 0, 0); os.chmod(out, 0o700)

    runner = regular_root_file(root / "runner/fair_hnsw_frozen_base_recall_e1_runner.py", "runner")
    validator = regular_root_file(root / "tools/validate_hnsw_frozen_base_recall_output.py", "validator")
    split_path = regular_root_file(root / "inputs/hnsw_frozen_base_calibration_test_split_v1.json", "split")
    bundle = real_root_dir(root / "inputs/e1_frozen_base_knn_projection_v1", "bundle")
    admission = regular_root_file(root / "preflight/e1_frozen_base_knn_projection_v1.v2.admission.env", "admission")
    runtime = real_root_dir(root / "trusted_hnsw_venv_v1", "trusted runtime")
    runtime_python = runtime / "bin/python"
    require(runtime_python.is_file() or runtime_python.is_symlink(), "trusted runtime lacks python")
    target_overlap, target_exact, split_hash = split_target(json.loads(split_path.read_text(encoding="utf-8")))

    card = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "scope": "CPU-only/no-timing frozen immutable-base HNSW calibration; selection consumes calibration metrics only",
        "efs": list(EFS),
        "selection_policy": {
            "primary": "minimize absolute calibration merged_overlap_sum gap to sealed Native-GTS target",
            "tie_1": "smaller ef_search",
            "tie_2": "smaller absolute calibration merged_exact_match_count gap to sealed target",
            "forbidden_input": "held_out metrics are not read, printed, or serialized by the selector",
        },
        "calibration_target": {"merged_overlap_sum": target_overlap, "merged_exact_match_count": target_exact, "knn": 68, "k": 10},
        "input_sha256": {"runner": sha256_file(runner), "validator": sha256_file(validator), "split": sha256_file(split_path), "split_declared_sha256": split_hash, "admission": sha256_file(admission)},
        "runtime": {"root": str(runtime), "python": str(runtime_python)},
        "environment": command_env(),
        "gpu_used": False,
        "timing_claim": False,
    }
    write_new(out / "run_card.json", card)
    calibration_rows: list[dict[str, Any]] = []
    env = command_env()
    for ef in EFS:
        condition = out / f"ef_{ef:04d}"
        os.mkdir(condition, 0o700); os.chown(condition, 0, 0); os.chmod(condition, 0o700)
        runner_argv = [str(runtime_python), "-I", str(runner), "--bundle", str(bundle), "--preflight", str(admission),
                       "--out", str(condition / "engine.jsonl"), "--summary", str(condition / "summary.json"),
                       "--runtime-root", str(runtime), "--ef-search", str(ef), "--mode", "recall-probe"]
        write_new(condition / "runner_command.json", {"argv": runner_argv, "env": env, "gpu_used": False, "timing_claim": False})
        run_logged(runner_argv, env, condition / "runner.stdout", condition / "runner.stderr", f"runner ef={ef}")
        validator_argv = ["/usr/bin/python3.12", "-I", "-S", str(validator), "--bundle", str(bundle), "--admission", str(admission),
                          "--output", str(condition / "engine.jsonl"), "--summary", str(condition / "summary.json"), "--split", str(split_path)]
        run_logged(validator_argv, env, condition / "validator.stdout", condition / "validator.stderr", f"validator ef={ef}")
        validation = json.loads((condition / "validator.stdout").read_text(encoding="utf-8"))
        metrics = cal_counts(validation)
        require(validation.get("hnsw_ef_search") == ef, f"validator ef mismatch at {ef}")
        row = {"ef_search": ef, **metrics,
               "merged_overlap_gap_to_native_gts_calibration": abs(metrics["merged_overlap_sum"] - target_overlap),
               "merged_exact_gap_to_native_gts_calibration": abs(metrics["merged_exact_match_count"] - target_exact),
               "validator_sha256": sha256_file(condition / "validator.stdout")}
        calibration_rows.append(row)
        print("PASS calibration ef=%d merged_overlap=%d/%d merged_exact=%d/%d" %
              (ef, metrics["merged_overlap_sum"], metrics["knn_count"] * metrics["k"], metrics["merged_exact_match_count"], metrics["knn_count"]), flush=True)

    selected = min(calibration_rows, key=lambda row: (row["merged_overlap_gap_to_native_gts_calibration"], row["ef_search"], row["merged_exact_gap_to_native_gts_calibration"]))
    selection = {
        "schema": "safe-c1-hnsw-calibration-selection-v1",
        "status": "PASS_CALIBRATION_SELECTION",
        "split_declared_sha256": split_hash,
        "selection_policy": card["selection_policy"],
        "calibration_target": card["calibration_target"],
        "calibration_conditions": calibration_rows,
        "selected": selected,
        "next_required_step": "run one fresh locked-ef replay, then read/report held_out only; do not alter ef_search",
        "gpu_used": False,
        "timing_claim": False,
    }
    write_new(out / "calibration_selection.json", selection)
    completed = dict(card); completed["status"] = "PASS_CALIBRATION_SELECTION"; completed["selected_ef_search"] = selected["ef_search"]
    write_new(out / "completed_run_card.json", completed)
    print("PASS selected_ef=%d calibration_gap=%d calibration_exact_gap=%d; held_out intentionally not read" %
          (selected["ef_search"], selected["merged_overlap_gap_to_native_gts_calibration"], selected["merged_exact_gap_to_native_gts_calibration"]), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL calibration sweep: {error}", file=sys.stderr)
        raise SystemExit(2)
