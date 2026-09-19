#!/usr/bin/env python3
"""Run the frozen end-to-end Gate-5 blocker recovery matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any


DETERMINISTIC_POINTS = (
    "before_run_stage",
    "run_temp_created",
    "run_data_fsynced",
    "run_renamed",
    "run_dir_fsynced",
    "before_gpu_upload",
    "after_gpu_upload",
    "before_manifest_commit",
    "manifest_temp_created",
    "manifest_fsynced",
    "manifest_renamed",
    "manifest_dir_fsynced",
    "after_manifest_commit",
    "before_host_publish",
    "after_host_publish",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def gpu_isolation(gpu: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "-i",
        str(gpu),
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"nvidia-smi failed: {process.stderr}")
    rows = [row.strip() for row in process.stdout.splitlines() if row.strip()]
    if rows:
        raise RuntimeError(f"GPU {gpu} is not isolated: {rows}")
    return {"gpu": gpu, "compute_processes": rows}


def run_json(command: list[str], stdout: Path, stderr: Path) -> tuple[int, Any | None]:
    process = subprocess.run(command, capture_output=True, text=True)
    stdout.write_text(process.stdout)
    stderr.write_text(process.stderr)
    value = None
    if process.returncode == 0 and process.stdout.strip():
        value = json.loads(process.stdout)
    return process.returncode, value


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gate5-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--durable-script", type=Path, required=True)
    parser.add_argument("--exact-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--random-trials", type=int, default=100)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    exact = json.loads(args.exact_result.read_text())
    expected_hashes = {
        0: int(exact["recovery_base_hash"]),
        1: int(exact["recovery_union_hash"]),
    }
    delta = args.root / "data/stage_a/delta_u64x6.bin"
    delta_sha = sha256_file(delta)
    rng = random.Random(args.seed)
    trial_points = [("deterministic", point) for point in DETERMINISTIC_POINTS]
    trial_points += [
        ("randomized", rng.choice((*DETERMINISTIC_POINTS, "none")))
        for _ in range(args.random_trials)
    ]
    records = []
    for trial, (kind, point) in enumerate(trial_points):
        trial_root = args.output / f"trial-{trial:04d}-{kind}-{point}"
        trial_root.mkdir()
        store = trial_root / "store"
        worker_result = trial_root / "worker.json"
        preflight_worker = gpu_isolation(args.gpu)
        worker_command = [
            str(args.binary),
            "--mode",
            "blocker-recovery-worker",
            "--root",
            str(args.root),
            "--gpu",
            str(args.gpu),
            "--durable-script",
            str(args.durable_script),
            "--store-root",
            str(store),
            "--crash-point",
            point,
            "--expected-hash",
            str(expected_hashes[1]),
            "--output",
            str(worker_result),
        ]
        (trial_root / "worker_command.json").write_text(
            json.dumps(worker_command, indent=2) + "\n"
        )
        worker = subprocess.run(worker_command, capture_output=True, text=True)
        (trial_root / "worker.stdout").write_text(worker.stdout)
        (trial_root / "worker.stderr").write_text(worker.stderr)
        expected_exit = 0 if point == "none" else 99

        recover_command = [
            "python3",
            str(args.durable_script),
            "recover",
            "--store",
            str(store),
        ]
        recover_code, state = run_json(
            recover_command, trial_root / "recover.stdout", trial_root / "recover.stderr"
        )
        state = state or {}
        epoch = int(state.get("epoch", -1))
        rows = int(state.get("total_rows", 0))
        recovered_payload_sha = None
        materialize_code = 0
        recovered_delta = trial_root / "recovered_delta_u64x6.bin"
        if epoch == 1:
            materialize_command = [
                "python3",
                str(args.durable_script),
                "materialize",
                "--store",
                str(store),
                "--output",
                str(recovered_delta),
            ]
            materialize_code, materialized = run_json(
                materialize_command,
                trial_root / "materialize.stdout",
                trial_root / "materialize.stderr",
            )
            if materialized:
                recovered_payload_sha = materialized.get("output_sha256")

        preflight_verify = gpu_isolation(args.gpu)
        verify_result = trial_root / "verify.json"
        verify_command = [
            str(args.binary),
            "--mode",
            "blocker-recovery-verify",
            "--root",
            str(args.root),
            "--gpu",
            str(args.gpu),
            "--expected-epoch",
            str(epoch),
            "--expected-hash",
            str(expected_hashes.get(epoch, 0)),
            "--output",
            str(verify_result),
        ]
        if epoch == 1:
            verify_command += ["--recovered-delta", str(recovered_delta)]
        (trial_root / "verify_command.json").write_text(
            json.dumps(verify_command, indent=2) + "\n"
        )
        verify = subprocess.run(verify_command, capture_output=True, text=True)
        (trial_root / "verify.stdout").write_text(verify.stdout)
        (trial_root / "verify.stderr").write_text(verify.stderr)
        verify_value = (
            json.loads(verify_result.read_text()) if verify_result.is_file() else {}
        )
        complete_epoch = (epoch == 0 and rows == 0) or (epoch == 1 and rows == 20760)
        payload_valid = epoch == 0 or (
            materialize_code == 0 and recovered_payload_sha == delta_sha
        )
        passed = (
            worker.returncode == expected_exit
            and recover_code == 0
            and complete_epoch
            and payload_valid
            and verify.returncode == 0
            and verify_value.get("B5_RECOVERY_REHYDRATE") is True
        )
        record = {
            "trial": trial,
            "kind": kind,
            "crash_point": point,
            "worker_returncode": worker.returncode,
            "expected_worker_returncode": expected_exit,
            "recovered_epoch": epoch,
            "recovered_rows": rows,
            "recovered_payload_sha256": recovered_payload_sha,
            "expected_payload_sha256": delta_sha if epoch == 1 else None,
            "query_hash": verify_value.get("query_hash"),
            "expected_query_hash": expected_hashes.get(epoch),
            "orphan_count": len(state.get("orphan_files", [])),
            "store_bytes": directory_bytes(store),
            "trial_bytes": directory_bytes(trial_root),
            "worker_preflight": preflight_worker,
            "verify_preflight": preflight_verify,
            "passed": passed,
        }
        (trial_root / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)

    point_counts: dict[str, int] = {}
    for record in records:
        point_counts[record["crash_point"]] = point_counts.get(record["crash_point"], 0) + 1
    failures = sum(not record["passed"] for record in records)
    summary = {
        "experiment_id": "tide_20260828_gate5_blocker_recovery",
        "scope": "real-delta local process crash, GPU rehydration, exact query verification",
        "seed": args.seed,
        "deterministic_trials": len(DETERMINISTIC_POINTS),
        "randomized_trials": args.random_trials,
        "total_trials": len(records),
        "point_counts": point_counts,
        "failures": failures,
        "old_epoch_recoveries": sum(record["recovered_epoch"] == 0 for record in records),
        "new_epoch_recoveries": sum(record["recovered_epoch"] == 1 for record in records),
        "max_store_bytes": max(record["store_bytes"] for record in records),
        "max_trial_bytes": max(record["trial_bytes"] for record in records),
        "delta_payload_bytes": delta.stat().st_size,
        "delta_payload_sha256": delta_sha,
        "all_deterministic_points_covered": all(
            point_counts.get(point, 0) >= 1 for point in DETERMINISTIC_POINTS
        ),
        "B5_RECOVERY_CLOSURE": failures == 0
        and args.random_trials >= 100
        and all(point_counts.get(point, 0) >= 1 for point in DETERMINISTIC_POINTS),
        "records": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["B5_RECOVERY_CLOSURE"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
