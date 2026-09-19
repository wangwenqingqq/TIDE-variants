#!/usr/bin/env python3
"""Crash-injected durable immutable-run manifest for Gate 5.

This tests local process-crash recovery. It does not emulate loss of a storage
device or distributed replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


MAGIC = b"TIDRUN5\0"
ROW = struct.Struct("<6Q")
U64 = struct.Struct("<Q")
CRASH_STEPS = (
    "run_temp_created",
    "run_data_fsynced",
    "run_renamed",
    "run_dir_fsynced",
    "manifest_temp_created",
    "manifest_fsynced",
    "manifest_renamed",
    "manifest_dir_fsynced",
)


def sha256_file(path: Path, offset: int = 0) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if offset:
            stream.seek(offset)
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def popcount64(value: int) -> int:
    # The remote experiment environment may use Python 3.8, before int.bit_count.
    return bin(value).count("1")


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_dir(path.parent)


def initial_manifest() -> dict[str, Any]:
    return {"format": "tide-gate5-manifest-v1", "epoch": 0, "runs": []}


def initialize(store: Path) -> None:
    store.mkdir(parents=True, exist_ok=True)
    manifest = store / "CURRENT.json"
    if not manifest.exists():
        atomic_json(manifest, initial_manifest())


def input_metadata(path: Path, epoch: int) -> dict[str, Any]:
    size = path.stat().st_size
    if size % ROW.size:
        raise ValueError(f"input size {size} is not divisible by {ROW.size}")
    counts = [0] * 257
    previous = 0
    rows = 0
    with path.open("rb") as stream:
        while True:
            raw = stream.read(ROW.size)
            if not raw:
                break
            _, w0, w1, w2, w3, popcount = ROW.unpack(raw)
            actual = popcount64(w0) + popcount64(w1) + popcount64(w2) + popcount64(w3)
            if popcount != actual or popcount > 256:
                raise ValueError(f"invalid popcount at row {rows}: {popcount} != {actual}")
            if rows and popcount < previous:
                raise ValueError("input run is not population-count sorted")
            previous = popcount
            counts[popcount] += 1
            rows += 1
    cumulative = [0]
    for count in counts:
        cumulative.append(cumulative[-1] + count)
    return {
        "format": "tide-gate5-run-v1",
        "epoch": epoch,
        "rows": rows,
        "payload_bytes": size,
        "payload_sha256": sha256_file(path),
        "cumulative": cumulative,
    }


def maybe_crash(selected: str | None, step: str) -> None:
    if selected == step:
        os._exit(99)


def commit(store: Path, source: Path, epoch: int, crash_after: str | None) -> dict[str, Any]:
    initialize(store)
    before = recover(store)
    if epoch <= before["epoch"]:
        raise ValueError(f"epoch {epoch} must exceed current epoch {before['epoch']}")

    header = input_metadata(source, epoch)
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    final_name = f"run-{epoch:020d}-{header['payload_sha256'][:16]}.bin"
    final_path = store / final_name
    temporary = store / (final_name + f".tmp.{os.getpid()}")

    with temporary.open("wb") as output, source.open("rb") as payload:
        output.write(MAGIC)
        output.write(U64.pack(len(header_bytes)))
        output.write(header_bytes)
        shutil.copyfileobj(payload, output, length=1 << 20)
        output.flush()
        maybe_crash(crash_after, "run_temp_created")
        os.fsync(output.fileno())
    maybe_crash(crash_after, "run_data_fsynced")

    os.replace(temporary, final_path)
    maybe_crash(crash_after, "run_renamed")
    fsync_dir(store)
    maybe_crash(crash_after, "run_dir_fsynced")

    run_record = {
        "path": final_name,
        "epoch": epoch,
        "rows": header["rows"],
        "payload_bytes": header["payload_bytes"],
        "payload_sha256": header["payload_sha256"],
        "file_sha256": sha256_file(final_path),
        "cumulative": header["cumulative"],
    }
    manifest_value = {
        "format": "tide-gate5-manifest-v1",
        "epoch": epoch,
        "runs": before["runs"] + [run_record],
    }
    manifest = store / "CURRENT.json"
    manifest_temp = store / f"CURRENT.json.tmp.{os.getpid()}"
    with manifest_temp.open("w", encoding="utf-8") as output:
        json.dump(manifest_value, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
        output.flush()
        maybe_crash(crash_after, "manifest_temp_created")
        os.fsync(output.fileno())
    maybe_crash(crash_after, "manifest_fsynced")
    os.replace(manifest_temp, manifest)
    maybe_crash(crash_after, "manifest_renamed")
    fsync_dir(store)
    maybe_crash(crash_after, "manifest_dir_fsynced")
    return recover(store)


def read_run(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"bad run magic: {path}")
        length_bytes = stream.read(U64.size)
        if len(length_bytes) != U64.size:
            raise ValueError(f"truncated run header length: {path}")
        header_length = U64.unpack(length_bytes)[0]
        header = json.loads(stream.read(header_length))
        payload_offset = len(MAGIC) + U64.size + header_length
    return header, payload_offset


def recover(store: Path) -> dict[str, Any]:
    manifest_path = store / "CURRENT.json"
    if not manifest_path.exists():
        return initial_manifest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "tide-gate5-manifest-v1":
        raise ValueError("bad manifest format")
    previous_epoch = 0
    total_rows = 0
    for run in manifest.get("runs", []):
        path = store / run["path"]
        if not path.is_file():
            raise ValueError(f"committed run missing: {path}")
        if sha256_file(path) != run["file_sha256"]:
            raise ValueError(f"run file hash mismatch: {path}")
        header, payload_offset = read_run(path)
        if header["payload_sha256"] != run["payload_sha256"]:
            raise ValueError(f"run header/manifest hash mismatch: {path}")
        if sha256_file(path, payload_offset) != run["payload_sha256"]:
            raise ValueError(f"run payload hash mismatch: {path}")
        if header["rows"] != run["rows"] or header["cumulative"] != run["cumulative"]:
            raise ValueError(f"run metadata mismatch: {path}")
        if run["epoch"] <= previous_epoch:
            raise ValueError("run epochs are not strictly increasing")
        previous_epoch = run["epoch"]
        total_rows += run["rows"]
    if manifest["epoch"] != previous_epoch and manifest["runs"]:
        raise ValueError("manifest epoch does not match last run")
    if not manifest["runs"] and manifest["epoch"] != 0:
        raise ValueError("nonzero empty manifest")
    manifest["total_rows"] = total_rows
    manifest["orphan_files"] = sorted(
        entry.name
        for entry in store.iterdir()
        if entry.name != "CURRENT.json"
        and entry.name not in {run["path"] for run in manifest["runs"]}
    )
    return manifest


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * quantile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def make_fixture(path: Path, rows: int = 257) -> None:
    with path.open("wb") as output:
        for index in range(rows):
            bit = index % 256
            words = [0, 0, 0, 0]
            words[bit // 64] = 1 << (bit % 64)
            output.write(ROW.pack(index + 1, *words, 1))


def crash_matrix(output: Path, trials: int, seed: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "fixture_u64x6.bin"
    make_fixture(fixture)
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for trial in range(trials):
        store = output / f"trial-{trial:04d}"
        initialize(store)
        crash_step = rng.choice((*CRASH_STEPS, None))
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "commit",
            "--store",
            str(store),
            "--input",
            str(fixture),
            "--epoch",
            "1",
        ]
        if crash_step:
            command += ["--crash-after", crash_step]
        process = subprocess.run(command, capture_output=True, text=True)
        state = recover(store)
        allowed_epoch = state["epoch"] in (0, 1)
        complete = state["total_rows"] in (0, 257)
        expected_exit = process.returncode == (99 if crash_step else 0)
        passed = allowed_epoch and complete and expected_exit
        records.append(
            {
                "trial": trial,
                "crash_after": crash_step,
                "returncode": process.returncode,
                "recovered_epoch": state["epoch"],
                "recovered_rows": state["total_rows"],
                "orphan_count": len(state["orphan_files"]),
                "passed": passed,
                "stderr": process.stderr[-1000:],
            }
        )
        if not passed:
            raise RuntimeError(f"crash trial failed: {records[-1]}")
    result = {
        "experiment_id": "tide_20260827_gate5_crash_recovery",
        "scope": "local process-crash injection, not storage-device failure",
        "seed": seed,
        "trials": trials,
        "crash_steps": list(CRASH_STEPS),
        "failures": sum(not row["passed"] for row in records),
        "old_epoch_recoveries": sum(row["recovered_epoch"] == 0 for row in records),
        "new_epoch_recoveries": sum(row["recovered_epoch"] == 1 for row in records),
        "records": records,
        "G5_RECOVERY": all(row["passed"] for row in records),
    }
    (output / "crash_matrix.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def bench(output: Path, source: Path, repetitions: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        store = output / f"rep-{repetition:03d}"
        initialize(store)
        started = time.perf_counter_ns()
        state = commit(store, source, 1, None)
        visible_ns = time.perf_counter_ns()
        recovered = recover(store)
        recovered_ns = time.perf_counter_ns()
        if state["epoch"] != 1 or recovered["epoch"] != 1:
            raise RuntimeError("durable visibility/recovery mismatch")
        samples.append(
            {
                "repetition": repetition,
                "persistent_to_manifest_visible_ms": (visible_ns - started) / 1e6,
                "recovery_ms": (recovered_ns - visible_ns) / 1e6,
                "rows": recovered["total_rows"],
            }
        )
    visible = [row["persistent_to_manifest_visible_ms"] for row in samples]
    recovery = [row["recovery_ms"] for row in samples]
    result = {
        "experiment_id": "tide_20260827_gate5_persistent_manifest",
        "scope": "durable local run + manifest; excludes GPU upload/publication",
        "repetitions": repetitions,
        "input": str(source),
        "input_bytes": source.stat().st_size,
        "visible_ms": {
            "min": min(visible),
            "median": statistics.median(visible),
            "p95": percentile(visible, 0.95),
            "max": max(visible),
        },
        "recovery_ms": {
            "min": min(recovery),
            "median": statistics.median(recovery),
            "p95": percentile(recovery, 0.95),
            "max": max(recovery),
        },
        "gate_ms": 1905.55,
        "G5_PERSIST_MANIFEST_PARTIAL": percentile(visible, 0.95) <= 1905.55,
        "samples": samples,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--store", type=Path, required=True)

    commit_parser = subparsers.add_parser("commit")
    commit_parser.add_argument("--store", type=Path, required=True)
    commit_parser.add_argument("--input", type=Path, required=True)
    commit_parser.add_argument("--epoch", type=int, required=True)
    commit_parser.add_argument("--crash-after", choices=CRASH_STEPS)

    recover_parser = subparsers.add_parser("recover")
    recover_parser.add_argument("--store", type=Path, required=True)

    crash_parser = subparsers.add_parser("crash-matrix")
    crash_parser.add_argument("--output", type=Path, required=True)
    crash_parser.add_argument("--trials", type=int, default=100)
    crash_parser.add_argument("--seed", type=int, default=20260827)

    bench_parser = subparsers.add_parser("bench")
    bench_parser.add_argument("--output", type=Path, required=True)
    bench_parser.add_argument("--input", type=Path, required=True)
    bench_parser.add_argument("--repetitions", type=int, default=10)

    args = parser.parse_args()
    if args.command == "init":
        initialize(args.store)
        result: Any = recover(args.store)
    elif args.command == "commit":
        result = commit(args.store, args.input, args.epoch, args.crash_after)
    elif args.command == "recover":
        result = recover(args.store)
    elif args.command == "crash-matrix":
        result = crash_matrix(args.output, args.trials, args.seed)
    elif args.command == "bench":
        result = bench(args.output, args.input, args.repetitions)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
