#!/usr/bin/env python3
"""Assemble the amended Gate-5 sanitizer gate without rewriting raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def error_summary(path: Path) -> int:
    match = re.search(r"ERROR SUMMARY: (\d+) errors", path.read_text())
    if not match:
        raise RuntimeError(f"missing Compute Sanitizer error summary: {path}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    root = args.root
    raw = root / "results" / "raw"
    original = raw / "20260827T160500Z_sanitizer_concurrent_gpu2_v1"
    static_dir = raw / "20260828T132100Z_static_racecheck_applicability_v1"
    bounded_dir = raw / "20260828T132000Z_bounded_racecheck_gpu2_v1"
    tsan_dir = raw / "20260828T131700Z_actual_program_tsan_gpu2_v2"

    tools = {}
    for tool in ("memcheck", "synccheck", "initcheck"):
        result_path = original / f"{tool}_result.json"
        stdout_path = original / f"{tool}.stdout"
        result = load(result_path)
        errors = error_summary(stdout_path)
        tools[tool] = {
            "errors": errors,
            "functional_result_pass": result.get(
                "G5_SYNTHETIC_CONCURRENT", False
            ),
            "result": str(result_path),
            "stdout": str(stdout_path),
            "pass": errors == 0
            and result.get("G5_SYNTHETIC_CONCURRENT", False),
        }

    intervention = load(original / "racecheck_intervention.json")
    static = load(static_dir / "summary.json")
    bounded = load(bounded_dir / "summary_v2.json")
    tsan = load(tsan_dir / "summary.json")
    tsan_result = load(tsan_dir / "result.json")

    replacement_pass = all(item["pass"] for item in tools.values())
    replacement_pass &= static["G5_RACECHECK_STATIC_NA"]
    replacement_pass &= bounded["G5_BOUNDED_RACECHECK_DIAGNOSTIC"]
    replacement_pass &= bounded["not_host_or_global_race_proof"]
    replacement_pass &= tsan["G5_HOST_TSAN"]
    replacement_pass &= tsan_result["G5_SYNTHETIC_CONCURRENT"]
    replacement_pass &= intervention["status"] == "aborted_tool_pathology"
    replacement_pass &= intervention["not_a_pass"]

    result = {
        "experiment_id": "tide_20260828_gate5_sanitizer_amendment",
        "protocol": str(root / "PROTOCOL_AMENDMENT_20260828.md"),
        "device_tools": tools,
        "full_racecheck": {
            "status": intervention["status"],
            "formal_gate_result": intervention["formal_gate_result"],
            "not_a_pass": intervention["not_a_pass"],
            "not_a_correctness_failure": intervention[
                "not_a_correctness_failure"
            ],
            "raw_evidence": str(original),
        },
        "static_applicability": static,
        "bounded_racecheck_diagnostic": bounded,
        "actual_program_host_tsan": tsan,
        "functional_stress_inputs": {
            "concurrent_publication": str(
                raw
                / "20260827T151000Z_concurrent_snapshot_gpu2_v1/result.json"
            ),
            "snapshot_stress": str(
                raw
                / "20260827T151500Z_snapshot_stress_gpu2_v1/result.json"
            ),
        },
        "interpretation": (
            "Racecheck was statically inapplicable to the actual concurrency "
            "mechanism because the only CUDA kernel has zero shared memory and "
            "no barrier. The one-launch run is diagnostic only. Host "
            "ThreadSanitizer covers atomic snapshot publication in the actual "
            "program; Memcheck and Initcheck cover device access and "
            "initialization."
        ),
        "G5_SANITIZER": bool(replacement_pass),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    inputs = [
        root / "PROTOCOL_AMENDMENT_20260828.md",
        original / "racecheck_intervention.json",
        static_dir / "summary.json",
        bounded_dir / "summary_v2.json",
        tsan_dir / "summary.json",
        tsan_dir / "result.json",
    ]
    for tool in ("memcheck", "synccheck", "initcheck"):
        inputs.extend(
            [original / f"{tool}.stdout", original / f"{tool}_result.json"]
        )
    checksums = args.output_dir / "inputs.sha256"
    checksums.write_text(
        "".join(f"{sha256(path)}  {path}\n" for path in inputs)
    )
    print(json.dumps(result, indent=2))
    return 0 if result["G5_SANITIZER"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
