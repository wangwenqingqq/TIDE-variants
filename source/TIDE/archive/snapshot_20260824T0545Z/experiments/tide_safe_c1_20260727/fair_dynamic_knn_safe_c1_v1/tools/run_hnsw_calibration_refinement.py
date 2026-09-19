#!/usr/bin/env python3
"""Root-private CPU-only HNSW calibration-refinement stage.

This is a separate stage from the immutable coarse sweep.  It runs EF values
17..23 in fresh runner processes/new HNSW indexes, runs the pure validator for
each, retains the complete per-condition artifacts, and selects only from the
validator's calibration object.

The calibration object is streamed byte-by-byte and the reader stops at its
closing brace, before the sibling partition begins.  Therefore the selector
does not consume, print, or serialize any non-calibration metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "safe-c1-hnsw-calibration-refinement-v1"
STAGE = "calibration-refinement"
EFS = (17, 18, 19, 20, 21, 22, 23)
TARGET_OVERLAP, OVERLAP_DENOMINATOR = 651, 680
TARGET_EXACT, EXACT_DENOMINATOR = 53, 68
SYSTEM_PYTHON = Path("/usr/bin/python3.12")
COARSE_V1_BASENAME = "safe-c1-hnsw-calibration-sweep-v1-20260729"
MAX_BYTES = 1 << 20


class Fail(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Fail(message)


def lst(path: Path, label: str) -> os.stat_result:
    try:
        return path.lstat()
    except FileNotFoundError as exc:
        raise Fail(f"missing {label}: {path}") from exc


def private_dir(path: Path, label: str) -> Path:
    st = lst(path, label)
    need(stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode),
         f"{label} is not a real directory")
    need(st.st_uid == 0 and st.st_gid == 0 and (st.st_mode & 0o077) == 0,
         f"{label} is not root-private")
    return path.resolve(strict=True)


def rooted_file(path: Path, label: str) -> Path:
    st = lst(path, label)
    need(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode),
         f"{label} is not a regular non-symlink")
    need(st.st_uid == 0 and st.st_gid == 0 and (st.st_mode & 0o022) == 0,
         f"{label} is not root-owned/non-writable")
    return path.resolve(strict=True)


def sha(path: Path, label: str) -> str:
    rooted_file(path, label)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def new_dir(path: Path, parent: Path, label: str) -> Path:
    need(path.is_absolute() and path.parent.resolve(strict=True) == parent,
         f"{label} must be a direct child of its approved parent")
    need(not os.path.lexists(path), f"refusing to overwrite {label}: {path}")
    try:
        os.mkdir(path, 0o700)
        os.chown(path, 0, 0)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise Fail(f"cannot create {label}: {exc}") from exc
    return private_dir(path, label)


def new_json(path: Path, parent: Path, value: Any, label: str) -> None:
    need(path.parent.resolve(strict=True) == parent and not os.path.lexists(path),
         f"refusing to overwrite/escape {label}: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise Fail(f"cannot create {label}: {exc}") from exc
    try:
        data = memoryview(dump(value))
        while data:
            written = os.write(fd, data)
            need(written > 0, f"short write to {label}")
            data = data[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chown(path, 0, 0)
    os.chmod(path, 0o600)
    rooted_file(path, label)


def env_cpu_only() -> dict[str, str]:
    return {
        "HOME": "/root", "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void",
        "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    }


def run_to_logs(argv: list[str], env: dict[str, str], cwd: Path,
                stdout: Path, stderr: Path, label: str) -> None:
    need(not os.path.lexists(stdout) and not os.path.lexists(stderr),
         f"refusing to overwrite {label} logs")
    try:
        with stdout.open("xb") as out_handle, stderr.open("xb") as err_handle:
            result = subprocess.run(argv, cwd=str(cwd), env=env,
                                    stdin=subprocess.DEVNULL,
                                    stdout=out_handle, stderr=err_handle,
                                    check=False)
    except OSError as exc:
        raise Fail(f"{label} could not start; inspect saved logs: {exc}") from exc
    for path, name in ((stdout, "stdout"), (stderr, "stderr")):
        os.chown(path, 0, 0)
        os.chmod(path, 0o600)
        rooted_file(path, f"{label} {name}")
    need(result.returncode == 0,
         f"{label} failed with exit={result.returncode}; inspect saved stderr")


def stream_calibration(path: Path, marker: bytes, label: str) -> tuple[bytes, bytes]:
    """Read only through the calibration object, never past it."""
    rooted_file(path, label)
    prefix = bytearray()
    with path.open("rb") as handle:
        while not prefix.endswith(marker):
            byte = handle.read(1)
            need(byte, f"{label} ended before calibration marker")
            prefix.extend(byte)
            need(len(prefix) <= MAX_BYTES, f"{label} prefix safety limit exceeded")
        payload = bytearray()
        depth, quoted, escaped = 0, False, False
        while True:
            byte = handle.read(1)
            need(byte, f"{label} ended inside calibration object")
            value = byte[0]
            payload.extend(byte)
            need(len(payload) <= MAX_BYTES, f"{label} object safety limit exceeded")
            if depth == 0:
                need(value == ord("{"), f"{label} marker lacks JSON object")
                depth = 1
            elif quoted:
                if escaped:
                    escaped = False
                elif value == ord("\\"):
                    escaped = True
                elif value == ord('"'):
                    quoted = False
            elif value == ord('"'):
                quoted = True
            elif value == ord("{"):
                depth += 1
            elif value == ord("}"):
                depth -= 1
                need(depth >= 0, f"{label} malformed JSON")
                if depth == 0:
                    break
    return bytes(prefix[:-len(marker)]), bytes(payload)


def as_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Fail(f"{label} calibration JSON is invalid: {exc}") from exc
    need(isinstance(value, dict), f"{label} calibration object is not an object")
    return value


def uint(value: Any, label: str) -> int:
    need(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
         f"{label} must be a nonnegative integer")
    return int(value)


def sealed_target(split: Path) -> tuple[dict[str, int], str]:
    prefix, raw = stream_calibration(
        split, b'"native_gts_overlap_targets":{"calibration":', "sealed split")
    # The split schema/self-hash are independently enforced by the pure validator.
    # Do not scan after this calibration object merely to reach a later schema key.
    item = as_object(raw, "sealed split")
    overlap, exact = item.get("merged_overlap"), item.get("merged_exact_result_sets")
    need(isinstance(overlap, dict) and isinstance(exact, dict),
         "sealed calibration target shape drift")
    target = {
        "merged_overlap_sum": uint(overlap.get("numerator"), "target overlap numerator"),
        "merged_overlap_denominator": uint(overlap.get("denominator"), "target overlap denominator"),
        "merged_exact_match_count": uint(exact.get("numerator"), "target exact numerator"),
        "merged_exact_denominator": uint(exact.get("denominator"), "target exact denominator"),
        "knn_count": uint(item.get("knn_count"), "target KNN count"),
        "k": uint(item.get("k"), "target K"),
    }
    need(tuple(target.values()) == (TARGET_OVERLAP, OVERLAP_DENOMINATOR,
                                    TARGET_EXACT, EXACT_DENOMINATOR, 68, 10),
         "sealed calibration target is not 651/680 and 53/68")
    return target, hashlib.sha256(raw).hexdigest()


def validator_calibration(stdout: Path, expected_ef: int) -> tuple[dict[str, int], str]:
    """Extract exactly the selection object; close file before sibling content."""
    prefix, raw = stream_calibration(
        stdout, b'"split_metrics":{"calibration":', "validator stdout")
    matches = re.findall(rb'"hnsw_ef_search":([0-9]+)', prefix)
    need(len(matches) == 1 and int(matches[0]) == expected_ef,
         "validator does not bind its EF to this condition")
    item = as_object(raw, "validator stdout")
    metrics = {key: uint(item.get(key), f"validator calibration {key}")
               for key in ("knn_count", "k", "merged_exact_match_count",
                           "merged_overlap_sum")}
    need((metrics["knn_count"], metrics["k"]) == (68, 10),
         "validator calibration cardinality drift")
    need(metrics["merged_overlap_sum"] <= OVERLAP_DENOMINATOR and
         metrics["merged_exact_match_count"] <= EXACT_DENOMINATOR,
         "validator calibration metric exceeds protocol denominator")
    return metrics, hashlib.sha256(raw).hexdigest()


def runtime_python(runtime: Path) -> Path:
    python = runtime / "bin/python"
    st = lst(python, "runtime python")
    need(st.st_uid == 0 and st.st_gid == 0 and
         (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)),
         "runtime python is not root-owned")
    system = rooted_file(SYSTEM_PYTHON, "system Python")
    need(python.resolve(strict=True) == system,
         "runtime python must resolve to the approved system Python")
    return python


def contract(root: Path) -> dict[str, Any]:
    private_dir(root / "tools", "tools")
    private_dir(root / "runner", "runner directory")
    private_dir(root / "inputs", "inputs")
    runs = private_dir(root / "runs", "runs")
    harness = rooted_file(Path(__file__).resolve(), "refinement harness")
    runner = rooted_file(root / "runner/fair_hnsw_frozen_base_recall_e1_runner.py", "runner")
    validator = rooted_file(root / "tools/validate_hnsw_frozen_base_recall_output.py", "validator")
    split = rooted_file(root / "inputs/hnsw_frozen_base_calibration_test_split_v1.json", "sealed split")
    admission = rooted_file(root / "preflight/e1_frozen_base_knn_projection_v1.v2.admission.env", "admission")
    bundle = private_dir(root / "inputs/e1_frozen_base_knn_projection_v1", "projection bundle")
    manifest = rooted_file(bundle / "manifest.json", "manifest")
    metadata = rooted_file(bundle / "metadata.json", "metadata")
    trace = rooted_file(bundle / "trace.e1gtrc", "trace")
    runtime = private_dir(root / "trusted_hnsw_venv_v1", "trusted HNSW runtime")
    py = runtime_python(runtime)
    target, target_sha = sealed_target(split)
    validator_source = validator.read_bytes()
    need(b'json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)' in
         validator_source, "validator canonical-output contract drift")
    hashes = {
        "harness": sha(harness, "refinement harness"),
        "runner": sha(runner, "runner"),
        "validator": sha(validator, "validator"),
        "admission": sha(admission, "admission"),
        "manifest": sha(manifest, "manifest"),
        "metadata": sha(metadata, "metadata"),
        "trace": sha(trace, "trace"),
        "sealed_calibration_target_fragment": target_sha,
    }
    return {"runs": runs, "runner": runner, "validator": validator, "split": split,
            "admission": admission, "bundle": bundle, "runtime": runtime, "python": py,
            "target": target, "target_sha": target_sha, "hashes": hashes}


def assert_static(root: Path, original: dict[str, Any]) -> None:
    current = contract(root)
    need(current["hashes"] == original["hashes"] and current["target"] == original["target"],
         "static source/input/calibration target drift during sweep")


def condition_card(ef: int, c: dict[str, Any], directory: Path,
                   env: dict[str, str]) -> dict[str, Any]:
    runner = [str(c["python"]), "-I", str(c["runner"]), "--bundle", str(c["bundle"]),
              "--preflight", str(c["admission"]), "--out", str(directory / "engine.jsonl"),
              "--summary", str(directory / "summary.json"), "--runtime-root", str(c["runtime"]),
              "--ef-search", str(ef), "--mode", "recall-probe"]
    validator = [str(SYSTEM_PYTHON), "-I", "-S", str(c["validator"]),
                 "--bundle", str(c["bundle"]), "--admission", str(c["admission"]),
                 "--output", str(directory / "engine.jsonl"),
                 "--summary", str(directory / "summary.json"), "--split", str(c["split"])]
    return {
        "schema": "safe-c1-hnsw-calibration-refinement-condition-v1",
        "stage": STAGE, "ef_search": ef, "fresh_process": True, "fresh_hnsw_index": True,
        "scope": "CPU-only/no-timing frozen immutable-base HNSW recall probe plus pure validation",
        "gpu_used": False, "timing_claim": False, "environment": env,
        "commands": {"runner": runner, "validator": validator}, "input_sha256": c["hashes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only HNSW calibration-refinement, selected from validator calibration only.")
    parser.add_argument("--stage", required=True, choices=(STAGE,))
    parser.add_argument("--out", required=True, type=Path,
                        help="new direct child of the root-private runs directory")
    args = parser.parse_args()
    need(os.geteuid() == 0, "must execute as root")
    root = private_dir(Path(__file__).resolve().parents[1], "experiment root")
    c = contract(root)
    out = args.out.resolve(strict=False)
    need(out.is_absolute() and out.parent.resolve(strict=True) == c["runs"],
         "--out must be a direct child of root/runs")
    need(out.name != COARSE_V1_BASENAME,
         "coarse v1 is immutable; refinement needs a new output directory")
    run = new_dir(out, c["runs"], "refinement output directory")
    env = env_cpu_only()
    policy = {
        "primary": "min abs(calibration merged_overlap_sum - 651)",
        "tie_1": "lower ef_search",
        "tie_2": "min abs(calibration merged_exact_match_count - 53)",
        "selection_input": "only the pure validator calibration object",
    }
    card = {
        "schema": SCHEMA, "stage": STAGE, "status": "RUNNING", "efs": list(EFS),
        "scope": "CPU-only/no-timing immutable-base calibration refinement; no performance claim",
        "calibration_target": c["target"], "calibration_target_fragment_sha256": c["target_sha"],
        "selection_policy": policy, "input_sha256": c["hashes"],
        "paths": {"runner": str(c["runner"]), "validator": str(c["validator"]),
                  "sealed_split": str(c["split"]), "admission": str(c["admission"]),
                  "bundle": str(c["bundle"]), "runtime": str(c["runtime"]),
                  "runtime_python": str(c["python"])},
        "environment": env, "fresh_process_per_ef": True, "fresh_hnsw_index_per_ef": True,
        "gpu_used": False, "timing_claim": False, "elapsed_time_recorded": False,
    }
    new_json(run / "run_card.json", run, card, "run card")
    rows: list[dict[str, Any]] = []
    for ef in EFS:
        assert_static(root, c)
        condition = new_dir(run / f"ef_{ef:04d}", run, f"EF={ef} condition directory")
        item_card = condition_card(ef, c, condition, env)
        new_json(condition / "condition_card.json", condition, item_card, "condition card")
        commands = item_card["commands"]
        run_to_logs(commands["runner"], env, root, condition / "runner.stdout",
                    condition / "runner.stderr", f"runner EF={ef}")
        rooted_file(condition / "engine.jsonl", f"runner JSONL EF={ef}")
        rooted_file(condition / "summary.json", f"runner summary EF={ef}")
        run_to_logs(commands["validator"], env, root, condition / "validator.stdout",
                    condition / "validator.stderr", f"validator EF={ef}")
        metrics, fragment_sha = validator_calibration(condition / "validator.stdout", ef)
        row = {
            "ef_search": ef, **metrics,
            "merged_overlap_gap_to_target": abs(metrics["merged_overlap_sum"] - TARGET_OVERLAP),
            "merged_exact_gap_to_target": abs(metrics["merged_exact_match_count"] - TARGET_EXACT),
            "validator_calibration_fragment_sha256": fragment_sha, "validator_exit_code": 0,
        }
        new_json(condition / "calibration_metrics.json", condition, row, "calibration metrics")
        rows.append(row)
        print("PASS calibration-refinement ef=%d merged_overlap=%d/%d merged_exact=%d/%d" %
              (ef, metrics["merged_overlap_sum"], OVERLAP_DENOMINATOR,
               metrics["merged_exact_match_count"], EXACT_DENOMINATOR), flush=True)
    selected = min(rows, key=lambda row: (
        row["merged_overlap_gap_to_target"], row["ef_search"], row["merged_exact_gap_to_target"]))
    selection = {
        "schema": "safe-c1-hnsw-calibration-refinement-selection-v1",
        "stage": STAGE, "status": "PASS_CALIBRATION_SELECTION", "efs": list(EFS),
        "scope": "calibration-only fixed-EF selection; no performance claim",
        "calibration_target": c["target"], "calibration_target_fragment_sha256": c["target_sha"],
        "selection_policy": policy, "calibration_conditions": rows, "selected": selected,
        "next_required_step": "lock selected EF in one fresh replay before any separately controlled test report",
        "gpu_used": False, "timing_claim": False, "elapsed_time_recorded": False,
    }
    new_json(run / "calibration_selection.json", run, selection, "calibration selection")
    final_card = dict(card)
    final_card.update({"status": "PASS_CALIBRATION_SELECTION", "selected_ef_search": selected["ef_search"]})
    new_json(run / "completed_run_card.json", run, final_card, "completed run card")
    print("PASS selected_ef=%d calibration_overlap_gap=%d calibration_exact_gap=%d" %
          (selected["ef_search"], selected["merged_overlap_gap_to_target"],
           selected["merged_exact_gap_to_target"]), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Fail as exc:
        print(f"FAIL calibration-refinement: {exc}", file=sys.stderr)
        raise SystemExit(2)
