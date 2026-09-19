#!/usr/bin/python3.12
"""Create an immutable held-out lock from a strict calibration-only selection.

No held-out result is opened or computed.  The lock is valid only if the
pre-registered grid and conservative two-metric calibration policy selected the
smallest qualifying EF.  The held-out seal is checked solely for its immutable
membership/provenance and explicit absence of a quality target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

SCHEMA = "safe-c1-hnsw-strict-c2-heldout-lock-v1"
SELECTION_SCHEMA = "safe-c1-hnsw-strict-c2-calibration-selection-v1"
SEAL_SCHEMA = "safe-c1-hnsw-strict-c2-partition-seal-v1"
SOURCE_FILE_SHA = "92394b5aa7cee61d6f2583b08061d30a1a287ef5776c7dbe5c9a6a79ddadd585"
SOURCE_CANONICAL_SHA = "c825f1e29f80fcb93b0d302e8a3da3273de0de578a839bd72d86eceb5b532355"
GRID = [17, 18, 19, 20, 21, 22, 23]
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


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


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
         (info.st_mode & 0o077) == 0, f"{label} must be a root-owned private file")
    return path.resolve(strict=True)


def read_object(path: Path, label: str) -> dict[str, Any]:
    root_private_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Fail(f"invalid {label} JSON: {exc}") from exc
    need(isinstance(value, dict), f"{label} must be an object")
    return value


def need_sha(value: Any, label: str) -> str:
    need(isinstance(value, str) and len(value) == 64 and
         all(ch in "0123456789abcdef" for ch in value), f"{label} must be a SHA-256")
    return value


def write_new(path: Path, payload: bytes) -> None:
    need(not os.path.lexists(path), f"refusing to overwrite existing lock: {path}")
    root_private_dir(path.parent, "lock output parent")
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


def verify_seal(path: Path, partition: str) -> tuple[str, dict[str, Any]]:
    item = read_object(path, f"{partition} seal")
    need(item.get("schema") == SEAL_SCHEMA and item.get("partition") == partition,
         f"{partition} seal schema/partition mismatch")
    sha = need_sha(item.get("seal_sha256"), f"{partition} seal SHA")
    raw = dict(item)
    raw.pop("seal_sha256", None)
    need(hashlib.sha256(canonical(raw)).hexdigest() == sha,
         f"{partition} seal self-hash mismatch")
    need(item.get("source_split_file_sha256") == SOURCE_FILE_SHA and
         item.get("source_split_canonical_sha256") == SOURCE_CANONICAL_SHA,
         f"{partition} seal official split binding mismatch")
    if partition == "held_out":
        need(item.get("held_out_quality_target_embedded") is False and
             "calibration_native_gts_target" not in item,
             "held-out seal must not embed any quality target")
    return sha, item


def verify_selection(selection: dict[str, Any], calibration_seal_sha: str) -> tuple[str, int, dict[str, str]]:
    need(selection.get("schema") == SELECTION_SCHEMA and selection.get("status") == "PASS_SELECTION",
         "calibration selection schema/status mismatch")
    raw_sha = need_sha(selection.get("selection_sha256"), "selection self SHA")
    raw = dict(selection)
    raw.pop("selection_sha256", None)
    need(hashlib.sha256(canonical(raw)).hexdigest() == raw_sha,
         "calibration selection self-hash mismatch")
    expected_keys = {"schema", "status", "scope", "selection_policy", "candidate_ef_grid",
                     "calibration_target", "calibration_seal_sha256", "manifest_sha256",
                     "metadata_sha256", "trace_sha256", "sources", "conditions",
                     "selected_ef_search", "selected_calibration_metrics",
                     "no_cross_partition_metrics", "selection_sha256_scope", "selection_sha256"}
    need(set(selection) == expected_keys, "calibration selection key set mismatch")
    need(selection.get("selection_policy") == POLICY and selection.get("candidate_ef_grid") == GRID and
         selection.get("calibration_target") == TARGET and
         selection.get("calibration_seal_sha256") == calibration_seal_sha,
         "calibration selection policy/grid/target/seal binding mismatch")
    need(selection.get("no_cross_partition_metrics") == {
        "heldout_files_opened": 0, "heldout_hnsw_query_invocations": 0,
        "heldout_oracle_evaluations": 0, "heldout_records_written": 0,
        "heldout_metric_aggregates_written": 0, "heldout_quality_target_read": False,
    }, "calibration selection does not prove no held-out access")
    conditions = selection.get("conditions")
    need(isinstance(conditions, list) and len(conditions) == len(GRID),
         "calibration selection condition count mismatch")
    by_ef: dict[int, dict[str, Any]] = {}
    for item in conditions:
        need(isinstance(item, dict) and set(item) ==
             {"ef_search", "runner_output_sha256", "runner_summary_sha256", "validator_json_sha256",
              "calibration_metrics", "no_cross_partition_metrics"},
             "calibration condition shape mismatch")
        ef = item.get("ef_search")
        need(isinstance(ef, int) and not isinstance(ef, bool) and ef in GRID and ef not in by_ef,
             "calibration condition EF mismatch")
        for field in ("runner_output_sha256", "runner_summary_sha256", "validator_json_sha256"):
            need_sha(item.get(field), f"condition {ef} {field}")
        metrics = item.get("calibration_metrics")
        need(isinstance(metrics, dict) and set(metrics) ==
             {"knn_count", "k", "merged_exact_match_count", "merged_overlap_sum"} and
             metrics["knn_count"] == 68 and metrics["k"] == 10 and
             isinstance(metrics["merged_exact_match_count"], int) and
             isinstance(metrics["merged_overlap_sum"], int),
             "calibration condition metric shape/type mismatch")
        need(item.get("no_cross_partition_metrics") == {
            "other_partition_hnsw_query_invocations": 0,
            "other_partition_oracle_evaluations": 0,
            "other_partition_records_written": 0,
            "other_partition_metric_aggregates_written": 0,
        }, "calibration condition cross-partition evidence mismatch")
        by_ef[ef] = item
    need(sorted(by_ef) == GRID, "calibration selection grid conditions mismatch")
    eligible = [ef for ef in GRID if
                by_ef[ef]["calibration_metrics"]["merged_overlap_sum"] >= TARGET["merged_overlap_sum"] and
                by_ef[ef]["calibration_metrics"]["merged_exact_match_count"] >= TARGET["merged_exact_match_count"]]
    need(eligible, "calibration selection has no qualifying conservative EF")
    selected = selection.get("selected_ef_search")
    need(selected == min(eligible), "calibration selection did not choose smallest qualifying EF")
    need(selection.get("selected_calibration_metrics") == by_ef[selected]["calibration_metrics"],
         "selected calibration metrics mismatch condition")
    sources = selection.get("sources")
    need(isinstance(sources, dict) and set(sources) ==
         {"sweep_sha256", "runner_sha256", "runner_core_sha256", "validator_sha256", "validator_core_sha256"},
         "calibration selection source binding shape mismatch")
    for key in sources:
        need_sha(sources[key], f"selection source {key}")
    for key in ("manifest_sha256", "metadata_sha256", "trace_sha256"):
        need_sha(selection.get(key), f"selection {key}")
    return raw_sha, selected, {key: selection[key] for key in ("manifest_sha256", "metadata_sha256", "trace_sha256")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lock strict-C2 held-out EF after calibration-only selection")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--calibration-selection", required=True, type=Path)
    parser.add_argument("--calibration-seal", required=True, type=Path)
    parser.add_argument("--heldout-seal", required=True, type=Path)
    parser.add_argument("--out-lock", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        root = root_private_dir(args.root, "protocol root")
        selection_path = root_private_file(args.calibration_selection, "calibration selection")
        calibration_path = root_private_file(args.calibration_seal, "calibration seal")
        heldout_path = root_private_file(args.heldout_seal, "held-out seal")
        need(selection_path.is_relative_to(root) and calibration_path.is_relative_to(root) and
             heldout_path.is_relative_to(root) and args.out_lock.parent.resolve(strict=True).is_relative_to(root),
             "lock inputs/output parent must be within protocol root")
        calibration_sha, _ = verify_seal(calibration_path, "calibration")
        heldout_sha, _ = verify_seal(heldout_path, "held_out")
        selection_sha, selected_ef, input_hashes = verify_selection(
            read_object(selection_path, "calibration selection"), calibration_sha)
        lock: dict[str, Any] = {
            "schema": SCHEMA, "status": "LOCKED",
            "scope": ("Immutable strict-C2 held-out EF lock created only from a calibration-only selection; "
                      "it contains no held-out quality metric or target."),
            "partition": "held_out", "selected_on_partition": "calibration",
            "selection_policy": POLICY, "candidate_ef_grid": GRID, "locked_ef_search": selected_ef,
            "calibration_selection_sha256": selection_sha,
            "calibration_seal_sha256": calibration_sha, "heldout_seal_sha256": heldout_sha,
            "source_split_file_sha256": SOURCE_FILE_SHA,
            "source_split_canonical_sha256": SOURCE_CANONICAL_SHA,
            **input_hashes,
            "lock_sha256_scope": "SHA-256 of canonical UTF-8 JSON for this object with lock_sha256 omitted; sort_keys=true, separators=(',', ':'), trailing LF.",
        }
        lock["lock_sha256"] = hashlib.sha256(canonical(lock)).hexdigest()
        write_new(args.out_lock, canonical(lock))
        print(json.dumps({"schema": SCHEMA, "status": "LOCKED", "out_lock": str(args.out_lock.resolve()),
                          "lock_sha256": lock["lock_sha256"], "locked_ef_search": selected_ef,
                          "heldout_quality_metrics_embedded": False},
                         sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except Fail as exc:
        print(f"strict_c2_heldout_lock fail-stop: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
