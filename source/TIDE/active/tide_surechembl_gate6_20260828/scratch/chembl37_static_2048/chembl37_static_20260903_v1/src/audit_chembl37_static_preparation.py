#!/usr/bin/env python3
"""Independently audit the one-source ChEMBL 37 static preparation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np


ID_PATTERN = re.compile(r"CHEMBL([0-9]+)\Z")
WORDS = 32
ROW_WORDS = WORDS + 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def compare_component(
    union_fp: np.memmap,
    union_pc: np.memmap,
    row_by_id: np.ndarray,
    ids: np.ndarray,
    fp: np.ndarray,
    pc: np.ndarray,
    label: str,
) -> None:
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError(f"{label}: duplicate IDs")
    if len(pc) and np.any(pc[1:] < pc[:-1]):
        raise RuntimeError(f"{label}: population-count order mismatch")
    indexes = row_by_id[ids]
    if np.any(indexes < 0):
        raise RuntimeError(f"{label}: ID absent from union")
    for start in range(0, len(ids), 100_000):
        stop = min(len(ids), start + 100_000)
        selected = indexes[start:stop]
        if not np.array_equal(union_pc[selected], pc[start:stop]):
            raise RuntimeError(f"{label}: population-count content mismatch")
        if not np.array_equal(union_fp[selected], fp[start:stop]):
            raise RuntimeError(f"{label}: fingerprint content mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    begin = time.monotonic()
    manifest_path = args.manifest or args.root / "PREPARATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    expected_rows = int(manifest["rows"])

    gate0 = args.root / "data/gate0_prepared"
    stage_a = args.root / "data/stage_a"
    union_fp = np.memmap(
        gate0 / "union_fp_u64x32.bin",
        mode="r",
        dtype="<u8",
        shape=(expected_rows, WORDS),
    )
    union_ids = np.memmap(
        gate0 / "union_id_i64.bin", mode="r", dtype="<i8", shape=(expected_rows,)
    )
    union_pc = np.memmap(
        gate0 / "union_popcnt_u16.bin",
        mode="r",
        dtype="<u2",
        shape=(expected_rows,),
    )
    if len(np.unique(union_ids)) != expected_rows:
        raise RuntimeError("union contains duplicate IDs")
    if np.any(union_pc[1:] < union_pc[:-1]):
        raise RuntimeError("union population-count order mismatch")
    maximum_id = int(union_ids.max())
    if int(union_ids.min()) < 0:
        raise RuntimeError("union contains a negative numeric ID")
    row_by_id = np.full(maximum_id + 1, -1, dtype=np.int64)
    row_by_id[union_ids] = np.arange(expected_rows, dtype=np.int64)

    source_rows = 0
    source_seen = np.zeros(expected_rows, dtype=np.bool_)
    source_mismatches = 0
    source_first_mismatch: dict[str, object] | None = None
    with gzip.open(args.fps, "rt", encoding="ascii", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.rstrip("\r\n")
            if not text or text.startswith("#"):
                continue
            fingerprint_hex, identifier = text.split("\t")
            match = ID_PATTERN.fullmatch(identifier)
            if not match:
                raise RuntimeError(f"source line {line_number}: invalid identifier")
            numeric_id = int(match.group(1))
            if numeric_id > maximum_id or row_by_id[numeric_id] < 0:
                raise RuntimeError(f"source ID absent from union: {identifier}")
            row = int(row_by_id[numeric_id])
            if source_seen[row]:
                raise RuntimeError(f"source duplicate ID: {identifier}")
            fingerprint = bytes.fromhex(fingerprint_hex)
            fingerprint_integer = int.from_bytes(fingerprint, "little")
            actual_pc = (
                fingerprint_integer.bit_count()
                if hasattr(fingerprint_integer, "bit_count")
                else bin(fingerprint_integer).count("1")
            )
            same = (
                len(fingerprint) == WORDS * 8
                and union_fp[row].tobytes() == fingerprint
                and int(union_pc[row]) == actual_pc
            )
            if not same:
                source_mismatches += 1
                if source_first_mismatch is None:
                    source_first_mismatch = {
                        "line": line_number,
                        "identifier": identifier,
                        "stored_popcount": int(union_pc[row]),
                        "source_popcount": actual_pc,
                    }
            source_seen[row] = True
            source_rows += 1
            if source_rows % 250_000 == 0:
                print(json.dumps({"phase": "source-union", "rows": source_rows}), flush=True)
    if source_rows != expected_rows or not bool(np.all(source_seen)):
        raise RuntimeError("source/union row coverage mismatch")

    base_rows = int(manifest["base_rows"])
    base_fp = np.memmap(
        gate0 / "base_fp_u64x32.bin", mode="r", dtype="<u8", shape=(base_rows, WORDS)
    )
    base_ids = np.memmap(
        gate0 / "base_id_i64.bin", mode="r", dtype="<i8", shape=(base_rows,)
    )
    base_pc = np.memmap(
        gate0 / "base_popcnt_u16.bin", mode="r", dtype="<u2", shape=(base_rows,)
    )
    compare_component(union_fp, union_pc, row_by_id, base_ids, base_fp, base_pc, "base")

    delta_rows = int(manifest["delta_rows"])
    delta = np.memmap(
        stage_a / "delta_u64x34.bin",
        mode="r",
        dtype="<u8",
        shape=(delta_rows, ROW_WORDS),
    )
    delta_ids = delta[:, 0].view("<i8")
    delta_fp = delta[:, 1 : 1 + WORDS]
    delta_pc = delta[:, -1].astype("<u2")
    compare_component(
        union_fp, union_pc, row_by_id, delta_ids, delta_fp, delta_pc, "delta"
    )
    membership = np.zeros(expected_rows, dtype=np.uint8)
    membership[row_by_id[base_ids]] += 1
    membership[row_by_id[delta_ids]] += 1
    if not bool(np.all(membership == 1)):
        raise RuntimeError("base/delta do not form a disjoint union partition")

    query_rows = int(manifest["query_rows"])
    queries = np.memmap(
        stage_a / "queries_u64x34.bin",
        mode="r",
        dtype="<u8",
        shape=(query_rows, ROW_WORDS),
    )
    query_ids = queries[:, 0].view("<i8")
    delta_row_by_id = {int(value): index for index, value in enumerate(delta_ids)}
    for row, numeric_id in enumerate(query_ids):
        delta_row = delta_row_by_id.get(int(numeric_id))
        if delta_row is None or not np.array_equal(queries[row], delta[delta_row]):
            raise RuntimeError(f"query row {row} is not an exact delta row")

    artifact_hash_mismatches: list[dict[str, str]] = []
    for relative, record in manifest["artifacts"].items():
        path = args.root / relative
        actual = sha256(path)
        if actual != record["sha256"]:
            artifact_hash_mismatches.append(
                {"path": relative, "expected": record["sha256"], "actual": actual}
            )
    passed = source_mismatches == 0 and not artifact_hash_mismatches
    result = {
        "experiment_id": "tide_20260903_chembl37_static_preparation_audit_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha256": sha256(args.fps),
        "manifest_sha256": sha256(manifest_path),
        "rows": expected_rows,
        "base_rows": base_rows,
        "delta_rows": delta_rows,
        "query_rows": query_rows,
        "source_union_rows_checked": source_rows,
        "source_union_mismatches": source_mismatches,
        "source_first_mismatch": source_first_mismatch,
        "base_delta_disjoint_complete": bool(np.all(membership == 1)),
        "queries_exact_delta_rows": True,
        "artifact_hash_mismatches": artifact_hash_mismatches,
        "pass": passed,
        "elapsed_s": time.monotonic() - begin,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
