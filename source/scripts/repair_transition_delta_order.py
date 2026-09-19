#!/usr/bin/env python3
"""Atomically repair Gate-6 delta row order without changing membership."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def multiset_digest(rows: np.ndarray) -> str:
    row_digests = np.empty(len(rows), dtype="S32")
    raw = memoryview(np.ascontiguousarray(rows)).cast("B")
    row_bytes = rows.shape[1] * rows.dtype.itemsize
    for row in range(len(rows)):
        start = row * row_bytes
        row_digests[row] = hashlib.sha256(raw[start : start + row_bytes]).digest()
    row_digests.sort()
    return hashlib.sha256(memoryview(row_digests).cast("B")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = json.loads((root / "results" / "TRANSITION_REALITY.json").read_text())
    lut = np.fromiter((value.bit_count() for value in range(256)), dtype=np.uint8, count=256)
    results = []
    for transition in audit["transitions"]:
        path = Path(str(transition["delta_u64x6_binary"]))
        retained = path.with_name("delta_u64x6.unsorted_v1.bin")
        if retained.exists():
            raise RuntimeError(f"retained original already exists; refuse ambiguous rerun: {retained}")
        packed = np.fromfile(path, dtype="<u8")
        if packed.size % 6:
            raise RuntimeError(f"invalid delta size: {path}")
        packed = packed.reshape(-1, 6)
        before_sha = sha256(path)
        before_multiset = multiset_digest(packed)
        actual = lut[np.ascontiguousarray(packed[:, 1:5]).view(np.uint8)].reshape(len(packed), 32).sum(axis=1)
        mismatch = np.flatnonzero(actual != packed[:, 5])
        if len(mismatch):
            raise RuntimeError(f"popcount mismatch in {path} at row {int(mismatch[0])}")
        ids = packed[:, 0].view("<i8")
        order = np.lexsort((ids, packed[:, 5]))
        repaired = np.ascontiguousarray(packed[order])
        after_multiset = multiset_digest(repaired)
        if before_multiset != after_multiset:
            raise RuntimeError(f"membership changed while sorting {path}")
        if len(repaired) and np.any(repaired[1:, 5] < repaired[:-1, 5]):
            raise RuntimeError(f"population count still unsorted: {path}")
        ties = repaired[1:, 5] == repaired[:-1, 5]
        if np.any(repaired[1:, 0].view("<i8")[ties] < repaired[:-1, 0].view("<i8")[ties]):
            raise RuntimeError(f"ID tie-break is unsorted: {path}")
        os.replace(path, retained)
        part = path.with_suffix(path.suffix + ".part")
        repaired.tofile(part)
        os.replace(part, path)
        results.append({
            "transition": transition["transition"],
            "rows": len(repaired),
            "retained_unsorted_v1": str(retained),
            "retained_unsorted_v1_sha256": before_sha,
            "repaired_path": str(path),
            "repaired_sha256": sha256(path),
            "before_multiset_sha256": before_multiset,
            "after_multiset_sha256": after_multiset,
            "membership_equal": before_multiset == after_multiset,
            "order": "population_count_then_signed_id",
        })
    result = {
        "experiment_id": "tide_20260829_gate6_delta_order_repair",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "amendment": "DELTA_ORDER_AMENDMENT_20260829.md",
        "transitions": results,
        "repair_pass": len(results) == 6 and all(x["membership_equal"] for x in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["repair_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
