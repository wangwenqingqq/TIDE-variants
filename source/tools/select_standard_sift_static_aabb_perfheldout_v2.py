#!/usr/bin/env python3
"""CPU-only deterministic selection of a fresh SIFT1M performance-held-out set."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
BASE = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs")
QUERY = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs")
GT = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs")
IDS_OUT = ROOT / "inputs/standard_sift_static_aabb_perfheldout256_v2.ids"
META_OUT = ROOT / "provenance/static_aabb_perfheldout_selection_v2.json"

DIM = 128
BASE_COUNT = 1_000_000
QUERY_COUNT = 10_000
GT_WIDTH = 100
TARGET_COUNT = 256
SCAN_START = 5260
SCAN_STOP_EXCLUSIVE = 10_000
EXCLUDED_INTERVALS = ((0, 31), (1000, 1259), (5000, 5259))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_regular(path: Path, expected_bytes: int) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size != expected_bytes:
        raise RuntimeError(f"invalid input: {path}")


def read_fvec_at(path: Path, row: int, count: int) -> tuple[float, ...]:
    if not 0 <= row < count:
        raise RuntimeError(f"fvec row out of range: {row}")
    stride = 4 + 4 * DIM
    with path.open("rb") as f:
        f.seek(row * stride)
        raw_dim = f.read(4)
        raw = f.read(4 * DIM)
    if len(raw_dim) != 4 or struct.unpack("<i", raw_dim)[0] != DIM or len(raw) != 4 * DIM:
        raise RuntimeError(f"bad fvec: {row}")
    return struct.unpack("<128f", raw)


def read_gt_at(row: int) -> tuple[int, ...]:
    if not 0 <= row < QUERY_COUNT:
        raise RuntimeError(f"GT row out of range: {row}")
    stride = 4 + 4 * GT_WIDTH
    with GT.open("rb") as f:
        f.seek(row * stride)
        raw_dim = f.read(4)
        raw = f.read(4 * GT_WIDTH)
    if len(raw_dim) != 4 or struct.unpack("<i", raw_dim)[0] != GT_WIDTH or len(raw) != 4 * GT_WIDTH:
        raise RuntimeError(f"bad GT: {row}")
    ids = struct.unpack("<100i", raw)
    if any(x < 0 or x >= BASE_COUNT for x in ids[:11]):
        raise RuntimeError(f"invalid GT ID: {row}")
    return ids


def sq_l2(q: tuple[float, ...], x: tuple[float, ...]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(q, x))


def eligible(qid: int) -> bool:
    return not any(lo <= qid <= hi for lo, hi in EXCLUDED_INTERVALS)


def atomic_write(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing existing output: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    check_regular(BASE, BASE_COUNT * (4 + 4 * DIM))
    check_regular(QUERY, QUERY_COUNT * (4 + 4 * DIM))
    check_regular(GT, QUERY_COUNT * (4 + 4 * GT_WIDTH))
    if IDS_OUT.exists() or IDS_OUT.is_symlink() or META_OUT.exists() or META_OUT.is_symlink():
        raise RuntimeError("v2 selection output already exists")

    selected: list[int] = []
    witness: list[dict[str, object]] = []
    for qid in range(SCAN_START, SCAN_STOP_EXCLUSIVE):
        if not eligible(qid):
            continue
        q = read_fvec_at(QUERY, qid, QUERY_COUNT)
        gt_ids = read_gt_at(qid)
        d10 = sq_l2(q, read_fvec_at(BASE, gt_ids[9], BASE_COUNT))
        d11 = sq_l2(q, read_fvec_at(BASE, gt_ids[10], BASE_COUNT))
        if d10 < d11:
            selected.append(qid)
            witness.append({"qid": qid, "d10_sq_float64": d10, "d11_sq_float64": d11})
            if len(selected) == TARGET_COUNT:
                break
    if len(selected) != TARGET_COUNT or selected != sorted(selected) or len(set(selected)) != TARGET_COUNT:
        raise RuntimeError(f"selected {len(selected)} IDs")
    if any(not eligible(qid) for qid in selected) or any(qid < SCAN_START for qid in selected):
        raise RuntimeError("selection interval violation")

    atomic_write(IDS_OUT, "".join(f"{qid}\n" for qid in selected))
    meta = {
        "schema": "safe-c2-static-aabb-fresh-performance-heldout-selection-v2",
        "status": "FROZEN_CPU_ONLY",
        "scope": "fresh standard-SIFT1M performance-held-out IDs for a successor static-AABB telemetry batch only; no external/blinded-test claim",
        "selection_policy": {
            "candidate_qid_interval_half_open": [SCAN_START, SCAN_STOP_EXCLUSIVE],
            "excluded_qid_intervals_inclusive": [list(x) for x in EXCLUDED_INTERVALS],
            "scan_order": "increasing integer query ID",
            "target_count": TARGET_COUNT,
            "eligibility": "recompute GT rank-10/rank-11 raw squared L2 in CPU binary64 and retain only d10_sq < d11_sq",
            "does_not_read": ["AABB bounds", "tree traversal", "GPU state", "GPU timing", "prior run artifacts", "failed-batch timing values"],
        },
        "freshness_boundary": {
            "excludes_entire_prior_observed_interval": [5000, 5259],
            "reason": "predecessor telemetry batch produced visible raw timings but failed monitoring acceptance; this selection is disjoint and does not read those values",
        },
        "selected_ids": selected,
        "boundary_witness": witness,
        "inputs": {"base": artifact(BASE), "query": artifact(QUERY), "groundtruth": artifact(GT)},
        "output_ids": artifact(IDS_OUT),
        "selector": artifact(Path(__file__).resolve()),
        "gpu_executed": False,
        "nvidia_smi_called": False,
        "formal_claim_eligible": False,
    }
    atomic_write(META_OUT, json.dumps(meta, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "FROZEN_CPU_ONLY", "ids": str(IDS_OUT), "metadata": str(META_OUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_STATIC_AABB_PERFHELDOUT_SELECTION_V2: {exc}", file=sys.stderr)
        raise SystemExit(2)
