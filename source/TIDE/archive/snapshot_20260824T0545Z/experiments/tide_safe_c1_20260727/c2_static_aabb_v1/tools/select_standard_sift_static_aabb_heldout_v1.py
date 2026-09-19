#!/usr/bin/env python3
"""Freeze a deterministic development-held-out standard-SIFT ID set.

This is CPU-only.  Selection uses only the raw SIFT files and the public GT
rank-10/rank-11 boundary to avoid an ambiguous exact-top-10 set.  It does not
read AABB, traversal, timing, GPU, or prior run artifacts.
"""
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
IDS_OUT = ROOT / "inputs/standard_sift_static_aabb_heldout256_v1.ids"
EXACT_IDS_OUT = ROOT / "inputs/standard_sift_static_aabb_heldout_exact64_v1.ids"
META_OUT = ROOT / "provenance/static_aabb_heldout_selection_v1.json"
DIM = 128
BASE_COUNT = 1_000_000
QUERY_COUNT = 10_000
GT_WIDTH = 100
TARGET_COUNT = 256
SCAN_START = 5000
SCAN_STOP_EXCLUSIVE = 10_000
EXCLUDED_INTERVALS = ((0, 31), (1000, 1259))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_regular(path: Path, expected_bytes: int) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid input path: {path}")
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"unexpected byte count for {path}: {path.stat().st_size}")


def read_fvec_at(path: Path, row: int, count: int) -> tuple[float, ...]:
    if not 0 <= row < count:
        raise RuntimeError(f"fvec row out of range: {row}")
    stride = 4 + 4 * DIM
    with path.open("rb") as f:
        f.seek(row * stride)
        raw_dim = f.read(4)
        if len(raw_dim) != 4 or struct.unpack("<i", raw_dim)[0] != DIM:
            raise RuntimeError(f"bad fvec header at row {row}")
        raw = f.read(4 * DIM)
        if len(raw) != 4 * DIM:
            raise RuntimeError(f"short fvec at row {row}")
        return struct.unpack("<128f", raw)


def read_gt_at(path: Path, row: int) -> tuple[int, ...]:
    if not 0 <= row < QUERY_COUNT:
        raise RuntimeError(f"GT row out of range: {row}")
    stride = 4 + 4 * GT_WIDTH
    with path.open("rb") as f:
        f.seek(row * stride)
        raw_dim = f.read(4)
        if len(raw_dim) != 4 or struct.unpack("<i", raw_dim)[0] != GT_WIDTH:
            raise RuntimeError(f"bad GT header at row {row}")
        raw = f.read(4 * GT_WIDTH)
        if len(raw) != 4 * GT_WIDTH:
            raise RuntimeError(f"short GT row {row}")
        ids = struct.unpack("<100i", raw)
    if any(index < 0 or index >= BASE_COUNT for index in ids[:11]):
        raise RuntimeError(f"invalid top-11 GT ID at row {row}")
    return ids


def eligible(qid: int) -> bool:
    return not any(lo <= qid <= hi for lo, hi in EXCLUDED_INTERVALS)


def sq_l2(q: tuple[float, ...], x: tuple[float, ...]) -> float:
    # Python binary64 arithmetic is an independent CPU-only boundary witness.
    return sum((float(a) - float(b)) ** 2 for a, b in zip(q, x))


def atomic_write(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing pre-existing output: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    check_regular(BASE, BASE_COUNT * (4 + 4 * DIM))
    check_regular(QUERY, QUERY_COUNT * (4 + 4 * DIM))
    check_regular(GT, QUERY_COUNT * (4 + 4 * GT_WIDTH))
    if (IDS_OUT.exists() or IDS_OUT.is_symlink() or EXACT_IDS_OUT.exists() or
            EXACT_IDS_OUT.is_symlink() or META_OUT.exists() or META_OUT.is_symlink()):
        raise RuntimeError("held-out selection outputs already exist; refusing to overwrite")

    selected: list[int] = []
    witness: list[dict[str, object]] = []
    for qid in range(SCAN_START, SCAN_STOP_EXCLUSIVE):
        if not eligible(qid):
            continue
        q = read_fvec_at(QUERY, qid, QUERY_COUNT)
        gt_ids = read_gt_at(GT, qid)
        d10 = sq_l2(q, read_fvec_at(BASE, gt_ids[9], BASE_COUNT))
        d11 = sq_l2(q, read_fvec_at(BASE, gt_ids[10], BASE_COUNT))
        if d10 < d11:
            selected.append(qid)
            witness.append({"qid": qid, "d10_sq_float64": d10, "d11_sq_float64": d11})
            if len(selected) == TARGET_COUNT:
                break
    if len(selected) != TARGET_COUNT:
        raise RuntimeError(f"only selected {len(selected)} of {TARGET_COUNT}")
    if len(set(selected)) != len(selected) or selected != sorted(selected):
        raise RuntimeError("non-unique or non-monotone selected IDs")
    if any(not eligible(qid) for qid in selected):
        raise RuntimeError("selected excluded ID")

    exact_ids = selected[:64]
    if len(exact_ids) != 64:
        raise RuntimeError("short exact-oracle prefix")
    atomic_write(IDS_OUT, "".join(f"{qid}\n" for qid in selected))
    atomic_write(EXACT_IDS_OUT, "".join(f"{qid}\n" for qid in exact_ids))
    metadata = {
        "schema": "safe-c2-static-aabb-standard-sift-heldout-selection-v1",
        "status": "FROZEN_CPU_ONLY",
        "scope": (
            "development-held-out static-AABB evaluation input for standard SIFT1M only; "
            "not old-C2 validation/sealed data, not a blinded external test, and not a performance result"
        ),
        "selection_policy": {
            "candidate_qid_interval_half_open": [SCAN_START, SCAN_STOP_EXCLUSIVE],
            "excluded_qid_intervals_inclusive": [list(x) for x in EXCLUDED_INTERVALS],
            "scan_order": "increasing integer query ID",
            "target_count": TARGET_COUNT,
            "eligibility": (
                "recompute public GT rank-10 and rank-11 squared L2 distances in CPU binary64; "
                "retain when d10_sq < d11_sq"
            ),
            "does_not_read": [
                "AABB bounds",
                "tree traversal outputs",
                "GPU state",
                "GPU timing",
                "prior run artifacts",
            ],
        },
        "selected_ids": selected,
        "boundary_witness": witness,
        "inputs": {
            str(BASE): {"bytes": BASE.stat().st_size, "sha256": sha256(BASE)},
            str(QUERY): {"bytes": QUERY.stat().st_size, "sha256": sha256(QUERY)},
            str(GT): {"bytes": GT.stat().st_size, "sha256": sha256(GT)},
        },
        "output_ids": {"path": str(IDS_OUT), "sha256": sha256(IDS_OUT), "bytes": IDS_OUT.stat().st_size},
        "gpu_heldout_correctness_subset": {
            "ids_path": str(EXACT_IDS_OUT),
            "ids_sha256": sha256(EXACT_IDS_OUT),
            "query_count": len(exact_ids),
            "selected_ids": exact_ids,
            "relation_to_heldout256": "ordered prefix of the same frozen heldout256 list",
            "required_oracle": "independent CPU exact top-10 over all 1,000,000 base vectors; do not use GT",
        },
        "gt_gated_heldout_performance_set": {
            "ids_path": str(IDS_OUT),
            "ids_sha256": sha256(IDS_OUT),
            "query_count": len(selected),
            "GT_role": (
                "fixed standard label for the performance correctness gate only; "
                "not a projection/calibration/timing tuning signal"
            ),
        },
        "selector": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "gpu_executed": False,
        "nvidia_smi_called": False,
        "formal_claim_eligible": False,
    }
    atomic_write(META_OUT, json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    print(META_OUT)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_STATIC_AABB_HELDOUT_SELECTION_V1: {exc}", file=sys.stderr)
        raise SystemExit(2)
