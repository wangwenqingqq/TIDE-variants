#!/usr/bin/env python3
"""Generate Safe-C2 v2 tie-free IDs from frozen v1 memberships without GPU use."""
import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

D = 128
W = 100
FVEC_DTYPE = np.dtype([("dim", "<i4"), ("v", "<f4", (D,))])
IVEC_DTYPE = np.dtype([("dim", "<i4"), ("ids", "<i4", (W,))])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256("".join(f"{q}\n" for q in ids).encode()).hexdigest()


def read_ids(path: Path) -> list[int]:
    data = [int(x) for x in path.read_text().split()]
    if len(data) != len(set(data)):
        raise RuntimeError(f"duplicate IDs in {path}")
    return data


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        f.write(text)
        tmp = Path(f.name)
    os.replace(tmp, path)


def ids_payload(ids: list[int]) -> str:
    return "".join(f"{q}\n" for q in ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-protocol", type=Path, required=True)
    ap.add_argument("--v2-preflight-dir", type=Path, required=True)
    args = ap.parse_args()

    p1 = json.loads(args.v1_protocol.read_text())
    if p1.get("schema") != "safe-c2-corrected-after-submit-protocol-v1":
        raise RuntimeError("not the frozen Safe-C2 v1 protocol")
    data = p1["data"]
    base_path = Path(data["base"]["path"])
    query_path = Path(data["query"]["path"])
    gt_path = Path(data["groundtruth"]["path"])
    parts = p1["split"]["files"]
    original = {part: read_ids(Path(parts[part]["path"])) for part in ("calibration", "validation", "test")}
    all_original = sum((original[k] for k in ("calibration", "validation", "test")), [])
    if len(all_original) != 10_000 or set(all_original) != set(range(10_000)):
        raise RuntimeError("v1 split is not disjoint/exhaustive 0..9999")

    base = np.memmap(base_path, dtype=FVEC_DTYPE, mode="r")
    queries = np.memmap(query_path, dtype=FVEC_DTYPE, mode="r")
    gt = np.memmap(gt_path, dtype=IVEC_DTYPE, mode="r")
    if len(base) != 1_000_000 or len(queries) != 10_000 or len(gt) != 10_000:
        raise RuntimeError("unexpected SIFT1M record counts")
    if not np.all(base["dim"] == D) or not np.all(queries["dim"] == D) or not np.all(gt["dim"] == W):
        raise RuntimeError("unexpected fvecs/ivecs row headers")

    ambiguous: dict[int, dict[str, object]] = {}
    for q in range(10_000):
        candidates = gt["ids"][q]
        if np.any(candidates < 0) or np.any(candidates >= len(base)):
            raise RuntimeError(f"invalid public GT ID at q={q}")
        # Frozen preflight establishes integral raw SIFT floats. Convert only the
        # 100 public candidates/query to int64: squared L2 and equality are exact.
        b = base["v"][candidates]
        qq = queries["v"][q]
        if not np.all(b == np.floor(b)) or not np.all(qq == np.floor(qq)):
            raise RuntimeError(f"non-integral coordinate violates v2 exact-L2 premise at q={q}")
        diff = b.astype(np.int64) - qq.astype(np.int64)
        d2 = np.sum(diff * diff, axis=1, dtype=np.int64)
        order = np.lexsort((candidates, d2))
        stored_top10 = {int(x) for x in candidates[:10]}
        stable_top10 = {int(x) for x in candidates[order[:10]]}
        boundary_tie = bool(d2[order[9]] == d2[order[10]])
        set_mismatch = stored_top10 != stable_top10
        if boundary_tie or set_mismatch:
            ambiguous[q] = {
                "boundary_tie": boundary_tie,
                "stored_top10_set_mismatch": set_mismatch,
                "boundary_d2_rank10": int(d2[order[9]]),
                "boundary_d2_rank11": int(d2[order[10]]),
            }

    out_dir = args.v2_preflight_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, list[int]] = {}
    excluded: dict[str, list[int]] = {}
    entries: dict[str, object] = {}
    for part in ("calibration", "validation", "test"):
        selected[part] = [q for q in original[part] if q not in ambiguous]
        excluded[part] = [q for q in original[part] if q in ambiguous]
        selected_path = out_dir / f"{part}.ids"
        excluded_path = out_dir / f"{part}.excluded_ambiguous.ids"
        atomic_text(selected_path, ids_payload(selected[part]))
        atomic_text(excluded_path, ids_payload(excluded[part]))
        entries[part] = {
            "original": {
                "path": str(Path(parts[part]["path"])),
                "count": len(original[part]),
                "sha256": sha(Path(parts[part]["path"])),
            },
            "selected": {
                "path": str(selected_path),
                "count": len(selected[part]),
                "sha256": sha(selected_path),
                "first_ids": selected[part][:8],
            },
            "excluded_ambiguous": {
                "path": str(excluded_path),
                "count": len(excluded[part]),
                "sha256": sha(excluded_path),
                "ids": excluded[part],
            },
        }

    all_excluded = [q for part in ("calibration", "validation", "test") for q in excluded[part]]
    all_excluded_path = out_dir / "all.excluded_ambiguous.ids"
    atomic_text(all_excluded_path, ids_payload(all_excluded))
    audit = {
        "schema": "safe-c2-tiefree-selection-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_used": False,
        "implementation_identity": "Safe-C2 v2 tie-free corrected-after-submit; not submitted legacy C2",
        "source_v1_protocol": {"path": str(args.v1_protocol), "sha256": sha(args.v1_protocol)},
        "raw_data": {
            key: {"path": data[key]["path"], "protocol_sha256": data[key]["sha256"], "actual_sha256": sha(Path(data[key]["path"]))}
            for key in ("base", "query", "groundtruth")
        },
        "split_seed_unchanged": p1["split"]["seed"],
        "membership_rule": "Preserve frozen v1 membership and order within each stage; remove only IDs that violate the public-GT tie-free predicate.",
        "eligibility_rule": {
            "distance": "exact int64 squared L2 on integral raw SIFT128 values; equivalent ordering/equality to raw L2",
            "stable_order": "ascending (squared_L2, ID) within stored public top-100",
            "selected_id_requirements": [
                "rank-10/rank-11 squared-L2 are unequal",
                "stored public GT first-10 ID set equals stable raw-L2 top-10 ID set",
            ],
        },
        "partitions": entries,
        "all_excluded_ambiguous": {
            "path": str(all_excluded_path),
            "count": len(all_excluded),
            "sha256": sha(all_excluded_path),
            "ids": all_excluded,
        },
        "selected_total": sum(len(v) for v in selected.values()),
        "excluded_total": len(all_excluded),
        "selected_disjoint": len(set().union(*map(set, selected.values()))) == sum(len(v) for v in selected.values()),
        "selected_subset_of_v1": all(set(selected[k]).issubset(original[k]) for k in selected),
    }
    if audit["excluded_total"] != len(ambiguous) or set(all_excluded) != set(ambiguous):
        raise RuntimeError("per-stage exclusions do not exactly cover ambiguity set")
    if audit["selected_total"] != 9863:
        raise RuntimeError(f"unexpected tie-free selected total {audit['selected_total']}")
    audit_path = out_dir / "selection_audit.json"
    atomic_text(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS",
        "cuda_used": False,
        "selection_audit": str(audit_path),
        "selected_counts": {k: len(v) for k, v in selected.items()},
        "excluded_counts": {k: len(v) for k, v in excluded.items()},
        "selected_total": audit["selected_total"],
        "excluded_total": audit["excluded_total"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
