#!/usr/bin/env python3
"""CPU-only, fail-closed raw-data verifier for Safe-C2 v2 tie-free admission.

The frozen selection is accepted only if BOTH checks agree: exact int64 squared
L2 on integral SIFT coordinates, and the explicit source-level C++ evaluator
recurrence (ordered fp32 accumulation followed by float32 sqrt).  No CUDA,
NVIDIA-SMI, or GTS binary is invoked by this program.
"""
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
PARTS = ("calibration", "validation", "test")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_ids(path: Path) -> list[int]:
    data = [int(x) for x in path.read_text().split()]
    if len(data) != len(set(data)):
        raise RuntimeError(f"duplicate IDs in {path}")
    return data


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        temp = Path(f.name)
    os.replace(temp, path)


def need(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def entry_check(entry: dict, failures: list[str], label: str) -> list[int]:
    path = Path(entry.get("path", ""))
    need(path.is_file(), f"{label}: missing file {path}", failures)
    if not path.is_file():
        return []
    ids = read_ids(path)
    need(len(ids) == entry.get("count"), f"{label}: count mismatch", failures)
    need(sha(path) == entry.get("sha256"), f"{label}: SHA mismatch", failures)
    return ids


def cxx_raw_l2_fp32_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized across rows, but strictly sequential fp32 across dimensions."""
    total = np.zeros(a.shape[0], dtype=np.float32)
    for dim in range(D):
        delta = np.subtract(a[:, dim], b[dim], dtype=np.float32)
        term = np.multiply(delta, delta, dtype=np.float32)
        total = np.add(total, term, dtype=np.float32)
    return np.sqrt(total, dtype=np.float32)


def classify_query(base, queries, gt, q: int) -> dict:
    candidates = gt["ids"][q]
    if np.any(candidates < 0) or np.any(candidates >= len(base)):
        raise RuntimeError(f"q={q}: invalid public GT ID")
    b = base["v"][candidates]
    qq = queries["v"][q]
    if not np.all(b == np.floor(b)) or not np.all(qq == np.floor(qq)):
        raise RuntimeError(f"q={q}: non-integral coordinate violates v2 exact-L2 premise")
    diff = b.astype(np.int64) - qq.astype(np.int64)
    d2 = np.sum(diff * diff, axis=1, dtype=np.int64)
    exact_order = np.lexsort((candidates, d2))
    fp32 = cxx_raw_l2_fp32_rows(b, qq)
    fp32_order = np.lexsort((candidates, fp32))
    stored_top10 = {int(x) for x in candidates[:10]}
    exact_top10 = {int(x) for x in candidates[exact_order[:10]]}
    fp32_top10 = {int(x) for x in candidates[fp32_order[:10]]}
    # Count only collisions of different exact squared distances in the same
    # fp32 result bucket. Equal true distances are not a representation drift.
    by_bits: dict[int, set[int]] = {}
    for dist, sq in zip(fp32, d2):
        bits = int(np.asarray(dist, dtype=np.float32).view(np.uint32))
        by_bits.setdefault(bits, set()).add(int(sq))
    collapse_pairs = sum(len(v) * (len(v) - 1) // 2 for v in by_bits.values() if len(v) > 1)
    return {
        "exact_boundary_tie": bool(d2[exact_order[9]] == d2[exact_order[10]]),
        "exact_stored_top10_mismatch": stored_top10 != exact_top10,
        "fp32_boundary_tie": bool(fp32[fp32_order[9]] == fp32[fp32_order[10]]),
        "fp32_stored_top10_mismatch": stored_top10 != fp32_top10,
        "fp32_equal_distance_distinct_d2_pairs": int(collapse_pairs),
        "max_d2": int(np.max(d2)),
    }


def is_ambiguous(c: dict) -> bool:
    return any((c["exact_boundary_tie"], c["exact_stored_top10_mismatch"],
                c["fp32_boundary_tie"], c["fp32_stored_top10_mismatch"],
                c["fp32_equal_distance_distinct_d2_pairs"] != 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--preflight", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    failures: list[str] = []
    try:
        protocol = json.loads(args.protocol.read_text())
        preflight = json.loads(args.preflight.read_text())
    except Exception as exc:
        atomic_json(args.out, {"schema": "safe-c2-tiefree-reverification-v2", "status": "FAIL", "cuda_used": False, "failures": [str(exc)]})
        return 2

    root = args.protocol.parent.parent.resolve()
    need(protocol.get("schema") == "safe-c2-corrected-after-submit-tiefree-protocol-v2", "wrong v2 protocol schema", failures)
    need(preflight.get("schema") == "safe-c2-corrected-after-submit-tiefree-preflight-v2", "wrong v2 preflight schema", failures)
    need(protocol.get("implementation_identity", {}).get("v2_tiefree") is True, "protocol lacks v2_tiefree identity", failures)
    need(protocol.get("resource_policy", {}).get("write_boundary") == str(root), "protocol write boundary does not equal v2 root", failures)
    split = protocol.get("split", {})
    need(split.get("selected_evaluation_exhaustive_over_original_10000") is False, "protocol selected-coverage field is wrong", failures)
    need(split.get("selected_plus_excluded_partition_original_10000") is True, "protocol partition field is wrong", failures)
    need(protocol.get("data", {}).get("preflight_path") == str(args.preflight), "protocol preflight path mismatch", failures)
    need(protocol.get("data", {}).get("preflight_sha256") == sha(args.preflight), "protocol preflight SHA mismatch", failures)
    need(preflight.get("tie_free_v2", {}).get("selection_audit_path") == protocol.get("tie_free_v2", {}).get("selection_audit_path"), "protocol/preflight selection-audit path mismatch", failures)
    need(preflight.get("tie_free_v2", {}).get("selection_audit_sha256") == protocol.get("tie_free_v2", {}).get("selection_audit_sha256"), "protocol/preflight selection-audit SHA mismatch", failures)
    need(preflight.get("tie_free_v2", {}).get("fp32_selected_checks") == protocol.get("tie_free_v2", {}).get("fp32_selected_checks"), "protocol/preflight fp32 admission statistics mismatch", failures)

    data_entries = {
        "base": (protocol.get("data", {}).get("base", {}), preflight.get("inputs", {}).get("base", {}), FVEC_DTYPE, 1_000_000),
        "query": (protocol.get("data", {}).get("query", {}), preflight.get("inputs", {}).get("query", {}), FVEC_DTYPE, 10_000),
        "groundtruth": (protocol.get("data", {}).get("groundtruth", {}), preflight.get("inputs", {}).get("groundtruth", {}), IVEC_DTYPE, 10_000),
    }
    raw_actual: dict[str, str] = {}
    maps = {}
    for label, (p_entry, q_entry, dtype, expected_n) in data_entries.items():
        path = Path(p_entry.get("path", ""))
        need(path.is_file(), f"{label}: missing raw input", failures)
        need(p_entry.get("path") == q_entry.get("path"), f"{label}: protocol/preflight path disagreement", failures)
        need(p_entry.get("sha256") == q_entry.get("sha256"), f"{label}: protocol/preflight SHA disagreement", failures)
        if path.is_file():
            raw_actual[label] = sha(path)
            need(raw_actual[label] == p_entry.get("sha256"), f"{label}: actual raw SHA mismatch", failures)
            try:
                mm = np.memmap(path, dtype=dtype, mode="r")
                need(len(mm) == expected_n, f"{label}: record count mismatch", failures)
                need(np.all(mm["dim"] == (D if label != "groundtruth" else W)), f"{label}: row dimension mismatch", failures)
                maps[label] = mm
            except Exception as exc:
                failures.append(f"{label}: cannot memmap/check raw file: {exc}")

    v1_info = split.get("v1_original_protocol", {})
    v1_path = Path(v1_info.get("path", ""))
    need(v1_path.is_file() and sha(v1_path) == v1_info.get("sha256"), "frozen v1 protocol provenance mismatch", failures)
    original = {}; selected = {}; excluded = {}
    for part in PARTS:
        original[part] = entry_check(split.get("v1_original_files", {}).get(part, {}), failures, f"{part}.v1_original")
        selected[part] = entry_check(split.get("files", {}).get(part, {}), failures, f"{part}.selected")
        excluded[part] = entry_check(split.get("excluded_ambiguous_files", {}).get(part, {}), failures, f"{part}.excluded")
    all_original = sum((original[p] for p in PARTS), [])
    all_selected = sum((selected[p] for p in PARTS), [])
    all_excluded = sum((excluded[p] for p in PARTS), [])
    need(len(all_original) == 10_000 and set(all_original) == set(range(10_000)), "v1 memberships not disjoint/exhaustive 0..9999", failures)
    need(len(all_selected) == len(set(all_selected)), "v2 selected split overlap", failures)
    need(set(all_selected).isdisjoint(set(all_excluded)), "v2 selected/excluded overlap", failures)
    need(set(all_selected) | set(all_excluded) == set(all_original), "v2 selected/excluded do not partition v1 membership", failures)
    need(len(all_selected) == split.get("selected_total") == 9863, "v2 selected total/count invariant failed", failures)
    need(len(all_excluded) == split.get("excluded_total") == 137, "v2 excluded total/count invariant failed", failures)

    classes: dict[int, dict] = {}
    if not failures and len(maps) == 3:
        for q in all_original:
            classes[q] = classify_query(maps["base"], maps["query"], maps["groundtruth"], q)
        for part in PARTS:
            expected_selected = [q for q in original[part] if not is_ambiguous(classes[q])]
            expected_excluded = [q for q in original[part] if is_ambiguous(classes[q])]
            need(selected[part] == expected_selected, f"{part}: selected IDs do not preserve v1 order after exact+fp32 tie-free exclusion", failures)
            need(excluded[part] == expected_excluded, f"{part}: excluded IDs do not equal exact+fp32 ambiguity set", failures)

    selection_info = protocol.get("tie_free_v2", {})
    selection_path = Path(selection_info.get("selection_audit_path", ""))
    need(selection_path.is_file() and sha(selection_path) == selection_info.get("selection_audit_sha256"), "selection audit provenance mismatch", failures)
    if selection_path.is_file():
        try:
            selection = json.loads(selection_path.read_text())
            need(selection.get("schema") == "safe-c2-tiefree-selection-v2", "wrong selection audit schema", failures)
            need(selection.get("selected_total") == len(all_selected), "selection audit selected total mismatch", failures)
            need(selection.get("excluded_total") == len(all_excluded), "selection audit excluded total mismatch", failures)
        except Exception as exc:
            failures.append(f"cannot parse selection audit: {exc}")

    def stats(ids: list[int]) -> dict:
        cs = [classes[q] for q in ids]
        return {
            "fp32_boundary_ties": sum(c["fp32_boundary_tie"] for c in cs),
            "fp32_stored_top10_mismatches": sum(c["fp32_stored_top10_mismatch"] for c in cs),
            "fp32_equal_distance_distinct_d2_pairs": sum(c["fp32_equal_distance_distinct_d2_pairs"] for c in cs),
            "max_d2": max((c["max_d2"] for c in cs), default=0),
        }
    exact_ties = {p: sum(classes[q]["exact_boundary_tie"] for q in original[p]) for p in PARTS} if classes else {p: 0 for p in PARTS}
    exact_mismatches = {p: sum(classes[q]["exact_stored_top10_mismatch"] for q in original[p]) for p in PARTS} if classes else {p: 0 for p in PARTS}
    fp32_selected = {p: stats(selected[p]) for p in PARTS} if classes else {p: {} for p in PARTS}
    selected_bad = sum(1 for q in all_selected if q in classes and is_ambiguous(classes[q]))
    for part, s in fp32_selected.items():
        need(s.get("fp32_boundary_ties", 0) == 0, f"{part}: selected fp32 boundary tie", failures)
        need(s.get("fp32_stored_top10_mismatches", 0) == 0, f"{part}: selected fp32 stored-top10 mismatch", failures)
        need(s.get("fp32_equal_distance_distinct_d2_pairs", 0) == 0, f"{part}: selected fp32 equal-distance distinct-d2 collapse", failures)

    payload = {
        "schema": "safe-c2-tiefree-reverification-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_used": False,
        "status": "PASS" if not failures else "FAIL",
        "protocol_path": str(args.protocol), "protocol_sha256": sha(args.protocol),
        "preflight_path": str(args.preflight), "preflight_sha256": sha(args.preflight),
        "raw_actual_sha256": raw_actual,
        "selected_counts": {p: len(selected[p]) for p in PARTS},
        "excluded_counts": {p: len(excluded[p]) for p in PARTS},
        "recomputed_ambiguous_counts": {p: sum(1 for q in original[p] if q in classes and is_ambiguous(classes[q])) for p in PARTS},
        "recomputed_set_mismatch_counts": exact_mismatches,
        "recomputed_exact_boundary_tie_counts": exact_ties,
        "selected_tie_or_mismatch_count": selected_bad,
        "fp32_source_semantics": "ordered volatile float32 accumulation of delta*delta, then float32 sqrt; matches raw_l2 source-level recurrence",
        "fp32_selected_checks": fp32_selected,
        "rule": "selected IDs must pass exact int64 squared-L2 stable order AND source-level fp32 raw_l2 stable order within public GT100",
        "failures": failures,
    }
    atomic_json(args.out, payload)
    print(f"{payload['status']} {args.out}")
    return 0 if not failures else 3

if __name__ == "__main__":
    raise SystemExit(main())
