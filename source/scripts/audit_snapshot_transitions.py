#!/usr/bin/env python3
"""Audit real snapshot transitions without assuming insert-only behavior."""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import time
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np

DATES = [
    "2026-06-01", "2026-06-15", "2026-07-01", "2026-07-17",
    "2026-08-04", "2026-08-18", "2026-08-25",
]


def schema(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        dataset = handle["fps"]
        return {
            "shape": list(dataset.shape),
            "dtype": str(dataset.dtype),
            "dtype_names": list(dataset.dtype.names or ()),
            "chunks": list(dataset.chunks or ()),
            "compression": str(dataset.compression),
            "file_attrs": {key: str(value) for key, value in handle.attrs.items()},
            "dataset_attrs": {key: str(value) for key, value in dataset.attrs.items()},
        }


def load(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return handle["fps"][:]


def sorted_unique_ids(rows: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray, int]:
    order = np.argsort(rows["fp_id"], kind="stable")
    ids = rows["fp_id"][order]
    duplicates = int(np.count_nonzero(ids[1:] == ids[:-1]))
    return order, ids, duplicates


def fp_fields(dtype: np.dtype) -> list[str]:
    names = list(dtype.names or ())
    fields = [name for name in names if name.startswith("f") and name[1:].isdigit()]
    return sorted(fields, key=lambda name: int(name[1:]))


def equal_fp(left: np.ndarray, right: np.ndarray, fields: list[str]) -> np.ndarray:
    same = np.ones(len(left), dtype=bool)
    for field in fields:
        same &= left[field] == right[field]
    if "popcnt" in (left.dtype.names or ()) and "popcnt" in (right.dtype.names or ()):
        same &= left["popcnt"] == right["popcnt"]
    return same


def sample_queries(rows: np.ndarray, row_indices: np.ndarray, transition: str, count: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    # Keep only the smallest requested SHA-256 keys. Some real transitions can
    # add millions of rows, so materializing and sorting one Python tuple per
    # addition would distort the audit's memory contract.
    heap: list[tuple[int, int, int]] = []
    for row in row_indices:
        fp_id = int(rows["fp_id"][row])
        key_int = int.from_bytes(
            hashlib.sha256(f"20260828:{transition}:{fp_id}".encode()).digest(), "big"
        )
        entry = (-key_int, -int(row), fp_id)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    chosen = [((-neg_key).to_bytes(32, "big"), -neg_row, fp_id) for neg_key, neg_row, fp_id in heap]
    chosen.sort()
    selected = rows[np.array([item[1] for item in chosen], dtype=np.int64)]
    manifest = [
        {"rank": rank, "fp_id": fp_id, "source_row": row, "selection_sha256": key.hex()}
        for rank, (key, row, fp_id) in enumerate(chosen)
    ]
    return selected, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=512)
    args = parser.parse_args()
    root = args.root.resolve()
    official = root / "data" / "official" / "surechembl"
    query_root = root / "data" / "prepared" / "transition_queries"
    query_root.mkdir(parents=True, exist_ok=True)

    file_schemas = {}
    for date in DATES:
        path = official / date / "fpsim2_fingerprints.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        file_schemas[date] = schema(path)

    transitions = []
    for base_date, union_date in zip(DATES, DATES[1:]):
        transition = f"{base_date}_to_{union_date}"
        begin = time.monotonic()
        base_path = official / base_date / "fpsim2_fingerprints.h5"
        union_path = official / union_date / "fpsim2_fingerprints.h5"
        print(f"loading {transition}", flush=True)
        base = load(base_path)
        union = load(union_path)
        if base.dtype != union.dtype:
            raise RuntimeError(f"dtype mismatch for {transition}: {base.dtype} vs {union.dtype}")
        fields = fp_fields(base.dtype)
        if not fields:
            raise RuntimeError(f"no fingerprint fields for {transition}")
        b_order, b_ids, b_duplicates = sorted_unique_ids(base, f"{transition}:base")
        u_order, u_ids, u_duplicates = sorted_unique_ids(union, f"{transition}:union")

        bpos_for_u = np.searchsorted(b_ids, u_ids)
        added = bpos_for_u == len(b_ids)
        in_range = ~added
        added[in_range] = b_ids[bpos_for_u[in_range]] != u_ids[in_range]
        upos_for_b = np.searchsorted(u_ids, b_ids)
        removed = upos_for_b == len(u_ids)
        in_range = ~removed
        removed[in_range] = u_ids[upos_for_b[in_range]] != b_ids[in_range]

        common_u_sorted = np.flatnonzero(~added)
        common_b_sorted = bpos_for_u[common_u_sorted]
        changed_count = 0
        changed_ids_sample: list[int] = []
        for start in range(0, len(common_u_sorted), 1_000_000):
            us = common_u_sorted[start : start + 1_000_000]
            bs = common_b_sorted[start : start + 1_000_000]
            unequal = ~equal_fp(union[u_order[us]], base[b_order[bs]], fields)
            if np.any(unequal):
                values = u_ids[us[unequal]]
                changed_count += int(len(values))
                if len(changed_ids_sample) < 100:
                    take = min(100 - len(changed_ids_sample), len(values))
                    changed_ids_sample.extend(int(x) for x in values[:take])

        added_rows = u_order[np.flatnonzero(added)]
        removed_rows = b_order[np.flatnonzero(removed)]
        selected, selection = sample_queries(union, added_rows, transition, args.query_count)
        query_dir = query_root / transition
        query_dir.mkdir(parents=True, exist_ok=True)
        selected.tofile(query_dir / "queries_struct.bin")
        (query_dir / "queries.json").write_text(json.dumps(selection, indent=2) + "\n")

        # Gate-5-compatible precomputed row layout: id, fingerprint words, popcount.
        delta_rows_ordered = added_rows[
            np.lexsort((union["fp_id"][added_rows], union["popcnt"][added_rows]))
        ]
        delta_u64 = np.empty((len(delta_rows_ordered), 2 + len(fields)), dtype="<u8")
        delta_u64[:, 0] = union["fp_id"][delta_rows_ordered].view("<u8")
        for word, field in enumerate(fields):
            delta_u64[:, word + 1] = union[field][delta_rows_ordered]
        delta_u64[:, -1] = union["popcnt"][delta_rows_ordered].astype("<u8")
        delta_path = query_dir / "delta_u64x6.bin"
        delta_u64.tofile(delta_path)

        query_u64 = np.empty((len(selected), 2 + len(fields)), dtype="<u8")
        query_u64[:, 0] = selected["fp_id"].view("<u8")
        for word, field in enumerate(fields):
            query_u64[:, word + 1] = selected[field]
        query_u64[:, -1] = selected["popcnt"].astype("<u8")
        query_path = query_dir / "queries_u64x6.bin"
        query_u64.tofile(query_path)
        counts = {
            "base_rows": int(len(base)),
            "union_rows": int(len(union)),
            "added": int(len(added_rows)),
            "removed": int(len(removed_rows)),
            "changed_common": changed_count,
            "base_duplicate_ids": b_duplicates,
            "union_duplicate_ids": u_duplicates,
            "selected_queries": len(selection),
            "insert_only_exact": b_duplicates == 0 and u_duplicates == 0 and len(removed_rows) == 0 and changed_count == 0,
            "row_balance": int(len(base) + len(added_rows) - len(removed_rows) - len(union)),
        }
        transitions.append({
            "transition": transition,
            "base": base_date,
            "union": union_date,
            "fingerprint_fields": fields,
            "fingerprint_bits": len(fields) * 64,
            "counts": counts,
            "removed_ids_sample": [int(base["fp_id"][row]) for row in removed_rows[:100]],
            "changed_ids_sample": changed_ids_sample,
            "query_manifest": str(query_dir / "queries.json"),
            "query_struct_binary": str(query_dir / "queries_struct.bin"),
            "query_u64x6_binary": str(query_path),
            "delta_u64x6_binary": str(delta_path),
            "elapsed_s": time.monotonic() - begin,
        })
        print(json.dumps({"transition": transition, **counts}), flush=True)
        del base, union, b_order, b_ids, u_order, u_ids
        gc.collect()

    chembl_path = root / "data" / "official" / "chembl_37" / "chembl_37.h5"
    result = {
        "experiment_id": "tide_20260828_gate6_transition_reality",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "surechembl_schemas": file_schemas,
        "chembl_37_schema": schema(chembl_path),
        "transitions": transitions,
        "complete_transition_count": len(transitions),
        "all_unique_ids": all(t["counts"]["base_duplicate_ids"] == 0 and t["counts"]["union_duplicate_ids"] == 0 for t in transitions),
        "insert_only_transition_count": sum(bool(t["counts"]["insert_only_exact"]) for t in transitions),
        # This is only the transition/schema component. The aggregate
        # G6-DATA-REALITY gate also requires provenance, source hashes, and the
        # separate ChEMBL content-integrity result, so this script must not
        # relabel its narrower result as the complete gate.
        "transition_audit_component_pass": len(transitions) == 6 and all(
            t["counts"]["base_duplicate_ids"] == 0 and t["counts"]["union_duplicate_ids"] == 0
            for t in transitions
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "complete_transition_count", "all_unique_ids", "insert_only_transition_count",
        "transition_audit_component_pass",
    )}, indent=2))
    return 0 if result["transition_audit_component_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
