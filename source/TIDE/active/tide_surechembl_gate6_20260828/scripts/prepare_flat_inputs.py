#!/usr/bin/env python3
"""Prepare hash-tracked, width-explicit flat inputs for Gate 6."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np

DATES = [
    "2026-06-01", "2026-06-15", "2026-07-01", "2026-07-17",
    "2026-08-04", "2026-08-18", "2026-08-25",
]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            value.update(block)
    return value.hexdigest()


def fields(dtype: np.dtype) -> list[str]:
    names = list(dtype.names or ())
    result = [name for name in names if name.startswith("f") and name[1:].isdigit()]
    return sorted(result, key=lambda name: int(name[1:]))


def atomic_replace(part: Path, final: Path) -> None:
    os.replace(part, final)


def write_snapshot(source: Path, output: Path, row_mask: np.ndarray | None = None) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as handle:
        dataset = handle["fps"]
        names = fields(dataset.dtype)
        word_count = len(names)
        if word_count not in (4, 32):
            raise RuntimeError(
                f"Gate-6 supports only frozen 256/2048-bit contracts, got "
                f"{word_count} u64 fields in {source}"
            )
        if "fp_id" not in (dataset.dtype.names or ()) or "popcnt" not in (dataset.dtype.names or ()):
            raise RuntimeError(f"required fp_id/popcnt fields absent in {source}: {dataset.dtype}")
        if row_mask is not None and row_mask.shape != (int(dataset.shape[0]),):
            raise RuntimeError(f"invalid row mask shape {row_mask.shape} for {dataset.shape}")
        count = int(np.count_nonzero(row_mask)) if row_mask is not None else int(dataset.shape[0])
        fp_final = output / f"fp_u64x{word_count}.bin"
        id_final = output / "id_i64.bin"
        pc_final = output / "popcnt_u16.bin"
        fp_part = output / f"fp_u64x{word_count}.bin.part"
        id_part = output / "id_i64.bin.part"
        pc_part = output / "popcnt_u16.bin.part"
        fp = np.memmap(fp_part, mode="w+", dtype="<u8", shape=(count, word_count))
        ids = np.memmap(id_part, mode="w+", dtype="<i8", shape=(count,))
        pc = np.memmap(pc_part, mode="w+", dtype="<u2", shape=(count,))
        popcnt_lut = np.fromiter((int(value).bit_count() for value in range(256)), dtype=np.uint8, count=256)
        offset = 0
        # Bound temporary memory for the 2048-bit corpus while retaining large
        # sequential HDF5 reads for the 256-bit snapshots.
        chunk = 1_000_000 if word_count == 4 else 200_000
        for start in range(0, int(dataset.shape[0]), chunk):
            block = dataset[start : start + chunk]
            if row_mask is not None:
                block = block[row_mask[start : start + len(block)]]
            stop = offset + len(block)
            for word, name in enumerate(names):
                fp[offset:stop, word] = block[name]
            ids[offset:stop] = block["fp_id"]
            pc[offset:stop] = block["popcnt"]
            packed = np.empty((len(block), word_count), dtype="<u8")
            for word, name in enumerate(names):
                packed[:, word] = block[name]
            recomputed = popcnt_lut[packed.view(np.uint8)].reshape(len(block), word_count * 8).sum(axis=1)
            mismatch = np.flatnonzero(recomputed != block["popcnt"])
            if len(mismatch):
                row = start + int(mismatch[0])
                raise RuntimeError(
                    f"stored/recomputed popcount mismatch at source row {row} in {source}: "
                    f"stored={int(block['popcnt'][mismatch[0]])} recomputed={int(recomputed[mismatch[0]])}"
                )
            offset = stop
        if offset != count:
            raise RuntimeError(f"row-count mismatch {offset} != {count}")
        fp.flush(); ids.flush(); pc.flush()
        del fp, ids, pc
        atomic_replace(fp_part, fp_final)
        atomic_replace(id_part, id_final)
        atomic_replace(pc_part, pc_final)
    pc_view = np.memmap(pc_final, mode="r", dtype="<u2")
    sorted_pc = bool(len(pc_view) == 0 or np.all(pc_view[1:] >= pc_view[:-1]))
    if not sorted_pc:
        raise RuntimeError(f"source is not population-count sorted: {source}")
    return {
        "source": str(source),
        "rows": count,
        "word_count": word_count,
        "fingerprint_bits": word_count * 64,
        "population_count_sorted": sorted_pc,
        "fp": str(fp_final), "fp_sha256": sha256(fp_final),
        "ids": str(id_final), "ids_sha256": sha256(id_final),
        "popcnt": str(pc_final), "popcnt_sha256": sha256(pc_final),
    }


def symlink_checked(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"unexpected existing link {link} -> {link.resolve()}")
        return
    link.symlink_to(target)


def install_transition_layout(
    root: Path,
    transition: str,
    base: dict[str, object],
    union: dict[str, object],
    query_dir: Path,
) -> Path:
    if base["word_count"] != union["word_count"]:
        raise RuntimeError(f"width mismatch for {transition}: {base['word_count']} vs {union['word_count']}")
    word_count = int(base["word_count"])
    row_words = word_count + 2
    layout = root / "data" / "prepared" / "transition_roots" / transition
    gate0 = layout / "data" / "gate0_prepared"
    stage_a = layout / "data" / "stage_a"
    mapping = {
        gate0 / f"base_fp_u64x{word_count}.bin": Path(str(base["fp"])),
        gate0 / "base_id_i64.bin": Path(str(base["ids"])),
        gate0 / "base_popcnt_u16.bin": Path(str(base["popcnt"])),
        gate0 / f"union_fp_u64x{word_count}.bin": Path(str(union["fp"])),
        gate0 / "union_id_i64.bin": Path(str(union["ids"])),
        gate0 / "union_popcnt_u16.bin": Path(str(union["popcnt"])),
        stage_a / f"delta_u64x{row_words}.bin": query_dir / f"delta_u64x{row_words}.bin",
        stage_a / f"queries_u64x{row_words}.bin": query_dir / f"queries_u64x{row_words}.bin",
    }
    for link, target in mapping.items():
        symlink_checked(target, link)
    manifest = {
        "transition": transition,
        "word_count": word_count,
        "fingerprint_bits": word_count * 64,
        "row_words": row_words,
        "base": base,
        "union": union,
        "query_dir": str(query_dir),
        "links": {str(link): str(target) for link, target in mapping.items()},
    }
    manifest_path = layout / "LAYOUT_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return layout


def prepare_chembl(root: Path) -> tuple[dict[str, object], Path]:
    source = root / "data" / "official" / "chembl_37" / "chembl_37.h5"
    with h5py.File(source, "r") as handle:
        dataset = handle["fps"]
        names = fields(dataset.dtype)
        if len(names) != 32:
            raise RuntimeError(f"ChEMBL 37 must retain its 2048-bit contract, got fields: {names}")
        ids = dataset["fp_id"][:]
        if len(ids) <= 20760:
            raise RuntimeError("ChEMBL 37 too small for frozen synthetic partition")
        keys = np.fromiter(
            (int.from_bytes(hashlib.sha256(f"20260828:chembl37:{int(value)}".encode()).digest()[:8], "little") for value in ids),
            dtype=np.uint64,
            count=len(ids),
        )
        delta_rows = 20_760
        selected = np.argpartition(keys, delta_rows)[:delta_rows]
        selected = np.sort(selected)
        delta_mask = np.zeros(len(ids), dtype=bool)
        delta_mask[selected] = True
        selected_ids = ids[selected].copy()
        del ids, keys
    union = write_snapshot(source, root / "data" / "prepared" / "snapshots" / "chembl_37")
    base = write_snapshot(source, root / "data" / "prepared" / "snapshots" / "chembl_37_base", row_mask=~delta_mask)
    with h5py.File(source, "r") as handle:
        dataset = handle["fps"]
        delta = dataset[selected]
    names = fields(delta.dtype)
    row_words = len(names) + 2
    delta_u64 = np.empty((len(delta), row_words), dtype="<u8")
    delta_u64[:, 0] = delta["fp_id"].view("<u8")
    for word, name in enumerate(names):
        delta_u64[:, word + 1] = delta[name]
    delta_u64[:, -1] = delta["popcnt"].astype("<u8")
    query_order = sorted(
        range(len(delta)),
        key=lambda index: hashlib.sha256(f"20260828:chembl37-query:{int(delta['fp_id'][index])}".encode()).digest(),
    )[:512]
    queries = delta_u64[np.asarray(query_order, dtype=np.int64)]
    query_dir = root / "data" / "prepared" / "transition_queries" / "chembl37_synthetic"
    query_dir.mkdir(parents=True, exist_ok=True)
    delta_u64.tofile(query_dir / f"delta_u64x{row_words}.bin")
    queries.tofile(query_dir / f"queries_u64x{row_words}.bin")
    (query_dir / "partition.json").write_text(json.dumps({
        "scope": "synthetic update partition for cross-dataset query validation",
        "seed": "SHA256(20260828:chembl37:<fp_id>)",
        "selection": "20,760 smallest unsigned little-endian first-8-byte SHA256 keys",
        "delta_rows": len(delta),
        "base_rows": base["rows"],
        "union_rows": union["rows"],
        "query_rows": len(queries),
        "word_count": len(names),
        "fingerprint_bits": len(names) * 64,
        "row_words": row_words,
        "selected_id_sha256": hashlib.sha256(selected_ids.astype("<i8", copy=False).tobytes()).hexdigest(),
    }, indent=2) + "\n")
    layout = install_transition_layout(root, "chembl37_synthetic", base, union, query_dir)
    return {"union": union, "base": base, "delta_rows": len(delta)}, layout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--include-unvalidated-chembl",
        action="store_true",
        help="prepare the source-blocked ChEMBL corpus only for explicitly labeled exploratory use",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    begin = time.monotonic()
    snapshots: dict[str, dict[str, object]] = {}
    for date in DATES:
        source = root / "data" / "official" / "surechembl" / date / "fpsim2_fingerprints.h5"
        print(f"preparing {date}", flush=True)
        snapshots[date] = write_snapshot(source, root / "data" / "prepared" / "snapshots" / date)
    layouts = {}
    for base_date, union_date in zip(DATES, DATES[1:]):
        transition = f"{base_date}_to_{union_date}"
        query_dir = root / "data" / "prepared" / "transition_queries" / transition
        layouts[transition] = str(install_transition_layout(root, transition, snapshots[base_date], snapshots[union_date], query_dir))
    chembl: dict[str, object] | None = None
    if args.include_unvalidated_chembl:
        print("preparing UNVALIDATED chembl37 exploratory corpus", flush=True)
        chembl, chembl_layout = prepare_chembl(root)
        layouts["chembl37_synthetic"] = str(chembl_layout)
    result = {
        "experiment_id": "tide_20260828_gate6_flat_input_preparation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snapshots": snapshots,
        "chembl_37": chembl,
        "chembl_37_source_status": (
            "unvalidated_exploratory_included" if args.include_unvalidated_chembl
            else "blocked_and_skipped_per_SECOND_DATASET_BLOCKER_20260829"
        ),
        "layouts": layouts,
        "elapsed_s": time.monotonic() - begin,
    }
    output = root / "data" / "prepared" / "FLAT_INPUT_MANIFEST.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"snapshot_count": len(snapshots), "layouts": layouts, "elapsed_s": result["elapsed_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
