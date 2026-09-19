#!/usr/bin/env python3
"""Measure actual FPSim2 0.7.4 post-fingerprint correctness restoration."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import tables as tb

import FPSim2
from FPSim2 import FPSim2Engine
from FPSim2.io.backends.pytables import sort_db_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_transition(audit_path: Path, transition_name: str) -> dict[str, object]:
    audit = json.loads(audit_path.read_text())
    for transition in audit["transitions"]:
        if transition["transition"] == transition_name:
            return transition
    raise RuntimeError(f"transition not found: {transition_name}")


def structured_delta(path: Path, dtype: np.dtype) -> np.ndarray:
    names = list(dtype.names or ())
    fp_fields = sorted(
        [name for name in names if name.startswith("f") and name[1:].isdigit()],
        key=lambda name: int(name[1:]),
    )
    row_words = len(fp_fields) + 2
    packed = np.fromfile(path, dtype="<u8")
    if packed.size % row_words:
        raise RuntimeError(f"invalid delta size {packed.size} words for {row_words}-word rows")
    packed = packed.reshape(-1, row_words)
    rows = np.empty(len(packed), dtype=dtype)
    rows["fp_id"] = packed[:, 0].view("<i8")
    for word, field in enumerate(fp_fields):
        rows[field] = packed[:, word + 1]
    rows["popcnt"] = packed[:, -1].astype(dtype["popcnt"], copy=False)
    actual = np.zeros(len(rows), dtype=np.uint16)
    lut = np.fromiter((value.bit_count() for value in range(256)), dtype=np.uint8, count=256)
    fp_matrix = np.ascontiguousarray(packed[:, 1:-1], dtype="<u8")
    actual[:] = lut[fp_matrix.view(np.uint8)].reshape(len(rows), len(fp_fields) * 8).sum(axis=1)
    mismatch = np.flatnonzero(actual != rows["popcnt"])
    if len(mismatch):
        raise RuntimeError(f"delta popcount mismatch at row {int(mismatch[0])}")
    if len(rows) and np.any(rows["popcnt"][1:] < rows["popcnt"][:-1]):
        raise RuntimeError("delta is not population-count sorted")
    return rows


def copy_scratch(source: Path, target: Path) -> tuple[str, float]:
    if target.exists():
        raise RuntimeError(f"scratch target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    method = "copy2"
    try:
        subprocess.run(
            ["cp", "--reflink=always", "--sparse=always", str(source), str(target)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        method = "cp_reflink_always"
    except subprocess.CalledProcessError:
        shutil.copy2(source, target)
    return method, time.monotonic() - begin


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--transition", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-scratch", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    audit_path = root / "results" / "TRANSITION_REALITY.json"
    transition = load_transition(audit_path, args.transition)
    counts = transition["counts"]
    if not counts["insert_only_exact"]:
        raise RuntimeError(
            f"{args.transition} is not insert-only; current FPSim2 append lifecycle is ineligible"
        )

    base_path = root / "data" / "official" / "surechembl" / str(transition["base"]) / "fpsim2_fingerprints.h5"
    union_path = root / "data" / "official" / "surechembl" / str(transition["union"]) / "fpsim2_fingerprints.h5"
    delta_path = Path(str(transition["delta_u64x6_binary"]))
    scratch = root / "scratch" / "fpsim2_lifecycle" / args.run_id / f"{args.transition}.h5"

    with tb.open_file(base_path, mode="r") as handle:
        source_dtype = handle.root.fps.dtype
        base_rows = int(handle.root.fps.nrows)
    with tb.open_file(union_path, mode="r") as handle:
        union_rows = int(handle.root.fps.nrows)
        union_dtype = handle.root.fps.dtype
    if source_dtype != union_dtype:
        raise RuntimeError(f"base/union dtype mismatch: {source_dtype} vs {union_dtype}")
    delta = structured_delta(delta_path, source_dtype)
    if len(delta) != int(counts["added"]):
        raise RuntimeError(f"delta count mismatch {len(delta)} != {counts['added']}")
    if base_rows + len(delta) != union_rows:
        raise RuntimeError(f"row-balance mismatch {base_rows}+{len(delta)}!={union_rows}")

    copy_method, copy_s = copy_scratch(base_path, scratch)
    copy_sha = sha256(scratch)
    base_sha = sha256(base_path)
    if copy_sha != base_sha:
        raise RuntimeError("scratch copy SHA-256 mismatch before timing")

    append_begin = time.monotonic()
    with tb.open_file(scratch, mode="a") as handle:
        handle.root.fps.append(delta)
        handle.root.fps.flush()
    append_s = time.monotonic() - append_begin

    index_begin = time.monotonic()
    with tb.open_file(scratch, mode="a") as handle:
        column = handle.root.fps.cols.popcnt
        if column.is_indexed:
            column.remove_index()
        # FPSim2's sort_db_file delegates to PyTables copy(sortby="popcnt").
        # PyTables requires a completely sorted index (CSI) for a globally
        # sorted result.  A default full index is only optlevel=6 and can emit
        # separately sorted slices at this scale.
        column.create_csindex()
        index_is_csi = bool(column.index.is_csi)
        if not index_is_csi:
            raise RuntimeError("population-count index is not a CSI")
        handle.root.fps.flush()
    csi_creation_s = time.monotonic() - index_begin

    sort_begin = time.monotonic()
    sort_db_file(str(scratch))
    sort_s = time.monotonic() - sort_begin

    reload_begin = time.monotonic()
    engine = FPSim2Engine(
        str(scratch), storage_backend="pytables", in_memory_fps=True, fps_sort=False
    )
    reload_s = time.monotonic() - reload_begin
    lifecycle_s = append_s + csi_creation_s + sort_s + reload_s

    fps = engine.fps
    rows_after = int(fps.shape[0])
    popcnt_sorted = bool(rows_after == 0 or np.all(fps[1:, -1] >= fps[:-1, -1]))
    bin_end = max(int(pair[1][1]) for pair in engine.popcnt_bins)
    config = {
        "fp_type": engine.fp_type,
        "fp_params": engine.fp_params,
        "source_rdkit_version": engine.rdkit_ver,
        "source_fpsim2_version": engine.fpsim2_ver,
        "runtime_fpsim2_version": getattr(FPSim2, "__version__", "unknown"),
    }
    query_ready_component_pass = bool(
        rows_after == union_rows and popcnt_sorted and bin_end == union_rows
    )
    result = {
        "experiment_id": "tide_20260828_gate6_fpsim2_lifecycle",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "run_id": args.run_id,
        "transition": args.transition,
        "scope": (
            "post-fingerprint local artifact append + complete popcount CSI creation + "
            "global sort/bin repair + actual engine reload"
        ),
        "excluded_scope": "source scratch copy, network, parsing, fingerprint generation",
        "base_path": str(base_path),
        "base_sha256": base_sha,
        "union_path": str(union_path),
        "union_sha256": sha256(union_path),
        "delta_path": str(delta_path),
        "delta_sha256": sha256(delta_path),
        "base_rows": base_rows,
        "delta_rows": len(delta),
        "union_rows": union_rows,
        "copy": {"method": copy_method, "seconds_excluded": copy_s},
        "timed": {
            "append_precomputed_fingerprints_s": append_s,
            "complete_popcount_csi_creation_s": csi_creation_s,
            "global_sort_and_bin_repair_s": sort_s,
            "actual_engine_reload_s": reload_s,
            "lifecycle_total_s": lifecycle_s,
        },
        "pre_sort_index": {
            "kind": "full",
            "optlevel": 9,
            "is_csi": index_is_csi,
        },
        "engine": config,
        "rows_after": rows_after,
        "population_count_sorted_after": popcnt_sorted,
        "last_bin_end": bin_end,
        "query_ready_component_pass": query_ready_component_pass,
        "complete_result_equality_gate": "pending separate exact-query campaign",
        "scratch_path": str(scratch),
        "scratch_cleanup": "kept" if args.keep_scratch else "pending_cleanup_on_pass",
    }
    del fps, engine, delta
    gc.collect()
    if query_ready_component_pass and not args.keep_scratch:
        os.unlink(scratch)
        parent = scratch.parent
        try:
            parent.rmdir()
        except OSError:
            pass
        result["scratch_cleanup"] = "removed_exact_owned_scratch_after_component_pass"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if query_ready_component_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
