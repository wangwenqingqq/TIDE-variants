#!/usr/bin/env python3
"""Measure the post-fingerprint FPSim2 append/sort/reload lifecycle."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tables as tb

from FPSim2 import FPSim2Engine
from FPSim2.io.backends.pytables import sort_db_file


def table_state(path: Path) -> dict:
    with tb.open_file(path, "r") as handle:
        table = handle.root.fps
        rows = int(table.nrows)
        first_popcount = int(table[0]["popcnt"])
        last_popcount = int(table[-1]["popcnt"])
        bins = handle.root.config[4]
        column = table.cols.popcnt
        is_indexed = bool(column.is_indexed)
        index = column.index if is_indexed else None
        index_kind = getattr(index, "kind", None)
        index_is_csi = bool(getattr(index, "is_csi", False))
        index_dirty = bool(getattr(index, "dirty", False))
    return {
        "rows": rows,
        "first_popcount": first_popcount,
        "last_popcount": last_popcount,
        "bin_count": len(bins),
        "popcnt_index": {
            "is_indexed": is_indexed,
            "kind": index_kind,
            "is_csi": index_is_csi,
            "dirty": index_dirty,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-file", type=Path, required=True)
    parser.add_argument("--delta-u64x6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    before = table_state(args.work_file)
    delta = np.memmap(args.delta_u64x6, dtype="<u8", mode="r").reshape(-1, 6)

    start = time.perf_counter_ns()
    with tb.open_file(args.work_file, "a") as handle:
        table = handle.root.fps
        rows = np.empty(delta.shape[0], dtype=table.dtype)
        rows["fp_id"] = delta[:, 0].view("<i8")
        for index in range(4):
            rows[f"f{index + 1}"] = delta[:, index + 1]
        rows["popcnt"] = delta[:, 5].view("<i8")
        table.append(rows)
        table.flush()
    append_ms = (time.perf_counter_ns() - start) / 1e6
    after_append = table_state(args.work_file)

    start = time.perf_counter_ns()
    with tb.open_file(args.work_file, "a") as handle:
        column = handle.root.fps.cols.popcnt
        if column.is_indexed:
            index_action = "reindex"
            column.reindex()
        else:
            index_action = "create_index_full"
            column.create_index(kind="full")
        handle.root.fps.flush()
    index_repair_ms = (time.perf_counter_ns() - start) / 1e6
    after_index_repair = table_state(args.work_file)
    if not after_index_repair["popcnt_index"]["is_csi"]:
        raise RuntimeError("popcnt index repair did not produce a CSI index")

    start = time.perf_counter_ns()
    sort_db_file(str(args.work_file))
    sort_ms = (time.perf_counter_ns() - start) / 1e6
    after_sort = table_state(args.work_file)

    start = time.perf_counter_ns()
    engine = FPSim2Engine(str(args.work_file), in_memory_fps=True, fps_sort=False)
    cpu_reload_ms = (time.perf_counter_ns() - start) / 1e6

    output = {
        "scope": (
            "local post-fingerprint artifact lifecycle: append precomputed rows, "
            "full sort/bin repair, and CPU in-memory engine reload"
        ),
        "before": before,
        "delta_rows": int(delta.shape[0]),
        "after_append": after_append,
        "index_action": index_action,
        "after_index_repair": after_index_repair,
        "after_sort": after_sort,
        "append_ms": append_ms,
        "index_repair_ms": index_repair_ms,
        "sort_ms": sort_ms,
        "cpu_reload_ms": cpu_reload_ms,
        "total_update_to_cpu_visible_ms": (
            append_ms + index_repair_ms + sort_ms + cpu_reload_ms
        ),
        "engine_rows": int(engine.fps.shape[0]),
        "checks": {
            "append_row_count": after_append["rows"]
            == before["rows"] + int(delta.shape[0]),
            "sort_row_count": after_sort["rows"] == after_append["rows"],
            "engine_row_count": int(engine.fps.shape[0]) == after_sort["rows"],
            "sorted_boundary": after_sort["first_popcount"]
            <= after_sort["last_popcount"],
        },
    }
    if not all(output["checks"].values()):
        raise RuntimeError(json.dumps(output, indent=2))
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
