#!/usr/bin/env python3
"""Probe FPSim2 append visibility without mutating an upstream fixture."""

import argparse
import json
import shutil
from pathlib import Path

import tables as tb

from FPSim2 import FPSim2Engine
from FPSim2.io.backends.pytables import create_db_file, sort_db_file


def state(path: Path, label: str) -> dict:
    with tb.open_file(path, "r") as handle:
        rows = handle.root.fps[:]
        bins = list(handle.root.config[4]) if len(handle.root.config) > 4 else None
    return {
        "label": label,
        "rows": int(rows.shape[0]),
        "table_popcnt_sequence": [int(value) for value in rows["popcnt"]],
        "stored_bins": (
            [
                [int(popcount), [int(begin), int(end)]]
                for popcount, (begin, end) in bins
            ]
            if bins is not None
            else None
        ),
    }


def result_ids(engine: FPSim2Engine, query: str, threshold: float) -> list[int]:
    return [int(value) for value in engine.similarity(query, threshold=threshold)["mol_id"]]


def query_matrix(path: Path, fps_sort: bool) -> dict:
    engine = FPSim2Engine(str(path), in_memory_fps=True, fps_sort=fps_sort)
    return {
        query: {
            "threshold_1.0": result_ids(engine, query, 1.0),
            "threshold_0.7": result_ids(engine, query, 0.7),
        }
        for query in ("CC", "CCC", "CCCC")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--work-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.work_file.parent.mkdir(parents=True, exist_ok=True)
    if args.fixture.suffix == ".smi":
        create_db_file(
            str(args.fixture),
            str(args.work_file),
            mol_format="smiles",
            fp_type="Morgan",
            fp_params={"radius": 2, "fpSize": 256},
            sort_by_popcnt=True,
        )
    else:
        shutil.copy2(args.fixture, args.work_file)

    before = state(args.work_file, "before")
    writer = FPSim2Engine(
        str(args.work_file), in_memory_fps=False, storage_backend="pytables"
    )
    writer.storage.append_fps(
        [["CC", 11], ["CCC", 12], ["CCCC", 13]], mol_format="smiles"
    )

    after_append = state(args.work_file, "after_append_without_sort")
    query_after_append = {
        "fps_sort_false": query_matrix(args.work_file, fps_sort=False),
        "fps_sort_true": query_matrix(args.work_file, fps_sort=True),
    }

    sort_db_file(str(args.work_file))
    after_disk_sort = state(args.work_file, "after_sort_db_file")
    query_after_disk_sort = query_matrix(args.work_file, fps_sort=False)

    output = {
        "before": before,
        "after_append": after_append,
        "queries_after_append": query_after_append,
        "after_disk_sort": after_disk_sort,
        "queries_after_disk_sort": query_after_disk_sort,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
