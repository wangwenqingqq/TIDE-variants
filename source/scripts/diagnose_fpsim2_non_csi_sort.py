#!/usr/bin/env python3
"""Read-only diagnostic for retained FPSim2/PyTables non-CSI sort output."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import time
from pathlib import Path

import numpy as np
import tables as tb

import FPSim2
from FPSim2.io.backends.pytables import sort_db_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    args = parser.parse_args()

    input_path = args.input.resolve()
    inversions: list[dict[str, object]] = []
    index_metadata: dict[str, object]
    neighborhoods: dict[str, list[int]] = {}
    minimum: int | None = None
    maximum: int | None = None
    previous: int | None = None
    rows = 0
    with tb.open_file(input_path, "r") as handle:
        table = handle.root.fps
        rows = int(table.nrows)
        column = table.cols.popcnt
        if not column.is_indexed:
            raise RuntimeError("retained failed output unexpectedly has no popcnt index")
        index = column.index
        index_metadata = {
            "is_indexed": True,
            "kind": str(index.kind),
            "optlevel": int(index.optlevel),
            "is_csi": bool(index.is_csi),
            "nelements": int(index.nelements),
            "slicesize": int(index.slicesize),
            "blocksize": int(index.blocksize),
            "superblocksize": int(index.superblocksize),
            "chunksize": int(index.chunksize),
            "index_rows": int(index.nrows),
            "table_chunkshape": [int(value) for value in table.chunkshape],
        }
        for start in range(0, rows, args.chunk_rows):
            values = column[start : min(rows, start + args.chunk_rows)].astype(
                np.int64, copy=False
            )
            if not len(values):
                continue
            block_min = int(values.min())
            block_max = int(values.max())
            minimum = block_min if minimum is None else min(minimum, block_min)
            maximum = block_max if maximum is None else max(maximum, block_max)
            if previous is not None and int(values[0]) < previous:
                inversions.append(
                    {
                        "row": start,
                        "left": previous,
                        "right": int(values[0]),
                    }
                )
            positions = np.flatnonzero(values[1:] < values[:-1])
            for position_value in positions:
                position = int(position_value)
                inversions.append(
                    {
                        "row": start + position + 1,
                        "left": int(values[position]),
                        "right": int(values[position + 1]),
                    }
                )
            previous = int(values[-1])
        for inversion in inversions:
            row = int(inversion["row"])
            lo = max(0, row - 8)
            hi = min(rows, row + 8)
            neighborhoods[str(row)] = [int(value) for value in column[lo:hi]]

    fpsim2_source = Path(inspect.getsourcefile(sort_db_file) or "")
    result = {
        "experiment_id": "tide_20260829_gate6_fpsim2_non_csi_diagnostic",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "read-only retained failed scratch diagnostic; not a timing run",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "rows": rows,
        "popcount_min": minimum,
        "popcount_max": maximum,
        "population_count_sorted": len(inversions) == 0,
        "inversion_count": len(inversions),
        "inversions": inversions,
        "neighborhoods": neighborhoods,
        "index": index_metadata,
        "runtime": {
            "python": platform.python_version(),
            "pytables": tb.__version__,
            "fpsim2": getattr(FPSim2, "__version__", "unknown"),
            "fpsim2_pytables_source": str(fpsim2_source),
            "fpsim2_pytables_source_sha256": sha256(fpsim2_source),
        },
        "interpretation_boundary": (
            "The diagnostic proves the retained output and index state only. "
            "The causal CSI correction is separately tested by probe05 and trace02."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
