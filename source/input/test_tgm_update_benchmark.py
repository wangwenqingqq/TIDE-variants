#!/usr/bin/env python3
"""Fixture for resettable LES3 group insertion and first-query visibility."""

from __future__ import annotations

import csv
import pathlib
import struct
import subprocess
import sys
import tempfile


def pack_row(identifier: int, tokens: set[int]) -> bytes:
    words = [0, 0, 0, 0]
    for token in tokens:
        words[token // 64] |= 1 << (token % 64)
    return struct.pack("<6Q", identifier, *words, len(tokens))


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: test_tgm_update_benchmark.py SOURCE CXX", file=sys.stderr)
        return 2
    source = pathlib.Path(sys.argv[1]).resolve()
    compiler = sys.argv[2]
    base = [(10, {0, 1}), (20, {1, 2}), (30, {4, 5}), (40, {5, 6})]
    with tempfile.TemporaryDirectory(prefix="les3_update_") as temp_string:
        temp = pathlib.Path(temp_string)
        binary = temp / "tgm_update_benchmark"
        subprocess.run([compiler, "-std=c++20", "-O2", str(source), "-o", str(binary)], check=True)
        fp = temp / "fp.bin"
        ids = temp / "ids.bin"
        pop = temp / "pop.bin"
        with fp.open("wb") as fp_output, ids.open("wb") as id_output, pop.open("wb") as pop_output:
            for identifier, tokens in base:
                packed = pack_row(identifier, tokens)
                _, *words, count = struct.unpack("<6Q", packed)
                fp_output.write(struct.pack("<4Q", *words))
                id_output.write(struct.pack("<Q", identifier))
                pop_output.write(struct.pack("<H", count))
        order = temp / "order.bin"
        order.write_bytes(struct.pack("<4I", 0, 1, 2, 3))
        offsets = temp / "offsets.bin"
        offsets.write_bytes(struct.pack("<3Q", 0, 2, 4))
        groups = temp / "groups.tsv"
        groups.write_text(
            "group\trows\tunion_popcount\tword0\tword1\tword2\tword3\n"
            "0\t2\t3\t7\t0\t0\t0\n"
            "1\t2\t3\t70\t0\t0\t0\n"
        )
        delta = temp / "delta.bin"
        delta.write_bytes(pack_row(99, {0, 1, 2}))
        queries = temp / "queries.bin"
        queries.write_bytes(pack_row(99, {0, 1, 2}))
        output = temp / "results.csv"
        completed = subprocess.run(
            [str(binary), str(fp), str(ids), str(pop), str(order), str(offsets),
             str(groups), str(delta), str(queries), str(output), "--base-rows", "4",
             "--query-index", "0", "--repetitions", "3", "--warmup", "1",
             "--threshold", "7/10"],
            text=True, capture_output=True, check=True,
        )
        print(completed.stdout, end="")
        records = list(csv.DictReader(output.open()))
        assert len(records) == 3
        assert {row["delta_rows"] for row in records} == {"1"}
        assert {row["groups_receiving_rows"] for row in records} == {"1"}
        assert {row["hits"] for row in records} == {"1"}
        assert {row["visible_query_id_count"] for row in records} == {"1"}
        assert {row["duplicate_id"] for row in records} == {"0"}
        for row in records:
            assert abs(
                float(row["update_to_visible_seconds"])
                - float(row["update_seconds"])
                - float(row["query_materialize_seconds"])
            ) < 1e-12
        print("ALL_TGM_UPDATE_GATES_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
