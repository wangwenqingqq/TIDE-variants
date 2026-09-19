#!/usr/bin/env python3
"""Fixture for the independent exact transition oracle."""

from __future__ import annotations

import csv
import pathlib
import struct
import subprocess
import sys
import tempfile


def pack_words(tokens: set[int]) -> tuple[int, int, int, int]:
    words = [0, 0, 0, 0]
    for token in tokens:
        words[token // 64] |= 1 << (token % 64)
    return tuple(words)


def fnv(ids: list[int]) -> int:
    value = 1469598103934665603
    for identifier in ids:
        for byte in struct.pack("<Q", identifier):
            value ^= byte
            value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: test_exact_transition_oracle.py SOURCE CXX", file=sys.stderr)
        return 2
    source = pathlib.Path(sys.argv[1]).resolve()
    compiler = sys.argv[2]
    rows = [
        (10, {0, 1}),
        (20, {0, 1, 2}),
        (30, {0, 1, 2, 3}),
        (40, {4, 5, 6, 7}),
    ]
    rows.sort(key=lambda item: len(item[1]))
    with tempfile.TemporaryDirectory(prefix="les3_transition_oracle_") as temp_string:
        root = pathlib.Path(temp_string)
        prepared = root / "data" / "gate0_prepared"
        stage_a = root / "data" / "stage_a"
        prepared.mkdir(parents=True)
        stage_a.mkdir(parents=True)
        binary = root / "exact_transition_oracle"
        subprocess.run(
            [compiler, "-std=c++20", "-O2", str(source), "-o", str(binary)],
            check=True,
        )
        with (prepared / "union_fp_u64x4.bin").open("wb") as fp_out, (
            prepared / "union_id_i64.bin"
        ).open("wb") as id_out, (prepared / "union_popcnt_u16.bin").open("wb") as pop_out:
            for identifier, tokens in rows:
                fp_out.write(struct.pack("<4Q", *pack_words(tokens)))
                id_out.write(struct.pack("<Q", identifier))
                pop_out.write(struct.pack("<H", len(tokens)))
        query_tokens = {0, 1, 2, 3}
        (stage_a / "queries_u64x6.bin").write_bytes(
            struct.pack("<6Q", 30, *pack_words(query_tokens), len(query_tokens))
        )
        output = root / "oracle.csv"
        completed = subprocess.run(
            [str(binary), str(root), str(output), "1"],
            text=True,
            capture_output=True,
            check=True,
        )
        print(completed.stdout, end="")
        records = list(csv.DictReader(output.open()))
        assert len(records) == 2
        expected = {(7, 10): [20, 30], (4, 5): [30]}
        for record in records:
            threshold = (int(record["threshold_num"]), int(record["threshold_den"]))
            identifiers = sorted(expected[threshold])
            assert int(record["hits"]) == len(identifiers)
            assert int(record["id_hash"]) == fnv(identifiers)
            assert record["duplicate_id"] == "0"
        print("ALL_EXACT_TRANSITION_ORACLE_GATES_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
