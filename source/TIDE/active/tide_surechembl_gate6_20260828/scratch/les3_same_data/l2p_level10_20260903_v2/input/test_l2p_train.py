#!/usr/bin/env python3
"""End-to-end deterministic coverage test for the scalable L2P driver."""

from __future__ import annotations

import hashlib
import math
import os
import pathlib
import random
import shlex
import struct
import subprocess
import sys
import tempfile


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dense_fixture(fp: pathlib.Path, pop: pathlib.Path, rows: int) -> None:
    randomizer = random.Random(314159)
    with fp.open("wb") as fp_output, pop.open("wb") as pop_output:
        for row in range(rows):
            center = (row % 4) * 64
            tokens = {center + randomizer.randrange(64) for _ in range(12)}
            tokens.update({randomizer.randrange(256) for _ in range(4)})
            words = [0, 0, 0, 0]
            for token in sorted(tokens):
                words[token // 64] |= 1 << (token % 64)
            count = sum(bin(word).count("1") for word in words)
            fp_output.write(struct.pack("<4Q", *words))
            pop_output.write(struct.pack("<H", count))


def read_u32(path: pathlib.Path) -> list[int]:
    payload = path.read_bytes()
    return list(struct.unpack(f"<{len(payload) // 4}I", payload))


def read_u64(path: pathlib.Path) -> list[int]:
    payload = path.read_bytes()
    return list(struct.unpack(f"<{len(payload) // 8}Q", payload))


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: test_l2p_train.py BUILD_PTR_CPP TRAIN_CPP CXX", file=sys.stderr)
        return 2
    build_source = pathlib.Path(sys.argv[1]).resolve()
    train_source = pathlib.Path(sys.argv[2]).resolve()
    compiler = sys.argv[3]
    compile_flags = shlex.split(os.environ.get("CXXFLAGS", "-std=c++20 -O2"))
    rows = 400
    with tempfile.TemporaryDirectory(prefix="les3_l2p_train_") as temp_string:
        temp = pathlib.Path(temp_string)
        build_binary = temp / "l2p_build_ptr"
        train_binary = temp / "l2p_train"
        subprocess.run([compiler, *compile_flags, str(build_source), "-o", str(build_binary)], check=True)
        subprocess.run([compiler, *compile_flags, "-pthread", str(train_source), "-o", str(train_binary)], check=True)
        fp = temp / "fixture.fp"
        pop = temp / "fixture.pop"
        write_dense_fixture(fp, pop, rows)
        representation = temp / "representation" / "fixture"
        representation.parent.mkdir()
        subprocess.run(
            [str(build_binary), str(fp), str(pop), str(representation), "--expected-rows", str(rows)],
            check=True,
        )

        output_hashes: list[tuple[str, str, str]] = []
        for run, workers in (("a", "1"), ("b", "2")):
            prefix = temp / run / "partition"
            completed = subprocess.run(
                [
                    str(train_binary),
                    str(fp),
                    str(representation) + ".ptr_u8x16.bin",
                    str(representation) + ".min_token_order_u32.bin",
                    str(representation) + ".ptr_meta.txt",
                    str(prefix),
                    "--rows", str(rows),
                    "--initial-level", "2",
                    "--initial-groups", "4",
                    "--target-level", "3",
                    "--emit-from-level", "3",
                    "--seed", "20260902",
                    "--pairs-per-model", "200",
                    "--epochs", "3",
                    "--batch-size", "64",
                    "--workers", workers,
                    "--stop-below", "50",
                    "--write-text",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            print(completed.stdout, end="")
            stem = pathlib.Path(str(prefix) + ".level3")
            order_path = pathlib.Path(str(stem) + ".order_u32.bin")
            offsets_path = pathlib.Path(str(stem) + ".offsets_u64.bin")
            groups_path = pathlib.Path(str(stem) + ".groups.txt")
            order = read_u32(order_path)
            offsets = read_u64(offsets_path)
            assert sorted(order) == list(range(rows))
            assert offsets[0] == 0 and offsets[-1] == rows
            assert all(left < right for left, right in zip(offsets, offsets[1:]))
            text_groups = [list(map(int, line.split())) for line in groups_path.read_text().splitlines()]
            assert [value for group in text_groups for value in group] == order
            assert [len(group) for group in text_groups] == [
                right - left for left, right in zip(offsets, offsets[1:])
            ]
            metrics_path = pathlib.Path(str(stem) + ".metrics.tsv")
            metrics = metrics_path.read_text()
            assert "nan" not in metrics.lower()
            output_hashes.append((sha256(order_path), sha256(offsets_path), sha256(groups_path)))
        assert output_hashes[0] == output_hashes[1]
        final_offsets = read_u64(
            pathlib.Path(str(temp / "a" / "partition.level3.offsets_u64.bin"))
        )
        print(
            "ALL_L2P_TRAIN_GATES_PASS "
            f"rows={rows} groups={len(final_offsets) - 1} "
            f"order_sha256={output_hashes[0][0]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
