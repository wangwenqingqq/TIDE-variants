#!/usr/bin/env python3
"""Extract a normalized per-kernel SASS ledger for Gate-4 binaries."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


KERNELS = (
    "exact_segment_scan_kernel",
    "released_taniRAW",
    "collect_released_kernel",
)


def parse(path: Path) -> dict[str, list[str]]:
    output = {kernel: [] for kernel in KERNELS}
    active = None
    for line in path.read_text().splitlines():
        if "Function :" in line:
            active = next((kernel for kernel in KERNELS if kernel in line), None)
            continue
        if active is None:
            continue
        match = re.search(r"/\*[0-9a-f]+\*/\s+(.*?)\s*/\*", line)
        if match:
            output[active].append(" ".join(match.group(1).split()))
    return output


def summarize(instructions: list[str]) -> dict:
    opcodes = collections.Counter()
    for instruction in instructions:
        text = instruction
        if text.startswith("@"):
            text = text.split(maxsplit=1)[1]
        opcode = text.split(maxsplit=1)[0].rstrip(";")
        opcodes[opcode] += 1
    families = {
        "POPC": sum(count for opcode, count in opcodes.items() if opcode.startswith("POPC")),
        "LDG": sum(count for opcode, count in opcodes.items() if opcode.startswith("LDG")),
        "STG": sum(count for opcode, count in opcodes.items() if opcode.startswith("STG")),
        "ATOM_or_RED": sum(
            count
            for opcode, count in opcodes.items()
            if opcode.startswith("ATOM") or opcode.startswith("RED")
        ),
        "BAR": sum(count for opcode, count in opcodes.items() if opcode.startswith("BAR")),
    }
    return {
        "instruction_count": len(instructions),
        "families": families,
        "opcodes": dict(sorted(opcodes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-sass", type=Path, required=True)
    parser.add_argument("--sustained-sass", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    timing = parse(args.timing_sass)
    sustained = parse(args.sustained_sass)
    result = {
        "timing": {kernel: summarize(timing[kernel]) for kernel in KERNELS},
        "sustained": {
            kernel: summarize(sustained[kernel]) for kernel in KERNELS
        },
        "normalized_instruction_sequences_identical": {
            kernel: timing[kernel] == sustained[kernel] for kernel in KERNELS
        },
    }
    result["all_kernel_sequences_identical"] = all(
        result["normalized_instruction_sequences_identical"].values()
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
