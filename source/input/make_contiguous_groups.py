#!/usr/bin/env python3
"""Create deterministic contiguous group memberships for resource smoke tests."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=int)
    parser.add_argument("group_size", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.rows <= 0 or args.group_size <= 0:
        raise SystemExit("rows and group_size must be positive")
    if args.output.exists():
        raise SystemExit(f"refusing to replace existing output: {args.output}")
    with args.output.open("x") as output:
        for start in range(0, args.rows, args.group_size):
            stop = min(args.rows, start + args.group_size)
            output.write(" ".join(map(str, range(start, stop))))
            output.write("\n")


if __name__ == "__main__":
    main()
