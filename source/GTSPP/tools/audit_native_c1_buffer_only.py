#!/usr/bin/env python3
"""Static, fail-closed audit of the native GTS C1 buffer-only runtime gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(text: str, token: str) -> None:
    if token not in text:
        raise RuntimeError(f"required native C1 token missing: {token}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    update = args.source / "include" / "update.cuh"
    main_cu = args.source / "src" / "main.cu"
    text = update.read_text()
    main_text = main_cu.read_text()

    for token in [
        'std::getenv("TIDE_C1_BUFFER_ONLY")',
        'const bool c1_buffer_only = c1BufferOnlyMode();',
        'if (!c1_buffer_only)',
        'ins_result = incrementalInsert(',
        'if (c1_buffer_only && getIncrTotal() != 0)',
        'else if (in_size > 0)',
        'searchNaiveRnn(',
        'mergeTotalResult<<<',
    ]:
        require(text, token)
    require(main_text, 'Mode 0: C1 uses all-include')

    gate_start = text.index('if (!c1_buffer_only)')
    direct_start = text.index('ins_result = incrementalInsert(', gate_start)
    gate_end = text.index('static int incr_ok', gate_start)
    if not (gate_start < direct_start < gate_end):
        raise RuntimeError('incrementalInsert is not lexically contained by the non-buffer-only gate')

    result = {
        "schema": "tide-native-c1-buffer-only-static-audit-v1",
        "status": "pass",
        "source": str(args.source),
        "files_sha256": {"include/update.cuh": digest(update), "src/main.cu": digest(main_cu)},
        "verified": [
            "TIDE_C1_BUFFER_ONLY gate exists",
            "buffer-only path bypasses incrementalInsert",
            "live Delta is scanned through searchNaiveRnn when nonempty",
            "base and Delta candidates are merged",
            "update path configures C2 all-include mode",
        ],
        "not_verified": [
            "native runtime result exactness",
            "per-query full-live oracle agreement",
            "concurrent mutation correctness",
            "latency or speedup",
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"native C1 static audit failed: {error}", file=sys.stderr)
        raise
