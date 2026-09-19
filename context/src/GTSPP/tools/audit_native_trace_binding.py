#!/usr/bin/env python3
"""Expose whether a native GTS update trace has a valid base/payload binding."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(text: str, token: str, path: pathlib.Path) -> None:
    if token not in text:
        raise RuntimeError(f"expected token missing in {path.name}: {token}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    tree = args.source / "include" / "tree.cuh"
    incr = args.source / "include" / "incremental_insert.cuh"
    update = args.source / "include" / "update.cuh"
    tree_text, incr_text, update_text = (p.read_text() for p in (tree, incr, update))

    # These exact source facts show that data_info[1] currently controls initial
    # index construction, while insert IDs address the same loaded payload array.
    need(tree_text, 'CHECK(cudaMalloc((void **)&id_list, data_info[1] * sizeof(int)))', tree)
    need(tree_text, 'initIndexData<<<(data_info[1] - 1)', tree)
    need(incr_text, 'data_d[data_id * data_info[0] + j]', incr)
    need(update_text, 'int ins_data_id = update_list[i].update_id;', update)

    finding = {
        "schema": "tide-native-trace-binding-audit-v1",
        "status": "needs_adapter_or_manifest",
        "source_files_sha256": {
            "include/tree.cuh": sha256(tree),
            "include/incremental_insert.cuh": sha256(incr),
            "include/update.cuh": sha256(update),
        },
        "finding": (
            "The current native path uses data_info[1] to populate the initial tree and "
            "uses update_id as an address into that same loaded payload array. A generic "
            "fresh-insert trace therefore has no demonstrated separation between backing "
            "payload rows and the initial active Base."
        ),
        "safe_interpretation": (
            "An old native update trace may be treated as a reactivation/legacy trace only "
            "if it supplies a stable-ID-to-row mapping and proves that every inserted ID was "
            "not active immediately before its insert. It is not automatically evidence for "
            "generic fresh insertion."
        ),
        "required_before_native_runtime_claim": [
            "frozen Base active-ID manifest",
            "backing payload-row manifest and hash",
            "stable-ID to native logical-position mapping receipt",
            "operation trace with pre/post live-set checks",
            "independent full-live oracle comparison after every query",
        ],
    }
    args.output.write_text(json.dumps(finding, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": finding["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"native trace binding audit failed: {exc}", file=sys.stderr)
        raise
