#!/usr/bin/env python3
"""Strict-C2 calibration-only HNSW runner entrypoint.

The only query partition accepted here is the presealed calibration partition.
All trace inserts are replayed; non-calibration KNNs are skipped without a query
or oracle call by the shared core.  This command has no timing mode.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import importlib.util


def _load_core():
    path = Path(__file__).resolve().with_name("strict_c2_hnsw_core.py")
    spec = importlib.util.spec_from_file_location("strict_c2_hnsw_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load root-private strict-C2 runner core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_core = _load_core()
Fail = _core.Fail
RunArgs = _core.RunArgs
run_partition = _core.run_partition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict-C2 calibration-only CPU HNSW quality runner")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--calibration-seal", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--ef-search", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        return run_partition(RunArgs(
            bundle=args.bundle, preflight=args.preflight, out=args.out, summary=args.summary,
            runtime_root=args.runtime_root, partition_seal=args.calibration_seal,
            ef_search=args.ef_search, partition="calibration", entrypoint=Path(__file__),
            lock_receipt=None,
        ))
    except Fail as exc:
        print(f"strict_c2_calibration_runner fail-stop: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # fail-stop if a library/runtime error escapes
        print(f"strict_c2_calibration_runner unexpected fail-stop: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
