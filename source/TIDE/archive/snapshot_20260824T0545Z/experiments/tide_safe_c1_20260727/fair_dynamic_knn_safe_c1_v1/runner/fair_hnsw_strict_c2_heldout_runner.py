#!/usr/bin/env python3
"""Strict-C2 locked held-out HNSW runner entrypoint.

There is intentionally no --ef-search / tuning argument.  The only EF value is
read from a self-hashed calibration-selection lock receipt and then rechecked by
the shared core against the held-out seal and immutable input bundle.
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
LOCK_SCHEMA = _core.LOCK_SCHEMA
RunArgs = _core.RunArgs
read_json_object = _core.read_json_object
require_uint = _core.require_uint
run_partition = _core.run_partition


def locked_ef(path: Path) -> int:
    lock = read_json_object(path, "held-out lock receipt")
    if lock.get("schema") != LOCK_SCHEMA or lock.get("status") != "LOCKED":
        raise Fail("held-out entrypoint refuses an unsealed lock receipt")
    return require_uint(lock.get("locked_ef_search"), "lock locked_ef_search")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict-C2 locked held-out CPU HNSW quality runner")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--heldout-seal", required=True, type=Path)
    parser.add_argument("--lock-receipt", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        # The lock is read only to obtain its immutable EF; no calibration result
        # or tuning input is accepted by this entrypoint.
        return run_partition(RunArgs(
            bundle=args.bundle, preflight=args.preflight, out=args.out, summary=args.summary,
            runtime_root=args.runtime_root, partition_seal=args.heldout_seal,
            ef_search=locked_ef(args.lock_receipt), partition="held_out", entrypoint=Path(__file__),
            lock_receipt=args.lock_receipt,
        ))
    except Fail as exc:
        print(f"strict_c2_heldout_runner fail-stop: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"strict_c2_heldout_runner unexpected fail-stop: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
