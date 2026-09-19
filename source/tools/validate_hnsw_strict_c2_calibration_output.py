#!/usr/bin/python3.12
"""Independent pure-stdlib validator for strict-C2 calibration-only output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import importlib.util


def _load_core():
    path = Path(__file__).resolve().with_name("strict_c2_pure_validator_core.py")
    spec = importlib.util.spec_from_file_location("strict_c2_pure_validator_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pure strict-C2 validator core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_core = _load_core()
Fail = _core.Fail
validate_partition = _core.validate_partition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pure-stdlib strict-C2 calibration output validator")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--calibration-seal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_partition(
            args.bundle, args.admission, args.output, args.summary, args.calibration_seal,
            "calibration", Path(__file__).resolve().parent.parent / "runner" /
            "fair_hnsw_strict_c2_calibration_runner.py",
            None,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except Fail as exc:
        print(f"strict_c2_calibration_validator FAIL: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"strict_c2_calibration_validator unexpected FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
