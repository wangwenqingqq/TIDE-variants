#!/usr/bin/env python3
"""Validate a sealed E1 lifecycle trace and raw integer-vector catalog."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tide_protocol_lib import ProtocolError, load_and_validate_trace, print_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    args = parser.parse_args()
    try:
        catalog, trace = load_and_validate_trace(args.trace, args.catalog)
        print_result({
            "status": "PASS_TRACE_SYNTAX_AND_LIFECYCLE_ONLY",
            "trace_id": trace["trace_id"],
            "trace_sha256": trace["sha256"],
            "catalog_sha256": catalog["sha256"],
            "operations": len(trace["operations"]),
            "metric_contract": trace["header"]["metric_contract"],
            "k": trace["k"],
            "declared_rebuilds": trace["rebuild_count"],
            "nonclaim": "No host/GPU/oracle execution occurred.",
        })
        return 0
    except ProtocolError as error:
        print_result({"status": "FAIL_CLOSED", "error": str(error)})
        return 2


if __name__ == "__main__":
    sys.exit(main())
