#!/usr/bin/env python3
"""Independent stdlib-only full-live integer-L2 oracle/replayer.

It reads a sealed catalog and trace, never imports the host or runner, and
creates a new JSONL output using exclusive creation. It is intentionally not
invoked by static audit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from tide_protocol_lib import (
    ProtocolError,
    active_set_digest,
    canonical_json_bytes,
    load_and_validate_trace,
    print_result,
    squared_l2,
)


def ordered_topk(catalog: Dict[str, Any], live: Set[str], query_id: str, k: int) -> List[Dict[str, Any]]:
    query = catalog["queries"][query_id]["vector"]
    ranked = []
    for stable_id in live:
        vector = catalog["objects"][stable_id]["vector"]
        ranked.append((squared_l2(query, vector), stable_id))
    ranked.sort()
    return [
        {"rank": rank, "stable_id": stable_id, "distance_key": str(distance)}
        for rank, (distance, stable_id) in enumerate(ranked[:k], 1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path,
                        help="new output path; existing path is rejected")
    args = parser.parse_args()
    try:
        catalog, trace = load_and_validate_trace(args.trace, args.catalog)
        require_parent = args.out.parent
        if not require_parent.is_dir():
            raise ProtocolError(f"{args.out}: parent does not exist")
        live: Set[str] = {
            stable_id for stable_id, row in catalog["objects"].items()
            if row["initial_membership"] == "base"
        }
        base: Set[str] = set(live)
        overlay: Set[str] = set()
        epoch = 0
        records: List[Dict[str, Any]] = []
        for operation in trace["operations"]:
            seq = operation["seq"]
            pre_digest = active_set_digest(live)
            op = operation["op"]
            rebuilt = False
            if op == "insert":
                stable_id = operation["stable_id"]
                live.add(stable_id)
                overlay.add(stable_id)
                if operation.get("post_op_rebuild", False):
                    base = set(live)
                    overlay.clear()
                    epoch += 1
                    rebuilt = True
                record: Dict[str, Any] = {
                    "schema": "tide.oracle-replay-record.v1", "seq": seq,
                    "op": op, "epoch_before": epoch - int(rebuilt), "epoch_after": epoch,
                    "pre_active_set_digest": pre_digest,
                    "post_active_set_digest": active_set_digest(live),
                    "rebuild": rebuilt,
                }
            elif op == "delete":
                stable_id = operation["stable_id"]
                was_base = stable_id in base
                live.remove(stable_id)
                if was_base:
                    base = set(live)
                    overlay.clear()
                    epoch += 1
                    rebuilt = True
                else:
                    overlay.remove(stable_id)
                record = {
                    "schema": "tide.oracle-replay-record.v1", "seq": seq,
                    "op": op, "epoch_before": epoch - int(rebuilt), "epoch_after": epoch,
                    "pre_active_set_digest": pre_digest,
                    "post_active_set_digest": active_set_digest(live),
                    "rebuild": rebuilt, "deleted_stable_id": stable_id,
                }
            else:
                if len(live) < trace["k"]:
                    raise ProtocolError(f"trace seq {seq}: fewer than k live objects for query")
                answer = ordered_topk(catalog, live, operation["query_id"], trace["k"])
                record = {
                    "schema": "tide.oracle-replay-record.v1", "seq": seq,
                    "op": op, "epoch_before": epoch, "epoch_after": epoch,
                    "pre_active_set_digest": pre_digest,
                    "post_active_set_digest": pre_digest,
                    "rebuild": False, "query_id": operation["query_id"],
                    "answer": answer,
                }
            records.append(record)
        with args.out.open("xb") as handle:
            for record in records:
                handle.write(canonical_json_bytes(record))
        print_result({
            "status": "ORACLE_REPLAY_CREATED",
            "trace_id": trace["trace_id"],
            "records": len(records),
            "out": str(args.out),
            "nonclaim": "Independent raw-vector replay only; no host result was accepted.",
        })
        return 0
    except (ProtocolError, OSError) as error:
        print_result({"status": "FAIL_CLOSED", "error": str(error)})
        return 2


if __name__ == "__main__":
    sys.exit(main())
