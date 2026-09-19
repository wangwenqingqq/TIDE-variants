#!/usr/bin/env python3
"""CPU-only deterministic input derivation for C1 workspace microbenchmark.

Reads archived inputs without modification and writes only below the protocol
isolation root.  It extracts the first fixed number of type-2 operations from
the exact update trace; it never imports CUDA or invokes GTS.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def ensure_under(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(f"refusing to write outside isolation root: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path,
                        default=Path(__file__).with_name("protocol_c1_workspace_onequery_sift1m_v1.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--verify-source-sha256", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("schema") != "gtspp-c1-workspace-onequery-microbenchmark-protocol-v1":
        raise SystemExit("unexpected protocol schema")
    root = Path(protocol["isolation"]["write_root"])
    out = args.out_dir
    ensure_under(out, root)
    out.mkdir(parents=True, exist_ok=True)

    data = protocol["data_and_input"]
    base = Path(data["base_data"]["path"])
    trace = Path(data["source_update_trace"]["path"])
    if not base.is_file() or not trace.is_file():
        raise SystemExit("missing pre-registered base/trace source")
    header = base.open().readline().strip()
    if header != data["base_data"]["header"]:
        raise SystemExit(f"base header mismatch: {header!r}")
    if args.verify_source_sha256:
        for name, path, expect in [
            ("base", base, data["base_data"]["sha256"]),
            ("trace", trace, data["source_update_trace"]["sha256"]),
        ]:
            actual = sha256(path)
            if actual != expect:
                raise SystemExit(f"{name} SHA-256 mismatch: {actual} != {expect}")

    expected = data["source_update_trace"]["observed_flags"]
    count = int(data["derived_measured_trace"]["selection"].split()[1])  # fixed "first 1024 ..."
    # Keep the protocol parser intentionally strict to prevent hidden selection changes.
    if count != 1024:
        raise SystemExit("protocol must pre-register exactly 1024 type-2 operations")
    selected: list[tuple[int, int, int]] = []  # (source event ordinal, source line number, id)
    op_counts = {0: 0, 1: 0, 2: 0}
    with trace.open() as fh:
        declared_line = fh.readline().strip()
        declared = int(declared_line)
        if declared != data["source_update_trace"]["declared_operations"]:
            raise SystemExit(f"declared count mismatch: {declared}")
        rows = 0
        for source_line_number, line in enumerate(fh, start=2):
            parts = line.split()
            if len(parts) != 2:
                raise SystemExit(f"malformed trace line {source_line_number}")
            op, item_id = map(int, parts)
            if op not in op_counts:
                raise SystemExit(f"unexpected operation {op} at line {source_line_number}")
            op_counts[op] += 1
            if op == 2 and len(selected) < count:
                selected.append((rows, source_line_number, item_id))
            rows += 1
    if rows != declared or op_counts != {0: expected["0_insert"], 1: expected["1_delete"], 2: expected["2_range_query"]}:
        raise SystemExit(f"source trace count mismatch: rows={rows}, ops={op_counts}")
    if len(selected) != count:
        raise SystemExit(f"only found {len(selected)} type-2 operations")
    # IDs reference the 1M base in this query-only workload.
    if any(item_id < 0 or item_id >= 1000000 for _, _, item_id in selected):
        raise SystemExit("selected query ID outside static SIFT1M base")

    trace_out = out / "sift1m_10k_442_first1024_type2_query_only.txt"
    with trace_out.open("w") as fh:
        fh.write(f"{len(selected)}\n")
        for _, _, item_id in selected:
            fh.write(f"2 {item_id}\n")
    manifest = {
        "schema": "gtspp-c1-query-only-trace-provenance-v1",
        "status": "PASS_CPU_ONLY_TRACE_DERIVATION",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "input derivation only; no CUDA/GTS execution",
        "protocol": {"path": str(args.protocol), "sha256": sha256(args.protocol)},
        "base": {"path": str(base), "header": header, "expected_sha256": data["base_data"]["sha256"]},
        "source_trace": {"path": str(trace), "expected_sha256": data["source_update_trace"]["sha256"],
                         "declared_operations": declared, "observed_flags": op_counts},
        "selection": "first 1024 original-order type-2 operations",
        "selected_source_event_ordinals_zero_based": [x[0] for x in selected],
        "selected_source_line_numbers_one_based": [x[1] for x in selected],
        "selected_id_min": min(x[2] for x in selected),
        "selected_id_max": max(x[2] for x in selected),
        "derived_trace": {"path": str(trace_out), "sha256": sha256(trace_out), "operations": len(selected)},
        "verify_source_sha256": bool(args.verify_source_sha256)
    }
    manifest_out = out / "sift1m_10k_442_first1024_type2_provenance.json"
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(trace_out)
    print(manifest_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
