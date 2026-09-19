#!/usr/bin/env python3
"""Independently validate Gate-6 flat-file sizes, hashes, order, and layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "data" / "prepared" / "FLAT_INPUT_MANIFEST.json"
    reality_path = root / "results" / "TRANSITION_REALITY.json"
    manifest = json.loads(manifest_path.read_text())
    reality = json.loads(reality_path.read_text())

    snapshot_checks = {}
    for date, snapshot in manifest["snapshots"].items():
        rows = int(snapshot["rows"])
        words = int(snapshot["word_count"])
        if words != 4:
            raise RuntimeError(f"unexpected SureChEMBL word count for {date}: {words}")
        expected_rows = int(reality["surechembl_schemas"][date]["shape"][0])
        if rows != expected_rows:
            raise RuntimeError(f"row mismatch for {date}: {rows} != {expected_rows}")
        fp = Path(snapshot["fp"])
        ids = Path(snapshot["ids"])
        pc = Path(snapshot["popcnt"])
        expected_sizes = {
            "fp": rows * words * 8,
            "ids": rows * 8,
            "popcnt": rows * 2,
        }
        actual_sizes = {
            "fp": fp.stat().st_size,
            "ids": ids.stat().st_size,
            "popcnt": pc.stat().st_size,
        }
        if actual_sizes != expected_sizes:
            raise RuntimeError(
                f"component-size mismatch for {date}: {actual_sizes} != {expected_sizes}"
            )
        actual_hashes = {
            "fp": sha256(fp),
            "ids": sha256(ids),
            "popcnt": sha256(pc),
        }
        expected_hashes = {
            "fp": snapshot["fp_sha256"],
            "ids": snapshot["ids_sha256"],
            "popcnt": snapshot["popcnt_sha256"],
        }
        if actual_hashes != expected_hashes:
            raise RuntimeError(f"component-hash mismatch for {date}")
        popcounts = np.memmap(pc, mode="r", dtype="<u2")
        monotone = bool(
            len(popcounts) == 0 or np.all(popcounts[1:] >= popcounts[:-1])
        )
        if not monotone:
            raise RuntimeError(f"population-count order mismatch for {date}")
        snapshot_checks[date] = {
            "rows": rows,
            "word_count": words,
            "component_sizes": actual_sizes,
            "component_sha256": actual_hashes,
            "population_count_sorted": monotone,
        }

    transition_by_name = {
        transition["transition"]: transition for transition in reality["transitions"]
    }
    layout_checks = {}
    for transition, layout_text in manifest["layouts"].items():
        if transition == "chembl37_synthetic":
            raise RuntimeError("source-blocked ChEMBL layout was unexpectedly prepared")
        layout = Path(layout_text)
        layout_manifest = json.loads((layout / "LAYOUT_MANIFEST.json").read_text())
        if layout_manifest["transition"] != transition:
            raise RuntimeError(f"layout label mismatch for {transition}")
        checked_links = {}
        for link_text, target_text in layout_manifest["links"].items():
            link = Path(link_text)
            target = Path(target_text)
            if not link.is_symlink() or link.resolve() != target.resolve():
                raise RuntimeError(f"layout link mismatch: {link} -> {target}")
            checked_links[str(link)] = str(target.resolve())
        audit = transition_by_name[transition]
        counts = audit["counts"]
        query_dir = Path(layout_manifest["query_dir"])
        delta = query_dir / "delta_u64x6.bin"
        queries = query_dir / "queries_u64x6.bin"
        expected_delta_bytes = int(counts["added"]) * 6 * 8
        expected_query_bytes = int(counts["selected_queries"]) * 6 * 8
        if delta.stat().st_size != expected_delta_bytes:
            raise RuntimeError(f"delta byte mismatch for {transition}")
        if queries.stat().st_size != expected_query_bytes:
            raise RuntimeError(f"query byte mismatch for {transition}")
        layout_checks[transition] = {
            "link_count": len(checked_links),
            "links": checked_links,
            "delta_bytes": delta.stat().st_size,
            "query_bytes": queries.stat().st_size,
        }

    if len(snapshot_checks) != 7 or len(layout_checks) != 6:
        raise RuntimeError("incomplete flat-input denominator")
    if manifest.get("chembl_37") is not None:
        raise RuntimeError("source-blocked ChEMBL payload was unexpectedly included")
    result = {
        "experiment_id": "tide_20260829_gate6_flat_input_validation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "snapshot_checks": snapshot_checks,
        "layout_checks": layout_checks,
        "chembl_37_status": manifest["chembl_37_source_status"],
        "flat_input_validation_pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
