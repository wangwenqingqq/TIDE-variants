#!/usr/bin/env python3
"""Static safety gate for the archived GTS incremental implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def hits(path: Path, needle: str) -> list[int]:
    return [i for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1) if needle in line]


def check(src: Path, file: str, needle: str, why: str, required: str) -> dict[str, Any]:
    path = src / file
    locs = hits(path, needle) if path.exists() else []
    return {
        "id": needle,
        "file": file,
        "lines": locs,
        "detected": bool(locs),
        "why_blocks_e1g": why,
        "required_fix": required,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    src = args.source.resolve()
    checks = [
        check(src, "include/incremental_insert.cuh", "leaf.size += 1", "Direct insertion expands the legacy tree leaf rather than using a bounded sidecar; it changes query-visible occupancy and exceeds MAX_SIZE result slots.", "Replace direct leaf writes with immutable tree intervals plus per-leaf sidecar storage and sidecar-aware result capacity."),
        check(src, "include/incremental_insert.cuh", "dist > h_inc_max_dis[leaf_id]", "Direct insertion mutates the leaf coverage radius after construction; ancestor/query visibility is not certified.", "Freeze routing intervals/radii between rebuilds; route non-certified objects to exact delta."),
        check(src, "include/update.cuh", "int logical_id = update_list[i].update_id", "Deletes are addressed by current live rank, not a stable external ID; a trace cannot be replayed after compaction/rebuild.", "Carry stable IDs end-to-end and maintain stable_id -> current physical slot mapping across rebuild."),
        check(src, "include/update.cuh", "if (0 > 0)", "The direct-overflow scan branch is permanently disabled, so the intended separate direct tier is absent.", "Scan every certified leaf sidecar exactly once and scan global delta exactly once for each query."),
        check(src, "include/update.cuh", "fprintf(fcost, \"%d \", total_result_num)", "Update mode exports only a range result count, not IDs/distances; an exact oracle cannot detect false positives/negatives.", "Emit per-operation JSONL stable IDs and distances for range and top-k."),
        check(src, "include/update.cuh", "searchIndexRnnUpdate", "The archived update loop has only dynamic range-query invocation; no dynamic top-k result path is exported.", "Add Safe-C1 dynamic top-k path or limit E1-G claim to explicitly stated range-only scope."),
        check(src, "include/update.cuh", "getNewData<<<", "Rebuild rematerializes data rows and does not retain an external stable-ID mapping.", "Keep immutable pool rows or atomically update stable_id -> physical map during rebuild; verify active set hash."),
        check(src, "include/search.cuh", "allinclude_flags", "All-include propagation can include post-build objects without distance evaluation once a leaf is modified.", "Disable all-include for sidecar-bearing paths or prove its certificate covers sidecar objects."),
    ]
    source_files = [src / "include/incremental_insert.cuh", src / "include/update.cuh", src / "include/search.cuh", src / "include/tree.cuh", src / "src/main.cu"]
    detected = [c for c in checks if c["detected"]]
    result = {
        "schema": "e1g-legacy-static-audit-v1",
        "source": str(src),
        "source_files_sha256": {str(x.relative_to(src)): sha256(x) for x in source_files if x.exists()},
        "status": "BLOCKED_UNSAFE_LEGACY_SEMANTICS" if detected else "UNEXPECTED_AUDIT_RESULT",
        "gpu_used": False,
        "blocker_count": len(detected),
        "checks": checks,
        "minimum_safe_patch_order": [
            "1. immutable pool + base-only frozen tree + disjoint reservoir loader",
            "2. stable-ID registry and rebuild remapping",
            "3. certified leaf sidecar / exact delta (no leaf.size or interval mutation)",
            "4. sidecar/delta-aware exact range and top-k exporters",
            "5. independent verifier replay over stable trace before any performance measurement",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "blocker_count": result["blocker_count"], "out": str(args.out)}, sort_keys=True))
    return 0 if not detected else 2


if __name__ == "__main__":
    raise SystemExit(main())
