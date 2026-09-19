#!/usr/bin/env python3
"""CPU-only source contract audit for the isolated Safe-C1 G1 implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def need(text: str, token: str, errors: list[str], label: str) -> None:
    if token not in text:
        errors.append(f"missing:{label}:{token}")


def forbid(text: str, token: str, errors: list[str], label: str) -> None:
    if token in text:
        errors.append(f"forbidden:{label}:{token}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    source = root / "src/safe_c1_dynamic_gts.cu"
    header = root / "include/safe_search_v2.cuh"
    binary = root / "bin/GTS_safe_c1_g1"
    errors: list[str] = []
    for path in (source, header, binary):
        if not path.is_file():
            errors.append(f"missing_file:{path}")
    src = source.read_text(encoding="utf-8") if source.is_file() else ""
    hdr = header.read_text(encoding="utf-8") if header.is_file() else ""
    # Mechanism checks: real GTS receipt is captured at mergeLNodeKnn before
    # native leaf scan, and state sidecars are indexed only through that receipt.
    need(src, '#include "safe_search_v2.cuh"', errors, "source")
    need(src, 'run_gts_base_topk_with_receipt', errors, "source")
    need(src, 'state.sidecar_candidates_for(gts_answer.visited_leaf_ids)', errors, "source")
    need(src, 'exact_candidate_l2', errors, "source")
    need(src, 'G1 refuses base deletion', errors, "source")
    need(src, 'G1 deliberately rejects range events', errors, "source")
    forbid(src, 'exact_live_l2', errors, "source")
    # Comments may name forbidden legacy facilities to document the boundary;
    # reject only an actual include or executable call token.
    for bad in ('#include "incremental_insert.cuh"', '#include "update.cuh"',
                'updateIndexRnn(', 'deleteIncrementalInsert('):
        forbid(src, bad, errors, "source")
    need(hdr, 'safe_c1_last_visited_leaf_pairs.clear()', errors, "header")
    need(hdr, 'Safe-C1 instrumentation: materialize only the (query, leaf) pairs', errors, "header")
    need(hdr, 'mergeLNodeKnn error', errors, "header")
    need(hdr, 'dataProcessKnnVec<<<', errors, "header")
    m = hdr.find('Safe-C1 instrumentation: materialize only the (query, leaf) pairs')
    d = hdr.find('dataProcessKnnVec<<<', m if m >= 0 else 0)
    if m < 0 or d < 0 or m > d:
        errors.append('receipt_not_before_native_leaf_scan')
    result = {
        "schema": "safe-c1-g1-static-contract-audit-v1",
        "gpu_used": False,
        "root": str(root),
        "status": "PASS_CPU_ONLY_STATIC_CONTRACT" if not errors else "FAIL_CPU_ONLY_STATIC_CONTRACT",
        "formal_claim_eligible": False,
        "scope": "static source/binary contract only; not an execution or performance result",
        "files": {str(x.relative_to(root)): sha256(x) for x in (source, header, binary) if x.is_file()},
        "errors": errors,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": len(errors), "gpu_used": False}, sort_keys=True))
    return 0 if not errors else 2

if __name__ == "__main__":
    raise SystemExit(main())
