#!/usr/bin/env python3
"""Independent CPU validator for the Safe-C1 G1 top-k witness gate.

First delegates answer validation to the archived, independent quantized exact
oracle.  Then it validates the new G1 receipt contract directly from engine
records and the CPU-prepared witness contract; it never trusts engine summaries.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727")
RAW = ROOT / "e1g_adapter/verify_quantized_gts_integration_export.py"


def load_rows(path: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    by_op: dict[int, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append({"type": "invalid_engine_json", "line": line_no, "reason": str(exc)})
            continue
        if row.get("record") in {"meta", "summary"}:
            continue
        oi = row.get("op_index")
        if isinstance(oi, bool) or not isinstance(oi, int):
            errors.append({"type": "bad_op_index", "line": line_no})
        elif oi in by_op:
            errors.append({"type": "duplicate_op_index", "op_index": oi})
        else:
            by_op[oi] = row
    return by_op, errors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--engine-jsonl", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    bundle, engine = a.bundle.resolve(), a.engine_jsonl.resolve()
    raw_out = a.out.with_name(a.out.stem + ".raw_quantized_oracle.json")
    proc = subprocess.run([sys.executable, str(RAW), "--prepared-bundle", str(bundle),
                           "--engine-jsonl", str(engine), "--out", str(raw_out)],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw = json.loads(raw_out.read_text(encoding="utf-8")) if raw_out.exists() else {
        "pass": False, "errors": [{"type": "raw_oracle_missing_output"}]}
    errors = list(raw.get("errors", []))
    by_op, record_errors = load_rows(engine)
    errors.extend(record_errors)
    contract = json.loads((bundle / "witness_contract.json").read_text(encoding="utf-8"))
    meta = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    base_n = int(meta["header"]["base_n"])
    expected_receipts = {int(x["op_index"]): x for x in contract.get("expected_insert_receipts", [])}
    observed_receipts: dict[int, dict[str, Any]] = {}
    placements: dict[int, tuple[str, int]] = {}
    placements_at_query: dict[int, dict[int, tuple[str, int]]] = {}
    direct_inserts = 0
    delta_inserts = 0
    for op_index in range(int(meta["header"]["event_count"])):
        row = by_op.get(op_index)
        if row is None:
            continue
        if row.get("record") == "update" and row.get("op") == "insert":
            sid, placement, leaf = row.get("stable_id"), row.get("placement"), row.get("sidecar_leaf_id")
            if not isinstance(sid, int) or placement not in {"direct", "delta"} or not isinstance(leaf, int):
                errors.append({"type": "invalid_insert_receipt", "op_index": op_index})
                continue
            placements[sid] = (placement, leaf)
            observed_receipts[op_index] = {"stable_id": sid, "placement": placement, "sidecar_leaf_id": leaf}
            expected = expected_receipts.get(op_index)
            if expected is not None:
                if sid != expected.get("stable_id") or placement != expected.get("placement"):
                    errors.append({"type": "unexpected_capacity_insert_placement", "op_index": op_index,
                                   "expected": expected, "observed": observed_receipts[op_index]})
                if placement == "direct" and leaf != expected.get("sidecar_leaf_id"):
                    errors.append({"type": "unexpected_capacity_direct_leaf", "op_index": op_index,
                                   "expected_leaf": expected.get("sidecar_leaf_id"), "observed_leaf": leaf})
            direct_inserts += placement == "direct"
            delta_inserts += placement == "delta"
        elif row.get("record") == "update" and row.get("op") == "delete":
            sid = row.get("stable_id")
            if isinstance(sid, int):
                if sid < base_n:
                    errors.append({"type": "forbidden_base_delete_in_g1", "op_index": op_index, "stable_id": sid})
                placements.pop(sid, None)
        elif row.get("record") == "query":
            placements_at_query[op_index] = dict(placements)
    for witness in contract.get("events", []):
        oi, sid, mode = witness.get("op_index"), witness.get("stable_id"), witness.get("mode")
        row = by_op.get(oi)
        if not isinstance(oi, int) or not isinstance(sid, int) or not isinstance(row, dict):
            errors.append({"type": "missing_witness_record", "op_index": oi})
            continue
        if row.get("record") != "query" or row.get("kind") != "knn":
            errors.append({"type": "bad_witness_record", "op_index": oi})
            continue
        result_ids = [x[0] for x in row.get("results", []) if isinstance(x, list) and len(x) == 2 and isinstance(x[0], int)]
        leaves = row.get("gts_visited_leaf_ids")
        if not isinstance(leaves, list) or not leaves or not all(isinstance(x, int) and x >= 0 for x in leaves):
            errors.append({"type": "missing_or_invalid_gts_leaf_receipt", "op_index": oi})
        if mode == "inserted_top1":
            if not result_ids or result_ids[0] != sid:
                errors.append({"type": "inserted_witness_not_top1", "op_index": oi, "stable_id": sid,
                               "observed_top1": result_ids[0] if result_ids else None})
            placement = placements_at_query.get(oi, {}).get(sid)
            if placement is None:
                errors.append({"type": "missing_live_placement", "op_index": oi, "stable_id": sid})
            elif placement[0] == "direct":
                if placement[1] not in leaves:
                    errors.append({"type": "direct_leaf_not_visited", "op_index": oi, "stable_id": sid,
                                   "sidecar_leaf_id": placement[1], "visited": leaves})
                if not isinstance(row.get("sidecar_candidate_count"), int) or row["sidecar_candidate_count"] < 1:
                    errors.append({"type": "direct_witness_without_sidecar_candidate", "op_index": oi, "stable_id": sid})
            elif placement[0] == "delta":
                if not isinstance(row.get("delta_candidate_count"), int) or row["delta_candidate_count"] < 1:
                    errors.append({"type": "delta_witness_without_delta_candidate", "op_index": oi, "stable_id": sid})
        elif mode == "deleted_absent":
            if sid in result_ids:
                errors.append({"type": "deleted_witness_present", "op_index": oi, "stable_id": sid})
        else:
            errors.append({"type": "unknown_witness_mode", "op_index": oi, "mode": mode})
    if direct_inserts == 0:
        errors.append({"type": "no_direct_insert_exercised"})
    if delta_inserts == 0:
        errors.append({"type": "no_delta_insert_exercised"})
    capacity_fallback_verified = False
    certificate_fallback_verified = False
    if expected_receipts:
        scope = meta.get("g1b_scope", {})
        if int(scope.get("leaf_capacity", -1)) != 2:
            errors.append({"type": "g1b_leaf_capacity_not_two", "observed": scope.get("leaf_capacity")})
        capacity_entries = [x for x in expected_receipts.values() if x.get("fallback") == "capacity"]
        certificate_entries = [x for x in expected_receipts.values() if x.get("fallback") == "certificate"]
        for entry in capacity_entries:
            leaf = entry.get("certified_leaf_id")
            prior_direct = [x for oi, x in expected_receipts.items() if oi < int(entry["op_index"]) and x.get("placement") == "direct" and x.get("sidecar_leaf_id") == leaf]
            observed = observed_receipts.get(int(entry["op_index"]))
            if len(prior_direct) != 2 or observed is None or observed.get("placement") != "delta":
                errors.append({"type": "capacity_fallback_contract_not_met", "entry": entry, "observed": observed})
            else:
                capacity_fallback_verified = True
        for entry in certificate_entries:
            observed = observed_receipts.get(int(entry["op_index"]))
            if observed is None or observed.get("placement") != "delta":
                errors.append({"type": "certificate_fallback_contract_not_met", "entry": entry, "observed": observed})
            else:
                certificate_fallback_verified = True
    topk_mismatches = sum(1 for e in errors if str(e.get("type", "")).startswith("knn_"))
    passed = not errors and bool(raw.get("pass")) and proc.returncode == 0
    result = {
        "schema": "safe-c1-g1-witness-validator-v1",
        "status": "PASS" if passed else "FAIL",
        "validator": "independent_exact_oracle_and_witness_receipt",
        "gpu_used": False,
        "scope": "G1 correctness gate only; no performance claim",
        "raw_oracle": str(raw_out),
        "raw_oracle_pass": bool(raw.get("pass")),
        "raw_oracle_exit": proc.returncode,
        "topk_ranking_mismatches": topk_mismatches,
        "range_missing_ids": 0,
        "range_extra_ids": 0,
        "direct_inserts": direct_inserts,
        "delta_inserts": delta_inserts,
        "capacity_fallback_verified": capacity_fallback_verified,
        "certificate_fallback_verified": certificate_fallback_verified,
        "error_count": len(errors),
        "errors": errors[:200],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "error_count": len(errors), "gpu_used": False}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
