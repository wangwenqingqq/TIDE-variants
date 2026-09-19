#!/usr/bin/env python3
"""CPU-only validator for one Safe-C1/buffer-only Fable5 matched GPU receipt pair."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: malformed JSONL line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}: non-object JSONL line {line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: empty JSONL")
    return rows


def index_rows(rows: list[dict[str, Any]], kind: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("record") != kind:
            continue
        op = row.get("op_index")
        if not isinstance(op, int) or op in result:
            raise ValueError(f"duplicate/invalid {kind} op_index")
        result[op] = row
    return result


def close_number(left: Any, right: Any) -> bool:
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return False
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10)


def equal_results(left: list[Any], right: list[Any]) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if not (isinstance(a, list) and isinstance(b, list) and len(a) == 2 and len(b) == 2):
            return False
        if a[0] != b[0] or not close_number(a[1], b[1]):
            return False
    return True


def validate_pair(safe_rows: list[dict[str, Any]], buffer_rows: list[dict[str, Any]], trace_sha: str) -> list[str]:
    errors: list[str] = []
    safe_meta = safe_rows[0]
    buffer_meta = buffer_rows[0]
    for label, meta, policy in (("safe", safe_meta, "safe_c1"), ("buffer", buffer_meta, "buffer_only")):
        if meta.get("record") != "meta" or meta.get("schema") != "fable5-matched-gpu-runner-v1":
            errors.append(label + "_wrong_meta_schema")
        if meta.get("policy") != policy:
            errors.append(label + "_wrong_policy")
        if meta.get("trace_sha256") != trace_sha:
            errors.append(label + "_trace_sha_mismatch")
        if meta.get("legacy_incremental_updater_used") is not False:
            errors.append(label + "_legacy_updater_not_false")
        if meta.get("rebuild_policy") != "base_delete_immediate":
            errors.append(label + "_wrong_rebuild_policy")
    safe_updates = index_rows(safe_rows, "update")
    buffer_updates = index_rows(buffer_rows, "update")
    if set(safe_updates) != set(buffer_updates):
        errors.append("update_op_set_mismatch")
    safe_direct = safe_capacity = safe_cert_reject = safe_direct_delete = safe_delta_delete = 0
    for op in sorted(set(safe_updates).intersection(buffer_updates)):
        s, b = safe_updates[op], buffer_updates[op]
        if s.get("op") != b.get("op") or s.get("stable_id") != b.get("stable_id"):
            errors.append(f"trace_update_mismatch_op_{op}")
            continue
        if s.get("op") == "insert":
            if s.get("native_certificate_leaf") != b.get("native_certificate_leaf"):
                errors.append(f"native_certificate_receipt_mismatch_op_{op}")
            if b.get("placement") != "delta" or b.get("fallback_reason") != "buffer_only_policy":
                errors.append(f"buffer_not_forced_delta_op_{op}")
            if s.get("placement") == "direct":
                safe_direct += 1
            if s.get("fallback_reason") == "capacity_full":
                safe_capacity += 1
            if s.get("fallback_reason") == "certificate_reject":
                safe_cert_reject += 1
        elif s.get("op") == "delete":
            if s.get("prior_placement") == "direct":
                safe_direct_delete += 1
            if s.get("prior_placement") == "delta":
                safe_delta_delete += 1
    if not safe_direct:
        errors.append("safe_missing_direct_admission")
    if not safe_capacity:
        errors.append("safe_missing_capacity_fallback")
    if not safe_cert_reject:
        errors.append("safe_missing_certificate_reject")
    if not safe_direct_delete:
        errors.append("safe_missing_direct_delete")
    if not safe_delta_delete:
        errors.append("safe_missing_delta_delete")

    safe_queries = index_rows(safe_rows, "query")
    buffer_queries = index_rows(buffer_rows, "query")
    if set(safe_queries) != set(buffer_queries) or not safe_queries:
        errors.append("query_op_set_mismatch_or_empty")
    for op in sorted(set(safe_queries).intersection(buffer_queries)):
        s, b = safe_queries[op], buffer_queries[op]
        if s.get("query_id") != b.get("query_id"):
            errors.append(f"query_id_mismatch_op_{op}")
        if s.get("active_hash") != b.get("active_hash"):
            errors.append(f"active_hash_mismatch_op_{op}")
        if s.get("gts_visited_leaf_ids") != b.get("gts_visited_leaf_ids"):
            errors.append(f"native_traversal_receipt_mismatch_op_{op}")
        if s.get("oracle_full_active_set_checked") is not True or b.get("oracle_full_active_set_checked") is not True:
            errors.append(f"oracle_missing_op_{op}")
        if not equal_results(s.get("results", []), b.get("results", [])):
            errors.append(f"oracle_result_mismatch_op_{op}")

    safe_rebuilds = index_rows(safe_rows, "rebuild")
    buffer_rebuilds = index_rows(buffer_rows, "rebuild")
    if set(safe_rebuilds) != set(buffer_rebuilds) or len(safe_rebuilds) != 1:
        errors.append("rebuild_op_set_mismatch_or_not_singleton")
    for op in sorted(set(safe_rebuilds).intersection(buffer_rebuilds)):
        s, b = safe_rebuilds[op], buffer_rebuilds[op]
        for key in ("trigger", "pre_rebuild_active_hash", "post_rebuild_active_hash", "immutable_pool_hash_before", "immutable_pool_hash_after"):
            if s.get(key) != b.get(key):
                errors.append(f"rebuild_{key}_mismatch_op_{op}")
        if s.get("trigger") != "base_delete_immediate":
            errors.append(f"bad_rebuild_trigger_op_{op}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-results", type=Path, required=True)
    parser.add_argument("--buffer-results", type=Path, required=True)
    parser.add_argument("--trace-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, Any] = {
        "schema": "fable5-matched-gpu-receipt-validator-v1",
        "gpu_used": False,
        "cuda_binary_executed": False,
        "trace_sha256": args.trace_sha256,
        "safe_results": str(args.safe_results.resolve()),
        "buffer_results": str(args.buffer_results.resolve()),
        "errors": [],
        "scope": "CPU-only post-run receipt comparison; no timing or performance interpretation.",
    }
    try:
        if len(args.trace_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in args.trace_sha256):
            raise ValueError("trace SHA must be 64 hexadecimal characters")
        safe_rows = load_jsonl(args.safe_results)
        buffer_rows = load_jsonl(args.buffer_results)
        report["errors"] = validate_pair(safe_rows, buffer_rows, args.trace_sha256)
        report["status"] = "PASS_FABLE5_MATCHED_RECEIPT_VALIDATION" if not report["errors"] else "FAIL_FABLE5_MATCHED_RECEIPT_VALIDATION"
    except Exception as exc:
        report["errors"].append(f"exception:{type(exc).__name__}:{exc}")
        report["status"] = "FAIL_FABLE5_MATCHED_RECEIPT_VALIDATION"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["status"])
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
