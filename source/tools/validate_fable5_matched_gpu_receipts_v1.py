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


def _as_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        return None
    return list(value)


def _result_ids(value: Any) -> set[int] | None:
    if not isinstance(value, list):
        return None
    ids: set[int] = set()
    for item in value:
        if not (isinstance(item, list) and len(item) == 2 and isinstance(item[0], int)):
            return None
        ids.add(item[0])
    return ids


def validate_pair(safe_rows: list[dict[str, Any]], buffer_rows: list[dict[str, Any]], trace_sha: str) -> list[str]:
    """Validate semantic matching, not merely equal final result lines.

    The accepted Fable5 witness is intentionally a fixed 12-op contract:
    I(A), Q, I(B capacity), I(R certificate-reject), Q, D(A direct), D(B
    delta), I(C slot reuse), Q, D(base), REBUILD, Q.  The pre-NVML trace
    preflight binds ABI/payload; this post-run validator binds *runtime
    receipts* to that contract.  It does not interpret timings.
    """
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

    def indexed_records(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for row in rows:
            record = row.get("record")
            if record not in {"update", "query", "rebuild"}:
                continue
            op = row.get("op_index")
            if not isinstance(op, int) or op in result:
                errors.append(label + "_duplicate_or_invalid_op_index")
                continue
            result[op] = row
        return result

    safe_ops = indexed_records(safe_rows, "safe")
    buffer_ops = indexed_records(buffer_rows, "buffer")
    expected_pattern = [
        ("update", "insert"), ("query", None), ("update", "insert"), ("update", "insert"),
        ("query", None), ("update", "delete"), ("update", "delete"), ("update", "insert"),
        ("query", None), ("update", "delete"), ("rebuild", None), ("query", None),
    ]
    expected_indices = set(range(len(expected_pattern)))
    if set(safe_ops) != expected_indices or set(buffer_ops) != expected_indices:
        errors.append("fable5_12op_set_mismatch")
    for op, (record, update_op) in enumerate(expected_pattern):
        for label, rows in (("safe", safe_ops), ("buffer", buffer_ops)):
            row = rows.get(op)
            if row is None:
                continue
            if row.get("record") != record or (update_op is not None and row.get("op") != update_op):
                errors.append(f"{label}_fable5_pattern_mismatch_op_{op}")

    # Require that both policies consumed exactly the same stable-ID trace.
    for op in sorted(expected_indices.intersection(safe_ops).intersection(buffer_ops)):
        s, b = safe_ops[op], buffer_ops[op]
        if s.get("record") != b.get("record"):
            errors.append(f"record_kind_mismatch_op_{op}")
            continue
        if s.get("record") == "update":
            if s.get("op") != b.get("op") or s.get("stable_id") != b.get("stable_id"):
                errors.append(f"trace_update_mismatch_op_{op}")
        elif s.get("record") == "query":
            if s.get("query_id") != b.get("query_id"):
                errors.append(f"query_id_mismatch_op_{op}")

    # The trace's named roles are derived from actual records rather than a
    # user-provided leaf label.  Thus capacity and reuse must be proved by the
    # native certificate receipts emitted at runtime.
    role_ids: dict[str, int] = {}
    for role, op in (("A", 0), ("B", 2), ("R", 3), ("C", 7), ("base", 9)):
        row = safe_ops.get(op)
        if row is None or not isinstance(row.get("stable_id"), int):
            errors.append("missing_role_id_" + role)
        else:
            role_ids[role] = row["stable_id"]
    if all(role in role_ids for role in ("A", "B", "R", "C", "base")):
        if len(set(role_ids.values())) != len(role_ids):
            errors.append("fable5_role_ids_not_distinct")
        if safe_ops.get(5, {}).get("stable_id") != role_ids["A"]:
            errors.append("direct_delete_does_not_target_A")
        if safe_ops.get(6, {}).get("stable_id") != role_ids["B"]:
            errors.append("delta_delete_does_not_target_B")

    # Replay only dynamic placements.  Initial IDs are base; any first-seen
    # delete must therefore be the base-delete/rebuild trigger.
    safe_dynamic: dict[int, tuple[str, int]] = {}
    buffer_dynamic: set[int] = set()
    direct_visible_in_exact_result = False
    capacity_same_leaf_proved = False
    for op in range(len(expected_pattern)):
        s, b = safe_ops.get(op), buffer_ops.get(op)
        if s is None or b is None:
            continue
        record = s.get("record")
        if record == "update":
            sid = s.get("stable_id")
            if not isinstance(sid, int):
                errors.append(f"safe_update_missing_id_op_{op}")
                continue
            if s.get("op") == "insert":
                cert_s, cert_b = s.get("native_certificate_leaf"), b.get("native_certificate_leaf")
                if not isinstance(cert_s, int) or cert_s != cert_b:
                    errors.append(f"native_certificate_receipt_mismatch_op_{op}")
                if b.get("placement") != "delta" or b.get("fallback_reason") != "buffer_only_policy":
                    errors.append(f"buffer_not_forced_delta_op_{op}")
                if b.get("sidecar_leaf_id") != -1:
                    errors.append(f"buffer_has_sidecar_op_{op}")
                if s.get("placement") == "direct":
                    if not isinstance(cert_s, int) or cert_s < 0 or s.get("sidecar_leaf_id") != cert_s or s.get("fallback_reason") != "none":
                        errors.append(f"malformed_direct_receipt_op_{op}")
                    safe_dynamic[sid] = ("direct", int(cert_s) if isinstance(cert_s, int) else -1)
                elif s.get("fallback_reason") == "capacity_full":
                    if not isinstance(cert_s, int) or cert_s < 0 or s.get("placement") != "delta" or s.get("sidecar_leaf_id") != -1:
                        errors.append(f"malformed_capacity_receipt_op_{op}")
                    if not any(kind == "direct" and leaf == cert_s for kind, leaf in safe_dynamic.values()):
                        errors.append(f"capacity_without_active_same_leaf_direct_op_{op}")
                    else:
                        capacity_same_leaf_proved = True
                    safe_dynamic[sid] = ("delta", -1)
                elif s.get("fallback_reason") == "certificate_reject":
                    if cert_s != -1 or s.get("placement") != "delta" or s.get("sidecar_leaf_id") != -1:
                        errors.append(f"malformed_certificate_reject_receipt_op_{op}")
                    safe_dynamic[sid] = ("delta", -1)
                else:
                    errors.append(f"unknown_safe_insert_route_op_{op}")
                if sid in buffer_dynamic:
                    errors.append(f"buffer_duplicate_dynamic_insert_op_{op}")
                buffer_dynamic.add(sid)
            elif s.get("op") == "delete":
                prior_s, prior_b = s.get("prior_placement"), b.get("prior_placement")
                if sid in safe_dynamic:
                    expected_kind, _leaf = safe_dynamic[sid]
                    if prior_s != expected_kind:
                        errors.append(f"safe_delete_prior_mismatch_op_{op}")
                    if prior_b != "delta":
                        errors.append(f"buffer_dynamic_delete_not_delta_op_{op}")
                    del safe_dynamic[sid]
                    if sid not in buffer_dynamic:
                        errors.append(f"buffer_delete_missing_dynamic_id_op_{op}")
                    else:
                        buffer_dynamic.remove(sid)
                else:
                    if prior_s != "base" or prior_b != "base":
                        errors.append(f"non_dynamic_delete_not_base_op_{op}")
                    if op != 9:
                        errors.append(f"base_delete_outside_rebuild_trigger_op_{op}")
                if op == 5 and prior_s != "direct":
                    errors.append("A_not_deleted_as_direct")
                if op == 6 and prior_s != "delta":
                    errors.append("B_not_deleted_as_delta")
            else:
                errors.append(f"unknown_update_op_{op}")
        elif record == "query":
            visited_s, visited_b = _as_int_list(s.get("gts_visited_leaf_ids")), _as_int_list(b.get("gts_visited_leaf_ids"))
            side_s, side_b = _as_int_list(s.get("sidecar_ids")), _as_int_list(b.get("sidecar_ids"))
            delta_s, delta_b = _as_int_list(s.get("delta_ids")), _as_int_list(b.get("delta_ids"))
            if visited_s is None or visited_b is None or visited_s != visited_b:
                errors.append(f"native_traversal_receipt_mismatch_op_{op}")
            if side_s is None or side_b is None or delta_s is None or delta_b is None:
                errors.append(f"malformed_dynamic_candidate_list_op_{op}")
                continue
            expected_side = sorted(sid for sid, (kind, leaf) in safe_dynamic.items()
                                   if kind == "direct" and leaf in set(visited_s))
            expected_safe_delta = sorted(sid for sid, (kind, _leaf) in safe_dynamic.items() if kind == "delta")
            if sorted(side_s) != expected_side or len(side_s) != len(set(side_s)):
                errors.append(f"receipt_sidecar_selection_mismatch_op_{op}")
            if side_b:
                errors.append(f"buffer_has_sidecar_query_op_{op}")
            if sorted(delta_s) != expected_safe_delta or len(delta_s) != len(set(delta_s)):
                errors.append(f"safe_delta_set_mismatch_op_{op}")
            if sorted(delta_b) != sorted(buffer_dynamic) or len(delta_b) != len(set(delta_b)):
                errors.append(f"buffer_delta_set_mismatch_op_{op}")
            if s.get("active_hash") != b.get("active_hash"):
                errors.append(f"active_hash_mismatch_op_{op}")
            if s.get("oracle_full_active_set_checked") is not True or b.get("oracle_full_active_set_checked") is not True:
                errors.append(f"oracle_missing_op_{op}")
            if not equal_results(s.get("results", []), b.get("results", [])):
                errors.append(f"oracle_result_mismatch_op_{op}")
            result_ids = _result_ids(s.get("results"))
            if result_ids is None:
                errors.append(f"malformed_query_result_op_{op}")
            elif result_ids.intersection(side_s):
                direct_visible_in_exact_result = True
        elif record == "rebuild":
            for key in ("trigger", "pre_rebuild_active_hash", "post_rebuild_active_hash",
                        "immutable_pool_hash_before", "immutable_pool_hash_after"):
                if s.get(key) != b.get(key):
                    errors.append(f"rebuild_{key}_mismatch_op_{op}")
            if s.get("trigger") != "base_delete_immediate":
                errors.append(f"bad_rebuild_trigger_op_{op}")
            # At rebuild, C and R may remain live but are reseeded as base.
            safe_dynamic.clear()
            buffer_dynamic.clear()
        else:
            errors.append(f"unknown_record_op_{op}")

    # The last query is after metadata-only rebuild, so sidecar/delta are empty.
    post = safe_ops.get(11)
    if post is not None:
        if _as_int_list(post.get("sidecar_ids")) != []:
            errors.append("post_rebuild_sidecar_not_empty")
        if _as_int_list(post.get("delta_ids")) != []:
            errors.append("post_rebuild_delta_not_empty")
    if not capacity_same_leaf_proved:
        errors.append("safe_missing_capacity_same_leaf_proof")
    if not direct_visible_in_exact_result:
        errors.append("safe_direct_never_visible_in_exact_result")
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
