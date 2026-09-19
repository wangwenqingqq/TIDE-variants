#!/usr/bin/env python3
"""CPU-only validator for the isolated Safe-C1 G3 P0 repair.

This module does not import or launch CUDA code, start a subprocess, inspect a
GPU, or produce a native correctness/performance result.  Fixture-only mode
checks the control-plane contracts that a future native trace must satisfy:

* an immutable frozen base is distinct from the changing active partition;
* active = frozen-base disjoint-union source-plan sidecars disjoint-union delta;
* the current source closure keeps the direct-sidecar runtime gate closed;
* a destructive rebuild creates a new frozen generation and requires two
  independently verified first queries (KNN and range).

The fixture selector is deliberately a CPU source mirror, not an actual GTS
receipt.  A compact synthetic test also exercises the *host contract* for a
padded physical id_list receipt; the source audit is the evidence that the
native adapter uses the corresponding all-row materializer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from g3_common import exact_results, read_json, sha256_file, stable_set_sha256, write_json


# This mirrors the compile-only G3 source closure.  It is intentionally closed
# until a future authorized native runtime establishes the missing evidence.
DIRECT_SIDECAR_RUNTIME_GUARD_OPEN = False
LEAF_PAD_SLOTS = 64


def err(errors: list[dict[str, Any]], kind: str, **kwargs: Any) -> None:
    errors.append({"type": kind, **kwargs})


def local_to_stable_sha256(ids: Iterable[int]) -> str:
    values = [int(x) for x in ids]
    raw = "safe-c1-g3-local-to-stable-v1\n" + "".join(
        f"{row}:{stable}\n" for row, stable in enumerate(values)
    )
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def fixture_frozen_identity(ids: Iterable[int]) -> dict[str, Any]:
    """CPU-fixture identities, not a claim about native GTS byte hashes."""
    values = [int(x) for x in ids]
    return {
        "fixture_frozen_base_stable_ids_sha256": stable_set_sha256(values),
        "fixture_frozen_local_to_stable_sha256": local_to_stable_sha256(values),
        "fixture_frozen_base_count": len(values),
    }


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_jsonl(path: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    records: dict[int, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            err(errors, "invalid_json", line=line_no, reason=str(exc))
            continue
        if not isinstance(row, dict):
            err(errors, "not_object", line=line_no)
            continue
        if row.get("record") in {"summary", "meta"}:
            continue
        index = row.get("op_index")
        if isinstance(index, bool) or not isinstance(index, int):
            err(errors, "missing_or_invalid_op_index", line=line_no)
            continue
        if index in records:
            err(errors, "duplicate_op_index", op_index=index)
        else:
            records[index] = row
    return records, errors


def as_results(value: Any) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError("results must be a list")
    parsed: list[list[int]] = []
    seen: set[int] = set()
    for pos, item in enumerate(value):
        if isinstance(item, list) and len(item) == 2:
            sid, d2 = item
        elif isinstance(item, dict):
            sid, d2 = item.get("stable_id"), item.get("distance_sq")
        else:
            raise ValueError(f"results[{pos}] must be [stable_id,distance_sq] or object")
        if (isinstance(sid, bool) or not isinstance(sid, int) or
                isinstance(d2, bool) or not isinstance(d2, int)):
            raise ValueError(f"results[{pos}] noninteger stable_id/distance_sq")
        if sid in seen:
            raise ValueError(f"duplicate stable_id {sid}")
        if d2 < 0:
            raise ValueError("negative distance_sq")
        seen.add(sid)
        parsed.append([int(sid), int(d2)])
    return parsed


def ensure_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        raise ValueError(f"{name} must be an integer list")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicate IDs")
    return [int(x) for x in value]


def as_native_res_ids(value: Any) -> list[int]:
    """KNN diagnostic IDs: repeated -1 sentinels are permitted."""
    if not isinstance(value, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        raise ValueError("native_final_res_ids must be an integer list")
    non_sentinel = [int(x) for x in value if int(x) != -1]
    if len(set(non_sentinel)) != len(non_sentinel):
        raise ValueError("native_final_res_ids contains duplicate non-sentinel local rows")
    return [int(x) for x in value]


def as_leaf_pairs(value: Any) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError("sidecar_source_leaf_pairs must be a list")
    output: list[list[int]] = []
    for pos, item in enumerate(value):
        if isinstance(item, list) and len(item) == 2:
            leaf, sid = item
        elif isinstance(item, dict):
            leaf, sid = item.get("leaf_id"), item.get("stable_id")
        else:
            raise ValueError(f"sidecar_source_leaf_pairs[{pos}] malformed")
        if (isinstance(leaf, bool) or not isinstance(leaf, int) or leaf < 0 or
                isinstance(sid, bool) or not isinstance(sid, int)):
            raise ValueError(f"sidecar_source_leaf_pairs[{pos}] noninteger")
        output.append([int(leaf), int(sid)])
    if len({tuple(x) for x in output}) != len(output):
        raise ValueError("duplicate sidecar source leaf pair")
    return sorted(output)


def as_receipt_base_rows(value: Any) -> list[list[int]]:
    """Parse physical `(leaf,id_list_slot,local,stable)` receipt rows."""
    if not isinstance(value, list):
        raise ValueError("base_receipt_rows must be a list")
    rows: list[list[int]] = []
    seen_leaf_slot: set[tuple[int, int]] = set()
    seen_local: set[int] = set()
    for pos, item in enumerate(value):
        if isinstance(item, list) and len(item) == 4:
            leaf, slot, local, stable = item
        elif isinstance(item, dict):
            leaf = item.get("leaf_id")
            slot = item.get("id_list_slot")
            local = item.get("local_row")
            stable = item.get("stable_id")
        else:
            raise ValueError(f"base_receipt_rows[{pos}] malformed")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (leaf, slot, local, stable)):
            raise ValueError(f"base_receipt_rows[{pos}] noninteger")
        if leaf < 0 or slot < 0 or local < 0 or stable < 0:
            raise ValueError(f"base_receipt_rows[{pos}] negative physical receipt field")
        key = (int(leaf), int(slot))
        if key in seen_leaf_slot:
            raise ValueError("duplicate receipt leaf/id_list physical slot")
        if int(local) in seen_local:
            raise ValueError("receipt rows duplicate a compact local row across leaves")
        seen_leaf_slot.add(key)
        seen_local.add(int(local))
        rows.append([int(leaf), int(slot), int(local), int(stable)])
    return rows


def as_receipt_leaf_spans(value: Any) -> list[list[int]]:
    """Parse exact physical `(leaf,lid,size)` spans exported by the adapter."""
    if not isinstance(value, list):
        raise ValueError("receipt_leaf_spans must be a list")
    spans: list[list[int]] = []
    seen_leaf: set[int] = set()
    for pos, item in enumerate(value):
        if isinstance(item, list) and len(item) == 3:
            leaf, lid, size = item
        elif isinstance(item, dict):
            leaf, lid, size = item.get("leaf_id"), item.get("id_list_lid"), item.get("size")
        else:
            raise ValueError(f"receipt_leaf_spans[{pos}] malformed")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (leaf, lid, size)):
            raise ValueError(f"receipt_leaf_spans[{pos}] noninteger")
        if leaf < 0 or lid < 0 or size < 0:
            raise ValueError(f"receipt_leaf_spans[{pos}] negative")
        if int(leaf) in seen_leaf:
            raise ValueError("receipt_leaf_spans contains duplicate leaf")
        seen_leaf.add(int(leaf))
        spans.append([int(leaf), int(lid), int(size)])
    return sorted(spans)


def exact_receipt_results(pool: np.ndarray, query: np.ndarray, stable_ids: Iterable[int],
                          kind: str, radius_sq: int | None) -> list[list[int]]:
    ids = sorted(set(int(x) for x in stable_ids))
    if not ids:
        return []
    # KNN base_results intentionally contains *all* exact rows from the
    # receipt, whereas final QueryExport.results is the k-trimmed merge.
    return exact_results(pool, query, ids, kind, len(ids), radius_sq)


def load_bundle(bundle: Path) -> tuple[
    dict[str, Any], np.ndarray, np.ndarray, list[int], list[dict[str, Any]],
    list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]
]:
    errors: list[dict[str, Any]] = []
    metadata = read_json(bundle / "metadata.json")
    manifest = read_json(bundle / "manifest.json")
    selection = read_json(bundle / "selection_receipt.json")
    header = metadata.get("header", {})
    if metadata.get("schema") != "safe-c1-g3-cpu-fixture-v1":
        err(errors, "bad_metadata_schema", observed=metadata.get("schema"))
    if manifest.get("schema") != "safe-c1-g3-bundle-manifest-v1":
        err(errors, "bad_manifest_schema", observed=manifest.get("schema"))
    for name, digest in manifest.get("files_sha256", {}).items():
        path = bundle / name
        if not path.is_file():
            err(errors, "manifest_file_missing", file=name)
        elif sha256_file(path) != digest:
            err(errors, "manifest_sha_mismatch", file=name)
    try:
        pool_n, query_n, dim = int(header["pool_n"]), int(header["query_n"]), int(header["dim"])
        raw_pool = np.fromfile(bundle / "pool.i16", dtype="<i2")
        raw_query = np.fromfile(bundle / "queries.i16", dtype="<i2")
        if raw_pool.size != pool_n * dim:
            err(errors, "pool_size_mismatch", actual=int(raw_pool.size), expected=pool_n * dim)
        if raw_query.size != query_n * dim:
            err(errors, "query_size_mismatch", actual=int(raw_query.size), expected=query_n * dim)
        physical_pool = raw_pool.reshape(pool_n, dim)
        queries = raw_query.reshape(query_n, dim)
        stable_to_pool = np.fromfile(bundle / "stable_id_to_pool_row.i32", dtype="<i4")
        if stable_to_pool.size != pool_n:
            err(errors, "stable_to_pool_size_mismatch", actual=int(stable_to_pool.size), expected=pool_n)
            stable_to_pool = np.arange(pool_n, dtype=np.int32)
        if (np.any(stable_to_pool < 0) or np.any(stable_to_pool >= pool_n) or
                len(np.unique(stable_to_pool)) != pool_n):
            err(errors, "stable_to_pool_not_bijection")
            stable_to_pool = np.arange(pool_n, dtype=np.int32)
        # Stable IDs index this logical pool; do not assume the mapping is identity.
        pool = physical_pool[stable_to_pool.astype(np.int64, copy=False)]
        base_ids = np.fromfile(bundle / "initial_base_stable_ids.i32", dtype="<i4").astype(np.int64).tolist()
        if len(base_ids) != int(header["base_n"]) or len(set(base_ids)) != len(base_ids):
            err(errors, "initial_base_ids_invalid", count=len(base_ids), expected=int(header["base_n"]))
        if any(x < 0 or x >= pool_n for x in base_ids):
            err(errors, "initial_base_id_out_of_range")
    except Exception as exc:
        err(errors, "payload_read_error", reason=str(exc))
        pool = np.empty((0, 0), np.int16)
        queries = np.empty((0, 0), np.int16)
        base_ids = []
    trace: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    try:
        trace = [json.loads(x) for x in (bundle / "trace.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        oracle = [json.loads(x) for x in (bundle / "oracle_expected.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    except Exception as exc:
        err(errors, "jsonl_read_error", reason=str(exc))
    return metadata, pool, queries, base_ids, trace, oracle, selection, errors


def partition_violations(
    active: set[int], frozen_base: set[int], source_direct: set[int], source_delta: set[int]
) -> list[str]:
    issues: list[str] = []
    if frozen_base & source_direct:
        issues.append("base_overlaps_sidecar")
    if frozen_base & source_delta:
        issues.append("base_overlaps_delta")
    if source_direct & source_delta:
        issues.append("sidecar_overlaps_delta")
    if active != frozen_base | source_direct | source_delta:
        issues.append("active_not_exact_partition")
    return issues


def _synthetic_materialize(
    leaf_payloads: dict[int, tuple[int, list[int]]], visited_leaf_ids: Iterable[int],
    id_list_capacity: int, local_to_stable: list[int], native_res_ids: Iterable[int]
) -> list[list[int]]:
    """Pure host model of the P0 all-physical-row receipt contract."""
    rows: list[list[int]] = []
    seen_local: set[int] = set()
    for leaf in sorted(set(int(x) for x in visited_leaf_ids)):
        if leaf not in leaf_payloads:
            raise ValueError("receipt leaf unavailable")
        lid, physical_locals = leaf_payloads[leaf]
        if lid < 0 or lid + len(physical_locals) > id_list_capacity:
            raise ValueError("receipt leaf span outside padded id_list capacity")
        for offset, local in enumerate(physical_locals):
            if local < 0 or local >= len(local_to_stable):
                raise ValueError("receipt local outside compact map")
            if local in seen_local:
                raise ValueError("cross-leaf duplicate compact local row")
            seen_local.add(local)
            rows.append([leaf, lid + offset, local, local_to_stable[local]])
    receipt_locals = {row[2] for row in rows}
    for local in native_res_ids:
        if local == -1:
            continue
        if local not in receipt_locals:
            raise ValueError("native res_ids diagnostic not attributable to receipt")
    return rows


def padded_receipt_materializer_static_contract() -> dict[str, Any]:
    """Small CPU-only regression for padding, full leaves, and diagnostic res_ids.

    It intentionally uses a 25-row leaf at physical `lid=64` while logical N
    is 25.  Thus a legacy `MAX_SIZE=20` cap or an `lid+size <= N` assumption
    would fail this contract.
    """
    errors: list[str] = []
    base_count = 25
    capacity = base_count + LEAF_PAD_SLOTS + LEAF_PAD_SLOTS
    local_to_stable = [1000 + row for row in range(base_count)]
    primary = {7: (LEAF_PAD_SLOTS, list(range(base_count)))}
    happy_rows: list[list[int]] = []
    try:
        happy_rows = _synthetic_materialize(primary, [7, 7], capacity, local_to_stable, [0, 24, -1])
        if len(happy_rows) != base_count:
            errors.append("full_physical_leaf_not_materialized")
        if {row[2] for row in happy_rows} != set(range(base_count)):
            errors.append("receipt_local_rows_not_exact_full_leaf")
        if min(row[1] for row in happy_rows) != LEAF_PAD_SLOTS:
            errors.append("padded_lid_not_preserved")
        if max(row[1] for row in happy_rows) < base_count:
            errors.append("test_did_not_exercise_slot_beyond_logical_N")
    except Exception as exc:
        errors.append(f"happy_path_failed:{exc}")
    try:
        _synthetic_materialize({**primary, 8: (LEAF_PAD_SLOTS + base_count, [3])}, [7, 8],
                               capacity, local_to_stable, [])
        errors.append("cross_leaf_duplicate_not_rejected")
    except ValueError as exc:
        if "cross-leaf duplicate" not in str(exc):
            errors.append(f"wrong_cross_leaf_failure:{exc}")
    try:
        _synthetic_materialize(primary, [7], capacity, local_to_stable, [base_count])
        errors.append("out_of_receipt_res_id_not_rejected")
    except ValueError as exc:
        if "not attributable" not in str(exc):
            errors.append(f"wrong_res_id_failure:{exc}")
    return {
        "scope": "synthetic CPU host contract only; not a native GTS execution",
        "pass": not errors,
        "logical_base_count": base_count,
        "physical_id_list_capacity": capacity,
        "leaf_lid": LEAF_PAD_SLOTS,
        "leaf_size": base_count,
        "materialized_rows": len(happy_rows),
        "res_ids_diagnostic_only": True,
        "errors": errors,
    }


def fixture_only(bundle: Path) -> dict[str, Any]:
    metadata, pool, queries, base_ids, trace, oracle, selection, errors = load_bundle(bundle)
    header = metadata.get("header", {})
    pool_n = int(header.get("pool_n", 0))
    k = int(header.get("k", 0))
    if len(base_ids) != len(set(base_ids)):
        err(errors, "frozen_base_not_unique")

    # Source-plan state models the fixture’s deliberately marked selector plan.
    # It is useful for exercising delete/capacity/rebuild roles, but is never
    # native evidence.  The guard-closed model tracks current source behavior.
    active: set[int] = set(map(int, base_ids))
    source_direct: set[int] = set()
    source_delta: set[int] = set()
    source_placement: dict[int, str] = {sid: "base" for sid in active}
    guard_active: set[int] = set(active)
    guard_delta: set[int] = set()
    frozen_base_ids: list[int] = [int(x) for x in base_ids]
    frozen_generation = 1
    frozen_epochs: list[dict[str, Any]] = [{
        "generation": frozen_generation,
        "start_op_index": 0,
        **fixture_frozen_identity(frozen_base_ids),
    }]
    frozen_assertions = 0
    draft_commit_checks = 0
    source_direct_witnesses = 0
    guard_closed_insertions = 0
    expected_query_count = 0
    rebuild_contracts: list[dict[str, Any]] = []
    pending_after_rebuild: set[str] = set()
    observed_after_rebuild: list[dict[str, Any]] = []
    oracle_by_op = {row.get("op_index"): row for row in oracle if isinstance(row, dict)}
    selection_events = selection.get("events") if isinstance(selection.get("events"), list) else []
    selection_insert_events = [x for x in selection_events if isinstance(x, dict)]
    selection_insert_expected: Counter[tuple[int, str]] = Counter(
        (int(x["stable_id"]), str(x["role"]))
        for x in selection_insert_events
        if isinstance(x.get("stable_id"), int) and isinstance(x.get("role"), str)
    )

    if selection.get("selector") != "cpu_source_mirror_not_native_receipt":
        err(errors, "selection_not_marked_source_mirror")
    if not isinstance(selection.get("events"), list):
        err(errors, "selection_events_missing")

    padded_contract = padded_receipt_materializer_static_contract()
    if not padded_contract["pass"]:
        err(errors, "padded_receipt_materializer_static_contract_failed", details=padded_contract["errors"])

    def check_frozen_and_partition(op_index: int, stage: str) -> None:
        nonlocal frozen_assertions
        frozen_set = set(frozen_base_ids)
        identity = fixture_frozen_identity(frozen_base_ids)
        expected_epoch = frozen_epochs[-1]
        if any(identity[key] != expected_epoch[key] for key in identity):
            err(errors, "frozen_base_identity_changed_without_rebuild", op_index=op_index, stage=stage)
        for issue in partition_violations(active, frozen_set, source_direct, source_delta):
            err(errors, "source_plan_partition_violation", op_index=op_index, stage=stage, issue=issue)
        if guard_active != active:
            err(errors, "guard_closed_active_set_diverges_from_source_plan", op_index=op_index, stage=stage)
        if guard_delta != active - frozen_set:
            err(errors, "guard_closed_delta_not_all_dynamic_ids", op_index=op_index, stage=stage)
        frozen_assertions += 1

    def commit_source_plan_insert(stable: int, role: str, op_index: int) -> None:
        nonlocal active, source_direct, source_delta, source_placement, draft_commit_checks, source_direct_witnesses
        candidate_active = set(active)
        candidate_direct = set(source_direct)
        candidate_delta = set(source_delta)
        candidate_placement = dict(source_placement)
        candidate_active.add(stable)
        if role in {"direct", "direct_same_leaf"}:
            candidate_direct.add(stable)
            candidate_placement[stable] = "direct"
            source_direct_witnesses += 1
        else:
            candidate_delta.add(stable)
            candidate_placement[stable] = "delta"
        violations = partition_violations(candidate_active, set(frozen_base_ids), candidate_direct, candidate_delta)
        if violations:
            err(errors, "draft_insert_partition_failure", op_index=op_index, issues=violations)
            return
        active, source_direct, source_delta, source_placement = (
            candidate_active, candidate_direct, candidate_delta, candidate_placement
        )
        draft_commit_checks += 1

    def commit_source_plan_delete(stable: int, op_index: int) -> None:
        nonlocal active, source_direct, source_delta, source_placement, draft_commit_checks
        candidate_active = set(active)
        candidate_direct = set(source_direct)
        candidate_delta = set(source_delta)
        candidate_placement = dict(source_placement)
        candidate_active.remove(stable)
        candidate_direct.discard(stable)
        candidate_delta.discard(stable)
        candidate_placement[stable] = "deleted"
        violations = partition_violations(candidate_active, set(frozen_base_ids), candidate_direct, candidate_delta)
        if violations:
            err(errors, "draft_delete_partition_failure", op_index=op_index, issues=violations)
            return
        active, source_direct, source_delta, source_placement = (
            candidate_active, candidate_direct, candidate_delta, candidate_placement
        )
        draft_commit_checks += 1

    for position, event in enumerate(trace):
        if not isinstance(event, dict) or event.get("op_index") != position:
            err(errors, "noncontiguous_trace", position=position,
                observed=event.get("op_index") if isinstance(event, dict) else None)
            continue
        op = event.get("op")
        check_frozen_and_partition(position, "before")
        if pending_after_rebuild and op not in {"knn", "range"}:
            err(errors, "mutation_or_rebuild_before_post_rebuild_oracle_completion",
                op_index=position, op=op, pending=sorted(pending_after_rebuild))

        if op == "insert":
            sid = event.get("stable_id")
            role = event.get("expected_role")
            if not isinstance(sid, int) or sid < 0 or sid >= pool_n:
                err(errors, "invalid_insert_id", op_index=position, stable_id=sid)
                continue
            if sid in active:
                err(errors, "duplicate_insert", op_index=position, stable_id=sid)
                continue
            if role not in {"direct", "direct_same_leaf", "capacity_delta", "certificate_delta"}:
                err(errors, "unknown_insert_role", op_index=position, role=role)
                continue
            selection_key = (sid, str(role))
            if selection_insert_expected[selection_key] <= 0:
                err(errors, "selection_insert_event_missing_or_mismatched", op_index=position,
                    fixture_stable_id=sid, fixture_role=role)
            else:
                selection_insert_expected[selection_key] -= 1
            # Long mixed fixtures intentionally have no actual-tree leaf
            # binding.  Such entries remain source-mirror selection witnesses;
            # a missing expected_leaf is not silently promoted to a native
            # direct placement because the current runtime gate is closed.
            if event.get("expected_leaf") is not None and not isinstance(event.get("expected_leaf"), int):
                err(errors, "provisional_leaf_not_integer", op_index=position)
            commit_source_plan_insert(sid, str(role), position)
            # Current source behavior: the closed gate forces every non-base
            # insert into the global delta, even source-mirror direct witnesses.
            guard_active.add(sid)
            guard_delta.add(sid)
            guard_closed_insertions += 1

        elif op == "delete":
            sid = event.get("stable_id")
            if not isinstance(sid, int) or sid not in active:
                err(errors, "invalid_delete", op_index=position, stable_id=sid)
                continue
            prior = source_placement.get(sid)
            if prior == "base":
                err(errors, "forbidden_base_delete", op_index=position, stable_id=sid)
                continue
            if event.get("expected_prior") != prior:
                err(errors, "fixture_delete_prior_mismatch", op_index=position,
                    expected=prior, fixture_expected=event.get("expected_prior"))
            if sid not in guard_delta:
                err(errors, "guard_closed_delete_not_dynamic", op_index=position, stable_id=sid)
            commit_source_plan_delete(sid, position)
            guard_active.discard(sid)
            guard_delta.discard(sid)

        elif op == "rebuild":
            expected_hash = stable_set_sha256(active)
            if int(event.get("expected_delta_live", -1)) <= 0:
                err(errors, "bad_rebuild_threshold", op_index=position)
            if event.get("expected_live_ids_sha256") != expected_hash:
                err(errors, "bad_rebuild_active_hash", op_index=position)
            if event.get("expected_delta_live") != len(source_delta):
                err(errors, "fixture_rebuild_delta_count_mismatch", op_index=position,
                    expected=event.get("expected_delta_live"), observed=len(source_delta))
            old_identity = fixture_frozen_identity(frozen_base_ids)
            source_direct_before_rebuild = len(source_direct)
            source_delta_before_rebuild = len(source_delta)
            guard_delta_before_rebuild = len(guard_delta)
            planned_local_to_stable = sorted(active)
            stable_to_local = {stable: local for local, stable in enumerate(planned_local_to_stable)}
            if len(stable_to_local) != len(active):
                err(errors, "rebuild_mapping_not_bijection", op_index=position)
            # The replacement is only published after the planned new base and
            # empty dynamic tiers satisfy the same partition invariant.
            replacement_base = set(planned_local_to_stable)
            violations = partition_violations(replacement_base, replacement_base, set(), set())
            if violations:
                err(errors, "rebuild_replacement_partition_failure", op_index=position, issues=violations)
                continue
            frozen_base_ids = planned_local_to_stable
            frozen_generation += 1
            active = set(replacement_base)
            source_direct.clear()
            source_delta.clear()
            source_placement = {sid: "base" for sid in active}
            guard_active = set(active)
            guard_delta.clear()
            new_identity = fixture_frozen_identity(frozen_base_ids)
            frozen_epochs.append({
                "generation": frozen_generation,
                "start_op_index": position,
                **new_identity,
            })
            rebuild_contracts.append({
                "op_index": position,
                "old_fixture_frozen_base": old_identity,
                "new_fixture_frozen_base": new_identity,
                "live_ids_sha256": expected_hash,
                "base_count": len(active),
                "planned_local_to_stable_sha256": local_to_stable_sha256(planned_local_to_stable),
                "planned_local_rows_cover": f"[0,{len(active)})",
                "planned_stable_to_local_bijection": len(stable_to_local) == len(active),
                "source_plan_sidecar_live_before_rebuild": source_direct_before_rebuild,
                "source_plan_delta_live_before_rebuild": source_delta_before_rebuild,
                "guard_closed_delta_live_before_rebuild": guard_delta_before_rebuild,
                "destructive_rebuild_contract_only": True,
                "post_rebuild_oracle_required": ["knn", "range"],
                "native_runtime_checks_required": [
                    "actual_leaf_raw_local_rows_are_permutation_of_0_to_N_minus_1",
                    "actual_leaf_local_to_stable_set_equals_all_and_only_live_ids",
                    "actual_tree_version_increments",
                    "actual_padded_id_list_spans_are_within_capacity",
                    "actual_sidecar_and_delta_empty_after_generation_publish",
                    "independent_first_KNN_and_range_oracles_complete_before_mutation",
                ],
            })
            # Preserve observed pre-rebuild values in record above; after the
            # destructive replacement dynamic tiers must be empty.
            pending_after_rebuild = {"knn", "range"}
            draft_commit_checks += 1

        elif op in {"knn", "range"}:
            expected_query_count += 1
            qid = event.get("query_id")
            if not isinstance(qid, int) or qid < 0 or qid >= len(queries):
                err(errors, "bad_external_vector_query_id", op_index=position, query_id=qid)
                continue
            if op == "knn":
                expected = exact_results(pool, queries[qid], active, "knn", k)
            else:
                radius = event.get("radius_sq")
                if not isinstance(radius, int) or radius < 0:
                    err(errors, "bad_range_radius", op_index=position, radius_sq=radius)
                    continue
                expected = exact_results(pool, queries[qid], active, "range", k, radius)
            stored = oracle_by_op.get(position)
            if not isinstance(stored, dict):
                err(errors, "missing_oracle_record", op_index=position)
            elif stored.get("results") != expected or stored.get("active_ids_sha256") != stable_set_sha256(active):
                err(errors, "oracle_mismatch", op_index=position)
            if op in pending_after_rebuild:
                observed_after_rebuild.append({
                    "op_index": position,
                    "kind": op,
                    "oracle_rows": len(expected),
                    "independent_fixture_oracle_only": True,
                })
                pending_after_rebuild.remove(op)
        else:
            err(errors, "unknown_trace_op", op_index=position, op=op)
        check_frozen_and_partition(position, "after")

    unconsumed_selection = [
        {"stable_id": stable, "role": role, "count": count}
        for (stable, role), count in sorted(selection_insert_expected.items()) if count
    ]
    if unconsumed_selection:
        err(errors, "unconsumed_selection_insert_events", entries=unconsumed_selection[:100])
    if len(oracle_by_op) != expected_query_count:
        err(errors, "oracle_query_count_mismatch", expected=expected_query_count, actual=len(oracle_by_op))
    if pending_after_rebuild:
        err(errors, "missing_post_rebuild_query_kind", kinds=sorted(pending_after_rebuild))

    return {
        "schema": "safe-c1-g3-fixture-check-v4-p0",
        "status": "PASS" if not errors else "FAIL",
        "gpu_used": False,
        "scope": "CPU fixture/source-mirror and synthetic receipt contracts only; no native execution, correctness result, or performance claim",
        "bundle": str(bundle),
        "trace_events": len(trace),
        "oracle_queries": expected_query_count,
        "source_mirror_selection_only": True,
        "direct_runtime_guard": {
            "open": DIRECT_SIDECAR_RUNTIME_GUARD_OPEN,
            "source_mirror_direct_witnesses": source_direct_witnesses,
            "guard_closed_insertions_modeled_as_delta": guard_closed_insertions,
            "not_native_direct_visibility_evidence": True,
        },
        "frozen_base_epochs": frozen_epochs,
        "frozen_base_invariant_assertions": frozen_assertions,
        "draft_partition_commit_checks": draft_commit_checks,
        "padded_receipt_materializer_static_contract": padded_contract,
        "rebuild_contracts": rebuild_contracts,
        "post_rebuild_vector_oracle_queries": observed_after_rebuild,
        "range_contract_scope": {
            "fixture_range_answers_are_independent_CPU_oracles": True,
            "native_range_correctness_not_claimed": True,
            "direct_range_correctness_not_claimed": True,
        },
        "error_count": len(errors),
        "errors": errors[:200],
    }


def validate_engine(bundle: Path, engine_path: Path) -> dict[str, Any]:
    """Validate a future JSONL receipt against the P0 source contract.

    This function still runs only CPU code.  It does not make an engine trace
    native evidence by itself; it rejects traces that contradict the current
    direct-gate, frozen-generation, receipt, or post-rebuild schema.
    """
    fixture = fixture_only(bundle)
    metadata, pool, queries, base_ids, trace, _, _, errors = load_bundle(bundle)
    errors.extend(fixture.get("errors", []))
    records, parse_errors = load_jsonl(engine_path)
    errors.extend(parse_errors)
    header = metadata["header"]
    k = int(header["k"])

    current_base = sorted(map(int, base_ids))
    active: set[int] = set(current_base)
    delta: set[int] = set()
    placement: dict[int, str] = {sid: "base" for sid in active}
    tree_version = 1
    tree_payload_by_generation: str | None = None
    base_payload_by_generation: str | None = None
    native_tree_bytes_by_generation: str | None = None
    post_rebuild_pending: set[str] = set()
    delta_seen = rebuild_seen = 0
    query_receipt_rows = 0

    def expected_frozen_set_hash() -> str:
        return stable_set_sha256(current_base)

    def expected_mapping_hash() -> str:
        return local_to_stable_sha256(current_base)

    for event in trace:
        oi = int(event["op_index"])
        op = event["op"]
        row = records.pop(oi, None)
        if row is None:
            err(errors, "missing_engine_record", op_index=oi, op=op)
            continue
        if post_rebuild_pending and op not in {"knn", "range"}:
            err(errors, "engine_mutation_or_rebuild_before_post_rebuild_oracle_completion",
                op_index=oi, pending=sorted(post_rebuild_pending))

        if op == "insert":
            sid = event["stable_id"]
            if row.get("record") != "update" or row.get("op") != "insert" or row.get("stable_id") != sid:
                err(errors, "bad_insert_record", op_index=oi)
                continue
            if row.get("direct_runtime_guard_open") is not False:
                err(errors, "direct_runtime_guard_not_closed", op_index=oi)
            if row.get("placement") != "delta":
                err(errors, "direct_placement_forbidden_while_guard_closed", op_index=oi,
                    placement=row.get("placement"))
            if row.get("sidecar_leaf_id") not in {-1, None}:
                err(errors, "sidecar_leaf_present_while_direct_guard_closed", op_index=oi)
            if sid in active:
                err(errors, "engine_duplicate_insert", op_index=oi, stable_id=sid)
                continue
            # A source-mirror direct witness can have a true certificate, but
            # current source still falls back to delta.  If it says so, demand
            # the explicit gate reason; certificate failure has its own reason.
            if row.get("certificate_ok") is True and row.get("fallback_reason") != "direct_visibility_runtime_guard_closed":
                err(errors, "guard_closed_certificate_true_missing_fallback_reason", op_index=oi)
            active.add(sid)
            delta.add(sid)
            placement[sid] = "delta"
            delta_seen += 1

        elif op == "delete":
            sid = event["stable_id"]
            if row.get("record") != "update" or row.get("op") != "delete" or row.get("stable_id") != sid:
                err(errors, "bad_delete_record", op_index=oi)
                continue
            prior = placement.get(sid)
            if prior != "delta":
                err(errors, "engine_delete_not_current_guard_closed_dynamic_delta", op_index=oi, prior=prior)
            if row.get("prior_placement") != prior:
                err(errors, "delete_prior_mismatch", op_index=oi, expected=prior,
                    got=row.get("prior_placement"))
            # Fixture expected_prior belongs to the source-mirror selector and
            # is intentionally not imposed on the current closed-gate trace.
            active.discard(sid)
            delta.discard(sid)
            placement[sid] = "deleted"

        elif op == "rebuild":
            if row.get("record") != "rebuild":
                err(errors, "bad_rebuild_record", op_index=oi)
                continue
            expected_live_hash = stable_set_sha256(active)
            if row.get("live_ids_sha256") != expected_live_hash:
                err(errors, "rebuild_active_hash_mismatch", op_index=oi)
            if row.get("immutable_base_stable_ids_sha256") != expected_live_hash:
                err(errors, "rebuild_immutable_base_hash_mismatch", op_index=oi)
            if row.get("base_count") != len(active):
                err(errors, "rebuild_base_count_mismatch", op_index=oi)
            if row.get("sidecar_live") != 0 or row.get("delta_live") != 0:
                err(errors, "rebuild_nonempty_dynamic_tier", op_index=oi)
            for field in (
                "compact_mapping_bijection_ok", "raw_leaf_rows_cover_compact_range",
                "destructive_fail_stop_contract", "old_generation_retired_before_candidate_build",
                "dynamic_tiers_replaced_after_generation_publish",
                "post_rebuild_knn_oracle_required", "post_rebuild_range_oracle_required",
            ):
                if row.get(field) is not True:
                    err(errors, "missing_or_false_rebuild_p0_attestation", op_index=oi, field=field)
            for field in (
                "immutable_base_payload_sha256", "native_tree_bytes_sha256",
                "local_to_stable_sha256", "logical_leaf_stable_ids_sha256", "tree_payload_sha256",
            ):
                if not is_sha256(row.get(field)):
                    err(errors, "rebuild_hash_missing", op_index=oi, field=field)
            if not isinstance(row.get("tree_version"), int) or row["tree_version"] <= tree_version:
                err(errors, "bad_tree_version", op_index=oi)
            else:
                tree_version = int(row["tree_version"])
            current_base = sorted(active)
            if row.get("local_to_stable_sha256") != expected_mapping_hash():
                err(errors, "rebuild_mapping_hash_mismatch", op_index=oi)
            placement = {sid: "base" for sid in active}
            delta.clear()
            tree_payload_by_generation = row.get("tree_payload_sha256") if is_sha256(row.get("tree_payload_sha256")) else None
            base_payload_by_generation = row.get("immutable_base_payload_sha256") if is_sha256(row.get("immutable_base_payload_sha256")) else None
            native_tree_bytes_by_generation = row.get("native_tree_bytes_sha256") if is_sha256(row.get("native_tree_bytes_sha256")) else None
            post_rebuild_pending = {"knn", "range"}
            rebuild_seen += 1

        elif op in {"knn", "range"}:
            if row.get("record") != "query" or row.get("kind") != op or row.get("query_id") != event.get("query_id"):
                err(errors, "bad_query_record", op_index=oi)
                continue
            if row.get("production_full_live_scan") is not False:
                err(errors, "forbidden_production_full_live_scan", op_index=oi)
            if row.get("tree_version") != tree_version:
                err(errors, "query_tree_version_mismatch", op_index=oi)
            if row.get("receipt_before_leaf_materialization") is not True:
                err(errors, "generic_receipt_order_missing", op_index=oi)
            if isinstance(row.get("id_list_capacity"), bool) or not isinstance(row.get("id_list_capacity"), int) or row.get("id_list_capacity") <= 0:
                err(errors, "missing_or_invalid_padded_id_list_capacity", op_index=oi)
            if row.get("immutable_base_stable_ids_sha256") != expected_frozen_set_hash():
                err(errors, "query_frozen_base_set_hash_mismatch", op_index=oi)
            if row.get("local_to_stable_sha256") != expected_mapping_hash():
                err(errors, "query_frozen_mapping_hash_mismatch", op_index=oi)
            for field in ("tree_payload_sha256", "immutable_base_payload_sha256", "native_tree_bytes_sha256"):
                if not is_sha256(row.get(field)):
                    err(errors, "query_frozen_hash_missing", op_index=oi, field=field)
            # Establish per-generation identities on the first observed query;
            # then require immutable values until an explicit rebuild.
            if tree_payload_by_generation is None and is_sha256(row.get("tree_payload_sha256")):
                tree_payload_by_generation = row.get("tree_payload_sha256")
                base_payload_by_generation = row.get("immutable_base_payload_sha256")
                native_tree_bytes_by_generation = row.get("native_tree_bytes_sha256")
            elif (row.get("tree_payload_sha256") != tree_payload_by_generation or
                  row.get("immutable_base_payload_sha256") != base_payload_by_generation or
                  row.get("native_tree_bytes_sha256") != native_tree_bytes_by_generation):
                err(errors, "frozen_native_identity_changed_without_rebuild", op_index=oi)

            if op == "knn":
                if row.get("base_path") != "real_gts_vector_topk_receipt_all_native_leaf_rows":
                    err(errors, "wrong_knn_base_path", op_index=oi)
                if row.get("native_final_res_ids_cross_checked") is not True:
                    err(errors, "knn_res_ids_not_diagnostic_cross_checked", op_index=oi)
            else:
                if row.get("base_path") != "gts_vector_range_branch_aligned_predicate_mirror_receipt":
                    err(errors, "wrong_range_base_path", op_index=oi)
                if row.get("range_receipt_before_leaf_materialization") is not True:
                    err(errors, "range_receipt_order_missing", op_index=oi)
                if row.get("range_predicate_mirror_branch_aligned") is not True:
                    err(errors, "range_branch_aligned_mirror_missing", op_index=oi)
                if row.get("direct_sidecars_exact_range_filtered") is not True:
                    err(errors, "range_direct_sidecars_not_exact_filtered", op_index=oi)
                if row.get("direct_range_correctness_claim") is True:
                    err(errors, "forbidden_direct_range_correctness_claim", op_index=oi)

            try:
                observed = as_results(row.get("results"))
                base_observed = as_results(row.get("base_results"))
                visited = ensure_int_list(row.get("gts_visited_leaf_ids"), "gts_visited_leaf_ids")
                if any(x < 0 for x in visited):
                    raise ValueError("negative leaf ID")
                receipt_spans = as_receipt_leaf_spans(row.get("receipt_leaf_spans"))
                receipt_rows = as_receipt_base_rows(row.get("base_receipt_rows"))
                native_res_ids = row.get("native_final_res_ids")
                if native_res_ids is None:
                    native_res_ids = []
                native_res_ids = as_native_res_ids(native_res_ids)
                side_ids = ensure_int_list(row.get("sidecar_candidate_ids"), "sidecar_candidate_ids")
                delta_ids = ensure_int_list(row.get("delta_candidate_ids"), "delta_candidate_ids")
                side_pairs = as_leaf_pairs(row.get("sidecar_source_leaf_pairs"))
            except ValueError as exc:
                err(errors, "receipt_or_results_malformed", op_index=oi, reason=str(exc))
                observed = []
                base_observed = []
                visited = []
                receipt_spans = []
                receipt_rows = []
                native_res_ids = []
                side_ids = []
                delta_ids = []
                side_pairs = []

            local_to_stable = current_base
            stable_to_local = {stable: local for local, stable in enumerate(local_to_stable)}
            visited_set = set(visited)
            span_by_leaf = {span[0]: span for span in receipt_spans}
            if set(span_by_leaf) != visited_set:
                err(errors, "receipt_spans_not_exactly_visited_leaves", op_index=oi,
                    spans=sorted(span_by_leaf), visited=sorted(visited_set))
            capacity = row.get("id_list_capacity") if isinstance(row.get("id_list_capacity"), int) else -1
            expected_slots_by_leaf: dict[int, set[int]] = {}
            for leaf, lid, size in receipt_spans:
                if lid + size > capacity:
                    err(errors, "receipt_span_exceeds_padded_id_list_capacity", op_index=oi,
                        leaf_id=leaf, lid=lid, size=size, capacity=capacity)
                expected_slots_by_leaf[leaf] = set(range(lid, lid + size))
            observed_slots_by_leaf: dict[int, set[int]] = {leaf: set() for leaf in span_by_leaf}
            receipt_locals = {receipt[2] for receipt in receipt_rows}
            receipt_stables = {receipt[3] for receipt in receipt_rows}
            for leaf, slot, local, stable in receipt_rows:
                if local >= len(local_to_stable) or local_to_stable[local] != stable:
                    err(errors, "receipt_local_to_stable_mapping_mismatch", op_index=oi,
                        leaf_id=leaf, id_list_slot=slot, local_row=local, stable_id=stable)
                if leaf not in span_by_leaf:
                    err(errors, "receipt_row_leaf_without_selected_span", op_index=oi, leaf_id=leaf)
                else:
                    observed_slots_by_leaf[leaf].add(slot)
            for leaf, expected_slots in expected_slots_by_leaf.items():
                if observed_slots_by_leaf.get(leaf, set()) != expected_slots:
                    err(errors, "receipt_rows_do_not_cover_exact_full_physical_span", op_index=oi,
                        leaf_id=leaf, expected_count=len(expected_slots),
                        observed_count=len(observed_slots_by_leaf.get(leaf, set())))
            if op == "knn":
                for local in native_res_ids:
                    if local != -1 and local not in receipt_locals:
                        err(errors, "res_ids_not_diagnostic_receipt_subset", op_index=oi, local_row=local)
            elif native_res_ids or row.get("native_final_res_ids_cross_checked") is not False:
                err(errors, "range_must_not_export_knn_res_ids_diagnostic", op_index=oi)
            for sid, _d2 in base_observed:
                if sid not in receipt_stables or stable_to_local.get(sid) not in receipt_locals:
                    err(errors, "base_result_not_attributable_to_full_receipt_rows", op_index=oi, stable_id=sid)
            qid_for_base = int(event["query_id"])
            base_expected_from_receipt = exact_receipt_results(
                pool, queries[qid_for_base], receipt_stables, op,
                event.get("radius_sq") if op == "range" else None,
            )
            if base_observed != base_expected_from_receipt:
                err(errors, "base_results_not_exactly_full_receipt_scan", op_index=oi)
            query_receipt_rows += len(receipt_rows)

            if side_ids or side_pairs:
                err(errors, "direct_sidecars_present_while_runtime_guard_closed", op_index=oi)
            if delta_ids != sorted(delta):
                err(errors, "delta_not_global_or_exact", op_index=oi)

            qid = int(event["query_id"])
            radius = event.get("radius_sq")
            expected = exact_results(pool, queries[qid], active, op, k, radius)
            base_expected = exact_results(pool, queries[qid], set(current_base), op, k, radius)
            if observed != expected:
                err(errors, "exact_answer_mismatch", op_index=oi)
            if base_observed != base_expected:
                err(errors, "base_stable_export_mismatch", op_index=oi)

            if op in post_rebuild_pending:
                if row.get("post_rebuild_oracle_required") is not True:
                    err(errors, "post_rebuild_query_not_marked_required", op_index=oi, kind=op)
                if row.get("post_rebuild_oracle_match") is not True:
                    err(errors, "post_rebuild_oracle_not_attested", op_index=oi, kind=op)
                post_rebuild_pending.remove(op)
            elif row.get("post_rebuild_oracle_required") is True:
                err(errors, "unexpected_post_rebuild_oracle_required", op_index=oi, kind=op)
        else:
            err(errors, "unknown_trace_op", op_index=oi)

    if post_rebuild_pending:
        err(errors, "missing_post_rebuild_engine_oracle_attestation", kinds=sorted(post_rebuild_pending))
    if records:
        err(errors, "unexpected_engine_records", op_indices=sorted(records)[:50])
    return {
        "schema": "safe-c1-g3-native-validator-v4-p0",
        "status": "PASS" if not errors else "FAIL",
        "gpu_used": False,
        "scope": "independent CPU validation of a future receipt schema only; no native execution or performance claim",
        "bundle": str(bundle),
        "engine_jsonl": str(engine_path),
        "direct_runtime_guard_open_required": False,
        "delta_inserts": delta_seen,
        "rebuilds": rebuild_seen,
        "receipt_rows_observed": query_receipt_rows,
        "error_count": len(errors),
        "errors": errors[:200],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--engine-jsonl", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    result = validate_engine(bundle, args.engine_jsonl.resolve()) if args.engine_jsonl else fixture_only(bundle)
    write_json(args.out, result)
    print(json.dumps({"status": result["status"], "gpu_used": False, "error_count": result["error_count"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
