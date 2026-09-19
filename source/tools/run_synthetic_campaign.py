#!/usr/bin/env python3
"""One-shot CPU-only synthetic E2/E3 receipt-sidecar vs exact-delta runner."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import heapq
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

EXPECTED_UID = 1001
RUN_SCHEMA = "tide.synthetic-e2e3-receipt-sidecar-execution.v2"
CELL_EDGE = 64
CELLS_PER_AXIS = 4
Point = Tuple[int, int, int]
Ranked = Tuple[int, int]
Overlay = Dict[int, Optional[Point]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_bytes(path: Path, payload: bytes) -> None:
    require(path.parent.is_dir(), f"missing parent for {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    write_new_bytes(path, canonical_bytes(value))


def write_new_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_new_bytes(path, b"".join(canonical_bytes(row) for row in rows))


def mkdir_new(path: Path) -> None:
    os.mkdir(path, 0o700)


def point_for_id(stable_id: int) -> Point:
    require(stable_id >= 0, "negative stable ID")
    return ((stable_id * 17 + 13) & 255, (stable_id * 31 + 29) & 255, (stable_id * 47 + 71) & 255)


def squared_distance(left: Point, right: Point) -> int:
    dx = left[0] - right[0]
    dy = left[1] - right[1]
    dz = left[2] - right[2]
    return dx * dx + dy * dy + dz * dz


def receipt_leaf_for_point(point: Point) -> int:
    return ((point[0] // CELL_EDGE) * CELLS_PER_AXIS * CELLS_PER_AXIS +
            (point[1] // CELL_EDGE) * CELLS_PER_AXIS + (point[2] // CELL_EDGE))


def receipt_leaf_bounds(leaf_id: int) -> Tuple[Point, Point]:
    require(0 <= leaf_id < CELLS_PER_AXIS ** 3, "receipt leaf is out of range")
    xcell = leaf_id // (CELLS_PER_AXIS * CELLS_PER_AXIS)
    rem = leaf_id % (CELLS_PER_AXIS * CELLS_PER_AXIS)
    ycell = rem // CELLS_PER_AXIS
    zcell = rem % CELLS_PER_AXIS
    low = (xcell * CELL_EDGE, ycell * CELL_EDGE, zcell * CELL_EDGE)
    high = (low[0] + CELL_EDGE - 1, low[1] + CELL_EDGE - 1, low[2] + CELL_EDGE - 1)
    return low, high


def receipt_leaf_lower_bound_squared(query: Point, leaf_id: int) -> int:
    low, high = receipt_leaf_bounds(leaf_id)
    total = 0
    for coordinate, lo, hi in zip(query, low, high):
        delta = lo - coordinate if coordinate < lo else coordinate - hi if coordinate > hi else 0
        total += delta * delta
    return total


def catalog_identity(object_count: int) -> str:
    require(object_count > 0, "invalid base catalog size")
    return sha256_bytes(canonical_bytes({
        "kind": "implicit_synthetic_uint8_l2_v2",
        "object_count": object_count,
        "stable_id_range": [0, object_count - 1],
        "coordinate_formula": "((17*i+13)&255,(31*i+29)&255,(47*i+71)&255)",
        "receipt_cells": "4x4x4 axis-aligned uint8 cells of edge 64",
    }))


def trace_identity(label: str, operations: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_bytes({"kind": "synthetic_matched_trace_v2", "label": label, "operations": list(operations)}))


def semantic_state_identity(base_size: int, overlay: Overlay) -> str:
    rows: List[List[Any]] = []
    for stable_id in sorted(overlay):
        point = overlay[stable_id]
        rows.append([stable_id, None if point is None else list(point)])
    return sha256_bytes(canonical_bytes({"base_size": base_size, "semantic_overlay": rows}))


def offer_sorted(best: List[Ranked], candidate: Ranked, k: int) -> None:
    if len(best) < k:
        bisect.insort(best, candidate)
    elif candidate < best[-1]:
        bisect.insort(best, candidate)
        best.pop()


def scan_base_top_k(base_size: int, overrides: Overlay, query: Point, k: int) -> Tuple[List[Ranked], int]:
    best: List[Ranked] = []
    scanned = 0
    for stable_id in range(base_size):
        if stable_id in overrides:
            continue
        offer_sorted(best, (squared_distance(query, point_for_id(stable_id)), stable_id), k)
        scanned += 1
    require(len(best) == k, "base scan cannot provide k candidates")
    return best, scanned


def full_live_oracle_top_k(base_size: int, semantic_overlay: Overlay, query: Point, k: int) -> List[Ranked]:
    """Independently structured full-live oracle: a max heap over all live items."""
    heap: List[Tuple[int, int]] = []

    def offer(stable_id: int, point: Point) -> None:
        transformed = (-squared_distance(query, point), -stable_id)
        if len(heap) < k:
            heapq.heappush(heap, transformed)
        elif transformed > heap[0]:
            heapq.heapreplace(heap, transformed)

    for stable_id in range(base_size):
        if stable_id not in semantic_overlay:
            offer(stable_id, point_for_id(stable_id))
    for stable_id, point in semantic_overlay.items():
        if point is not None:
            offer(stable_id, point)
    result = sorted([(-distance, -stable_id) for distance, stable_id in heap])
    require(len(result) == k, "full-live oracle result has wrong cardinality")
    return result


def ranked_hash(rows: Sequence[Ranked]) -> str:
    return sha256_bytes(canonical_bytes({"ranked": [[distance, stable_id] for distance, stable_id in rows]}))


class CertifiedSidecarState:
    """Synthetic receipt-indexed state: eligible arrivals live under 64 cells."""

    def __init__(self, base_size: int) -> None:
        self.base_size = base_size
        self.sidecars: Dict[int, Dict[int, Point]] = {}
        self.global_delta: Overlay = {}
        self.pending_base_deletes: set[int] = set()
        self.barrier_epoch = 0

    def semantic_overlay(self) -> Overlay:
        merged: Overlay = dict(self.global_delta)
        for leaf_id in sorted(self.sidecars):
            for stable_id, point in self.sidecars[leaf_id].items():
                require(stable_id not in merged, "sidecar/global-delta semantic collision")
                merged[stable_id] = point
        return merged

    def _remove_sidecar_arrival(self, stable_id: int) -> bool:
        for leaf_id in list(self.sidecars):
            bucket = self.sidecars[leaf_id]
            if stable_id in bucket:
                del bucket[stable_id]
                if not bucket:
                    del self.sidecars[leaf_id]
                return True
        return False

    def apply(self, operation: Mapping[str, Any]) -> str:
        kind = operation["op"]
        if kind == "insert":
            stable_id = operation["stable_id"]
            point_values = operation["point"]
            route = operation["route"]
            require(isinstance(stable_id, int) and stable_id >= self.base_size, "sidecar insert ID is invalid")
            point = (int(point_values[0]), int(point_values[1]), int(point_values[2]))
            already_sidecar = any(stable_id in bucket for bucket in self.sidecars.values())
            require(stable_id not in self.global_delta and not already_sidecar,
                    "sidecar insert ID is not fresh")
            if route == "receipt_sidecar":
                leaf_id = receipt_leaf_for_point(point)
                require(operation.get("receipt_leaf") == leaf_id, "declared receipt leaf disagrees with point")
                self.sidecars.setdefault(leaf_id, {})[stable_id] = point
                return "PASS_RECEIPT_SIDECAR_ROUTE"
            require(route == "global_delta", "unknown synthetic arrival route")
            self.global_delta[stable_id] = point
            return "PASS_GLOBAL_DELTA_ROUTE"
        if kind == "delete":
            stable_id = operation["stable_id"]
            require(isinstance(stable_id, int) and stable_id >= 0, "sidecar delete ID is invalid")
            if stable_id < self.base_size:
                require(stable_id not in self.global_delta, "base delete repeats")
                self.global_delta[stable_id] = None
                self.pending_base_deletes.add(stable_id)
                return "PASS_BASE_DELETE_TOMBSTONE"
            if stable_id in self.global_delta and self.global_delta[stable_id] is not None:
                del self.global_delta[stable_id]
                return "PASS_GLOBAL_DELTA_DELETE"
            require(self._remove_sidecar_arrival(stable_id), "sidecar delete misses a live arrival")
            return "PASS_RECEIPT_SIDECAR_DELETE"
        if kind == "rebuild_barrier":
            require(self.pending_base_deletes, "synchronous barrier has no pending base delete")
            self.pending_base_deletes.clear()
            self.barrier_epoch += 1
            return "PASS_SYNCHRONOUS_REBUILD_BARRIER"
        raise ValueError("unsupported sidecar operation")

    def query_top_k(self, query: Point, k: int, capacity: int) -> Tuple[List[Ranked], Dict[str, Any]]:
        best, scanned_base = scan_base_top_k(self.base_size, self.global_delta, query, k)
        scanned_delta = 0
        for stable_id in sorted(self.global_delta):
            point = self.global_delta[stable_id]
            scanned_delta += 1
            if point is not None:
                offer_sorted(best, (squared_distance(query, point), stable_id), k)
        cutoff = best[-1][0]
        receipt_leaves = [leaf_id for leaf_id in sorted(self.sidecars)
                          if receipt_leaf_lower_bound_squared(query, leaf_id) <= cutoff]
        scanned_sidecar = 0
        for leaf_id in receipt_leaves:
            for stable_id, point in sorted(self.sidecars[leaf_id].items()):
                offer_sorted(best, (squared_distance(query, point), stable_id), k)
                scanned_sidecar += 1
        counts = self.counts(capacity)
        return best, {
            "scanned_base": scanned_base,
            "scanned_sidecar": scanned_sidecar,
            "scanned_delta": scanned_delta,
            "receipt_leaf_ids": receipt_leaves,
            "receipt_leaf_count": len(receipt_leaves),
            **counts,
        }

    def counts(self, capacity: int) -> Dict[str, int]:
        sidecar_count = sum(len(bucket) for bucket in self.sidecars.values())
        delta_count = len(self.global_delta)
        require(sidecar_count + delta_count <= capacity, "C_ov exceeded in Certified-Sidecar")
        return {"live_sidecars": sidecar_count, "live_delta": delta_count,
                "overlay_total": sidecar_count + delta_count, "C_ov": capacity}


class ExactDeltaState:
    """Synthetic reference state: every arrival is stored and scanned in global delta."""

    def __init__(self, base_size: int) -> None:
        self.base_size = base_size
        self.delta: Overlay = {}
        self.pending_base_deletes: set[int] = set()
        self.barrier_epoch = 0

    def semantic_overlay(self) -> Overlay:
        return dict(self.delta)

    def apply(self, operation: Mapping[str, Any]) -> str:
        kind = operation["op"]
        if kind == "insert":
            stable_id = operation["stable_id"]
            point_values = operation["point"]
            require(isinstance(stable_id, int) and stable_id >= self.base_size and stable_id not in self.delta,
                    "delta insert ID is invalid/not fresh")
            route = operation["route"]
            require(route in {"receipt_sidecar", "global_delta"}, "unknown synthetic arrival route")
            self.delta[stable_id] = (int(point_values[0]), int(point_values[1]), int(point_values[2]))
            return "PASS_ALL_ARRIVALS_GLOBAL_DELTA"
        if kind == "delete":
            stable_id = operation["stable_id"]
            require(isinstance(stable_id, int) and stable_id >= 0, "delta delete ID is invalid")
            if stable_id < self.base_size:
                require(stable_id not in self.delta, "base delete repeats")
                self.delta[stable_id] = None
                self.pending_base_deletes.add(stable_id)
                return "PASS_BASE_DELETE_TOMBSTONE"
            require(stable_id in self.delta and self.delta[stable_id] is not None, "delta delete misses live arrival")
            del self.delta[stable_id]
            return "PASS_GLOBAL_DELTA_DELETE"
        if kind == "rebuild_barrier":
            require(self.pending_base_deletes, "synchronous barrier has no pending base delete")
            self.pending_base_deletes.clear()
            self.barrier_epoch += 1
            return "PASS_SYNCHRONOUS_REBUILD_BARRIER"
        raise ValueError("unsupported delta operation")

    def query_top_k(self, query: Point, k: int, capacity: int) -> Tuple[List[Ranked], Dict[str, Any]]:
        best, scanned_base = scan_base_top_k(self.base_size, self.delta, query, k)
        scanned_delta = 0
        for stable_id in sorted(self.delta):
            point = self.delta[stable_id]
            scanned_delta += 1
            if point is not None:
                offer_sorted(best, (squared_distance(query, point), stable_id), k)
        counts = self.counts(capacity)
        return best, {
            "scanned_base": scanned_base,
            "scanned_sidecar": 0,
            "scanned_delta": scanned_delta,
            "receipt_leaf_ids": [],
            "receipt_leaf_count": 0,
            **counts,
        }

    def counts(self, capacity: int) -> Dict[str, int]:
        delta_count = len(self.delta)
        require(delta_count <= capacity, "C_ov exceeded in Exact-Delta")
        return {"live_sidecars": 0, "live_delta": delta_count,
                "overlay_total": delta_count, "C_ov": capacity}


def make_state(variant: str, base_size: int) -> Any:
    if variant == "certified_sidecar":
        return CertifiedSidecarState(base_size)
    if variant == "exact_delta":
        return ExactDeltaState(base_size)
    raise ValueError("unknown variant")


def make_e2_trace(base_size: int) -> List[Dict[str, Any]]:
    sidecar_id = base_size + 17
    sidecar_point = point_for_id(sidecar_id)
    delta_id = base_size + 19
    delta_point = point_for_id(delta_id)
    return [
        {"op": "query", "query": list(point_for_id(base_size * 3 + 1)), "label": "initial"},
        {"op": "insert", "stable_id": sidecar_id, "point": list(sidecar_point), "route": "receipt_sidecar", "receipt_leaf": receipt_leaf_for_point(sidecar_point), "label": "eligible_arrival"},
        {"op": "insert", "stable_id": delta_id, "point": list(delta_point), "route": "global_delta", "label": "ineligible_arrival"},
        {"op": "query", "query": list(sidecar_point), "label": "after_arrivals"},
        {"op": "delete", "stable_id": 17, "label": "base_delete"},
        {"op": "rebuild_barrier", "label": "synchronous_base_delete_barrier"},
        {"op": "query", "query": list(delta_point), "label": "after_rebuild_barrier"},
        {"op": "delete", "stable_id": sidecar_id, "label": "eligible_arrival_delete"},
        {"op": "query", "query": list(point_for_id(base_size * 3 + 4)), "label": "after_sidecar_delete"},
    ]


def make_e3_trace(base_size: int) -> List[Dict[str, Any]]:
    sidecar_id = base_size + 23
    sidecar_point = point_for_id(sidecar_id)
    delta_id = base_size + 29
    delta_point = point_for_id(delta_id)
    return [
        {"op": "insert", "stable_id": sidecar_id, "point": list(sidecar_point), "route": "receipt_sidecar", "receipt_leaf": receipt_leaf_for_point(sidecar_point), "label": "bounded_eligible_arrival"},
        {"op": "insert", "stable_id": delta_id, "point": list(delta_point), "route": "global_delta", "label": "bounded_ineligible_arrival"},
        {"op": "delete", "stable_id": 17, "label": "bounded_base_delete"},
        {"op": "rebuild_barrier", "label": "bounded_synchronous_barrier"},
        {"op": "query", "query": list(sidecar_point), "label": "bounded_every_query_oracle"},
    ]


def load_plan_or_fail(path: Path) -> Tuple[Dict[str, Any], str]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(plan, dict), "plan root is invalid")
    require(plan.get("schema") == "tide.e2e3-campaign-plan.v1" and plan.get("template_only") is False,
            "plan schema/template state is invalid")
    require(plan.get("synthetic_development_only") is True and plan.get("metric_contract") == "integer_l2_squared_v1",
            "runner accepts only the validated synthetic integer-L2 plan")
    common = {"expression": "live_sidecars + live_delta", "threshold_name": "C_ov", "C_ov": 4096}
    require(plan.get("common_overlay_budget") == common and isinstance(plan.get("k"), int) and plan["k"] == 10,
            "unexpected plan C_ov/k")
    e2 = plan.get("primary_e2")
    require(isinstance(e2, dict) and e2.get("variants") == ["certified_sidecar", "exact_delta"], "E2 variants differ")
    require(e2.get("overlay_accounting") == common and e2.get("sidecar_cap_L") == 64 and
            e2.get("base_delete_barrier") == "synchronous" and "delta_only_B" not in e2, "E2 contract differs")
    reps = e2.get("repetitions")
    require(isinstance(reps, list) and len(reps) == 5, "requires five E2 repetitions")
    starts: List[str] = []
    identifiers: List[str] = []
    for index, rep in enumerate(reps):
        require(isinstance(rep, dict) and rep.get("replication") == index + 1, "bad E2 replication")
        order = rep.get("order"); ids = rep.get("run_ids")
        require(isinstance(order, list) and len(order) == 2 and set(order) == {"certified_sidecar", "exact_delta"}, "bad E2 order")
        require(isinstance(ids, list) and len(ids) == 2 and all(isinstance(value, str) and len(value) == 32 for value in ids), "bad E2 IDs")
        starts.append(order[0]); identifiers.extend(ids)
    require(all(starts[index] != starts[index + 1] for index in range(len(starts) - 1)) and
            abs(starts.count("certified_sidecar") - starts.count("exact_delta")) <= 1 and
            len(identifiers) == len(set(identifiers)), "E2 schedule is not balanced/unique")
    e3 = plan.get("e3_scale_points")
    require(isinstance(e3, list), "E3 list missing")
    counts: List[int] = []
    for point in e3:
        require(isinstance(point, dict) and point.get("overlay_accounting") == common and
                point.get("sidecar_cap_L") == 64 and point.get("base_delete_barrier") == "synchronous" and
                point.get("oracle_coverage") == "every_query", "E3 matching contract differs")
        require(isinstance(point.get("objects"), int) and isinstance(point.get("query_count"), int), "E3 types invalid")
        counts.append(point["objects"])
    require(100000 in counts and 1000000 in counts, "required 100K/1M plan points missing")
    return plan, sha256_file(path)


def create_fresh_run_root(path: Path, plan_sha256: str) -> None:
    require(not path.exists(), f"run root exists; retry prohibited: {path}")
    require(path.parent.is_dir(), "run root parent missing")
    mkdir_new(path)
    write_new_json(path / "RUN_ROOT_O_EXCL_CREATED.json", {
        "schema": RUN_SCHEMA,
        "status": "RUN_ROOT_CREATED_O_EXCL_NO_RETRY",
        "executor_uid": os.geteuid(),
        "plan_sha256": plan_sha256,
        "nonclaim": "Fresh synthetic-development root only; no result is implied.",
    })
    mkdir_new(path / "e2")
    mkdir_new(path / "e3")


def execute_variant_or_fail(run_dir: Path, suite: str, variant: str, run_id: str, base_size: int,
                             operations: Sequence[Mapping[str, Any]], capacity: int, k: int,
                             catalog_sha256: str, trace_sha256: str, plan_sha256: str,
                             planned_query_count: Optional[int]) -> Dict[str, Any]:
    mkdir_new(run_dir)
    state = make_state(variant, base_size)
    samples: List[Dict[str, Any]] = []
    executed_queries = 0
    for index, operation in enumerate(operations):
        started_ns = time.perf_counter_ns()
        kind = operation["op"]
        query_stats: Dict[str, Any] = {"scanned_base": 0, "scanned_sidecar": 0, "scanned_delta": 0,
                                       "receipt_leaf_ids": [], "receipt_leaf_count": 0, **state.counts(capacity)}
        result_sha256 = None
        oracle_sha256 = None
        if kind == "query":
            values = operation["query"]
            query = (int(values[0]), int(values[1]), int(values[2]))
            observed, query_stats = state.query_top_k(query, k, capacity)
            oracle = full_live_oracle_top_k(base_size, state.semantic_overlay(), query, k)
            require(observed == oracle, f"full-live exactness mismatch in {suite}/{run_id}/{index}")
            exactness = "PASS_FULL_INTEGER_L2_ORACLE"
            result_sha256 = ranked_hash(observed)
            oracle_sha256 = ranked_hash(oracle)
            executed_queries += 1
        else:
            exactness = state.apply(operation)
            query_stats = {"scanned_base": 0, "scanned_sidecar": 0, "scanned_delta": 0,
                           "receipt_leaf_ids": [], "receipt_leaf_count": 0, **state.counts(capacity)}
        elapsed_ns = time.perf_counter_ns() - started_ns
        semantic_overlay = state.semantic_overlay()
        sample: Dict[str, Any] = {
            "schema": "tide.synthetic-e2e3-receipt-sidecar-raw-operation.v2",
            "development_status": "SYNTHETIC_DEVELOPMENT_ONLY",
            "suite": suite,
            "variant": variant,
            "run_id": run_id,
            "operation_index": index,
            "operation": kind,
            "operation_label": operation["label"],
            "elapsed_ns": elapsed_ns,
            "catalog_sha256": catalog_sha256,
            "trace_sha256": trace_sha256,
            "plan_sha256": plan_sha256,
            "semantic_state_sha256": semantic_state_identity(base_size, semantic_overlay),
            "exactness": exactness,
            **query_stats,
        }
        if result_sha256 is not None:
            sample["result_sha256"] = result_sha256
            sample["oracle_result_sha256"] = oracle_sha256
        samples.append(sample)
    require(executed_queries > 0, "run contains no oracle-checked query")
    raw_path = run_dir / "raw_per_operation.jsonl"
    write_new_jsonl(raw_path, samples)
    summary: Dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "COMPLETED_SYNTHETIC_DEVELOPMENT_ONLY",
        "suite": suite,
        "variant": variant,
        "run_id": run_id,
        "executor_uid": os.geteuid(),
        "execution_backend": "cpu_stdlib_python_only",
        "synthetic_state_implementation": "receipt_indexed_leaf_sidecars" if variant == "certified_sidecar" else "all_arrivals_global_delta",
        "catalog_sha256": catalog_sha256,
        "trace_sha256": trace_sha256,
        "plan_sha256": plan_sha256,
        "C_ov": capacity,
        "k": k,
        "operation_count": len(samples),
        "executed_query_count": executed_queries,
        "oracle_coverage": "every_executed_query",
        "raw_per_operation_path": "raw_per_operation.jsonl",
        "raw_per_operation_sha256": sha256_file(raw_path),
        "nonclaim": "Raw CPU synthetic timers/exactness only; no native, GPU, or paper-performance claim.",
    }
    if planned_query_count is not None:
        summary["planned_query_count"] = planned_query_count
        summary["bounded_execution_note"] = "One query per variant was executed; every executed query is oracle-checked, but the 1,000-query predeclaration is not completed."
    write_new_json(run_dir / "RUN_SUMMARY.json", summary)
    return summary


def execute_e2_or_fail(root: Path, plan: Mapping[str, Any], plan_sha256: str) -> Dict[str, Any]:
    e2 = plan["primary_e2"]
    capacity = plan["common_overlay_budget"]["C_ov"]
    base_size = 2048
    operations = make_e2_trace(base_size)
    catalog_sha256 = catalog_identity(base_size)
    trace_sha256 = trace_identity("e2-matched-receipt-sidecar-vs-delta", operations)
    summaries: List[Dict[str, Any]] = []
    for rep in e2["repetitions"]:
        for position, variant in enumerate(rep["order"]):
            summaries.append(execute_variant_or_fail(
                root / "e2" / f"rep{rep['replication']:02d}_{variant}_{rep['run_ids'][position]}", "E2", variant,
                rep["run_ids"][position], base_size, operations, capacity, plan["k"], catalog_sha256,
                trace_sha256, plan_sha256, None))
    require(len(summaries) == 10 and all(row["catalog_sha256"] == catalog_sha256 and row["trace_sha256"] == trace_sha256 and row["C_ov"] == capacity for row in summaries),
            "E2 variants do not share catalog/trace/C_ov")
    summary = {
        "schema": RUN_SCHEMA, "status": "COMPLETED_SYNTHETIC_DEVELOPMENT_ONLY", "suite": "E2",
        "executor_uid": os.geteuid(), "catalog_sha256": catalog_sha256, "trace_sha256": trace_sha256,
        "C_ov": capacity, "sidecar_cap_L": e2["sidecar_cap_L"], "base_delete_barrier": e2["base_delete_barrier"],
        "repetitions_executed": 5, "variant_runs_executed": 10,
        "planned_start_order": [row["order"][0] for row in e2["repetitions"]],
        "oracle_coverage": "every_executed_query", "raw_samples_only": True,
        "nonclaim": "Synthetic receipt-sidecar vs exact-delta correctness/timer exercise; not native-GTS performance evidence.",
    }
    write_new_json(root / "e2" / "E2_EXECUTION_SCOPE.json", summary)
    return summary


def execute_e3_or_fail(root: Path, plan: Mapping[str, Any], plan_sha256: str) -> Dict[str, Any]:
    capacity = plan["common_overlay_budget"]["C_ov"]
    selected = [row for row in plan["e3_scale_points"] if row["objects"] in {100000, 1000000}]
    require({row["objects"] for row in selected} == {100000, 1000000}, "E3 required points absent")
    records: List[Dict[str, Any]] = []
    for scale_index, point in enumerate(sorted(selected, key=lambda row: row["objects"])):
        count = point["objects"]
        scale_dir = root / "e3" / f"objects_{count}"
        mkdir_new(scale_dir)
        operations = make_e3_trace(count)
        catalog_sha256 = catalog_identity(count)
        trace_sha256 = trace_identity(f"e3-{count}-bounded-receipt-sidecar-vs-delta", operations)
        order = ["certified_sidecar", "exact_delta"] if scale_index % 2 == 0 else ["exact_delta", "certified_sidecar"]
        summaries: List[Dict[str, Any]] = []
        for variant in order:
            run_id = sha256_bytes(f"synthetic-e3-v2|{count}|{variant}".encode("utf-8"))[:32]
            summaries.append(execute_variant_or_fail(
                scale_dir / f"{variant}_{run_id}", "E3", variant, run_id, count, operations,
                capacity, plan["k"], catalog_sha256, trace_sha256, plan_sha256, point["query_count"]))
        require(all(row["executed_query_count"] == 1 and row["oracle_coverage"] == "every_executed_query" and
                    row["catalog_sha256"] == catalog_sha256 and row["trace_sha256"] == trace_sha256 and row["C_ov"] == capacity
                    for row in summaries), "E3 bounded paired oracle contract failed")
        record = {
            "schema": RUN_SCHEMA, "status": "COMPLETED_SYNTHETIC_DEVELOPMENT_ONLY", "suite": "E3",
            "objects": count, "catalog_sha256": catalog_sha256, "trace_sha256": trace_sha256, "C_ov": capacity,
            "base_delete_barrier": point["base_delete_barrier"], "planned_query_count": point["query_count"],
            "executed_query_count_per_variant": 1, "executed_variant_runs": 2,
            "oracle_coverage": "every_executed_query",
            "bounded_execution_note": "One query per variant only; this is not completion of the 1,000-query E3 plan.",
            "nonclaim": "CPU implicit-catalog synthetic smoke evidence only; not scale/performance evidence.",
        }
        write_new_json(scale_dir / "E3_SCALE_SCOPE.json", record)
        records.append(record)
    summary = {
        "schema": RUN_SCHEMA, "status": "COMPLETED_SYNTHETIC_DEVELOPMENT_ONLY", "suite": "E3",
        "executor_uid": os.geteuid(), "executed_scale_points": records, "oracle_coverage": "every_executed_query",
        "raw_samples_only": True,
        "nonclaim": "Bounded synthetic CPU checks only; not full E3 execution or paper evidence.",
    }
    write_new_json(root / "e3" / "E3_EXECUTION_SCOPE.json", summary)
    return summary


def evidence_manifest(root: Path) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "EVIDENCE_MANIFEST.json":
            continue
        files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": "tide.synthetic-e2e3-receipt-sidecar-evidence-manifest.v2", "status": "SYNTHETIC_DEVELOPMENT_ONLY",
            "payload_files": files,
            "nonclaim": "Integrity listing only; no native/GPU/performance conclusion is made."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    require(os.getuid() == EXPECTED_UID and os.geteuid() == EXPECTED_UID, f"must execute as UID {EXPECTED_UID}")
    plan, plan_sha256 = load_plan_or_fail(args.plan.resolve())
    root = args.run_root.resolve()
    created = False
    try:
        create_fresh_run_root(root, plan_sha256)
        created = True
        write_new_json(root / "RUN_STARTED.json", {
            "schema": RUN_SCHEMA, "status": "RUNNING_SYNTHETIC_DEVELOPMENT_ONLY", "executor_uid": os.geteuid(),
            "execution_backend": "cpu_stdlib_python_only", "plan_path": str(args.plan.resolve()), "plan_sha256": plan_sha256,
            "nonclaim": "Start marker only; no completion claim.",
        })
        e2 = execute_e2_or_fail(root, plan, plan_sha256)
        e3 = execute_e3_or_fail(root, plan, plan_sha256)
        write_new_json(root / "RUN_COMPLETED.json", {
            "schema": RUN_SCHEMA, "status": "COMPLETED_SYNTHETIC_DEVELOPMENT_ONLY", "executor_uid": os.geteuid(),
            "execution_backend": "cpu_stdlib_python_only", "plan_sha256": plan_sha256, "e2": e2, "e3": e3,
            "nonclaim": "Synthetic CPU exactness/raw-timer bundle only; no native-GTS, GPU, or paper-performance conclusion.",
        })
        write_new_json(root / "EVIDENCE_MANIFEST.json", evidence_manifest(root))
        return 0
    except BaseException as error:
        if created and root.is_dir() and not (root / "RUN_COMPLETED.json").exists() and not (root / "RUN_FAILED.json").exists():
            try:
                write_new_json(root / "RUN_FAILED.json", {
                    "schema": RUN_SCHEMA, "status": "FAIL_CLOSED_SYNTHETIC_DEVELOPMENT_ONLY", "executor_uid": os.geteuid(),
                    "error_type": type(error).__name__, "error": str(error),
                    "nonclaim": "Failure record only; no partial result may be used.",
                })
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL_CLOSED: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
