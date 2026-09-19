#!/usr/bin/env python3
"""Execute a bounded, CPU-only synthetic E2/E3 developmental campaign.

The code is deliberately standalone: it does not import or invoke GTS, CUDA,
network clients, an external dataset, or a host runner.  It never retries a
run root and writes artifacts only through O_EXCL creation.
"""
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
RUN_SCHEMA = "tide.synthetic-e2e3-execution.v1"
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
    require(path.parent.is_dir(), f"parent is absent for {path}")
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


def create_fresh_run_root(path: Path, plan_sha256: str) -> None:
    require(not path.exists(), f"run root already exists (retry prohibited): {path}")
    require(path.parent.is_dir(), "run root parent is absent")
    mkdir_new(path)
    # os.mkdir is the atomic root creation; this explicit O_EXCL record makes
    # the no-retry creation contract observable in the evidence bundle.
    write_new_json(path / "RUN_ROOT_O_EXCL_CREATED.json", {
        "schema": RUN_SCHEMA,
        "status": "RUN_ROOT_CREATED_O_EXCL_NO_RETRY",
        "executor_uid": os.geteuid(),
        "plan_sha256": plan_sha256,
        "nonclaim": "Fresh developmental synthetic run root only; no result is implied.",
    })
    mkdir_new(path / "e2")
    mkdir_new(path / "e3")


def load_plan_or_fail(path: Path) -> Tuple[Dict[str, Any], str]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(plan, dict), "plan root is not an object")
    require(plan.get("schema") == "tide.e2e3-campaign-plan.v1", "unexpected plan schema")
    require(plan.get("template_only") is False, "plan is not executable")
    require(plan.get("synthetic_development_only") is True, "runner refuses non-synthetic plan")
    require(plan.get("metric_contract") == "integer_l2_squared_v1", "metric contract differs")
    require(isinstance(plan.get("k"), int) and 1 <= plan["k"] <= 32, "invalid k")
    common = plan.get("common_overlay_budget")
    require(isinstance(common, dict), "common overlay contract is absent")
    require(common == {"expression": "live_sidecars + live_delta", "threshold_name": "C_ov", "C_ov": 4096},
            "unexpected synthetic common overlay contract")
    e2 = plan.get("primary_e2")
    require(isinstance(e2, dict), "E2 plan is absent")
    require(e2.get("variants") == ["certified_sidecar", "exact_delta"], "E2 variants/order differs")
    require(e2.get("overlay_accounting") == common and e2.get("sidecar_cap_L") == 64,
            "E2 does not bind the common budget/L")
    require(e2.get("base_delete_barrier") == "synchronous" and "delta_only_B" not in e2,
            "E2 barrier or budget contract differs")
    repetitions = e2.get("repetitions")
    require(isinstance(repetitions, list) and len(repetitions) == 5, "exactly five E2 repetitions required")
    starts: List[str] = []
    run_ids: List[str] = []
    for index, row in enumerate(repetitions):
        require(isinstance(row, dict) and row.get("replication") == index + 1, "bad repetition index")
        order = row.get("order")
        ids = row.get("run_ids")
        require(isinstance(order, list) and len(order) == 2 and set(order) == {"certified_sidecar", "exact_delta"},
                "bad E2 run order")
        require(isinstance(ids, list) and len(ids) == 2 and all(isinstance(value, str) and len(value) == 32 for value in ids),
                "bad E2 run IDs")
        starts.append(order[0])
        run_ids.extend(ids)
    require(all(starts[index] != starts[index + 1] for index in range(len(starts) - 1)), "E2 starts are not alternating")
    require(abs(starts.count("certified_sidecar") - starts.count("exact_delta")) <= 1, "E2 starts are not balanced")
    require(len(run_ids) == len(set(run_ids)), "E2 run IDs are not unique")
    e3 = plan.get("e3_scale_points")
    require(isinstance(e3, list), "E3 scale list is absent")
    counts = []
    for point in e3:
        require(isinstance(point, dict), "E3 scale is not an object")
        require(point.get("overlay_accounting") == common and point.get("sidecar_cap_L") == 64,
                "E3 does not bind the common budget/L")
        require(point.get("oracle_coverage") == "every_query", "E3 oracle contract differs")
        require(isinstance(point.get("objects"), int) and point["objects"] > 0, "bad E3 object count")
        require(isinstance(point.get("query_count"), int) and point["query_count"] > 0, "bad E3 planned query count")
        counts.append(point["objects"])
    require(any(count >= 100000 for count in counts) and any(count >= 1000000 for count in counts),
            "plan lacks 100K/1M E3 points")
    return plan, sha256_file(path)


def point_for_id(stable_id: int) -> Point:
    require(stable_id >= 0, "negative stable ID")
    return ((stable_id * 17 + 13) & 255, (stable_id * 31 + 29) & 255, (stable_id * 47 + 71) & 255)


def squared_distance(left: Point, right: Point) -> int:
    dx = left[0] - right[0]
    dy = left[1] - right[1]
    dz = left[2] - right[2]
    return dx * dx + dy * dy + dz * dz


def catalog_identity(object_count: int) -> str:
    require(object_count > 0, "catalog object count is invalid")
    return sha256_bytes(canonical_bytes({
        "catalog_kind": "implicit_synthetic_uint8_l2_v1",
        "object_count": object_count,
        "stable_id_range": [0, object_count - 1],
        "coordinate_formula": "((17*i+13)&255,(31*i+29)&255,(47*i+71)&255)",
    }))


def trace_identity(label: str, operations: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_bytes({"trace_kind": "synthetic_overlay_trace_v1", "label": label, "operations": list(operations)}))


def state_identity(base_size: int, overlay: Overlay) -> str:
    rows: List[List[Any]] = []
    for stable_id in sorted(overlay):
        point = overlay[stable_id]
        rows.append([stable_id, None if point is None else list(point)])
    return sha256_bytes(canonical_bytes({"base_size": base_size, "overlay": rows}))


def overlay_counts(variant: str, overlay: Overlay, capacity: int) -> Dict[str, int]:
    require(variant in {"certified_sidecar", "exact_delta"}, "unknown synthetic variant")
    live = len(overlay)
    require(live <= capacity, "shared overlay capacity exceeded")
    return {
        "live_sidecars": live if variant == "certified_sidecar" else 0,
        "live_delta": live if variant == "exact_delta" else 0,
        "overlay_total": live,
        "C_ov": capacity,
    }


def offer_sorted(best: List[Ranked], candidate: Ranked, k: int) -> None:
    if len(best) < k:
        bisect.insort(best, candidate)
    elif candidate < best[-1]:
        bisect.insort(best, candidate)
        best.pop()


def variant_top_k(base_size: int, overlay: Overlay, query: Point, k: int) -> List[Ranked]:
    """Synthetic variant path: ordered fixed-k insertion over base + overlay."""
    best: List[Ranked] = []
    for stable_id in range(base_size):
        if stable_id in overlay:
            continue
        offer_sorted(best, (squared_distance(query, point_for_id(stable_id)), stable_id), k)
    for stable_id in sorted(overlay):
        point = overlay[stable_id]
        if point is not None:
            offer_sorted(best, (squared_distance(query, point), stable_id), k)
    return best


def full_exact_oracle_top_k(base_size: int, overlay: Overlay, query: Point, k: int) -> List[Ranked]:
    """Separate full oracle path: max-heap replacement, then canonical sort."""
    heap: List[Tuple[int, int]] = []

    def offer(stable_id: int, point: Point) -> None:
        distance = squared_distance(query, point)
        transformed = (-distance, -stable_id)
        if len(heap) < k:
            heapq.heappush(heap, transformed)
        elif transformed > heap[0]:
            heapq.heapreplace(heap, transformed)

    for stable_id in range(base_size):
        if stable_id not in overlay:
            offer(stable_id, point_for_id(stable_id))
    for stable_id, point in overlay.items():
        if point is not None:
            offer(stable_id, point)
    result = sorted([(-distance, -stable_id) for distance, stable_id in heap])
    require(len(result) == k, "oracle did not produce k rows")
    return result


def ranked_hash(rows: Sequence[Ranked]) -> str:
    return sha256_bytes(canonical_bytes({"ranked": [[distance, stable_id] for distance, stable_id in rows]}))


def apply_update_or_fail(overlay: Overlay, operation: Mapping[str, Any], base_size: int) -> None:
    kind = operation["op"]
    stable_id = operation["stable_id"]
    require(isinstance(stable_id, int) and stable_id >= 0, "invalid update stable ID")
    if kind == "insert":
        point_values = operation["point"]
        require(stable_id >= base_size and stable_id not in overlay, "synthetic insert is not fresh")
        require(isinstance(point_values, list) and len(point_values) == 3, "invalid insert point")
        overlay[stable_id] = (int(point_values[0]), int(point_values[1]), int(point_values[2]))
    elif kind == "delete":
        if stable_id >= base_size:
            require(stable_id in overlay and overlay[stable_id] is not None, "synthetic overlay delete misses live insert")
            del overlay[stable_id]
        else:
            require(stable_id not in overlay, "synthetic base delete repeats")
            overlay[stable_id] = None
    else:
        raise ValueError("non-update operation passed to update path")


def make_e2_trace(base_size: int) -> List[Dict[str, Any]]:
    inserted = base_size + 17
    return [
        {"op": "query", "query": list(point_for_id(base_size * 3 + 1)), "label": "initial"},
        {"op": "insert", "stable_id": inserted, "point": list(point_for_id(inserted)), "label": "insert"},
        {"op": "query", "query": list(point_for_id(base_size * 3 + 2)), "label": "after_insert"},
        {"op": "delete", "stable_id": 17, "label": "base_delete"},
        {"op": "query", "query": list(point_for_id(base_size * 3 + 3)), "label": "after_base_delete"},
        {"op": "delete", "stable_id": inserted, "label": "overlay_delete"},
        {"op": "query", "query": list(point_for_id(base_size * 3 + 4)), "label": "after_overlay_delete"},
    ]


def make_e3_trace(base_size: int) -> List[Dict[str, Any]]:
    inserted = base_size + 23
    return [
        {"op": "insert", "stable_id": inserted, "point": list(point_for_id(inserted)), "label": "bounded_insert"},
        {"op": "delete", "stable_id": 17, "label": "bounded_base_delete"},
        {"op": "query", "query": list(point_for_id(base_size * 5 + 11)), "label": "bounded_every_query_oracle"},
    ]


def execute_variant_or_fail(run_dir: Path, suite: str, variant: str, run_id: str, base_size: int,
                             operations: Sequence[Mapping[str, Any]], capacity: int, k: int,
                             catalog_sha256: str, trace_sha256: str, plan_sha256: str,
                             planned_query_count: Optional[int]) -> Dict[str, Any]:
    mkdir_new(run_dir)
    overlay: Overlay = {}
    samples: List[Dict[str, Any]] = []
    executed_queries = 0
    for index, operation in enumerate(operations):
        started_ns = time.perf_counter_ns()
        kind = operation["op"]
        result_sha256 = None
        oracle_sha256 = None
        if kind == "query":
            values = operation["query"]
            query = (int(values[0]), int(values[1]), int(values[2]))
            observed = variant_top_k(base_size, overlay, query, k)
            oracle = full_exact_oracle_top_k(base_size, overlay, query, k)
            require(observed == oracle, f"exactness mismatch in {suite}/{run_id} op {index}")
            result_sha256 = ranked_hash(observed)
            oracle_sha256 = ranked_hash(oracle)
            exactness = "PASS_FULL_INTEGER_L2_ORACLE"
            executed_queries += 1
        else:
            apply_update_or_fail(overlay, operation, base_size)
            exactness = "PASS_STATE_TRANSITION"
        timing_ns = time.perf_counter_ns() - started_ns
        counts = overlay_counts(variant, overlay, capacity)
        sample: Dict[str, Any] = {
            "schema": "tide.synthetic-e2e3-raw-operation.v1",
            "development_status": "SYNTHETIC_DEVELOPMENT_ONLY",
            "suite": suite,
            "variant": variant,
            "run_id": run_id,
            "operation_index": index,
            "operation": kind,
            "operation_label": operation["label"],
            "elapsed_ns": timing_ns,
            "catalog_sha256": catalog_sha256,
            "trace_sha256": trace_sha256,
            "plan_sha256": plan_sha256,
            "state_sha256": state_identity(base_size, overlay),
            "exactness": exactness,
            **counts,
        }
        if result_sha256 is not None:
            sample["result_sha256"] = result_sha256
            sample["oracle_result_sha256"] = oracle_sha256
        samples.append(sample)
    require(executed_queries > 0, "synthetic run has no query")
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
        "nonclaim": "Raw local timer samples only; no performance, native-GTS, or paper claim is made.",
    }
    if planned_query_count is not None:
        summary["planned_query_count"] = planned_query_count
        summary["bounded_execution_note"] = "Executed one query per variant, not the plan predeclaration; every executed query has full-oracle coverage."
    write_new_json(run_dir / "RUN_SUMMARY.json", summary)
    return summary


def execute_e2_or_fail(root: Path, plan: Mapping[str, Any], plan_sha256: str) -> Dict[str, Any]:
    e2_dir = root / "e2"
    e2 = plan["primary_e2"]
    capacity = plan["common_overlay_budget"]["C_ov"]
    base_size = 2048
    operations = make_e2_trace(base_size)
    catalog_sha256 = catalog_identity(base_size)
    trace_sha256 = trace_identity("e2-fixed-synthetic-trace", operations)
    summaries: List[Dict[str, Any]] = []
    for repetition in e2["repetitions"]:
        rep_index = repetition["replication"]
        for order_index, variant in enumerate(repetition["order"]):
            run_id = repetition["run_ids"][order_index]
            run_dir = e2_dir / f"rep{rep_index:02d}_{variant}_{run_id}"
            summaries.append(execute_variant_or_fail(
                run_dir, "E2", variant, run_id, base_size, operations, capacity, plan["k"],
                catalog_sha256, trace_sha256, plan_sha256, None))
    require(len(summaries) == 10, "E2 did not execute both variants in five repetitions")
    require(all(row["catalog_sha256"] == catalog_sha256 and row["trace_sha256"] == trace_sha256 and row["C_ov"] == capacity for row in summaries),
            "E2 runs do not share catalog/trace/C_ov")
    summary = {
        "schema": RUN_SCHEMA,
        "status": "COMPLETED_SYNTHETIC_DEVELOPMENT_ONLY",
        "suite": "E2",
        "executor_uid": os.geteuid(),
        "catalog_sha256": catalog_sha256,
        "trace_sha256": trace_sha256,
        "C_ov": capacity,
        "sidecar_cap_L": e2["sidecar_cap_L"],
        "repetitions_executed": 5,
        "variant_runs_executed": 10,
        "planned_start_order": [row["order"][0] for row in e2["repetitions"]],
        "raw_samples_only": True,
        "oracle_coverage": "every_executed_query",
        "nonclaim": "Synthetic CPU correctness/timer exercise only; not a matched native-GTS performance experiment.",
    }
    write_new_json(e2_dir / "E2_EXECUTION_SCOPE.json", summary)
    return summary


def execute_e3_or_fail(root: Path, plan: Mapping[str, Any], plan_sha256: str) -> Dict[str, Any]:
    e3_dir = root / "e3"
    capacity = plan["common_overlay_budget"]["C_ov"]
    selected = [point for point in plan["e3_scale_points"] if point["objects"] in {100000, 1000000}]
    require({point["objects"] for point in selected} == {100000, 1000000}, "required E3 points are absent")
    scale_records: List[Dict[str, Any]] = []
    for scale_index, point in enumerate(sorted(selected, key=lambda row: row["objects"])):
        object_count = point["objects"]
        scale_dir = e3_dir / f"objects_{object_count}"
        mkdir_new(scale_dir)
        operations = make_e3_trace(object_count)
        catalog_sha256 = catalog_identity(object_count)
        trace_sha256 = trace_identity(f"e3-{object_count}-bounded-synthetic-trace", operations)
        order = ["certified_sidecar", "exact_delta"] if scale_index % 2 == 0 else ["exact_delta", "certified_sidecar"]
        summaries: List[Dict[str, Any]] = []
        for variant in order:
            run_id = sha256_bytes(f"synthetic-e3|{object_count}|{variant}".encode("utf-8"))[:32]
            summaries.append(execute_variant_or_fail(
                scale_dir / f"{variant}_{run_id}", "E3", variant, run_id, object_count,
                operations, capacity, plan["k"], catalog_sha256, trace_sha256, plan_sha256,
                point["query_count"]))
        require(all(row["executed_query_count"] == 1 and row["oracle_coverage"] == "every_executed_query" for row in summaries),
                "E3 did not oracle-cover every bounded query")
        require(all(row["catalog_sha256"] == catalog_sha256 and row["trace_sha256"] == trace_sha256 and row["C_ov"] == capacity for row in summaries),
                "E3 paired variants do not share catalog/trace/C_ov")
        record = {
            "schema": RUN_SCHEMA,
            "status": "COMPLETED_SYNTHETIC_DEVELOPMENT_ONLY",
            "suite": "E3",
            "objects": object_count,
            "catalog_sha256": catalog_sha256,
            "trace_sha256": trace_sha256,
            "C_ov": capacity,
            "planned_query_count": point["query_count"],
            "executed_query_count_per_variant": 1,
            "executed_variant_runs": 2,
            "oracle_coverage": "every_executed_query",
            "bounded_execution_note": "Only one query per variant was executed; this is not completion of the 1,000-query predeclaration.",
            "nonclaim": "Implicit synthetic catalog, CPU-only exactness smoke check; no scale or performance claim.",
        }
        write_new_json(scale_dir / "E3_SCALE_SCOPE.json", record)
        scale_records.append(record)
    summary = {
        "schema": RUN_SCHEMA,
        "status": "COMPLETED_SYNTHETIC_DEVELOPMENT_ONLY",
        "suite": "E3",
        "executor_uid": os.geteuid(),
        "executed_scale_points": scale_records,
        "raw_samples_only": True,
        "nonclaim": "Bounded synthetic CPU checks only; not full E3 execution or paper evidence.",
    }
    write_new_json(e3_dir / "E3_EXECUTION_SCOPE.json", summary)
    return summary


def evidence_manifest(root: Path) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "EVIDENCE_MANIFEST.json":
            continue
        relative = path.relative_to(root).as_posix()
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "schema": "tide.synthetic-e2e3-evidence-manifest.v1",
        "status": "SYNTHETIC_DEVELOPMENT_ONLY",
        "payload_files": files,
        "nonclaim": "Integrity inventory only; no performance or native implementation claim is made.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    require(os.geteuid() == EXPECTED_UID and os.getuid() == EXPECTED_UID,
            f"runner must execute as UID {EXPECTED_UID}")
    plan, plan_sha256 = load_plan_or_fail(args.plan.resolve())
    root = args.run_root.resolve()
    created = False
    try:
        create_fresh_run_root(root, plan_sha256)
        created = True
        write_new_json(root / "RUN_STARTED.json", {
            "schema": RUN_SCHEMA,
            "status": "RUNNING_SYNTHETIC_DEVELOPMENT_ONLY",
            "executor_uid": os.geteuid(),
            "execution_backend": "cpu_stdlib_python_only",
            "plan_path": str(args.plan.resolve()),
            "plan_sha256": plan_sha256,
            "nonclaim": "Start marker only; no completed result is implied.",
        })
        e2 = execute_e2_or_fail(root, plan, plan_sha256)
        e3 = execute_e3_or_fail(root, plan, plan_sha256)
        write_new_json(root / "RUN_COMPLETED.json", {
            "schema": RUN_SCHEMA,
            "status": "COMPLETED_SYNTHETIC_DEVELOPMENT_ONLY",
            "executor_uid": os.geteuid(),
            "execution_backend": "cpu_stdlib_python_only",
            "plan_sha256": plan_sha256,
            "e2": e2,
            "e3": e3,
            "nonclaim": "Developmental synthetic exactness/raw-timer bundle only; no native-GTS or paper-performance conclusion.",
        })
        write_new_json(root / "EVIDENCE_MANIFEST.json", evidence_manifest(root))
        return 0
    except BaseException as error:
        if created and root.is_dir() and not (root / "RUN_FAILED.json").exists() and not (root / "RUN_COMPLETED.json").exists():
            try:
                write_new_json(root / "RUN_FAILED.json", {
                    "schema": RUN_SCHEMA,
                    "status": "FAIL_CLOSED_SYNTHETIC_DEVELOPMENT_ONLY",
                    "executor_uid": os.geteuid(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "nonclaim": "Failure record only; no partial result should be used.",
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
