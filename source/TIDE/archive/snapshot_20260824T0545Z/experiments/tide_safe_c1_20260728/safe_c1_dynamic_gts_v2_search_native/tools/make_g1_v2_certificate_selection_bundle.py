#!/usr/bin/env python3
"""Prepare, but never run, a fresh v2 Safe-C1 certificate-selection bundle.

Each screened reservoir object is inserted, queried at its own immutable vector,
and deleted before the next object.  With leaf capacity one this isolates the
search-native certificate decision from sidecar capacity.  The CPU screen only
selects inputs satisfying the bounded static-prefix/dynamic-boundary witness; it does not assert a direct placement.  Direct-vs-delta
is determined later only by the new v2 guarded GPU0 selection runner.  The only
selection hard gate is a same-sidecar-leaf group of three direct receipts for a
future, separately validated G1B capacity test; certificate-delta is recorded
as a diagnostic statistic, never required here.
"""
from __future__ import annotations

from array import array
import argparse
import json
import shutil
from pathlib import Path

from safe_c1_v2_bundle_contract import (
    OP_DELETE,
    OP_INSERT,
    OP_KNN,
    BOUNDED_TIE_DOMAIN,
    CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
    DYNAMIC_K_BOUNDARY_POLICY,
    STATIC_BASE_POLICY,
    BundleContractError,
    TraceEvent,
    TraceHeader,
    exact_squared_l2,
    load_current_runner_identity_stable_id_layout,
    load_i16,
    parse_trace,
    sha256,
    validate_tie_free_witness,
    write_i16,
    write_trace,
)

CERTIFICATE = {
    "schema": "safe-c1-search-native-sibling-boundary-v2",
    "native_search_boundary": "non-last: min_i + epsilon < radius < next_sibling_min - epsilon; final: radius > min_last + epsilon",
    "required_parent_shape": "all 10 physical siblings non-empty, common pivot, finite strictly increasing min_dis",
    "max_distance": "remains in frozen hash only; never used as a direct-routing upper bound",
    "on_failure": "fail closed to exact global delta",
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def query_from_pool(
    pool: array, stable_id: int, dimension: int, stable_id_to_pool_row: list[int]
) -> array:
    """Materialize a query from the explicitly verified runner-compatible layout."""
    output = array("h")
    begin = stable_id_to_pool_row[stable_id] * dimension
    output.extend(pool[begin:begin + dimension])
    return output


def has_unique_exact_self_top1(
    pool: array, source: TraceHeader, stable_id: int, stable_id_to_pool_row: list[int]
) -> bool:
    """Reject a candidate whose self-query has an ambiguous exact top-1.

    Dynamic k/k+1 distinctness makes top-k membership unique but does not by
    itself rule out a tie at rank one. The selection receipt is intended to
    show the inserted object itself at rank one, so screen out that ambiguity
    before any guarded run.
    """
    query = query_from_pool(pool, stable_id, source.dimension, stable_id_to_pool_row)
    for base_id in range(source.base_n):
        if exact_squared_l2(pool, stable_id_to_pool_row[base_id], query, 0, source.dimension) == 0:
            return False
    return True


def is_tie_free_single_candidate(
    pool: array, source: TraceHeader, stable_id: int, stable_id_to_pool_row: list[int]
) -> bool:
    query = query_from_pool(pool, stable_id, source.dimension, stable_id_to_pool_row)
    header = TraceHeader(source.dimension, source.base_n, source.reservoir_n,
                         source.pool_n, 1, source.k, source.radius, 3)
    events = [TraceEvent(0, OP_INSERT, stable_id), TraceEvent(1, OP_KNN, 0),
              TraceEvent(2, OP_DELETE, stable_id)]
    try:
        validate_tie_free_witness(pool, query, header, events)
        return True
    except BundleContractError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=512, help="bounded-tie candidates to screen (default: 512; use 2048 for a broader fresh selection bundle)")
    parser.add_argument("--start-stable-id", type=int, default=None)
    args = parser.parse_args()
    canonical_root = Path(__file__).resolve().parents[1]
    source = args.source_bundle.resolve()
    out = args.out.resolve()
    try:
        out.relative_to(canonical_root / "bundles")
    except ValueError:
        raise SystemExit("v2 bundle output must be under this v2 root's bundles directory")
    if args.count < 3:
        raise SystemExit("--count must be at least 3: selection needs a possible same-leaf direct group of three")
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing v2 selection bundle: {out}")

    source_header, _ = parse_trace(source / "trace.e1gtrc")
    pool = load_i16(source / "pool.i16", source_header.pool_n * source_header.dimension)
    try:
        stable_id_to_pool_row, _initial_base_stable_ids = \
            load_current_runner_identity_stable_id_layout(source, source_header)
    except BundleContractError as exc:
        raise SystemExit(f"certificate-selection source stable-ID layout rejected: {exc}") from exc
    start = source_header.base_n if args.start_stable_id is None else args.start_stable_id
    if start < source_header.base_n or start >= source_header.pool_n:
        raise SystemExit("selection start ID must be in the reservoir")

    selected: list[int] = []
    rejected_tie: list[int] = []
    rejected_nonunique_self_top1: list[int] = []
    for stable_id in range(start, source_header.pool_n):
        if not has_unique_exact_self_top1(
            pool, source_header, stable_id, stable_id_to_pool_row
        ):
            rejected_nonunique_self_top1.append(stable_id)
        elif is_tie_free_single_candidate(
            pool, source_header, stable_id, stable_id_to_pool_row
        ):
            selected.append(stable_id)
            if len(selected) == args.count:
                break
        else:
            rejected_tie.append(stable_id)
    if len(selected) != args.count:
        raise SystemExit(
            f"insufficient bounded-tie reservoir candidates: need={args.count} found={len(selected)}"
        )

    queries = array("h")
    events: list[TraceEvent] = []
    candidates: list[dict[str, int]] = []
    for query_id, stable_id in enumerate(selected):
        queries.extend(query_from_pool(
            pool, stable_id, source_header.dimension, stable_id_to_pool_row
        ))
        insert_op = len(events)
        events.append(TraceEvent(insert_op, OP_INSERT, stable_id))
        query_op = len(events)
        events.append(TraceEvent(query_op, OP_KNN, query_id))
        delete_op = len(events)
        events.append(TraceEvent(delete_op, OP_DELETE, stable_id))
        candidates.append({
            "stable_id": stable_id,
            "query_id": query_id,
            "insert_op_index": insert_op,
            "query_op_index": query_op,
            "delete_op_index": delete_op,
        })
    header = TraceHeader(source_header.dimension, source_header.base_n,
                         source_header.reservoir_n, source_header.pool_n,
                         len(selected), source_header.k, source_header.radius,
                         len(events))
    tie_preflight = validate_tie_free_witness(pool, queries, header, events)

    out.mkdir(parents=True)
    write_i16(out / "pool.i16", pool)
    write_i16(out / "queries.i16", queries)
    write_trace(out / "trace.e1gtrc", header, events)
    # `load_current_runner_identity_stable_id_layout` above has already
    # checked both mappings and their runner-compatible identity semantics.
    for name in ("initial_base_stable_ids.i32", "stable_id_to_pool_row.i32"):
        shutil.copyfile(source / name, out / name)
    bundle_spec = {
        "schema": "safe-c1-g1-v2-certificate-selection-bundle-spec",
        "requested_count": args.count,
        "start_stable_id": start,
        "selection_method": "CPU bounded-tie input screen only; no certificate placement result",
        "unique_self_top1_screen": "reject exact self-query top-1 ties before any guarded run",
        "stable_id_layout": CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
        "leaf_capacity": 1,
        "selection_hard_gate": "at least three direct receipts sharing one sidecar_leaf_id",
        "certificate_delta_rejections_policy": "record_only_not_selection_gate",
        "g1b_followup": {
            "leaf_capacity": 2,
            "same_leaf_group_size": 3,
            "purpose": "future independent capacity-fallback validation; no result is claimed by selection",
        },
    }
    selection_contract = {
        "schema": "safe-c1-g1-v2-certificate-selection-contract",
        "engine_schema": "safe-c1-g1-topk-v2-search-native",
        "certificate_schema": CERTIFICATE["schema"],
        "tie_domain": BOUNDED_TIE_DOMAIN,
        "static_base_oracle_policy": STATIC_BASE_POLICY,
        "dynamic_k_boundary_policy": DYNAMIC_K_BOUNDARY_POLICY,
        "v1_evidence_used": False,
        "stable_id_layout": CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
        "leaf_capacity": 1,
        "min_direct_candidates": 3,
        "min_same_sidecar_leaf_direct_candidates": 3,
        "certificate_delta_rejections_policy": "record_only_not_selection_gate",
        "g1b_followup": {
            "leaf_capacity": 2,
            "same_leaf_group_size": 3,
            "purpose": "future independent capacity-fallback validation; no result is claimed by this selection artifact",
        },
        "scope": (
            "new-v2 certificate selection only; each candidate is independently inserted/query/deleted; "
            "require at least three direct receipts sharing one sidecar leaf for a future G1B capacity-two test; "
            "certificate-delta rejections are recorded only, not a selection gate; no performance, capacity result, "
            "all-input tie, or prior-v1 selection claim; ties after the bounded static top-(k+1) prefix are permitted"
        ),
        "candidates": candidates,
    }
    write_json(out / "bundle_spec.json", bundle_spec)
    write_json(out / "selection_candidates.json", selection_contract)
    write_json(out / "tie_free_preflight.json", tie_preflight)
    files = {path.name: sha256(path) for path in sorted(out.iterdir()) if path.is_file()}
    manifest = {
        "schema": "safe-c1-g1-v2-tie-free-bundle-manifest",
        "status": "PREPARED_CPU_ONLY_READY_FOR_V2_CERTIFICATE_SELECTION_RUN",
        "bundle_kind": "certificate_selection",
        "gpu_used": False,
        "cuda_binary_executed": False,
        "engine_schema_required": "safe-c1-g1-topk-v2-search-native",
        "stable_id_layout": CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
        "scope": (
            "fresh v2 CPU-prepared certificate-selection input; no v1 execution/selection evidence, "
            "no GPU execution, no direct-placement result, and no performance claim"
        ),
        "source_payload": {
            "path": str(source),
            "role": "immutable input bytes only; not v1 execution or selection evidence",
            "sha256": {name: sha256(source / name) for name in (
                "pool.i16", "trace.e1gtrc", "initial_base_stable_ids.i32", "stable_id_to_pool_row.i32",
            )},
        },
        "header": {
            "magic": "E1GTRC01", "version": 1, "dimension": header.dimension,
            "base_n": header.base_n, "reservoir_n": header.reservoir_n,
            "pool_n": header.pool_n, "query_n": header.query_n, "k": header.k,
            "event_count": header.event_count, "radius": header.radius,
        },
        "certificate_contract": CERTIFICATE,
        "tie_free_witness_domain": tie_preflight,
        "candidate_selection": {
            "status": "PENDING_V2_CERTIFICATE_SELECTION",
            "required_schema": "safe-c1-g1-v2-candidate-selection",
            "selection_contract": "selection_candidates.json",
            "selection_contract_sha256": files["selection_candidates.json"],
            "v1_evidence_used": False,
            "required_binding": "new v2 binary + new v2 static audit + new v2 run root only",
        },
        "screening": {
            "candidate_ids_considered_start": start,
            "selected_count": len(selected),
            "rejected_tie_count_before_completion": len(rejected_tie),
            "rejected_nonunique_self_top1_count_before_completion": len(rejected_nonunique_self_top1),
        },
        "files_sha256": files,
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "selected": len(selected), "gpu_used": False, "out": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BundleContractError as exc:
        raise SystemExit(str(exc))
