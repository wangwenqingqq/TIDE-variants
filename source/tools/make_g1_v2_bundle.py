#!/usr/bin/env python3
"""Create a new CPU-prepared Safe-C1 G1 v2 bounded-tie input bundle.

This generator copies immutable payload bytes only.  It does not reuse a v1
execution result and deliberately marks the output PENDING until a separate,
new-v2 certificate-selection artifact exists.  It never calls CUDA/GPU tools.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from safe_c1_v2_bundle_contract import (
    OP_DELETE,
    OP_INSERT,
    OP_KNN,
    CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
    BundleContractError,
    TraceEvent,
    TraceHeader,
    load_current_runner_identity_stable_id_layout,
    load_i16,
    parse_trace,
    sha256,
    validate_tie_free_witness,
    write_i16,
    write_trace,
)

OP_NAMES = {"insert": OP_INSERT, "delete": OP_DELETE, "knn": OP_KNN}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_spec(path: Path) -> tuple[list[int], list[TraceEvent], int | None]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != "safe-c1-g1-v2-bundle-spec":
        raise BundleContractError("bad_v2_bundle_spec_schema")
    query_ids = raw.get("query_stable_ids")
    event_rows = raw.get("events")
    if not isinstance(query_ids, list) or not query_ids or not all(isinstance(x, int) for x in query_ids):
        raise BundleContractError("bad_query_stable_ids")
    if not isinstance(event_rows, list) or not event_rows:
        raise BundleContractError("bad_bundle_events")
    events: list[TraceEvent] = []
    for index, row in enumerate(event_rows):
        if not isinstance(row, dict) or set(row) != {"op", "argument"}:
            raise BundleContractError(f"bad_event_row:{index}")
        op = OP_NAMES.get(row["op"])
        argument = row["argument"]
        if op is None or not isinstance(argument, int):
            raise BundleContractError(f"bad_event_opcode_or_argument:{index}")
        events.append(TraceEvent(index, op, argument))
    leaf_capacity = raw.get("leaf_capacity")
    if leaf_capacity is not None and (not isinstance(leaf_capacity, int) or leaf_capacity <= 0):
        raise BundleContractError("bad_leaf_capacity")
    return query_ids, events, leaf_capacity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    canonical_root = Path(__file__).resolve().parents[1]
    source = args.source_bundle.resolve()
    out = args.out.resolve()
    try:
        out.relative_to(canonical_root / "bundles")
    except ValueError:
        raise SystemExit("v2 bundle output must be under this v2 root's bundles directory")
    spec_path = args.spec.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing v2 bundle: {out}")
    if not source.is_dir() or not spec_path.is_file():
        raise SystemExit("source bundle or spec does not exist")

    source_header, _ = parse_trace(source / "trace.e1gtrc")
    pool = load_i16(source / "pool.i16", source_header.pool_n * source_header.dimension)
    try:
        stable_id_to_pool_row, _initial_base_stable_ids = \
            load_current_runner_identity_stable_id_layout(source, source_header)
    except BundleContractError as exc:
        raise SystemExit(f"v2 source stable-ID layout rejected: {exc}") from exc
    query_ids, events, leaf_capacity = parse_spec(spec_path)
    if len(set(query_ids)) != len(query_ids):
        raise SystemExit("query_stable_ids must be unique in a v2 witness spec")
    if any(stable_id < 0 or stable_id >= source_header.pool_n for stable_id in query_ids):
        raise SystemExit("query_stable_id outside immutable pool")

    # Queries are immutable pool rows selected in the explicit new-v2 spec.
    from array import array

    queries = array("h")
    for stable_id in query_ids:
        begin = stable_id_to_pool_row[stable_id] * source_header.dimension
        queries.extend(pool[begin:begin + source_header.dimension])
    header = TraceHeader(
        dimension=source_header.dimension,
        base_n=source_header.base_n,
        reservoir_n=source_header.reservoir_n,
        pool_n=source_header.pool_n,
        query_n=len(query_ids),
        k=source_header.k,
        radius=source_header.radius,
        event_count=len(events),
    )
    tie_preflight = validate_tie_free_witness(pool, queries, header, events)

    out.mkdir(parents=True)
    write_i16(out / "pool.i16", pool)
    write_i16(out / "queries.i16", queries)
    write_trace(out / "trace.e1gtrc", header, events)
    # Source layout was validated above; preserve the exact audit payload in
    # the generated bundle because the current CUDA runner validates it again
    # before any CUDA/tree operation.
    for name in ("initial_base_stable_ids.i32", "stable_id_to_pool_row.i32"):
        shutil.copyfile(source / name, out / name)
    write_json(out / "bundle_spec.json", json.loads(spec_path.read_text(encoding="utf-8")))
    write_json(out / "tie_free_preflight.json", tie_preflight)

    files = {path.name: sha256(path) for path in sorted(out.iterdir()) if path.is_file()}
    manifest = {
        "schema": "safe-c1-g1-v2-tie-free-bundle-manifest",
        "status": "PREPARED_CPU_ONLY_PENDING_V2_CERTIFICATE_SELECTION",
        "gpu_used": False,
        "cuda_binary_executed": False,
        "engine_schema_required": "safe-c1-g1-topk-v2-search-native",
        "bundle_kind": "generic_witness",
        "scope": (
            "new v2 CPU-prepared bounded static-prefix/dynamic-boundary witness input only; no v1 execution evidence, "
            "no GPU execution, no direct-placement claim, and no all-input tie claim"
        ),
        "source_payload": {
            "path": str(source),
            "role": "immutable input bytes only; not v1 execution evidence",
            "sha256": {
                name: sha256(source / name)
                for name in (
                    "pool.i16", "trace.e1gtrc", "initial_base_stable_ids.i32", "stable_id_to_pool_row.i32",
                )
            },
        },
        "stable_id_layout": CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
        "header": {
            "magic": "E1GTRC01",
            "version": 1,
            "dimension": header.dimension,
            "base_n": header.base_n,
            "reservoir_n": header.reservoir_n,
            "pool_n": header.pool_n,
            "query_n": header.query_n,
            "k": header.k,
            "event_count": header.event_count,
            "radius": header.radius,
        },
        "certificate_contract": {
            "schema": "safe-c1-search-native-sibling-boundary-v2",
            "native_search_boundary": "non-last: min_i + epsilon < radius < next_sibling_min - epsilon; final: radius > min_last + epsilon",
            "required_parent_shape": "all 10 physical siblings non-empty, common pivot, finite strictly increasing min_dis",
            "max_distance": "remains in frozen hash only; never used as a direct-routing upper bound",
            "on_failure": "fail closed to exact global delta",
        },
        "tie_free_witness_domain": tie_preflight,
        "candidate_selection": {
            "status": "PENDING_V2_CERTIFICATE_SELECTION",
            "required_schema": "safe-c1-g1-v2-candidate-selection",
            "v1_evidence_used": False,
            "required_binding": "must reference this new v2 bundle, v2 static audit, and v2 binary SHA; v1 run evidence is rejected",
            "leaf_capacity": leaf_capacity,
        },
        "files_sha256": files,
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "gpu_used": False, "out": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BundleContractError as exc:
        raise SystemExit(str(exc))
