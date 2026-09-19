#!/usr/bin/env python3
"""CPU-only, KNN-only stable-ID HNSW replay for the frozen E1 SIFT trace.

This is deliberately a minimal semantic baseline, not a GTS comparison and not
an HNSW range-search benchmark.  It consumes the frozen quantized E1 bundle
without modifying it, applies every insert/delete immediately by stable ID,
and independently recomputes an exact active-set int64 L2 oracle for every
KNN event.  Range events are recorded as unsupported/skipped while retaining
their place in the trace; they do not modify the active set.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import socket
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np

MAGIC = b"E1GTRC01"
VERSION = 1
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
OP_BY_CODE = {1: "insert", 2: "delete", 3: "knn", 4: "range"}
SCHEMA = "hnsw-stable-id-dynamic-knn-replay-v1"


@dataclass(frozen=True)
class TraceEvent:
    op_index: int
    op: str
    argument: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_set_sha256(active: set[int]) -> str:
    # Same canonical encoding as the frozen E1 adapter; it is a stable-ID
    # state witness, not an HNSW-internal-state hash.
    return hashlib.sha256("".join(f"{sid}\n" for sid in sorted(active)).encode("ascii")).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def canonical_jsonl_row(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_bundle(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    require(bundle.is_dir(), f"bundle does not exist: {bundle}")
    manifest_path = bundle / "manifest.json"
    metadata_path = bundle / "metadata.json"
    require(manifest_path.is_file() and metadata_path.is_file(), "bundle missing manifest.json or metadata.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "e1gi-b-quantized-gts-integration-manifest-v1", "unexpected frozen manifest schema")
    require(metadata.get("schema") == "e1gi-b-quantized-gts-integration-bundle-v1", "unexpected frozen metadata schema")

    expected_hashes = manifest.get("files_sha256")
    require(isinstance(expected_hashes, dict) and expected_hashes, "manifest has no files_sha256 map")
    input_hashes: dict[str, str] = {"manifest.json": sha256_file(manifest_path)}
    for name, wanted in sorted(expected_hashes.items()):
        require(isinstance(name, str) and "/" not in name and name not in {"", ".", ".."}, f"unsafe manifest file name: {name!r}")
        path = bundle / name
        require(path.is_file(), f"bundle file missing: {name}")
        observed = sha256_file(path)
        require(observed == wanted, f"frozen input hash mismatch for {name}: expected {wanted}, got {observed}")
        input_hashes[name] = observed

    trace_path = bundle / "trace.e1gtrc"
    blob = trace_path.read_bytes()
    require(len(blob) >= HEADER.size, "trace is shorter than fixed header")
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = HEADER.unpack_from(blob, 0)
    require(magic == MAGIC and version == VERSION, "unsupported trace magic/version")
    require(len(blob) == HEADER.size + EVENT.size * event_count, "trace byte length disagrees with header")
    header = {
        "magic": magic.decode("ascii"), "version": int(version), "dim": int(dim), "base_n": int(base_n),
        "reservoir_n": int(reservoir_n), "pool_n": int(pool_n), "query_n": int(query_n), "k": int(k),
        "radius": float(radius), "event_count": int(event_count), "header_bytes": HEADER.size, "event_bytes": EVENT.size,
    }
    for key in ("dim", "base_n", "reservoir_n", "pool_n", "query_n", "k", "event_count"):
        require(int(metadata.get("header", {}).get(key, -1)) == header[key], f"metadata/header mismatch for {key}")
    require(np.isclose(np.float32(metadata.get("header", {}).get("radius")), np.float32(radius), rtol=0.0, atol=0.0), "metadata/header mismatch for radius")

    events: list[TraceEvent] = []
    for position in range(event_count):
        op_index, code, argument = EVENT.unpack_from(blob, HEADER.size + position * EVENT.size)
        require(op_index == position, f"trace event {position}: noncontiguous op_index={op_index}")
        op = OP_BY_CODE.get(code)
        require(op is not None, f"trace event {position}: unknown opcode={code}")
        events.append(TraceEvent(int(op_index), op, int(argument)))

    raw_pool = np.fromfile(bundle / "pool.i16", dtype="<i2")
    raw_queries = np.fromfile(bundle / "queries.i16", dtype="<i2")
    require(raw_pool.size == pool_n * dim, f"pool payload length {raw_pool.size} != {pool_n}*{dim}")
    require(raw_queries.size == query_n * dim, f"query payload length {raw_queries.size} != {query_n}*{dim}")
    pool = raw_pool.reshape(pool_n, dim)
    queries = raw_queries.reshape(query_n, dim)

    mapping = np.fromfile(bundle / "stable_id_to_pool_row.i32", dtype="<i4")
    require(mapping.size == pool_n, "stable_id_to_pool_row length mismatch")
    require(np.all((mapping >= 0) & (mapping < pool_n)), "stable_id_to_pool_row contains out-of-range row")
    require(np.unique(mapping).size == pool_n, "stable_id_to_pool_row is not a one-to-one mapping")
    initial_ids = np.fromfile(bundle / "initial_base_stable_ids.i32", dtype="<i4")
    require(initial_ids.size == base_n, "initial base stable-ID length mismatch")
    require(np.unique(initial_ids).size == base_n, "initial base stable IDs are not unique")
    require(np.array_equal(np.sort(initial_ids.astype(np.int64)), np.arange(base_n, dtype=np.int64)), "frozen SIFT trace violates initial stable-ID contract [0,base_n)")

    # Validate dynamic stable-ID semantics before constructing HNSW.  HNSW has
    # no support for duplicate labels/reinsertion under this protocol.
    active = set(int(x) for x in initial_ids)
    seen_insert: set[int] = set()
    trace_counts = {"insert": 0, "delete": 0, "knn": 0, "range": 0}
    for event in events:
        sid = event.argument
        if event.op == "insert":
            require(base_n <= sid < base_n + reservoir_n <= pool_n, f"op {event.op_index}: insert {sid} is outside disjoint reservoir")
            require(sid not in active and sid not in seen_insert, f"op {event.op_index}: duplicate/reinsert stable ID {sid}")
            active.add(sid); seen_insert.add(sid)
        elif event.op == "delete":
            require(sid in active, f"op {event.op_index}: delete non-live stable ID {sid}")
            active.remove(sid)
        else:
            require(0 <= sid < query_n, f"op {event.op_index}: query ID {sid} out of range")
        trace_counts[event.op] += 1
    trace_counts["final_active_count"] = len(active)

    expected_path = bundle / "quantized_oracle_expected.jsonl"
    require(expected_path.is_file(), "frozen bundle lacks quantized_oracle_expected.jsonl")
    expected: dict[int, dict[str, Any]] = {}
    with expected_path.open("r", encoding="utf-8") as fh:
        for line_number, raw in enumerate(fh, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            oi = row.get("op_index")
            require(isinstance(oi, int) and oi not in expected, f"provided oracle line {line_number}: duplicate/invalid op_index")
            expected[oi] = row
    require(set(expected) == set(range(event_count)), "provided oracle does not contain exactly one record per trace event")

    return {
        "bundle": bundle, "manifest": manifest, "metadata": metadata, "header": header, "events": events,
        "pool": pool, "queries": queries, "mapping": mapping.astype(np.int64, copy=False),
        "initial_ids": initial_ids.astype(np.int64, copy=False), "expected": expected,
        "input_hashes": input_hashes, "trace_counts": trace_counts,
        "event_stream_sha256": hashlib.sha256(blob[HEADER.size:]).hexdigest(),
    }


def exact_topk(pool: np.ndarray, mapping: np.ndarray, query: np.ndarray, active: set[int], k: int) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """Independent exact oracle: int64 squared coordinate differences + stable ID tie break."""
    ids = np.fromiter(sorted(active), dtype=np.int64)
    require(ids.size >= k, f"active set has only {ids.size} elements, smaller than k={k}")
    rows = mapping[ids]
    delta = pool[rows].astype(np.int64, copy=False) - query.astype(np.int64, copy=False)
    sqdist = np.sum(delta * delta, axis=1, dtype=np.int64)
    order = np.lexsort((ids, sqdist))
    top = order[:k]
    kth_sq = sqdist[top[-1]]
    boundary_tie = bool(np.any(sqdist[order[k:]] == kth_sq))
    tie_count = int(np.count_nonzero(sqdist == kth_sq))
    return ids[top], sqdist[top], boundary_tie, tie_count


def exact_sq_for_ids(pool: np.ndarray, mapping: np.ndarray, query: np.ndarray, stable_ids: np.ndarray) -> np.ndarray:
    rows = mapping[stable_ids]
    delta = pool[rows].astype(np.int64, copy=False) - query.astype(np.int64, copy=False)
    return np.sum(delta * delta, axis=1, dtype=np.int64)


def check_provided_update(row: dict[str, Any], event: TraceEvent, active: set[int]) -> list[str]:
    errors: list[str] = []
    if row.get("record") != "update": errors.append("provided record is not update")
    if row.get("op") != event.op: errors.append(f"provided op={row.get('op')!r}, expected={event.op!r}")
    if row.get("stable_id") != event.argument: errors.append("provided stable_id mismatch")
    if row.get("active_count") != len(active): errors.append("provided active_count mismatch")
    if row.get("verifier_active_set_sha256") != stable_set_sha256(active): errors.append("provided active-set hash mismatch")
    return errors


def check_provided_query(row: dict[str, Any], event: TraceEvent, active: set[int], oracle_ids: np.ndarray | None, oracle_sq: np.ndarray | None) -> list[str]:
    errors: list[str] = []
    if row.get("record") != "query": errors.append("provided record is not query")
    if row.get("kind") != event.op: errors.append(f"provided kind={row.get('kind')!r}, expected={event.op!r}")
    if row.get("query_id") != event.argument: errors.append("provided query_id mismatch")
    if row.get("active_count") != len(active): errors.append("provided active_count mismatch")
    if row.get("verifier_active_set_sha256") != stable_set_sha256(active): errors.append("provided active-set hash mismatch")
    if event.op == "knn" and oracle_ids is not None and oracle_sq is not None:
        results = row.get("results")
        if not isinstance(results, list) or len(results) != len(oracle_ids):
            errors.append("provided KNN results length mismatch")
        else:
            for position, (wanted_id, wanted_sq, item) in enumerate(zip(oracle_ids.tolist(), oracle_sq.tolist(), results)):
                if not isinstance(item, list) or len(item) != 2 or int(item[0]) != int(wanted_id):
                    errors.append(f"provided KNN stable-ID mismatch at rank {position}")
                    break
                # Distances are source evidence only.  The independently recomputed
                # int64 exact oracle above is the validation basis here.
                wanted_distance = math.sqrt(int(wanted_sq))
                if not math.isclose(float(item[1]), wanted_distance, rel_tol=1e-12, abs_tol=1e-9):
                    errors.append(f"provided KNN distance mismatch at rank {position}")
                    break
    return errors


def replay(bundle_data: dict[str, Any], out: Path, args: argparse.Namespace, run_card: dict[str, Any]) -> dict[str, Any]:
    header = bundle_data["header"]
    pool: np.ndarray = bundle_data["pool"]
    queries: np.ndarray = bundle_data["queries"]
    mapping: np.ndarray = bundle_data["mapping"]
    initial_ids: np.ndarray = bundle_data["initial_ids"]
    expected: dict[int, dict[str, Any]] = bundle_data["expected"]
    events: list[TraceEvent] = bundle_data["events"]
    k = int(header["k"])

    index = hnswlib.Index(space="l2", dim=int(header["dim"]))
    build_started = time.perf_counter()
    index.init_index(max_elements=int(header["pool_n"]), ef_construction=args.ef_construction, M=args.m,
                     random_seed=args.seed, allow_replace_deleted=False)
    initial_payload = pool[mapping[initial_ids]].astype(np.float32, copy=False)
    index.add_items(initial_payload, initial_ids, num_threads=1)
    index.set_ef(args.ef)
    build_seconds = time.perf_counter() - build_started

    active = set(int(x) for x in initial_ids)
    trace_path = out / "hnsw_replay.jsonl"
    query_metrics = {
        "knn_events": 0, "range_events_skipped": 0, "updates": 0, "total_topk_hits": 0,
        "total_topk_slots": 0, "exact_membership_matches": 0, "exact_order_matches": 0,
        "invalid_hnsw_responses": 0, "k_boundary_tie_queries": 0,
    }
    crosscheck_errors: list[dict[str, Any]] = []
    wall_started = time.perf_counter()

    with trace_path.open("w", encoding="utf-8") as log:
        for event in events:
            provided = expected[event.op_index]
            if event.op == "insert":
                sid = event.argument
                require(sid not in active, f"runtime op {event.op_index}: insert active ID {sid}")
                payload = pool[mapping[np.asarray([sid], dtype=np.int64)]].astype(np.float32, copy=False)
                index.add_items(payload, np.asarray([sid], dtype=np.int64), num_threads=1)
                active.add(sid)
                provided_errors = check_provided_update(provided, event, active)
                record = {
                    "schema": SCHEMA, "record": "update", "op_index": event.op_index, "op": "insert", "stable_id": sid,
                    "payload_source": "pool.i16[stable_id_to_pool_row[stable_id]]", "applied_immediately": True,
                    "active_count": len(active), "active_set_sha256": stable_set_sha256(active),
                    "provided_oracle_crosscheck_errors": provided_errors,
                }
                query_metrics["updates"] += 1
            elif event.op == "delete":
                sid = event.argument
                require(sid in active, f"runtime op {event.op_index}: delete non-live ID {sid}")
                index.mark_deleted(sid)
                active.remove(sid)
                provided_errors = check_provided_update(provided, event, active)
                record = {
                    "schema": SCHEMA, "record": "update", "op_index": event.op_index, "op": "delete", "stable_id": sid,
                    "applied_immediately": True, "delete_api": "hnswlib.Index.mark_deleted(stable_id)",
                    "active_count": len(active), "active_set_sha256": stable_set_sha256(active),
                    "provided_oracle_crosscheck_errors": provided_errors,
                }
                query_metrics["updates"] += 1
            elif event.op == "range":
                # Deliberately no HNSW range query: this runner has no range result,
                # timing, or correctness claim.  The event is retained for trace audit.
                provided_errors = check_provided_query(provided, event, active, None, None)
                record = {
                    "schema": SCHEMA, "record": "range_unsupported", "op_index": event.op_index,
                    "query_id": event.argument, "active_count": len(active), "active_set_sha256": stable_set_sha256(active),
                    "executed": False, "reason": "intentionally KNN-only baseline; hnswlib range search is not implemented/evaluated",
                    "provided_oracle_crosscheck_errors": provided_errors,
                }
                query_metrics["range_events_skipped"] += 1
            else:  # knn
                query = queries[event.argument]
                oracle_ids, oracle_sq, boundary_tie, tie_count = exact_topk(pool, mapping, query, active, k)
                provided_errors = check_provided_query(provided, event, active, oracle_ids, oracle_sq)
                labels, hnsw_sq_f32 = index.knn_query(query.astype(np.float32, copy=False).reshape(1, -1), k=k, num_threads=1)
                raw_ids = labels.reshape(-1).astype(np.int64, copy=False)
                raw_hnsw_sq = hnsw_sq_f32.reshape(-1).astype(np.float64, copy=False)
                response_errors: list[str] = []
                if raw_ids.size != k: response_errors.append(f"returned {raw_ids.size} labels, expected {k}")
                if np.unique(raw_ids).size != raw_ids.size: response_errors.append("duplicate stable ID in HNSW result")
                inactive = [int(sid) for sid in raw_ids.tolist() if int(sid) not in active]
                if inactive: response_errors.append(f"returned deleted/non-live stable IDs: {inactive[:10]}")
                if response_errors:
                    normalized_ids = raw_ids
                    returned_exact_sq = np.full(raw_ids.shape, -1, dtype=np.int64)
                else:
                    returned_exact_sq = exact_sq_for_ids(pool, mapping, query, raw_ids)
                    normalized_ids = raw_ids[np.lexsort((raw_ids, returned_exact_sq))]
                exact_order = bool(np.array_equal(normalized_ids, oracle_ids))
                exact_membership = bool(set(int(x) for x in raw_ids.tolist()) == set(int(x) for x in oracle_ids.tolist())) if not response_errors else False
                hits = len(set(int(x) for x in raw_ids.tolist()) & set(int(x) for x in oracle_ids.tolist())) if not response_errors else 0
                record = {
                    "schema": SCHEMA, "record": "knn", "op_index": event.op_index, "query_id": event.argument,
                    "active_count": len(active), "active_set_sha256": stable_set_sha256(active),
                    "oracle_method": "independent int64 squared-L2 active-set scan; deterministic (squared_distance, stable_id)",
                    "exact_oracle_stable_ids": [int(x) for x in oracle_ids.tolist()],
                    "exact_oracle_squared_distances": [int(x) for x in oracle_sq.tolist()],
                    "exact_oracle_distances": [math.sqrt(int(x)) for x in oracle_sq.tolist()],
                    "k_boundary_tie": boundary_tie, "kth_distance_tie_count": tie_count,
                    "hnsw_returned_stable_ids_raw": [int(x) for x in raw_ids.tolist()],
                    "hnsw_returned_squared_distances_f32": [float(x) for x in raw_hnsw_sq.tolist()],
                    "hnsw_returned_exact_squared_distances": [int(x) for x in returned_exact_sq.tolist()],
                    "hnsw_returned_stable_ids_normalized_by_exact_contract": [int(x) for x in normalized_ids.tolist()],
                    "response_errors": response_errors, "topk_intersection": hits, "recall_at_k": hits / float(k),
                    "exact_membership_match": exact_membership, "exact_order_match": exact_order,
                    "provided_oracle_crosscheck_errors": provided_errors,
                }
                query_metrics["knn_events"] += 1
                query_metrics["total_topk_hits"] += hits
                query_metrics["total_topk_slots"] += k
                query_metrics["exact_membership_matches"] += int(exact_membership)
                query_metrics["exact_order_matches"] += int(exact_order)
                query_metrics["invalid_hnsw_responses"] += int(bool(response_errors))
                query_metrics["k_boundary_tie_queries"] += int(boundary_tie)
            if provided_errors:
                crosscheck_errors.append({"op_index": event.op_index, "errors": provided_errors})
            log.write(canonical_jsonl_row(record) + "\n")

    wall_seconds = time.perf_counter() - wall_started
    require(len(active) == bundle_data["trace_counts"]["final_active_count"], "runtime final active count mismatch")
    require(stable_set_sha256(active) == bundle_data["metadata"]["quantized_oracle"]["verifier_active_set_sha256"], "runtime final active-set hash mismatch vs frozen metadata")
    require(not crosscheck_errors, f"independent replay disagrees with provided frozen oracle on {len(crosscheck_errors)} events")

    query_metrics["mean_recall_at_k"] = (query_metrics["total_topk_hits"] / query_metrics["total_topk_slots"] if query_metrics["total_topk_slots"] else 0.0)
    query_metrics["all_knn_exact_membership"] = query_metrics["exact_membership_matches"] == query_metrics["knn_events"]
    query_metrics["all_knn_exact_order"] = query_metrics["exact_order_matches"] == query_metrics["knn_events"]
    status = "PASS_VALIDATED_HNSW_KNN_REPLAY"
    if args.require_exact and not query_metrics["all_knn_exact_order"]:
        status = "FAIL_REQUIRED_EXACT_MATCH"

    return {
        "schema": SCHEMA,
        "status": status,
        "scope": "CPU-only dynamic HNSW KNN semantic replay over frozen E1 quantized SIFT trace; not a GTS comparison",
        "gpu_used": False,
        "bundle": str(bundle_data["bundle"]),
        "header": header,
        "trace_counts": bundle_data["trace_counts"],
        "event_stream_sha256": bundle_data["event_stream_sha256"],
        "hnsw": {
            "library": getattr(hnswlib, "__file__", "unknown"), "space": "l2", "M": args.m,
            "ef_construction": args.ef_construction, "ef_search": args.ef, "random_seed": args.seed,
            "threads": 1, "max_elements": int(header["pool_n"]), "build_wall_seconds": build_seconds,
            "delete_semantics": "hnswlib.Index.mark_deleted(stable_id); no compaction/rebuild measured",
        },
        "dynamic_contract": {
            "stable_id_labels": True,
            "payload_lookup": "pool.i16[stable_id_to_pool_row[stable_id]]",
            "inserts_immediate": True, "deletes_immediate": True,
            "knn_oracle_each_event": "independent int64 active-set L2 scan", "topk_order": "(squared_distance, stable_id)",
        },
        "knn_accuracy": query_metrics,
        "provided_quantized_oracle_crosscheck": {
            "used_only_as_consistency_check": True, "error_count": len(crosscheck_errors), "errors": crosscheck_errors[:100],
        },
        "final_active_count": len(active), "final_active_set_sha256": stable_set_sha256(active),
        "trace_output": str(trace_path), "trace_output_sha256": sha256_file(trace_path),
        "wall_seconds_excluding_index_build": wall_seconds,
        "limitations": [
            "KNN only: all 136 range events are recorded as unsupported/skipped; no HNSW range-query correctness, latency, or recall result is claimed.",
            "CPU hnswlib only; CUDA/GPU is not used. This is not a dynamic GTS performance comparison and supplies no relative-speed claim.",
            "This is one frozen small SIFT-128 quantized trace (4096 initial objects, 512 events), not a paper-scale or production deployment result.",
            "HNSW is approximate. The measured per-query exact-oracle recall is evidence for this fixed configuration/trace only, not a universal exactness guarantee.",
            "mark_deleted models logical deletion; index compaction, memory growth, rebuild policy, and update-cost amortization are intentionally not evaluated.",
        ],
        "run_card_path": str(out / "run_card.json"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef", type=int, default=4096)
    p.add_argument("--seed", type=int, default=20260727)
    p.add_argument("--wrapper-path", type=Path, default=None)
    p.add_argument("--require-exact", action="store_true", help="return nonzero if any KNN differs from exact stable-ID top-K")
    args = p.parse_args()
    require(args.m > 0 and args.ef_construction > 0 and args.ef > 0, "HNSW parameters must be positive")
    out = args.out.resolve()
    require(not out.exists(), f"refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True)
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    source = Path(__file__).resolve()
    run_card: dict[str, Any] = {
        "schema": SCHEMA + "-run-card-v1", "status": "RUNNING", "started_utc": started_utc,
        "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version,
        "cpu_only": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runner": str(source), "runner_sha256": sha256_file(source),
        "wrapper": str(args.wrapper_path.resolve()) if args.wrapper_path else None,
        "wrapper_sha256": sha256_file(args.wrapper_path.resolve()) if args.wrapper_path else None,
        "command_argv": sys.argv, "hnsw_parameters": {"M": args.m, "ef_construction": args.ef_construction, "ef_search": args.ef, "seed": args.seed, "threads": 1},
        "allowed_operations": ["initial base build", "stable-ID immediate insert", "stable-ID immediate mark_deleted", "KNN with independent exact active-set oracle"],
        "forbidden_or_unsupported": ["GPU/CUDA", "range search", "GTS execution", "GTS-vs-HNSW timing or throughput comparison", "archive modification"],
    }
    write_json(out / "run_card.json", run_card)
    try:
        bundle_data = load_bundle(args.bundle)
        write_json(out / "input_hashes.json", {
            "bundle": str(bundle_data["bundle"]), "manifest_validated": True,
            "files_sha256": bundle_data["input_hashes"], "event_stream_sha256": bundle_data["event_stream_sha256"],
        })
        summary = replay(bundle_data, out, args, run_card)
        summary["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(out / "summary.json", summary)
        run_card["status"] = summary["status"]
        run_card["finished_utc"] = summary["finished_utc"]
        run_card["summary_path"] = str(out / "summary.json")
        run_card["input_hashes_path"] = str(out / "input_hashes.json")
        run_card["trace_path"] = str(out / "hnsw_replay.jsonl")
        write_json(out / "run_card.json", run_card)
        print(json.dumps({"status": summary["status"], "out": str(out), "gpu_used": False, "knn_accuracy": summary["knn_accuracy"]}, sort_keys=True))
        return 0 if summary["status"].startswith("PASS") else 2
    except Exception as exc:
        failure = {"schema": SCHEMA + "-failure-v1", "status": "FAIL_RUNNER_EXCEPTION", "error_type": type(exc).__name__, "error": str(exc), "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        write_json(out / "failure.json", failure)
        run_card["status"] = "FAIL_RUNNER_EXCEPTION"
        run_card["finished_utc"] = failure["finished_utc"]
        run_card["failure_path"] = str(out / "failure.json")
        write_json(out / "run_card.json", run_card)
        print(json.dumps({"status": "FAIL_RUNNER_EXCEPTION", "out": str(out), "error": str(exc), "gpu_used": False}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
