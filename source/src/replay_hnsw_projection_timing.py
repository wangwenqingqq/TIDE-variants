#!/usr/bin/env python3
"""Strict CPU HNSW adapter for an explicitly KNN-only frozen trace projection.

This program is intentionally independent from the historical HNSW replay.  It
accepts only a manifest whose projection explicitly contains immediate stable-ID
inserts and KNNs; delete and range operations must already have been physically
removed from that projection.  It never skips either operation silently.

Timing scope is deliberately narrow: for a KNN event, ``engine_ns`` encloses
only ``hnswlib.Index.knn_query``.  The exact int64 oracle, response
canonicalization, hashing, JSONL emission, and all input conversion happen
outside that timed interval.  Base construction and each immediate insert are
reported separately.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Frozen binary format.  The projection keeps the standard E1 record encoding,
# but has a fresh manifest and only opcodes 1 (insert) and 3 (KNN).
MAGIC = b"E1GTRC01"
VERSION = 1
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
OP_BY_CODE = {1: "insert", 2: "delete", 3: "knn", 4: "range"}

MANIFEST_SCHEMA = "e1-frozen-base-knn-projection-manifest-v1"
METADATA_SCHEMA = "e1-frozen-base-knn-projection-bundle-v1"
RUN_SCHEMA = "hnsw-stable-id-knn-projection-timing-v1"

# These are fixed experiment parameters, not CLI tuning knobs.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 4096
HNSW_SEED = 20260727
HNSW_THREADS = 1
HNSW_VERSION = "0.8.0"

REQUIRED_PAYLOADS = {
    "metadata.json",
    "trace.e1gtrc",
    "pool.i16",
    "queries.i16",
    "stable_id_to_pool_row.i32",
    "initial_base_stable_ids.i32",
}


@dataclass(frozen=True)
class TraceEvent:
    op_index: int
    op: str
    argument: int


@dataclass(frozen=True)
class Bundle:
    path: Path
    manifest: dict[str, Any]
    metadata: dict[str, Any]
    header: dict[str, Any]
    events: list[TraceEvent]
    pool_i16: np.ndarray
    queries_i16: np.ndarray
    mapping: np.ndarray
    initial_ids: np.ndarray
    trace_counts: dict[str, int]
    final_active_hash: str
    event_stream_sha256: str
    input_hashes: dict[str, str]
    source_trace_sha256: str
    source_event_count: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_set_sha256(active: Iterable[int]) -> str:
    """Canonical state witness: sorted decimal stable IDs, one per line."""
    return hashlib.sha256("".join(f"{sid}\n" for sid in sorted(active)).encode("ascii")).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n"


def canonical_jsonl(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(canonical_json(obj), encoding="utf-8")


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def require_simple_file_name(name: Any, label: str) -> str:
    require(isinstance(name, str) and name not in {"", ".", ".."}, f"{label} must be a nonempty file name")
    require("/" not in name and "\\" not in name and Path(name).name == name, f"unsafe {label}: {name!r}")
    return name


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value),
            f"{label} must be a lowercase SHA-256 hex digest")
    return value


def require_exact_string_list(value: Any, wanted: set[str], label: str) -> list[str]:
    require(isinstance(value, list) and all(isinstance(item, str) for item in value), f"{label} must be a string list")
    actual = set(value)
    require(len(actual) == len(value), f"{label} has duplicate entries")
    require(actual == wanted, f"{label} must be exactly {sorted(wanted)}, got {sorted(actual)}")
    return sorted(actual)


def parse_trace(trace_path: Path) -> tuple[dict[str, Any], list[TraceEvent], str]:
    blob = trace_path.read_bytes()
    require(len(blob) >= HEADER.size, "trace is shorter than the fixed header")
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = HEADER.unpack_from(blob, 0)
    require(magic == MAGIC, f"unsupported trace magic {magic!r}")
    require(version == VERSION, f"unsupported trace version {version}")
    require(len(blob) == HEADER.size + EVENT.size * event_count,
            "trace byte length disagrees with header event_count")
    header = {
        "magic": magic.decode("ascii"),
        "version": int(version),
        "dim": int(dim),
        "base_n": int(base_n),
        "reservoir_n": int(reservoir_n),
        "pool_n": int(pool_n),
        "query_n": int(query_n),
        "k": int(k),
        "radius": float(radius),
        "event_count": int(event_count),
        "header_bytes": HEADER.size,
        "event_bytes": EVENT.size,
    }
    require(header["dim"] > 0 and header["base_n"] > 0 and header["reservoir_n"] >= 0,
            "trace has invalid nonpositive dimension/base/reservoir")
    require(header["pool_n"] >= header["base_n"] + header["reservoir_n"],
            "pool_n cannot cover base plus disjoint reservoir")
    require(header["query_n"] > 0 and header["k"] > 0 and header["base_n"] >= header["k"],
            "trace has invalid query/k/base cardinalities")

    events: list[TraceEvent] = []
    for position in range(event_count):
        op_index, opcode, argument = EVENT.unpack_from(blob, HEADER.size + position * EVENT.size)
        require(op_index == position, f"trace event {position} has noncontiguous op_index={op_index}")
        op = OP_BY_CODE.get(opcode)
        require(op is not None, f"trace event {position} has unknown opcode {opcode}")
        # Do not let a projection runner merely skip unsupported work.  A source
        # trace with delete/range is invalid input for this adapter.
        require(op in {"insert", "knn"},
                f"projection trace event {position} is forbidden op={op}; projection must physically exclude delete/range")
        events.append(TraceEvent(int(op_index), op, int(argument)))
    return header, events, hashlib.sha256(blob[HEADER.size:]).hexdigest()


def _validate_projection_metadata(
    manifest: dict[str, Any], metadata: dict[str, Any], header: dict[str, Any], events: list[TraceEvent],
    trace_path: Path, event_stream_sha256: str, bundle_path: Path,
) -> tuple[str, int, str]:
    """Validate the explicit source-to-projection provenance emitted by the derivation tool.

    The metadata intentionally keeps the source header (including its original
    event_count), whereas trace.e1gtrc has the projected event_count.  The two
    must agree on every immutable data dimension and be linked by sealed hashes.
    """
    projection = metadata.get("projection")
    require(isinstance(projection, dict), "metadata.projection must be an object")
    require_exact_string_list(projection.get("allowed_ops"), {"insert", "knn"}, "metadata.projection.allowed_ops")
    require_exact_string_list(projection.get("excluded_ops"), {"delete", "range"}, "metadata.projection.excluded_ops")
    require(isinstance(projection.get("filter_policy"), str) and projection["filter_policy"],
            "metadata.projection.filter_policy must state the explicit projection rule")
    require(projection.get("source_quantized_oracle_role") == "source provenance only; never a projection oracle",
            "metadata.projection must explicitly forbid using the source full-trace oracle as a projection oracle")

    source_trace_sha256 = require_sha256(metadata.get("source_trace_sha256"), "metadata.source_trace_sha256")
    require(manifest.get("source_trace_sha256") == source_trace_sha256,
            "manifest.source_trace_sha256 disagrees with metadata.source_trace_sha256")

    # Current bundle schema carries both headers explicitly: ``header`` is the
    # projected binary trace and ``source_header`` seals its full-trace origin.
    projected_header = metadata.get("header")
    source_header = metadata.get("source_header")
    require(isinstance(projected_header, dict), "metadata.header must be the projected trace header object")
    require(isinstance(source_header, dict), "metadata.source_header must be the source trace header object")
    header_pairs = {
        "magic": "magic", "version": "version", "dimension": "dim", "base_n": "base_n",
        "reservoir_n": "reservoir_n", "pool_n": "pool_n", "query_n": "query_n", "k": "k",
        "event_count": "event_count",
    }
    for metadata_key, trace_key in header_pairs.items():
        require(projected_header.get(metadata_key) == header[trace_key],
                f"metadata.header.{metadata_key} disagrees with projected trace {trace_key}")
    require(np.float32(projected_header.get("radius")) == np.float32(header["radius"]),
            "metadata.header.radius disagrees with projected trace")

    source_event_count: Any = source_header.get("event_count")
    require(isinstance(source_event_count, int) and source_event_count >= header["event_count"],
            "metadata.source_header.event_count must be the source event count >= projected event_count")
    source_pairs = {
        "magic": "magic", "version": "version", "dimension": "dim", "base_n": "base_n",
        "reservoir_n": "reservoir_n", "pool_n": "pool_n", "query_n": "query_n", "k": "k",
    }
    for source_key, trace_key in source_pairs.items():
        require(source_header.get(source_key) == header[trace_key],
                f"metadata.source_header.{source_key} disagrees with projected trace {trace_key}")
    require(np.float32(source_header.get("radius")) == np.float32(header["radius"]),
            "metadata.source_header.radius disagrees with projected trace")

    observed_trace_sha = sha256_file(trace_path)
    projection_trace_sha = require_sha256(projection.get("projection_trace_sha256"),
                                           "metadata.projection.projection_trace_sha256")
    require(projection_trace_sha == observed_trace_sha,
            "metadata.projection.projection_trace_sha256 disagrees with trace.e1gtrc")
    require(manifest.get("projection_trace_sha256") == observed_trace_sha,
            "manifest.projection_trace_sha256 disagrees with trace.e1gtrc")
    require(require_sha256(projection.get("projection_event_stream_sha256"),
                           "metadata.projection.projection_event_stream_sha256") == event_stream_sha256,
            "metadata.projection.projection_event_stream_sha256 disagrees with trace events")
    require(projection.get("source_indices_count") == len(events),
            "metadata.projection.source_indices_count disagrees with projected trace event_count")
    for key in ("source_indices_sha256", "stable_id_mapping_sha256", "initial_base_stable_ids_sha256"):
        require_sha256(projection.get(key), f"metadata.projection.{key}")
    require(projection["stable_id_mapping_sha256"] == sha256_file(bundle_path / "stable_id_to_pool_row.i32"),
            "metadata.projection stable-ID mapping hash mismatch")
    require(projection["initial_base_stable_ids_sha256"] == sha256_file(bundle_path / "initial_base_stable_ids.i32"),
            "metadata.projection initial base ID hash mismatch")
    for key in ("first_source_index", "last_source_index"):
        require(isinstance(projection.get(key), int) and 0 <= projection[key] < source_event_count,
                f"metadata.projection.{key} must be a valid source event index")
    require(projection["first_source_index"] <= projection["last_source_index"],
            "metadata.projection source index interval is reversed")
    return source_trace_sha256, int(source_event_count), observed_trace_sha


def load_bundle(bundle_path: Path) -> Bundle:
    bundle_path = bundle_path.resolve()
    require(bundle_path.is_dir(), f"bundle directory does not exist: {bundle_path}")
    manifest_path = bundle_path / "manifest.json"
    metadata_path = bundle_path / "metadata.json"
    manifest = read_json_object(manifest_path, "manifest.json")
    metadata = read_json_object(metadata_path, "metadata.json")
    require(manifest.get("schema") == MANIFEST_SCHEMA,
            f"unexpected manifest schema {manifest.get('schema')!r}; expected {MANIFEST_SCHEMA!r}")
    require(metadata.get("schema") == METADATA_SCHEMA,
            f"unexpected metadata schema {metadata.get('schema')!r}; expected {METADATA_SCHEMA!r}")

    # The manifest is the frozen-input integrity boundary.  Hash each declared
    # file and require every data payload needed by this adapter to be declared.
    declared = manifest.get("files_sha256")
    require(isinstance(declared, dict) and declared, "manifest.files_sha256 must be a nonempty object")
    declared_names: set[str] = set()
    input_hashes: dict[str, str] = {"manifest.json": sha256_file(manifest_path)}
    for name, expected_hash in sorted(declared.items()):
        safe_name = require_simple_file_name(name, "manifest.files_sha256 key")
        require(safe_name not in declared_names, f"duplicate manifest file name: {safe_name}")
        declared_names.add(safe_name)
        expected_hash = require_sha256(expected_hash, f"manifest hash for {safe_name}")
        path = bundle_path / safe_name
        require(path.is_file(), f"manifest-declared file is missing: {safe_name}")
        observed = sha256_file(path)
        require(observed == expected_hash,
                f"frozen input hash mismatch for {safe_name}: expected {expected_hash}, got {observed}")
        input_hashes[safe_name] = observed
    missing = REQUIRED_PAYLOADS - declared_names
    require(not missing, f"manifest.files_sha256 omits required payloads: {sorted(missing)}")

    # Standard payload names are part of the projection contract.  Unlike the
    # historical full-trace runner, there is deliberately no projection oracle
    # file to consume: any copied quantized_oracle_expected.jsonl is provenance
    # only and is never read by this adapter.
    header, events, event_stream_sha256 = parse_trace(bundle_path / "trace.e1gtrc")
    source_trace_sha256, source_event_count, observed_projection_trace_sha = _validate_projection_metadata(
        manifest, metadata, header, events, bundle_path / "trace.e1gtrc", event_stream_sha256, bundle_path
    )

    raw_pool = np.fromfile(bundle_path / "pool.i16", dtype="<i2")
    raw_queries = np.fromfile(bundle_path / "queries.i16", dtype="<i2")
    require(raw_pool.size == header["pool_n"] * header["dim"],
            f"pool.i16 length {raw_pool.size} != pool_n*dim")
    require(raw_queries.size == header["query_n"] * header["dim"],
            f"queries.i16 length {raw_queries.size} != query_n*dim")
    pool_i16 = raw_pool.reshape(header["pool_n"], header["dim"])
    queries_i16 = raw_queries.reshape(header["query_n"], header["dim"])

    mapping = np.fromfile(bundle_path / "stable_id_to_pool_row.i32", dtype="<i4")
    require(mapping.size == header["pool_n"], "stable_id_to_pool_row.i32 length mismatch")
    require(np.all((mapping >= 0) & (mapping < header["pool_n"])),
            "stable_id_to_pool_row.i32 contains an out-of-range row")
    require(np.unique(mapping).size == header["pool_n"],
            "stable_id_to_pool_row.i32 must be a bijection over pool rows")
    mapping_i64 = mapping.astype(np.int64, copy=False)

    initial_ids = np.fromfile(bundle_path / "initial_base_stable_ids.i32", dtype="<i4")
    require(initial_ids.size == header["base_n"], "initial_base_stable_ids.i32 length mismatch")
    require(np.unique(initial_ids).size == header["base_n"], "initial base stable IDs are not unique")
    require(np.array_equal(np.sort(initial_ids.astype(np.int64)), np.arange(header["base_n"], dtype=np.int64)),
            "initial base stable-ID contract must be exactly [0, base_n)")
    initial_i64 = initial_ids.astype(np.int64, copy=False)

    active = set(int(value) for value in initial_i64)
    seen_inserts: set[int] = set()
    counts = {"insert": 0, "delete": 0, "knn": 0, "range": 0}
    for event in events:
        if event.op == "insert":
            sid = event.argument
            require(header["base_n"] <= sid < header["base_n"] + header["reservoir_n"],
                    f"op {event.op_index}: insert stable ID {sid} is outside disjoint reservoir")
            require(sid not in active and sid not in seen_inserts,
                    f"op {event.op_index}: duplicate/reinsert stable ID {sid}")
            active.add(sid)
            seen_inserts.add(sid)
        elif event.op == "knn":
            require(0 <= event.argument < header["query_n"],
                    f"op {event.op_index}: query ID {event.argument} out of range")
        else:  # parse_trace makes this unreachable; retain the explicit guard.
            raise ValueError(f"op {event.op_index}: forbidden op {event.op}")
        counts[event.op] += 1
    counts["final_active_count"] = len(active)

    meta_counts = metadata.get("trace_counts")
    require(isinstance(meta_counts, dict), "metadata.trace_counts must be an object")
    source_counts = meta_counts.get("source")
    projection_counts = meta_counts.get("projection")
    require(isinstance(source_counts, dict) and isinstance(projection_counts, dict),
            "metadata.trace_counts must contain source and projection objects")
    for key in ("insert", "delete", "knn", "range"):
        require(isinstance(source_counts.get(key), int) and source_counts[key] >= 0,
                f"metadata.trace_counts.source.{key} must be a nonnegative integer")
    require(sum(source_counts[key] for key in ("insert", "delete", "knn", "range")) == source_event_count,
            "metadata.trace_counts.source does not sum to source event_count")
    # Requiring a real deleted source operation makes the exclusion visible: the
    # timing adapter cannot present a no-delete workload as if deletion had been
    # supported.  The derivation tool additionally proves those are base IDs.
    require(source_counts["delete"] > 0,
            "projection provenance must show at least one excluded source delete")
    require(source_counts["range"] > 0,
            "projection provenance must show at least one excluded source range")
    require(set(projection_counts).issubset({"insert", "knn"}),
            "metadata.trace_counts.projection may contain only insert and knn")
    for key in ("insert", "knn"):
        require(projection_counts.get(key) == counts[key],
                f"metadata.trace_counts.projection.{key} disagrees with projected trace")

    # This is a state witness generated for the projected active set.  Do not
    # read any copied original full-trace per-query oracle: it is deliberately
    # source provenance, not an oracle for this projected active history.
    projection_meta = metadata.get("projection")
    final_active_hash = require_sha256(projection_meta.get("final_active_set_sha256"),
                                       "metadata.projection.final_active_set_sha256")
    require(projection_meta.get("final_active_count") == len(active),
            "metadata.projection.final_active_count disagrees with independently replayed trace")
    observed_final_hash = stable_set_sha256(active)
    require(final_active_hash == observed_final_hash,
            "projection final active-set hash disagrees with independently replayed trace")

    return Bundle(
        path=bundle_path,
        manifest=manifest,
        metadata=metadata,
        header=header,
        events=events,
        pool_i16=pool_i16,
        queries_i16=queries_i16,
        mapping=mapping_i64,
        initial_ids=initial_i64,
        trace_counts={key: int(value) for key, value in counts.items()},
        final_active_hash=final_active_hash,
        event_stream_sha256=event_stream_sha256,
        input_hashes=input_hashes,
        source_trace_sha256=source_trace_sha256,
        source_event_count=source_event_count,
    )


def exact_topk(bundle: Bundle, query_id: int, active: set[int]) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """Independent int64 oracle sorted by (squared distance, stable ID)."""
    ids = np.fromiter(sorted(active), dtype=np.int64)
    k = bundle.header["k"]
    require(ids.size >= k, f"active set has only {ids.size} IDs, smaller than k={k}")
    rows = bundle.mapping[ids]
    query = bundle.queries_i16[query_id].astype(np.int64, copy=False)
    delta = bundle.pool_i16[rows].astype(np.int64, copy=False) - query
    sqdist = np.sum(delta * delta, axis=1, dtype=np.int64)
    order = np.lexsort((ids, sqdist))
    top = order[:k]
    kth_sq = sqdist[top[-1]]
    return ids[top], sqdist[top], bool(np.any(sqdist[order[k:]] == kth_sq)), int(np.count_nonzero(sqdist == kth_sq))


def exact_sq_for_ids(bundle: Bundle, query_id: int, ids: np.ndarray) -> np.ndarray:
    rows = bundle.mapping[ids]
    delta = bundle.pool_i16[rows].astype(np.int64, copy=False) - bundle.queries_i16[query_id].astype(np.int64, copy=False)
    return np.sum(delta * delta, axis=1, dtype=np.int64)


def fixed_hnsw_distribution_version() -> str:
    """Preflight the exact package version without constructing an index."""
    try:
        version = importlib.metadata.version("hnswlib")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("hnswlib distribution is not installed; refusing to substitute another engine") from exc
    require(version == HNSW_VERSION,
            f"hnswlib must be exactly {HNSW_VERSION}, found {version!r}; refusing an unpinned baseline")
    return version


def fixed_hnsw_module() -> tuple[Any, str]:
    version = fixed_hnsw_distribution_version()
    try:
        import hnswlib  # Imported only after version pinning.
    except Exception as exc:  # pragma: no cover - environment diagnostic
        raise ValueError(f"unable to import pinned hnswlib {HNSW_VERSION}: {exc}") from exc
    return hnswlib, version


def _percentiles_ns(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "sum_ns": 0, "mean_ns": None, "p50_ns": None, "p95_ns": None, "max_ns": None}
    arr = np.asarray(values, dtype=np.int64)
    return {
        "count": int(arr.size),
        "sum_ns": int(arr.sum(dtype=np.int64)),
        "mean_ns": float(arr.mean()),
        "p50_ns": float(np.percentile(arr, 50)),
        "p95_ns": float(np.percentile(arr, 95)),
        "max_ns": int(arr.max()),
    }


def compute_projection_oracle_digest(bundle: Bundle) -> dict[str, Any]:
    """Dry-run-only independent oracle pass; it does not instantiate HNSW."""
    active = set(int(value) for value in bundle.initial_ids)
    digest = hashlib.sha256()
    knn_count = 0
    boundary_ties = 0
    for event in bundle.events:
        if event.op == "insert":
            active.add(event.argument)
            digest.update(f"U|{event.op_index}|{event.argument}|{stable_set_sha256(active)}\n".encode("ascii"))
        elif event.op == "knn":
            ids, sq, boundary_tie, tie_count = exact_topk(bundle, event.argument, active)
            digest.update(
                (f"Q|{event.op_index}|{event.argument}|{stable_set_sha256(active)}|"
                 f"{','.join(str(int(x)) for x in ids)}|{','.join(str(int(x)) for x in sq)}|"
                 f"{int(boundary_tie)}|{tie_count}\n").encode("ascii")
            )
            knn_count += 1
            boundary_ties += int(boundary_tie)
        else:  # Defensive explicit no-skip guard.
            raise ValueError(f"dry run encountered forbidden op {event.op}")
    final_hash = stable_set_sha256(active)
    require(final_hash == bundle.final_active_hash, "dry-run final active state does not match projection witness")
    return {
        "independent_oracle": "int64 squared-L2 active-set scan, ordered by (squared_distance, stable_id)",
        "knn_events_oracled": knn_count,
        "k_boundary_tie_queries": boundary_ties,
        "oracle_stream_sha256": digest.hexdigest(),
        "final_active_count": len(active),
        "final_active_set_sha256": final_hash,
    }


def replay_timed(bundle: Bundle, out: Path, hnswlib: Any, hnsw_version: str) -> dict[str, Any]:
    """Replay the projected trace and record only index-call durations as engine_ns."""
    header = bundle.header
    # These conversions and views deliberately precede all timing.  Timing is
    # engine only, not quantization conversion, hashing, JSON, or exact oracle.
    pool_f32 = bundle.pool_i16.astype(np.float32, copy=False)
    queries_f32 = bundle.queries_i16.astype(np.float32, copy=False)
    base_labels = bundle.initial_ids.astype(np.int64, copy=False)
    base_payload = pool_f32[bundle.mapping[base_labels]]

    index = hnswlib.Index(space="l2", dim=header["dim"])
    init_start = time.perf_counter_ns()
    index.init_index(
        max_elements=header["pool_n"],
        ef_construction=HNSW_EF_CONSTRUCTION,
        M=HNSW_M,
        random_seed=HNSW_SEED,
        allow_replace_deleted=False,
    )
    init_index_ns = time.perf_counter_ns() - init_start
    base_start = time.perf_counter_ns()
    index.add_items(base_payload, base_labels, num_threads=HNSW_THREADS)
    base_add_items_ns = time.perf_counter_ns() - base_start
    # Configuration is intentionally outside build/query timing; it is a fixed
    # contract setting and is applied before the event trace begins.
    index.set_ef(HNSW_EF_SEARCH)

    active = set(int(value) for value in bundle.initial_ids)
    trace_path = out / "hnsw_projection_replay.jsonl"
    insert_ns: list[int] = []
    knn_ns: list[int] = []
    metrics: dict[str, int] = {
        "insert_events": 0,
        "knn_events": 0,
        "response_cardinality_failures": 0,
        "response_duplicate_failures": 0,
        "response_inactive_or_out_of_range_failures": 0,
        "exact_membership_matches": 0,
        "canonical_exact_order_matches": 0,
        "raw_exact_order_matches": 0,
        "k_boundary_tie_queries": 0,
    }

    with trace_path.open("x", encoding="utf-8") as log:
        build_hash = stable_set_sha256(active)
        log.write(canonical_jsonl({
            "schema": RUN_SCHEMA,
            "record": "build",
            "op_index": None,
            "active_count": len(active),
            "active_set_sha256": build_hash,
            "engine_ns": int(init_index_ns + base_add_items_ns),
            "init_index_ns": int(init_index_ns),
            "base_add_items_ns": int(base_add_items_ns),
            "oracle_result_flags": {
                "applicable": False,
                "projection_trace_validated": True,
                "provided_full_trace_oracle_used": False,
            },
            "fixed_hnsw": {
                "library_version": hnsw_version,
                "M": HNSW_M,
                "ef_construction": HNSW_EF_CONSTRUCTION,
                "ef_search": HNSW_EF_SEARCH,
                "random_seed": HNSW_SEED,
                "threads": HNSW_THREADS,
                "max_elements": header["pool_n"],
            },
        }) + "\n")

        for event in bundle.events:
            if event.op == "insert":
                sid = event.argument
                require(sid not in active, f"runtime insert {event.op_index} repeats active stable ID {sid}")
                # Prepare data outside timing; the timed call is HNSW update only.
                payload = pool_f32[bundle.mapping[np.asarray([sid], dtype=np.int64)]].reshape(1, -1)
                labels = np.asarray([sid], dtype=np.int64)
                update_start = time.perf_counter_ns()
                index.add_items(payload, labels, num_threads=HNSW_THREADS)
                engine_ns = time.perf_counter_ns() - update_start
                active.add(sid)
                active_hash = stable_set_sha256(active)
                insert_ns.append(int(engine_ns))
                metrics["insert_events"] += 1
                log.write(canonical_jsonl({
                    "schema": RUN_SCHEMA,
                    "record": "insert",
                    "op_index": event.op_index,
                    "stable_id": sid,
                    "applied_immediately": True,
                    "active_count": len(active),
                    "active_set_sha256": active_hash,
                    "engine_ns": int(engine_ns),
                    "oracle_result_flags": {
                        "applicable": False,
                        "projection_trace_validated": True,
                        "provided_full_trace_oracle_used": False,
                    },
                }) + "\n")
                continue

            if event.op != "knn":
                # parse_trace and load_bundle already reject these, but do not
                # leave a future source edit a path to silent skipping.
                raise ValueError(f"runtime encountered forbidden operation {event.op!r}")

            # Exact oracle is deliberately outside the time window.
            oracle_ids, oracle_sq, boundary_tie, tie_count = exact_topk(bundle, event.argument, active)
            query_f32 = queries_f32[event.argument].reshape(1, -1)
            query_start = time.perf_counter_ns()
            hnsw_result = index.knn_query(query_f32, k=header["k"], num_threads=HNSW_THREADS)
            engine_ns = time.perf_counter_ns() - query_start
            raw_labels, raw_sq_f32 = hnsw_result
            raw_ids = np.asarray(raw_labels).reshape(-1).astype(np.int64, copy=False)
            raw_distances = np.asarray(raw_sq_f32).reshape(-1).astype(np.float64, copy=False)

            response_cardinality_ok = raw_ids.size == header["k"]
            response_unique_ok = response_cardinality_ok and np.unique(raw_ids).size == raw_ids.size
            response_live_and_in_range_ok = response_unique_ok and bool(np.all((raw_ids >= 0) & (raw_ids < header["pool_n"])))
            if response_live_and_in_range_ok:
                response_live_and_in_range_ok = all(int(sid) in active for sid in raw_ids.tolist())

            if response_cardinality_ok and response_unique_ok and response_live_and_in_range_ok:
                returned_exact_sq = exact_sq_for_ids(bundle, event.argument, raw_ids)
                canonical_ids = raw_ids[np.lexsort((raw_ids, returned_exact_sq))]
                exact_membership_match = set(int(x) for x in raw_ids.tolist()) == set(int(x) for x in oracle_ids.tolist())
                canonical_exact_order_match = bool(np.array_equal(canonical_ids, oracle_ids))
                raw_exact_order_match = bool(np.array_equal(raw_ids, oracle_ids))
                returned_sq_json: list[int | None] = [int(x) for x in returned_exact_sq.tolist()]
                canonical_ids_json: list[int] = [int(x) for x in canonical_ids.tolist()]
            else:
                returned_exact_sq = None
                canonical_ids = None
                exact_membership_match = False
                canonical_exact_order_match = False
                raw_exact_order_match = False
                returned_sq_json = []
                canonical_ids_json = []

            active_hash = stable_set_sha256(active)
            knn_ns.append(int(engine_ns))
            metrics["knn_events"] += 1
            metrics["response_cardinality_failures"] += int(not response_cardinality_ok)
            metrics["response_duplicate_failures"] += int(response_cardinality_ok and not response_unique_ok)
            metrics["response_inactive_or_out_of_range_failures"] += int(response_unique_ok and not response_live_and_in_range_ok)
            metrics["exact_membership_matches"] += int(exact_membership_match)
            metrics["canonical_exact_order_matches"] += int(canonical_exact_order_match)
            metrics["raw_exact_order_matches"] += int(raw_exact_order_match)
            metrics["k_boundary_tie_queries"] += int(boundary_tie)
            all_oracle_checks_pass = bool(response_cardinality_ok and response_unique_ok and response_live_and_in_range_ok and canonical_exact_order_match)

            log.write(canonical_jsonl({
                "schema": RUN_SCHEMA,
                "record": "knn",
                "op_index": event.op_index,
                "query_id": event.argument,
                "active_count": len(active),
                "active_set_sha256": active_hash,
                "engine_ns": int(engine_ns),
                "engine_timing_scope": "only hnswlib.Index.knn_query(query_f32, k, num_threads=1)",
                "oracle_contract": "independent int64 squared-L2 active-set scan; order=(squared_distance, stable_id)",
                "oracle_topk_stable_ids": [int(x) for x in oracle_ids.tolist()],
                "oracle_topk_squared_distances": [int(x) for x in oracle_sq.tolist()],
                "returned_stable_ids_raw": [int(x) for x in raw_ids.tolist()],
                "returned_hnsw_squared_distances_f32": [float(x) for x in raw_distances.tolist()],
                "returned_exact_squared_distances": returned_sq_json,
                "returned_stable_ids_canonical_int64_order": canonical_ids_json,
                "k_boundary_tie": bool(boundary_tie),
                "kth_distance_tie_count": int(tie_count),
                "oracle_result_flags": {
                    "applicable": True,
                    "computed_independently_outside_engine_timed_window": True,
                    "provided_full_trace_oracle_used": False,
                    "response_cardinality_ok": bool(response_cardinality_ok),
                    "response_unique_ok": bool(response_unique_ok),
                    "response_live_and_in_range_ok": bool(response_live_and_in_range_ok),
                    "exact_membership_match": bool(exact_membership_match),
                    "canonical_int64_order_match": bool(canonical_exact_order_match),
                    "raw_engine_order_match": bool(raw_exact_order_match),
                    "all_required_checks_pass": all_oracle_checks_pass,
                },
            }) + "\n")

    final_active_hash = stable_set_sha256(active)
    require(final_active_hash == bundle.final_active_hash,
            "runtime final active state disagrees with projection final state witness")
    require(metrics["insert_events"] == bundle.trace_counts["insert"], "runtime insert count mismatch")
    require(metrics["knn_events"] == bundle.trace_counts["knn"], "runtime KNN count mismatch")

    all_exact = metrics["canonical_exact_order_matches"] == metrics["knn_events"]
    status = "PASS_EXACT_HNSW_KNN_PROJECTION" if all_exact else "FAIL_HNSW_NOT_EXACT_ON_PROJECTED_TRACE"
    return {
        "schema": RUN_SCHEMA,
        "status": status,
        "scope": "CPU-only hnswlib timing adapter over a frozen KNN-only insert projection; no GPU, range, delete, rebuild, or GTS execution",
        "bundle": str(bundle.path),
        "header": bundle.header,
        "trace_counts": bundle.trace_counts,
        "event_stream_sha256": bundle.event_stream_sha256,
        "source_projection_provenance": {
            "source_trace_sha256_declared_by_projection_manifest": bundle.source_trace_sha256,
            "source_event_count_declared_by_projection_manifest": bundle.source_event_count,
            "allowed_ops": ["insert", "knn"],
            "explicitly_excluded_ops": ["delete", "range"],
        },
        "gpu_used": False,
        "hnsw": {
            "library_version": hnsw_version,
            "library_path": getattr(hnswlib, "__file__", "unknown"),
            "space": "l2",
            "M": HNSW_M,
            "ef_construction": HNSW_EF_CONSTRUCTION,
            "ef_search": HNSW_EF_SEARCH,
            "random_seed": HNSW_SEED,
            "threads": HNSW_THREADS,
            "max_elements": header["pool_n"],
        },
        "engine_timing_contract": {
            "build": "init_index and initial add_items measured as separate engine calls",
            "insert": "engine_ns encloses only hnswlib.Index.add_items for that stable ID",
            "knn": "engine_ns encloses only hnswlib.Index.knn_query",
            "excluded_from_engine_ns": ["int16-to-float32 conversion", "exact int64 oracle", "response canonicalization", "active-set hashing", "JSONL output", "input integrity checks"],
        },
        "timing_ns": {
            "init_index_ns": int(init_index_ns),
            "base_add_items_ns": int(base_add_items_ns),
            "build_total_ns": int(init_index_ns + base_add_items_ns),
            "immediate_insert_add_items": _percentiles_ns(insert_ns),
            "knn_query": _percentiles_ns(knn_ns),
        },
        "knn_oracle_validation": {
            "contract": "independent int64 squared-L2 active-set scan; deterministic (squared_distance, stable_id)",
            "provided_full_trace_oracle_used": False,
            **metrics,
            "all_canonical_order_exact": all_exact,
        },
        "final_active_count": len(active),
        "final_active_set_sha256": final_active_hash,
        "trace_output": str(trace_path),
        "trace_output_sha256": sha256_file(trace_path),
        "limitations": [
            "The projection intentionally contains only insert and KNN events. It is not evidence for delete, range, rebuild, compaction, concurrency, or generic dynamic-index correctness.",
            "This is CPU hnswlib only. A CPU/GPU timing ratio is not a fair cross-hardware performance claim without a separately controlled hardware protocol.",
            "The exact oracle and all validation execute outside engine_ns. They establish this trace's correctness gate, not engine-only end-to-end latency.",
            "A PASS is limited to this pinned hnswlib 0.8.0 configuration and this frozen projected bundle.",
        ],
    }


def run_card_base(args: argparse.Namespace, source: Path) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-run-card-v1",
        "status": "RUNNING",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runner": str(source),
        "runner_sha256": sha256_file(source),
        "command_argv": sys.argv,
        "dry_run": bool(args.dry_run),
        "fixed_hnsw_parameters": {
            "required_version": HNSW_VERSION,
            "M": HNSW_M,
            "ef_construction": HNSW_EF_CONSTRUCTION,
            "ef_search": HNSW_EF_SEARCH,
            "random_seed": HNSW_SEED,
            "threads": HNSW_THREADS,
            "max_elements": "trace.header.pool_n",
        },
        "allowed_trace_operations": ["insert", "knn"],
        "rejected_trace_operations": ["delete", "range"],
        "forbidden": ["GPU/CUDA execution", "range-query execution", "delete skipping", "base-delete skipping", "GTS execution", "archive modification"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="frozen KNN-only projection bundle")
    parser.add_argument("--out", required=True, type=Path, help="new output directory; must not exist")
    parser.add_argument("--dry-run", action="store_true", help="validate projection and run only independent oracle; do not instantiate/query HNSW")
    args = parser.parse_args()

    out = args.out.resolve()
    require(not out.exists(), f"refusing to overwrite existing --out path: {out}")
    out.mkdir(parents=True, mode=0o700)
    source = Path(__file__).resolve()
    run_card = run_card_base(args, source)
    write_json(out / "run_card.json", run_card)

    try:
        bundle = load_bundle(args.bundle)
        write_json(out / "input_hashes.json", {
            "bundle": str(bundle.path),
            "manifest_validated": True,
            "files_sha256": bundle.input_hashes,
            "event_stream_sha256": bundle.event_stream_sha256,
            "source_trace_sha256_declared_by_projection_manifest": bundle.source_trace_sha256,
        })
        if args.dry_run:
            # Distribution-only preflight: no hnswlib import, index construction,
            # update, query, GPU call, or engine timing occurs in dry-run mode.
            pinned_hnsw_version = fixed_hnsw_distribution_version()
            validation = compute_projection_oracle_digest(bundle)
            validation.update({
                "schema": RUN_SCHEMA + "-dry-run-v1",
                "status": "PASS_DRY_RUN_PROJECTION_VALIDATED",
                "bundle": str(bundle.path),
                "trace_counts": bundle.trace_counts,
                "event_stream_sha256": bundle.event_stream_sha256,
                "pinned_hnswlib_distribution_version": pinned_hnsw_version,
                "gpu_used": False,
                "hnsw_engine_executed": False,
                "note": "dry-run validates manifest/trace/stable-ID state, pins the hnswlib distribution, and computes independent int64 KNN oracle; it does not import, instantiate, or time HNSW.",
            })
            write_json(out / "dry_run_validation.json", validation)
            run_card["status"] = validation["status"]
            run_card["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            run_card["validation_path"] = str(out / "dry_run_validation.json")
            write_json(out / "run_card.json", run_card)
            print(json.dumps({"status": validation["status"], "out": str(out), "gpu_used": False, "hnsw_engine_executed": False}, sort_keys=True))
            return 0

        hnswlib, hnsw_version = fixed_hnsw_module()
        summary = replay_timed(bundle, out, hnswlib, hnsw_version)
        summary["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(out / "summary.json", summary)
        run_card["status"] = summary["status"]
        run_card["finished_utc"] = summary["finished_utc"]
        run_card["summary_path"] = str(out / "summary.json")
        run_card["input_hashes_path"] = str(out / "input_hashes.json")
        run_card["trace_path"] = str(out / "hnsw_projection_replay.jsonl")
        write_json(out / "run_card.json", run_card)
        print(json.dumps({
            "status": summary["status"], "out": str(out), "gpu_used": False,
            "knn_events": summary["knn_oracle_validation"]["knn_events"],
            "all_canonical_order_exact": summary["knn_oracle_validation"]["all_canonical_order_exact"],
        }, sort_keys=True))
        return 0 if summary["status"].startswith("PASS") else 2
    except Exception as exc:
        failure = {
            "schema": RUN_SCHEMA + "-failure-v1",
            "status": "FAIL_RUNNER_EXCEPTION",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "gpu_used": False,
        }
        write_json(out / "failure.json", failure)
        run_card["status"] = failure["status"]
        run_card["finished_utc"] = failure["finished_utc"]
        run_card["failure_path"] = str(out / "failure.json")
        write_json(out / "run_card.json", run_card)
        print(json.dumps({"status": failure["status"], "out": str(out), "error": str(exc), "gpu_used": False}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
