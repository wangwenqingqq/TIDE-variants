#!/usr/bin/env python3
"""CPU-only contract helpers for the isolated Safe-C1 G2 rebuild witness v4.

This module deliberately has no CUDA imports or subprocess calls.  The witness
uses a fixed, auditable sequence: direct -> capacity delta -> direct-slot reuse
-> base delete -> immediate explicit rebuild -> post-rebuild query.  It does
not establish a general dynamic/rebuild or performance result.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
import sys
from typing import Any, Iterable

MAGIC = b"E1GTRC02"
VERSION = 2
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
OP_INSERT = 1
OP_DELETE = 2
OP_KNN = 3
OP_RANGE = 4
OP_REBUILD = 5

SCHEMA = "safe-c1-g2-rebuild-witness-contract-v4"
ENGINE_SCHEMA = "safe-c1-g2-rebuild-witness-v4-search-native"
CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE = "identity_full_immutable_pool_seeded_stable_ids_v4"
BOUNDED_TIE_ABORT = "ABORT_UNPROVED_BOUNDED_TIE_WITNESS_DOMAIN"
STATIC_POLICY = (
    "for every epoch base set, first min(k+1,epoch_base_n) exact squared-L2 and "
    "modeled GTS float-L2 keys are pairwise distinct; first-k ID order agrees"
)
DYNAMIC_POLICY = "k and k+1 exact quantized squared-L2 keys distinct for every trace kNN state"


class BundleContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class TraceHeader:
    dimension: int
    base_n: int
    reservoir_n: int
    pool_n: int
    query_n: int
    k: int
    radius: float
    event_count: int


@dataclass(frozen=True)
class TraceEvent:
    op_index: int
    op: int
    argument: int


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fnv1a_u64_bytes(data: bytes, seed: int = 1469598103934665603) -> int:
    value = seed
    for item in data:
        value ^= item
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def fnv1a_i32(values: Iterable[int]) -> int:
    return fnv1a_u64_bytes(b"".join(struct.pack("<i", int(item)) for item in values))


def load_i16(path: Path, expected_values: int) -> array:
    raw = path.read_bytes()
    if len(raw) != expected_values * 2:
        raise BundleContractError(
            f"wrong_i16_byte_count:{path.name}:expected={expected_values * 2}:actual={len(raw)}"
        )
    values = array("h")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def write_i16(path: Path, values: array) -> None:
    copy = array("h", values)
    if sys.byteorder != "little":
        copy.byteswap()
    path.write_bytes(copy.tobytes())


def load_i32_le(path: Path, expected_values: int | None = None) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise BundleContractError(f"wrong_i32_byte_count:{path.name}:actual={len(raw)}")
    values = [item[0] for item in struct.iter_unpack("<i", raw)]
    if expected_values is not None and len(values) != expected_values:
        raise BundleContractError(
            f"wrong_i32_value_count:{path.name}:expected={expected_values}:actual={len(values)}"
        )
    return values


def write_i32_le(path: Path, values: Iterable[int]) -> None:
    path.write_bytes(b"".join(struct.pack("<i", int(value)) for value in values))


def parse_trace(path: Path) -> tuple[TraceHeader, list[TraceEvent]]:
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise BundleContractError("truncated_trace_header")
    magic, version, dimension, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = HEADER.unpack_from(raw, 0)
    if magic != MAGIC or version != VERSION:
        raise BundleContractError("unsupported_g2_trace_magic_or_version")
    if dimension <= 0 or base_n <= 0 or pool_n < base_n or reservoir_n != pool_n - base_n:
        raise BundleContractError("invalid_trace_population_header")
    if query_n <= 0 or k <= 0 or k > base_n or radius < 0:
        raise BundleContractError("invalid_trace_query_header")
    expected_bytes = HEADER.size + event_count * EVENT.size
    if len(raw) != expected_bytes:
        raise BundleContractError(
            f"wrong_trace_byte_count:expected={expected_bytes}:actual={len(raw)}"
        )
    events: list[TraceEvent] = []
    for index in range(event_count):
        op_index, op, argument = EVENT.unpack_from(raw, HEADER.size + index * EVENT.size)
        if op_index != index or op not in {OP_INSERT, OP_DELETE, OP_KNN, OP_RANGE, OP_REBUILD}:
            raise BundleContractError(f"invalid_trace_event:{index}")
        events.append(TraceEvent(op_index, op, argument))
    return TraceHeader(dimension, base_n, reservoir_n, pool_n, query_n, k, radius, event_count), events


def write_trace(path: Path, header: TraceHeader, events: list[TraceEvent]) -> None:
    if header.event_count != len(events):
        raise BundleContractError("trace_event_count_mismatch")
    raw = HEADER.pack(MAGIC, VERSION, header.dimension, header.base_n, header.reservoir_n,
                      header.pool_n, header.query_n, header.k, float(header.radius), header.event_count)
    for index, event in enumerate(events):
        if event.op_index != index:
            raise BundleContractError("noncontiguous_output_event_index")
        raw += EVENT.pack(event.op_index, event.op, event.argument)
    path.write_bytes(raw)


def load_identity_layout(bundle: Path, header: TraceHeader) -> tuple[list[int], list[int]]:
    mapping = load_i32_le(bundle / "stable_id_to_pool_row.i32", header.pool_n)
    initial = load_i32_le(bundle / "initial_base_stable_ids.i32", header.base_n)
    if mapping != list(range(header.pool_n)):
        raise BundleContractError("stable_id_to_pool_row_not_identity_for_v4")
    if initial != list(range(header.base_n)):
        raise BundleContractError("initial_base_not_identity_prefix_for_v4")
    return mapping, initial


def read_contract(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("=") != 1:
            raise BundleContractError("malformed_g2_contract_line")
        key, value = (item.strip() for item in line.split("=", 1))
        if not key or not value or key in values:
            raise BundleContractError("empty_or_duplicate_g2_contract_field")
        values[key] = value
    required = {
        "schema", "leaf_capacity", "sidecar_leaf_id", "stable_id_a", "stable_id_b", "stable_id_c",
        "query_id_before", "query_id_middle", "query_id_after", "base_delete_id",
        "v3_g1b_result_sha256", "v2_selection_sha256",
    }
    if set(values) != required:
        raise BundleContractError(f"unexpected_g2_contract_fields:{sorted(set(values) ^ required)}")
    if values["schema"] != SCHEMA:
        raise BundleContractError("wrong_g2_contract_schema")
    for key in required - {"schema", "v3_g1b_result_sha256", "v2_selection_sha256"}:
        try:
            parsed = int(values[key])
        except ValueError as exc:
            raise BundleContractError(f"noninteger_g2_contract_field:{key}") from exc
        if str(parsed) != values[key]:
            raise BundleContractError(f"noncanonical_integer_g2_contract_field:{key}")
    for key in ("v3_g1b_result_sha256", "v2_selection_sha256"):
        value = values[key]
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise BundleContractError(f"bad_sha256_g2_contract_field:{key}")
    return values


def cint(contract: dict[str, str], key: str) -> int:
    return int(contract[key])


def validate_strict_g2_trace(header: TraceHeader, events: list[TraceEvent], contract: dict[str, str]) -> None:
    if cint(contract, "leaf_capacity") != 1:
        raise BundleContractError("g2_requires_leaf_capacity_one")
    a, b, c = (cint(contract, key) for key in ("stable_id_a", "stable_id_b", "stable_id_c"))
    qa, qc, qpost = (cint(contract, key) for key in ("query_id_before", "query_id_middle", "query_id_after"))
    base_delete = cint(contract, "base_delete_id")
    if len({a, b, c}) != 3 or min(a, b, c) < header.base_n or max(a, b, c) >= header.pool_n:
        raise BundleContractError("g2_insert_ids_not_distinct_reservoir_ids")
    if not 0 <= base_delete < header.base_n:
        raise BundleContractError("g2_base_delete_not_in_initial_base")
    if any(not 0 <= q < header.query_n for q in (qa, qc, qpost)):
        raise BundleContractError("g2_query_id_out_of_range")
    expected = [
        (OP_INSERT, a), (OP_INSERT, b), (OP_KNN, qa), (OP_DELETE, a),
        (OP_INSERT, c), (OP_KNN, qc), (OP_DELETE, base_delete), (OP_REBUILD, 0),
        (OP_KNN, qpost),
    ]
    if len(events) != len(expected) or header.event_count != len(expected):
        raise BundleContractError("g2_trace_not_exact_nine_event_sequence")
    for index, (op, arg) in enumerate(expected):
        event = events[index]
        if event.op_index != index or event.op != op or event.argument != arg:
            raise BundleContractError(f"g2_trace_not_exact_direct_delta_reuse_delete_rebuild_sequence:op={index}")


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def exact_squared_l2(pool: array, stable_id: int, queries: array, query_id: int, dimension: int) -> int:
    start = stable_id * dimension
    qstart = query_id * dimension
    return sum((int(pool[start + axis]) - int(queries[qstart + axis])) ** 2 for axis in range(dimension))


def modeled_gts_float_l2(pool: array, stable_id: int, queries: array, query_id: int, dimension: int) -> float:
    start = stable_id * dimension
    qstart = query_id * dimension
    accumulated = 0.0
    for axis in range(dimension):
        diff = _f32(_f32(pool[start + axis]) - _f32(queries[qstart + axis]))
        accumulated += float(_f32(diff * diff))
    return _f32(math.sqrt(_f32(accumulated)))


def require_static_epoch_tie_free(pool: array, queries: array, base_ids: Iterable[int], header: TraceHeader,
                                  query_ids: Iterable[int], label: str) -> list[dict[str, Any]]:
    ids = sorted(base_ids)
    if len(ids) < header.k or len(ids) != len(set(ids)):
        raise BundleContractError(f"{BOUNDED_TIE_ABORT}:invalid_epoch_base_set:{label}")
    checks: list[dict[str, Any]] = []
    prefix = min(header.k + 1, len(ids))
    for qid in query_ids:
        exact = sorted((exact_squared_l2(pool, ident, queries, qid, header.dimension), ident) for ident in ids)
        modeled = sorted((modeled_gts_float_l2(pool, ident, queries, qid, header.dimension), ident) for ident in ids)
        for rank in range(1, prefix):
            if exact[rank - 1][0] == exact[rank][0]:
                raise BundleContractError(
                    f"{BOUNDED_TIE_ABORT}:static_{label}_exact_tie:q={qid}:ranks={rank-1},{rank}"
                )
            if modeled[rank - 1][0] == modeled[rank][0]:
                raise BundleContractError(
                    f"{BOUNDED_TIE_ABORT}:static_{label}_modeled_float_tie:q={qid}:ranks={rank-1},{rank}"
                )
        for rank in range(header.k):
            if exact[rank][1] != modeled[rank][1]:
                raise BundleContractError(
                    f"{BOUNDED_TIE_ABORT}:static_{label}_exact_modeled_order:q={qid}:rank={rank}"
                )
        checks.append({"epoch": label, "query_id": qid, "base_candidates": len(ids),
                       "static_prefix_count": prefix,
                       "exact_prefix_keys_pairwise_distinct": True,
                       "modeled_gts_float_prefix_keys_pairwise_distinct": True,
                       "top_k_id_order_matches_exact_vs_modeled_gts_float": True})
    return checks


def require_dynamic_k_boundary(pool: array, queries: array, active: Iterable[int], header: TraceHeader,
                               query_id: int, op_index: int) -> dict[str, int]:
    ranked = sorted((exact_squared_l2(pool, ident, queries, query_id, header.dimension), ident) for ident in active)
    if len(ranked) < header.k:
        raise BundleContractError(f"active_population_below_k:op={op_index}")
    if len(ranked) > header.k and ranked[header.k - 1][0] == ranked[header.k][0]:
        raise BundleContractError(f"{BOUNDED_TIE_ABORT}:dynamic_k_boundary_tie:op={op_index}:q={query_id}")
    return {"op_index": op_index, "query_id": query_id, "active_count": len(ranked)}


def topk_exact(pool: array, queries: array, active: Iterable[int], header: TraceHeader, query_id: int) -> list[tuple[int, int]]:
    return sorted((exact_squared_l2(pool, ident, queries, query_id, header.dimension), ident) for ident in active)[:header.k]


def replay_g2_trace(header: TraceHeader, events: list[TraceEvent], contract: dict[str, str]) -> dict[str, Any]:
    """Replay only stable-ID state semantics; no claim about certificate execution."""
    validate_strict_g2_trace(header, events, contract)
    base = set(range(header.base_n))
    active = set(base)
    placement: dict[int, str] = {ident: "base" for ident in base}
    a, b, c = (cint(contract, key) for key in ("stable_id_a", "stable_id_b", "stable_id_c"))
    deleted_base = cint(contract, "base_delete_id")
    query_states: dict[int, list[int]] = {}
    transient_before: dict[str, list[int]] | None = None
    active_before_rebuild: list[int] | None = None
    active_hash_before: int | None = None
    rebuild_epoch = 0
    for event in events:
        if event.op == OP_INSERT:
            if event.argument in active:
                raise BundleContractError(f"replay_duplicate_active_insert:op={event.op_index}")
            active.add(event.argument)
            if event.argument == a:
                placement[a] = "direct"
            elif event.argument == b:
                placement[b] = "delta"
            elif event.argument == c:
                placement[c] = "direct"
            else:
                raise BundleContractError(f"replay_unexpected_insert:op={event.op_index}")
        elif event.op == OP_DELETE:
            if event.argument not in active:
                raise BundleContractError(f"replay_inactive_delete:op={event.op_index}")
            active.remove(event.argument)
            placement[event.argument] = "deleted"
        elif event.op == OP_KNN:
            query_states[event.op_index] = sorted(active)
        elif event.op == OP_REBUILD:
            if event.op_index == 0 or events[event.op_index - 1].op != OP_DELETE or events[event.op_index - 1].argument != deleted_base:
                raise BundleContractError("rebuild_not_immediately_after_base_delete")
            active_before_rebuild = sorted(active)
            active_hash_before = fnv1a_i32(active_before_rebuild)
            transient_before = {
                "direct_ids": sorted(ident for ident, kind in placement.items() if kind == "direct"),
                "delta_ids": sorted(ident for ident, kind in placement.items() if kind == "delta"),
                "deleted_ids": sorted(ident for ident, kind in placement.items() if kind == "deleted"),
            }
            if transient_before["direct_ids"] != [c] or transient_before["delta_ids"] != [b] or a not in transient_before["deleted_ids"] or deleted_base not in transient_before["deleted_ids"]:
                raise BundleContractError("rebuild_prestate_not_expected_direct_delta_deleted_partition")
            placement = {ident: "base" for ident in active}
            rebuild_epoch += 1
        else:
            raise BundleContractError(f"replay_unknown_op:{event.op_index}")
    if rebuild_epoch != 1 or active_before_rebuild is None or transient_before is None or active_hash_before is None:
        raise BundleContractError("replay_missing_single_rebuild")
    final_active = sorted(active)
    if fnv1a_i32(final_active) != active_hash_before:
        raise BundleContractError("rebuild_changed_active_stable_id_set")
    if any(kind != "base" for kind in placement.values()) or set(placement) != set(final_active):
        raise BundleContractError("rebuild_did_not_clear_transient_tiers")
    return {
        "initial_base_ids": sorted(base),
        "query_active_ids": {str(op): values for op, values in query_states.items()},
        "active_before_rebuild": active_before_rebuild,
        "active_after_rebuild": final_active,
        "active_hash_before_rebuild": active_hash_before,
        "active_hash_after_rebuild": fnv1a_i32(final_active),
        "transient_before_rebuild": transient_before,
        "transient_after_rebuild": {"direct_ids": [], "delta_ids": []},
        "final_placement": {str(ident): placement[ident] for ident in final_active},
    }
