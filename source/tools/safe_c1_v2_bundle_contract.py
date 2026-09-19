"""CPU-only bounded-tie witness preflight shared by Safe-C1 G1 v2.

No CUDA/GPU calls are present here. The static base contract proves only a
deterministic top-(k+1) prefix; ties after that prefix are permitted. This is
not a canonical tie semantics implementation for all inputs.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import sys
from typing import Any

MAGIC = b"E1GTRC01"
VERSION = 1
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
OP_INSERT = 1
OP_DELETE = 2
OP_KNN = 3
OP_RANGE = 4

# These strings are part of the cross-language bounded witness contract.  They
# intentionally restrict only the static top-(k+1) prefix and dynamic k/k+1
# membership boundary; they do not claim all-input canonical tie semantics.
BOUNDED_TIE_DOMAIN = "bounded static-prefix and dynamic-boundary witness only"
STATIC_BASE_POLICY = (
    "first min(k+1,base_n) exact squared-L2 and modeled GTS float-L2 keys "
    "pairwise distinct; first-k ID order agrees"
)
DYNAMIC_K_BOUNDARY_POLICY = (
    "k and k+1 exact quantized squared-L2 keys distinct for every trace kNN state"
)
BOUNDED_TIE_ABORT = "ABORT_UNPROVED_BOUNDED_TIE_WITNESS_DOMAIN"
BOUNDED_TIE_PREFLIGHT_STATUS = "PASS_CPU_ONLY_BOUNDED_TIE_WITNESS"
CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE = "explicit_identity_only_current_v2_runner"


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
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_i16(path: Path, expected_values: int) -> array:
    raw = path.read_bytes()
    if len(raw) != expected_values * 2:
        raise BundleContractError(
            f"wrong_i16_byte_count:{path.name}:expected={expected_values * 2}:actual={len(raw)}"
        )
    output = array("h")
    output.frombytes(raw)
    if sys.byteorder != "little":
        output.byteswap()
    return output


def write_i16(path: Path, values: array) -> None:
    copy = array("h", values)
    if sys.byteorder != "little":
        copy.byteswap()
    path.write_bytes(copy.tobytes())


def load_i32_le(path: Path, expected_values: int | None = None) -> list[int]:
    """Read an explicitly little-endian signed-i32 payload.

    Stable IDs are part of the runner's data contract, not host-native array
    bytes. Keep the endianness explicit so CPU preflight and finalization do
    not silently reinterpret a mapping on a different host architecture.
    """
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise BundleContractError(
            f"wrong_i32_byte_count:{path.name}:actual={len(raw)}"
        )
    count = len(raw) // 4
    if expected_values is not None and count != expected_values:
        raise BundleContractError(
            f"wrong_i32_value_count:{path.name}:expected={expected_values}:actual={count}"
        )
    return [value[0] for value in struct.iter_unpack("<i", raw)]


def write_i32_le(path: Path, values: list[int]) -> None:
    """Write signed i32 values in the explicit little-endian bundle format."""
    try:
        path.write_bytes(b"".join(struct.pack("<i", int(value)) for value in values))
    except struct.error as exc:
        raise BundleContractError(f"i32_value_out_of_range:{path.name}:{exc}") from exc


def load_current_runner_identity_stable_id_layout(
    bundle: Path, header: TraceHeader
) -> tuple[list[int], list[int]]:
    """Load the stable-ID layout accepted by the current v2 CUDA runner.

    The current search-native runner addresses ``pool.i16`` directly as
    ``pool[stable_id]`` and initializes the base as IDs ``[0, base_n)``. It
    does *not* consume an arbitrary stable-ID mapping file. Therefore a
    mapping permutation would make a CPU oracle validate different data than
    the executable. New certificate-selection bundles must carry both mapping
    files and they must explicitly certify this identity-only layout.
    """
    mapping_path = bundle / "stable_id_to_pool_row.i32"
    base_ids_path = bundle / "initial_base_stable_ids.i32"
    if not mapping_path.is_file():
        raise BundleContractError("stable_id_layout_missing:stable_id_to_pool_row.i32")
    if not base_ids_path.is_file():
        raise BundleContractError("stable_id_layout_missing:initial_base_stable_ids.i32")
    stable_to_row = load_i32_le(mapping_path, header.pool_n)
    initial_base_ids = load_i32_le(base_ids_path, header.base_n)
    expected_mapping = list(range(header.pool_n))
    expected_base = list(range(header.base_n))
    if stable_to_row != expected_mapping:
        raise BundleContractError(
            "stable_id_layout_not_identity_for_current_runner:stable_id_to_pool_row.i32"
        )
    if initial_base_ids != expected_base:
        raise BundleContractError(
            "stable_id_layout_not_identity_for_current_runner:initial_base_stable_ids.i32"
        )
    return stable_to_row, initial_base_ids


def parse_trace(path: Path) -> tuple[TraceHeader, list[TraceEvent]]:
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise BundleContractError("truncated_trace_header")
    magic, version, dimension, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = (
        HEADER.unpack_from(raw, 0)
    )
    if magic != MAGIC or version != VERSION:
        raise BundleContractError("unsupported_trace_magic_or_version")
    if dimension <= 0 or base_n <= 0 or pool_n < base_n or reservoir_n != pool_n - base_n:
        raise BundleContractError("invalid_trace_population_header")
    if query_n <= 0 or k <= 0 or k > base_n or radius < 0:
        raise BundleContractError("invalid_trace_query_header")
    expected_bytes = HEADER.size + event_count * EVENT.size
    if len(raw) != expected_bytes:
        raise BundleContractError(
            f"wrong_trace_byte_count:expected={expected_bytes}:actual={len(raw)}"
        )
    header = TraceHeader(dimension, base_n, reservoir_n, pool_n, query_n, k, radius, event_count)
    events: list[TraceEvent] = []
    for index in range(event_count):
        op_index, op, argument = EVENT.unpack_from(raw, HEADER.size + index * EVENT.size)
        if op_index != index or op not in {OP_INSERT, OP_DELETE, OP_KNN, OP_RANGE}:
            raise BundleContractError(f"invalid_trace_event:{index}")
        events.append(TraceEvent(op_index, op, argument))
    return header, events


def write_trace(path: Path, header: TraceHeader, events: list[TraceEvent]) -> None:
    if header.event_count != len(events):
        raise BundleContractError("trace_event_count_mismatch")
    raw = HEADER.pack(
        MAGIC,
        VERSION,
        header.dimension,
        header.base_n,
        header.reservoir_n,
        header.pool_n,
        header.query_n,
        header.k,
        float(header.radius),
        header.event_count,
    )
    for expected_index, event in enumerate(events):
        if event.op_index != expected_index:
            raise BundleContractError("noncontiguous_output_event_index")
        raw += EVENT.pack(event.op_index, event.op, event.argument)
    path.write_bytes(raw)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def exact_squared_l2(pool: array, point_id: int, queries: array, query_id: int, dimension: int) -> int:
    point_offset = point_id * dimension
    query_offset = query_id * dimension
    total = 0
    for axis in range(dimension):
        delta = int(pool[point_offset + axis]) - int(queries[query_offset + axis])
        total += delta * delta
    return total


def modeled_gts_float_l2_key(
    pool: array, point_id: int, queries: array, query_id: int, dimension: int
) -> float:
    """Model dataProcessKnnVec L2's float product/double sum/sqrtf(float(sum))."""
    point_offset = point_id * dimension
    query_offset = query_id * dimension
    accumulated = 0.0
    for axis in range(dimension):
        diff = _f32(_f32(pool[point_offset + axis]) - _f32(queries[query_offset + axis]))
        accumulated += float(_f32(diff * diff))
    return _f32(math.sqrt(_f32(accumulated)))


def validate_tie_free_witness(
    pool: array, queries: array, header: TraceHeader, events: list[TraceEvent]
) -> dict[str, Any]:
    """Raise on any forbidden tie or invalid G1 event; return CPU-only metrics."""
    if len(pool) != header.pool_n * header.dimension:
        raise BundleContractError("pool_shape_does_not_match_trace")
    if len(queries) != header.query_n * header.dimension:
        raise BundleContractError("query_shape_does_not_match_trace")

    static_checks: list[dict[str, int | bool]] = []
    static_prefix_count = min(header.base_n, header.k + 1)
    for query_id in range(header.query_n):
        exact_rows = sorted(
            (exact_squared_l2(pool, stable_id, queries, query_id, header.dimension), stable_id)
            for stable_id in range(header.base_n)
        )
        modeled_rows = sorted(
            (modeled_gts_float_l2_key(pool, stable_id, queries, query_id, header.dimension), stable_id)
            for stable_id in range(header.base_n)
        )
        for rank in range(1, static_prefix_count):
            left, right = exact_rows[rank - 1], exact_rows[rank]
            if left[0] == right[0]:
                raise BundleContractError(
                    BOUNDED_TIE_ABORT + ":"
                    f"static_base_top_k_plus_one_exact_squared_l2_tie:q={query_id}:"
                    f"ranks={rank - 1},{rank}:ids={left[1]},{right[1]}:squared_l2={right[0]}"
                )
            left, right = modeled_rows[rank - 1], modeled_rows[rank]
            if left[0] == right[0]:
                raise BundleContractError(
                    BOUNDED_TIE_ABORT + ":"
                    f"static_base_top_k_plus_one_modeled_gts_float_key_tie:q={query_id}:"
                    f"ranks={rank - 1},{rank}:ids={left[1]},{right[1]}:modeled_float_l2={right[0]}"
                )
        for rank in range(header.k):
            if exact_rows[rank][1] != modeled_rows[rank][1]:
                raise BundleContractError(
                    BOUNDED_TIE_ABORT + ":"
                    f"static_base_top_k_order_mismatch_exact_vs_modeled_gts_float:q={query_id}:"
                    f"rank={rank}:exact_id={exact_rows[rank][1]}:modeled_gts_id={modeled_rows[rank][1]}"
                )
        static_checks.append({
            "query_id": query_id,
            "base_candidates": header.base_n,
            "static_prefix_count": static_prefix_count,
            "exact_prefix_keys_pairwise_distinct": True,
            "modeled_gts_float_prefix_keys_pairwise_distinct": True,
            "top_k_id_order_matches_exact_vs_modeled_gts_float": True,
        })

    active = [False] * header.pool_n
    for stable_id in range(header.base_n):
        active[stable_id] = True
    dynamic_checks: list[dict[str, int]] = []
    for event in events:
        stable_id = event.argument
        if event.op == OP_INSERT:
            if not header.base_n <= stable_id < header.pool_n:
                raise BundleContractError(f"invalid_g1_reservoir_insert:op={event.op_index}")
            if active[stable_id]:
                raise BundleContractError(f"duplicate_active_insert:op={event.op_index}:id={stable_id}")
            active[stable_id] = True
        elif event.op == OP_DELETE:
            if not 0 <= stable_id < header.pool_n:
                raise BundleContractError(f"invalid_delete_id:op={event.op_index}")
            if stable_id < header.base_n:
                raise BundleContractError(f"forbidden_base_delete:op={event.op_index}:id={stable_id}")
            if not active[stable_id]:
                raise BundleContractError(f"inactive_delete:op={event.op_index}:id={stable_id}")
            active[stable_id] = False
        elif event.op == OP_KNN:
            query_id = stable_id
            if not 0 <= query_id < header.query_n:
                raise BundleContractError(f"invalid_query_id:op={event.op_index}")
            rows = sorted(
                (exact_squared_l2(pool, object_id, queries, query_id, header.dimension), object_id)
                for object_id, live in enumerate(active)
                if live
            )
            if len(rows) < header.k:
                raise BundleContractError(f"active_population_below_k:op={event.op_index}")
            if len(rows) > header.k and rows[header.k - 1][0] == rows[header.k][0]:
                raise BundleContractError(
                    BOUNDED_TIE_ABORT + ":"
                    f"dynamic_k_boundary_distance_tie:op={event.op_index}:q={query_id}:"
                    f"ids={rows[header.k - 1][1]},{rows[header.k][1]}:squared_l2={rows[header.k - 1][0]}"
                )
            dynamic_checks.append(
                {"op_index": event.op_index, "query_id": query_id, "active_count": len(rows)}
            )
        elif event.op == OP_RANGE:
            raise BundleContractError(f"forbidden_g1_range_event:op={event.op_index}")
        else:
            raise BundleContractError(f"unknown_event_opcode:op={event.op_index}")

    return {
        "schema": "safe-c1-g1-v2-tie-free-witness-preflight",
        "status": BOUNDED_TIE_PREFLIGHT_STATUS,
        "gpu_used": False,
        "cuda_binary_executed": False,
        "static_base_queries_checked": len(static_checks),
        "dynamic_k_boundary_queries_checked": len(dynamic_checks),
        "static_base_policy": STATIC_BASE_POLICY,
        "dynamic_k_boundary_policy": DYNAMIC_K_BOUNDARY_POLICY,
        "on_violation": BOUNDED_TIE_ABORT,
        "global_canonical_distance_stable_id_proof": False,
        "static_checks": static_checks,
        "dynamic_checks": dynamic_checks,
    }
