#!/usr/bin/env python3
"""CPU-only selector for a *native-exported* Fable5 frozen E0 snapshot.

It never constructs a GTS tree, invokes a CUDA binary, probes a device, or
assigns fixture certificate labels.  It only accepts a snapshot emitted by the
separate native exporter, verifies its byte pins and frozen-tree hash, mirrors
the source's strict sibling-boundary certificate against that frozen metadata,
and writes a 12-op trace only when every required condition is satisfied.

A successful bundle is only ``CPU_CANDIDATE_NOT_NATIVE_RUNTIME_VALIDATED``:
the matched GPU runner must recompute every certificate and prove receipt-based
query visibility.  A missing snapshot or absent natural certificate reject
produces only a blocking report, never an invented trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "fable5-native-frozen-snapshot-v1"
TRACE_MAGIC = b"E1GTRC02"
TRACE_VERSION = 2
OP_INSERT = 1
OP_DELETE = 2
OP_KNN = 3
OP_REBUILD = 5
TREE_ORDER = 10
STRICT_EPSILON = struct.unpack("<f", struct.pack("<f", 1.0e-5))[0]
PREFLIGHT_CONTRACT_PATH = ROOT / "manifests" / "fable5_matched_trace_preflight_v1.json"
TRACE_FILENAME = "trace.fable5.e1gtrc"
CANDIDATE_FILENAME = "fable5_matched_trace_candidate_v1.json"
CANDIDATE_SCHEMA = "fable5-matched-trace-candidate-v1"
CANDIDATE_STATUS = "CPU_CANDIDATE_NOT_NATIVE_RUNTIME_VALIDATED"
OP_NAMES = {OP_INSERT: "INSERT", OP_DELETE: "DELETE", OP_KNN: "KNN", OP_REBUILD: "REBUILD"}


class Blocked(RuntimeError):
    """A truthful inability to select a native-backed trace."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fnv1a(raw: bytes, seed: int = 1469598103934665603) -> int:
    value = seed
    for byte in raw:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def float_from_bits(bits: int) -> float:
    if not isinstance(bits, int) or bits < 0 or bits > 0xFFFFFFFF:
        raise Blocked("invalid_f32_bits")
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Blocked(message)


def read_header(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    require(len(raw) >= 48, "truncated_header_trace")
    magic, version, dimension, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = struct.unpack(
        "<8sIIIIIIIfQ", raw[:48]
    )
    require(magic == TRACE_MAGIC and version == TRACE_VERSION, "unsupported_header_trace_abi")
    require(
        dimension > 0
        and base_n > 0
        and reservoir_n == pool_n - base_n
        and pool_n >= base_n
        and query_n > 0
        and 0 < k <= base_n
        and math.isfinite(radius)
        and radius >= 0.0,
        "invalid_header_trace",
    )
    require(len(raw) == 48 + event_count * 12, "header_trace_event_length_mismatch")
    return {
        "dimension": dimension,
        "base_n": base_n,
        "reservoir_n": reservoir_n,
        "pool_n": pool_n,
        "query_n": query_n,
        "k": k,
        "radius": radius,
        "event_count": event_count,
    }


def read_i16(path: Path, expected_count: int) -> list[int]:
    raw = path.read_bytes()
    require(len(raw) == expected_count * 2, f"unexpected_i16_size:{path.name}")
    return list(struct.unpack(f"<{expected_count}h", raw))


def read_i32(path: Path, expected_count: int) -> list[int]:
    raw = path.read_bytes()
    require(len(raw) == expected_count * 4, f"unexpected_i32_size:{path.name}")
    return list(struct.unpack(f"<{expected_count}i", raw))


def source_semantics(source: Path) -> dict[str, bool]:
    text = source.read_text(encoding="utf-8")
    required = {
        "strict_lower": "distance > nodes[child].min_dis + kStrictEpsilon",
        "strict_upper": "distance < nodes[next].min_dis - kStrictEpsilon",
        "full_sibling_block": "for (int slot = 0; slot < fanout; ++slot)",
        "common_pivot": "common_pivot_or_reject",
        "fail_closed": "if (child < 0) return -1;",
        "native_leaf_return": "if (nodes[child].is_leaf == 1) return child;",
    }
    result = {name: needle in text for name, needle in required.items()}
    require(all(result.values()), "source_certificate_semantic_anchor_missing")
    return result


def parse_snapshot(snapshot_path: Path, bundle: Path, header_trace: Path, source: Path,
                   exporter_source: Path, matched_runner_source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not snapshot_path.is_file():
        raise Blocked("missing_native_frozen_snapshot")
    try:
        value = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        raise Blocked(f"invalid_snapshot_json:{error}") from error
    require(value.get("schema") == SCHEMA, "wrong_snapshot_schema")
    require(value.get("status") == "NATIVE_E0_CAPTURED_PENDING_CPU_SELECTOR", "snapshot_not_native_e0_capture")
    require(value.get("residual_pruning", {}).get("mode") == 0, "snapshot_not_baseline_residual_mode_0")
    require(value.get("residual_pruning", {}).get("query_executed") is False, "snapshot_export_must_not_claim_query")
    require(value.get("c2_used") is False, "snapshot_must_record_c2_used_false")
    require(value.get("c3_used") is False, "snapshot_must_record_c3_used_false")

    immutable = value.get("immutable_input")
    require(isinstance(immutable, dict), "snapshot_missing_immutable_input")
    expected_paths = {
        "header_trace_sha256": header_trace,
        "pool_i16_sha256": bundle / "pool.i16",
        "queries_i16_sha256": bundle / "queries.i16",
        "initial_base_stable_ids_i32_sha256": bundle / "initial_base_stable_ids.i32",
        "stable_id_to_pool_row_i32_sha256": bundle / "stable_id_to_pool_row.i32",
        "core_source_sha256": source,
        "exporter_source_sha256": exporter_source,
        "matched_runner_source_sha256": matched_runner_source,
    }
    observed_hashes: dict[str, str] = {}
    for key, path in expected_paths.items():
        require(path.is_file(), f"missing_pinned_input:{path}")
        actual = sha256_file(path)
        observed_hashes[key] = actual
        require(immutable.get(key) == actual, f"snapshot_hash_mismatch:{key}")

    return value, observed_hashes


def normalize_snapshot(value: dict[str, Any], header: dict[str, Any], initial_ids: list[int]) -> dict[str, Any]:
    snapshot_header = value.get("header")
    require(isinstance(snapshot_header, dict), "snapshot_missing_header")
    for key in ("dimension", "base_n", "reservoir_n", "pool_n", "query_n", "k"):
        require(snapshot_header.get(key) == header[key], f"snapshot_header_mismatch:{key}")
    require(f32(float(snapshot_header.get("radius"))) == f32(float(header["radius"])), "snapshot_header_mismatch:radius")
    require(value.get("fanout") == TREE_ORDER, "snapshot_wrong_fanout")
    require(isinstance(value.get("tree_height"), int) and value["tree_height"] > 1, "snapshot_invalid_tree_height")
    require(value.get("tn_abi") == "live-only:little-endian:i32,f32bits,i32,i32,i32;empty-nodes=null", "snapshot_unknown_tn_abi")
    require(value.get("tn_size_bytes") == 20, "snapshot_unknown_tn_size")

    raw_nodes = value.get("nodes")
    raw_empty = value.get("empty")
    raw_max = value.get("max_distance_f32_bits")
    require(isinstance(raw_nodes, list) and isinstance(raw_empty, list) and isinstance(raw_max, list), "snapshot_arrays_missing")
    require(len(raw_nodes) > 0 and len(raw_nodes) == len(raw_empty) == len(raw_max), "snapshot_array_length_mismatch")

    empty: list[int] = []
    for flag in raw_empty:
        require(isinstance(flag, int) and flag in (0, 1), "invalid_snapshot_empty_flag")
        empty.append(flag)

    # Empty TN slots are not initialized by the legacy cudaMalloc constructor.
    # The exporter serializes those slots as JSON null; no field from them may
    # participate in the selector or canonical identity.
    nodes: list[dict[str, Any] | None] = []
    max_bits: list[int | None] = []
    for index, (raw_node, raw_maximum) in enumerate(zip(raw_nodes, raw_max)):
        if empty[index] != 0:
            require(raw_node is None and raw_maximum is None, "empty_node_must_be_null")
            nodes.append(None)
            max_bits.append(None)
            continue
        require(isinstance(raw_node, dict), "invalid_live_snapshot_node")
        pid = raw_node.get("pid")
        size = raw_node.get("size")
        lid = raw_node.get("lid")
        leaf = raw_node.get("is_leaf")
        bits = raw_node.get("min_dis_f32_bits")
        require(all(isinstance(item, int) for item in (pid, size, lid, leaf, bits)), "noninteger_live_snapshot_node_field")
        require(-2**31 <= pid < 2**31 and -2**31 <= size < 2**31 and -2**31 <= lid < 2**31, "live_snapshot_i32_out_of_range")
        minimum = float_from_bits(bits)
        require(leaf in (0, 1) and size >= 0 and lid >= 0 and math.isfinite(minimum), "invalid_live_snapshot_node_shape")
        require(isinstance(raw_maximum, int), "missing_live_max_distance_bits")
        maximum_bits = int(raw_maximum)
        _ = float_from_bits(maximum_bits)
        nodes.append({"pid": pid, "min": minimum, "min_bits": bits, "size": size, "lid": lid, "leaf": leaf})
        max_bits.append(maximum_bits)

    logical_payload = value.get("logical_leaf_payload")
    require(isinstance(logical_payload, list), "snapshot_missing_logical_leaf_payload")
    logical_layout: list[int] = []
    flattened: list[int] = []
    seen_leaf_nodes: set[int] = set()
    for item in logical_payload:
        require(isinstance(item, dict), "invalid_logical_leaf_payload")
        node_id = item.get("node_id")
        ids = item.get("ids")
        require(isinstance(node_id, int) and isinstance(ids, list), "invalid_logical_leaf_shape")
        require(0 <= node_id < len(nodes) and node_id not in seen_leaf_nodes, "duplicate_or_invalid_leaf_node")
        seen_leaf_nodes.add(node_id)
        node = nodes[node_id]
        require(node is not None and empty[node_id] == 0 and node["leaf"] == 1, "payload_not_for_live_leaf")
        require(item.get("lid") == node["lid"] and item.get("size") == node["size"], "payload_node_metadata_mismatch")
        require(len(ids) == node["size"] and all(isinstance(ident, int) for ident in ids), "payload_id_shape_mismatch")
        logical_layout.extend([node_id, node["size"], *ids])
        flattened.extend(ids)
    expected_leaf_nodes = {index for index, node in enumerate(nodes) if empty[index] == 0 and node is not None and node["leaf"] == 1}
    require(seen_leaf_nodes == expected_leaf_nodes, "logical_leaf_payload_incomplete")
    require(sorted(flattened) == initial_ids, "snapshot_leaf_ids_do_not_equal_initial_base")
    logical_raw = struct.pack(f"<{len(logical_layout)}i", *logical_layout) if logical_layout else b""
    logical_hash = fnv1a(logical_raw)
    logical_count = len(flattened)

    native_hashes = value.get("native_hashes")
    require(isinstance(native_hashes, dict), "snapshot_missing_native_hashes")
    require(native_hashes.get("logical_leaf_id_fnv1a64") == logical_hash, "logical_leaf_hash_mismatch")
    require(native_hashes.get("logical_leaf_id_count") == logical_count, "logical_leaf_count_mismatch")

    # Mirror canonical_live_tree_hash in the standalone exporter.  It includes
    # all initialized empty flags and only initialized live TN/max-distance
    # fields, so it is stable across allocations that leave empty TN bytes
    # indeterminate.
    canonical = fnv1a(struct.pack("<i", value["tree_height"]))
    canonical = fnv1a(struct.pack("<i", value["fanout"]), canonical)
    canonical = fnv1a(struct.pack("<i", len(nodes)), canonical)
    for node_id, node in enumerate(nodes):
        canonical = fnv1a(struct.pack("<i", node_id), canonical)
        canonical = fnv1a(struct.pack("<i", empty[node_id]), canonical)
        if node is not None:
            canonical = fnv1a(struct.pack("<iIiii", node["pid"], node["min_bits"], node["size"], node["lid"], node["leaf"]), canonical)
            assert max_bits[node_id] is not None
            canonical = fnv1a(struct.pack("<I", max_bits[node_id]), canonical)
    canonical = fnv1a(struct.pack("<i", logical_count), canonical)
    canonical = fnv1a(struct.pack("<Q", logical_hash), canonical)
    require(native_hashes.get("canonical_live_tree_fnv1a64") == canonical, "canonical_live_tree_hash_mismatch")
    return {"nodes": nodes, "empty": empty, "height": value["tree_height"], "fanout": value["fanout"], "canonical_live_hash": canonical}


def encoded_pool_fnv1a(pool: list[int]) -> int:
    # GtsScalar is float in the pinned V5 configuration.  This exactly mirrors
    # the exporter's immutable_pool vector bytes and binds the snapshot to the
    # actual immutable vector payload rather than only its file SHA.
    return fnv1a(b"".join(struct.pack("<f", float(value)) for value in pool))

def valid_nonempty(tree: dict[str, Any], node_id: int) -> bool:
    return (0 <= node_id < len(tree["nodes"]) and tree["empty"][node_id] == 0
            and tree["nodes"][node_id] is not None)


def child_id(parent: int, slot: int) -> int:
    return parent * TREE_ORDER + slot + 1


def common_pivot_or_reject(tree: dict[str, Any], parent: int) -> tuple[int, str | None]:
    if parent < 0:
        return -1, "negative_parent"
    pivot = -1
    previous: float | None = None
    for slot in range(TREE_ORDER):
        child = child_id(parent, slot)
        if not valid_nonempty(tree, child):
            return -1, f"missing_or_empty_sibling:{parent}:{slot}"
        node = tree["nodes"][child]
        if node["pid"] < 0 or not math.isfinite(node["min"]):
            return -1, f"invalid_pivot_or_min:{child}"
        if pivot < 0:
            pivot = node["pid"]
        elif node["pid"] != pivot:
            return -1, f"inconsistent_sibling_pivot:{parent}"
        if previous is not None and not (node["min"] > f32(previous + STRICT_EPSILON)):
            return -1, f"non_strict_sibling_min:{parent}:{slot}"
        previous = node["min"]
    return pivot, None


def host_l2(pool: list[int], dimension: int, left: int, right: int) -> float:
    # Matches HostVectorPool::l2's float accumulator; the native runner will
    # recompute it again at runtime, so this is selection evidence, not a
    # substitute for that receipt.
    total = f32(0.0)
    offset_l = left * dimension
    offset_r = right * dimension
    for axis in range(dimension):
        delta = f32(float(pool[offset_l + axis]) - float(pool[offset_r + axis]))
        total = f32(total + f32(delta * delta))
    return f32(math.sqrt(total))


def certify(tree: dict[str, Any], pool: list[int], dimension: int, inserted: int) -> dict[str, Any]:
    parent = 0
    path: list[dict[str, Any]] = []
    for level in range(tree["height"] - 1):
        pivot, reason = common_pivot_or_reject(tree, parent)
        if pivot < 0:
            return {"leaf": -1, "reason": reason, "path": path}
        distance = host_l2(pool, dimension, inserted, pivot)
        if not math.isfinite(distance) or distance < 0.0:
            return {"leaf": -1, "reason": f"invalid_distance:{level}", "path": path}
        match = -1
        matched_lower = None
        matched_upper = None
        for slot in range(TREE_ORDER):
            child = child_id(parent, slot)
            if not valid_nonempty(tree, child):
                return {"leaf": -1, "reason": f"missing_child_after_audit:{parent}:{slot}", "path": path}
            node = tree["nodes"][child]
            lower_ok = distance > f32(node["min"] + STRICT_EPSILON)
            upper_ok = True
            upper = None
            if slot + 1 < TREE_ORDER:
                next_child = child_id(parent, slot + 1)
                if not valid_nonempty(tree, next_child):
                    return {"leaf": -1, "reason": f"missing_next_sibling:{parent}:{slot}", "path": path}
                upper = tree["nodes"][next_child]["min"]
                upper_ok = distance < f32(upper - STRICT_EPSILON)
            if lower_ok and upper_ok:
                if match != -1:
                    return {"leaf": -1, "reason": f"multiple_strict_children:{parent}", "path": path}
                match = child
                matched_lower = node["min"]
                matched_upper = upper
        if match < 0:
            return {
                "leaf": -1,
                "reason": f"no_strict_child:{parent}:level{level}",
                "path": path + [{"level": level, "parent": parent, "pivot": pivot, "distance_f32": distance, "child": -1}],
            }
        step = {
            "level": level,
            "parent": parent,
            "pivot": pivot,
            "distance_f32": distance,
            "child": match,
            "lower_min_f32": matched_lower,
            "upper_next_min_f32": matched_upper,
            "lower_margin_f32": f32(distance - f32(matched_lower + STRICT_EPSILON)),
            "upper_margin_f32": None if matched_upper is None else f32(f32(matched_upper - STRICT_EPSILON) - distance),
        }
        path.append(step)
        if tree["nodes"][match]["leaf"] == 1:
            return {"leaf": match, "reason": None, "path": path}
        parent = match
    return {"leaf": -1, "reason": "path_ended_without_leaf", "path": path}


def squared_l2(pool: list[int], queries: list[int], dimension: int, ident: int, query_id: int) -> int:
    total = 0
    point = ident * dimension
    query = query_id * dimension
    for axis in range(dimension):
        delta = pool[point + axis] - queries[query + axis]
        total += delta * delta
    return total


def modeled_gts_l2(pool: list[int], queries: list[int], dimension: int, ident: int, query_id: int) -> float:
    # Exact source structure: float delta/product -> double sum -> float sqrt.
    total = 0.0
    point = ident * dimension
    query = query_id * dimension
    for axis in range(dimension):
        delta = f32(float(pool[point + axis]) - float(queries[query + axis]))
        total += float(f32(delta * delta))
    return f32(math.sqrt(f32(total)))


def sorted_exact(active: Iterable[int], pool: list[int], queries: list[int], dimension: int, query_id: int) -> list[tuple[int, int]]:
    return sorted((squared_l2(pool, queries, dimension, ident, query_id), ident) for ident in active)


def static_epoch_gate(active: set[int], pool: list[int], queries: list[int], dimension: int, k: int) -> tuple[bool, str | None]:
    if len(active) < k:
        return False, "epoch_below_k"
    for query_id in range(len(queries) // dimension):
        exact = sorted_exact(active, pool, queries, dimension, query_id)
        modeled = sorted((modeled_gts_l2(pool, queries, dimension, ident, query_id), ident) for ident in active)
        prefix = min(len(active), k + 1)
        for rank in range(1, prefix):
            if exact[rank - 1][0] == exact[rank][0]:
                return False, f"static_exact_prefix_tie:q{query_id}:rank{rank}"
            if modeled[rank - 1][0] == modeled[rank][0]:
                return False, f"static_modeled_prefix_tie:q{query_id}:rank{rank}"
        for rank in range(k):
            if exact[rank][1] != modeled[rank][1]:
                return False, f"static_exact_modeled_order_mismatch:q{query_id}:rank{rank}"
    return True, None


def dynamic_gate(active: set[int], pool: list[int], queries: list[int], dimension: int, k: int, query_id: int) -> tuple[bool, str | None]:
    """Whole-set reference gate retained for synthetic equivalence checks only."""
    ordered = sorted_exact(active, pool, queries, dimension, query_id)
    if len(ordered) < k:
        return False, "dynamic_below_k"
    if len(ordered) > k and ordered[k - 1][0] == ordered[k][0]:
        return False, f"dynamic_boundary_tie:q{query_id}"
    return True, None


# The selector never explores the reservoir/cartesian product exhaustively.
# These are deterministic CPU work limits.  A limit hit is a blocker, not
# evidence that a real native-backed candidate cannot exist elsewhere.
MAX_QUERY_RANKED_PER_CLASS = 64
MAX_DIRECT_LEAF_GROUPS = 16
MAX_DIRECT_IDS_PER_LEAF = 6
MAX_BASE_DELETE_IDS = 32
MAX_TUPLE_EVALUATIONS = 50_000


def load_preflight_contract(source_bundle: Path, header: dict[str, Any]) -> dict[str, Any]:
    """Load and byte-pin the exact pre-NVML contract without importing its tool."""
    require(PREFLIGHT_CONTRACT_PATH.is_file(), "missing_fable5_preflight_contract")
    try:
        contract = json.loads(PREFLIGHT_CONTRACT_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Blocked(f"invalid_fable5_preflight_contract:{type(error).__name__}") from error
    require(contract.get("schema") == "fable5-matched-trace-preflight-contract-v1", "wrong_preflight_contract_schema")
    require(contract.get("status") == "STATIC_PRE_NVML_CONTRACT_ONLY", "wrong_preflight_contract_status")
    require(contract.get("candidate_contract") == {
        "filename": CANDIDATE_FILENAME,
        "schema": CANDIDATE_SCHEMA,
        "required_status": CANDIDATE_STATUS,
    }, "preflight_candidate_schema_mismatch")
    expected_header = contract.get("trace_abi", {}).get("header")
    require(isinstance(expected_header, dict), "preflight_header_missing")
    observed_header = {
        "magic": "E1GTRC02", "version": 2,
        "dimension": header["dimension"], "base_n": header["base_n"],
        "reservoir_n": header["reservoir_n"], "pool_n": header["pool_n"],
        "query_n": header["query_n"], "k": header["k"],
        "radius": float(header["radius"]), "event_count": 12,
    }
    require(observed_header == expected_header, "header_does_not_match_exact_fable5_preflight_contract")
    sequence = contract.get("event_sequence")
    runtime_requirements = contract.get("runtime_requirements")
    require(isinstance(sequence, list) and len(sequence) == 12, "preflight_event_sequence_missing")
    require(isinstance(runtime_requirements, dict), "preflight_runtime_requirements_missing")
    layout = contract.get("bundle_layout")
    require(isinstance(layout, dict) and layout.get("trace_filename") == TRACE_FILENAME, "preflight_trace_filename_mismatch")
    require(layout.get("stable_id_layout") == "identity_full_immutable_pool_seeded_stable_ids_v5", "preflight_stable_layout_mismatch")
    required_files = layout.get("required_files")
    require(isinstance(required_files, dict) and set(required_files) == {
        "pool.i16", "queries.i16", "stable_id_to_pool_row.i32", "initial_base_stable_ids.i32"
    }, "preflight_immutable_file_set_mismatch")
    for filename, detail in required_files.items():
        require(isinstance(detail, dict), f"preflight_file_detail_invalid:{filename}")
        path = source_bundle / filename
        require(path.is_file() and path.stat().st_size == detail.get("bytes"), f"source_bundle_file_size_mismatch:{filename}")
        require(sha256_file(path) == detail.get("sha256"), f"source_bundle_file_sha256_mismatch:{filename}")
    source_hashes = contract.get("root_source_hashes")
    require(isinstance(source_hashes, dict), "preflight_root_source_hashes_missing")
    for relative, digest in source_hashes.items():
        require(isinstance(relative, str) and isinstance(digest, str), "preflight_source_hash_field_invalid")
        path = ROOT / relative
        require(path.is_file() and sha256_file(path) == digest, f"preflight_root_source_hash_mismatch:{relative}")
    return contract


def assert_exact_event_contract(events: list[tuple[int, int]], slots: dict[str, int], contract: dict[str, Any]) -> None:
    expected = contract["event_sequence"]
    require(len(events) == len(expected) == 12, "fable5_event_count_not_exactly_12")
    for index, (expected_event, actual) in enumerate(zip(expected, events)):
        operation, argument = actual
        require(expected_event.get("op_index") == index, f"preflight_sequence_bad_index:{index}")
        expected_op = expected_event.get("op")
        require(OP_NAMES.get(operation) == expected_op, f"fable5_event_opcode_mismatch:{index}")
        slot = expected_event.get("slot")
        expected_argument = 0 if slot == "rebuild" else slots.get(str(slot))
        require(expected_argument is not None and argument == expected_argument, f"fable5_event_argument_mismatch:{index}")


class PrefixRanker:
    """Exact-prefix cache for bounded candidate selection.

    Only the first k+2 base records are needed: every E1 state removes at
    most one base record and adds at most two reservoir records.  Thus the
    merged prefix contains every possible top-(k+1) record while avoiding
    repeated full 4096-record sorts for each tuple.
    """

    def __init__(self, base: set[int], pool: list[int], queries: list[int], dimension: int, k: int) -> None:
        self.base_ids = tuple(sorted(base))
        self.base_set = set(base)
        self.pool = pool
        self.queries = queries
        self.dimension = dimension
        self.k = k
        self.query_n = len(queries) // dimension
        require(self.query_n > 0 and len(queries) == self.query_n * dimension, "invalid_query_vector_layout")
        require(len(self.base_ids) >= k + 2, "base_too_small_for_prefix_cache")
        self.prefix_length = k + 2
        self._exact_cache: dict[tuple[int, int], int] = {}
        self._modeled_cache: dict[tuple[int, int], float] = {}
        self._base_exact_prefix: dict[int, list[tuple[int, int]]] = {}
        self._base_modeled_prefix: dict[int, list[tuple[float, int]]] = {}
        for query_id in range(self.query_n):
            exact: list[tuple[int, int]] = []
            modeled: list[tuple[float, int]] = []
            for ident in self.base_ids:
                exact_distance = squared_l2(pool, queries, dimension, ident, query_id)
                modeled_distance = modeled_gts_l2(pool, queries, dimension, ident, query_id)
                self._exact_cache[(ident, query_id)] = exact_distance
                self._modeled_cache[(ident, query_id)] = modeled_distance
                exact.append((exact_distance, ident))
                modeled.append((modeled_distance, ident))
            exact.sort()
            modeled.sort()
            self._base_exact_prefix[query_id] = exact[:self.prefix_length]
            self._base_modeled_prefix[query_id] = modeled[:self.prefix_length]

    def _exact_distance(self, ident: int, query_id: int) -> int:
        key = (ident, query_id)
        if key not in self._exact_cache:
            self._exact_cache[key] = squared_l2(self.pool, self.queries, self.dimension, ident, query_id)
        return self._exact_cache[key]

    def _modeled_distance(self, ident: int, query_id: int) -> float:
        key = (ident, query_id)
        if key not in self._modeled_cache:
            self._modeled_cache[key] = modeled_gts_l2(self.pool, self.queries, self.dimension, ident, query_id)
        return self._modeled_cache[key]

    def _merged(self, extras: Iterable[int], query_id: int, removed_base: int | None,
                modeled: bool) -> list[tuple[float | int, int]]:
        require(0 <= query_id < self.query_n, "query_id_out_of_range")
        if removed_base is not None:
            require(removed_base in self.base_set, "removed_id_not_in_base")
        extra_ids = tuple(sorted(set(extras)))
        require(all(ident not in self.base_set for ident in extra_ids), "extra_id_must_not_be_initial_base")
        prefix = self._base_modeled_prefix[query_id] if modeled else self._base_exact_prefix[query_id]
        merged: list[tuple[float | int, int]] = [pair for pair in prefix if pair[1] != removed_base]
        for ident in extra_ids:
            distance: float | int
            if modeled:
                distance = self._modeled_distance(ident, query_id)
            else:
                distance = self._exact_distance(ident, query_id)
            merged.append((distance, ident))
        merged.sort()
        require(len(merged) >= self.k + 1, "prefix_merge_too_short")
        return merged

    def dynamic(self, extras: Iterable[int], query_id: int, removed_base: int | None = None) -> tuple[bool, str | None, list[int]]:
        exact = self._merged(extras, query_id, removed_base, modeled=False)
        if exact[self.k - 1][0] == exact[self.k][0]:
            return False, f"dynamic_boundary_tie:q{query_id}", []
        return True, None, [ident for _, ident in exact[:self.k]]

    def static_epoch(self, extras: Iterable[int] = (), removed_base: int | None = None) -> tuple[bool, str | None]:
        for query_id in range(self.query_n):
            exact = self._merged(extras, query_id, removed_base, modeled=False)
            modeled = self._merged(extras, query_id, removed_base, modeled=True)
            for rank in range(1, self.k + 1):
                if exact[rank - 1][0] == exact[rank][0]:
                    return False, f"static_exact_prefix_tie:q{query_id}:rank{rank}"
                if modeled[rank - 1][0] == modeled[rank][0]:
                    return False, f"static_modeled_prefix_tie:q{query_id}:rank{rank}"
            for rank in range(self.k):
                if exact[rank][1] != modeled[rank][1]:
                    return False, f"static_exact_modeled_order_mismatch:q{query_id}:rank{rank}"
        return True, None


def exact_visible_queries(ranker: PrefixRanker, extras: Iterable[int], target: int,
                          removed_base: int | None = None) -> list[int]:
    """Return only exact-top-k, boundary-safe query IDs from the cached merge."""
    visible: list[int] = []
    for query_id in range(ranker.query_n):
        gate, _reason, topk = ranker.dynamic(extras, query_id, removed_base)
        if gate and target in topk:
            visible.append(query_id)
    return visible


def query_ranked_candidates(ranker: PrefixRanker, candidates: Iterable[int]) -> list[int]:
    """Distance-ordered union of each query's exact nearest 64 candidates."""
    available = sorted(set(candidates))
    retained: dict[int, tuple[int, int]] = {}
    for query_id in range(ranker.query_n):
        ranked = sorted((ranker._exact_distance(ident, query_id), ident) for ident in available)
        for distance, ident in ranked[:MAX_QUERY_RANKED_PER_CLASS]:
            key = (distance, ident)
            previous = retained.get(ident)
            if previous is None or key < previous:
                retained[ident] = key
    return [ident for _key, ident in sorted((key, ident) for ident, key in retained.items())]


def bounded_direct_ids(ids: Iterable[int], rank: dict[int, int]) -> list[int]:
    """Deterministic sub-cap after the per-query nearest-candidate filter."""
    return sorted(set(ids), key=lambda ident: (rank[ident], ident))[:MAX_DIRECT_IDS_PER_LEAF]


def bounded_base_delete_ids(initial_ids: list[int], requested_base_delete: str) -> list[int]:
    base = set(initial_ids)
    if requested_base_delete != "auto":
        try:
            requested = int(requested_base_delete)
        except ValueError as error:
            raise Blocked("invalid_base_delete_id") from error
        require(requested in base, "requested_base_delete_not_in_initial_base")
        return [requested]
    ordered = sorted(base)
    # Keep the known low-ID deletion first but do not assert that it is valid.
    preferred = [0] if 0 in base else []
    return (preferred + [ident for ident in ordered if ident not in preferred])[:MAX_BASE_DELETE_IDS]


def candidate_selection(tree: dict[str, Any], header: dict[str, Any], pool: list[int], queries: list[int],
                        initial_ids: list[int], leaf_capacity: int, requested_base_delete: str,
                        contract: dict[str, Any], search: dict[str, Any]) -> tuple[dict[str, int], bytes]:
    """Boundedly choose a candidate; never serialize CPU certificate labels.

    The selected trace is valid only as a CPU candidate.  Native execution must
    recompute all certificate, capacity, routing-receipt, and oracle evidence.
    """
    require(leaf_capacity == 1, "selector_requires_leaf_capacity_one_for_capacity_coverage")
    dimension = header["dimension"]
    base = set(initial_ids)
    require(len(base) == header["base_n"], "initial_base_duplicate_ids")
    ranker = PrefixRanker(base, pool, queries, dimension, header["k"])
    search.update({
        "strategy": "bounded_prefix_cache_v1",
        "base_query_prefix_length": ranker.prefix_length,
        "limits": {
            "query_ranked_per_class_per_query": MAX_QUERY_RANKED_PER_CLASS,
            "direct_leaf_groups": MAX_DIRECT_LEAF_GROUPS,
            "direct_ids_per_leaf": MAX_DIRECT_IDS_PER_LEAF,
            "base_delete_ids": MAX_BASE_DELETE_IDS,
            "full_tuple_evaluations": MAX_TUPLE_EVALUATIONS,
        },
        "reservoir_ids_scanned": 0,
        "candidate_leaf_groups_examined": 0,
        "qmid_query_checks": 0,
        "full_tuple_evaluations": 0,
        "qmid_B_and_R_exact_topk_found": False,
        "cap_reached": False,
    })

    # This mirrors only the frozen E0 routing metadata to form a bounded
    # candidate pool.  It is deliberately not emitted in the candidate JSON.
    by_leaf: dict[int, list[int]] = defaultdict(list)
    rejects: list[int] = []
    for ident in range(header["base_n"], header["pool_n"]):
        search["reservoir_ids_scanned"] += 1
        record = certify(tree, pool, dimension, ident)
        if record["leaf"] >= 0:
            by_leaf[record["leaf"]].append(ident)
        else:
            rejects.append(ident)

    base_static_ok, base_static_reason = ranker.static_epoch()
    require(base_static_ok, f"E0_static_gate_failed:{base_static_reason}")
    if not rejects:
        raise Blocked("no_natural_certificate_reject_in_immutable_reservoir")

    # Do not enumerate all certified/rejected IDs.  For each immutable query,
    # retain at most its nearest 64 direct and nearest 64 reject candidates;
    # then use only the union of those deterministic sets below.
    ranked_direct_ids = query_ranked_candidates(ranker, (ident for ids in by_leaf.values() for ident in ids))
    ranked_direct = set(ranked_direct_ids)
    ranked_direct_rank = {ident: index for index, ident in enumerate(ranked_direct_ids)}
    ranked_reject = query_ranked_candidates(ranker, rejects)
    search["query_ranked_direct_union"] = len(ranked_direct)
    search["query_ranked_reject_union"] = len(ranked_reject)
    if not ranked_reject:
        raise Blocked("bounded_query_ranked_reject_pool_empty")

    leaf_groups: list[tuple[int, list[int]]] = []
    for leaf in sorted(by_leaf):
        bounded = bounded_direct_ids(
            (ident for ident in by_leaf[leaf] if ident in ranked_direct), ranked_direct_rank
        )
        if len(bounded) >= 3:
            leaf_groups.append((leaf, bounded))
        if len(leaf_groups) >= MAX_DIRECT_LEAF_GROUPS:
            break
    if not leaf_groups:
        raise Blocked("bounded_query_ranked_direct_pool_has_no_same_leaf_triple")
    reject_ids = ranked_reject
    base_delete_candidates = bounded_base_delete_ids(initial_ids, requested_base_delete)
    require(base_delete_candidates, "bounded_base_delete_pool_empty")

    # Deterministic bounded order: certified leaf, A, B, R, C, base-delete,
    # then query ID.  qmid is checked before any C/deletion expansion and must
    # include *both* B and R in the exact top-k; no visibility surrogate is
    # accepted.
    for _leaf, direct_ids in leaf_groups:
        search["candidate_leaf_groups_examined"] += 1
        for a in direct_ids:
            qa_options = exact_visible_queries(ranker, (a,), a)
            if not qa_options:
                continue
            for b in direct_ids:
                if b == a:
                    continue
                for r in reject_ids:
                    qmid_options: list[int] = []
                    for query_id in range(ranker.query_n):
                        search["qmid_query_checks"] += 1
                        gate, _reason, topk = ranker.dynamic((a, b, r), query_id)
                        if gate and b in topk and r in topk:
                            qmid_options.append(query_id)
                    if not qmid_options:
                        continue
                    search["qmid_B_and_R_exact_topk_found"] = True
                    for c in direct_ids:
                        if c == a or c == b:
                            continue
                        qc_options = exact_visible_queries(ranker, (r, c), c)
                        if not qc_options:
                            continue
                        for base_delete in base_delete_candidates:
                            search["full_tuple_evaluations"] += 1
                            if search["full_tuple_evaluations"] > MAX_TUPLE_EVALUATIONS:
                                search["cap_reached"] = True
                                raise Blocked(f"bounded_selector_tuple_cap_reached:{MAX_TUPLE_EVALUATIONS};no_trace_written")
                            e1_static_ok, _e1_static_reason = ranker.static_epoch((r, c), base_delete)
                            if not e1_static_ok:
                                continue
                            qpost_options: list[int] = []
                            for query_id in qc_options:
                                gate, _reason, topk = ranker.dynamic((r, c), query_id, base_delete)
                                if gate and c in topk:
                                    qpost_options.append(query_id)
                            if not qpost_options:
                                continue
                            slots = {
                                "A": a, "B": b, "R": r, "C": c,
                                "query_A": qa_options[0], "query_mid": qmid_options[0],
                                "query_C": qc_options[0], "base_delete": base_delete,
                                "query_post": qpost_options[0],
                            }
                            events = [
                                (OP_INSERT, a), (OP_KNN, slots["query_A"]), (OP_INSERT, b), (OP_INSERT, r),
                                (OP_KNN, slots["query_mid"]), (OP_DELETE, a), (OP_DELETE, b), (OP_INSERT, c),
                                (OP_KNN, slots["query_C"]), (OP_DELETE, base_delete), (OP_REBUILD, 0),
                                (OP_KNN, slots["query_post"]),
                            ]
                            assert_exact_event_contract(events, slots, contract)
                            return slots, encode_trace(header, events)

    if not search["qmid_B_and_R_exact_topk_found"]:
        raise Blocked("bounded_candidate_space_no_query_mid_with_B_and_R_in_exact_topk")
    raise Blocked("bounded_candidate_space_no_tuple_passed_A_C_visibility_qmid_B_R_exact_topk_and_E1_static_gates")


def encode_trace(header: dict[str, Any], events: list[tuple[int, int]]) -> bytes:
    packed = struct.pack(
        "<8sIIIIIIIfQ",
        TRACE_MAGIC,
        TRACE_VERSION,
        header["dimension"],
        header["base_n"],
        header["reservoir_n"],
        header["pool_n"],
        header["query_n"],
        header["k"],
        float(header["radius"]),
        len(events),
    )
    for index, (operation, argument) in enumerate(events):
        packed += struct.pack("<IB3xi", index, operation, argument)
    require(len(packed) == 48 + 12 * 12, "encoded_trace_wrong_abi_size")
    return packed


def atomic_write_bytes(path: Path, content: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_text(path: Path, value: dict[str, Any]) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def canonical_output_bundle(path: Path) -> Path:
    parent = (ROOT / "bundles").resolve(strict=True)
    target = path.expanduser().resolve(strict=False)
    require(target.parent == parent and target.name and not target.name.startswith("."), "out_bundle_must_be_direct_child_of_root_bundles")
    require(not target.exists(), "out_bundle_already_exists")
    return target


def candidate_contract(contract: dict[str, Any], slots: dict[str, int], trace_sha256: str) -> dict[str, Any]:
    header = contract["trace_abi"]["header"]
    required_files = contract["bundle_layout"]["required_files"]
    immutable_payload_sha256 = {name: detail["sha256"] for name, detail in required_files.items()}
    source_hashes = contract["root_source_hashes"]
    selector = Path(__file__).resolve()
    relative = selector.relative_to(ROOT).as_posix()
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "status": CANDIDATE_STATUS,
        "trace": {"filename": TRACE_FILENAME, "sha256": trace_sha256},
        "trace_header": header,
        "leaf_capacity": 1,
        "stable_id_layout": "identity_full_immutable_pool_seeded_stable_ids_v5",
        "immutable_payload_sha256": immutable_payload_sha256,
        "root_source_hashes": source_hashes,
        "archived_core_sha256": source_hashes["src/safe_c1_dynamic_gts.cu"],
        "slot_ids": slots,
        "runtime_requirements": contract["runtime_requirements"],
        "static_certificate_outcomes": "NOT_ESTABLISHED_STATICALLY",
        "candidate_source": {"relative_path": relative, "sha256": sha256_file(selector)},
    }
    # Keep the exact mappings that preflight compares, including field types.
    require(candidate["trace_header"] == contract["trace_abi"]["header"], "candidate_trace_header_mismatch")
    require(candidate["runtime_requirements"] == contract["runtime_requirements"], "candidate_runtime_requirements_mismatch")
    return candidate


def atomic_publish_bundle(target: Path, source_bundle: Path, contract: dict[str, Any], trace: bytes,
                          candidate: dict[str, Any]) -> None:
    parent = target.parent
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(parent)))
    try:
        for filename, detail in contract["bundle_layout"]["required_files"].items():
            source = source_bundle / filename
            destination = stage / filename
            shutil.copyfile(source, destination)
            require(destination.stat().st_size == detail["bytes"], f"published_bundle_size_mismatch:{filename}")
            require(sha256_file(destination) == detail["sha256"], f"published_bundle_hash_mismatch:{filename}")
        trace_path = stage / TRACE_FILENAME
        atomic_write_bytes(trace_path, trace)
        trace_sha = sha256_file(trace_path)
        require(candidate["trace"] == {"filename": TRACE_FILENAME, "sha256": trace_sha}, "published_candidate_trace_binding_mismatch")
        atomic_write_text(stage / CANDIDATE_FILENAME, candidate)
        # Verify all writes before one directory rename makes them visible.
        require(sha256_file(trace_path) == candidate["trace"]["sha256"], "published_trace_sha_mismatch")
        parsed = json.loads((stage / CANDIDATE_FILENAME).read_text(encoding="utf-8"))
        require(parsed == candidate, "published_candidate_json_mismatch")
        os.replace(stage, target)
        stage = None  # type: ignore[assignment]
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        atomic_write_text(temporary, report)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="source immutable bundle; read-only")
    parser.add_argument("--header-trace", required=True, type=Path)
    parser.add_argument("--native-snapshot", required=True, type=Path)
    parser.add_argument("--out-bundle", required=True, type=Path, help="new direct child of ROOT/bundles")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=ROOT / "src" / "safe_c1_dynamic_gts.cu")
    parser.add_argument("--exporter-source", type=Path, default=ROOT / "src" / "fable5_native_snapshot_exporter_v1.cu")
    parser.add_argument("--matched-runner-source", type=Path, default=ROOT / "src" / "fable5_matched_gpu_runner_v1.cu")
    parser.add_argument("--leaf-capacity", type=int, default=1)
    parser.add_argument("--base-delete-id", default="auto", help="initial-base ID or auto (first ascending E1-static-gated ID)")
    args = parser.parse_args()

    selection_search: dict[str, Any] = {
        "strategy": "not_started",
        "cap_reached": False,
    }
    report: dict[str, Any] = {
        "schema": "fable5-native-snapshot-selector-v1",
        "selector_search": selection_search,
        "execution": {"cpu_only": True, "cuda_binary_executed": False, "gpu_used": False, "nvml_used": False, "nvidia_smi_called": False},
        "scope": "CPU candidate construction only; no native runtime certificate, traversal receipt, timing, performance, C2, C3, or deployment claim",
        "static_certificate_outcomes": "NOT_ESTABLISHED_STATICALLY",
        "required_runtime_followup": [
            "recompute every native sibling-boundary certificate in fable5_matched_gpu_runner_v1",
            "observe native_certificate_leaf=-1 for R",
            "observe capacity fallback for B and direct reuse for C",
            "verify native traversal receipt visits the selected direct leaf for A/C query visibility",
            "validate full-active-set oracle under both matched policies",
        ],
    }
    try:
        require(args.bundle.is_dir(), "missing_source_bundle")
        target = canonical_output_bundle(args.out_bundle)
        header = read_header(args.header_trace)
        contract = load_preflight_contract(args.bundle, header)
        report["header_trace_sha256"] = sha256_file(args.header_trace)
        report["certificate_source_sha256"] = sha256_file(args.source)
        report["certificate_source_semantics"] = source_semantics(args.source)
        snapshot, observed_hashes = parse_snapshot(
            args.native_snapshot, args.bundle, args.header_trace, args.source, args.exporter_source, args.matched_runner_source
        )
        initial = read_i32(args.bundle / "initial_base_stable_ids.i32", header["base_n"])
        mapping = read_i32(args.bundle / "stable_id_to_pool_row.i32", header["pool_n"])
        require(initial == list(range(header["base_n"])), "initial_base_not_identity_prefix")
        require(mapping == list(range(header["pool_n"])), "stable_map_not_identity")
        tree = normalize_snapshot(snapshot, header, initial)
        pool = read_i16(args.bundle / "pool.i16", header["pool_n"] * header["dimension"])
        queries = read_i16(args.bundle / "queries.i16", header["query_n"] * header["dimension"])
        native_hashes = snapshot.get("native_hashes")
        require(isinstance(native_hashes, dict), "snapshot_missing_native_hashes_after_parse")
        immutable_pool_hash = encoded_pool_fnv1a(pool)
        require(native_hashes.get("immutable_pool_fnv1a64") == immutable_pool_hash, "immutable_pool_fnv1a64_mismatch")
        slots, trace = candidate_selection(
            tree, header, pool, queries, initial, args.leaf_capacity, args.base_delete_id, contract, selection_search
        )
        trace_sha = hashlib.sha256(trace).hexdigest()
        candidate = candidate_contract(contract, slots, trace_sha)
        atomic_publish_bundle(target, args.bundle, contract, trace, candidate)
        report.update({
            "status": CANDIDATE_STATUS,
            "native_snapshot": str(args.native_snapshot),
            "native_snapshot_canonical_live_hash": tree["canonical_live_hash"],
            "input_hashes_verified": observed_hashes,
            "output_bundle": str(target),
            "trace": {"path": str(target / TRACE_FILENAME), "sha256": trace_sha, "event_count": 12},
            "candidate_contract": {"path": str(target / CANDIDATE_FILENAME), "sha256": sha256_file(target / CANDIDATE_FILENAME)},
            "slot_ids": slots,
            "residual_pruning": snapshot["residual_pruning"],
            "c2_used": snapshot["c2_used"],
            "c3_used": snapshot["c3_used"],
        })
        write_report(args.report, report)
        print(CANDIDATE_STATUS, args.report)
        return 0
    except Blocked as error:
        report.update({
            "status": "BLOCKED_NO_TRACE_WRITTEN",
            "blocker": str(error),
            "native_snapshot": str(args.native_snapshot),
        })
        write_report(args.report, report)
        print("BLOCKED_NO_TRACE_WRITTEN", error, args.report, file=sys.stderr)
        return 2



if __name__ == "__main__":
    raise SystemExit(main())
