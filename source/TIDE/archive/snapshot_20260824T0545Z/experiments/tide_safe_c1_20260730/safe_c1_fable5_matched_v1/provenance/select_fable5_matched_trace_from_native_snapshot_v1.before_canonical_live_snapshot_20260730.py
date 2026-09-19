#!/usr/bin/env python3
"""CPU-only selector for a *native-exported* Fable5 frozen E0 snapshot.

It never constructs a GTS tree, invokes a CUDA binary, probes a device, or
assigns fixture certificate labels.  It only accepts a snapshot emitted by the
separate native exporter, verifies its byte pins and frozen-tree hash, mirrors
the source's strict sibling-boundary certificate against that frozen metadata,
and writes a 12-op trace only when every required condition is satisfied.

A successful output is still ``PENDING_NATIVE_RUNTIME_RECEIPT``: the matched
GPU runner must recompute every certificate and prove receipt-based query
visibility.  A missing snapshot or absent natural certificate reject produces
only a blocking report, never an invented trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
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
    require(value.get("fanout") == TREE_ORDER, "snapshot_wrong_fanout")
    require(isinstance(value.get("tree_height"), int) and value["tree_height"] > 1, "snapshot_invalid_tree_height")
    require(value.get("tn_abi") == "little-endian:i32,f32bits,i32,i32,i32", "snapshot_unknown_tn_abi")
    require(value.get("tn_size_bytes") == 20, "snapshot_unknown_tn_size")

    raw_nodes = value.get("nodes")
    raw_empty = value.get("empty")
    raw_max = value.get("max_distance_f32_bits")
    require(isinstance(raw_nodes, list) and isinstance(raw_empty, list) and isinstance(raw_max, list), "snapshot_arrays_missing")
    require(len(raw_nodes) > 0 and len(raw_nodes) == len(raw_empty) == len(raw_max), "snapshot_array_length_mismatch")

    nodes: list[dict[str, Any]] = []
    for node in raw_nodes:
        require(isinstance(node, dict), "invalid_snapshot_node")
        pid = node.get("pid")
        size = node.get("size")
        lid = node.get("lid")
        leaf = node.get("is_leaf")
        minimum = float_from_bits(node.get("min_dis_f32_bits"))
        require(all(isinstance(item, int) for item in (pid, size, lid, leaf)), "noninteger_snapshot_node_field")
        require(leaf in (0, 1) and size >= 0 and lid >= 0, "invalid_snapshot_node_shape")
        nodes.append({"pid": pid, "min": minimum, "min_bits": node["min_dis_f32_bits"], "size": size, "lid": lid, "leaf": leaf})
    empty: list[int] = []
    for flag in raw_empty:
        require(isinstance(flag, int) and flag in (0, 1), "invalid_snapshot_empty_flag")
        empty.append(flag)
    max_bits: list[int] = []
    for bits in raw_max:
        max_bits.append(int(bits))
        _ = float_from_bits(int(bits))

    # Verify byte-exact FrozenTreeSnapshot::compute_hash fields that are now
    # exportable.  This makes the selector reject a hand-written interval JSON.
    native_hashes = value.get("native_hashes")
    require(isinstance(native_hashes, dict), "snapshot_missing_native_hashes")
    node_raw = b"".join(
        struct.pack("<iIiii", node["pid"], node["min_bits"], node["size"], node["lid"], node["leaf"])
        for node in nodes
    )
    empty_raw = struct.pack(f"<{len(empty)}i", *empty)
    max_raw = struct.pack(f"<{len(max_bits)}I", *max_bits)
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
        require(empty[node_id] == 0 and node["leaf"] == 1, "payload_not_for_live_leaf")
        require(item.get("lid") == node["lid"] and item.get("size") == node["size"], "payload_node_metadata_mismatch")
        require(len(ids) == node["size"] and all(isinstance(ident, int) for ident in ids), "payload_id_shape_mismatch")
        logical_layout.extend([node_id, node["size"], *ids])
        flattened.extend(ids)
    expected_leaf_nodes = {index for index, node in enumerate(nodes) if empty[index] == 0 and node["leaf"] == 1}
    require(seen_leaf_nodes == expected_leaf_nodes, "logical_leaf_payload_incomplete")
    require(sorted(flattened) == initial_ids, "snapshot_leaf_ids_do_not_equal_initial_base")
    logical_raw = struct.pack(f"<{len(logical_layout)}i", *logical_layout) if logical_layout else b""
    logical_hash = fnv1a(logical_raw)
    logical_count = len(flattened)
    require(native_hashes.get("logical_leaf_id_fnv1a64") == logical_hash, "logical_leaf_hash_mismatch")
    require(native_hashes.get("logical_leaf_id_count") == logical_count, "logical_leaf_count_mismatch")
    frozen = fnv1a(struct.pack("<i", value["tree_height"]))
    frozen = fnv1a(struct.pack("<i", value["fanout"]), frozen)
    frozen = fnv1a(node_raw, frozen)
    frozen = fnv1a(empty_raw, frozen)
    frozen = fnv1a(max_raw, frozen)
    frozen = fnv1a(struct.pack("<i", logical_count), frozen)
    frozen = fnv1a(struct.pack("<Q", logical_hash), frozen)
    require(native_hashes.get("frozen_tree_fnv1a64") == frozen, "frozen_tree_hash_mismatch")
    return {"nodes": nodes, "empty": empty, "height": value["tree_height"], "fanout": value["fanout"], "frozen_hash": frozen}


def valid_nonempty(tree: dict[str, Any], node_id: int) -> bool:
    return 0 <= node_id < len(tree["nodes"]) and tree["empty"][node_id] == 0


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
    ordered = sorted_exact(active, pool, queries, dimension, query_id)
    if len(ordered) < k:
        return False, "dynamic_below_k"
    if len(ordered) > k and ordered[k - 1][0] == ordered[k][0]:
        return False, f"dynamic_boundary_tie:q{query_id}"
    return True, None


def visibility_queries(active: set[int], target: int, pool: list[int], queries: list[int], dimension: int, k: int) -> list[int]:
    valid: list[int] = []
    for query_id in range(len(queries) // dimension):
        gate, _ = dynamic_gate(active, pool, queries, dimension, k, query_id)
        if gate and target in {ident for _, ident in sorted_exact(active, pool, queries, dimension, query_id)[:k]}:
            valid.append(query_id)
    return valid


def candidate_selection(tree: dict[str, Any], header: dict[str, Any], pool: list[int], queries: list[int],
                        initial_ids: list[int], leaf_capacity: int, requested_base_delete: str) -> tuple[dict[str, Any], bytes]:
    require(leaf_capacity == 1, "selector_requires_leaf_capacity_one_for_capacity_coverage")
    dimension = header["dimension"]
    base = set(initial_ids)
    results: dict[int, dict[str, Any]] = {}
    by_leaf: dict[int, list[int]] = defaultdict(list)
    rejects: list[int] = []
    for ident in range(header["base_n"], header["pool_n"]):
        record = certify(tree, pool, dimension, ident)
        results[ident] = record
        if record["leaf"] >= 0:
            by_leaf[record["leaf"]].append(ident)
        else:
            rejects.append(ident)

    base_static_ok, base_static_reason = static_epoch_gate(base, pool, queries, dimension, header["k"])
    require(base_static_ok, f"E0_static_gate_failed:{base_static_reason}")
    group_summary = [{"leaf": leaf, "candidate_count": len(ids), "ids": ids[:12]} for leaf, ids in sorted(by_leaf.items())]
    if not rejects:
        raise Blocked("no_natural_certificate_reject_in_immutable_reservoir")
    if not any(len(ids) >= 3 for ids in by_leaf.values()):
        raise Blocked("no_same_native_cert_leaf_group_of_three")

    base_delete_candidates: list[int]
    if requested_base_delete == "auto":
        base_delete_candidates = list(initial_ids)
    else:
        try:
            requested = int(requested_base_delete)
        except ValueError as error:
            raise Blocked("invalid_base_delete_id") from error
        require(requested in base, "requested_base_delete_not_in_initial_base")
        base_delete_candidates = [requested]

    # Deterministic nested order: leaf, A, B, C, R, base-delete, query IDs.
    # A/B/C all pass the same actual frozen certificate; B becomes capacity
    # delta because A owns the sole sidecar slot; R is a genuine -1 result.
    for leaf in sorted(by_leaf):
        direct_ids = by_leaf[leaf]
        if len(direct_ids) < 3:
            continue
        for a_index, a in enumerate(direct_ids):
            active_after_a = base | {a}
            qa_options = visibility_queries(active_after_a, a, pool, queries, dimension, header["k"])
            if not qa_options:
                continue
            for b_index, b in enumerate(direct_ids):
                if b_index == a_index:
                    continue
                for c_index, c in enumerate(direct_ids):
                    if c_index in (a_index, b_index):
                        continue
                    # Reuse state after deleting A (direct) and B (capacity delta): C must be visible.
                    for r in rejects:
                        active_mid = base | {a, b, r}
                        qmid_options = [
                            query_id for query_id in range(header["query_n"])
                            if dynamic_gate(active_mid, pool, queries, dimension, header["k"], query_id)[0]
                        ]
                        if not qmid_options:
                            continue
                        active_after_reuse = base | {r, c}
                        qc_options = visibility_queries(active_after_reuse, c, pool, queries, dimension, header["k"])
                        if not qc_options:
                            continue
                        for base_delete in base_delete_candidates:
                            e1 = (base - {base_delete}) | {r, c}
                            e1_static_ok, _ = static_epoch_gate(e1, pool, queries, dimension, header["k"])
                            if not e1_static_ok:
                                continue
                            # Prefer C-observable query after rebuild as well; it makes the
                            # post-reuse/post-rebuild linkage directly inspectable.
                            qpost_options = [
                                query_id for query_id in qc_options
                                if dynamic_gate(e1, pool, queries, dimension, header["k"], query_id)[0]
                                and c in {ident for _, ident in sorted_exact(e1, pool, queries, dimension, query_id)[:header["k"]]}
                            ]
                            if not qpost_options:
                                continue
                            qa = qa_options[0]
                            qmid = qmid_options[0]
                            qc = qc_options[0]
                            qpost = qpost_options[0]
                            events = [
                                (OP_INSERT, a), (OP_KNN, qa), (OP_INSERT, b), (OP_INSERT, r),
                                (OP_KNN, qmid), (OP_DELETE, a), (OP_DELETE, b), (OP_INSERT, c),
                                (OP_KNN, qc), (OP_DELETE, base_delete), (OP_REBUILD, 0), (OP_KNN, qpost),
                            ]
                            raw = encode_trace(header, events)
                            selection = {
                                "direct_leaf": leaf,
                                "A_direct": a,
                                "B_capacity_delta": b,
                                "C_reuse_direct": c,
                                "R_certificate_reject": r,
                                "base_delete": base_delete,
                                "queries": {"after_A": qa, "middle": qmid, "after_C_reuse": qc, "post_rebuild": qpost},
                                "certificate_witness": {str(ident): results[ident] for ident in (a, b, c, r)},
                                "observability": {
                                    "A_exact_topk_before_rebuild": [ident for _, ident in sorted_exact(active_after_a, pool, queries, dimension, qa)[:header["k"]]],
                                    "C_exact_topk_after_reuse": [ident for _, ident in sorted_exact(active_after_reuse, pool, queries, dimension, qc)[:header["k"]]],
                                    "C_exact_topk_post_rebuild": [ident for _, ident in sorted_exact(e1, pool, queries, dimension, qpost)[:header["k"]]],
                                    "runtime_still_required": "matched runner must prove native base traversal receipt visits direct_leaf for A/C query visibility",
                                },
                                "E1_active_formula": "initial_base - {base_delete} + {C_reuse_direct,R_certificate_reject}",
                                "E1_active_count": len(e1),
                                "candidate_scan": {"reservoir_count": header["reservoir_n"], "direct_count": sum(len(ids) for ids in by_leaf.values()), "certificate_reject_count": len(rejects), "groups": group_summary},
                            }
                            return selection, raw
    raise Blocked("no_candidate_tuple_passed_observability_dynamic_and_E1_static_gates")


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
    return packed


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--header-trace", required=True, type=Path)
    parser.add_argument("--native-snapshot", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=ROOT / "src" / "safe_c1_dynamic_gts.cu")
    parser.add_argument("--exporter-source", type=Path, default=ROOT / "src" / "fable5_native_snapshot_exporter_v1.cu")
    parser.add_argument("--matched-runner-source", type=Path, default=ROOT / "src" / "fable5_matched_gpu_runner_v1.cu")
    parser.add_argument("--leaf-capacity", type=int, default=1)
    parser.add_argument("--base-delete-id", default="auto", help="initial-base ID or auto (first ascending that passes E1 gates)")
    args = parser.parse_args()

    report_path = args.out_dir / "selection_report.json"
    report: dict[str, Any] = {
        "schema": "fable5-native-snapshot-selector-v1",
        "execution": {"cpu_only": True, "cuda_binary_executed": False, "gpu_used": False, "nvml_used": False, "nvidia_smi_called": False},
        "scope": "native-snapshot-backed candidate selection only; no runtime receipt, timing, performance, C2, C3, or deployment claim",
        "required_runtime_followup": [
            "recompute every native sibling-boundary certificate in fable5_matched_gpu_runner_v1",
            "observe native_certificate_leaf=-1 for R",
            "observe capacity fallback for B and direct reuse for C",
            "verify native traversal receipt visits the selected direct leaf for A/C observability queries",
            "validate full-active-set oracle under both matched policies",
        ],
    }
    try:
        require(args.bundle.is_dir(), "missing_bundle")
        header = read_header(args.header_trace)
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
        selection, trace = candidate_selection(
            tree, header, pool, queries, initial, args.leaf_capacity, args.base_delete_id
        )
        trace_path = args.out_dir / "trace.fable5.e1gtrc"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        trace_path.write_bytes(trace)
        report.update({
            "status": "READY_FOR_NATIVE_RUNTIME_RECEIPT",
            "native_snapshot": str(args.native_snapshot),
            "native_snapshot_frozen_hash": tree["frozen_hash"],
            "input_hashes_verified": observed_hashes,
            "selection": selection,
            "trace": {"path": str(trace_path), "sha256": sha256_file(trace_path), "event_count": 12},
            "residual_pruning": snapshot["residual_pruning"],
        })
        write_report(report_path, report)
        print("READY_FOR_NATIVE_RUNTIME_RECEIPT", report_path)
        return 0
    except Blocked as error:
        report.update({
            "status": "BLOCKED_NO_TRACE_WRITTEN",
            "blocker": str(error),
            "native_snapshot": str(args.native_snapshot),
        })
        write_report(report_path, report)
        print("BLOCKED_NO_TRACE_WRITTEN", error, report_path, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
