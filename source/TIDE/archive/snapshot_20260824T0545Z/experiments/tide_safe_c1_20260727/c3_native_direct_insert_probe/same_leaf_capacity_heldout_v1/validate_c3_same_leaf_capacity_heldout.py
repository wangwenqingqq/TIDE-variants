#!/workspace/legacy_workspace/GTS/bench_env/bin/python
"""Independent CPU validator for C3 same-leaf capacity + held-out KNN probe.

No CUDA binary, GPU command, or archived updater is invoked.  The validator
recomputes all routing/capacity and exact int64 active-set answers from frozen
inputs and the runner's exported real-GTS geometry.  It also requires the
runner's initial geometry to equal the SHA-pinned selection witness exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"E1GTRC01"
HEADER = struct.Struct("<8sI6IfQ")
SCHEMA = "c3-same-leaf-capacity-heldout-cpu-validator-v1"
EPSILON = 1.0e-4
EXPECTED_TARGET = 221
EXPECTED_INITIAL_IDS = [3546, 158, 1747, 1659]
EXPECTED_ACCEPTED = [4462, 4507, 4577, 4593, 4603, 4671, 4694, 4698,
                     4728, 4729, 4740, 4741, 4793, 4826, 4942, 5085]
EXPECTED_BOUNDARY = 5430
EXPECTED_QUERY_IDS = [0, 1, 2]
EXPECTED_KS = [1, 10, 20]
EXPECTED_FINGERPRINT = "fnv1a64:a4a1881cd82237fb"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(errors: list[dict[str, Any]], kind: str, **detail: Any) -> None:
    errors.append({"type": kind, **detail})


def f32_from_bits(bits: int) -> np.float32:
    return np.frombuffer(struct.pack("<I", int(bits)), dtype="<f4")[0]


def f32_l2_pool(pool: np.ndarray, left: int, right: int) -> float:
    total = np.float32(0.0)
    for a, b in zip(pool[left], pool[right]):
        delta = np.float32(np.float32(a) - np.float32(b))
        total = np.float32(total + np.float32(delta * delta))
    return float(np.sqrt(total, dtype=np.float32))


def f32_ulp(value: float) -> float:
    x = np.float32(value)
    return float(np.nextafter(x, np.float32(np.inf), dtype=np.float32) - x)


def close_static_distance(observed: float, squared: int) -> bool:
    reference = float(np.float32(math.sqrt(int(squared))))
    magnitude = max(1.0, abs(float(observed)), abs(reference))
    return abs(float(observed) - reference) <= max(0.002, 2.0 * f32_ulp(magnitude))


def fnv1a64_geometry(geometry: dict[str, Any]) -> str:
    h = 0xCBF29CE484222325
    prime = 0x100000001B3

    def mix(value: int) -> None:
        nonlocal h
        for byte in (int(value) & ((1 << 64) - 1)).to_bytes(8, "little", signed=False):
            h ^= byte
            h = (h * prime) & ((1 << 64) - 1)

    nodes = geometry["nodes"]
    for value in (geometry["tree_height"], geometry["tree_order"], geometry["max_size"],
                  geometry["leaf_pad_slots"], len(nodes)):
        mix(value)
    for node in nodes:
        for key in ("node_id", "empty", "pid", "min_dis_bits", "max_dis_bits", "size", "lid", "is_leaf"):
            mix(node[key])
        ids = node.get("logical_ids", [])
        mix(len(ids))
        for stable_id in ids:
            mix(stable_id)
    return f"fnv1a64:{h:016x}"


def load_bundle(bundle: Path, errors: list[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "e1gi-b-quantized-gts-integration-manifest-v1":
        fail(errors, "bundle_manifest_schema", observed=manifest.get("schema"))
    for name, expected in manifest.get("files_sha256", {}).items():
        path = bundle / name
        observed = sha256_file(path) if path.is_file() else None
        if observed != expected:
            fail(errors, "frozen_input_hash_mismatch", file=name, expected=expected, observed=observed)
    blob = (bundle / "trace.e1gtrc").read_bytes()
    if len(blob) < HEADER.size:
        raise ValueError("truncated trace header")
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = HEADER.unpack_from(blob, 0)
    header = {"dim": int(dim), "base_n": int(base_n), "reservoir_n": int(reservoir_n),
              "pool_n": int(pool_n), "query_n": int(query_n), "k": int(k),
              "radius": float(radius), "event_count": int(event_count)}
    if magic != MAGIC or version != 1 or reservoir_n != pool_n - base_n:
        fail(errors, "trace_header_contract", header=header)
    pool = np.fromfile(bundle / "pool.i16", dtype="<i2")
    queries = np.fromfile(bundle / "queries.i16", dtype="<i2")
    if pool.size != pool_n * dim or queries.size != query_n * dim:
        raise ValueError("frozen payload shape mismatch")
    return header, pool.reshape(pool_n, dim), queries.reshape(query_n, dim), manifest


def load_snapshot(path: Path, errors: list[dict[str, Any]], label: str) -> dict[str, Any]:
    snap = json.loads(path.read_text())
    if snap.get("schema") != "c3-native-gts-geometry-snapshot-v1":
        fail(errors, "geometry_schema", label=label, observed=snap.get("schema"))
    if (snap.get("tree_order"), snap.get("max_size"), snap.get("leaf_pad_slots")) != (10, 20, 64):
        fail(errors, "geometry_constants", label=label, observed=[snap.get("tree_order"), snap.get("max_size"), snap.get("leaf_pad_slots")])
    nodes = snap.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"{label}: empty geometry")
    for expected, node in enumerate(nodes):
        if not isinstance(node, dict) or node.get("node_id") != expected:
            fail(errors, "geometry_node_id", label=label, expected=expected, observed=node.get("node_id") if isinstance(node, dict) else None)
            continue
        for key in ("empty", "pid", "min_dis_bits", "max_dis_bits", "size", "lid", "is_leaf"):
            if not isinstance(node.get(key), int):
                fail(errors, "geometry_node_field", label=label, node_id=expected, field=key)
        if node.get("empty") == 0 and node.get("is_leaf") == 1:
            ids = node.get("logical_ids")
            if not isinstance(ids, list) or len(ids) != node.get("size") or any(not isinstance(x, int) for x in ids):
                fail(errors, "geometry_leaf_ids", label=label, node_id=expected)
    return snap


def leaf_ids(node: dict[str, Any]) -> list[int]:
    return list(node.get("logical_ids", []))


def search_upper_invariant(snap: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    nodes = snap["nodes"]
    order = int(snap["tree_order"])
    issues: list[dict[str, Any]] = []
    for parent, parent_node in enumerate(nodes):
        if parent_node.get("empty") != 0:
            continue
        for slot in range(order):
            child_id = parent * order + slot + 1
            if child_id < 0 or child_id >= len(nodes) or nodes[child_id].get("empty") != 0:
                continue
            child = nodes[child_id]
            lower = float(f32_from_bits(child["min_dis_bits"]))
            upper = float(f32_from_bits(child["max_dis_bits"]))
            if not lower <= upper:
                issues.append({"type": "min_exceeds_max", "child": child_id})
            if child_id % order == 0:
                continue
            nxt = child_id + 1
            if nxt >= len(nodes) or nodes[nxt].get("empty") != 0 or (nxt - 1) // order != parent:
                issues.append({"type": "missing_next_sibling", "child": child_id, "next": nxt})
                continue
            next_min = float(f32_from_bits(nodes[nxt]["min_dis_bits"]))
            if upper > next_min:
                issues.append({"type": "max_exceeds_next_min", "child": child_id, "next": nxt, "upper": upper, "next_min": next_min})
    return not issues, issues


def certificate(initial: dict[str, Any], pool: np.ndarray, stable_id: int,
                appended: dict[int, list[int]], search_upper_ok: bool) -> dict[str, Any]:
    nodes = initial["nodes"]
    order = int(initial["tree_order"])
    height = int(initial["tree_height"])
    if not search_upper_ok:
        return {"ok": False, "reason": "search_upper_invariant", "path": []}
    parent = 0
    path: list[dict[str, Any]] = []
    for _ in range(height - 1):
        child_ids = [parent * order + slot + 1 for slot in range(order)]
        children = [cid for cid in child_ids if 0 <= cid < len(nodes) and nodes[cid].get("empty") == 0]
        if not children:
            return {"ok": False, "reason": "missing_or_invalid_path", "path": path}
        pivot = nodes[children[0]].get("pid")
        if not isinstance(pivot, int) or pivot < 0 or pivot >= pool.shape[0] or any(nodes[c].get("pid") != pivot for c in children):
            return {"ok": False, "reason": "missing_or_invalid_path", "path": path}
        distance = f32_l2_pool(pool, stable_id, pivot)
        matches: list[tuple[int, float, float, float, bool]] = []
        for cid in children:
            child = nodes[cid]
            lower = float(f32_from_bits(child["min_dis_bits"]))
            upper = float(f32_from_bits(child["max_dis_bits"]))
            has_next = cid % order != 0
            search_upper = upper
            if has_next:
                nxt = cid + 1
                if nxt >= len(nodes) or nodes[nxt].get("empty") != 0 or (nxt - 1) // order != parent:
                    return {"ok": False, "reason": "search_upper_invariant", "path": path}
                search_upper = float(f32_from_bits(nodes[nxt]["min_dis_bits"]))
            if distance > lower + EPSILON and distance < upper - EPSILON and (not has_next or distance < search_upper - EPSILON):
                matches.append((cid, lower, upper, search_upper, has_next))
        chosen = matches[0] if len(matches) == 1 else None
        path.append({"parent": parent, "pivot": pivot, "child": chosen[0] if chosen else -1,
                     "matching_children": len(matches), "distance": distance,
                     "lower": chosen[1] if chosen else None, "max_upper": chosen[2] if chosen else None,
                     "search_upper": chosen[3] if chosen else None,
                     "has_next_sibling_upper": chosen[4] if chosen else None})
        if chosen is None:
            return {"ok": False, "reason": "no_unique_strict_child", "path": path}
        if nodes[chosen[0]].get("is_leaf") == 1:
            leaf = chosen[0]
            pre = int(nodes[leaf]["size"]) + len(appended.get(leaf, []))
            if pre >= int(initial["max_size"]):
                return {"ok": False, "reason": "static_scan_limit", "leaf": leaf, "pre_leaf_size": pre, "path": path}
            if len(appended.get(leaf, [])) >= int(initial["leaf_pad_slots"]):
                return {"ok": False, "reason": "padded_leaf_capacity", "leaf": leaf, "pre_leaf_size": pre, "path": path}
            return {"ok": True, "reason": "accepted", "leaf": leaf, "pre_leaf_size": pre, "path": path}
        parent = chosen[0]
    return {"ok": False, "reason": "not_leaf", "path": path}


def exact_external_topk(pool: np.ndarray, active: set[int], query: np.ndarray, k: int) -> tuple[list[int], list[int], bool]:
    ids = np.asarray(sorted(active), dtype=np.int64)
    diff = pool[ids].astype(np.int64, copy=False) - query.astype(np.int64, copy=False)
    squared = np.sum(diff * diff, axis=1, dtype=np.int64)
    order = np.lexsort((ids, squared))
    boundary = bool(len(order) > k and int(squared[order[k - 1]]) == int(squared[order[k]]))
    top = order[:k]
    return [int(ids[i]) for i in top], [int(squared[i]) for i in top], boundary


def validate_topk_row(row: dict[str, Any], event_index: int, stable_id: int, query_id: int, k: int,
                      active: set[int], pool: np.ndarray, queries: np.ndarray,
                      errors: list[dict[str, Any]], record_index: int) -> None:
    if row.get("record") != "heldout_knn":
        fail(errors, "missing_heldout_record", index=record_index, observed=row.get("record")); return
    expected_meta = {"event_index": event_index, "triggering_stable_id": stable_id, "query_id": query_id, "k": k,
                     "active_count": len(active), "external_nonself_query": True}
    for key, expected in expected_meta.items():
        if row.get(key) != expected:
            fail(errors, "heldout_record_meta", index=record_index, field=key, expected=expected, observed=row.get(key))
    expected_ids, expected_sq, boundary = exact_external_topk(pool, active, queries[query_id], k)
    if boundary:
        fail(errors, "heldout_k_boundary_tie", event_index=event_index, query_id=query_id, k=k)
        return
    if row.get("k_boundary_tie") is not False:
        fail(errors, "runner_boundary_tie_field", event_index=event_index, query_id=query_id, k=k, observed=row.get("k_boundary_tie"))
    got_ids, got_dist = row.get("gts_ids"), row.get("gts_distances")
    if not isinstance(got_ids, list) or not isinstance(got_dist, list) or len(got_ids) != k or len(got_dist) != k:
        fail(errors, "heldout_gts_shape", event_index=event_index, query_id=query_id, k=k); return
    if any(not isinstance(x, int) for x in got_ids) or len(set(got_ids)) != len(got_ids):
        fail(errors, "heldout_gts_ids", event_index=event_index, query_id=query_id, k=k); return
    pos = 0
    while pos < k:
        end = pos + 1
        while end < k and expected_sq[end] == expected_sq[pos]:
            end += 1
        if sorted(got_ids[pos:end]) != sorted(expected_ids[pos:end]):
            fail(errors, "heldout_topk_ids", event_index=event_index, query_id=query_id, k=k,
                 ranks=[pos, end - 1], expected=expected_ids[pos:end], observed=got_ids[pos:end])
        for rank in range(pos, end):
            try:
                observed = float(got_dist[rank])
            except (TypeError, ValueError):
                fail(errors, "heldout_distance_type", event_index=event_index, query_id=query_id, k=k, rank=rank); continue
            if not math.isfinite(observed) or not close_static_distance(observed, expected_sq[rank]):
                fail(errors, "heldout_distance", event_index=event_index, query_id=query_id, k=k, rank=rank,
                     expected_squared=expected_sq[rank], observed=got_dist[rank])
        pos = end


def validate_geometry(initial: dict[str, Any], final: dict[str, Any], appended: dict[int, list[int]], errors: list[dict[str, Any]]) -> None:
    for key in ("tree_height", "tree_order", "max_size", "leaf_pad_slots"):
        if initial.get(key) != final.get(key): fail(errors, "geometry_header_changed", field=key)
    if len(initial["nodes"]) != len(final["nodes"]):
        fail(errors, "geometry_node_count_changed"); return
    for before, after in zip(initial["nodes"], final["nodes"]):
        node_id = before.get("node_id")
        for key in ("node_id", "empty", "pid", "min_dis_bits", "max_dis_bits", "lid", "is_leaf"):
            if before.get(key) != after.get(key): fail(errors, "frozen_geometry_mutation", node_id=node_id, field=key)
        added = appended.get(node_id, [])
        if before.get("empty") != 0 or before.get("is_leaf") != 1:
            if before.get("size") != after.get("size"): fail(errors, "nonleaf_size_mutation", node_id=node_id)
            continue
        if after.get("size") != before.get("size") + len(added):
            fail(errors, "leaf_size_mutation", node_id=node_id, expected=before.get("size") + len(added), observed=after.get("size"))
        if leaf_ids(after) != leaf_ids(before) + added:
            fail(errors, "leaf_logical_ids", node_id=node_id)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--initial-geometry", required=True, type=Path)
    p.add_argument("--final-geometry", required=True, type=Path)
    p.add_argument("--engine-jsonl", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    errors: list[dict[str, Any]] = []
    a.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        protocol = json.loads(a.protocol.read_text())
        if protocol.get("schema") != "c3-same-leaf-capacity-heldout-protocol-v1": fail(errors, "protocol_schema", observed=protocol.get("schema"))
        if protocol.get("status") != "PRE_REGISTERED_AFTER_CPU_SELECTION_AND_COMPILE_ONLY_BEFORE_GPU_EXECUTION": fail(errors, "protocol_status", observed=protocol.get("status"))
        if protocol.get("candidate_trace", {}).get("accepted_stable_ids") != EXPECTED_ACCEPTED: fail(errors, "protocol_accepted_trace")
        if protocol.get("candidate_trace", {}).get("boundary_stable_id") != EXPECTED_BOUNDARY: fail(errors, "protocol_boundary")
        if protocol.get("heldout_nonself_knn", {}).get("query_ids") != EXPECTED_QUERY_IDS or protocol.get("heldout_nonself_knn", {}).get("k_values") != EXPECTED_KS: fail(errors, "protocol_heldout")
        if protocol.get("selection_witness", {}).get("full_structure_fingerprint") != EXPECTED_FINGERPRINT: fail(errors, "protocol_fingerprint")
        header, pool, queries, manifest = load_bundle(a.bundle.resolve(), errors)
        if (header["base_n"], header["pool_n"], header["dim"]) != (4096, 6144, 128): fail(errors, "frozen_header", observed=header)
        initial = load_snapshot(a.initial_geometry.resolve(), errors, "initial")
        final = load_snapshot(a.final_geometry.resolve(), errors, "final")
        selection_path = Path(protocol["selection_witness"]["path"])
        if not selection_path.is_file() or sha256_file(selection_path) != protocol["selection_witness"]["sha256"]:
            fail(errors, "selection_witness_sha")
            selection = None
        else:
            selection = json.loads(selection_path.read_text())
            witness_path = Path(selection["frozen_input"]["geometry_witness"])
            if not witness_path.is_file() or sha256_file(witness_path) != selection["frozen_input"]["geometry_witness_sha256"]:
                fail(errors, "geometry_witness_sha")
            else:
                witness = load_snapshot(witness_path, errors, "selection_witness")
                if initial != witness: fail(errors, "runtime_initial_geometry_differs_from_witness")
                if fnv1a64_geometry(witness) != EXPECTED_FINGERPRINT: fail(errors, "witness_fingerprint")
        if fnv1a64_geometry(initial) != EXPECTED_FINGERPRINT: fail(errors, "runtime_initial_fingerprint", observed=fnv1a64_geometry(initial))
        upper_ok, upper_errors = search_upper_invariant(initial)
        if not upper_ok: fail(errors, "search_upper_invariant", details=upper_errors[:30])
        target = initial["nodes"][EXPECTED_TARGET] if EXPECTED_TARGET < len(initial["nodes"]) else None
        if target is None or target.get("empty") != 0 or target.get("is_leaf") != 1 or target.get("size") != 4 or leaf_ids(target) != EXPECTED_INITIAL_IDS:
            fail(errors, "runtime_target_leaf_contract", observed=target)
        for query_id in EXPECTED_QUERY_IDS:
            if query_id >= queries.shape[0] or bool(np.any(np.all(pool == queries[query_id], axis=1))):
                fail(errors, "heldout_query_not_nonself", query_id=query_id)

        rows = [json.loads(line) for line in a.engine_jsonl.read_text().splitlines() if line.strip()]
        if not rows or rows[0].get("record") != "meta":
            fail(errors, "missing_meta")
        else:
            meta = rows[0]
            checks = {"schema": "c3-same-leaf-capacity-heldout-runner-v1", "archived_incremental_updater_used": False,
                      "buffer_merge_used": False, "residual_pruning_mode": 0, "residual_pruning_runtime_verified": True,
                      "target_leaf": EXPECTED_TARGET, "initial_target_occupancy": 4,
                      "expected_final_target_occupancy": 20, "capacity_boundary_stable_id": EXPECTED_BOUNDARY,
                      "selection_geometry_structural_fingerprint": EXPECTED_FINGERPRINT,
                      "heldout_query_ids": EXPECTED_QUERY_IDS, "heldout_k_values": EXPECTED_KS,
                      "search_upper_invariant_ok": upper_ok}
            for key, expected in checks.items():
                if meta.get(key) != expected: fail(errors, "meta_contract", field=key, expected=expected, observed=meta.get(key))

        active = set(range(header["base_n"]))
        appended: dict[int, list[int]] = {}
        index = 1
        heldout_checks = 0
        for event_index, stable_id in enumerate(EXPECTED_ACCEPTED, start=1):
            if index >= len(rows): fail(errors, "missing_candidate", event_index=event_index); break
            row = rows[index]
            cert = certificate(initial, pool, stable_id, appended, upper_ok)
            if row.get("record") != "candidate" or row.get("stable_id") != stable_id or row.get("outcome") != "accepted" or row.get("reason") != "accepted":
                fail(errors, "accepted_candidate_record", event_index=event_index, observed=row)
            expected_before = leaf_ids(initial["nodes"][EXPECTED_TARGET]) + appended.get(EXPECTED_TARGET, [])
            expected_after = expected_before + [stable_id]
            fields = {"leaf_id": EXPECTED_TARGET, "pre_leaf_size": 3 + event_index, "leaf_lid": initial["nodes"][EXPECTED_TARGET]["lid"],
                      "write_slot": initial["nodes"][EXPECTED_TARGET]["lid"] + 3 + event_index,
                      "leaf_ids_before": expected_before, "leaf_ids_after": expected_after}
            if not cert.get("ok") or cert.get("leaf") != EXPECTED_TARGET or cert.get("pre_leaf_size") != 3 + event_index:
                fail(errors, "accepted_certificate", event_index=event_index, certificate=cert)
            for key, expected in fields.items():
                if row.get(key) != expected: fail(errors, "accepted_append_record", event_index=event_index, field=key, expected=expected, observed=row.get(key))
            appended.setdefault(EXPECTED_TARGET, []).append(stable_id)
            active.add(stable_id)
            index += 1
            for query_id in EXPECTED_QUERY_IDS:
                for k in EXPECTED_KS:
                    if index >= len(rows): fail(errors, "missing_heldout_record", event_index=event_index, query_id=query_id, k=k); break
                    validate_topk_row(rows[index], event_index, stable_id, query_id, k, active, pool, queries, errors, index)
                    heldout_checks += 1
                    index += 1
        if index >= len(rows):
            fail(errors, "missing_capacity_boundary")
        else:
            row = rows[index]
            cert = certificate(initial, pool, EXPECTED_BOUNDARY, appended, upper_ok)
            checks = {"record": "capacity_boundary", "stable_id": EXPECTED_BOUNDARY, "outcome": "rejected", "reason": "static_scan_limit",
                      "leaf_id": EXPECTED_TARGET, "pre_leaf_size": 20, "expected_static_scan_max_size": 20,
                      "native_append_called": False, "write_attempted": False, "buffer_merge_used": False,
                      "target_leaf_size_before": 20, "target_leaf_size_after": 20}
            for key, expected in checks.items():
                if row.get(key) != expected: fail(errors, "capacity_boundary_record", field=key, expected=expected, observed=row.get(key))
            if cert.get("ok") or cert.get("reason") != "static_scan_limit" or cert.get("leaf") != EXPECTED_TARGET or cert.get("pre_leaf_size") != 20:
                fail(errors, "capacity_boundary_certificate", certificate=cert)
            index += 1
        if index != len(rows): fail(errors, "unexpected_trailing_records", count=len(rows)-index)
        if heldout_checks != 144: fail(errors, "heldout_check_count", observed=heldout_checks)
        validate_geometry(initial, final, appended, errors)
        final_target = final["nodes"][EXPECTED_TARGET] if EXPECTED_TARGET < len(final["nodes"]) else None
        if final_target is None or final_target.get("size") != 20 or leaf_ids(final_target) != EXPECTED_INITIAL_IDS + EXPECTED_ACCEPTED:
            fail(errors, "final_target_leaf_contract", observed=final_target)
        if final_target is not None and EXPECTED_BOUNDARY in leaf_ids(final_target): fail(errors, "boundary_candidate_written")
        result = {
            "schema": SCHEMA, "status": "PASS" if not errors else "FAIL", "gpu_used": False,
            "scope": "independent CPU validation of one same-leaf static-scan capacity boundary and external non-self held-out KNN trace; not complete C3/performance evidence",
            "accepted": len(EXPECTED_ACCEPTED), "heldout_checks": heldout_checks, "boundary_stable_id": EXPECTED_BOUNDARY,
            "error_count": len(errors), "errors": errors[:300],
            "search_upper_invariant": {"pass": upper_ok, "errors": upper_errors[:30]},
            "inputs": {"protocol": {"path": str(a.protocol.resolve()), "sha256": sha256_file(a.protocol.resolve())},
                       "bundle": str(a.bundle.resolve()),
                       "initial_geometry": {"path": str(a.initial_geometry.resolve()), "sha256": sha256_file(a.initial_geometry.resolve())},
                       "final_geometry": {"path": str(a.final_geometry.resolve()), "sha256": sha256_file(a.final_geometry.resolve())},
                       "engine_jsonl": {"path": str(a.engine_jsonl.resolve()), "sha256": sha256_file(a.engine_jsonl.resolve())},
                       "validator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())}},
            "limitations": ["one frozen leaf/sequence/query set", "no buffer merge", "no delete/range/concurrency/rebuild", "not performance evidence"],
        }
        a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": result["status"], "errors": len(errors), "out": str(a.out.resolve())}, sort_keys=True))
        return 0 if not errors else 2
    except Exception as exc:
        result = {"schema": SCHEMA, "status": "FAIL_VALIDATOR_EXCEPTION", "gpu_used": False,
                  "error_type": type(exc).__name__, "error": str(exc), "error_count": len(errors), "errors": errors[:300]}
        a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": result["status"], "error": str(exc), "out": str(a.out.resolve())}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
