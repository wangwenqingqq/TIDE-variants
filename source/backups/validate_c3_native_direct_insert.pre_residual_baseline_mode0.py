#!/workspace/legacy_workspace/GTS/bench_env/bin/python
"""Independent CPU validator for the pre-registered native C3 direct-insert probe.

It does not trust the runner summary or its embedded exact answers.  It reads
only the frozen int16 payload, exported initial/final real-GTS geometry
snapshots, and per-event runner JSONL; reconstructs stable-ID active state,
strict certificate paths, leaf append invariants, and every self-query exact
int64 L2 top-K answer.
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
SCHEMA = "c3-native-certified-direct-insert-cpu-validator-v1"


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
    """Sequential float32 accumulation matching the runner's host certificate."""
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
    tolerance = max(0.002, 2.0 * f32_ulp(magnitude))
    return abs(float(observed) - reference) <= tolerance


def load_bundle(bundle: Path, errors: list[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray]:
    manifest_path = bundle / "manifest.json"
    metadata_path = bundle / "metadata.json"
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if manifest.get("schema") != "e1gi-b-quantized-gts-integration-manifest-v1":
        fail(errors, "manifest_schema", observed=manifest.get("schema"))
    for name, expected in manifest.get("files_sha256", {}).items():
        p = bundle / name
        if not p.is_file() or sha256_file(p) != expected:
            fail(errors, "frozen_input_hash_mismatch", file=name, expected=expected,
                 observed=sha256_file(p) if p.is_file() else None)
    blob = (bundle / "trace.e1gtrc").read_bytes()
    if len(blob) < HEADER.size:
        raise ValueError("truncated trace")
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, radius, events = HEADER.unpack_from(blob, 0)
    header = {"dim": int(dim), "base_n": int(base_n), "reservoir_n": int(reservoir_n),
              "pool_n": int(pool_n), "query_n": int(query_n), "k": int(k),
              "radius": float(radius), "event_count": int(events)}
    if magic != MAGIC or version != 1:
        fail(errors, "trace_magic_version", magic=magic.decode("ascii", "replace"), version=int(version))
    if reservoir_n != pool_n - base_n or k <= 0 or k > base_n:
        fail(errors, "trace_header_contract", header=header)
    for key in ("dim", "base_n", "reservoir_n", "pool_n", "query_n", "k", "event_count"):
        if int(metadata.get("header", {}).get(key, -1)) != header[key]:
            fail(errors, "metadata_header_mismatch", key=key)
    raw_pool = np.fromfile(bundle / "pool.i16", dtype="<i2")
    if raw_pool.size != pool_n * dim:
        raise ValueError("pool payload size mismatch")
    pool = raw_pool.reshape(pool_n, dim)
    return header, pool


def load_snapshot(path: Path, errors: list[dict[str, Any]], label: str) -> dict[str, Any]:
    snapshot = json.loads(path.read_text())
    if snapshot.get("schema") != "c3-native-gts-geometry-snapshot-v1":
        fail(errors, "geometry_schema", label=label, observed=snapshot.get("schema"))
    nodes = snapshot.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"{label}: empty nodes")
    for expected_id, node in enumerate(nodes):
        if not isinstance(node, dict) or node.get("node_id") != expected_id:
            fail(errors, "geometry_node_id", label=label, expected=expected_id, observed=node.get("node_id") if isinstance(node, dict) else None)
            continue
        for key in ("empty", "pid", "min_dis_bits", "max_dis_bits", "size", "lid", "is_leaf"):
            if not isinstance(node.get(key), int):
                fail(errors, "geometry_node_field", label=label, node_id=expected_id, field=key)
        if node.get("empty") == 0 and node.get("is_leaf") == 1:
            ids = node.get("logical_ids")
            if not isinstance(ids, list) or len(ids) != node.get("size") or any(not isinstance(x, int) for x in ids):
                fail(errors, "geometry_leaf_ids", label=label, node_id=expected_id)
    return snapshot


def node(snapshot: dict[str, Any], idx: int) -> dict[str, Any]:
    return snapshot["nodes"][idx]


def leaf_ids(n: dict[str, Any]) -> list[int]:
    return list(n.get("logical_ids", []))


def search_upper_invariant(initial: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    """Audit actual `search_v2::nodeProcessKnn` next-sibling upper semantics."""
    nodes = initial["nodes"]
    order = int(initial["tree_order"])
    errors: list[dict[str, Any]] = []
    for parent in range(len(nodes)):
        if node(initial, parent)["empty"] != 0:
            continue
        for slot in range(order):
            child_id = parent * order + slot + 1
            if child_id < 0 or child_id >= len(nodes) or node(initial, child_id)["empty"] != 0:
                continue
            child = node(initial, child_id)
            lower = float(f32_from_bits(child["min_dis_bits"]))
            max_upper = float(f32_from_bits(child["max_dis_bits"]))
            if not lower <= max_upper:
                errors.append({"type": "min_exceeds_max", "child": child_id, "lower": lower, "max_upper": max_upper})
            # Exact static branch: `if (nid % TREE_ORDER != 0) dis_lb2 = dis_q - node_list[nid+1].min_dis`.
            if child_id % order == 0:
                continue
            next_id = child_id + 1
            if next_id >= len(nodes) or node(initial, next_id)["empty"] != 0 or (next_id - 1) // order != parent:
                errors.append({"type": "missing_next_sibling", "child": child_id, "next": next_id})
                continue
            next_min = float(f32_from_bits(node(initial, next_id)["min_dis_bits"]))
            if not max_upper <= next_min:
                errors.append({"type": "max_exceeds_next_min", "child": child_id, "max_upper": max_upper,
                               "next": next_id, "next_min": next_min})
    return not errors, errors


def certificate(initial: dict[str, Any], pool: np.ndarray, stable_id: int,
                appended: dict[int, list[int]], epsilon: float, search_upper_ok: bool) -> dict[str, Any]:
    nodes = initial["nodes"]
    tree_order = int(initial["tree_order"])
    tree_height = int(initial["tree_height"])
    max_size = int(initial["max_size"])
    pad = int(initial["leaf_pad_slots"])
    parent = 0
    path: list[dict[str, Any]] = []
    if not search_upper_ok:
        return {"ok": False, "reason": "search_upper_invariant", "path": path}
    if tree_height <= 1 or tree_order <= 1:
        return {"ok": False, "reason": "missing_or_invalid_path", "path": path}
    for _level in range(tree_height - 1):
        child_ids = [parent * tree_order + slot + 1 for slot in range(tree_order)]
        existing = [cid for cid in child_ids if 0 <= cid < len(nodes) and node(initial, cid)["empty"] == 0]
        if not existing:
            return {"ok": False, "reason": "missing_or_invalid_path", "path": path}
        pivot = node(initial, existing[0])["pid"]
        if pivot < 0 or pivot >= pool.shape[0] or any(node(initial, cid)["pid"] != pivot for cid in existing):
            return {"ok": False, "reason": "missing_or_invalid_path", "path": path}
        distance = f32_l2_pool(pool, stable_id, pivot)
        matches: list[int] = []
        for cid in existing:
            child = node(initial, cid)
            lower = float(f32_from_bits(child["min_dis_bits"]))
            max_upper = float(f32_from_bits(child["max_dis_bits"]))
            has_next = cid % tree_order != 0
            search_upper = max_upper
            if has_next:
                next_id = cid + 1
                if next_id >= len(nodes) or node(initial, next_id)["empty"] != 0 or (next_id - 1) // tree_order != parent:
                    return {"ok": False, "reason": "search_upper_invariant", "path": path}
                search_upper = float(f32_from_bits(node(initial, next_id)["min_dis_bits"]))
            if distance > lower + epsilon and distance < max_upper - epsilon and (not has_next or distance < search_upper - epsilon):
                matches.append(cid)
        chosen = matches[0] if len(matches) == 1 else -1
        path.append({"parent": parent, "pivot": pivot, "child": chosen, "matching_children": len(matches), "distance": distance})
        if len(matches) != 1:
            return {"ok": False, "reason": "no_unique_strict_child", "path": path}
        if node(initial, chosen)["is_leaf"] == 1:
            leaf = chosen
            pre_size = int(node(initial, leaf)["size"]) + len(appended.get(leaf, []))
            if pre_size >= max_size:
                return {"ok": False, "reason": "static_scan_limit", "path": path, "leaf": leaf, "pre_leaf_size": pre_size}
            if len(appended.get(leaf, [])) >= pad:
                return {"ok": False, "reason": "padded_leaf_capacity", "path": path, "leaf": leaf, "pre_leaf_size": pre_size}
            return {"ok": True, "reason": "accepted", "path": path, "leaf": leaf, "pre_leaf_size": pre_size}
        parent = chosen
    return {"ok": False, "reason": "not_leaf", "path": path}

def exact_topk(pool: np.ndarray, active: set[int], query_id: int, k: int) -> tuple[list[int], list[int], bool]:
    ids = np.asarray(sorted(active), dtype=np.int64)
    diff = pool[ids].astype(np.int64, copy=False) - pool[query_id].astype(np.int64, copy=False)
    squared = np.sum(diff * diff, axis=1, dtype=np.int64)
    order = np.lexsort((ids, squared))
    boundary = bool(len(order) > k and int(squared[order[k - 1]]) == int(squared[order[k]]))
    top = order[:k]
    return [int(ids[i]) for i in top], [int(squared[i]) for i in top], boundary


def validate_query(row: dict[str, Any], stable_id: int, active: set[int], pool: np.ndarray, k: int,
                   errors: list[dict[str, Any]], record_index: int) -> None:
    if row.get("record") != "self_query" or row.get("stable_id") != stable_id:
        fail(errors, "missing_or_wrong_self_query", record_index=record_index, stable_id=stable_id)
        return
    if row.get("active_count") != len(active):
        fail(errors, "self_query_active_count", stable_id=stable_id, expected=len(active), observed=row.get("active_count"))
    expected_ids, expected_sq, boundary = exact_topk(pool, active, stable_id, k)
    if boundary:
        fail(errors, "k_boundary_tie", stable_id=stable_id)
        return
    got_ids = row.get("gts_ids")
    got_dist = row.get("gts_distances")
    if not isinstance(got_ids, list) or not isinstance(got_dist, list) or len(got_ids) != k or len(got_dist) != k:
        fail(errors, "gts_result_shape", stable_id=stable_id)
        return
    if any(not isinstance(x, int) for x in got_ids) or len(set(got_ids)) != len(got_ids):
        fail(errors, "gts_result_ids", stable_id=stable_id)
        return
    pos = 0
    while pos < k:
        end = pos + 1
        while end < k and expected_sq[end] == expected_sq[pos]:
            end += 1
        if sorted(got_ids[pos:end]) != sorted(expected_ids[pos:end]):
            fail(errors, "gts_topk_ids", stable_id=stable_id, ranks=[pos, end - 1],
                 expected=expected_ids[pos:end], observed=got_ids[pos:end])
        for rank in range(pos, end):
            try:
                observed = float(got_dist[rank])
            except (TypeError, ValueError):
                fail(errors, "gts_distance_type", stable_id=stable_id, rank=rank)
                continue
            if not math.isfinite(observed) or not close_static_distance(observed, expected_sq[rank]):
                fail(errors, "gts_distance", stable_id=stable_id, rank=rank,
                     expected_squared=expected_sq[rank], observed=got_dist[rank])
        pos = end
    if stable_id not in got_ids:
        fail(errors, "self_not_visible", stable_id=stable_id)
    else:
        self_rank = got_ids.index(stable_id)
        if float(got_dist[self_rank]) != 0.0:
            fail(errors, "self_distance_not_zero", stable_id=stable_id, observed=got_dist[self_rank])


def validate_geometry(initial: dict[str, Any], final: dict[str, Any], appended: dict[int, list[int]],
                      errors: list[dict[str, Any]]) -> None:
    if (initial.get("tree_height"), initial.get("tree_order"), initial.get("max_size"), initial.get("leaf_pad_slots")) != \
       (final.get("tree_height"), final.get("tree_order"), final.get("max_size"), final.get("leaf_pad_slots")):
        fail(errors, "geometry_header_changed")
    if len(initial["nodes"]) != len(final["nodes"]):
        fail(errors, "geometry_node_count_changed")
        return
    for before, after in zip(initial["nodes"], final["nodes"]):
        node_id = before.get("node_id")
        for key in ("node_id", "empty", "pid", "min_dis_bits", "max_dis_bits", "lid", "is_leaf"):
            if before.get(key) != after.get(key):
                fail(errors, "frozen_geometry_mutation", node_id=node_id, field=key,
                     before=before.get(key), after=after.get(key))
        added = appended.get(node_id, [])
        if before.get("empty") != 0 or before.get("is_leaf") != 1:
            if before.get("size") != after.get("size"):
                fail(errors, "nonleaf_size_mutation", node_id=node_id)
            continue
        expected_ids = leaf_ids(before) + added
        if after.get("size") != before.get("size") + len(added):
            fail(errors, "leaf_size_mutation", node_id=node_id,
                 expected=before.get("size") + len(added), observed=after.get("size"))
        if leaf_ids(after) != expected_ids:
            fail(errors, "leaf_logical_ids", node_id=node_id, expected=expected_ids, observed=leaf_ids(after))


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
        if protocol.get("schema") != "c3-native-certified-direct-insert-protocol-v2-search-upper":
            fail(errors, "protocol_schema", observed=protocol.get("schema"))
        if protocol.get("status") != "PRE_REGISTERED_AFTER_STATIC_SEARCH_UPPER_AUDIT_BEFORE_GPU_EXECUTION":
            fail(errors, "protocol_not_preregistered", observed=protocol.get("status"))
        epsilon = float(protocol.get("acceptance_certificate", {}).get("epsilon", -1))
        max_accepts = int(protocol.get("candidate_selection", {}).get("maximum_accepted_inserts", -1))
        if epsilon != 0.0001 or max_accepts != 16:
            fail(errors, "protocol_parameters", epsilon=epsilon, max_accepts=max_accepts)
        header, pool = load_bundle(a.bundle.resolve(), errors)
        initial = load_snapshot(a.initial_geometry.resolve(), errors, "initial")
        final = load_snapshot(a.final_geometry.resolve(), errors, "final")
        if initial.get("tree_order") != 10 or initial.get("max_size") != 20 or initial.get("leaf_pad_slots") != 64:
            fail(errors, "geometry_constants", tree_order=initial.get("tree_order"), max_size=initial.get("max_size"), pad=initial.get("leaf_pad_slots"))
        search_upper_ok, search_upper_errors = search_upper_invariant(initial)
        if not search_upper_ok:
            fail(errors, "search_upper_invariant", details=search_upper_errors[:40])
        # Initial leaf payload must exactly realize the frozen base stable-ID set.
        initial_leaf_union: list[int] = []
        for n in initial["nodes"]:
            if n.get("empty") == 0 and n.get("is_leaf") == 1:
                initial_leaf_union.extend(leaf_ids(n))
        if sorted(initial_leaf_union) != list(range(header["base_n"])):
            fail(errors, "initial_base_leaf_payload", count=len(initial_leaf_union))

        rows = [json.loads(line) for line in a.engine_jsonl.read_text().splitlines() if line.strip()]
        if not rows or rows[0].get("record") != "meta":
            fail(errors, "missing_runner_meta")
        elif rows[0].get("archived_incremental_updater_used") is not False:
            fail(errors, "legacy_updater_not_disabled")
        elif rows[0].get("search_upper_invariant_ok") is not search_upper_ok:
            fail(errors, "runner_search_upper_audit", expected=search_upper_ok, observed=rows[0].get("search_upper_invariant_ok"))
        active = set(range(header["base_n"]))
        appended: dict[int, list[int]] = {}
        accepted = 0
        rejected = 0
        candidates = 0
        next_stable_id = header["base_n"]
        i = 1
        while i < len(rows):
            row = rows[i]
            if row.get("record") != "candidate":
                fail(errors, "unexpected_record", index=i, record=row.get("record"))
                i += 1
                continue
            stable_id = row.get("stable_id")
            if stable_id != next_stable_id:
                fail(errors, "candidate_order", index=i, expected=next_stable_id, observed=stable_id)
            next_stable_id += 1
            candidates += 1
            if not isinstance(stable_id, int) or not (header["base_n"] <= stable_id < header["pool_n"]):
                fail(errors, "candidate_stable_id", index=i, observed=stable_id)
                i += 1
                continue
            cert = certificate(initial, pool, stable_id, appended, epsilon, search_upper_ok)
            expected_outcome = "accepted" if cert.get("ok") else "rejected"
            if row.get("outcome") != expected_outcome or row.get("reason") != cert.get("reason"):
                fail(errors, "certificate_outcome", stable_id=stable_id,
                     expected={"outcome": expected_outcome, "reason": cert.get("reason")},
                     observed={"outcome": row.get("outcome"), "reason": row.get("reason")})
            if cert.get("ok"):
                if accepted >= max_accepts:
                    fail(errors, "accepted_over_preregistered_limit", stable_id=stable_id)
                leaf = int(cert["leaf"])
                pre_size = int(cert["pre_leaf_size"])
                initial_leaf = node(initial, leaf)
                expected_before = leaf_ids(initial_leaf) + appended.get(leaf, [])
                expected_after = expected_before + [stable_id]
                checks = {
                    "leaf_id": leaf,
                    "pre_leaf_size": pre_size,
                    "leaf_lid": initial_leaf["lid"],
                    "write_slot": initial_leaf["lid"] + pre_size,
                    "leaf_ids_before": expected_before,
                    "leaf_ids_after": expected_after,
                }
                for key, expected in checks.items():
                    if row.get(key) != expected:
                        fail(errors, "native_append_record", stable_id=stable_id, field=key,
                             expected=expected, observed=row.get(key))
                appended.setdefault(leaf, []).append(stable_id)
                active.add(stable_id)
                accepted += 1
                i += 1
                if i >= len(rows):
                    fail(errors, "missing_self_query", stable_id=stable_id)
                    break
                validate_query(rows[i], stable_id, active, pool, header["k"], errors, i)
            else:
                rejected += 1
            i += 1
        if accepted == 0:
            fail(errors, "inconclusive_no_certified_insert")
        if accepted > max_accepts:
            fail(errors, "accepted_count_limit", accepted=accepted, max_accepts=max_accepts)
        validate_geometry(initial, final, appended, errors)
        result = {
            "schema": SCHEMA,
            "status": "PASS" if not errors else "FAIL",
            "scope": "independent CPU validation of a bounded native C3 direct-insert correctness probe; not complete C3/performance evidence",
            "gpu_used": False,
            "accepted": accepted,
            "rejected": rejected,
            "candidates_examined": candidates,
            "final_active_count": len(active),
            "search_upper_invariant": {"pass": search_upper_ok, "errors": search_upper_errors[:40]},
            "error_count": len(errors), "errors": errors[:200],
            "inputs": {
                "protocol": {"path": str(a.protocol.resolve()), "sha256": sha256_file(a.protocol.resolve())},
                "bundle": str(a.bundle.resolve()),
                "initial_geometry": {"path": str(a.initial_geometry.resolve()), "sha256": sha256_file(a.initial_geometry.resolve())},
                "final_geometry": {"path": str(a.final_geometry.resolve()), "sha256": sha256_file(a.final_geometry.resolve())},
                "engine_jsonl": {"path": str(a.engine_jsonl.resolve()), "sha256": sha256_file(a.engine_jsonl.resolve())},
                "validator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            },
            "limitations": [
                "insert-only certified native direct tier",
                "rejected buffer/merge is intentionally not validated",
                "no range/delete/concurrency/rebuild/performance claim",
                "strict certificate and exact answers apply only to the exported frozen geometry and bounded SIFT trace",
            ],
        }
        a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": result["status"], "accepted": accepted, "errors": len(errors), "out": str(a.out.resolve())}, sort_keys=True))
        return 0 if not errors else 2
    except Exception as exc:
        result = {"schema": SCHEMA, "status": "FAIL_VALIDATOR_EXCEPTION", "gpu_used": False,
                  "error_type": type(exc).__name__, "error": str(exc), "error_count": len(errors), "errors": errors[:200]}
        a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": result["status"], "error": str(exc), "out": str(a.out.resolve())}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
