#!/workspace/legacy_workspace/GTS/bench_env/bin/python
"""CPU-only selection/provenance gate for a bounded C3 same-leaf capacity test.

This tool never calls CUDA or nvidia-smi.  It reads the immutable frozen SIFT
bundle and an already-exported real-GTS initial geometry witness, finds exactly
one deterministic target leaf with 17 strict interior reservoir candidates,
and chooses external query vectors that are non-self under the quantized L2
contract.  The resulting JSON is a selection witness, not runtime evidence:
the CUDA runner and independent validator recompute every route/capacity rule
from their own exported geometry before accepting a result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"E1GTRC01"
HEADER = struct.Struct("<8sI6IfQ")
SCHEMA = "c3-same-leaf-capacity-selection-v1"
EPSILON = 1.0e-4
MAX_SIZE = 20
LEAF_PAD_SLOTS = 64
INITIAL_OCCUPANCY = 4
ACCEPT_COUNT = MAX_SIZE - INITIAL_OCCUPANCY
HELDOUT_QUERY_COUNT = 3
KS = (1, 10, 20)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def f32_from_bits(bits: int) -> np.float32:
    return np.frombuffer(struct.pack("<I", int(bits)), dtype="<f4")[0]


def f32_l2_pool(pool: np.ndarray, left: int, right: int) -> float:
    total = np.float32(0.0)
    for a, b in zip(pool[left], pool[right]):
        delta = np.float32(np.float32(a) - np.float32(b))
        total = np.float32(total + np.float32(delta * delta))
    return float(np.sqrt(total, dtype=np.float32))


def read_bundle(bundle: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("schema") != "e1gi-b-quantized-gts-integration-manifest-v1":
        raise ValueError("unexpected frozen bundle schema")
    for name, expected in manifest.get("files_sha256", {}).items():
        observed = sha256_file(bundle / name)
        if observed != expected:
            raise ValueError(f"frozen input hash mismatch: {name}: {observed} != {expected}")
    blob = (bundle / "trace.e1gtrc").read_bytes()
    if len(blob) < HEADER.size:
        raise ValueError("truncated trace")
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = HEADER.unpack_from(blob)
    if magic != MAGIC or version != 1:
        raise ValueError("unexpected trace magic/version")
    if reservoir_n != pool_n - base_n or dim <= 0 or base_n <= 0 or query_n <= 0:
        raise ValueError("invalid trace header")
    pool = np.fromfile(bundle / "pool.i16", dtype="<i2")
    if pool.size != pool_n * dim:
        raise ValueError("pool byte shape mismatch")
    queries = np.fromfile(bundle / "queries.i16", dtype="<i2")
    if queries.size != query_n * dim:
        raise ValueError("query byte shape mismatch")
    header = {
        "dim": int(dim), "base_n": int(base_n), "reservoir_n": int(reservoir_n),
        "pool_n": int(pool_n), "query_n": int(query_n), "k": int(k),
        "radius": float(radius), "event_count": int(event_count),
    }
    return header, pool.reshape(pool_n, dim), queries.reshape(query_n, dim), manifest


def read_geometry(path: Path, base_n: int) -> dict[str, Any]:
    geometry = json.loads(path.read_text())
    if geometry.get("schema") != "c3-native-gts-geometry-snapshot-v1":
        raise ValueError("unexpected geometry schema")
    if geometry.get("tree_order") != 10 or geometry.get("max_size") != MAX_SIZE or geometry.get("leaf_pad_slots") != LEAF_PAD_SLOTS:
        raise ValueError("geometry constants disagree with static source contract")
    nodes = geometry.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("empty geometry")
    ids: list[int] = []
    for expected_id, node in enumerate(nodes):
        if node.get("node_id") != expected_id:
            raise ValueError("geometry node IDs are not contiguous")
        if node.get("empty") == 0 and node.get("is_leaf") == 1:
            logical = node.get("logical_ids")
            if not isinstance(logical, list) or len(logical) != node.get("size"):
                raise ValueError(f"invalid leaf payload at {expected_id}")
            ids.extend(int(x) for x in logical)
    if sorted(ids) != list(range(base_n)):
        raise ValueError("geometry initial leaves are not exactly the frozen base stable-ID set")
    return geometry


def fnv1a64_geometry(geometry: dict[str, Any]) -> str:
    """Canonical full geometry fingerprint for a pre-write runtime gate.

    FNV-1a is not a cryptographic replacement for the recorded SHA-256 of the
    witness file.  It is a dependency-free, cross-language exact-structure
    gate: header plus every node field plus every initial leaf stable ID.
    """
    h = 0xCBF29CE484222325
    prime = 0x100000001B3

    def mix_u64(value: int) -> None:
        nonlocal h
        for byte in (int(value) & ((1 << 64) - 1)).to_bytes(8, "little", signed=False):
            h ^= byte
            h = (h * prime) & ((1 << 64) - 1)

    nodes = geometry["nodes"]
    for value in (geometry["tree_height"], geometry["tree_order"], geometry["max_size"],
                  geometry["leaf_pad_slots"], len(nodes)):
        mix_u64(value)
    for node in nodes:
        for key in ("node_id", "empty", "pid", "min_dis_bits", "max_dis_bits", "size", "lid", "is_leaf"):
            mix_u64(node[key])
        ids = node.get("logical_ids", [])
        mix_u64(len(ids))
        for stable_id in ids:
            mix_u64(stable_id)
    return f"fnv1a64:{h:016x}"


def strict_route(geometry: dict[str, Any], pool: np.ndarray, stable_id: int) -> dict[str, Any]:
    nodes = geometry["nodes"]
    order = int(geometry["tree_order"])
    height = int(geometry["tree_height"])
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
        for child in children:
            node = nodes[child]
            lower = float(f32_from_bits(node["min_dis_bits"]))
            upper = float(f32_from_bits(node["max_dis_bits"]))
            if not lower <= upper:
                return {"ok": False, "reason": "search_upper_invariant", "path": path}
            has_next = child % order != 0
            search_upper = upper
            if has_next:
                nxt = child + 1
                if nxt >= len(nodes) or nodes[nxt].get("empty") != 0 or (nxt - 1) // order != parent:
                    return {"ok": False, "reason": "search_upper_invariant", "path": path}
                search_upper = float(f32_from_bits(nodes[nxt]["min_dis_bits"]))
                if upper > search_upper:
                    return {"ok": False, "reason": "search_upper_invariant", "path": path}
            if distance > lower + EPSILON and distance < upper - EPSILON and (not has_next or distance < search_upper - EPSILON):
                matches.append((child, lower, upper, search_upper, has_next))
        chosen = matches[0] if len(matches) == 1 else None
        path.append({
            "parent": parent, "pivot": pivot, "child": chosen[0] if chosen else -1,
            "matching_children": len(matches), "distance": distance,
            "lower": chosen[1] if chosen else None, "max_upper": chosen[2] if chosen else None,
            "search_upper": chosen[3] if chosen else None,
            "has_next_sibling_upper": chosen[4] if chosen else None,
        })
        if chosen is None:
            return {"ok": False, "reason": "no_unique_strict_child", "path": path}
        if nodes[chosen[0]].get("is_leaf") == 1:
            return {"ok": True, "reason": "strict_route", "leaf": chosen[0], "path": path}
        parent = chosen[0]
    return {"ok": False, "reason": "not_leaf", "path": path}


def exact_topk_external(pool: np.ndarray, active: np.ndarray, query: np.ndarray, k: int) -> tuple[list[int], list[int], bool]:
    ids = np.flatnonzero(active).astype(np.int64, copy=False)
    diff = pool[ids].astype(np.int64, copy=False) - query.astype(np.int64, copy=False)
    squared = np.sum(diff * diff, axis=1, dtype=np.int64)
    order = np.lexsort((ids, squared))
    boundary_tie = bool(len(order) > k and int(squared[order[k - 1]]) == int(squared[order[k]]))
    chosen = order[:k]
    return [int(ids[i]) for i in chosen], [int(squared[i]) for i in chosen], boundary_tie


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--geometry-witness", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    bundle = args.bundle.resolve()
    witness = args.geometry_witness.resolve()
    header, pool, queries, manifest = read_bundle(bundle)
    geometry = read_geometry(witness, header["base_n"])

    by_leaf: dict[int, list[int]] = defaultdict(list)
    rejected: dict[str, int] = defaultdict(int)
    for stable_id in range(header["base_n"], header["pool_n"]):
        route = strict_route(geometry, pool, stable_id)
        if route.get("ok"):
            by_leaf[int(route["leaf"])].append(stable_id)
        else:
            rejected[str(route.get("reason"))] += 1

    viable: list[tuple[int, list[int]]] = []
    for leaf, ids in sorted(by_leaf.items()):
        node = geometry["nodes"][leaf]
        if int(node["size"]) == INITIAL_OCCUPANCY and len(ids) >= ACCEPT_COUNT + 1:
            viable.append((leaf, ids))
    if not viable:
        counts = sorted(((leaf, len(ids), geometry["nodes"][leaf].get("size")) for leaf, ids in by_leaf.items()), key=lambda x: (-x[1], x[0]))[:32]
        raise SystemExit(f"no leaf has occupancy={INITIAL_OCCUPANCY} and at least {ACCEPT_COUNT + 1} strict candidates; top={counts}")
    target_leaf, target_candidates = viable[0]
    accepted_ids = target_candidates[:ACCEPT_COUNT]
    boundary_id = target_candidates[ACCEPT_COUNT]

    # The query rule deliberately sees neither static GTS answers nor any
    # post-update GPU result: choose the first query indices with no quantized
    # duplicate in the immutable pool and no exact-oracle K boundary tie across
    # all requested active states.  This is a comparability/tie filter only.
    active = np.zeros(header["pool_n"], dtype=bool)
    active[:header["base_n"]] = True
    selected_queries: list[int] = []
    query_audit: list[dict[str, Any]] = []
    for query_id, query in enumerate(queries):
        if np.any(np.all(pool == query, axis=1)):
            continue
        no_tie = True
        state_audit: list[dict[str, Any]] = []
        tmp = active.copy()
        for event_index, inserted in enumerate(accepted_ids, start=1):
            tmp[inserted] = True
            ks_ok: dict[str, bool] = {}
            for k in KS:
                _, _, boundary = exact_topk_external(pool, tmp, query, k)
                ks_ok[str(k)] = not boundary
                no_tie = no_tie and not boundary
            state_audit.append({"event_index": event_index, "active_count": int(tmp.sum()), "no_boundary_tie": ks_ok})
        if not no_tie:
            continue
        selected_queries.append(query_id)
        query_audit.append({"query_id": query_id, "nonself_against_entire_pool": True, "state_tie_audit": state_audit})
        if len(selected_queries) == HELDOUT_QUERY_COUNT:
            break
    if len(selected_queries) != HELDOUT_QUERY_COUNT:
        raise SystemExit("could not select required non-self external query vectors without oracle K-boundary ties")

    # Re-evaluate the boundary candidate after the 16 accepted writes.  Its
    # strict route must remain target_leaf; only the static MAX_SIZE gate makes
    # the safe tier reject it.
    route_boundary = strict_route(geometry, pool, boundary_id)
    if not route_boundary.get("ok") or int(route_boundary["leaf"]) != target_leaf:
        raise SystemExit("boundary candidate no longer strict-routes to the selected target leaf")
    node = geometry["nodes"][target_leaf]
    result = {
        "schema": SCHEMA,
        "status": "CPU_ONLY_SELECTION_PASS_NO_CUDA",
        "gpu_used": False,
        "selection_geometry_is_witness_only": True,
        "selection_geometry_structural_fingerprint": fnv1a64_geometry(geometry),
        "frozen_input": {
            "bundle": str(bundle),
            "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "bundle_files_sha256": manifest.get("files_sha256", {}),
            "geometry_witness": str(witness),
            "geometry_witness_sha256": sha256_file(witness),
            "header": header,
        },
        "static_leaf_scan_contract": {
            "max_size": MAX_SIZE,
            "initial_target_occupancy": INITIAL_OCCUPANCY,
            "accepted_append_count": ACCEPT_COUNT,
            "expected_final_occupancy": MAX_SIZE,
            "next_strict_candidate_expected_safe_rejection": "static_scan_limit",
            "leaf_pad_slots": LEAF_PAD_SLOTS,
        },
        "candidate_selection_rule": {
            "rule": "scan reservoir stable IDs ascending; route with the strict v2 search-upper certificate; choose the smallest leaf ID with initial occupancy 4 and >=17 qualifying candidates; take that leaf's qualifying stable IDs in ascending order",
            "target_leaf": target_leaf,
            "target_leaf_lid": int(node["lid"]),
            "initial_target_ids": list(node["logical_ids"]),
            "accepted_stable_ids": accepted_ids,
            "boundary_stable_id": boundary_id,
            "boundary_route": route_boundary,
            "strict_candidates_for_target_count": len(target_candidates),
            "viable_leaf_count": len(viable),
            "rejected_route_counts": dict(sorted(rejected.items())),
        },
        "heldout_nonself_query_rule": {
            "rule": "after target/candidate selection, choose the first three frozen external query indices ascending that have no bitwise-equal int16 vector anywhere in pool.i16 and no exact int64 active-set K-boundary tie at K in {1,10,20} after each of the 16 predetermined inserts; no static-GTS result is inspected during selection",
            "query_ids": selected_queries,
            "query_audit": query_audit,
            "k_values": list(KS),
        },
        "limitations": [
            "selection witness is CPU-only and is not a CUDA result",
            "external query selection establishes non-self/tie-free comparability only, not statistical generalization",
            "this protocol intentionally excludes buffer merge, delete, range, concurrency, rebuild, and performance claims",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "target_leaf": target_leaf, "accepted": len(accepted_ids), "boundary": boundary_id, "heldout_query_ids": selected_queries, "out": str(args.out.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
