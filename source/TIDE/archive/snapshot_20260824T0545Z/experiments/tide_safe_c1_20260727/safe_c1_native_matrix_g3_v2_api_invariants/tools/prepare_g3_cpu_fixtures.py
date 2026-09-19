#!/usr/bin/env python3
"""Prepare CPU-only, hash-bound G3 Safe-C1 native-matrix fixtures.

No CUDA binary, GPU telemetry, profiler, or subprocess is invoked.  The
selector is a source-mirror only; native execution remains blocked until a
later actual-GTS certificate/range/rebuild preflight accepts the bundles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from g3_common import (
    SCALE,
    SourceMirrorTree,
    canonical_json_bytes,
    choose_nonboundary_radius_sq,
    classify_reservoir,
    exact_results,
    expected_gts_height,
    fbin_memmap,
    has_knn_boundary_tie,
    parse_gts_tree_config,
    quantize_f32,
    read_fbin_layout,
    sha256_file,
    stable_set_sha256,
    unique_within_radius_sq,
    write_json,
)

SCHEMA = "safe-c1-g3-cpu-fixture-v1"


class Planner:
    """CPU source-mirror state used only to choose pre-registered fixtures."""

    def __init__(self, pool: np.ndarray, base_n: int, order: int, max_size: int, leaf_cap: int, delta_threshold: int):
        self.pool = np.asarray(pool, dtype=np.int16)
        self.base_n = int(base_n)
        self.order, self.max_size = int(order), int(max_size)
        self.leaf_cap, self.delta_threshold = int(leaf_cap), int(delta_threshold)
        self.active: set[int] = set(range(base_n))
        self.base: set[int] = set(range(base_n))
        self.unused: set[int] = set(range(base_n, len(pool)))
        self.placement: dict[int, str] = {i: "base" for i in self.base}
        self.direct_leaf: dict[int, int] = {}
        self.sidecars: dict[int, list[int]] = {}
        self.delta: list[int] = []
        self.tree = SourceMirrorTree(self.pool, self.base, self.order, self.max_size)
        self._classification_cache: tuple[dict[int, list[int]], list[int], dict[int, dict[str, Any]]] | None = None
        self.tree_versions: list[dict[str, Any]] = [self.tree.metadata()]

    def _refresh_cache(self) -> tuple[dict[int, list[int]], list[int], dict[int, dict[str, Any]]]:
        if self._classification_cache is None:
            self._classification_cache = classify_reservoir(self.tree, sorted(self.unused))
        return self._classification_cache

    def _invalidate(self) -> None:
        self._classification_cache = None

    def _eligible_direct_groups(self) -> dict[int, list[int]]:
        groups, _, _ = self._refresh_cache()
        return {leaf: [sid for sid in values if sid in self.unused and len(self.sidecars.get(leaf, [])) < self.leaf_cap] for leaf, values in groups.items()}

    def select_same_leaf(self, count: int, require_narrow_unique: bool = True) -> tuple[int, list[int], dict[int, dict[str, Any]]]:
        groups, _, receipts = self._refresh_cache()
        # Only witness candidates need the narrow-radius uniqueness check. Do
        # not accidentally turn this CPU selector into an O(reservoir*pool)
        # exhaustive scan for every leaf.
        for leaf in sorted(groups):
            usable = [sid for sid in groups[leaf] if sid in self.unused]
            if len(usable) < count:
                continue
            chosen = usable[:count]
            if require_narrow_unique and not all(unique_within_radius_sq(self.pool, sid, range(len(self.pool)), 1) for sid in chosen):
                continue
            return int(leaf), list(map(int, chosen)), receipts
        raise RuntimeError(f"no source-mirror same-leaf group of {count} candidates")

    def select_direct(self) -> tuple[int, int, dict[str, Any]]:
        groups, _, receipts = self._refresh_cache()
        choices: list[tuple[int, int]] = []
        for leaf, values in groups.items():
            if len(self.sidecars.get(leaf, [])) >= self.leaf_cap:
                continue
            for sid in values:
                if sid in self.unused:
                    choices.append((len(self.sidecars.get(leaf, [])), int(leaf), int(sid)))
                    break
        if not choices:
            raise RuntimeError("no eligible source-mirror direct candidate")
        _, leaf, sid = min(choices)
        return sid, leaf, receipts[sid]

    def select_rejected_delta(self, require_narrow_unique: bool = False) -> tuple[int, dict[str, Any]]:
        _, rejected, receipts = self._refresh_cache()
        for sid in rejected:
            if sid not in self.unused:
                continue
            if require_narrow_unique and not unique_within_radius_sq(self.pool, sid, range(len(self.pool)), 1):
                continue
            return int(sid), receipts[sid]
        raise RuntimeError("no source-mirror certificate-rejected candidate")

    def insert(self, sid: int, expected: str, certificate: dict[str, Any], leaf: int | None = None) -> dict[str, Any]:
        sid = int(sid)
        if sid not in self.unused or sid in self.active:
            raise RuntimeError(f"invalid fixture insert {sid}")
        self.unused.remove(sid)
        self.active.add(sid)
        # Tree intervals are frozen until an explicit rebuild. Candidate
        # classifications therefore remain valid; selector methods filter the
        # mutable `unused` set and sidecar capacity separately.
        if expected in {"direct_same_leaf", "direct"}:
            actual_leaf = int(leaf if leaf is not None else certificate["leaf"])
            if not certificate.get("ok") or len(self.sidecars.get(actual_leaf, [])) >= self.leaf_cap:
                raise RuntimeError(f"source-mirror direct selection invalid sid={sid}")
            self.sidecars.setdefault(actual_leaf, []).append(sid)
            self.placement[sid] = "direct"
            self.direct_leaf[sid] = actual_leaf
            return {"placement": "direct", "sidecar_leaf_id": actual_leaf}
        if expected == "capacity_delta":
            actual_leaf = int(leaf if leaf is not None else certificate.get("leaf", -1))
            if not certificate.get("ok") or len(self.sidecars.get(actual_leaf, [])) < self.leaf_cap:
                raise RuntimeError(f"source-mirror capacity selection invalid sid={sid}")
            self.delta.append(sid)
            self.placement[sid] = "delta"
            return {"placement": "delta", "fallback": "capacity", "certified_leaf_id": actual_leaf}
        if expected == "certificate_delta":
            if certificate.get("ok"):
                raise RuntimeError(f"source-mirror certificate fallback invalid sid={sid}")
            self.delta.append(sid)
            self.placement[sid] = "delta"
            return {"placement": "delta", "fallback": "certificate", "certificate_reason": certificate.get("reason")}
        raise ValueError(expected)

    def delete(self, sid: int) -> dict[str, Any]:
        sid = int(sid)
        prior = self.placement.get(sid)
        if sid not in self.active or prior not in {"direct", "delta"}:
            raise RuntimeError(f"fixture refuses non-mutable/base delete sid={sid} prior={prior}")
        if prior == "direct":
            leaf = self.direct_leaf.pop(sid)
            self.sidecars[leaf].remove(sid)
        else:
            self.delta.remove(sid)
        self.active.remove(sid)
        self.placement[sid] = "deleted"
        # Deleting a direct/delta sidecar member does not alter frozen-tree
        # interval certificates, so keep the classification cache.
        return {"prior_placement": prior}

    def rebuild(self) -> dict[str, Any]:
        if len(self.delta) < self.delta_threshold:
            raise RuntimeError(f"fixture rebuild requires delta_live >= {self.delta_threshold}, got {len(self.delta)}")
        old = self.tree.metadata()
        self.base = set(self.active)
        self.sidecars.clear()
        self.delta.clear()
        self.direct_leaf.clear()
        for sid in self.active:
            self.placement[sid] = "base"
        self.tree = SourceMirrorTree(self.pool, self.base, self.order, self.max_size)
        self._invalidate()
        new = self.tree.metadata()
        self.tree_versions.append(new)
        return {
            "old_tree": old,
            "new_tree": new,
            "live_ids_sha256": stable_set_sha256(self.active),
            "base_count": len(self.base),
            "sidecar_live": 0,
            "delta_live": 0,
        }


def make_event(op: str, **values: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"op": op}
    value.update(values)
    return value


def assign_indices(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, event in enumerate(events):
        event["op_index"] = index
    return events


def bind_witness_queries(events: list[dict[str, Any]], original_queries: np.ndarray, pool: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    query_rows = [np.asarray(q, dtype=np.int16).copy() for q in original_queries]
    witness_to_qid: dict[int, int] = {}
    for event in events:
        sid = event.pop("witness_stable_id", None)
        if sid is not None:
            sid = int(sid)
            if sid not in witness_to_qid:
                witness_to_qid[sid] = len(query_rows)
                query_rows.append(pool[sid].copy())
            event["query_id"] = witness_to_qid[sid]
    return np.asarray(query_rows, dtype=np.int16), events


def replay_oracle(pool: np.ndarray, queries: np.ndarray, base_n: int, trace: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    active: set[int] = set(range(base_n))
    answers: list[dict[str, Any]] = []
    for event in trace:
        op = event["op"]
        if op == "insert":
            sid = int(event["stable_id"])
            if sid in active:
                raise RuntimeError(f"oracle duplicate insert {sid}")
            active.add(sid)
        elif op == "delete":
            sid = int(event["stable_id"])
            if sid not in active:
                raise RuntimeError(f"oracle delete nonlive {sid}")
            active.remove(sid)
        elif op == "rebuild":
            pass
        elif op in {"knn", "range"}:
            qid = int(event["query_id"])
            if qid < 0 or qid >= len(queries):
                raise RuntimeError(f"query id out of bounds {qid}")
            if op == "knn" and has_knn_boundary_tie(pool, queries[qid], active, k):
                raise RuntimeError(f"K-boundary tie in fixture at op {event['op_index']}")
            radius_sq = event.get("radius_sq")
            results = exact_results(pool, queries[qid], active, op, k, radius_sq)
            answers.append({
                "record": "oracle_query",
                "op_index": int(event["op_index"]),
                "kind": op,
                "query_id": qid,
                "radius_sq": radius_sq,
                "active_ids_sha256": stable_set_sha256(active),
                "results": results,
            })
        else:
            raise RuntimeError(f"unknown trace op {op}")
    return answers


def materialize_bundle(root: Path, name: str, pool: np.ndarray, queries: np.ndarray, trace: list[dict[str, Any]], selection: dict[str, Any], base_n: int, k: int, data_info: dict[str, Any], roles: dict[str, Any]) -> dict[str, Any]:
    bundle = root / "inputs" / name
    if bundle.exists():
        raise RuntimeError(f"refuse overwrite {bundle}")
    bundle.mkdir(parents=True)
    trace = assign_indices(trace)
    oracle = replay_oracle(pool, queries, base_n, trace, k)
    (bundle / "pool.i16").write_bytes(np.asarray(pool, dtype="<i2").tobytes(order="C"))
    (bundle / "queries.i16").write_bytes(np.asarray(queries, dtype="<i2").tobytes(order="C"))
    np.arange(base_n, dtype="<i4").tofile(bundle / "initial_base_stable_ids.i32")
    np.arange(len(pool), dtype="<i4").tofile(bundle / "stable_id_to_pool_row.i32")
    (bundle / "trace.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in trace), encoding="utf-8")
    (bundle / "oracle_expected.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in oracle), encoding="utf-8")
    write_json(bundle / "selection_receipt.json", selection)
    metadata = {
        "schema": SCHEMA,
        "status": "CPU_PREPARED_NOT_NATIVE_EXECUTED",
        "gpu_used": False,
        "scope": "G3 Safe-C1 correctness fixture; no CUDA/native execution or performance claim",
        "data": data_info,
        "header": {"base_n": base_n, "reservoir_n": len(pool) - base_n, "pool_n": len(pool), "query_n": len(queries), "dim": int(pool.shape[1]), "k": k, "event_count": len(trace)},
        "metric_contract": {"metric": "L2", "coordinate_type": "int16", "scale": SCALE, "oracle_distance": "int64 squared L2", "knn_order": "(distance_sq, stable_id)", "range_contract": "inclusive distance_sq <= radius_sq"},
        "trace_contract": {"ops": ["insert", "delete", "knn", "range", "rebuild"], "base_delete_forbidden": True, "explicit_rebuild": True, "roles": roles},
        "native_acceptance_required": ["actual frozen GTS tree height equals expected source height", "actual strict-certificate receipts match role/leaf contracts", "real GTS range traversal receipt", "fresh stable-ID rebuild", "no active-pool full scan in production query path"],
        "selection": selection,
    }
    write_json(bundle / "metadata.json", metadata)
    files = sorted(p for p in bundle.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {"schema": "safe-c1-g3-bundle-manifest-v1", "status": "CPU_PREPARED_NOT_NATIVE_EXECUTED", "gpu_used": False, "files_sha256": {p.name: sha256_file(p) for p in files}}
    write_json(bundle / "manifest.json", manifest)
    return {"name": name, "path": str(bundle), "event_count": len(trace), "query_count": len(queries), "oracle_queries": len(oracle), "manifest_sha256": sha256_file(bundle / "manifest.json")}


def deterministic_trace(planner: Planner, leaf_cap: int, delta_threshold: int, label: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    leaf, direct_ids, receipts = planner.select_same_leaf(leaf_cap + 1)
    direct = direct_ids[:leaf_cap]
    cap_id = direct_ids[-1]
    events: list[dict[str, Any]] = []
    selection_events: list[dict[str, Any]] = []
    for sid in direct:
        rec = receipts[sid]
        selected = planner.insert(sid, "direct_same_leaf", rec, leaf)
        selection_events.append({"stable_id": sid, "role": "direct_same_leaf", "source_mirror_certificate": rec, **selected})
        events.append(make_event("insert", stable_id=sid, expected_role="direct_same_leaf", expected_leaf=leaf, witness_label="direct_live"))
    # Direct visibility.
    target = direct[0]
    events += [make_event("knn", witness_stable_id=target, witness_label="direct_must_be_visible"), make_event("range", witness_stable_id=target, radius_sq=1, witness_label="direct_must_be_visible")]
    selected = planner.insert(cap_id, "capacity_delta", receipts[cap_id], leaf)
    selection_events.append({"stable_id": cap_id, "role": "capacity_delta", "source_mirror_certificate": receipts[cap_id], **selected})
    events.append(make_event("insert", stable_id=cap_id, expected_role="capacity_delta", expected_leaf=leaf, witness_label="capacity_delta_live"))
    events += [make_event("knn", witness_stable_id=cap_id, witness_label="capacity_delta_must_be_visible"), make_event("range", witness_stable_id=cap_id, radius_sq=1, witness_label="capacity_delta_must_be_visible")]
    prior = planner.delete(target)
    events.append(make_event("delete", stable_id=target, expected_prior="direct", witness_label="direct_deleted"))
    events += [make_event("knn", witness_stable_id=target, witness_label="direct_must_be_absent"), make_event("range", witness_stable_id=target, radius_sq=1, witness_label="direct_must_be_absent")]
    prior = planner.delete(cap_id)
    events.append(make_event("delete", stable_id=cap_id, expected_prior="delta", witness_label="delta_deleted"))
    events += [make_event("knn", witness_stable_id=cap_id, witness_label="delta_must_be_absent"), make_event("range", witness_stable_id=cap_id, radius_sq=1, witness_label="delta_must_be_absent")]
    gap_ids: list[int] = []
    for index in range(delta_threshold):
        sid, receipt = planner.select_rejected_delta(require_narrow_unique=index == 0)
        selected = planner.insert(sid, "certificate_delta", receipt)
        selection_events.append({"stable_id": sid, "role": "certificate_delta", "source_mirror_certificate": receipt, **selected})
        gap_ids.append(sid)
        events.append(make_event("insert", stable_id=sid, expected_role="certificate_delta", witness_label="certificate_delta_live"))
        if index == 0:
            events += [make_event("knn", witness_stable_id=sid, witness_label="certificate_delta_must_be_visible"), make_event("range", witness_stable_id=sid, radius_sq=1, witness_label="certificate_delta_must_be_visible")]
    rebuilt = planner.rebuild()
    events.append(make_event("rebuild", expected_delta_live=delta_threshold, expected_live_ids_sha256=rebuilt["live_ids_sha256"], witness_label="fresh_stable_id_rebuild"))
    events += [make_event("knn", query_id=0, witness_label="first_post_rebuild_knn"), make_event("range", query_id=0, radius_sq=None, radius_label="mid", witness_label="first_post_rebuild_range")]
    expected_events = leaf_cap + delta_threshold + 16
    if len(events) != expected_events:
        raise AssertionError(f"deterministic trace count {len(events)} != {expected_events}")
    selection = {"selector": "cpu_source_mirror_not_native_receipt", "label": label, "initial_tree": planner.tree_versions[0], "events": selection_events, "rebuild": rebuilt, "native_preflight_required": True}
    stats = {"direct_insert": leaf_cap, "capacity_delta": 1, "certificate_delta": delta_threshold, "delete_direct": 1, "delete_delta": 1, "rebuild": 1}
    return events, selection, stats


def mixed_trace(planner: Planner, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    rng = random.Random(int(seed) ^ 0x5AFE_C1)
    updates: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    direct_live_by_epoch: list[int] = []

    def insert_direct(count: int) -> list[int]:
        values: list[int] = []
        for _ in range(count):
            sid, leaf, cert = planner.select_direct()
            selected = planner.insert(sid, "direct", cert, leaf)
            values.append(sid)
            updates.append(make_event("insert", stable_id=sid, expected_role="direct"))
            receipts.append({"stable_id": sid, "role": "direct", "source_mirror_certificate": cert, **selected})
        return values

    def insert_delta(count: int) -> list[int]:
        values: list[int] = []
        for _ in range(count):
            sid, cert = planner.select_rejected_delta()
            selected = planner.insert(sid, "certificate_delta", cert)
            values.append(sid)
            updates.append(make_event("insert", stable_id=sid, expected_role="certificate_delta"))
            receipts.append({"stable_id": sid, "role": "certificate_delta", "source_mirror_certificate": cert, **selected})
        return values

    # Two epochs force two fresh trees (16 live certificate deltas) while the
    # third epoch retains mutable state for deletion coverage.
    for epoch in range(2):
        direct = insert_direct(56)
        insert_delta(8)
        delete_ids = rng.sample(direct, 30)
        for sid in delete_ids:
            planner.delete(sid)
            updates.append(make_event("delete", stable_id=sid, expected_prior="direct"))
        rebuilt = planner.rebuild()
        updates.append(make_event("rebuild", expected_delta_live=8, expected_live_ids_sha256=rebuilt["live_ids_sha256"], witness_label=f"mixed_epoch_{epoch}_rebuild"))
        direct_live_by_epoch.extend([sid for sid in direct if sid not in delete_ids])
    direct = insert_direct(30)
    delta = insert_delta(2)
    delete_direct = rng.sample(direct, 17)
    for sid in delete_direct:
        planner.delete(sid)
        updates.append(make_event("delete", stable_id=sid, expected_prior="direct"))
    planner.delete(delta[0])
    updates.append(make_event("delete", stable_id=delta[0], expected_prior="delta"))

    count = {"insert": sum(1 for x in updates if x["op"] == "insert"), "delete": sum(1 for x in updates if x["op"] == "delete"), "rebuild": sum(1 for x in updates if x["op"] == "rebuild")}
    if count != {"insert": 160, "delete": 78, "rebuild": 2}:
        raise AssertionError(f"mixed update counts wrong: {count}")

    # Preserve update order. Insert one forced exact query pair after each
    # rebuild; distribute the other 268 query operations deterministically.
    events: list[dict[str, Any]] = []
    remaining_knn, remaining_range = 136, 136
    upcoming_rebuilds = sum(1 for update in updates if update["op"] == "rebuild")
    for update in updates:
        events.append(update)
        if update["op"] == "rebuild":
            # Reserve one KNN and one range operation for *each* upcoming
            # rebuild before emitting arbitrary queries, so later forced pairs
            # never push the 512-event contract over budget.
            if remaining_knn <= 0 or remaining_range <= 0:
                raise AssertionError("post-rebuild query budget exhausted")
            events.append(make_event("knn", query_id=rng.randrange(128), witness_label="mixed_first_post_rebuild_knn"))
            events.append(make_event("range", query_id=rng.randrange(128), radius_sq=None, radius_label="wide", witness_label="mixed_first_post_rebuild_range"))
            remaining_knn -= 1
            remaining_range -= 1
            upcoming_rebuilds -= 1
        # 0--3 arbitrary queries after ordinary updates. This does not affect
        # state selection and keeps the event stream update/query mixed.
        if update["op"] != "rebuild":
            for _ in range(rng.randrange(4)):
                available_knn = remaining_knn - upcoming_rebuilds
                available_range = remaining_range - upcoming_rebuilds
                if available_knn <= 0 and available_range <= 0:
                    break
                if available_range <= 0 or (available_knn > 0 and rng.random() < 0.5):
                    events.append(make_event("knn", query_id=rng.randrange(128)))
                    remaining_knn -= 1
                else:
                    events.append(make_event("range", query_id=rng.randrange(128), radius_sq=None, radius_label="mid" if rng.random() < 0.5 else "wide"))
                    remaining_range -= 1
    if upcoming_rebuilds != 0:
        raise AssertionError("rebuild accounting mismatch")
    while remaining_knn > 0 or remaining_range > 0:
        if remaining_knn > 0 and (remaining_range <= 0 or rng.random() < 0.5):
            events.append(make_event("knn", query_id=rng.randrange(128)))
            remaining_knn -= 1
        else:
            events.append(make_event("range", query_id=rng.randrange(128), radius_sq=None, radius_label="mid" if rng.random() < 0.5 else "wide"))
            remaining_range -= 1
    if len(events) != 512:
        raise AssertionError(f"mixed trace must have 512 physical events, got {len(events)}")
    roles = {"direct_insert": 142, "certificate_delta": 18, "rebuild": 2, "no_base_delete": True}
    selection = {"selector": "cpu_source_mirror_not_native_receipt", "seed": seed, "initial_tree": planner.tree_versions[0], "tree_versions": planner.tree_versions, "events": receipts, "native_preflight_required": True}
    return events, selection, roles


def replace_range_radius(events: list[dict[str, Any]], mid_radius_sq: int, wide_radius_sq: int) -> None:
    for event in events:
        if event.get("op") != "range" or event.get("radius_sq") is not None:
            continue
        label = event.get("radius_label", "mid")
        event["radius_sq"] = int(wide_radius_sq if label == "wide" else mid_radius_sq)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-fbin", type=Path, required=True)
    parser.add_argument("--query-fbin", type=Path, required=True)
    parser.add_argument("--tree-header", type=Path, required=True)
    parser.add_argument("--source-g1", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260727, 20260728, 20260729])
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"root missing: {root}")
    if any((root / x).exists() for x in ("inputs/g3_sift4k_capacity_l8_d8", "preflight/cpu_fixture_summary.json")):
        raise SystemExit("refuse to overwrite existing G3 fixtures")
    tree_cfg = parse_gts_tree_config(args.tree_header.resolve())
    base_rows, base_dim, _ = read_fbin_layout(args.base_fbin)
    query_rows, query_dim, _ = read_fbin_layout(args.query_fbin)
    if base_dim != 128 or query_dim != 128 or base_dim != query_dim:
        raise SystemExit(f"expected SIFT128 fbin, saw base={base_rows}x{base_dim} query={query_rows}x{query_dim}")
    if base_rows < 49152 or query_rows < 128:
        raise SystemExit("insufficient SIFT fbin rows for G3 N=32768/reservoir=N/2/Q=128")
    source_info = {
        "base_fbin": {"path": str(args.base_fbin.resolve()), "sha256": sha256_file(args.base_fbin.resolve()), "rows": base_rows, "dim": base_dim},
        "query_fbin": {"path": str(args.query_fbin.resolve()), "sha256": sha256_file(args.query_fbin.resolve()), "rows": query_rows, "dim": query_dim},
        "tree_source": tree_cfg,
        "g1_source": {"path": str(args.source_g1.resolve()), "sha256": sha256_file(args.source_g1.resolve())},
        "gpu_used": False,
    }
    base_mmap = fbin_memmap(args.base_fbin.resolve())
    query_mmap = fbin_memmap(args.query_fbin.resolve())
    summaries: list[dict[str, Any]] = []
    heights: dict[str, Any] = {}
    for n in (4096, 32768):
        reservoir = n // 2
        pool = quantize_f32(np.asarray(base_mmap[: n + reservoir]), f"SIFT base slice N={n}")
        queries_base = quantize_f32(np.asarray(query_mmap[:128]), "SIFT query first128")
        height = expected_gts_height(tree_cfg["tree_order"], tree_cfg["max_size"], n)
        heights[str(n)] = {"n": n, "reservoir": reservoir, "expected_source_height": height, "pool_sha256": hashlib.sha256(pool.tobytes(order="C")).hexdigest(), "base_slice_sha256": hashlib.sha256(pool[:n].tobytes(order="C")).hexdigest(), "query128_sha256": hashlib.sha256(queries_base.tobytes(order="C")).hexdigest()}
        data_info = {**source_info, "slice": {"base_n": n, "reservoir_n": reservoir, "pool_rows": n + reservoir, "query_rows_source": 128, "pool_i16_sha256": heights[str(n)]["pool_sha256"], "query128_i16_sha256": heights[str(n)]["query128_sha256"]}, "height_preflight": heights[str(n)]}
        # Standard deterministic L=8,D=8 for both distinct source-derived tree heights.
        standard = Planner(pool, n, tree_cfg["tree_order"], tree_cfg["max_size"], leaf_cap=8, delta_threshold=8)
        trace, selection, roles = deterministic_trace(standard, 8, 8, f"sift{n}_capacity_l8_d8")
        queries, trace = bind_witness_queries(trace, queries_base, pool)
        mid = choose_nonboundary_radius_sq(pool, queries_base[0], range(n), 0.10)
        wide = choose_nonboundary_radius_sq(pool, queries_base[0], range(n), 0.50)
        replace_range_radius(trace, mid, wide)
        summaries.append(materialize_bundle(root, f"g3_sift{n}_capacity_l8_d8", pool, queries, trace, selection, n, 10, data_info, roles))
        # One small-capacity branch stress at 4K covers range+rebuild as G1B did not.
        if n == 4096:
            stress = Planner(pool, n, tree_cfg["tree_order"], tree_cfg["max_size"], leaf_cap=2, delta_threshold=3)
            trace, selection, roles = deterministic_trace(stress, 2, 3, "sift4096_branchstress_l2_d3")
            queries, trace = bind_witness_queries(trace, queries_base, pool)
            replace_range_radius(trace, mid, wide)
            summaries.append(materialize_bundle(root, "g3_sift4096_branchstress_l2_d3", pool, queries, trace, selection, n, 10, data_info, roles))
        # Three 512-event mixed traces per scale.  Each contains explicit
        # rebuild controls, so query counts stay 136+136 with 512 total events.
        for seed in args.seeds:
            mixed = Planner(pool, n, tree_cfg["tree_order"], tree_cfg["max_size"], leaf_cap=8, delta_threshold=8)
            trace, selection, roles = mixed_trace(mixed, seed)
            replace_range_radius(trace, mid, wide)
            summaries.append(materialize_bundle(root, f"g3_sift{n}_mixed_seed{seed}", pool, queries_base, trace, selection, n, 10, data_info, roles))
    if heights["4096"]["expected_source_height"] == heights["32768"]["expected_source_height"]:
        raise RuntimeError("scale pair does not change source-derived GTS height")
    summary = {"schema": "safe-c1-g3-cpu-fixture-summary-v1", "status": "PASS_CPU_ONLY_PREPARATION", "gpu_used": False, "scope": "CPU/source fixture only; no native GTS execution/performance result", "source": source_info, "height_preflight": heights, "bundles": summaries, "required_future_native_gates": ["runtime height receipt matches source-derived preflight", "actual selector receipt binds each expected role", "range receipt is real GTS traversal", "fresh stable-ID rebuild", "independent exact validator pass"]}
    write_json(root / "preflight" / "cpu_fixture_summary.json", summary)
    print(json.dumps({"status": summary["status"], "gpu_used": False, "bundle_count": len(summaries), "height_4096": heights["4096"]["expected_source_height"], "height_32768": heights["32768"]["expected_source_height"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
