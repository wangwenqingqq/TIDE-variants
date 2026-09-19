#!/usr/bin/env python3
"""Stable-ID Safe-C1 reference implementation and exact-oracle experiment harness.

This harness is intentionally separate from the archival GTS update path.  It
implements the Safe-C1 contract without legacy logical-rank IDs: frozen radial
intervals, a bounded per-leaf sidecar, an exact global delta buffer, immutable
source vectors, rebuild materialisation, and per-query exact range/top-k oracle
checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from heapq import heappop, heappush, heapreplace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

EPS = 1e-10
BASE, EXTENSION, DELTA, INACTIVE = 1, 2, 3, 0


def json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def l2(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


@dataclass
class Node:
    nid: int
    pivot: int
    base_ids: list[int]
    children: list[tuple[float, float, int]] = field(default_factory=list)
    ext_ids: list[int] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


class SafeC1Index:
    """Frozen metric tree with Safe-C1 sidecar and exact global delta buffer."""

    def __init__(
        self,
        pool: np.ndarray,
        base_ids: Iterable[int],
        *,
        fanout: int,
        base_leaf_size: int,
        leaf_extension_cap: int,
        delta_capacity: int,
        certificate_margin: float,
        allow_direct: bool,
    ) -> None:
        self.pool = np.asarray(pool, dtype=np.float64)
        self.n_total, self.dim = self.pool.shape
        self.fanout = int(fanout)
        self.base_leaf_size = int(base_leaf_size)
        self.leaf_extension_cap = int(leaf_extension_cap)
        self.delta_capacity = int(delta_capacity)
        self.certificate_margin = float(certificate_margin)
        self.allow_direct = bool(allow_direct)
        self.active = np.zeros(self.n_total, dtype=bool)
        self.kind = np.full(self.n_total, INACTIVE, dtype=np.int8)
        self.nodes: list[Node] = []
        self.root: int = -1
        self.delta_ids: list[int] = []
        self.tree_version = 0
        self.rebuilds = 0
        self.frozen_digest = ""
        self._materialize_tree(list(map(int, base_ids)), initial=True)

    def _new_node(self, pivot: int, base_ids: list[int]) -> Node:
        node = Node(nid=len(self.nodes), pivot=int(pivot), base_ids=[int(x) for x in base_ids])
        self.nodes.append(node)
        return node

    def _build(self, ids: list[int]) -> int:
        if not ids:
            raise ValueError("cannot build an empty subtree")
        pivot = int(ids[0])
        node = self._new_node(pivot, ids)
        if len(ids) <= self.base_leaf_size:
            return node.nid
        ids_arr = np.asarray(ids, dtype=np.int64)
        distances = np.linalg.norm(self.pool[ids_arr] - self.pool[pivot], axis=1)
        order = np.lexsort((ids_arr, distances))
        sorted_ids = ids_arr[order]
        sorted_distances = distances[order]
        chunks = np.array_split(np.arange(len(sorted_ids)), self.fanout)
        for positions in chunks:
            if len(positions) == 0:
                continue
            child_ids = [int(x) for x in sorted_ids[positions]]
            lo = float(sorted_distances[positions[0]])
            hi = float(sorted_distances[positions[-1]])
            child_nid = self._build(child_ids)
            node.children.append((lo, hi, child_nid))
        node.base_ids = []
        return node.nid

    def _digest_tree(self) -> str:
        payload = []
        for node in self.nodes:
            payload.append(
                {
                    "nid": node.nid,
                    "pivot": node.pivot,
                    "leaf": node.is_leaf,
                    "base": node.base_ids if node.is_leaf else [],
                    "children": [[round(lo, 12), round(hi, 12), cid] for lo, hi, cid in node.children],
                }
            )
        return sha256_bytes(canonical_bytes(payload))

    def _materialize_tree(self, active_ids: list[int], *, initial: bool = False) -> None:
        active_ids = sorted(map(int, active_ids))
        if not active_ids:
            raise ValueError("Safe-C1 requires a nonempty active set after rebuild")
        self.nodes = []
        self.root = self._build(active_ids)
        self.delta_ids = []
        self.kind.fill(INACTIVE)
        self.active.fill(False)
        self.active[np.asarray(active_ids, dtype=np.int64)] = True
        self.kind[np.asarray(active_ids, dtype=np.int64)] = BASE
        self.frozen_digest = self._digest_tree()
        if not initial:
            self.rebuilds += 1
            self.tree_version += 1

    def _check_frozen(self, before: str, rebuilt: bool) -> None:
        current = self._digest_tree()
        if rebuilt:
            if current == before:
                raise AssertionError("rebuild did not produce a new frozen tree")
            self.frozen_digest = current
        else:
            if current != before or current != self.frozen_digest:
                raise AssertionError("Safe-C1 mutated a frozen tree interval or base layout")

    def probe_certificate(self, stable_id: int) -> dict[str, Any]:
        """Return a frozen root-to-leaf certificate or an exact fallback reason."""
        stable_id = int(stable_id)
        x = self.pool[stable_id]
        nid = self.root
        certificate: list[dict[str, Any]] = []
        while True:
            node = self.nodes[nid]
            if node.is_leaf:
                if len(node.ext_ids) >= self.leaf_extension_cap:
                    return {"ok": False, "reason": "leaf_full", "leaf": nid, "certificate": certificate}
                return {"ok": True, "reason": "certified", "leaf": nid, "certificate": certificate}
            d = l2(x, self.pool[node.pivot])
            matches: list[tuple[float, float, int]] = []
            for lo, hi, child in node.children:
                # Strict interior membership rejects numerical boundaries.  The
                # exact delta fallback handles every rejected candidate.
                if d > lo + self.certificate_margin and d < hi - self.certificate_margin:
                    matches.append((lo, hi, child))
            if len(matches) != 1:
                return {
                    "ok": False,
                    "reason": "gap_or_boundary" if len(matches) == 0 else "overlap_or_ambiguous",
                    "leaf": None,
                    "certificate": certificate,
                    "distance": d,
                    "matching_children": len(matches),
                }
            lo, hi, child = matches[0]
            certificate.append({"parent": nid, "child": child, "lo": lo, "hi": hi, "distance": d})
            nid = child

    def insert(self, stable_id: int) -> dict[str, Any]:
        stable_id = int(stable_id)
        if stable_id < 0 or stable_id >= self.n_total or self.active[stable_id]:
            raise ValueError(f"invalid insert stable ID {stable_id}")
        before = self._digest_tree()
        probe = self.probe_certificate(stable_id) if self.allow_direct else {"ok": False, "reason": "buffer_only"}
        self.active[stable_id] = True
        rebuilt = False
        if probe.get("ok"):
            leaf = int(probe["leaf"])
            self.nodes[leaf].ext_ids.append(stable_id)
            self.kind[stable_id] = EXTENSION
            policy = "direct"
        else:
            self.delta_ids.append(stable_id)
            self.kind[stable_id] = DELTA
            policy = "delta"
        if len(self.delta_ids) >= self.delta_capacity:
            live = np.flatnonzero(self.active).astype(int).tolist()
            self._materialize_tree(live)
            rebuilt = True
        self._check_frozen(before, rebuilt)
        return {
            "policy": policy,
            "reason": probe.get("reason"),
            "certificate_depth": len(probe.get("certificate", [])),
            "rebuild": rebuilt,
            "tree_version": self.tree_version,
        }

    def delete(self, stable_id: int) -> dict[str, Any]:
        stable_id = int(stable_id)
        if stable_id < 0 or stable_id >= self.n_total or not self.active[stable_id]:
            raise ValueError(f"invalid delete stable ID {stable_id}")
        before = self._digest_tree()
        old_kind = int(self.kind[stable_id])
        if old_kind == EXTENSION:
            for node in self.nodes:
                if stable_id in node.ext_ids:
                    node.ext_ids.remove(stable_id)
                    break
            else:
                raise AssertionError("extension stable ID missing from sidecar")
        elif old_kind == DELTA:
            self.delta_ids.remove(stable_id)
        elif old_kind != BASE:
            raise AssertionError("live ID lacks a placement kind")
        self.active[stable_id] = False
        self.kind[stable_id] = INACTIVE
        self._check_frozen(before, rebuilt=False)
        return {"policy": {BASE: "base", EXTENSION: "direct", DELTA: "delta"}[old_kind], "rebuild": False}

    def _range_candidates(self, query: np.ndarray, radius: float) -> tuple[list[int], int]:
        candidates: list[int] = []
        stack = [self.root]
        visited = 0
        while stack:
            nid = stack.pop()
            node = self.nodes[nid]
            visited += 1
            if node.is_leaf:
                candidates.extend(i for i in node.base_ids if self.active[i])
                candidates.extend(i for i in node.ext_ids if self.active[i])
                continue
            dqp = l2(query, self.pool[node.pivot])
            for lo, hi, child in node.children:
                if dqp + radius + EPS >= lo and dqp - radius - EPS <= hi:
                    stack.append(child)
        candidates.extend(i for i in self.delta_ids if self.active[i])
        return candidates, visited

    def range_query(self, query: np.ndarray, radius: float) -> tuple[list[tuple[int, float]], dict[str, int]]:
        ids, visited = self._range_candidates(query, radius)
        if ids:
            arr = np.asarray(ids, dtype=np.int64)
            dist = np.linalg.norm(self.pool[arr] - query, axis=1)
            keep = dist <= radius + EPS
            out = [(int(i), float(d)) for i, d, yes in zip(arr, dist, keep) if yes]
            out.sort(key=lambda x: (x[1], x[0]))
        else:
            out = []
        return out, {"candidate_evals": len(ids), "visited_nodes": visited, "delta_evals": sum(self.kind[i] == DELTA for i in ids)}

    def knn_query(self, query: np.ndarray, k: int) -> tuple[list[tuple[int, float]], dict[str, int]]:
        frontier: list[tuple[float, int, int]] = [(0.0, 0, self.root)]
        serial = 1
        # Heap root is current worst result: largest distance, then largest ID.
        best: list[tuple[float, int, int, float]] = []
        evaluated = 0
        visited = 0

        def worst_threshold() -> float:
            return -best[0][0] if len(best) >= k else float("inf")

        def consider(stable_id: int) -> None:
            nonlocal evaluated
            if not self.active[stable_id]:
                return
            evaluated += 1
            d = l2(query, self.pool[stable_id])
            item = (-d, -stable_id, stable_id, d)
            if len(best) < k:
                heappush(best, item)
            else:
                worst = best[0]
                if (d, stable_id) < (-worst[0], -worst[1]):
                    heapreplace(best, item)

        while frontier:
            lower, _, nid = heappop(frontier)
            if lower > worst_threshold() + EPS:
                break
            node = self.nodes[nid]
            visited += 1
            if node.is_leaf:
                for stable_id in node.base_ids:
                    consider(stable_id)
                for stable_id in node.ext_ids:
                    consider(stable_id)
                continue
            dqp = l2(query, self.pool[node.pivot])
            for lo, hi, child in node.children:
                lower_child = max(0.0, lo - dqp, dqp - hi)
                if lower_child <= worst_threshold() + EPS:
                    heappush(frontier, (lower_child, serial, child))
                    serial += 1
        # Delta is globally visible and therefore must be checked regardless of
        # tree bounds; scanning it after the tree can only add work, never lose a result.
        delta_before = evaluated
        for stable_id in self.delta_ids:
            consider(stable_id)
        out = [(sid, d) for _, _, sid, d in best]
        out.sort(key=lambda x: (x[1], x[0]))
        return out, {"candidate_evals": evaluated, "visited_nodes": visited, "delta_evals": evaluated - delta_before}

    def oracle_range(self, query: np.ndarray, radius: float) -> list[tuple[int, float]]:
        ids = np.flatnonzero(self.active)
        dist = np.linalg.norm(self.pool[ids] - query, axis=1)
        out = [(int(i), float(d)) for i, d, yes in zip(ids, dist, dist <= radius + EPS) if yes]
        out.sort(key=lambda x: (x[1], x[0]))
        return out

    def oracle_knn(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        ids = np.flatnonzero(self.active)
        dist = np.linalg.norm(self.pool[ids] - query, axis=1)
        order = np.lexsort((ids, dist))[:k]
        return [(int(ids[i]), float(dist[i])) for i in order]

    def active_set_digest(self) -> str:
        return sha256_bytes(canonical_bytes(np.flatnonzero(self.active).astype(int).tolist()))

    def state_digest(self) -> str:
        payload = {
            "active": np.flatnonzero(self.active).astype(int).tolist(),
            "kind": self.kind.astype(int).tolist(),
            "delta": list(map(int, self.delta_ids)),
            "extensions": {str(n.nid): list(map(int, n.ext_ids)) for n in self.nodes if n.ext_ids},
            "tree": self.frozen_digest,
            "version": self.tree_version,
        }
        return sha256_bytes(canonical_bytes(payload))


def make_pool_and_queries(seed: int, base_n: int, reservoir_n: int, dim: int, query_n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    clusters = 8
    centers = rng.normal(0.0, 5.0, size=(clusters, dim))
    assignments = rng.integers(0, clusters, size=base_n + reservoir_n)
    pool = centers[assignments] + rng.normal(0.0, 0.35, size=(base_n + reservoir_n, dim))
    q_assignments = rng.integers(0, clusters, size=query_n)
    queries = centers[q_assignments] + rng.normal(0.0, 0.40, size=(query_n, dim))
    return pool.astype(np.float64), queries.astype(np.float64)


def make_normal_trace(seed: int, base_n: int, reservoir_n: int, query_n: int, events: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed ^ 0x5AFE_C1)
    active = set(range(base_n))
    available = list(range(base_n, base_n + reservoir_n))
    rng.shuffle(available)
    trace: list[dict[str, Any]] = []
    for op_index in range(events):
        roll = float(rng.random())
        if (roll < 0.32 and available) or len(active) < max(64, base_n // 4):
            stable_id = int(available.pop())
            active.add(stable_id)
            trace.append({"op_index": op_index, "op": "insert", "id": stable_id})
        elif roll < 0.46 and len(active) > max(64, base_n // 4):
            choices = np.asarray(sorted(active), dtype=np.int64)
            stable_id = int(choices[int(rng.integers(len(choices)))])
            active.remove(stable_id)
            trace.append({"op_index": op_index, "op": "delete", "id": stable_id})
        else:
            trace.append(
                {
                    "op_index": op_index,
                    "op": "knn" if rng.random() < 0.5 else "range",
                    "query_id": int(rng.integers(query_n)),
                }
            )
    return trace


def _probe_groups(index: SafeC1Index, candidates: Iterable[int]) -> tuple[dict[int, list[int]], list[int]]:
    groups: dict[int, list[int]] = {}
    gaps: list[int] = []
    for stable_id in candidates:
        probe = index.probe_certificate(int(stable_id))
        if probe.get("ok"):
            groups.setdefault(int(probe["leaf"]), []).append(int(stable_id))
        elif probe.get("reason") in {"gap_or_boundary", "overlap_or_ambiguous"}:
            gaps.append(int(stable_id))
    return groups, gaps


def make_adversarial_trace(
    pool: np.ndarray,
    queries: np.ndarray,
    *,
    base_n: int,
    reservoir_n: int,
    fanout: int,
    base_leaf_size: int,
    leaf_extension_cap: int,
    delta_capacity: int,
    certificate_margin: float,
) -> list[dict[str, Any]]:
    planner = SafeC1Index(
        pool,
        range(base_n),
        fanout=fanout,
        base_leaf_size=base_leaf_size,
        leaf_extension_cap=leaf_extension_cap,
        delta_capacity=delta_capacity,
        certificate_margin=certificate_margin,
        allow_direct=True,
    )
    reservoir = list(range(base_n, base_n + reservoir_n))
    groups, gaps = _probe_groups(planner, reservoir)
    group = next((g for g in groups.values() if len(g) >= leaf_extension_cap + 1), None)
    if group is None or len(gaps) < delta_capacity + 3:
        raise RuntimeError("deterministic adversarial pool lacks required interior/gap candidates")
    used: set[int] = set()
    direct = group[: leaf_extension_cap + 1]
    gap_ids = [x for x in gaps if x not in direct][: delta_capacity + 3]
    if len(gap_ids) < delta_capacity + 3:
        raise RuntimeError("insufficient independent gap candidates")
    trace: list[dict[str, Any]] = []

    def add(op: str, **kwargs: Any) -> None:
        trace.append({"op_index": len(trace), "op": op, **kwargs})

    # Two certified inserts fill one sidecar leaf; the next one must fall back.
    for stable_id in direct[:leaf_extension_cap]:
        planner.insert(stable_id); used.add(stable_id); add("insert", id=stable_id, label="interior_certified")
    planner.insert(direct[-1]); used.add(direct[-1]); add("insert", id=direct[-1], label="leaf_full_fallback")
    # Delete one direct and one delta object before the rebuild to cover both paths.
    planner.delete(direct[0]); add("delete", id=direct[0], label="delete_after_direct")
    planner.delete(direct[-1]); add("delete", id=direct[-1], label="delete_after_delta")
    # Fill the exact delta with certificate failures; final insertion rebuilds.
    for stable_id in gap_ids[:delta_capacity]:
        planner.insert(stable_id); used.add(stable_id); add("insert", id=stable_id, label="gap_or_boundary_fallback")
    # Post-rebuild query events use independent query vectors.
    for qid in range(min(12, len(queries))):
        add("knn" if qid % 2 == 0 else "range", query_id=qid, label="post_update_oracle")
    return trace


def compare_answers(actual: list[tuple[int, float]], oracle: list[tuple[int, float]], *, kind: str) -> dict[str, Any]:
    actual_ids = [x[0] for x in actual]
    oracle_ids = [x[0] for x in oracle]
    if kind == "range":
        missing = sorted(set(oracle_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(oracle_ids))
        ordered = not missing and not extra
    else:
        missing = [i for i, (a, b) in enumerate(zip(actual_ids, oracle_ids)) if a != b]
        if len(actual_ids) != len(oracle_ids):
            missing.extend(range(min(len(actual_ids), len(oracle_ids)), max(len(actual_ids), len(oracle_ids))))
        extra = []
        ordered = not missing
    max_abs_dist = 0.0
    if ordered:
        for (_, da), (_, db) in zip(actual, oracle):
            max_abs_dist = max(max_abs_dist, abs(da - db))
    return {
        "pass": bool(ordered and max_abs_dist <= 1e-8),
        "missing": missing,
        "extra": extra,
        "max_abs_distance_error": max_abs_dist,
    }


def execute_trace(
    index: SafeC1Index,
    trace: list[dict[str, Any]],
    queries: np.ndarray,
    *,
    radius: float,
    k: int,
    audit_qids: list[int],
    output: Path,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    records = output.open("w", encoding="utf-8")
    counts = {"insert": 0, "delete": 0, "knn": 0, "range": 0, "direct": 0, "delta": 0, "rebuild": 0}
    checks = {"knn": 0, "range": 0, "knn_mismatch": 0, "range_mismatch": 0, "range_missing": 0, "range_extra": 0}
    candidate_stats = {"knn": [], "range": [], "delta_knn": [], "delta_range": []}
    labels: dict[str, int] = {}

    def validate(kind: str, qid: int, origin: str, op_index: int) -> None:
        query = queries[qid]
        if kind == "knn":
            actual, cost = index.knn_query(query, k)
            oracle = index.oracle_knn(query, k)
        else:
            actual, cost = index.range_query(query, radius)
            oracle = index.oracle_range(query, radius)
        comparison = compare_answers(actual, oracle, kind=kind)
        checks[kind] += 1
        if not comparison["pass"]:
            checks[f"{kind}_mismatch"] += 1
            checks["range_missing"] += len(comparison["missing"]) if kind == "range" else 0
            checks["range_extra"] += len(comparison["extra"]) if kind == "range" else 0
        candidate_stats[kind].append(cost["candidate_evals"])
        candidate_stats[f"delta_{kind}"].append(cost["delta_evals"])
        rec = {
            "record": "query",
            "origin": origin,
            "op_index": op_index,
            "kind": kind,
            "query_id": qid,
            "live_count": int(index.active.sum()),
            "actual": [[sid, dist] for sid, dist in actual],
            "oracle": [[sid, dist] for sid, dist in oracle],
            "cost": cost,
            "comparison": comparison,
            "state_hash": index.state_digest(),
        }
        records.write(json.dumps(rec, default=json_default, sort_keys=True) + "\n")
        if not comparison["pass"]:
            raise AssertionError(f"oracle mismatch at trace op {op_index}: {kind}")

    begin = time.perf_counter()
    for event in trace:
        op = event["op"]
        op_index = int(event["op_index"])
        if op == "insert":
            result = index.insert(int(event["id"]))
            counts["insert"] += 1
            counts[result["policy"]] += 1
            counts["rebuild"] += int(result["rebuild"])
            labels[event.get("label", result["reason"] or "unlabeled")] = labels.get(event.get("label", result["reason"] or "unlabeled"), 0) + 1
            records.write(json.dumps({"record": "update", "event": event, "result": result, "state_hash": index.state_digest()}, default=json_default, sort_keys=True) + "\n")
            for qid in audit_qids:
                validate("knn", qid, "post_update_audit", op_index)
                validate("range", qid, "post_update_audit", op_index)
        elif op == "delete":
            result = index.delete(int(event["id"]))
            counts["delete"] += 1
            labels[event.get("label", "delete")] = labels.get(event.get("label", "delete"), 0) + 1
            records.write(json.dumps({"record": "update", "event": event, "result": result, "state_hash": index.state_digest()}, default=json_default, sort_keys=True) + "\n")
            for qid in audit_qids:
                validate("knn", qid, "post_update_audit", op_index)
                validate("range", qid, "post_update_audit", op_index)
        elif op in {"knn", "range"}:
            counts[op] += 1
            validate(op, int(event["query_id"]), "trace", op_index)
        else:
            raise ValueError(f"unsupported event {op}")
    elapsed = time.perf_counter() - begin
    records.close()
    return {
        "counts": counts,
        "checks": checks,
        "candidate_evals": {key: float(np.mean(values)) if values else 0.0 for key, values in candidate_stats.items()},
        "labels": labels,
        "elapsed_s": elapsed,
        "final_live_count": int(index.active.sum()),
        "final_state_hash": index.state_digest(),
        "final_active_set_hash": index.active_set_digest(),
        "frozen_tree_hash": index.frozen_digest,
        "tree_version": index.tree_version,
    }


def gpu_inventory() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return {"used": False, "devices": [line.strip() for line in raw.splitlines() if line.strip()]}
    except Exception as exc:  # pragma: no cover - host dependent
        return {"used": False, "error": str(exc)}


def run_one(args: argparse.Namespace, scenario: str, seed: int, root: Path, source_manifest: str | None) -> dict[str, Any]:
    pool, queries = make_pool_and_queries(seed, args.base_n, args.reservoir_n, args.dim, args.query_n)
    policy = dict(
        fanout=args.fanout,
        base_leaf_size=args.base_leaf_size,
        leaf_extension_cap=args.leaf_extension_cap if scenario == "normal" else args.adversarial_leaf_extension_cap,
        delta_capacity=args.delta_capacity if scenario == "normal" else args.adversarial_delta_capacity,
        certificate_margin=args.certificate_margin if scenario == "normal" else args.adversarial_certificate_margin,
    )
    if scenario == "normal":
        trace = make_normal_trace(seed, args.base_n, args.reservoir_n, args.query_n, args.events)
    else:
        trace = make_adversarial_trace(pool, queries, base_n=args.base_n, reservoir_n=args.reservoir_n, **policy)
    run_dir = root / scenario / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "trace.jsonl").write_text("\n".join(json.dumps(x, sort_keys=True) for x in trace) + "\n", encoding="utf-8")
    np.save(run_dir / "pool.npy", pool)
    np.save(run_dir / "queries.npy", queries)
    initial_ids = list(range(args.base_n))
    safe = SafeC1Index(pool, initial_ids, allow_direct=True, **policy)
    buffer_only = SafeC1Index(pool, initial_ids, allow_direct=False, **policy)
    audit_qids = list(range(min(args.audit_queries, args.query_n)))
    safe_result = execute_trace(safe, trace, queries, radius=args.radius, k=args.k, audit_qids=audit_qids, output=run_dir / "safe_answers.jsonl")
    buffer_result = execute_trace(buffer_only, trace, queries, radius=args.radius, k=args.k, audit_qids=audit_qids, output=run_dir / "buffer_answers.jsonl")
    same_active_set = safe.active_set_digest() == buffer_only.active_set_digest()
    if not same_active_set:
        raise AssertionError("Safe-C1 and buffer-only diverged in stable active-set state")
    trace_hash = sha256_file(run_dir / "trace.jsonl")
    summary = {
        "scenario": scenario,
        "seed": seed,
        "policy": policy,
        "data": {"metric": "L2", "dimension": args.dim, "base_n": args.base_n, "reservoir_n": args.reservoir_n, "query_n": args.query_n},
        "trace": {"events": len(trace), "sha256": trace_hash},
        "safe_c1": safe_result,
        "buffer_only": buffer_result,
        "oracle_pass": safe_result["checks"]["knn_mismatch"] == 0 and safe_result["checks"]["range_mismatch"] == 0 and buffer_result["checks"]["knn_mismatch"] == 0 and buffer_result["checks"]["range_mismatch"] == 0,
        "same_final_active_set": same_active_set,
        "final_active_set_hash": safe.active_set_digest(),
        "source_manifest_sha256": source_manifest,
        "files": {name: sha256_file(run_dir / name) for name in ["trace.jsonl", "pool.npy", "queries.npy", "safe_answers.jsonl", "buffer_answers.jsonl"]},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260727, 20260728, 20260729, 20260730, 20260731])
    p.add_argument("--base-n", type=int, default=4096)
    p.add_argument("--reservoir-n", type=int, default=2048)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--query-n", type=int, default=128)
    p.add_argument("--events", type=int, default=256)
    p.add_argument("--fanout", type=int, default=8)
    p.add_argument("--base-leaf-size", type=int, default=32)
    p.add_argument("--leaf-extension-cap", type=int, default=8)
    p.add_argument("--delta-capacity", type=int, default=16)
    p.add_argument("--certificate-margin", type=float, default=1e-7)
    p.add_argument("--adversarial-leaf-extension-cap", type=int, default=2)
    p.add_argument("--adversarial-delta-capacity", type=int, default=3)
    p.add_argument("--adversarial-certificate-margin", type=float, default=1e-4)
    p.add_argument("--audit-queries", type=int, default=4)
    p.add_argument("--radius", type=float, default=3.0)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--source-manifest", type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.out}")
    args.out.mkdir(parents=True)
    source_manifest = sha256_file(args.source_manifest) if args.source_manifest and args.source_manifest.exists() else None
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results = []
    for seed in args.seeds:
        results.append(run_one(args, "normal", int(seed), args.out, source_manifest))
    # One deterministic scenario specifically exercises certificate rejection,
    # leaf capacity, delete-after-insert, and rebuild.
    results.append(run_one(args, "adversarial", int(args.seeds[0]), args.out, source_manifest))
    suite = {
        "schema": "safe-c1-oracle-v1",
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "gpu_inventory": gpu_inventory(),
        "gpu_used": False,
        "source_file_sha256": sha256_file(Path(__file__)),
        "source_manifest_sha256": source_manifest,
        "results": results,
    }
    (args.out / "suite_summary.json").write_text(json.dumps(suite, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps({"suite": str(args.out), "runs": len(results), "all_oracle_pass": all(x["oracle_pass"] for x in results)}, sort_keys=True))
    return 0 if all(x["oracle_pass"] for x in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
