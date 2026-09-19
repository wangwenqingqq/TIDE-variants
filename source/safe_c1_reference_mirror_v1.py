#!/usr/bin/env python3
"""CPU reference-policy mirror for Safe-C1 versus buffer-only dynamic search.

This is an intentionally separate, CPU-only semantic model.  It builds a
frozen radial metric tree over the base set.  An arrival is admitted to a leaf
sidecar only if a strict, unique root-to-leaf radial-envelope certificate
exists; otherwise it enters an exact global delta.  Queries use the frozen
radial tree, scan sidecars only in visited leaves, scan the delta globally,
and are checked against an exact current-live-set oracle.

It is *not* a GPU GTS implementation, a latency benchmark, or a proof of
native-GTS generality.  v1 refuses test/heldout protocols and is usable only
for development validation of the reference semantics.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import heapq
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


DEVELOPMENT_SCOPES = {"development_smoke_only", "development_calibration_only"}
INPUT_SCHEMAS = {"tide-hnsw-snapshot-protocol-v1"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(obj))


def fvecs_shape(path: Path) -> tuple[int, int]:
    size = path.stat().st_size
    with path.open("rb") as f:
        raw = f.read(4)
    if len(raw) != 4:
        raise ValueError(f"empty/truncated fvecs: {path}")
    d = int(np.frombuffer(raw, dtype="<i4")[0])
    row_bytes = 4 * (d + 1)
    if d <= 0 or size % row_bytes != 0:
        raise ValueError(f"invalid fvecs layout: {path}")
    return size // row_bytes, d


def open_fvecs(path: Path) -> np.ndarray:
    n, d = fvecs_shape(path)
    record = np.dtype([("dim", "<i4"), ("vec", "<f4", (d,))])
    data = np.memmap(path, dtype=record, mode="r", shape=(n,))
    probe = np.unique(np.asarray([0, n // 2, n - 1], dtype=np.int64))
    if not np.all(data[probe]["dim"] == d):
        raise ValueError(f"inconsistent fvec dimension: {path}")
    return data["vec"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_trace(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            if row.get("ordinal") != len(out):
                raise ValueError(f"noncanonical trace ordinal at line {line_no}")
            if row.get("op") not in {"I", "D", "Q"}:
                raise ValueError(f"unknown trace op at line {line_no}")
            out.append(row)
    return out


def stable_ids_sha256(ids: Iterable[int]) -> str:
    values = np.asarray(sorted(ids), dtype="<u8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def l2_batch(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    # Float64 is deliberate: the reference tree and oracle share one stable
    # numerical domain independent of a native GPU accumulation order.
    diff = vectors.astype(np.float64, copy=False) - query.astype(np.float64, copy=False)
    return np.einsum("ij,ij->i", diff, diff, dtype=np.float64)


def exact_topk(vectors: np.ndarray, ids: np.ndarray, query: np.ndarray, k: int) -> tuple[list[int], list[float]]:
    d = l2_batch(vectors, query)
    if len(ids) < k:
        raise ValueError("live set smaller than k")
    candidate = np.argpartition(d, k - 1)[:k]
    cutoff = float(np.max(d[candidate]))
    strict = np.flatnonzero(d < cutoff)
    equal = np.flatnonzero(d == cutoff)
    take = k - len(strict)
    equal = equal[np.argsort(ids[equal], kind="stable")[:take]]
    chosen = np.concatenate((strict, equal))
    order = np.lexsort((ids[chosen], d[chosen]))
    chosen = chosen[order]
    return [int(x) for x in ids[chosen]], [float(x) for x in d[chosen]]


@dataclasses.dataclass(frozen=True)
class Node:
    node_id: int
    pivot_id: Optional[int]
    left_id: Optional[int]
    right_id: Optional[int]
    left_lo: Optional[float]
    left_hi: Optional[float]
    right_lo: Optional[float]
    right_hi: Optional[float]
    leaf_ids: Optional[np.ndarray]

    @property
    def is_leaf(self) -> bool:
        return self.leaf_ids is not None


class FrozenRadialTree:
    """Balanced, frozen radial tree with exact triangle-inequality envelopes."""

    def __init__(self, base: np.ndarray, leaf_size: int):
        if base.ndim != 2 or len(base) == 0 or leaf_size < 2:
            raise ValueError("invalid base/leaf_size")
        self.base = np.asarray(base, dtype=np.float32)
        self.leaf_size = int(leaf_size)
        self.nodes: list[Node] = []
        self.root_id = self._build(np.arange(len(self.base), dtype=np.int64))
        self.fingerprint_sha256 = self._fingerprint()

    def _distance_to_pivot(self, ids: np.ndarray, pivot_id: int) -> np.ndarray:
        return np.sqrt(l2_batch(self.base[ids], self.base[pivot_id]))

    def _build(self, ids: np.ndarray) -> int:
        ids = np.asarray(np.sort(ids), dtype=np.int64)
        node_id = len(self.nodes)
        # reserve an index before recursive construction so child ids are stable
        self.nodes.append(Node(node_id, None, None, None, None, None, None, None, None))
        if len(ids) <= self.leaf_size:
            self.nodes[node_id] = Node(node_id, None, None, None, None, None, None, None, ids)
            return node_id
        pivot_pos = len(ids) // 2
        pivot_id = int(ids[pivot_pos])
        rest = np.concatenate((ids[:pivot_pos], ids[pivot_pos + 1:]))
        if len(rest) < 2:
            self.nodes[node_id] = Node(node_id, None, None, None, None, None, None, None, ids)
            return node_id
        radii = self._distance_to_pivot(rest, pivot_id)
        order = np.lexsort((rest, radii))
        rest = rest[order]
        radii = radii[order]
        split = len(rest) // 2
        left_ids, right_ids = rest[:split], rest[split:]
        left_r, right_r = radii[:split], radii[split:]
        if len(left_ids) == 0 or len(right_ids) == 0:
            self.nodes[node_id] = Node(node_id, None, None, None, None, None, None, None, ids)
            return node_id
        left_id = self._build(left_ids)
        right_id = self._build(right_ids)
        self.nodes[node_id] = Node(
            node_id, pivot_id, left_id, right_id,
            float(np.min(left_r)), float(np.max(left_r)),
            float(np.min(right_r)), float(np.max(right_r)), None,
        )
        return node_id

    def _fingerprint(self) -> str:
        rows: list[dict[str, Any]] = []
        for n in self.nodes:
            rows.append({
                "node_id": n.node_id, "pivot_id": n.pivot_id,
                "left_id": n.left_id, "right_id": n.right_id,
                "left_lo": n.left_lo, "left_hi": n.left_hi,
                "right_lo": n.right_lo, "right_hi": n.right_hi,
                "leaf_ids": n.leaf_ids.tolist() if n.leaf_ids is not None else None,
            })
        return hashlib.sha256(canonical_json_bytes({"leaf_size": self.leaf_size, "nodes": rows})).hexdigest()

    @staticmethod
    def _strict_inside(radius: float, lo: float, hi: float) -> bool:
        return radius > lo and radius < hi

    def certify_leaf(self, vector: np.ndarray) -> Optional[int]:
        """Return a leaf only for a unique strict envelope path; otherwise None."""
        node_id = self.root_id
        while True:
            n = self.nodes[node_id]
            if n.is_leaf:
                return node_id
            assert n.pivot_id is not None and n.left_id is not None and n.right_id is not None
            radius = float(np.sqrt(l2_batch(self.base[n.pivot_id:n.pivot_id + 1], vector)[0]))
            left_ok = self._strict_inside(radius, float(n.left_lo), float(n.left_hi))
            right_ok = self._strict_inside(radius, float(n.right_lo), float(n.right_hi))
            if left_ok == right_ok:  # both or neither: not a unique safe route
                return None
            node_id = int(n.left_id if left_ok else n.right_id)

    @staticmethod
    def _child_lb(q_to_pivot: float, lo: float, hi: float) -> float:
        return max(0.0, lo - q_to_pivot, q_to_pivot - hi)

    def search(
        self,
        query: np.ndarray,
        k: int,
        sidecars: dict[int, list[int]],
        delta_ids: list[int],
        all_vectors: np.ndarray,
        include_sidecars: bool,
    ) -> tuple[list[int], list[float], dict[str, int], list[int]]:
        """Search frozen base tree plus tiered dynamic state.

        The sidecar scan happens only on visited leaves.  The strict-envelope
        certificate ensures such an omitted leaf cannot contain a sidecar item
        closer than the current kth threshold.
        """
        # heap item = (-distance, -stable_id, stable_id, distance).  heap[0]
        # is the worst element under (distance, stable-id) ordering.
        heap: list[tuple[float, int, int, float]] = []
        visited_leaves: list[int] = []
        counts = {"base_candidates": 0, "sidecar_candidates": 0, "delta_candidates": 0, "visited_leaves": 0}

        def offer_many(ids: np.ndarray | list[int], tier: str) -> None:
            if len(ids) == 0:
                return
            arr_ids = np.asarray(ids, dtype=np.int64)
            distances = l2_batch(all_vectors[arr_ids], query)
            if tier == "base":
                counts["base_candidates"] += int(len(arr_ids))
            elif tier == "sidecar":
                counts["sidecar_candidates"] += int(len(arr_ids))
            elif tier == "delta":
                counts["delta_candidates"] += int(len(arr_ids))
            else:
                raise ValueError("unknown tier")
            for stable_id, distance in zip(arr_ids.tolist(), distances.tolist()):
                item = (-float(distance), -int(stable_id), int(stable_id), float(distance))
                if len(heap) < k:
                    heapq.heappush(heap, item)
                else:
                    worst_d, worst_neg_id, _, _ = heap[0]
                    worst_key = (-worst_d, -worst_neg_id)
                    if (float(distance), int(stable_id)) < worst_key:
                        heapq.heapreplace(heap, item)

        def tau() -> float:
            return float("inf") if len(heap) < k else -heap[0][0]

        def visit(node_id: int) -> None:
            n = self.nodes[node_id]
            if n.is_leaf:
                assert n.leaf_ids is not None
                visited_leaves.append(node_id)
                counts["visited_leaves"] += 1
                offer_many(n.leaf_ids, "base")
                if include_sidecars:
                    offer_many(sidecars.get(node_id, []), "sidecar")
                return
            assert n.pivot_id is not None and n.left_id is not None and n.right_id is not None
            offer_many([n.pivot_id], "base")
            q_to_pivot = float(np.sqrt(l2_batch(self.base[n.pivot_id:n.pivot_id + 1], query)[0]))
            children = [
                (self._child_lb(q_to_pivot, float(n.left_lo), float(n.left_hi)), int(n.left_id)),
                (self._child_lb(q_to_pivot, float(n.right_lo), float(n.right_hi)), int(n.right_id)),
            ]
            for lower_bound, child_id in sorted(children):
                # Strict > preserves tie-aware correctness: equality can still
                # improve the result by stable-id tie break.
                if lower_bound <= tau():
                    visit(child_id)

        visit(self.root_id)
        # Global delta is exact and deliberately consulted after the frozen tree.
        offer_many(delta_ids, "delta")
        answer = sorted(((distance, stable_id) for _, _, stable_id, distance in heap), key=lambda x: (x[0], x[1]))
        return [x[1] for x in answer], [x[0] for x in answer], counts, sorted(visited_leaves)


class TieredState:
    def __init__(self, policy: str, base_n: int, pool_n: int, sidecar_capacity: int, delta_bound: int, tree: FrozenRadialTree):
        if policy not in {"safe_c1", "buffer_only"}:
            raise ValueError("policy")
        self.policy = policy
        self.base_n = base_n
        self.pool_n = pool_n
        self.sidecar_capacity = sidecar_capacity
        self.delta_bound = delta_bound
        self.tree = tree
        self.active: set[int] = set()
        self.placement: dict[int, tuple[str, Optional[int]]] = {}
        self.sidecars: dict[int, list[int]] = {}
        self.delta: list[int] = []
        self.counters = {"direct_inserts": 0, "delta_inserts": 0, "dynamic_deletes": 0, "certificate_evaluations": 0, "certificate_accepts": 0, "certificate_rejects": 0, "capacity_fallbacks": 0, "rebuild_count": 0}

    def insert(self, stable_id: int, vector: np.ndarray) -> tuple[str, Optional[int], str]:
        if stable_id < self.base_n or stable_id >= self.pool_n or stable_id in self.active or stable_id in self.placement:
            raise ValueError("invalid/repeated arrival insert")
        self.active.add(stable_id)
        if self.policy == "buffer_only":
            self.delta.append(stable_id)
            self.placement[stable_id] = ("delta", None)
            self.counters["delta_inserts"] += 1
            fallback = "buffer_only"
            result = ("delta", None, fallback)
        else:
            self.counters["certificate_evaluations"] += 1
            leaf = self.tree.certify_leaf(vector)
            if leaf is None:
                self.delta.append(stable_id)
                self.placement[stable_id] = ("delta", None)
                self.counters["delta_inserts"] += 1
                self.counters["certificate_rejects"] += 1
                result = ("delta", None, "certificate_reject")
            elif len(self.sidecars.get(leaf, [])) >= self.sidecar_capacity:
                self.delta.append(stable_id)
                self.placement[stable_id] = ("delta", None)
                self.counters["delta_inserts"] += 1
                self.counters["certificate_accepts"] += 1
                self.counters["capacity_fallbacks"] += 1
                result = ("delta", leaf, "capacity_full")
            else:
                self.sidecars.setdefault(leaf, []).append(stable_id)
                self.placement[stable_id] = ("direct", leaf)
                self.counters["direct_inserts"] += 1
                self.counters["certificate_accepts"] += 1
                result = ("direct", leaf, "none")
        self._assert_invariants()
        return result

    def delete(self, stable_id: int) -> tuple[str, Optional[int]]:
        if stable_id not in self.active:
            raise ValueError("delete of inactive arrival")
        tier, leaf = self.placement[stable_id]
        if tier == "direct":
            assert leaf is not None
            values = self.sidecars.get(leaf, [])
            if values.count(stable_id) != 1:
                raise ValueError("direct delete does not identify one sidecar object")
            values.remove(stable_id)
            if not values:
                del self.sidecars[leaf]
        elif tier == "delta":
            if self.delta.count(stable_id) != 1:
                raise ValueError("delta delete does not identify one delta object")
            self.delta.remove(stable_id)
        else:
            raise ValueError("unknown placement")
        self.active.remove(stable_id)
        del self.placement[stable_id]
        self.counters["dynamic_deletes"] += 1
        self._assert_invariants()
        return tier, leaf

    def _assert_invariants(self) -> None:
        direct = set()
        for leaf, values in self.sidecars.items():
            if len(values) > self.sidecar_capacity:
                raise AssertionError("sidecar capacity violated")
            for stable_id in values:
                if stable_id in direct or stable_id not in self.active or self.placement.get(stable_id) != ("direct", leaf):
                    raise AssertionError("invalid sidecar partition")
                direct.add(stable_id)
        delta = set(self.delta)
        if len(delta) != len(self.delta) or direct.intersection(delta):
            raise AssertionError("dynamic tier duplication")
        if direct.union(delta) != self.active:
            raise AssertionError("active arrivals do not match tiers")
        for stable_id in delta:
            if self.placement.get(stable_id) != ("delta", None):
                raise AssertionError("invalid delta placement")
        if len(self.delta) >= self.delta_bound:
            raise AssertionError("forbidden rebuild threshold reached")


def require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(f"{field} mismatch: actual={actual!r}, expected={expected!r}")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def run(args: argparse.Namespace) -> None:
    protocol_dir = Path(args.protocol_dir).resolve()
    manifest_path = protocol_dir / "manifest.json"
    trace_path = protocol_dir / "trace.jsonl"
    manifest = load_json(manifest_path)
    if manifest.get("schema") not in INPUT_SCHEMAS:
        raise ValueError(f"unsupported input schema {manifest.get('schema')!r}")
    scope = manifest.get("evidence_scope")
    if scope not in DEVELOPMENT_SCOPES:
        raise ValueError("v1 mirror refuses non-development protocol; no heldout test execution is permitted")
    manifest_sha = sha256_file(manifest_path)
    trace_sha = sha256_file(trace_path)
    require_equal(trace_sha, manifest["artifacts"]["trace"]["sha256"], "trace sha256")
    trace = read_trace(trace_path)
    selection = manifest["selection"]
    for role in ("base", "arrival", "query"):
        path = Path(selection[role]["path"])
        require_equal(sha256_file(path), selection[role]["sha256"], f"{role} data sha256")
    workload = manifest["workload"]
    base_n = int(workload["base_n"])
    arrival_n = int(workload["arrival_n"])
    query_n = int(workload["query_count"])
    k = int(workload["k"])
    if args.delta_bound <= int(workload["warmup_inserts"]) + 1:
        raise ValueError("delta_bound must exceed warmup+1 for this no-rebuild development trace")

    base_source = open_fvecs(Path(selection["base"]["path"]))
    arrival_source = open_fvecs(Path(selection["arrival"]["path"]))
    query_source = open_fvecs(Path(selection["query"]["path"]))
    a_start = int(selection["arrival"]["row_start"])
    q_start = int(selection["query"]["row_start"])
    base = np.asarray(base_source[:base_n], dtype=np.float32)
    arrivals = np.asarray(arrival_source[a_start:a_start + arrival_n], dtype=np.float32)
    queries = np.asarray(query_source[q_start:q_start + query_n], dtype=np.float32)
    if not (len(base) == base_n and len(arrivals) == arrival_n and len(queries) == query_n):
        raise ValueError("selected vector count mismatch")
    if base.shape[1] != arrivals.shape[1] or base.shape[1] != queries.shape[1]:
        raise ValueError("dimension mismatch")
    all_vectors = np.concatenate((base, arrivals), axis=0)

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"refuse overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    t_build = time.perf_counter_ns()
    tree = FrozenRadialTree(base, args.leaf_size)
    tree_build_ns = time.perf_counter_ns() - t_build
    state = TieredState(args.policy, base_n, base_n + arrival_n, args.sidecar_capacity, args.delta_bound, tree)

    state_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for event in trace:
        ordinal = int(event["ordinal"])
        op = event["op"]
        placement: Optional[str] = None
        certificate_leaf: Optional[int] = None
        fallback: Optional[str] = None
        if op == "I":
            label = int(event["label"])
            a_idx = int(event["arrival_offset"]) - a_start
            if label != base_n + a_idx or not (0 <= a_idx < arrival_n):
                raise ValueError(f"insert map mismatch at ordinal {ordinal}")
            placement, certificate_leaf, fallback = state.insert(label, arrivals[a_idx])
        elif op == "D":
            label = int(event["label"])
            placement, certificate_leaf = state.delete(label)
            fallback = "delete"
        else:
            q_idx = int(event["query_offset"]) - q_start
            if not (0 <= q_idx < query_n):
                raise ValueError(f"query map mismatch at ordinal {ordinal}")
            result_ids, result_d, counts, visited = tree.search(
                queries[q_idx], k, state.sidecars, state.delta, all_vectors, args.policy == "safe_c1"
            )
            active_ids = np.asarray(list(range(base_n)) + sorted(state.active), dtype=np.int64)
            exact_ids, exact_d = exact_topk(all_vectors[active_ids], active_ids, queries[q_idx], k)
            ok = result_ids == exact_ids
            if not ok:
                errors.append(f"oracle mismatch at ordinal {ordinal}")
            query_rows.append({
                "ordinal": ordinal, "query_offset": int(event["query_offset"]), "after_pair": int(event["after_pair"]),
                "full_active_ids_sha256": stable_ids_sha256(active_ids.tolist()),
                "dynamic_active_ids_sha256": stable_ids_sha256(state.active),
                "result_ids": result_ids, "result_squared_l2": result_d,
                "oracle_ids": exact_ids, "oracle_squared_l2": exact_d,
                "oracle_match": ok, "visited_leaf_ids": visited, "counts": counts,
                "direct_active": sum(len(x) for x in state.sidecars.values()), "delta_active": len(state.delta),
            })
        state_rows.append({
            "ordinal": ordinal, "op": op, "label": int(event.get("label", -1)),
            "placement": placement, "certificate_leaf": certificate_leaf, "fallback": fallback,
            "dynamic_active_ids_sha256": stable_ids_sha256(state.active),
            "dynamic_active_count": len(state.active), "direct_active": sum(len(x) for x in state.sidecars.values()),
            "delta_active": len(state.delta), "counters": dict(state.counters),
        })

    state_blob = b"".join(canonical_json_bytes(row) for row in state_rows)
    query_blob = b"".join(canonical_json_bytes(row) for row in query_rows)
    atomic_write_bytes(out_dir / "event_state_records.jsonl", state_blob)
    atomic_write_bytes(out_dir / "query_records.jsonl", query_blob)
    summary = {
        "schema": "tide-safe-c1-reference-mirror-run-v1",
        "status": "PASS_DEVELOPMENT_REFERENCE" if not errors and len(query_rows) == query_n else "FAIL_ORACLE",
        "evidence_scope": "development_reference_policy_only",
        "claim_boundary": [
            "CPU reference policy only", "not native GTS", "not GPU latency/throughput evidence",
            "not a heldout or final result", "C2 and C3 disabled/unmodeled",
            "no base deletes, rebuilds, concurrency, or general-metric claim",
        ],
        "input_protocol": {"path": str(protocol_dir), "manifest_sha256": manifest_sha, "trace_sha256": trace_sha, "declared_scope": scope},
        "policy": {"name": args.policy, "leaf_size": args.leaf_size, "sidecar_capacity": args.sidecar_capacity, "delta_bound": args.delta_bound, "rebuild_policy": "forbidden"},
        "tree": {"node_count": len(tree.nodes), "leaf_size": tree.leaf_size, "fingerprint_sha256": tree.fingerprint_sha256, "build_elapsed_ns": tree_build_ns},
        "workload": {"base_n": base_n, "arrival_n": arrival_n, "query_count": query_n, "k": k},
        "state": dict(state.counters) | {"direct_active_final": sum(len(x) for x in state.sidecars.values()), "delta_active_final": len(state.delta)},
        "correctness": {"queries_completed": len(query_rows), "queries_expected": query_n, "oracle_mismatches": len(errors), "errors": errors, "event_state_records_sha256": hashlib.sha256(state_blob).hexdigest(), "query_records_sha256": hashlib.sha256(query_blob).hexdigest()},
        "environment": {"hostname": socket.gethostname(), "cpu_model": cpu_model(), "platform": platform.platform(), "python": sys.version, "gpu_used": False},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(out_dir / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "out_dir": str(out_dir), "direct_inserts": state.counters["direct_inserts"], "delta_inserts": state.counters["delta_inserts"], "oracle_mismatches": len(errors)}, sort_keys=True))
    if errors:
        raise SystemExit(2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--policy", choices=("safe_c1", "buffer_only"), required=True)
    p.add_argument("--leaf-size", type=int, default=32)
    p.add_argument("--sidecar-capacity", type=int, default=8)
    p.add_argument("--delta-bound", type=int, default=4098)
    args = p.parse_args()
    if args.leaf_size < 2 or args.sidecar_capacity <= 0 or args.delta_bound <= 0:
        raise ValueError("invalid policy configuration")
    run(args)


if __name__ == "__main__":
    main()
