#!/usr/bin/env python3
"""CPU-only primitives for the isolated Safe-C1 G3 fixture preparation.

This module deliberately has no CUDA, subprocess, or GPU telemetry calls.
The source-mirror tree is a *selector preflight*, not native-GTS evidence: a
later native preflight must bind its actual frozen-tree receipt to the fixture.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SCALE = 100.0
STRICT_EPSILON = np.float32(1.0e-5)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def stable_set_sha256(ids: Iterable[int]) -> str:
    return hashlib.sha256("".join(f"{int(i)}\n" for i in sorted(set(map(int, ids)))).encode("ascii")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_fbin_layout(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise ValueError(f"truncated fbin header: {path}")
    rows, dim = struct.unpack("<ii", header)
    if rows <= 0 or dim <= 0:
        raise ValueError(f"invalid fbin shape {rows}x{dim}: {path}")
    expected = 8 + rows * dim * 4
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"fbin size mismatch {path}: actual={actual} expected={expected}")
    return rows, dim, expected


def fbin_memmap(path: Path) -> np.memmap:
    rows, dim, _ = read_fbin_layout(path)
    return np.memmap(path, mode="r", dtype="<f4", offset=8, shape=(rows, dim), order="C")


def quantize_f32(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: nonfinite input")
    rounded = np.rint(values.astype(np.float64) * SCALE)
    lo, hi = float(rounded.min()), float(rounded.max())
    if lo < np.iinfo(np.int16).min or hi > np.iinfo(np.int16).max:
        raise ValueError(f"{label}: quantization overflow [{lo}, {hi}]")
    return rounded.astype("<i2", copy=False)


def parse_gts_tree_config(tree_header: Path) -> dict[str, Any]:
    text = tree_header.read_text(encoding="utf-8", errors="strict")
    order_match = re.search(r"__managed__\s+int\s+TREE_ORDER\s*=\s*(\d+)\s*;", text)
    max_match = re.search(r"__managed__\s+int\s+MAX_SIZE\s*=\s*(\d+)\s*;", text)
    if not order_match or not max_match:
        raise ValueError(f"cannot parse TREE_ORDER/MAX_SIZE from {tree_header}")
    if "find smallest h where order^(h-1) leaves * MAX_SIZE >= object_count" not in text:
        raise ValueError("tree height implementation marker missing")
    return {
        "tree_header": str(tree_header),
        "tree_header_sha256": sha256_file(tree_header),
        "tree_order": int(order_match.group(1)),
        "max_size": int(max_match.group(1)),
        "height_rule": "smallest h with order^(h-1)*max_size >= object_count",
    }


def expected_gts_height(order: int, max_size: int, object_count: int) -> int:
    if order <= 1 or max_size <= 0 or object_count <= 0:
        raise ValueError("invalid source-tree height arguments")
    height, leaves = 1, 1
    while leaves * max_size < object_count and height < 20:
        leaves *= order
        height += 1
    return height


def l2sq_int64(pool: np.ndarray, query: np.ndarray, ids: Sequence[int] | np.ndarray) -> np.ndarray:
    chosen = np.asarray(ids, dtype=np.int64)
    if chosen.size == 0:
        return np.empty(0, dtype=np.int64)
    delta = pool[chosen].astype(np.int64, copy=False) - np.asarray(query, dtype=np.int16).astype(np.int64, copy=False)
    return np.sum(delta * delta, axis=1, dtype=np.int64)


def exact_results(pool: np.ndarray, query: np.ndarray, active_ids: Iterable[int], kind: str, k: int, radius_sq: int | float | None = None) -> list[list[int]]:
    ids = np.asarray(sorted(set(map(int, active_ids))), dtype=np.int64)
    distances = l2sq_int64(pool, query, ids)
    if kind == "knn":
        if k <= 0:
            raise ValueError("k must be positive")
        position = np.lexsort((ids, distances))[: min(k, len(ids))]
    elif kind == "range":
        if radius_sq is None or float(radius_sq) < 0:
            raise ValueError("range requires nonnegative radius_sq")
        keep = np.flatnonzero(distances <= radius_sq)
        position = keep[np.lexsort((ids[keep], distances[keep]))] if len(keep) else np.empty(0, dtype=np.int64)
    else:
        raise ValueError(f"unknown query kind {kind}")
    return [[int(ids[i]), int(distances[i])] for i in position]


def has_knn_boundary_tie(pool: np.ndarray, query: np.ndarray, active_ids: Iterable[int], k: int) -> bool:
    ids = np.asarray(sorted(set(map(int, active_ids))), dtype=np.int64)
    if len(ids) <= k:
        return False
    distances = l2sq_int64(pool, query, ids)
    sorted_distances = np.sort(distances, kind="stable")
    return bool(sorted_distances[k - 1] == sorted_distances[k])


def unique_within_radius_sq(pool: np.ndarray, target_id: int, active_ids: Iterable[int], radius_sq: int) -> bool:
    ids = np.asarray(sorted(set(map(int, active_ids))), dtype=np.int64)
    ids = ids[ids != int(target_id)]
    if not len(ids):
        return True
    return bool(np.all(l2sq_int64(pool, pool[int(target_id)], ids) > radius_sq))


def choose_nonboundary_radius_sq(pool: np.ndarray, query: np.ndarray, active_ids: Iterable[int], quantile: float) -> int:
    ids = np.asarray(sorted(set(map(int, active_ids))), dtype=np.int64)
    values = np.sort(l2sq_int64(pool, query, ids), kind="stable")
    if not len(values):
        raise ValueError("empty active set")
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * quantile))))
    lo = int(values[index])
    hi = int(values[index + 1]) if index + 1 < len(values) else lo + 2
    # A positive integer strictly between distances if possible; otherwise use
    # the exact value and record tie-free status separately in fixture checks.
    return lo + max(0, (hi - lo) // 2)


@dataclass
class Node:
    nid: int
    lid: int
    size: int
    pid: int
    min_dis: np.float32
    max_dis: np.float32
    is_leaf: bool
    children: list[int] = field(default_factory=list)


class SourceMirrorTree:
    """CPU mirror of the relevant source-tree split/certificate shape.

    It mirrors the frozen L2 interval construction in `tree.cuh` sufficiently
    for deterministic fixture selection. It is intentionally stamped as a
    source-mirror selector rather than a native execution receipt.
    """

    def __init__(self, pool: np.ndarray, base_ids: Iterable[int], order: int, max_size: int):
        self.pool = np.asarray(pool, dtype=np.int16)
        self.base_ids = np.asarray(sorted(set(map(int, base_ids))), dtype=np.int64)
        if self.base_ids.size == 0:
            raise ValueError("source mirror requires nonempty base")
        self.order = int(order)
        self.max_size = int(max_size)
        self.height = expected_gts_height(self.order, self.max_size, int(self.base_ids.size))
        self.id_list = self.base_ids.copy()
        self.nodes: dict[int, Node] = {}
        self._build()

    def _distance_f32(self, ids: np.ndarray, pivot: int) -> np.ndarray:
        delta = self.pool[ids].astype(np.float32, copy=False) - self.pool[int(pivot)].astype(np.float32, copy=False)
        return np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float32), dtype=np.float32)

    def _build(self) -> None:
        root_leaf = self.height == 1 or self.id_list.size <= self.max_size
        self.nodes[0] = Node(0, 0, int(self.id_list.size), -1, np.float32(0), np.float32(0), root_leaf)
        frontier = [0]
        for level in range(self.height - 1):
            next_frontier: list[int] = []
            for nid in frontier:
                parent = self.nodes[nid]
                if parent.size <= self.max_size:
                    parent.is_leaf = True
                    continue
                start, stop = parent.lid, parent.lid + parent.size
                current = self.id_list[start:stop].copy()
                pivot = int(current[(parent.size - 1) // 2])
                distances = self._distance_f32(current, pivot)
                # Thrust's exact tie order is not portable; all fixture
                # candidates receive a later native tie/certificate preflight.
                reorder = np.lexsort((current, distances))
                current = current[reorder]
                distances = distances[reorder]
                self.id_list[start:stop] = current
                average = parent.size // self.order
                if average <= 0:
                    parent.is_leaf = True
                    continue
                for slot in range(self.order):
                    child_size = average if slot < self.order - 1 else parent.size - (self.order - 1) * average
                    if child_size <= 0:
                        continue
                    child_lid = start + average * slot
                    child_nid = nid * self.order + slot + 1
                    lo = np.float32(distances[average * slot])
                    hi = np.float32(distances[average * slot + child_size - 1])
                    leaf = child_size <= self.max_size or level + 1 >= self.height - 1
                    self.nodes[child_nid] = Node(child_nid, child_lid, child_size, pivot, lo, hi, leaf)
                    parent.children.append(child_nid)
                    if not leaf:
                        next_frontier.append(child_nid)
            frontier = next_frontier

    def _candidate_distance(self, stable_id: int, pivot: int) -> np.float32:
        delta = self.pool[int(stable_id)].astype(np.float32, copy=False) - self.pool[int(pivot)].astype(np.float32, copy=False)
        return np.sqrt(np.sum(delta * delta, dtype=np.float32), dtype=np.float32)

    def certify(self, stable_id: int, epsilon: np.float32 = STRICT_EPSILON) -> dict[str, Any]:
        current = 0
        certificate: list[dict[str, Any]] = []
        for _ in range(max(0, self.height - 1)):
            parent = self.nodes[current]
            if parent.is_leaf:
                return {"ok": True, "leaf": current, "reason": "certified", "certificate": certificate}
            if not parent.children:
                return {"ok": False, "leaf": None, "reason": "empty_children", "certificate": certificate}
            pivot = self.nodes[parent.children[0]].pid
            distance = self._candidate_distance(int(stable_id), pivot)
            matched = [cid for cid in parent.children if distance > self.nodes[cid].min_dis + epsilon and distance < self.nodes[cid].max_dis - epsilon]
            if len(matched) != 1:
                return {
                    "ok": False,
                    "leaf": None,
                    "reason": "gap_or_boundary" if not matched else "overlap_or_ambiguous",
                    "certificate": certificate,
                    "distance": float(distance),
                    "matching_children": len(matched),
                }
            current = matched[0]
            child = self.nodes[current]
            certificate.append({
                "parent": parent.nid,
                "child": child.nid,
                "pivot": int(pivot),
                "lo": float(child.min_dis),
                "hi": float(child.max_dis),
                "distance": float(distance),
            })
            if child.is_leaf:
                return {"ok": True, "leaf": child.nid, "reason": "certified", "certificate": certificate}
        node = self.nodes[current]
        return {"ok": bool(node.is_leaf), "leaf": current if node.is_leaf else None, "reason": "certified" if node.is_leaf else "depth_exhausted", "certificate": certificate}

    def logical_hash(self) -> str:
        rows = []
        for nid in sorted(self.nodes):
            node = self.nodes[nid]
            if node.is_leaf:
                ids = self.id_list[node.lid:node.lid + node.size].astype(int).tolist()
                rows.append([node.nid, node.lid, node.size, node.pid, float(node.min_dis), float(node.max_dis), ids])
        return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "selector": "cpu_source_mirror_not_native_receipt",
            "tree_order": self.order,
            "max_size": self.max_size,
            "height": self.height,
            "base_count": int(self.base_ids.size),
            "logical_leaf_id_hash": self.logical_hash(),
        }


def classify_reservoir(tree: SourceMirrorTree, candidates: Iterable[int]) -> tuple[dict[int, list[int]], list[int], dict[int, dict[str, Any]]]:
    by_leaf: dict[int, list[int]] = {}
    rejected: list[int] = []
    receipts: dict[int, dict[str, Any]] = {}
    for sid in map(int, candidates):
        receipt = tree.certify(sid)
        receipts[sid] = receipt
        if receipt.get("ok"):
            by_leaf.setdefault(int(receipt["leaf"]), []).append(sid)
        else:
            rejected.append(sid)
    return by_leaf, rejected, receipts
