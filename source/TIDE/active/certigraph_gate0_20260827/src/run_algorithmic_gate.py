#!/usr/bin/env python3
"""Run the frozen CertiGraph motivation and algorithmic Gate-0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
from cuvs.neighbors import cagra


K = 10
TARGET_K = 100
NCAL = 256
NTEST = 1024
PROPOSALS = (32, 64, 128, 256)
ITOPKS = (128, 256, 512)
WIDTHS = (1, 2, 4)
LEAF_CAPS = (256, 512, 1024, 2048, 4096)
WARMUPS = 2
REPEATS = 5


def sha256(path: Path, block: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def pctl(values: np.ndarray | list[float] | list[int], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def cuda_time_search(index: Any, queries: cp.ndarray, params: Any, k: int) -> tuple[float, np.ndarray]:
    for _ in range(WARMUPS):
        _, neighbors = cagra.search(params, index, queries, k)
        cp.cuda.runtime.deviceSynchronize()
    samples = []
    keep = None
    for _ in range(REPEATS):
        a, b = cp.cuda.Event(), cp.cuda.Event()
        a.record()
        _, neighbors = cagra.search(params, index, queries, k)
        b.record()
        b.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(a, b)))
        keep = neighbors
    assert keep is not None
    return float(np.median(samples)), cp.asnumpy(cp.asarray(keep)).astype(
        np.int64, copy=False
    )


def compute_lb_matrix(
    signature_object: cp.ndarray,
    query_pivot: cp.ndarray,
    out_path: Path,
    batch: int = 16,
) -> np.memmap:
    if out_path.exists():
        raise FileExistsError(out_path)
    nq, npivot = query_pivot.shape
    nbase = signature_object.shape[0]
    out = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(nq, nbase))
    sig_i16 = signature_object.astype(cp.int16)
    qp_i16 = query_pivot.astype(cp.int16)
    for start in range(0, nq, batch):
        q = qp_i16[start : start + batch]
        lb = cp.max(cp.abs(sig_i16[None, :, :] - q[:, None, :]), axis=2)
        out[start : start + len(q)] = cp.asnumpy(lb).astype(np.uint8, copy=False)
    out.flush()
    return out


def topk_radius(dist: np.ndarray, k: int) -> int:
    return int(np.partition(dist, k - 1)[k - 1])


def seed_metrics(
    exact: np.ndarray,
    lb: np.ndarray,
    seed_ids: np.ndarray,
    k: int = K,
) -> dict[str, Any]:
    seed = np.unique(np.asarray(seed_ids, dtype=np.int64))
    if len(seed) < k:
        raise ValueError("seed set smaller than k")
    seed_dist = exact[seed]
    radius = topk_radius(seed_dist, k)
    survivors = int(np.count_nonzero(lb <= radius))
    known_survivors = int(np.count_nonzero(lb[seed] <= radius))
    exact_calls = int(len(seed) + survivors - known_survivors)
    candidate_mask = lb <= radius
    candidate_dist = exact[candidate_mask]
    candidate_top = np.sort(np.partition(candidate_dist, k - 1)[:k])
    true_top = np.sort(np.partition(exact, k - 1)[:k])
    return {
        "radius": radius,
        "survivors": survivors,
        "seed_count": int(len(seed)),
        "known_survivors": known_survivors,
        "exact_calls": exact_calls,
        "distance_multiset_match": bool(np.array_equal(candidate_top, true_top)),
    }


@dataclass
class TreeNode:
    lo: int
    hi: int
    mins: np.ndarray
    maxs: np.ndarray
    left: int = -1
    right: int = -1

    @property
    def size(self) -> int:
        return self.hi - self.lo


def build_balanced_signature_tree(
    signature_pmajor: np.ndarray, min_leaf: int
) -> tuple[np.ndarray, list[TreeNode]]:
    nbase = signature_pmajor.shape[1]
    order = np.arange(nbase, dtype=np.int32)
    nodes: list[TreeNode] = []

    def build(lo: int, hi: int) -> int:
        ids = order[lo:hi]
        mins = np.empty(signature_pmajor.shape[0], dtype=np.uint8)
        maxs = np.empty_like(mins)
        for p in range(signature_pmajor.shape[0]):
            values = signature_pmajor[p, ids]
            mins[p] = values.min()
            maxs[p] = values.max()
        node_id = len(nodes)
        nodes.append(TreeNode(lo=lo, hi=hi, mins=mins, maxs=maxs))
        if hi - lo <= min_leaf:
            return node_id
        spread = maxs.astype(np.int16) - mins.astype(np.int16)
        dim = int(np.argmax(spread))
        mid = (lo + hi) // 2
        values = signature_pmajor[dim, ids]
        partition = np.argpartition(values, mid - lo)
        order[lo:hi] = ids[partition]
        left = build(lo, mid)
        right = build(mid, hi)
        nodes[node_id].left = left
        nodes[node_id].right = right
        return node_id

    build(0, nbase)
    return order, nodes


def node_lb(query: np.ndarray, node: TreeNode) -> int:
    q = query.astype(np.int16, copy=False)
    low = node.mins.astype(np.int16, copy=False) - q
    high = q - node.maxs.astype(np.int16, copy=False)
    return int(np.maximum(np.maximum(low, high), 0).max())


def tree_certificate_bytes(
    query: np.ndarray,
    radius: int,
    leaf_cap: int,
    nodes: list[TreeNode],
    npivot: int,
) -> dict[str, int]:
    stack = [0]
    visited = 0
    object_rows = 0
    pruned_nodes = 0
    while stack:
        node_id = stack.pop()
        node = nodes[node_id]
        visited += 1
        if node_lb(query, node) > radius:
            pruned_nodes += 1
            continue
        if node.size <= leaf_cap or node.left < 0:
            object_rows += node.size
        else:
            stack.append(node.left)
            stack.append(node.right)
    node_bytes = visited * 2 * npivot
    object_bytes = object_rows * npivot
    return {
        "visited_nodes": visited,
        "pruned_nodes": pruned_nodes,
        "object_signature_rows": object_rows,
        "node_envelope_bytes": node_bytes,
        "object_signature_bytes": object_bytes,
        "total_certificate_bytes": node_bytes + object_bytes,
    }


def gpu_snapshot() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pstate,power.draw",
        "--format=csv,noheader,nounits",
    ]
    return {
        "visible_device": int(cp.cuda.runtime.getDevice()),
        "inventory": subprocess.check_output(cmd, text=True).strip().splitlines(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    args = ap.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    final_path = args.results / "FINAL_VERDICT.json"
    if final_path.exists():
        raise FileExistsError(final_path)

    meta = json.loads((args.export / "export_metadata.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    nbase, nq, npivot = (int(meta[x]) for x in ("nbase", "nquery", "npivot"))
    if nq != NCAL + NTEST or npivot != 64:
        raise ValueError((nq, npivot))
    sig_pm = np.memmap(
        args.export / "signature.pmajor.u8", dtype=np.uint8, mode="r", shape=(npivot, nbase)
    )
    qp = np.memmap(
        args.export / "query_pivot.qmajor.u8", dtype=np.uint8, mode="r", shape=(nq, npivot)
    )
    exact = np.memmap(
        args.export / "exact.qmajor.u8", dtype=np.uint8, mode="r", shape=(nq, nbase)
    )
    pivots = np.fromfile(args.export / "pivots.i32", dtype=np.int32)
    if len(pivots) != npivot or len(set(map(int, pivots))) != npivot:
        raise ValueError("bad pivots")

    cp.cuda.runtime.setDevice(0)
    sig_obj_host = np.ascontiguousarray(sig_pm.T)
    sig_obj_gpu_u8 = cp.asarray(sig_obj_host)
    sig_obj_gpu_f32 = sig_obj_gpu_u8.astype(cp.float32)
    qp_gpu_f32 = cp.asarray(np.asarray(qp), dtype=cp.float32)
    lb_path = args.raw / "lower_bound.qmajor.u8"
    lb = compute_lb_matrix(sig_obj_gpu_u8, cp.asarray(np.asarray(qp)), lb_path)

    # Independent triangle-inequality and exact top-k facts.
    true_r10 = np.empty(nq, dtype=np.uint8)
    true_r100 = np.empty(nq, dtype=np.uint8)
    triangle_violations = 0
    target_hits100 = np.zeros(nq, dtype=bool)
    records = manifest["records"]
    for q in range(nq):
        d = np.asarray(exact[q])
        l = np.asarray(lb[q])
        triangle_violations += int(np.count_nonzero(l > d))
        true_r10[q] = topk_radius(d, K)
        true_r100[q] = topk_radius(d, TARGET_K)
        tids = np.asarray(records[q]["target_ids"], dtype=np.int64)
        target_hits100[q] = bool(np.any(d[tids] <= true_r100[q]))

    build_params = cagra.IndexParams(
        metric="sqeuclidean", graph_degree=32, intermediate_graph_degree=64
    )
    cp.cuda.runtime.deviceSynchronize()
    a, b = cp.cuda.Event(), cp.cuda.Event()
    a.record()
    index = cagra.build(build_params, sig_obj_gpu_f32)
    b.record()
    b.synchronize()
    graph_build_ms = float(cp.cuda.get_elapsed_time(a, b))

    configs = []
    all_neighbors: dict[tuple[int, int, int], np.ndarray] = {}
    for proposals in PROPOSALS:
        for itopk in ITOPKS:
            if itopk < proposals:
                continue
            for width in WIDTHS:
                params = cagra.SearchParams(itopk_size=itopk, search_width=width)
                cal_ms, _ = cuda_time_search(index, qp_gpu_f32[:NCAL], params, proposals)
                _, neighbors = cuda_time_search(index, qp_gpu_f32, params, proposals)
                key = (proposals, itopk, width)
                all_neighbors[key] = neighbors
                neighbors.astype(np.int32).tofile(
                    args.raw / f"neighbors_m{proposals}_i{itopk}_w{width}.i32"
                )
                cal_calls = []
                cal_radius_gap = []
                exact_ok = True
                for q in range(NCAL):
                    seeds = np.concatenate([pivots.astype(np.int64), neighbors[q]])
                    m = seed_metrics(np.asarray(exact[q]), np.asarray(lb[q]), seeds)
                    cal_calls.append(m["exact_calls"])
                    cal_radius_gap.append(m["radius"] - int(true_r10[q]))
                    exact_ok &= m["distance_multiset_match"]
                configs.append(
                    {
                        "proposals": proposals,
                        "itopk_size": itopk,
                        "search_width": width,
                        "calibration_cuda_median_ms_diagnostic": cal_ms,
                        "calibration_exact_calls_median": pctl(cal_calls, 50),
                        "calibration_exact_calls_p90": pctl(cal_calls, 90),
                        "calibration_radius_gap_median": pctl(cal_radius_gap, 50),
                        "calibration_radius_gap_p90": pctl(cal_radius_gap, 90),
                        "calibration_exact": bool(exact_ok),
                    }
                )
    configs.sort(
        key=lambda x: (
            x["calibration_exact_calls_median"],
            x["calibration_cuda_median_ms_diagnostic"],
            x["proposals"],
            x["itopk_size"],
            x["search_width"],
        )
    )
    selected = configs[0]
    selected_key = (
        int(selected["proposals"]),
        int(selected["itopk_size"]),
        int(selected["search_width"]),
    )
    selected_neighbors = all_neighbors[selected_key]

    # Keeper/candidate call counts and exactness on all queries.
    pivot_rows = []
    graph_rows = []
    for q in range(nq):
        pivot_rows.append(
            seed_metrics(np.asarray(exact[q]), np.asarray(lb[q]), pivots.astype(np.int64))
        )
        graph_seed = np.concatenate([pivots.astype(np.int64), selected_neighbors[q]])
        graph_rows.append(
            seed_metrics(np.asarray(exact[q]), np.asarray(lb[q]), graph_seed)
        )

    # Build a query-independent balanced hierarchy, then choose leaf capacity on calibration.
    tree_start = time.perf_counter()
    _, nodes = build_balanced_signature_tree(sig_pm, min(LEAF_CAPS))
    tree_build_seconds = time.perf_counter() - tree_start
    cap_results: dict[int, list[dict[str, int]]] = {}
    for cap in LEAF_CAPS:
        rows = []
        for q in range(nq):
            rows.append(
                tree_certificate_bytes(
                    np.asarray(qp[q]), int(graph_rows[q]["radius"]), cap, nodes, npivot
                )
            )
        cap_results[cap] = rows
    cap_cal = []
    for cap in LEAF_CAPS:
        values = [x["total_certificate_bytes"] for x in cap_results[cap][:NCAL]]
        cap_cal.append(
            {
                "leaf_capacity": cap,
                "calibration_certificate_bytes_median": pctl(values, 50),
                "calibration_certificate_bytes_p90": pctl(values, 90),
            }
        )
    cap_cal.sort(
        key=lambda x: (
            x["calibration_certificate_bytes_median"],
            x["calibration_certificate_bytes_p90"],
            x["leaf_capacity"],
        )
    )
    selected_cap = int(cap_cal[0]["leaf_capacity"])

    test = slice(NCAL, NCAL + NTEST)
    pivot_calls = np.asarray([x["exact_calls"] for x in pivot_rows[test]], dtype=np.int64)
    graph_calls = np.asarray([x["exact_calls"] for x in graph_rows[test]], dtype=np.int64)
    radius_gap = np.asarray(
        [graph_rows[q]["radius"] - int(true_r10[q]) for q in range(NCAL, nq)],
        dtype=np.int64,
    )
    tree_test = cap_results[selected_cap][test]
    tree_bytes = np.asarray(
        [x["total_certificate_bytes"] for x in tree_test], dtype=np.int64
    )
    flat_bytes = nbase * npivot
    exact_matches = all(x["distance_multiset_match"] for x in pivot_rows[test]) and all(
        x["distance_multiset_match"] for x in graph_rows[test]
    )

    metrics = {
        "eligible_external_queries": int(manifest["eligible_unique_queries"]),
        "test_queries": NTEST,
        "target_hit_exact_top100_rate": float(target_hits100[test].mean()),
        "true_r10": {
            "median": pctl(true_r10[test], 50),
            "p90": pctl(true_r10[test], 90),
            "min": int(true_r10[test].min()),
            "max": int(true_r10[test].max()),
        },
        "graph_radius_gap": {
            "median": pctl(radius_gap, 50),
            "p90": pctl(radius_gap, 90),
            "max": int(radius_gap.max()),
        },
        "exact_calls": {
            "scan": nbase,
            "pivot_flat_median": pctl(pivot_calls, 50),
            "pivot_flat_p90": pctl(pivot_calls, 90),
            "graph_flat_median": pctl(graph_calls, 50),
            "graph_flat_p90": pctl(graph_calls, 90),
            "graph_vs_scan_median_reduction": nbase / pctl(graph_calls, 50),
            "graph_vs_pivot_median_reduction": pctl(pivot_calls, 50)
            / pctl(graph_calls, 50),
            "graph_vs_scan_p90_reduction": nbase / pctl(graph_calls, 90),
        },
        "certificate_bytes": {
            "flat_per_query": flat_bytes,
            "tree_median": pctl(tree_bytes, 50),
            "tree_p90": pctl(tree_bytes, 90),
            "tree_vs_flat_median_reduction": flat_bytes / pctl(tree_bytes, 50),
            "tree_vs_flat_p90_reduction": flat_bytes / pctl(tree_bytes, 90),
        },
        "triangle_inequality_violations": triangle_violations,
        "distance_multiset_exact": bool(exact_matches),
    }
    checks = {
        "G_DATA": metrics["eligible_external_queries"] >= NTEST,
        "G_TASK": metrics["target_hit_exact_top100_rate"] >= 0.90,
        "G_BOUND": metrics["graph_radius_gap"]["median"] <= 1
        and metrics["graph_radius_gap"]["p90"] <= 2,
        "G_WORK": metrics["exact_calls"]["graph_vs_scan_median_reduction"] >= 5.0
        and metrics["exact_calls"]["graph_vs_pivot_median_reduction"] >= 2.0
        and metrics["exact_calls"]["graph_vs_scan_p90_reduction"] >= 2.0,
        "G_BLOCK": metrics["certificate_bytes"]["tree_vs_flat_median_reduction"] >= 2.0
        and metrics["certificate_bytes"]["tree_vs_flat_p90_reduction"] >= 1.25,
        "G_EXACT": triangle_violations == 0 and bool(exact_matches),
    }
    verdict = {
        "schema": "certigraph-gate0-verdict-v1",
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "pass": bool(all(checks.values())),
        "checks": checks,
        "metrics": metrics,
        "selected_graph": selected,
        "selected_leaf_capacity": selected_cap,
        "graph_grid": configs,
        "leaf_capacity_calibration": cap_cal,
        "diagnostic_only": {
            "graph_build_cuda_ms": graph_build_ms,
            "tree_build_wall_seconds": tree_build_seconds,
            "export_timing": {
                k: meta[k]
                for k in meta
                if k.endswith("_cuda_ms_diagnostic")
            },
        },
        "provenance": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": gpu_snapshot(),
            "export_metadata_sha256": sha256(args.export / "export_metadata.json"),
            "query_manifest_sha256": sha256(args.manifest),
            "signature_sha256": sha256(args.export / "signature.pmajor.u8"),
            "query_pivot_sha256": sha256(args.export / "query_pivot.qmajor.u8"),
            "exact_matrix_sha256": sha256(args.export / "exact.qmajor.u8"),
            "lower_bound_sha256": sha256(lb_path),
        },
        "interpretation": {
            "implementation_status": "completed",
            "mechanism_status": "accepted" if all(checks.values()) else "rejected_or_narrowed",
            "thesis_impact": "eligible_for_fused_kernel_gate" if all(checks.values()) else "do_not_promote_full_certigraph_thesis",
            "performance_boundary": "Counts and diagnostic timings are not an end-to-end GPU speedup claim.",
        },
    }
    final_path.write_text(json.dumps(verdict, indent=2, sort_keys=True))

    summary_csv = args.results / "SUMMARY.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for group, values in metrics.items():
            if isinstance(values, dict):
                for name, value in values.items():
                    w.writerow([f"{group}.{name}", value])
            else:
                w.writerow([group, values])

    lines = [
        "# CertiGraph Gate-0 results",
        "",
        f"**Verdict: {verdict['verdict']}**",
        "",
        "## Frozen gates",
        "",
    ]
    lines += [f"- `{name}`: **{'PASS' if ok else 'FAIL'}**" for name, ok in checks.items()]
    lines += [
        "",
        "## Primary test evidence",
        "",
        f"- Eligible external queries: {metrics['eligible_external_queries']}",
        f"- Recorded target within exact top-100: {metrics['target_hit_exact_top100_rate']:.3%}",
        f"- Exact k=10 radius median / p90: {metrics['true_r10']['median']:.1f} / {metrics['true_r10']['p90']:.1f}",
        f"- Graph radius gap median / p90: {metrics['graph_radius_gap']['median']:.1f} / {metrics['graph_radius_gap']['p90']:.1f}",
        f"- Graph exact-call reduction vs scan, median / p90-query: {metrics['exact_calls']['graph_vs_scan_median_reduction']:.3f}x / {metrics['exact_calls']['graph_vs_scan_p90_reduction']:.3f}x",
        f"- Graph exact-call reduction vs pivot-flat median: {metrics['exact_calls']['graph_vs_pivot_median_reduction']:.3f}x",
        f"- Tree certificate-byte reduction median / p90-query: {metrics['certificate_bytes']['tree_vs_flat_median_reduction']:.3f}x / {metrics['certificate_bytes']['tree_vs_flat_p90_reduction']:.3f}x",
        f"- Triangle-inequality violations: {triangle_violations}",
        f"- Exact top-k distance-multiset check: {'PASS' if exact_matches else 'FAIL'}",
        "",
        "## Selected calibration-only configuration",
        "",
        f"- Graph: proposals={selected_key[0]}, itopk={selected_key[1]}, width={selected_key[2]}",
        f"- Tree leaf capacity: {selected_cap}",
        "",
        "All CUDA timings in this gate are diagnostic only. No end-to-end speedup is claimed.",
    ]
    (args.results / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": verdict["verdict"], "results": str(final_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
