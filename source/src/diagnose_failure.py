#!/usr/bin/env python3
"""Post-hoc diagnosis for the frozen Gate-0 failure; never changes its verdict."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


K = 10
NCAL = 256


def kth(values: np.ndarray, k: int) -> int:
    return int(np.partition(values, k - 1)[k - 1])


def qstats(exact: np.ndarray, lb: np.ndarray, seed: np.ndarray) -> dict:
    seed = np.unique(seed.astype(np.int64, copy=False))
    radius = kth(exact[seed], K)
    survivors = int(np.count_nonzero(lb <= radius))
    known = int(np.count_nonzero(lb[seed] <= radius))
    return {
        "radius": radius,
        "exact_calls": int(len(seed) + survivors - known),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    meta = json.loads((args.export / "export_metadata.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    nbase, nq, p = (int(meta[x]) for x in ("nbase", "nquery", "npivot"))
    exact = np.memmap(
        args.export / "exact.qmajor.u8", dtype=np.uint8, mode="r", shape=(nq, nbase)
    )
    lb = np.memmap(
        args.evidence / "lower_bound.qmajor.u8", dtype=np.uint8, mode="r", shape=(nq, nbase)
    )
    pivots = np.fromfile(args.export / "pivots.i32", dtype=np.int32)
    neighbors = np.memmap(
        args.evidence / "neighbors_m256_i512_w1.i32",
        dtype=np.int32,
        mode="r",
        shape=(nq, 256),
    )
    rows = []
    for q in range(NCAL, nq):
        d, lower = np.asarray(exact[q]), np.asarray(lb[q])
        true_r10, true_r100 = kth(d, 10), kth(d, 100)
        targets = np.asarray(manifest["records"][q]["target_ids"], dtype=np.int64)
        graph = qstats(d, lower, np.concatenate([pivots, neighbors[q]]))
        pivot = qstats(d, lower, pivots)
        target_best = int(d[targets].min())
        rows.append(
            {
                "query_id": q,
                "query": manifest["records"][q]["query_display"],
                "targets": manifest["records"][q]["target_display"],
                "query_len": len(bytes.fromhex(manifest["records"][q]["query_latin1_hex"])),
                "true_r10": true_r10,
                "true_r100": true_r100,
                "target_best_distance": target_best,
                "target_hit100": target_best <= true_r100,
                "graph_radius": graph["radius"],
                "graph_exact_calls": graph["exact_calls"],
                "pivot_exact_calls": pivot["exact_calls"],
                "scan_reduction": nbase / graph["exact_calls"],
            }
        )

    buckets = []
    for radius in sorted(set(x["true_r10"] for x in rows)):
        group = [x for x in rows if x["true_r10"] == radius]
        buckets.append(
            {
                "true_r10": radius,
                "queries": len(group),
                "fraction": len(group) / len(rows),
                "target_hit100_rate": float(np.mean([x["target_hit100"] for x in group])),
                "graph_exact_calls_median": float(np.median([x["graph_exact_calls"] for x in group])),
                "graph_exact_calls_p90": float(np.percentile([x["graph_exact_calls"] for x in group], 90)),
                "scan_reduction_median": float(np.median([x["scan_reduction"] for x in group])),
            }
        )
    misses = sorted(
        (x for x in rows if not x["target_hit100"]),
        key=lambda x: (x["target_best_distance"] - x["true_r100"], x["query_id"]),
        reverse=True,
    )[:20]
    result = {
        "schema": "certigraph-gate0-posthoc-diagnosis-v1",
        "status": "post-hoc diagnostic; frozen verdict unchanged",
        "by_true_r10": buckets,
        "representative_target_misses": misses,
        "observation": "Tail verification work tracks larger exact kNN radii; exact edit distance also fails to retain many recorded spelling targets within top-100.",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    lines = [
        "# Gate-0 post-hoc failure diagnosis",
        "",
        "This diagnostic does not change the frozen verdict or thresholds.",
        "",
        "| Exact R10 | Queries | Share | Target hit@100 | Graph exact calls median | P90 | Scan reduction median |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in buckets:
        lines.append(
            f"| {x['true_r10']} | {x['queries']} | {x['fraction']:.1%} | "
            f"{x['target_hit100_rate']:.1%} | {x['graph_exact_calls_median']:.0f} | "
            f"{x['graph_exact_calls_p90']:.0f} | {x['scan_reduction_median']:.2f}x |"
        )
    lines += [
        "",
        "The graph supplies a tight incumbent radius, but certificate selectivity collapses as the true radius grows. The block hierarchy cannot repair this because its per-pivot min/max envelopes are looser than the per-object certificate. Separately, the external spelling targets show that exact edit distance is not a sufficiently strong application objective for this corpus.",
    ]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

