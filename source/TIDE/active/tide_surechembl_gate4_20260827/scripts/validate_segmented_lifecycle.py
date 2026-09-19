#!/usr/bin/env python3
"""Validate exact segmented publication and per-bin compaction semantics."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
from FPSim2 import FPSim2Engine
from FPSim2.FPSim2lib import GenericSearch


THRESHOLDS = ((7, 10), (4, 5))
CORRECTNESS_QUERIES = tuple(range(64)) + tuple(range(512, 768))


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def popcount_interval(query_popcount: int, num: int, den: int) -> tuple[int, int]:
    lower = ceil_div(num * query_popcount, den)
    upper = (den * query_popcount) // num
    return max(0, lower), min(256, upper)


def dense_counts_from_engine(engine: FPSim2Engine) -> np.ndarray:
    counts = np.zeros(257, dtype=np.int64)
    for popcount, (begin, end) in engine.popcnt_bins:
        counts[int(popcount)] = int(end) - int(begin)
    if int(counts.sum()) != int(engine.fps.shape[0]):
        raise RuntimeError("population-count directory does not cover the database")
    return counts


def dense_ranges(counts: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate(([0], np.cumsum(counts)))
    return np.stack((cumulative[:-1], cumulative[1:]), axis=1)


@dataclasses.dataclass(frozen=True)
class Segment:
    source: str
    begin: int
    end: int


class EpochDirectory:
    """Small logical model of copy-before-publish immutable descriptors."""

    def __init__(self, base_ranges: np.ndarray):
        self._epoch = 0
        self._visible = tuple(
            (Segment("base", int(begin), int(end)),) if end > begin else tuple()
            for begin, end in base_ranges
        )
        self._staged: tuple[tuple[Segment, ...], ...] | None = None

    def stage_delta(self, delta_ranges: np.ndarray) -> None:
        if self._staged is not None:
            raise RuntimeError("an unpublished epoch is already staged")
        staged = []
        for visible, (begin, end) in zip(
            self._visible, delta_ranges, strict=True
        ):
            descriptors = list(visible)
            if end > begin:
                descriptors.append(Segment("delta", int(begin), int(end)))
            staged.append(tuple(descriptors))
        self._staged = tuple(staged)

    def publish(self) -> int:
        if self._staged is None:
            raise RuntimeError("no staged epoch")
        self._visible = self._staged
        self._staged = None
        self._epoch += 1
        return self._epoch

    def snapshot(self) -> tuple[int, tuple[tuple[Segment, ...], ...]]:
        return self._epoch, self._visible


def canonical(results: list[np.ndarray]) -> tuple[tuple[int, int], ...]:
    rows: list[tuple[int, int]] = []
    for result in results:
        coeff_bits = result["coeff"].view(np.uint32)
        rows.extend(
            (int(mol_id), int(bits))
            for mol_id, bits in zip(
                result["mol_id"].tolist(), coeff_bits.tolist(), strict=True
            )
        )
    rows.sort(key=lambda row: (-row[1], row[0]))
    return tuple(rows)


def result_hash(rows: tuple[tuple[int, int], ...]) -> str:
    digest = hashlib.sha256()
    for mol_id, coeff_bits in rows:
        digest.update(struct.pack("<II", mol_id, coeff_bits))
    return digest.hexdigest()


def first_difference(
    left: tuple[tuple[int, int], ...], right: tuple[tuple[int, int], ...]
) -> dict | None:
    for index, (lhs, rhs) in enumerate(zip(left, right, strict=False)):
        if lhs != rhs:
            return {"index": index, "left": lhs, "right": rhs}
    if len(left) != len(right):
        return {
            "index": min(len(left), len(right)),
            "left": left[min(len(left), len(right)) : min(len(left), len(right)) + 1],
            "right": right[min(len(left), len(right)) : min(len(left), len(right)) + 1],
        }
    return None


def bounded_contiguous_search(
    query: np.ndarray,
    database: np.ndarray,
    ranges: np.ndarray,
    lower: int,
    upper: int,
    threshold: float,
) -> tuple[tuple[tuple[int, int], ...], int]:
    begin = int(ranges[lower, 0])
    end = int(ranges[upper, 1])
    result = GenericSearch(query, database, threshold, 0, 0, begin, end)
    if result.size:
        result_popcounts = database[result["idx"].astype(np.int64), 5]
        outside = int(
            np.count_nonzero(
                (result_popcounts < lower) | (result_popcounts > upper)
            )
        )
    else:
        outside = 0
    return canonical([result]), outside


def directory_search(
    query: np.ndarray,
    snapshot: tuple[int, tuple[tuple[Segment, ...], ...]],
    sources: dict[str, np.ndarray],
    lower: int,
    upper: int,
    threshold: float,
) -> tuple[tuple[tuple[int, int], ...], dict[str, int], int]:
    _, descriptors = snapshot
    parts: list[np.ndarray] = []
    source_hits: dict[str, int] = {}
    outside = 0
    for popcount in range(lower, upper + 1):
        for segment in descriptors[popcount]:
            database = sources[segment.source]
            result = GenericSearch(
                query,
                database,
                threshold,
                0,
                0,
                segment.begin,
                segment.end,
            )
            parts.append(result)
            source_hits[segment.source] = source_hits.get(segment.source, 0) + int(
                result.size
            )
            if result.size:
                pcs = database[result["idx"].astype(np.int64), 5]
                outside += int(
                    np.count_nonzero((pcs < lower) | (pcs > upper))
                )
    return canonical(parts), source_hits, outside


def build_per_bin_compaction(
    base: np.ndarray,
    delta: np.ndarray,
    union: np.ndarray,
    base_ranges: np.ndarray,
    delta_ranges: np.ndarray,
    union_ranges: np.ndarray,
) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    compacted = np.empty_like(union)
    mismatched_bins = []
    for popcount in range(257):
        base_begin, base_end = map(int, base_ranges[popcount])
        delta_begin, delta_end = map(int, delta_ranges[popcount])
        union_begin, union_end = map(int, union_ranges[popcount])
        pieces = []
        if base_end > base_begin:
            pieces.append(base[base_begin:base_end])
        if delta_end > delta_begin:
            pieces.append(delta[delta_begin:delta_end])
        if pieces:
            merged = pieces[0].copy() if len(pieces) == 1 else np.concatenate(pieces)
            order = np.argsort(merged[:, 0], kind="stable")
            merged = merged[order]
            compacted[union_begin:union_end] = merged
        if not np.array_equal(
            compacted[union_begin:union_end], union[union_begin:union_end]
        ):
            mismatched_bins.append(popcount)
    return compacted, {
        "elapsed_s": time.perf_counter() - started,
        "mismatched_bin_count": len(mismatched_bins),
        "mismatched_bins": mismatched_bins,
        "byte_identical_to_fresh_union": len(mismatched_bins) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-h5", type=Path, required=True)
    parser.add_argument("--union-h5", type=Path, required=True)
    parser.add_argument("--queries-u64x6", type=Path, required=True)
    parser.add_argument("--delta-u64x6", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    queries = np.fromfile(args.queries_u64x6, dtype="<u8").reshape(-1, 6)
    delta = np.fromfile(args.delta_u64x6, dtype="<u8").reshape(-1, 6)
    if queries.shape != (4608, 6) or delta.shape != (20760, 6):
        raise RuntimeError(
            f"unexpected input shapes: queries={queries.shape}, delta={delta.shape}"
        )
    delta_order = np.lexsort((delta[:, 0], delta[:, 5]))
    delta = np.ascontiguousarray(delta[delta_order])

    base_engine = FPSim2Engine(str(args.base_h5))
    union_engine = FPSim2Engine(str(args.union_h5))
    base = base_engine.fps
    union = union_engine.fps
    base_counts = dense_counts_from_engine(base_engine)
    union_counts = dense_counts_from_engine(union_engine)
    delta_counts = np.bincount(delta[:, 5].astype(np.int64), minlength=257)
    if not np.array_equal(base_counts + delta_counts, union_counts):
        raise RuntimeError("base plus delta bin counts do not equal the fresh union")
    base_ranges = dense_ranges(base_counts)
    delta_ranges = dense_ranges(delta_counts)
    union_ranges = dense_ranges(union_counts)

    base_id_min = int(np.min(base[:, 0]))
    base_id_max = int(np.max(base[:, 0]))
    delta_id_min = int(np.min(delta[:, 0]))
    delta_id_max = int(np.max(delta[:, 0]))
    delta_ids = set(delta[:, 0].astype(np.uint32, copy=False).tolist())
    if base_id_max < delta_id_min or delta_id_max < base_id_min:
        id_overlap = 0
        id_disjoint_proof = "disjoint_id_ranges"
    else:
        id_overlap = int(np.count_nonzero(np.isin(delta[:, 0], base[:, 0])))
        id_disjoint_proof = "exact_isin"
    if id_overlap:
        raise RuntimeError(f"base and delta have {id_overlap} overlapping IDs")

    directory = EpochDirectory(base_ranges)
    epoch0, pre_snapshot = directory.snapshot()
    directory.stage_delta(delta_ranges)
    staged_epoch, staged_visible_snapshot = directory.snapshot()
    staged_delta_descriptor_count = sum(
        segment.source == "delta"
        for descriptors in staged_visible_snapshot
        for segment in descriptors
    )
    epoch1 = directory.publish()
    published_epoch, post_snapshot = directory.snapshot()
    published_delta_descriptor_count = sum(
        segment.source == "delta"
        for descriptors in post_snapshot
        for segment in descriptors
    )

    compacted, compaction = build_per_bin_compaction(
        base, delta, union, base_ranges, delta_ranges, union_ranges
    )
    compacted_directory = EpochDirectory(union_ranges)
    _, compacted_snapshot_base = compacted_directory.snapshot()
    compacted_snapshot = (
        2,
        tuple(
            tuple(
                Segment("compacted", segment.begin, segment.end)
                for segment in descriptors
            )
            for descriptors in compacted_snapshot_base
        ),
    )

    sources = {"base": base, "delta": delta, "compacted": compacted}
    tasks = [
        (query_index, num, den)
        for query_index in CORRECTNESS_QUERIES
        for num, den in THRESHOLDS
    ]

    def validate(task: tuple[int, int, int]) -> dict:
        query_index, num, den = task
        query = np.array(queries[query_index], dtype=np.uint64, copy=True)
        threshold = num / den
        lower, upper = popcount_interval(int(query[5]), num, den)
        base_oracle, outside_base = bounded_contiguous_search(
            query, base, base_ranges, lower, upper, threshold
        )
        union_oracle, outside_union = bounded_contiguous_search(
            query, union, union_ranges, lower, upper, threshold
        )
        pre, pre_sources, outside_pre = directory_search(
            query, (epoch0, pre_snapshot), sources, lower, upper, threshold
        )
        post, post_sources, outside_post = directory_search(
            query, (epoch1, post_snapshot), sources, lower, upper, threshold
        )
        compact, compact_sources, outside_compact = directory_search(
            query, compacted_snapshot, sources, lower, upper, threshold
        )

        pre_ok = pre == base_oracle
        post_ok = post == union_oracle
        compact_ok = compact == post
        delta_id_set = delta_ids
        pre_delta_id_hits = sum(mol_id in delta_id_set for mol_id, _ in pre)
        post_delta_id_hits = sum(mol_id in delta_id_set for mol_id, _ in post)
        compact_delta_id_hits = sum(mol_id in delta_id_set for mol_id, _ in compact)
        duplicate_counts = {
            "base_oracle": len(base_oracle) - len({row[0] for row in base_oracle}),
            "union_oracle": len(union_oracle) - len({row[0] for row in union_oracle}),
            "pre": len(pre) - len({row[0] for row in pre}),
            "post": len(post) - len({row[0] for row in post}),
            "compact": len(compact) - len({row[0] for row in compact}),
        }
        return {
            "query_index": query_index,
            "query_mol_id": int(query[0]),
            "query_popcount": int(query[5]),
            "threshold": f"{threshold:.2f}",
            "threshold_rational": [num, den],
            "lower_popcount": lower,
            "upper_popcount": upper,
            "counts": {
                "base_oracle": len(base_oracle),
                "union_oracle": len(union_oracle),
                "pre": len(pre),
                "post": len(post),
                "compacted": len(compact),
                "post_delta_segment": int(post_sources.get("delta", 0)),
            },
            "self_hits": {
                "base_oracle": sum(row[0] == int(query[0]) for row in base_oracle),
                "union_oracle": sum(row[0] == int(query[0]) for row in union_oracle),
                "pre": sum(row[0] == int(query[0]) for row in pre),
                "post": sum(row[0] == int(query[0]) for row in post),
                "compacted": sum(row[0] == int(query[0]) for row in compact),
            },
            "delta_id_hits": {
                "pre": pre_delta_id_hits,
                "post": post_delta_id_hits,
                "compacted": compact_delta_id_hits,
            },
            "source_hit_counts": {
                "pre": pre_sources,
                "post": post_sources,
                "compacted": compact_sources,
            },
            "outside_bound_count": outside_base
            + outside_union
            + outside_pre
            + outside_post
            + outside_compact,
            "duplicate_id_counts": duplicate_counts,
            "checks": {
                "pre_equals_base_oracle": pre_ok,
                "post_equals_union_oracle": post_ok,
                "compacted_equals_post": compact_ok,
                "no_staged_delta_visible_pre": pre_delta_id_hits == 0,
                "all_delta_hits_preserved_by_compaction": (
                    post_delta_id_hits == compact_delta_id_hits
                ),
            },
            "hashes": {
                "base_oracle": result_hash(base_oracle),
                "union_oracle": result_hash(union_oracle),
                "pre": result_hash(pre),
                "post": result_hash(post),
                "compacted": result_hash(compact),
            },
            "first_differences": {
                "pre_vs_base": first_difference(pre, base_oracle),
                "post_vs_union": first_difference(post, union_oracle),
                "compacted_vs_post": first_difference(compact, post),
            },
        }

    started = time.perf_counter()
    records: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for completed, record in enumerate(executor.map(validate, tasks), start=1):
            records.append(record)
            if completed % 32 == 0:
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(tasks),
                            "wall_s": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )

    records.sort(key=lambda row: (row["query_index"], row["threshold"]))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    exact_ok = all(
        record["checks"]["pre_equals_base_oracle"]
        and record["checks"]["post_equals_union_oracle"]
        and record["checks"]["compacted_equals_post"]
        and all(value == 0 for value in record["duplicate_id_counts"].values())
        for record in records
    )
    epoch_ok = (
        epoch0 == 0
        and staged_epoch == 0
        and epoch1 == published_epoch == 1
        and staged_delta_descriptor_count == 0
        and published_delta_descriptor_count == int(np.count_nonzero(delta_counts))
        and all(
            record["checks"]["no_staged_delta_visible_pre"]
            and record["counts"]["post_delta_segment"]
            == record["delta_id_hits"]["post"]
            and record["delta_id_hits"]["post"] > 0
            for record in records
        )
    )
    compaction_ok = (
        compaction["byte_identical_to_fresh_union"]
        and all(
            record["checks"]["compacted_equals_post"]
            and record["checks"]["all_delta_hits_preserved_by_compaction"]
            for record in records
        )
    )
    bound_ok = all(record["outside_bound_count"] == 0 for record in records)
    summary = {
        "experiment_id": "tide_20260827_segmented_lifecycle_exactness",
        "rows": {
            "base": int(base.shape[0]),
            "delta": int(delta.shape[0]),
            "union": int(union.shape[0]),
        },
        "correctness_queries": len(CORRECTNESS_QUERIES),
        "thresholds": [num / den for num, den in THRESHOLDS],
        "comparison_count": len(records),
        "workers": args.workers,
        "diagnostic_validation_wall_s": time.perf_counter() - started,
        "epochs": {
            "pre": epoch0,
            "staged_snapshot": staged_epoch,
            "published": published_epoch,
            "staged_delta_descriptor_count": staged_delta_descriptor_count,
            "published_delta_descriptor_count": published_delta_descriptor_count,
            "nonempty_delta_bins": int(np.count_nonzero(delta_counts)),
        },
        "id_overlap_base_delta": id_overlap,
        "id_disjointness": {
            "proof": id_disjoint_proof,
            "base_min": base_id_min,
            "base_max": base_id_max,
            "delta_min": delta_id_min,
            "delta_max": delta_id_max,
        },
        "per_bin_compaction": compaction,
        "mismatch_counts": {
            "pre_vs_base": sum(
                not row["checks"]["pre_equals_base_oracle"] for row in records
            ),
            "post_vs_union": sum(
                not row["checks"]["post_equals_union_oracle"] for row in records
            ),
            "compacted_vs_post": sum(
                not row["checks"]["compacted_equals_post"] for row in records
            ),
            "outside_bound": sum(
                row["outside_bound_count"] for row in records
            ),
        },
        "gates": {
            "G4_A_BOUND": bound_ok,
            "G4_A_EXACT": exact_ok,
            "G4_A_EPOCH": epoch_ok,
            "G4_A_COMPACTION": compaction_ok,
        },
        "notes": [
            "This is a CPU logical-organization validation, not concurrent GPU publication evidence.",
            "Compaction sorts independently within each of 257 bins; it does not perform one global population-count sort.",
            "Diagnostic elapsed time is not Gate-4 performance evidence.",
        ],
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
