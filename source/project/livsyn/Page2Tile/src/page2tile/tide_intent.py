"""Exact population-count routing metrics for binary fingerprints."""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Sequence, Tuple


def eligible_popcount_range(
    query_popcount: int,
    numerator: int,
    denominator: int,
    width_bits: int = 256,
) -> Tuple[int, int]:
    """Return the exact Tanimoto-feasible target-popcount interval."""
    if not 0 <= query_popcount <= width_bits:
        raise ValueError("query_popcount is outside the fingerprint width")
    if numerator <= 0 or denominator <= 0 or numerator > denominator:
        raise ValueError("threshold must satisfy 0 < numerator <= denominator")
    lower = (numerator * query_popcount + denominator - 1) // denominator
    upper = (denominator * query_popcount) // numerator
    return max(0, lower), min(width_bits, upper)


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "n": 0,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right) or not left:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    if left_ss == 0.0 or right_ss == 0.0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def _spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _top_share(values: Sequence[int], fraction: float) -> float:
    positive = sorted((value for value in values if value > 0), reverse=True)
    if not positive:
        return 0.0
    count = max(1, math.ceil(len(positive) * fraction))
    return sum(positive[:count]) / sum(positive)


def summarize_windows(
    query_popcounts: Sequence[int],
    bin_rows: Sequence[int],
    numerator: int,
    denominator: int,
    window_size: int,
    fingerprint_bytes_per_row: int = 32,
) -> Dict[str, object]:
    """Summarize non-overlapping routing windows without reordering queries."""
    if len(bin_rows) < 257:
        raise ValueError("bin_rows must contain entries for popcounts 0 through 256")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    windows: List[Dict[str, object]] = []
    hotness_vectors: List[List[int]] = []
    total_pairs = 0
    grouped_rows_total = 0
    fanin_pair_totals = {2: 0, 4: 0, 8: 0}
    mma_useful_pairs = 0
    mma_padded_pairs = 0

    for start in range(0, len(query_popcounts), window_size):
        current = query_popcounts[start : start + window_size]
        fanin = [0] * len(bin_rows)
        for query_popcount in current:
            lower, upper = eligible_popcount_range(
                int(query_popcount), numerator, denominator, len(bin_rows) - 1
            )
            for popcount in range(lower, upper + 1):
                if bin_rows[popcount] > 0:
                    fanin[popcount] += 1

        edge_count = sum(fanin)
        candidate_pairs = sum(
            fanin[index] * int(bin_rows[index]) for index in range(len(bin_rows))
        )
        grouped_rows = sum(
            int(bin_rows[index]) for index in range(len(bin_rows)) if fanin[index] > 0
        )
        active_bins = sum(1 for value in fanin if value > 0)
        weighted_hotness = [
            fanin[index] * int(bin_rows[index]) for index in range(len(bin_rows))
        ]
        pair_by_fanin = {
            threshold: sum(
                fanin[index] * int(bin_rows[index])
                for index in range(len(bin_rows))
                if fanin[index] >= threshold
            )
            for threshold in fanin_pair_totals
        }

        window_mma_useful = 0
        window_mma_padded = 0
        for index, query_fanin in enumerate(fanin):
            rows = int(bin_rows[index])
            if rows <= 0 or query_fanin < 8:
                continue
            window_mma_useful += rows * query_fanin
            window_mma_padded += (
                ((rows + 15) // 16) * 16 * ((query_fanin + 7) // 8) * 8
            )

        windows.append(
            {
                "start_query": start,
                "query_count": len(current),
                "query_partition_edges": edge_count,
                "active_nonempty_bins": active_bins,
                "candidate_query_pairs": candidate_pairs,
                "logical_query_major_fingerprint_bytes": (
                    candidate_pairs * fingerprint_bytes_per_row
                ),
                "logical_grouped_once_fingerprint_bytes": (
                    grouped_rows * fingerprint_bytes_per_row
                ),
                "logical_grouping_ratio": (
                    candidate_pairs / grouped_rows if grouped_rows else None
                ),
                "pair_fraction_by_min_fanin": {
                    str(threshold): (
                        pair_by_fanin[threshold] / candidate_pairs
                        if candidate_pairs
                        else 0.0
                    )
                    for threshold in sorted(pair_by_fanin)
                },
                "top_partition_share_by_edges": {
                    "1pct": _top_share(fanin, 0.01),
                    "5pct": _top_share(fanin, 0.05),
                    "10pct": _top_share(fanin, 0.10),
                },
                "top_partition_share_by_candidate_pairs": {
                    "1pct": _top_share(weighted_hotness, 0.01),
                    "5pct": _top_share(weighted_hotness, 0.05),
                    "10pct": _top_share(weighted_hotness, 0.10),
                },
                "mma_n8_useful_pair_fraction": (
                    window_mma_useful / candidate_pairs if candidate_pairs else 0.0
                ),
                "mma_m16n8_logical_tile_occupancy": (
                    window_mma_useful / window_mma_padded
                    if window_mma_padded
                    else None
                ),
            }
        )
        hotness_vectors.append(weighted_hotness)
        total_pairs += candidate_pairs
        grouped_rows_total += grouped_rows
        for threshold in fanin_pair_totals:
            fanin_pair_totals[threshold] += pair_by_fanin[threshold]
        mma_useful_pairs += window_mma_useful
        mma_padded_pairs += window_mma_padded

    adjacent_jaccard: List[float] = []
    adjacent_spearman: List[float] = []
    for left, right in zip(hotness_vectors, hotness_vectors[1:]):
        left_active = {index for index, value in enumerate(left) if value > 0}
        right_active = {index for index, value in enumerate(right) if value > 0}
        union = left_active | right_active
        adjacent_jaccard.append(
            len(left_active & right_active) / len(union) if union else 1.0
        )
        correlation = _spearman(left, right)
        if correlation is not None:
            adjacent_spearman.append(correlation)

    return {
        "window_size": window_size,
        "window_count": len(windows),
        "query_count": len(query_popcounts),
        "aggregate": {
            "candidate_query_pairs": total_pairs,
            "logical_query_major_fingerprint_bytes": (
                total_pairs * fingerprint_bytes_per_row
            ),
            "logical_grouped_once_fingerprint_bytes": (
                grouped_rows_total * fingerprint_bytes_per_row
            ),
            "logical_grouping_ratio": (
                total_pairs / grouped_rows_total if grouped_rows_total else None
            ),
            "pair_fraction_by_min_fanin": {
                str(threshold): (
                    fanin_pair_totals[threshold] / total_pairs if total_pairs else 0.0
                )
                for threshold in sorted(fanin_pair_totals)
            },
            "mma_n8_useful_pair_fraction": (
                mma_useful_pairs / total_pairs if total_pairs else 0.0
            ),
            "mma_m16n8_logical_tile_occupancy": (
                mma_useful_pairs / mma_padded_pairs if mma_padded_pairs else None
            ),
            "active_nonempty_bins": _summary(
                [float(window["active_nonempty_bins"]) for window in windows]
            ),
            "logical_grouping_ratio_per_window": _summary(
                [float(window["logical_grouping_ratio"]) for window in windows]
            ),
            "adjacent_window_active_bin_jaccard": _summary(adjacent_jaccard),
            "adjacent_window_byte_hotness_spearman": _summary(adjacent_spearman),
            "top_partition_share_by_edges": {
                label: _summary(
                    [
                        float(window["top_partition_share_by_edges"][label])
                        for window in windows
                    ]
                )
                for label in ("1pct", "5pct", "10pct")
            },
            "top_partition_share_by_candidate_pairs": {
                label: _summary(
                    [
                        float(window["top_partition_share_by_candidate_pairs"][label])
                        for window in windows
                    ]
                )
                for label in ("1pct", "5pct", "10pct")
            },
        },
        "windows": windows,
    }
