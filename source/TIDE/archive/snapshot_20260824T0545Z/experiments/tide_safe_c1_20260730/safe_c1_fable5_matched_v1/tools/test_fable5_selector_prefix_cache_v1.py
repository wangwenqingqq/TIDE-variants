#!/usr/bin/env python3
"""Synthetic CPU-only regression test for Fable5 selector prefix caching.

For every tested active-set shape, compare PrefixRanker against a full sort.
This is an implementation check only: no native snapshot, CUDA binary, GPU,
NVML, trace candidate, or experiment artifact is created.
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

ROOT = Path('/workspace/experiments/tide_safe_c1_20260730/safe_c1_fable5_matched_v1')
SELECTOR = ROOT / 'tools' / 'select_fable5_matched_trace_from_native_snapshot_v1.py'


def load_selector():
    spec = importlib.util.spec_from_file_location('fable5_selector_prefix_cache_module', SELECTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def full_exact(module, active, pool, queries, dimension, query_id):
    return module.sorted_exact(active, pool, queries, dimension, query_id)


def full_modeled(module, active, pool, queries, dimension, query_id):
    return sorted((module.modeled_gts_l2(pool, queries, dimension, ident, query_id), ident) for ident in active)


def main() -> int:
    m = load_selector()
    rng = random.Random(20260730)
    dimension, base_n, pool_n, query_n, k = 5, 23, 29, 3, 7
    # The large value range makes accidental integer L2 ties highly unlikely;
    # equality would still be compared deterministically by stable ID.
    pool = [rng.randrange(-10_000, 10_001) for _ in range(pool_n * dimension)]
    queries = [rng.randrange(-10_000, 10_001) for _ in range(query_n * dimension)]
    base = set(range(base_n))
    ranker = m.PrefixRanker(base, pool, queries, dimension, k)
    assert ranker.prefix_length == k + 2

    # No deletion may add up to three objects; one deletion may add up to two.
    no_delete_cases = [(), (23,), (23, 24), (23, 24, 25)]
    delete_cases = [(), (23,), (23, 24)]
    checked = 0
    for extras in no_delete_cases:
        active = base | set(extras)
        for query_id in range(query_n):
            got_exact = ranker._merged(extras, query_id, None, modeled=False)[: k + 1]
            got_modeled = ranker._merged(extras, query_id, None, modeled=True)[: k + 1]
            assert got_exact == full_exact(m, active, pool, queries, dimension, query_id)[: k + 1]
            assert got_modeled == full_modeled(m, active, pool, queries, dimension, query_id)[: k + 1]
            checked += 2
    for removed in sorted(base):
        for extras in delete_cases:
            active = (base - {removed}) | set(extras)
            for query_id in range(query_n):
                got_exact = ranker._merged(extras, query_id, removed, modeled=False)[: k + 1]
                got_modeled = ranker._merged(extras, query_id, removed, modeled=True)[: k + 1]
                assert got_exact == full_exact(m, active, pool, queries, dimension, query_id)[: k + 1]
                assert got_modeled == full_modeled(m, active, pool, queries, dimension, query_id)[: k + 1]
                checked += 2
    print(f'PASS_FABLE5_SELECTOR_PREFIX_CACHE_SYNTHETIC checks={checked}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
