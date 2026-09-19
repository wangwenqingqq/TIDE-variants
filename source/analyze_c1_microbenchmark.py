#!/usr/bin/env python3
"""CPU-only analyzer for the pre-registered C1 four-variant microbenchmark."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
from typing import Iterable

VARIANTS = ('E_G_c1_off_reference', 'P_G_workspace_only', 'E_F_fastpath_only', 'P_F_full_C1')
PHASES = ('cold', 'provision', 'warmup', 'steady_state', 'teardown')


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError('empty values')
    xs = sorted(values)
    position = (len(xs) - 1) * q
    lo, hi = int(position), min(int(position) + 1, len(xs) - 1)
    frac = position - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        'n': len(values), 'mean': statistics.fmean(values), 'median': quantile(values, .5),
        'p95': quantile(values, .95), 'p99': quantile(values, .99),
    }


def bootstrap_mean_ci(values: list[float], seed: int = 20260727, draws: int = 10000) -> dict[str, float | int]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(draws))
    return {'draws': draws, 'seed': seed, 'mean': statistics.fmean(values),
            'ci95_low': quantile(means, .025), 'ci95_high': quantile(means, .975)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', type=Path, required=True)
    args = ap.parse_args()
    root = args.run_root.resolve()
    semantic = read_json(root / 'semantic_verification.json')
    if semantic.get('pass') is not True:
        raise SystemExit('refusing statistics: semantic_verification.json is not PASS')
    per_variant: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    replicate_names = []
    for rep in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith('rep')):
        replicate_names.append(rep.name)
        for variant in VARIANTS:
            out = rep / variant
            card = read_json(out / 'run_card.json')
            completion = read_json(out / 'completion.json')
            if card.get('run_mode') != 'MEASURED_PROTOCOL':
                raise SystemExit(f'{out}: not a measured protocol run')
            if completion.get('tree_invariance_pass') is not True:
                raise SystemExit(f'{out}: tree-invariance failure')
            records = read_jsonl(out / 'phase_steady_ops.jsonl')
            if len(records) != 1024 or any(r.get('phase') != 'steady' for r in records):
                raise SystemExit(f'{out}: invalid steady-state record set')
            if any(r.get('allocation_delta', {}).get('failed_calls', 0) for r in records):
                raise SystemExit(f'{out}: allocator failure reported')
            e2e = [float(r['end_to_end_wall_ms']) for r in records]
            gpu = [float(r['gpu_event_ms']) for r in records]
            phase_alloc = read_json(out / 'allocation_by_phase.json')['phases']
            per_variant[variant].append({
                'replicate': rep.name,
                'steady_e2e_ms': summary(e2e),
                'steady_gpu_event_ms': summary(gpu),
                'allocation_by_phase': phase_alloc,
            })
    if len(replicate_names) != 5:
        raise SystemExit(f'expected exactly 5 measured replicates, found {len(replicate_names)}')
    aggregate = {}
    for variant, reps in per_variant.items():
        medians = [float(x['steady_e2e_ms']['median']) for x in reps]
        aggregate[variant] = {
            'replicate_median_e2e_ms': medians,
            'bootstrap_mean_of_replicate_medians': bootstrap_mean_ci(medians),
            'per_replicate': reps,
        }
    paired = []
    for i, name in enumerate(replicate_names):
        base = aggregate['E_G_c1_off_reference']['replicate_median_e2e_ms'][i]
        full = aggregate['P_F_full_C1']['replicate_median_e2e_ms'][i]
        paired.append({'replicate': name, 'E_G_over_P_F_median_ratio': base / full if full > 0 else None})
    output = {
        'schema': 'gtspp-c1-microbench-analysis-v1',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Measured C1 SIFT1M query-only microbenchmark only; no C2/C3/end-to-end claim.',
        'semantic_gate': semantic,
        'aggregate': aggregate,
        'primary_paired_descriptive_ratio': paired,
        'warning': 'Interpret only with the pre-registered timing boundary and successful exact canonical-result/tree gates.',
    }
    (root / 'analysis_summary.json').write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'pass': True, 'replicates': len(replicate_names), 'output': str(root / 'analysis_summary.json')}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
