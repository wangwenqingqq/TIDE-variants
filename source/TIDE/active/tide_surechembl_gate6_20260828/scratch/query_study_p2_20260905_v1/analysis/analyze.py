#!/usr/bin/env python3
"""Frozen P1 process-paired estimator; service ratios are peer / TIDE."""
import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

SEED = 20260905
RESAMPLES = 10000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)


def gmean(values):
    if not values or any(x <= 0 or not math.isfinite(x) for x in values):
        raise ValueError('finite positive observations required')
    return math.exp(statistics.mean(math.log(x) for x in values))


def interval(process_ratios):
    if len(process_ratios) != 6:
        raise ValueError('six complete process pairs required')
    rng = random.Random(SEED)
    logs = [math.log(x) for x in process_ratios]
    boot = [math.exp(statistics.mean(rng.choices(logs, k=6))) for _ in range(RESAMPLES)]
    return [quantile(boot, .025), quantile(boot, .975)]


def row_key(row):
    return (row['cohort'], row['batch'], row['threshold_num'], row['threshold_den'],
            row['query_start'], row['query_ids_sha256'])


def matched(left, right):
    if left.keys() != right.keys():
        raise ValueError('unmatched requests')
    for key in left:
        a, b = left[key], right[key]
        if a['counts'] != b['counts']:
            raise ValueError('answer counts differ')
        yield key, a, b


def read_process(stage, item):
    label = f"{item['comparison']}_pair{item['pair']}_slot{item['slot']}_{item['backend']}"
    path = stage / (label + '.jsonl')
    meta = json.loads(path.with_suffix('.metadata.json').read_text())
    guard = json.loads(path.with_suffix('.guard.json').read_text())
    if (not meta['timing_admissible'] or meta['mode'] != 'timing' or
            meta['backend'] != item['backend'] or meta['warmup_batches_per_cell'] != 16 or
            meta['reverse'] != item['reverse'] or meta['rows'] != 1048576):
        raise ValueError('metadata contract mismatch: ' + label)
    if (guard['status'] != 'complete' or guard['exit_code'] != 0 or
            guard['before']['apps'] or guard['after']['apps'] or
            guard.get('foreign_gpu_processes', ['missing-monitor'])):
        raise ValueError('guard not clean: ' + label)
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if (row['backend'] != item['backend'] or row['mode'] != 'timing' or
                not row['complete_oracle_equal'] or row['repetition'] != 0 or
                not 0 < row['service_ms'] < float('inf') or len(row['counts']) != row['batch']):
            raise ValueError('invalid observation: ' + label)
        key = row_key(row)
        if key in rows:
            raise ValueError('duplicate request')
        rows[key] = row
    cells = defaultdict(list)
    for key, row in rows.items():
        cells[key[:4]].append(row)
    if len(cells) != 12 or len(rows) != 2336 or meta['batch_records'] != len(rows):
        raise ValueError('missing cells/records')
    for cell, values in cells.items():
        if (cell[1] not in [1, 8, 64] or cell[2:] not in [(7, 10), (4, 5)] or
                len(values) != 512 // cell[1]):
            raise ValueError('incomplete cell')
    return rows, meta, digest(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('never replace a prior analysis')
    stage = args.stage.resolve()
    completion = json.loads((stage / 'COMPLETE.json').read_text())
    if not completion['complete'] or completion['processes'] != 24:
        raise ValueError('formal campaign incomplete')
    gates = json.loads((stage.parent / 'validate/GATES.json').read_text())
    if not gates['complete'] or len(gates['invocations']) != 15:
        raise ValueError('validation incomplete')
    plan = json.loads((stage / 'PLAN.json').read_text())
    if len(plan) != 24:
        raise ValueError('plan incomplete')
    process = {}
    hashes = {}
    invariants = set()
    for item in plan:
        key = item['comparison'], item['pair'], item['backend']
        if key in process:
            raise ValueError('duplicate process')
        rows, meta, sha = read_process(stage, item)
        process[key] = rows
        hashes[str(key)] = sha
        invariants.add(tuple(meta[x] for x in ['data_manifest_sha256', 'oracle_sha256',
                                              'query_sha256', 'gpu_uuid', 'script_sha256']))
    if len(invariants) != 1:
        raise ValueError('cross-process input, device, or driver drift')
    results = []
    for peer in ['nv_row', 'tide_bounded']:
        ratios = defaultdict(list)
        samples = defaultdict(lambda: [[], []])
        completions = defaultdict(lambda: [[], []])
        throughput = defaultdict(lambda: [[], []])
        for pair in range(1, 7):
            a = process[peer, pair, peer]
            b = process[peer, pair, 'tide_batch']
            per_cell = defaultdict(list)
            local_samples = defaultdict(lambda: [[], []])
            for key, left, right in matched(a, b):
                cell = key[:4]
                per_cell[cell].append(left['service_ms'] / right['service_ms'])
                for i, row in enumerate([left, right]):
                    samples[cell][i].append(row['service_ms'])
                    local_samples[cell][i].append(row['service_ms'])
                    if row['individual_completion_ms'] is not None:
                        completions[cell][i].extend(row['individual_completion_ms'])
            for cell, values in per_cell.items():
                ratios[cell].append(gmean(values))
                for i in range(2):
                    throughput[cell][i].append(512000 / sum(local_samples[cell][i]))
        for cell in sorted(ratios):
            values = ratios[cell]
            lo, hi = interval(values)
            row = dict(comparison=peer, cohort=cell[0], batch=cell[1],
                       threshold=f'{cell[2]}/{cell[3]}', paired_gmean=gmean(values),
                       process_cluster_95pct=[lo, hi], process_ratios=values,
                       tide_process_wins=sum(x > 1 for x in values),
                       order_peer_first_gmean=gmean(values[::2]),
                       order_tide_first_gmean=gmean(values[1::2]),
                       marginal_median_ratio=statistics.median(samples[cell][0]) /
                                             statistics.median(samples[cell][1]))
            for i, name in enumerate(['peer', 'tide']):
                row[name + '_service_ms_p10_p50_p90'] = [quantile(samples[cell][i], q) for q in [.1, .5, .9]]
                row[name + '_process_throughput_queries_s'] = throughput[cell][i]
                row[name + '_completion_ms_p50_p95'] = ([quantile(completions[cell][i], q) for q in [.5, .95]]
                                                        if completions[cell][i] else None)
            row['screen_direction'] = ('tide_margin' if row['paired_gmean'] >= 1.2 and lo > 1
                                       else 'peer_margin' if row['paired_gmean'] <= 1 / 1.2 and hi < 1
                                       else 'no_registered_margin')
            results.append(row)
    report = {'scope': '1M-row 256-bit complete adapter service; not kernel-only or full-scale',
              'estimator': 'per-cell GM of matched request ratios within pair, then GM across six process pairs',
              'ratio_direction': 'peer / tide_batch; above one favors TIDE',
              'bootstrap_seed': SEED, 'bootstrap_resamples': RESAMPLES,
              'multiplicity': 'exploratory per-cell intervals; not simultaneous confidence or novelty proof',
              'whole_device_peak': 'not measured; pre/post snapshots are not peak memory',
              'raw_sha256': hashes, 'analyzer_sha256': digest(__file__), 'cells': results}
    args.out.mkdir(parents=True)
    (args.out / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# P2 ordinary-batch falsification', '', 'Ratio = peer / queued-batch TIDE. All six process pairs retained.',
             'Exploratory process-cluster intervals; no novelty or full-scale claim.', '',
             '| Peer | Cohort | Q | Threshold | Paired GM [95%] | TIDE process wins | Peer / TIDE median ms |',
             '|---|---|---:|---|---|---:|---|']
    for row in results:
        lo, hi = row['process_cluster_95pct']
        lines.append(f"| {row['comparison']} | {row['cohort']} | {row['batch']} | {row['threshold']} | "
                     f"{row['paired_gmean']:.4f} [{lo:.4f}, {hi:.4f}] | {row['tide_process_wins']}/6 | "
                     f"{row['peer_service_ms_p10_p50_p90'][1]:.4f} / {row['tide_service_ms_p10_p50_p90'][1]:.4f} |")
    (args.out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'cells': len(results), 'out': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
