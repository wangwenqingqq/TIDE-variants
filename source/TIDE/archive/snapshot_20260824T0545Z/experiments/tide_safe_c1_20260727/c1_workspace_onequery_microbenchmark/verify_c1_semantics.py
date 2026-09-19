#!/usr/bin/env python3
"""Exact, CPU-only comparator for C1 canonical result artifacts."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

VARIANTS = ('E_G_c1_off_reference', 'P_G_workspace_only', 'E_F_fastpath_only', 'P_F_full_C1')
PHASES = ('cold', 'provision', 'warmup', 'steady')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f'invalid JSON {path}: {exc}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', type=Path, required=True)
    ap.add_argument('--allow-smoke', action='store_true')
    args = ap.parse_args()
    root = args.run_root.resolve()
    reports = []
    ok = True
    for rep in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith('rep')):
        expected = [rep / v for v in VARIANTS]
        if not all(p.is_dir() for p in expected):
            ok = False
            reports.append({'replicate_dir': str(rep), 'pass': False, 'reason': 'missing one or more variants'})
            continue
        cards = {v: require_json(rep / v / 'run_card.json') for v in VARIANTS}
        completions = {v: require_json(rep / v / 'completion.json') for v in VARIANTS}
        mode = {cards[v].get('run_mode') for v in VARIANTS}
        if mode != {'MEASURED_PROTOCOL'} and not (args.allow_smoke and len(mode) == 1):
            ok = False
            reports.append({'replicate_dir': str(rep), 'pass': False, 'reason': f'invalid run mode(s): {sorted(mode)}'})
            continue
        tree = {v: completions[v].get('initial_tree_fingerprint') for v in VARIANTS}
        local_ok = all(completions[v].get('tree_invariance_pass') is True for v in VARIANTS) and len(set(tree.values())) == 1
        phase_hashes = {}
        for phase in PHASES:
            digests = {}
            for v in VARIANTS:
                artifact = rep / v / f'phase_{phase}_canonical.bin'
                if not artifact.is_file():
                    local_ok = False
                    digests[v] = None
                else:
                    digests[v] = sha256(artifact)
            phase_hashes[phase] = digests
            if len(set(digests.values())) != 1:
                local_ok = False
        reports.append({'replicate_dir': str(rep), 'pass': local_ok, 'initial_tree_fingerprints': tree,
                        'canonical_sha256_by_phase': phase_hashes})
        ok = ok and local_ok
    if not reports:
        raise SystemExit('no replicate directories found')
    out = {'schema': 'gtspp-c1-semantic-gate-v1', 'verified_utc': datetime.now(timezone.utc).isoformat(),
           'run_root': str(root), 'pass': ok, 'replicates': reports,
           'criterion': 'all variants share exact canonical binary output for every phase and identical initial/final tree fingerprints'}
    (root / 'semantic_verification.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'pass': ok, 'replicate_count': len(reports)}))
    return 0 if ok else 1

if __name__ == '__main__':
    raise SystemExit(main())
