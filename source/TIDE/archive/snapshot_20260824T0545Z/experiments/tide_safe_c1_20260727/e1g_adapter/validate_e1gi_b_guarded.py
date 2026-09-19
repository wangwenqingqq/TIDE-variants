#!/usr/bin/env python3
"""Adapter that makes the independent quantized oracle satisfy the guarded launcher contract."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/workspace/experiments/tide_safe_c1_20260727')
RAW = ROOT / 'e1g_adapter' / 'verify_quantized_gts_integration_export.py'

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    engine = a.candidate / 'engine_results.jsonl'
    raw_out = a.out.with_name(a.out.stem + '.raw_quantized_oracle.json')
    proc = subprocess.run([sys.executable, str(RAW), '--prepared-bundle', str(a.bundle),
                           '--engine-jsonl', str(engine), '--out', str(raw_out)], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if raw_out.exists():
        raw = json.loads(raw_out.read_text(encoding='utf-8'))
    else:
        raw = {'pass': False, 'errors': [{'type': 'raw_validator_no_output', 'stderr': proc.stderr[-1000:]}]}
    errors = raw.get('errors', [])
    topk = sum(1 for e in errors if str(e.get('type', '')).startswith('knn_'))
    range_missing = sum(int(e.get('counts', {}).get('missing', 0)) for e in errors if e.get('type') == 'range_mismatch')
    range_extra = sum(int(e.get('counts', {}).get('extra', 0)) for e in errors if e.get('type') == 'range_mismatch')
    passed = bool(raw.get('pass')) and proc.returncode == 0
    result = {
        'schema': 'e1gi-b-guarded-validator-adapter-v1',
        'status': 'PASS' if passed else 'FAIL',
        'validator': 'independent_exact_oracle',
        'topk_ranking_mismatches': topk,
        'range_missing_ids': range_missing,
        'range_extra_ids': range_extra,
        'raw_quantized_oracle': str(raw_out),
        'raw_pass': raw.get('pass', False),
        'raw_error_count': raw.get('error_count', len(errors)),
        'raw_validator_exit': proc.returncode,
        'raw_validator_stderr_tail': proc.stderr[-1000:],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'raw_pass': result['raw_pass'], 'out': str(a.out)}, sort_keys=True))
    return 0 if passed else 2

if __name__ == '__main__':
    raise SystemExit(main())
