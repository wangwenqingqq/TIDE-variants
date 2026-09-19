#!/usr/bin/env python3
"""Synthetic, ephemeral schema test for the Fable5 CPU selector.

This never opens a native snapshot, never selects an experiment candidate, and
never retains a trace/bundle artifact.  It only checks that in-memory fixture
slot IDs are encoded into the preflight-required ABI and candidate JSON shape.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path('/workspace/experiments/tide_safe_c1_20260730/safe_c1_fable5_matched_v1')
SELECTOR = ROOT / 'tools' / 'select_fable5_matched_trace_from_native_snapshot_v1.py'
CONTRACT = ROOT / 'manifests' / 'fable5_matched_trace_preflight_v1.json'


def load_selector():
    spec = importlib.util.spec_from_file_location('fable5_selector_schema_test_module', SELECTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    selector = load_selector()
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    h = contract['trace_abi']['header']
    header = {key: h[key] for key in ('dimension', 'base_n', 'reservoir_n', 'pool_n', 'query_n', 'k', 'radius')}
    slots = {
        'A': 4096, 'B': 4097, 'R': 4098, 'C': 4099,
        'query_A': 0, 'query_mid': 1, 'query_C': 2,
        'base_delete': 0, 'query_post': 2,
    }
    events = [
        (selector.OP_INSERT, slots['A']), (selector.OP_KNN, slots['query_A']),
        (selector.OP_INSERT, slots['B']), (selector.OP_INSERT, slots['R']),
        (selector.OP_KNN, slots['query_mid']), (selector.OP_DELETE, slots['A']),
        (selector.OP_DELETE, slots['B']), (selector.OP_INSERT, slots['C']),
        (selector.OP_KNN, slots['query_C']), (selector.OP_DELETE, slots['base_delete']),
        (selector.OP_REBUILD, 0), (selector.OP_KNN, slots['query_post']),
    ]
    selector.assert_exact_event_contract(events, slots, contract)
    trace = selector.encode_trace(header, events)
    assert len(trace) == 48 + 12 * 12
    with tempfile.TemporaryDirectory(prefix='fable5-selector-schema-only-') as temporary:
        trace_path = Path(temporary) / selector.TRACE_FILENAME
        trace_path.write_bytes(trace)
        decoded = selector.read_header(trace_path)
        assert decoded['event_count'] == 12
        assert decoded['dimension'] == 32 and decoded['pool_n'] == 6144
        trace_sha = hashlib.sha256(trace).hexdigest()
        candidate = selector.candidate_contract(contract, slots, trace_sha)
        assert candidate['schema'] == 'fable5-matched-trace-candidate-v1'
        assert candidate['status'] == 'CPU_CANDIDATE_NOT_NATIVE_RUNTIME_VALIDATED'
        assert candidate['trace'] == {'filename': selector.TRACE_FILENAME, 'sha256': trace_sha}
        assert candidate['trace_header'] == contract['trace_abi']['header']
        assert candidate['runtime_requirements'] == contract['runtime_requirements']
        assert candidate['static_certificate_outcomes'] == 'NOT_ESTABLISHED_STATICALLY'
        assert set(candidate['slot_ids']) == set(slots)
        # A malformed synthetic sequence must fail the exact fixed-contract gate.
        malformed = list(events)
        malformed[6] = (selector.OP_DELETE, slots['R'])
        try:
            selector.assert_exact_event_contract(malformed, slots, contract)
        except selector.Blocked:
            pass
        else:
            raise AssertionError('malformed synthetic 12-op sequence was accepted')

        # Machine-readable C2/C3 scope guards fire before any tree parsing or
        # candidate selection.  This JSON is intentionally incomplete and is
        # only an ephemeral negative schema fixture.
        snapshot_path = Path(temporary) / 'missing_c2c3_snapshot.json'
        snapshot_path.write_text(json.dumps({
            'schema': selector.SCHEMA,
            'status': 'NATIVE_E0_CAPTURED_PENDING_CPU_SELECTOR',
            'residual_pruning': {'mode': 0, 'query_executed': False},
        }), encoding='utf-8')
        try:
            selector.parse_snapshot(snapshot_path, ROOT, CONTRACT, ROOT / 'src/safe_c1_dynamic_gts.cu', SELECTOR, ROOT / 'src/fable5_matched_gpu_runner_v1.cu')
        except selector.Blocked as error:
            assert str(error) == 'snapshot_must_record_c2_used_false'
        else:
            raise AssertionError('snapshot missing c2_used=false was accepted')
    print('PASS_FABLE5_SELECTOR_SYNTHETIC_SCHEMA_ONLY')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
