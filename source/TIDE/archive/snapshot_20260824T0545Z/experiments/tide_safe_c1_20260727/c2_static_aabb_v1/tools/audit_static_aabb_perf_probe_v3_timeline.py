#!/usr/bin/env python3
"""CPU-only source audit for the independent v3 timeline probe.

This tool only reads source files and writes its requested JSON result.  It does
not invoke nvcc, any binary, CUDA runtime, or nvidia-smi.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ROOT = Path('/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1')
OLD = ROOT / 'src/static_aabb_perf_probe_v2_1_diskguard.cu'
NEW = ROOT / 'src/static_aabb_perf_probe_v3_timeline.cu'
EXPECTED_OLD_SHA256 = '5dc92a9660bc168e1325edae7a1c9d41f59a93e036e83f57f2ccd9d55fa6819f'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def between(text: str, begin: str, end: str) -> str:
    start = text.index(begin)
    stop = text.index(end, start)
    return text[start:stop]


def check(condition: bool, name: str, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    if not condition:
        raise AssertionError(name)


def audit() -> dict:
    old = OLD.read_text()
    new = NEW.read_text()
    checks: dict[str, bool] = {}
    check(sha256(OLD) == EXPECTED_OLD_SHA256, 'archived_v2_1_source_sha256_matches', checks)

    # No existing computational helper (read/build/tree/search/evaluation/result)
    # may change.  v3 inserts its timeline support just before read_base().
    check(
        between(old, 'std::vector<float> read_base', 'int main(')
        == between(new, 'std::vector<float> read_base', 'int main('),
        'all_non_main_computational_helpers_byte_identical',
        checks,
    )

    # Setup through the warmup loop remains byte-identical.  The only main-body
    # change is the post-warmup measurement instrumentation.
    check(
        between(old, 'int main(', '  for(int i=0;i<a.warmups;++i)')
        == between(new, 'int main(', '  for(int i=0;i<a.warmups;++i)'),
        'main_setup_through_warmup_loop_byte_identical',
        checks,
    )
    for label, begin, end in (
        ('kernel_dispatch_helpers_byte_identical', 'void run_baseline(', 'double median('),
        ('v2_result_writer_byte_identical', 'void write_perf_result(', '\n\nint main('),
    ):
        check(between(old, begin, end) == between(new, begin, end), label, checks)

    # Adding a timeline must not add/change CUDA API invocations.
    old_cuda_calls = re.findall(r'\bcuda[A-Za-z0-9_]+\s*\(', old)
    new_cuda_calls = re.findall(r'\bcuda[A-Za-z0-9_]+\s*\(', new)
    check(old_cuda_calls == new_cuda_calls, 'cuda_api_call_sequence_byte_identical', checks)

    required = (
        '--run-token',
        'CLOCK_MONOTONIC',
        'timed_envelope_start_ns',
        'timed_envelope_end_ns',
        'benchmark_timeline.json',
        'op_index',
        'host_start_ns',
        'host_end_ns',
        'cuda_ms',
        'PASS_TIMELINE_60_OPERATIONS',
        'a.reps!=30',
        'ops.size()!=60',
        'O_EXCL',
        '::fsync',
        '::rename',
    )
    for token in required:
        check(token in new, f'contains_{token}', checks)

    loop_at = new.index('for(int rep=0;rep<a.reps;++rep){if((rep&1)==0){record_baseline')
    write_at = new.index('write_benchmark_timeline_atomic(a,ops,envelope_start,envelope_end);')
    snapshot_at = new.index('const Snapshot final=capture_snapshot', loop_at)
    check(loop_at < write_at < snapshot_at, 'timeline_atomic_publish_after_measurements_before_snapshot', checks)

    alternating = (
        'for(int rep=0;rep<a.reps;++rep){if((rep&1)==0){record_baseline(rep,0);'
        'record_aabb(rep,1);}else{record_aabb(rep,0);record_baseline(rep,1);}}'
    )
    check(alternating in new, 'v2_1_alternating_pair_order_preserved', checks)
    check(new.count('searchIndexKnnV2(') == old.count('searchIndexKnnV2(') == 1,
          'baseline_kernel_dispatch_count_unchanged', checks)
    check(new.count('searchIndexKnnStaticAabbV2DiskGuard(')
          == old.count('searchIndexKnnStaticAabbV2DiskGuard(') == 1,
          'static_aabb_kernel_dispatch_count_unchanged', checks)

    return {
        'schema': 'safe-c2-static-aabb-perf-timeline-v3-source-audit-v1',
        'status': 'PASS_SOURCE_ONLY_AUDIT',
        'scope': (
            'source-only audit of independent v3 timeline instrumentation; '
            'no compilation, binary execution, CUDA runtime, nvidia-smi, or GPU use'
        ),
        'gpu_executed': False,
        'nvidia_smi_called': False,
        'source_v2_1': {'path': str(OLD), 'sha256': sha256(OLD)},
        'source_v3': {'path': str(NEW), 'sha256': sha256(NEW)},
        'checks': checks,
        'checks_passed': len(checks),
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    fd, temp = tempfile.mkstemp(prefix=path.name + '.tmp.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit()
    if args.out is None:
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        atomic_json(args.out, result)
        print(json.dumps({'status': result['status'], 'out': str(args.out)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
