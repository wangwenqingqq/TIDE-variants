#!/usr/bin/env python3
"""CPU-only fail-closed static/data audit for isolated Safe-C2 v2 tie-free."""
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/experiments/tide_safe_c1_20260727/c2_interval_bound_safe_v2_tiefree')
PYTHON = Path('/workspace/legacy_workspace/GTS/bench_env/bin/python')
SRC = ROOT / 'src/gts_safe_c2_v2_tiefree_sift1m.cu'
SEARCH = ROOT / 'include/search_v2.cuh'
TREE = ROOT / 'include/tree.cuh'
RESIDUAL = ROOT / 'include/residual_pruning.cuh'
CONFIG = ROOT / 'include/config.cuh'
GUARD = ROOT / 'tools/run_safe_c2_v2_tiefree_guarded.sh'
VERIFY = ROOT / 'tools/verify_tiefree_v2.py'
GENERATOR = ROOT / 'tools/generate_tiefree_v2_artifacts.py'
PROTOCOL = ROOT / 'protocols/safe_c2_interval_sift1m_v2_tiefree.json'
PREFLIGHT = ROOT / 'preflight/sift1m_tiefree_v2_20260727T102548Z/preflight_safe_c2_tiefree_v2.json'
OUT = ROOT / 'logs/static_audit_safe_c2_v2_tiefree_latest.json'
VERIFY_OUT = ROOT / 'logs/tiefree_reverification_static_audit.json'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    failures: list[str] = []
    required_paths = (SRC, SEARCH, TREE, RESIDUAL, CONFIG, GUARD, VERIFY, GENERATOR, PROTOCOL, PREFLIGHT)
    for path in required_paths:
        require(path.is_file(), f'missing required artifact: {path}', failures)
    if failures:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({'status': 'FAIL', 'cuda_used': False, 'failures': failures}, indent=2) + '\n')
        return 2

    src, search, tree, residual, config, guard = (x.read_text() for x in (SRC, SEARCH, TREE, RESIDUAL, CONFIG, GUARD))
    protocol = json.loads(PROTOCOL.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    require(protocol.get('schema') == 'safe-c2-corrected-after-submit-tiefree-protocol-v2', 'wrong v2 protocol schema', failures)
    require(preflight.get('schema') == 'safe-c2-corrected-after-submit-tiefree-preflight-v2', 'wrong v2 preflight schema', failures)
    require(protocol.get('implementation_identity', {}).get('corrected_after_submit') is True, 'protocol lacks corrected-after-submit identity', failures)
    require(protocol.get('implementation_identity', {}).get('not_submitted_artifact') is True, 'protocol does not disavow submitted-artifact equivalence', failures)
    require(protocol.get('implementation_identity', {}).get('v2_tiefree') is True, 'protocol lacks v2 tie-free identity', failures)
    require(protocol.get('data', {}).get('preflight_path') == str(PREFLIGHT), 'protocol preflight path mismatch', failures)
    require(protocol.get('data', {}).get('preflight_sha256') == sha(PREFLIGHT), 'protocol preflight SHA mismatch', failures)
    require(protocol.get('timing_protocol', {}).get('heldout_pairing', {}).get('required') is True, 'protocol lacks mandatory ABBA heldout pairing', failures)

    split = protocol.get('split', {})
    expect = {'calibration': 1966, 'validation': 1976, 'test': 5921}
    for part, count in expect.items():
        entry = split.get('files', {}).get(part, {})
        fp = Path(entry.get('path', ''))
        require(fp.is_file(), f'missing v2 selected IDs {part}', failures)
        if fp.is_file():
            ids = [int(v) for v in fp.read_text().split()]
            require(len(ids) == count == entry.get('count'), f'{part} v2 count mismatch', failures)
            require(len(ids) == len(set(ids)), f'{part} v2 duplicate IDs', failures)
            require(sha(fp) == entry.get('sha256'), f'{part} v2 IDs SHA mismatch', failures)
        ex = split.get('excluded_ambiguous_files', {}).get(part, {})
        ep = Path(ex.get('path', ''))
        require(ep.is_file() and ex.get('count') == {'calibration':34,'validation':24,'test':79}[part],
                f'{part} exclusion artifact missing/count mismatch', failures)
        if ep.is_file():
            require(sha(ep) == ex.get('sha256'), f'{part} exclusion SHA mismatch', failures)
    require(split.get('selected_total') == 9863 and split.get('excluded_total') == 137, 'v2 split total mismatch', failures)
    require(split.get('membership_preserved_after_exclusion') is True, 'v2 does not state membership preservation', failures)
    require(split.get('exhaustive_over_original_10000') is False, 'v2 falsely claims exhaustive 10k coverage', failures)

    # Exact Safe-C2 vector Eq. (1)/(2) contract is unchanged from v1.
    start = search.find('__global__ void nodeProcessKnn(')
    end = search.find('// Initialize p_list.', start)
    kernel = search[start:end] if start >= 0 and end > start else ''
    require(start >= 0 and end > start, 'cannot isolate nodeProcessKnn kernel', failures)
    require('const float *max_dis_d' in kernel, 'nodeProcessKnn lacks max_dis_d parameter', failures)
    require(kernel.count('node_list[nid + 1].min_dis') == 0, 'v2 vector nodeProcessKnn still uses successor-min upper bound', failures)
    require(kernel.count('const float lower_from_max = dis_q - max_dis_d[nid];') == 2, 'both vector branches must use max_dis_d', failures)
    require(kernel.count('fmaxf(0.0f, fmaxf(lower_from_min, lower_from_max))') == 2, 'both vector branches must form Eq.1 LB', failures)
    launches = re.findall(r'nodeProcessKnn<<<[^>]+>>>\(([^\n]+)', search)
    require(len(launches) == 4, f'expected 4 nodeProcessKnn launch sites, found {len(launches)}', failures)
    require(all(re.match(r'\s*node_list\s*,\s*max_dis_d\s*,\s*disk\s*,', x) for x in launches), 'one or more vector launch sites omit/max-order max_dis_d', failures)
    require(search.count('node_list[nid + 1].min_dis') == 1, 'unexpected copied RNN successor-min count', failures)
    require('nodeProcessRnn<<<' not in src, 'v2 runner launches out-of-scope RNN path', failures)
    require('max_dis_d[id_child] = max_result;' in tree, 'tree source lacks max_dis_d construction', failures)
    require('(scale - 1.0f) * dis_lb' in residual, 'rp_predict is not Eq.2 scale delta', failures)
    for forbidden in ('zero false negatives', 'Online Adaptation', 'gamma ~ 0.5', 'Piecewise-linear LUT for enhanced inference'):
        require(forbidden not in residual, f'residual header contains forbidden legacy claim: {forbidden}', failures)

    # Runner remains Safe-C2 but gains failure observability only.
    for required in (
        'certify_and_outward_repair_intervals', 'post-repair directional interval cover verification failed',
        'per_query_no_regression', 'run_paired_abba_stage',
        'safe-c2-corrected-after-submit-tiefree-run-v2', 'v2_tiefree',
        'write_baseline_invalid_diagnostic', 'safe-c2-tiefree-baseline-invalid-diagnostic-v2',
        'no_gamma_candidate_was_issued', 'baseline_invalid_diagnostic.json',
        'SafeC2EffectiveCoordinate', 'static_assert(std::is_same<SafeC2EffectiveCoordinate, float>::value',
    ):
        require(required in src, f'v2 runner lacks required contract: {required}', failures)
    for forbidden in ('#include "update.cuh"', '#include "incremental_insert.cuh"', 'submitted-vldb-c2-perlevel-run-v1'):
        require(forbidden not in src, f'v2 runner contains forbidden legacy/non-Safe identity: {forbidden}', failures)
    require('#define short float' in config, 'config does not freeze effective float32 type', failures)

    # Guard must run the exact raw re-verifier before all nvidia-smi/lock paths,
    # poll strict idle before a phase, and retain post-phase telemetry without
    # interpreting a transient utilization percentage as a result failure.
    for required in (
        'verify_frozen_data_and_tiefree()', 'TIEFREE_VERIFY=',
        'physical_nvidia_smi()', 'gpu0_snapshot_identity_and_no_compute()',
        'wait_for_gpu0_strict_idle()', 'GPU0_IDLE_POLL_ATTEMPTS=15',
        'GPU0_IDLE_POLL_SECONDS=2', 'setsid --wait /bin/bash -c',
        'child_session_verified_before_cuda_exec=PASS',
        'NVIDIA_VISIBLE_DEVICES="$GPU0_UUID"',
        'gpu0_snapshot_identity_and_no_compute "after_${phase}" "$phase_dir"',
        'wait_for_gpu0_strict_idle "immediately_pre_${phase}" "$phase_dir"',
        'SAFE_C2_V2_TIEFREE_GPU0_APPROVED',
        '[[ ! -e "$OUT" ]] || die', '[[ ! -e "$PRE" ]] || die',
        'TERM_then_reap_wait_before_lock_release',
    ):
        require(required in guard, f'v2 guard lacks required hardening: {required}', failures)
    execute_anchor = guard.find('static_check "$PRE/static_audit.log"')
    verify_anchor = guard.find('verify_frozen_data_and_tiefree "$PRE/data_hash_verification.env"')
    gpu_anchor = guard.find('\nresolve_physical_gpu0_uuid\n', verify_anchor)
    lock_anchor = guard.find('if ! mkdir "$LOCK"')
    require(execute_anchor >= 0 and execute_anchor < verify_anchor < gpu_anchor < lock_anchor,
            'guard does not reverify raw tie-free condition before GPU query/lock', failures)
    phase = guard[guard.find('run_phase()'):guard.find('[[ $# -eq 1 ]]')]
    require(phase.find('wait_for_gpu0_strict_idle "immediately_pre_${phase}"') < phase.find('launch_session_guarded_command'),
            'guard can launch CUDA before bounded strict-idle poll', failures)
    require(phase.find('gpu0_snapshot_identity_and_no_compute "after_${phase}"') > phase.find('wait "$CHILD_LAUNCHER_PID"'),
            'post phase does not use telemetry-only identity/no-compute check', failures)

    # Run exact CPU raw admission verifier now; no CUDA/nvidia-smi API is used.
    if not failures:
        proc = subprocess.run(
            [str(PYTHON), str(VERIFY), '--protocol', str(PROTOCOL), '--preflight', str(PREFLIGHT), '--out', str(VERIFY_OUT)],
            text=True, capture_output=True, check=False)
        require(proc.returncode == 0, f'tie-free verifier failed: {proc.stdout} {proc.stderr}', failures)
        if VERIFY_OUT.is_file():
            v = json.loads(VERIFY_OUT.read_text())
            require(v.get('status') == 'PASS' and v.get('cuda_used') is False, 'tie-free verifier output not PASS CPU-only', failures)

    manifest = ROOT / 'source_manifest.sha256'
    if manifest.exists():
        lines = [line.split(maxsplit=1) for line in manifest.read_text().splitlines() if line.strip()]
        manifest_paths = set()
        for pieces in lines:
            require(len(pieces) == 2, 'malformed source manifest line', failures)
            if len(pieces) == 2:
                expected, pathtext = pieces
                path = Path(pathtext.strip())
                manifest_paths.add(path)
                require(path.is_file() and sha(path) == expected, f'manifest SHA mismatch: {path}', failures)
        for pinned in (SRC, GUARD, VERIFY, GENERATOR, PROTOCOL, PREFLIGHT):
            require(pinned in manifest_paths, f'source manifest does not pin {pinned.name}', failures)

    payload = {
        'schema': 'safe-c2-v2-tiefree-static-audit-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'cuda_used': False,
        'status': 'PASS' if not failures else 'FAIL',
        'root': str(ROOT),
        'corrected_after_submit': True,
        'v2_tiefree': True,
        'checks': {
            'nodeProcessKnn_launch_count': len(launches),
            'nodeProcessKnn_successor_min_count': kernel.count('node_list[nid + 1].min_dis'),
            'tie_free_counts': expect,
            'guard': 'raw reverify before GPU/lock; bounded strict-idle poll before phase; post telemetry only',
            'baseline_failure_diagnostic': 'counters/first_error JSON emitted before calibration baseline fail',
        },
        'hashes': {str(x): sha(x) for x in required_paths},
        'failures': failures,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f"{payload['status']} {OUT}")
    return 0 if not failures else 3


if __name__ == '__main__':
    sys.exit(main())
