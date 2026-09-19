#!/usr/bin/env python3
"""CPU-only fail-closed audit for the isolated corrected-after-submit Safe-C2."""
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/experiments/tide_safe_c1_20260727/c2_interval_bound_safe')
SRC = ROOT / 'src/gts_safe_c2_interval_sift1m.cu'
SEARCH = ROOT / 'include/search_v2.cuh'
TREE = ROOT / 'include/tree.cuh'
RESIDUAL = ROOT / 'include/residual_pruning.cuh'
CONFIG = ROOT / 'include/config.cuh'
GUARD = ROOT / 'tools/run_safe_c2_guarded.sh'
PROTOCOL = ROOT / 'protocols/safe_c2_interval_sift1m_v1.json'
PREFLIGHT = ROOT / 'preflight/sift1m_full_raw_v1_20260727T164200CST/preflight_safe_c2.json'
OUT = ROOT / 'logs/static_audit_safe_c2_latest.json'


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
    for path in (SRC, SEARCH, TREE, RESIDUAL, CONFIG, GUARD, PROTOCOL, PREFLIGHT):
        require(path.is_file(), f'missing required artifact: {path}', failures)
    if failures:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({'status': 'FAIL', 'failures': failures}, indent=2) + '\n')
        return 2

    src, search, tree, residual, config, guard = (x.read_text() for x in (SRC, SEARCH, TREE, RESIDUAL, CONFIG, GUARD))
    protocol = json.loads(PROTOCOL.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    require(protocol.get('schema') == 'safe-c2-corrected-after-submit-protocol-v1', 'wrong Safe-C2 protocol schema', failures)
    require(preflight.get('schema') == 'safe-c2-corrected-after-submit-preflight-v1', 'wrong Safe-C2 preflight schema', failures)
    require(protocol.get('implementation_identity', {}).get('corrected_after_submit') is True, 'protocol lacks corrected-after-submit identity', failures)
    require(protocol.get('implementation_identity', {}).get('not_submitted_artifact') is True, 'protocol does not disavow submitted-artifact equivalence', failures)
    require(protocol.get('timing_protocol', {}).get('heldout_pairing', {}).get('required') is True, 'protocol lacks mandatory ABBA heldout pairing', failures)

    # Freeze and check the split itself; no GPU API is imported/called here.
    parts = protocol.get('split', {}).get('files', {})
    ids_by_part: dict[str, list[int]] = {}
    for part in ('calibration', 'validation', 'test'):
        entry = parts.get(part, {})
        fp = Path(entry.get('path', ''))
        require(fp.is_file(), f'missing split file {part}', failures)
        if fp.is_file():
            ids = [int(v) for v in fp.read_text().split()]
            ids_by_part[part] = ids
            require(len(ids) == entry.get('count'), f'{part} split count mismatch', failures)
            require(sha(fp) == entry.get('sha256'), f'{part} split SHA mismatch', failures)
            require(len(ids) == len(set(ids)), f'{part} split contains duplicates', failures)
    all_ids = sum((ids_by_part.get(p, []) for p in ('calibration', 'validation', 'test')), [])
    require(len(all_ids) == 10_000 and set(all_ids) == set(range(10_000)), 'splits are not disjoint/exhaustive 0..9999', failures)

    # The frozen data metadata must agree between protocol and preflight. Actual
    # raw-file hashing is deliberately done by the guarded execute path before GPU use.
    mapping = (('base', 'base', 'base'), ('query', 'query', 'query'), ('groundtruth', 'groundtruth', 'groundtruth'))
    for label, protocol_key, preflight_key in mapping:
        p = protocol.get('data', {}).get(protocol_key, {})
        q = preflight.get('inputs', {}).get(preflight_key, {})
        require(p.get('path') == q.get('path'), f'{label} protocol/preflight path mismatch', failures)
        require(p.get('sha256') == q.get('sha256'), f'{label} protocol/preflight SHA mismatch', failures)
        require(isinstance(p.get('sha256'), str) and len(p.get('sha256')) == 64, f'{label} missing frozen SHA', failures)
    require(protocol.get('data', {}).get('preflight_path') == str(PREFLIGHT), 'protocol preflight path not frozen to Safe-C2 preflight', failures)
    require(protocol.get('data', {}).get('preflight_sha256') == sha(PREFLIGHT), 'protocol preflight SHA mismatch', failures)

    # Exact Safe-C2 vector kernel contract.
    start = search.find('__global__ void nodeProcessKnn(')
    end = search.find('// Initialize p_list.', start)
    kernel = search[start:end] if start >= 0 and end > start else ''
    require(start >= 0 and end > start, 'cannot isolate nodeProcessKnn kernel', failures)
    require('const float *max_dis_d' in kernel, 'nodeProcessKnn lacks max_dis_d parameter', failures)
    require(kernel.count('node_list[nid + 1].min_dis') == 0, 'Safe-C2 vector nodeProcessKnn still uses successor-min upper bound', failures)
    require(kernel.count('const float lower_from_max = dis_q - max_dis_d[nid];') == 2, 'both nodeProcessKnn compile branches must use max_dis_d', failures)
    require(kernel.count('fmaxf(0.0f, fmaxf(lower_from_min, lower_from_max))') == 2, 'both nodeProcessKnn branches must form Eq.1 LB', failures)
    launches = re.findall(r'nodeProcessKnn<<<[^>]+>>>\(([^\n]+)', search)
    require(len(launches) == 4, f'expected 4 nodeProcessKnn launch sites, found {len(launches)}', failures)
    require(all(re.match(r'\s*node_list\s*,\s*max_dis_d\s*,\s*disk\s*,', x) for x in launches), 'one or more nodeProcessKnn launch sites omit/max-order max_dis_d', failures)
    legacy_rnn_successor_count = search.count('node_list[nid + 1].min_dis')
    require(legacy_rnn_successor_count == 1, f'unexpected successor-min count outside Safe-C2 vector kernel: {legacy_rnn_successor_count}', failures)
    require('nodeProcessRnn<<<' not in src, 'Safe-C2 runner source launches out-of-scope RNN path', failures)
    require('max_dis_d[id_child] = max_result;' in tree and 'int last_idx = node_child.lid + node_child.size - 1;' in tree, 'tree source lacks stored max_dis_d construction', failures)

    # Gamma must be exactly Eq.2 scale-only; copied marketing/adaptation text is forbidden.
    require('(scale - 1.0f) * dis_lb' in residual, 'rp_predict is not Eq.2 scale delta', failures)
    require('return fmaxf(0.0f, (scale - 1.0f) * dis_lb);' in residual, 'rp_predict does not clamp scale delta', failures)
    for forbidden in ('zero false negatives', 'Online Adaptation', 'gamma ~ 0.5', 'Piecewise-linear LUT for enhanced inference'):
        require(forbidden not in residual, f'residual header contains forbidden legacy claim: {forbidden}', failures)

    # Corrected interval certificate and strict held-out gate must be present in the runner.
    for required in (
        'certify_and_outward_repair_intervals', 'current padded leaf memberships',
        'post-repair directional interval cover verification failed', 'before_interval_hash',
        'after_interval_hash', 'post_reverify_passed', 'per_query_no_regression',
        'same_per_query_overlap', 'run_paired_abba_stage', 'timing_comparison', 'safe-c2-corrected-after-submit-run-v1',
        'SafeC2EffectiveCoordinate', 'static_assert(std::is_same<SafeC2EffectiveCoordinate, float>::value',
    ):
        require(required in src, f'Safe-C2 runner lacks required contract: {required}', failures)
    for forbidden in ('#include "update.cuh"', '#include "incremental_insert.cuh"', 'submitted-vldb-c2-perlevel-run-v1'):
        require(forbidden not in src, f'runner contains forbidden legacy/non-Safe identity: {forbidden}', failures)
    require('#define short float' in config, 'config does not freeze legacy effective float32 data type', failures)
    require('searchIndexKnnV2(runtime.data_d' in src and 'query_d, result_ids_d' in src, 'runner does not call vector top-k ID API', failures)

    # Guard hardening: static, no GPU query. It must pin raw inputs before any
    # nvidia-smi/lock, parse only an exact GPU0 row, reject ambiguous app output,
    # use a verified setsid child group, and fail on output collisions.
    for required in (
        'verify_frozen_data_hashes()', 'data_hash_verification=PASS',
        'resolve_physical_gpu0_uuid()', 'require_exactly_one_nonblank_line',
        'reject_nvidia_error_text', 'No devices were found',
        '--query-compute-apps=pid,process_name,used_memory',
        'No running processes found', 'No running compute processes found',
        'gpu_status_exactly_one_row=PASS',
        'setsid --wait /bin/bash -c', 'child_session_verified_before_cuda_exec=PASS',
        'request_child_abort', 'terminate_child_group', 'wait_for_launcher_exit',
        'TERM_then_reap_wait_before_lock_release',
        'NVIDIA_VISIBLE_DEVICES="$GPU0_UUID"',
        '[[ ! -e "$OUT" ]] || die', '[[ ! -e "$PRE" ]] || die',
    ):
        require(required in guard, f'guard lacks required hardening: {required}', failures)
    require('query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1 || true' not in guard, 'guard masks compute-app query failure with || true', failures)
    require('apps_clean="$(normalize_nonblank_lines "$apps")"' in guard, 'guard does not normalize app output before exact allowlist', failures)
    app_case = "case \"$apps_clean\" in\n    '') ;;\n    'No running processes found'|'No running compute processes found') ;;"
    require(app_case in guard, 'guard app query allows anything other than explicit no-process response', failures)
    execute_anchor = guard.find('static_check "$PRE/static_audit.log"')
    hash_anchor = guard.find('verify_frozen_data_hashes "$PRE/data_hash_verification.env"')
    gpu_anchor = guard.find('\nresolve_physical_gpu0_uuid\n', hash_anchor)
    lock_anchor = guard.find('if ! mkdir "$LOCK"')
    require(execute_anchor >= 0 and execute_anchor < hash_anchor < gpu_anchor < lock_anchor, 'guard does not hash inputs before GPU query/lock', failures)
    launch = guard[guard.find('launch_session_guarded_command()'):guard.find('run_phase()')]
    require(launch.find('if adopt_child_session_from_ready; then') >= 0 and launch.find('if adopt_child_session_from_ready; then') < launch.find(': > "$CHILD_GO_FILE"'), 'guard authorizes CUDA child before verified session handshake', failures)
    cleanup = guard[guard.find('cleanup_on_exit()'):guard.find('on_signal()')]
    require(cleanup.find('terminate_child_group || true') >= 0 and cleanup.find('terminate_child_group || true') < cleanup.find('release_lock || true'), 'guard can release lock before child TERM/reap/wait cleanup', failures)
    require('export NVIDIA_VISIBLE_DEVICES="$GPU0_UUID"' in guard, 'guard does not export NVIDIA_VISIBLE_DEVICES UUID binding', failures)

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
        require(GUARD in manifest_paths, 'source manifest does not pin guard implementation', failures)

    payload = {
        'schema': 'safe-c2-static-audit-v2',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'cuda_used': False,
        'status': 'PASS' if not failures else 'FAIL',
        'root': str(ROOT),
        'corrected_after_submit': True,
        'checks': {
            'nodeProcessKnn_launch_count': len(launches),
            'nodeProcessKnn_successor_min_count': kernel.count('node_list[nid + 1].min_dis'),
            'out_of_scope_legacy_rnn_successor_min_count': legacy_rnn_successor_count,
            'effective_coordinate_type': 'float32 via config.cuh short->float macro',
            'interval_certificate': 'leaf-to-ancestor membership reconstruction + outward repair + D2H reverify required',
            'heldout_gate': 'strict per-query overlap no-regression',
            'heldout_timing': 'ABBA interleaved paired baseline/Safe-C2 runs; calibration correctness-only',
            'guard': 'strict GPU0 parser, frozen input hashes before GPU/lock, verified setsid session cleanup, output collision refusal',
        },
        'hashes': {str(x): sha(x) for x in (SRC, SEARCH, TREE, RESIDUAL, CONFIG, GUARD, PROTOCOL, PREFLIGHT)},
        'failures': failures,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f"{payload['status']} {OUT}")
    return 0 if not failures else 3


if __name__ == '__main__':
    sys.exit(main())
