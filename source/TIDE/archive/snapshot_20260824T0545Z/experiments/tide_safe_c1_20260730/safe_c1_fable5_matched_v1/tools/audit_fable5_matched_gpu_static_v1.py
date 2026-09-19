#!/usr/bin/env python3
"""Static-only audit for the isolated Fable5 matched GPU runner and guard."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CANONICAL_ROOT = Path('/workspace/experiments/tide_safe_c1_20260730/safe_c1_fable5_matched_v1')
EXPECTED_ARCHIVED_CORE_SHA256 = '5d0bfbd56c974855881bd2639f1bc00b50e2157615e23fe9673f53c10078c6d5'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_text(checks: dict[str, bool], errors: list[str], text: str, name: str, token: str) -> None:
    ok = token in text
    checks[name] = ok
    if not ok:
        errors.append('missing:' + name)


def forbid_text(checks: dict[str, bool], errors: list[str], text: str, name: str, token: str) -> None:
    ok = token not in text
    checks[name] = ok
    if not ok:
        errors.append('forbidden:' + name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    report: dict[str, Any] = {
        'schema': 'fable5-matched-gpu-static-audit-v1',
        'root': str(root),
        'gpu_used': False,
        'cuda_binary_executed': False,
        'nvidia_smi_called': False,
        'scope': 'Static source/guard contract only; no compiler, CUDA binary, GPU, NVML, or timing invocation.',
        'checks': {},
        'errors': [],
    }
    checks: dict[str, bool] = report['checks']
    errors: list[str] = report['errors']
    try:
        checks['root_is_isolated_fable5_target'] = root == CANONICAL_ROOT
        if not checks['root_is_isolated_fable5_target']:
            errors.append('wrong_root')
        runner = root / 'src/fable5_matched_gpu_runner_v1.cu'
        archived = root / 'src/safe_c1_dynamic_gts.cu'
        guard = root / 'tools/run_fable5_matched_gpu1_guarded_v1.sh'
        validator = root / 'tools/validate_fable5_matched_gpu_receipts_v1.py'
        trace_preflight = root / 'tools/preflight_fable5_matched_trace_v1.py'
        trace_preflight_contract = root / 'manifests/fable5_matched_trace_preflight_v1.json'
        manifest = root / 'manifests/fable5_matched_gpu_runner_v1.json'
        for name, path in {'runner': runner, 'archived_core': archived, 'guard': guard, 'validator': validator, 'trace_preflight': trace_preflight, 'trace_preflight_contract': trace_preflight_contract, 'manifest': manifest}.items():
            checks['exists_' + name] = path.is_file()
            if not path.is_file():
                errors.append('missing_file:' + name)
        if runner.is_file():
            text = runner.read_text(encoding='utf-8')
            for name, token in {
                'isolated_legacy_entry_renamed': '#define main fable5_archived_g2_entry_not_invoked',
                'archived_core_reused_readonly': '#include "safe_c1_dynamic_gts.cu"',
                'two_policy_enum': 'enum class Policy { kSafeC1, kBufferOnly }',
                'policy_cli': '--policy',
                'trace_cli': '--trace',
                'trace_hash_cli': '--trace-sha256',
                'strict_native_certificate': 'frozen.snapshot.certify_leaf(pool, id)',
                'certificate_both_policies_comment': 'for BOTH policies',
                'native_traversal_receipt': 'run_gts_base_topk_with_receipt',
                'receipt_selected_sidecars': 'state.sidecar_candidates_for(base.visited_leaf_ids)',
                'global_exact_delta': 'state.delta_ids().begin()',
                'full_active_oracle': 'exact_active_oracle(pool, query, state.active()',
                'immediate_rebuild_barrier': 'base deletion requires immediate REBUILD before ',
                'metadata_only_rebuild': 'release_tree_metadata_only(runtime)',
                'reseed_active_ids': 'build_seeded_epoch(runtime, live)',
                'post_rebuild_state_reset': 'state.after_rebuild(live)',
                'no_timing_scope': 'no timing, throughput, latency, C2, C3, or deployment claim',
            }.items():
                require_text(checks, errors, text, name, token)
            for name, token in {
                'legacy_incremental_include': 'incremental_insert.cuh',
                'legacy_update_include': '#include "update.cuh"',
                'legacy_direct_call': 'incrementalInsert(',
                'legacy_rnn_update_call': 'updateIndexRnn(',
            }.items():
                forbid_text(checks, errors, text, name, token)
        if archived.is_file():
            report['archived_core_sha256'] = sha(archived)
            checks['archived_core_unchanged_from_pinned_v5'] = report['archived_core_sha256'] == EXPECTED_ARCHIVED_CORE_SHA256
            if not checks['archived_core_unchanged_from_pinned_v5']:
                errors.append('archived_core_sha_mismatch')
        if guard.is_file():
            gtext = guard.read_text(encoding='utf-8')
            for name, token in {
                'default_execute_gate': '[[ "$EXECUTE" == 1 ]] ||',
                'approval_id_required': '--approval-id',
                'approval_epoch_required': '--approval-issued-epoch',
                'ttl_required': '--ttl',
                'gpu_ordinal_required': '--gpu-ordinal',
                'gpu_uuid_required': '--gpu-uuid',
                'gpu1_only': "readonly REQUIRED_GPU_ORDINAL='1'",
                'gpu1_uuid_pinned': 'GPU-CONFIGURE-ARCHIVE-DEVICE',
                'trace_sha_pinned': '--trace-sha256',
                'candidate_contract_pinned': '--candidate-contract-sha256',
                'source_sha_pinned': '--source-sha256',
                'binary_sha_pinned': '--binary-sha256',
                'nvml_binding': 'import pynvml',
                'no_process_termination_contract': 'never terminates processes',
                'cuda_device_restriction': 'CUDA_VISIBLE_DEVICES="$GPU_ORDINAL"',
                'two_policy_invocations': '--policy safe_c1',
                'buffer_policy_invocation': '--policy buffer_only',
                'receipt_validator_invoked': '"$VALIDATOR" --safe-results',
                'trace_preflight_invoked': '"$PYTHON_BIN" "$TRACE_PREFLIGHT"',
                'trace_preflight_contract_passed': '--contract "$TRACE_PREFLIGHT_CONTRACT"',
                'trace_preflight_persisted_after_mkdir': '"$RUN_DIR/trace_preflight.json"',
                'static_audit_before_nvml': '"$STATIC_AUDIT" --root "$ROOT"',
            }.items():
                require_text(checks, errors, gtext, name, token)
            prohibited = 'nvidia' + '-smi'
            forbid_text(checks, errors, gtext, 'legacy_gpu_tool', prohibited)
            execute_pos = gtext.find('[[ "$EXECUTE" == 1 ]] ||')
            preflight_pos = gtext.find('"$PYTHON_BIN" "$TRACE_PREFLIGHT"')
            run_dir_pos = gtext.find('RUN_DIR="$ROOT/runs/$RUN_NAME"')
            mkdir_pos = gtext.find('mkdir -p "$RUN_DIR/safe_c1"')
            audit_pos = gtext.find('"$STATIC_AUDIT" --root "$ROOT"')
            nvml_pos = gtext.find('import pynvml')
            checks['execute_gate_before_trace_preflight'] = execute_pos >= 0 and preflight_pos >= 0 and execute_pos < preflight_pos
            checks['trace_preflight_before_run_directory_mutation'] = preflight_pos >= 0 and run_dir_pos >= 0 and mkdir_pos >= 0 and preflight_pos < run_dir_pos < mkdir_pos
            checks['trace_preflight_before_static_audit'] = preflight_pos >= 0 and audit_pos >= 0 and preflight_pos < audit_pos
            checks['static_audit_before_nvml'] = audit_pos >= 0 and nvml_pos >= 0 and audit_pos < nvml_pos
            if not checks['execute_gate_before_trace_preflight']:
                errors.append('guard_execute_gate_preflight_order')
            if not checks['trace_preflight_before_run_directory_mutation']:
                errors.append('guard_trace_preflight_directory_order')
            if not checks['trace_preflight_before_static_audit']:
                errors.append('guard_trace_preflight_static_audit_order')
            if not checks['static_audit_before_nvml']:
                errors.append('guard_static_audit_order')
        if validator.is_file():
            vtext = validator.read_text(encoding='utf-8')
            for name, token in {
                'validator_checks_shared_trace': 'trace_sha256',
                'validator_checks_fable5_12op_set': 'fable5_12op_set_mismatch',
                'validator_checks_certificate_receipts': 'native_certificate_receipt_mismatch',
                'validator_checks_native_receipts': 'native_traversal_receipt_mismatch',
                'validator_checks_oracle_results': 'oracle_result_mismatch',
                'validator_checks_capacity_same_leaf_proof': 'safe_missing_capacity_same_leaf_proof',
                'validator_checks_direct_visibility': 'safe_direct_never_visible_in_exact_result',
                'validator_checks_buffer_policy': 'buffer_not_forced_delta',
            }.items():
                require_text(checks, errors, vtext, name, token)
            prohibited = 'nvidia' + '-smi'
            forbid_text(checks, errors, vtext, 'validator_legacy_gpu_tool', prohibited)
        if trace_preflight.is_file():
            ptext = trace_preflight.read_text(encoding='utf-8')
            for name, token in {
                'preflight_e1gtrc02_abi': 'E1GTRC02',
                'preflight_header_abi': '<8sI6IfQ',
                'preflight_event_abi': '<IB3si',
                'preflight_exact_12op_sequence': 'EXPECTED_SEQUENCE',
                'preflight_candidate_contract_hash': 'candidate_contract_sha256',
                'preflight_candidate_archived_core_hash': 'candidate_archived_core_sha256_mismatch',
                'preflight_static_certificate_limit': 'NOT_ESTABLISHED_STATICALLY',
                'preflight_runtime_requirements_only': 'runtime_requirements_only',
            }.items():
                require_text(checks, errors, ptext, name, token)
            for name, token in {
                'preflight_nvml_binding': 'pynvml',
                'preflight_subprocess': 'subprocess',
                'preflight_cuda_visibility': 'CUDA_VISIBLE_DEVICES',
                'preflight_legacy_gpu_tool': 'nvidia' + '-smi',
            }.items():
                forbid_text(checks, errors, ptext, name, token)
        if trace_preflight_contract.is_file():
            c = json.loads(trace_preflight_contract.read_text(encoding='utf-8'))
            checks['trace_preflight_contract_schema'] = c.get('schema') == 'fable5-matched-trace-preflight-contract-v1'
            checks['trace_preflight_contract_32d'] = c.get('trace_abi', {}).get('header', {}).get('dimension') == 32
            checks['trace_preflight_contract_12ops'] = len(c.get('event_sequence', [])) == 12
            checks['trace_preflight_contract_static_cert_limit'] = 'native_certificate_outcomes_not_established_statically' in c.get('static_limitations', [])
            for name in ('trace_preflight_contract_schema', 'trace_preflight_contract_32d', 'trace_preflight_contract_12ops', 'trace_preflight_contract_static_cert_limit'):
                if not checks[name]:
                    errors.append('invalid:' + name)
        if manifest.is_file():
            m = json.loads(manifest.read_text(encoding='utf-8'))
            checks['manifest_schema'] = m.get('schema') == 'fable5-matched-gpu-runner-v1'
            checks['manifest_static_only'] = m.get('execution_boundary', {}).get('compiled') is False and m.get('execution_boundary', {}).get('gpu_used') is False
            if not checks['manifest_schema']:
                errors.append('manifest_schema')
            if not checks['manifest_static_only']:
                errors.append('manifest_not_static_only')
        report['runner_sha256'] = sha(runner) if runner.is_file() else None
        report['guard_sha256'] = sha(guard) if guard.is_file() else None
        report['validator_sha256'] = sha(validator) if validator.is_file() else None
        report['trace_preflight_sha256'] = sha(trace_preflight) if trace_preflight.is_file() else None
        report['trace_preflight_contract_sha256'] = sha(trace_preflight_contract) if trace_preflight_contract.is_file() else None
        report['manifest_sha256'] = sha(manifest) if manifest.is_file() else None
        report['status'] = 'PASS_FABLE5_MATCHED_GPU_STATIC_AUDIT' if not errors else 'FAIL_FABLE5_MATCHED_GPU_STATIC_AUDIT'
    except Exception as exc:
        errors.append(f'exception:{type(exc).__name__}:{exc}')
        report['status'] = 'FAIL_FABLE5_MATCHED_GPU_STATIC_AUDIT'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(report['status'])
    return 0 if not errors else 2


if __name__ == '__main__':
    raise SystemExit(main())
