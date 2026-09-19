#!/usr/bin/env python3
"""Static-only audit for the isolated Fable5 native E0 snapshot pipeline."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

CANONICAL_ROOT = Path('/workspace/experiments/tide_safe_c1_20260730/safe_c1_fable5_matched_v1')
PINNED_CORE_SHA256 = '5d0bfbd56c974855881bd2639f1bc00b50e2157615e23fe9673f53c10078c6d5'
PINNED_MATCHED_RUNNER_SHA256 = 'cb298cf23c3a36176c873ce22c2fadc94b61a75e38e67718a4e6f903ec4ff240'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(checks: dict[str, bool], errors: list[str], text: str, name: str, token: str) -> None:
    ok = token in text
    checks[name] = ok
    if not ok:
        errors.append('missing:' + name)


def forbid(checks: dict[str, bool], errors: list[str], text: str, name: str, token: str) -> None:
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
        'schema': 'fable5-native-snapshot-export-static-audit-v1',
        'root': str(root),
        'execution': {
            'cpu_static_only': True,
            'compiled': False,
            'cuda_binary_executed': False,
            'gpu_used': False,
            'nvml_used': False,
            'nvidia_smi_called': False,
        },
        'scope': 'Static source/selector/guard contract only; no compiler, CUDA binary, GPU, NVML, or timing invocation.',
        'checks': {},
        'errors': [],
    }
    checks: dict[str, bool] = report['checks']
    errors: list[str] = report['errors']
    try:
        checks['isolated_target_root'] = root == CANONICAL_ROOT
        if not checks['isolated_target_root']:
            errors.append('wrong_root')
        core = root / 'src/safe_c1_dynamic_gts.cu'
        runner = root / 'src/fable5_matched_gpu_runner_v1.cu'
        backup = root / 'backups/fable5_matched_gpu_runner_v1.cu.pre_snapshot_export_20260730'
        exporter = root / 'src/fable5_native_snapshot_exporter_v1.cu'
        selector = root / 'tools/select_fable5_matched_trace_from_native_snapshot_v1.py'
        guard = root / 'tools/run_fable5_native_snapshot_export_guarded_v1.sh'
        manifest = root / 'manifests/fable5_native_snapshot_exporter_v1.json'
        for name, path in {
            'core': core, 'matched_runner': runner, 'matched_runner_backup': backup,
            'exporter': exporter, 'selector': selector, 'guard': guard, 'manifest': manifest,
        }.items():
            checks['exists_' + name] = path.is_file()
            if not path.is_file():
                errors.append('missing_file:' + name)
        if core.is_file():
            report['core_sha256'] = sha(core)
            checks['pinned_core_unchanged'] = report['core_sha256'] == PINNED_CORE_SHA256
            if not checks['pinned_core_unchanged']:
                errors.append('core_sha_mismatch')
        if runner.is_file():
            report['matched_runner_sha256'] = sha(runner)
            checks['matched_runner_unchanged'] = report['matched_runner_sha256'] == PINNED_MATCHED_RUNNER_SHA256
            if not checks['matched_runner_unchanged']:
                errors.append('matched_runner_sha_mismatch')
        if backup.is_file() and runner.is_file():
            checks['backup_matches_unchanged_runner'] = sha(backup) == sha(runner)
            if not checks['backup_matches_unchanged_runner']:
                errors.append('runner_backup_mismatch')
        if exporter.is_file():
            text = exporter.read_text(encoding='utf-8')
            for name, token in {
                'standalone_main_renamed': '#define main fable5_archived_g2_entry_not_invoked',
                'pinned_core_reused': '#include "safe_c1_dynamic_gts.cu"',
                'explicit_header_trace_cli': '--header-trace',
                'explicit_output_cli': '--out',
                'all_immutable_hash_pins': '--stable-map-sha256',
                'exporter_hash_pin': '--exporter-source-sha256',
                'core_hash_pin': '--core-source-sha256',
                'matched_runner_hash_pin': '--matched-runner-source-sha256',
                'e0_only_comment': 'no Safe-C1/buffer-only update',
                'native_tree_capture': 'snapshot.capture(runtime)',
                'logical_payload_capture': 'capture_leaf_payload',
                'tn_min_bits_export': 'min_dis_f32_bits',
                'max_distance_bits_export': 'max_distance_f32_bits',
                'canonical_live_hash_export': 'canonical_live_tree_fnv1a64',
                'empty_nodes_null_abi': 'empty-nodes=null',
                'immutable_pool_hash_export': 'immutable_pool_fnv1a64',
                'leaf_set_exact_gate': 'validate_exact_e0_leaf_set',
                'mode_zero_upload': 'upload_rp_constants(alpha, beta, gamma, RP_MAX_LEVELS, nullptr, nullptr, nullptr, 0, 0)',
                'mode_zero_provenance': r'\"residual_pruning\": {\"mode\":0',
                'no_query_export_claim': r'\"query_executed\":false',
                'no_c2_machine_claim': r'\"c2_used\":false',
                'no_c3_machine_claim': r'\"c3_used\":false',
                'no_c2c3_scope': 'no policy replay, update trace, dynamic query, timing, C2, C3, or deployment claim',
                'metadata_cleanup': 'release_tree_full(runtime)',
            }.items():
                require(checks, errors, text, name, token)
            for name, token in {
                'matched_policy_cli': '--policy',
                'matched_update_state': 'MatchedState',
                'matched_policy_enum': 'enum class Policy',
                'legacy_incremental_include': 'incremental_insert.cuh',
                'legacy_update_include': '#include "update.cuh"',
            }.items():
                forbid(checks, errors, text, name, token)
            prohibited = 'nvidia' + '-smi'
            forbid(checks, errors, text, 'exporter_legacy_gpu_tool', prohibited)
        if selector.is_file():
            text = selector.read_text(encoding='utf-8')
            try:
                ast.parse(text, filename=str(selector))
                checks['selector_python_ast_parses'] = True
            except SyntaxError as exc:
                checks['selector_python_ast_parses'] = False
                errors.append('selector_syntax:' + str(exc))
            for name, token in {
                'native_snapshot_required': '--native-snapshot',
                'snapshot_hash_recompute': 'canonical_live_tree_hash_mismatch',
                'snapshot_immutable_pool_hash': 'immutable_pool_fnv1a64_mismatch',
                'snapshot_radius_binding': 'snapshot_header_mismatch:radius',
                'snapshot_empty_node_null_gate': 'empty_node_must_be_null',
                'snapshot_c2_false_gate': 'snapshot_export_must_not_use_c2',
                'snapshot_c3_false_gate': 'snapshot_export_must_not_use_c3',
                'strict_lower_mirror': 'distance > f32(node["min"] + STRICT_EPSILON)',
                'strict_upper_mirror': 'distance < f32(upper - STRICT_EPSILON)',
                'full_sibling_mirror': 'for slot in range(TREE_ORDER)',
                'same_leaf_three_gate': 'no_same_native_cert_leaf_group_of_three',
                'natural_reject_gate': 'no_natural_certificate_reject_in_immutable_reservoir',
                'capacity_one_gate': 'selector_requires_leaf_capacity_one_for_capacity_coverage',
                'direct_observability_A': 'A_exact_topk_before_rebuild',
                'direct_observability_C': 'C_exact_topk_after_reuse',
                'fallback_observability_mid': 'B_and_R_exact_topk_middle',
                'candidate_contract_write': 'fable5_matched_trace_candidate_v1.json',
                'candidate_static_limitation': 'CPU_CANDIDATE_NOT_NATIVE_RUNTIME_VALIDATED',
                'runtime_receipt_requirement': 'receipt visits direct_leaf',
                'e1_active_formula': 'E1_active_formula',
                'twelve_op_encode': 'len(events),',
                'no_trace_on_block': 'BLOCKED_NO_TRACE_WRITTEN',
                'mode_zero_requirement': 'snapshot_not_baseline_residual_mode_0',
                'non_c2c3_scope': 'no runtime receipt, timing, performance, C2, C3, or deployment claim',
            }.items():
                require(checks, errors, text, name, token)
            for name, token in {
                'selector_subprocess': 'subprocess',
                'selector_nvml': 'pynvml',
                'selector_cuda_visibility': 'CUDA_VISIBLE_DEVICES',
                'selector_legacy_gpu_tool': 'nvidia' + '-smi',
            }.items():
                forbid(checks, errors, text, name, token)
        if guard.is_file():
            text = guard.read_text(encoding='utf-8')
            for name, token in {
                'default_execute_gate': '[[ "$EXECUTE" == 1 ]] ||',
                'approval_id': '--approval-id',
                'approval_epoch': '--approval-issued-epoch',
                'ttl': '--ttl',
                'gpu_ordinal': '--gpu-ordinal',
                'gpu_uuid': '--gpu-uuid',
                'gpu1_only': "readonly REQUIRED_GPU_ORDINAL='1'",
                'gpu1_uuid': 'GPU-CONFIGURE-ARCHIVE-DEVICE',
                'no_process_termination': 'never terminates a process',
                'static_audit_invoked': 'audit_fable5_native_snapshot_export_static_v1.py',
                'exporter_binary_invoked': '--header-trace "$HEADER_TRACE"',
                'source_pins_passed': '--exporter-source-sha256',
                'mode_zero_manifest_pin': 'snapshot exporter manifest',
                'nvml_execute_only': 'import pynvml',
            }.items():
                require(checks, errors, text, name, token)
            prohibited = 'nvidia' + '-smi'
            forbid(checks, errors, text, 'guard_legacy_gpu_tool', prohibited)
            gate = text.find('[[ "$EXECUTE" == 1 ]] ||')
            audit = text.find('"$PYTHON_BIN" "$STATIC_AUDIT"')
            nvml = text.find('import pynvml')
            checks['guard_gate_before_static_audit'] = gate >= 0 and audit >= 0 and gate < audit
            checks['guard_static_audit_before_nvml'] = audit >= 0 and nvml >= 0 and audit < nvml
            if not checks['guard_gate_before_static_audit']:
                errors.append('guard_gate_order')
            if not checks['guard_static_audit_before_nvml']:
                errors.append('guard_static_audit_nvml_order')
        if manifest.is_file():
            m = json.loads(manifest.read_text(encoding='utf-8'))
            checks['manifest_schema'] = m.get('schema') == 'fable5-native-snapshot-exporter-v1'
            boundary = m.get('execution_boundary', {})
            checks['manifest_pre_gpu_boundary'] = (
                boundary.get('cuda_binary_executed') is False
                and boundary.get('gpu_used') is False
                and boundary.get('nvml_queried') is False
            )
            status = m.get('status')
            checks['manifest_cpu_build_state'] = (
                (status == 'STATIC_ONLY_PRE_COMPILE_PRE_GPU' and boundary.get('compiled') is False)
                or (status == 'CPU_BUILT_PRE_GPU' and boundary.get('compiled') is True)
            )
            checks['manifest_mode_zero'] = m.get('residual_pruning', {}).get('mode') == 0
            checks['manifest_machine_no_c2c3'] = (
                m.get('residual_pruning', {}).get('c2_used') is False
                and m.get('residual_pruning', {}).get('c3_used') is False
            )
            checks['manifest_no_c2c3'] = all(item in m.get('non_claims', []) for item in ('C2', 'C3'))
            checks['manifest_exporter_source_hash'] = (
                exporter.is_file() and m.get('exporter', {}).get('source_sha256') == sha(exporter)
            )
            checks['manifest_selector_source_hash'] = (
                selector.is_file() and m.get('exporter', {}).get('cpu_selector_source_sha256') == sha(selector)
            )
            for name in ('manifest_schema', 'manifest_pre_gpu_boundary', 'manifest_cpu_build_state', 'manifest_mode_zero', 'manifest_machine_no_c2c3', 'manifest_no_c2c3', 'manifest_exporter_source_hash', 'manifest_selector_source_hash'):
                if not checks[name]:
                    errors.append('invalid:' + name)
        for name, path in {'exporter': exporter, 'selector': selector, 'guard': guard, 'manifest': manifest}.items():
            report[name + '_sha256'] = sha(path) if path.is_file() else None
        report['status'] = 'PASS_FABLE5_NATIVE_SNAPSHOT_STATIC_AUDIT' if not errors else 'FAIL_FABLE5_NATIVE_SNAPSHOT_STATIC_AUDIT'
    except Exception as exc:  # noqa: BLE001
        errors.append(f'exception:{type(exc).__name__}:{exc}')
        report['status'] = 'FAIL_FABLE5_NATIVE_SNAPSHOT_STATIC_AUDIT'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(report['status'])
    return 0 if not errors else 2


if __name__ == '__main__':
    raise SystemExit(main())
