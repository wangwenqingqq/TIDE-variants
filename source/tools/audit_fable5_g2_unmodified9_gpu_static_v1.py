#!/usr/bin/env python3
"""Static-only audit for the unmodified G2 9-op Safe-C1 GPU1 guard.

This audit intentionally does not compile, execute CUDA, inspect a GPU, import
NVML, or invoke a process-control tool. It validates only the pinned artifact
chain and the fact that this path is correctness-only rather than performance.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path('/workspace/experiments/tide_safe_c1_20260730/safe_c1_fable5_matched_v1')
CORE_SHA = '5d0bfbd56c974855881bd2639f1bc00b50e2157615e23fe9673f53c10078c6d5'
RUNNER_SHA = 'cb298cf23c3a36176c873ce22c2fadc94b61a75e38e67718a4e6f903ec4ff240'
VARIANT = 'unmodified_g2_l2_correctness_witness_v1'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check(checks: dict[str, bool], errors: list[str], name: str, condition: bool) -> None:
    checks[name] = condition
    if not condition:
        errors.append('failed:' + name)


def need(checks: dict[str, bool], errors: list[str], text: str, name: str, token: str) -> None:
    check(checks, errors, name, token in text)


def forbid(checks: dict[str, bool], errors: list[str], text: str, name: str, token: str) -> None:
    check(checks, errors, name, token not in text)


def read_obj(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('json_object_required')
    return value


def under(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return None
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    report: dict[str, Any] = {
        'schema': 'fable5-g2-unmodified9-gpu-static-audit-v1',
        'root': str(root),
        'execution_boundary': {
            'cpu_static_only': True, 'compiler_invoked': False,
            'cuda_binary_executed': False, 'gpu_used': False,
            'nvml_used': False, 'processes_inspected': False,
        },
        'scope': 'Static audit of unmodified selected G2 L2 correctness only; no workload/timing/performance/C2/C3/deployment result.',
        'checks': {}, 'errors': [],
    }
    checks: dict[str, bool] = report['checks']
    errors: list[str] = report['errors']
    try:
        check(checks, errors, 'canonical_root', root == ROOT)
        paths = {
            'core': root / 'src/safe_c1_dynamic_gts.cu',
            'runner': root / 'src/fable5_matched_gpu_runner_v1.cu',
            'binary': root / 'build/GTS_fable5_matched_v1',
            'build_receipt': root / 'provenance/fable5_matched_gpu_runner_v1_cpu_build_20260730.json',
            'guard': root / 'tools/run_fable5_g2_unmodified9_gpu1_guarded_v1.sh',
            'preflight': root / 'tools/preflight_fable5_g2_unmodified9_trace_v1.py',
            'validator': root / 'tools/validate_fable5_g2_unmodified9_gpu_receipts_v1.py',
            'contract': root / 'manifests/fable5_g2_unmodified9_trace_preflight_v1.json',
            'manifest': root / 'manifests/fable5_g2_unmodified9_gpu_runner_v1.json',
            'audit': root / 'tools/audit_fable5_g2_unmodified9_gpu_static_v1.py',
            'bundle_trace': root / 'bundles/g2_rebuild_witness_v5_20260728T125440Z_cpu/trace.g2trc',
        }
        for name, path in paths.items():
            check(checks, errors, 'exists_' + name, path.is_file())
        if paths['core'].is_file():
            report['core_sha256'] = sha(paths['core'])
            check(checks, errors, 'core_pin', report['core_sha256'] == CORE_SHA)
        if paths['runner'].is_file():
            report['runner_sha256'] = sha(paths['runner'])
            check(checks, errors, 'runner_pin', report['runner_sha256'] == RUNNER_SHA)
            runner = paths['runner'].read_text(encoding='utf-8')
            for name, token in {
                'native_certificate': 'frozen.snapshot.certify_leaf(pool, id)',
                'native_receipt': 'run_gts_base_topk_with_receipt',
                'receipt_sidecar': 'state.sidecar_candidates_for(base.visited_leaf_ids)',
                'exact_delta': 'state.delta_ids().begin()',
                'full_active_oracle': 'exact_active_oracle(pool, query, state.active()',
                'rebuild_barrier': 'base deletion requires immediate REBUILD before ',
                'no_timing_scope': 'no timing, throughput, latency, C2, C3, or deployment claim',
            }.items():
                need(checks, errors, runner, 'runner_' + name, token)
            forbid(checks, errors, runner, 'runner_legacy_incremental', 'incremental_insert.cuh')
            forbid(checks, errors, runner, 'runner_legacy_update', '#include "update.cuh"')
        if paths['build_receipt'].is_file():
            receipt = read_obj(paths['build_receipt'])
            check(checks, errors, 'cpu_build_receipt',
                  receipt.get('schema') == 'fable5-matched-gpu-runner-v1-cpu-build-receipt'
                  and receipt.get('status') == 'PASS_CPU_BUILD_ONLY_NOT_EXECUTED'
                  and receipt.get('source_sha256') == RUNNER_SHA
                  and receipt.get('archived_core_sha256') == CORE_SHA
                  and receipt.get('binary_sha256') == (sha(paths['binary']) if paths['binary'].is_file() else None)
                  and receipt.get('cuda_binary_executed') is False
                  and receipt.get('gpu_used') is False and receipt.get('nvml_used') is False)
        if paths['contract'].is_file():
            contract = read_obj(paths['contract'])
            variant = contract.get('variant', {})
            abi = contract.get('trace_abi', {})
            check(checks, errors, 'contract_boundary',
                  contract.get('schema') == 'fable5-g2-unmodified9-trace-preflight-contract-v1'
                  and contract.get('status') == 'STATIC_PRE_NVML_CONTRACT_ONLY'
                  and variant.get('id') == VARIANT and variant.get('input_modification', '').startswith('none')
                  and 'performance' in variant.get('prohibited_claims', [])
                  and abi.get('trace_filename') == 'trace.g2trc'
                  and abi.get('trace_sha256') == (sha(paths['bundle_trace']) if paths['bundle_trace'].is_file() else None)
                  and abi.get('header', {}).get('event_count') == 9
                  and len(contract.get('event_sequence', [])) == 9)
            snap = under(root, contract.get('native_e0_snapshot', {}).get('relative_path'))
            check(checks, errors, 'contract_snapshot_pin',
                  snap is not None and snap.is_file()
                  and contract.get('native_e0_snapshot', {}).get('sha256') == sha(snap))
        for name in ('preflight', 'validator'):
            path = paths[name]
            if path.is_file():
                text = path.read_text(encoding='utf-8')
                try:
                    ast.parse(text, filename=str(path))
                    check(checks, errors, name + '_ast', True)
                except SyntaxError:
                    check(checks, errors, name + '_ast', False)
                forbid(checks, errors, text, name + '_no_nvml', 'pynvml')
                forbid(checks, errors, text, name + '_no_subprocess', 'subprocess')
                forbid(checks, errors, text, name + '_no_cuda_visibility', 'CUDA_VISIBLE_DEVICES')
                forbid(checks, errors, text, name + '_no_gpu_tool', 'nvidia' + '-smi')
        if paths['preflight'].is_file():
            text = paths['preflight'].read_text(encoding='utf-8')
            for name, token in {
                'contract_name': 'fable5_g2_unmodified9_trace_preflight_v1.json',
                'original_bundle': 'g2_rebuild_witness_v5_20260728T125440Z_cpu',
                'native_snapshot': 'native_e0_snapshot',
                'independent_oracle': 'exact_topk(',
                'no_static_certificate_claim': 'NOT_ESTABLISHED_STATICALLY',
                'nine_events': "'event_count': 9",
            }.items():
                need(checks, errors, text, 'preflight_' + name, token)
        if paths['validator'].is_file():
            text = paths['validator'].read_text(encoding='utf-8')
            for name, token in {
                'nine_ops': 'EXPECTED = [',
                'direct_A': 'A_not_direct',
                'capacity_B': 'B_not_same_leaf_capacity_delta',
                'reuse_C': 'C_not_direct_slot_reuse',
                'receipt_sidecar': 'sidecar_receipt_selection',
                'independent_oracle': 'independent_exact_oracle_mismatch',
                'policy_equivalence': 'policy_result_mismatch',
                'no_performance_scope': 'no timing/workload/performance/C2/C3/deployment',
            }.items():
                need(checks, errors, text, 'validator_' + name, token)
        if paths['guard'].is_file():
            text = paths['guard'].read_text(encoding='utf-8')
            for name, token in {
                'execute_gate': '[[ "$EXECUTE" == 1 ]] ||',
                'gpu_uuid': 'GPU-CONFIGURE-ARCHIVE-DEVICE',
                'preflight': 'preflight_fable5_g2_unmodified9_trace_v1.py',
                'validator': 'validate_fable5_g2_unmodified9_gpu_receipts_v1.py',
                'static_audit': 'audit_fable5_g2_unmodified9_gpu_static_v1.py',
                'pinned_nvml_python': "readonly NVML_PYTHON_BIN='/workspace/project/livsyn/.venv/bin/python'",
                'binding_gate': "-c 'import pynvml'",
                'safe_policy': '--policy safe_c1',
                'buffer_policy': '--policy buffer_only',
                'coverage_disabled_in_runner': '--require-coverage 0',
                'correctness_scope': 'unmodified selected G2 L2 matched correctness only',
            }.items():
                need(checks, errors, text, 'guard_' + name, token)
            forbid(checks, errors, text, 'guard_no_gpu_tool', 'nvidia' + '-smi')
            execute = text.find('[[ "$EXECUTE" == 1 ]] ||')
            preflight = text.find('"$PYTHON_BIN" "$TRACE_PREFLIGHT"')
            run_dir = text.find('RUN_DIR="$ROOT/runs/$RUN_NAME"')
            audit = text.find('"$PYTHON_BIN" "$STATIC_AUDIT"')
            nvml = text.find('try:\n import pynvml')
            cuda = text.find('export CUDA_VISIBLE_DEVICES')
            check(checks, errors, 'guard_order_execute_preflight', execute >= 0 and preflight >= 0 and execute < preflight)
            check(checks, errors, 'guard_order_preflight_before_run_dir', preflight >= 0 and run_dir >= 0 and preflight < run_dir)
            check(checks, errors, 'guard_order_static_before_nvml', audit >= 0 and nvml >= 0 and audit < nvml)
            check(checks, errors, 'guard_order_nvml_before_cuda', nvml >= 0 and cuda >= 0 and nvml < cuda)
        if paths['manifest'].is_file():
            manifest = read_obj(paths['manifest'])
            check(checks, errors, 'manifest_schema_boundary',
                  manifest.get('schema') == 'fable5-g2-unmodified9-gpu-runner-v1'
                  and manifest.get('status') == 'CPU_BUILT_PRE_GPU'
                  and manifest.get('execution_boundary') == {'compiled': True, 'cuda_binary_executed': False, 'gpu_used': False, 'nvml_queried': False})
            variant = manifest.get('variant', {})
            check(checks, errors, 'manifest_variant',
                  variant.get('id') == VARIANT and variant.get('input_modification') == 'none'
                  and 'performance' in variant.get('prohibited_claims', []))
            runner = manifest.get('runner', {})
            check(checks, errors, 'manifest_runner_pin',
                  runner.get('source_sha256') == RUNNER_SHA
                  and runner.get('binary_sha256') == (sha(paths['binary']) if paths['binary'].is_file() else None)
                  and runner.get('cpu_build_receipt_sha256') == (sha(paths['build_receipt']) if paths['build_receipt'].is_file() else None))
            input_doc = manifest.get('input', {})
            bundle = under(root, input_doc.get('bundle'))
            trace = under(root, input_doc.get('trace'))
            contract = under(root, input_doc.get('preflight_contract'))
            snapshot = under(root, input_doc.get('native_e0_snapshot'))
            check(checks, errors, 'manifest_input_pins',
                  bundle is not None and bundle.is_dir() and trace == paths['bundle_trace']
                  and input_doc.get('trace_sha256') == (sha(trace) if trace is not None and trace.is_file() else None)
                  and contract == paths['contract']
                  and input_doc.get('preflight_contract_sha256') == (sha(contract) if contract is not None and contract.is_file() else None)
                  and snapshot is not None and snapshot.is_file()
                  and input_doc.get('native_e0_snapshot_sha256') == sha(snapshot))
            tools = manifest.get('tools', {})
            expected_paths = {'guard': paths['guard'], 'receipt_validator': paths['validator'], 'static_audit': paths['audit'], 'trace_preflight': paths['preflight']}
            ok = True
            for name, path in expected_paths.items():
                candidate = under(root, tools.get(name))
                ok = ok and candidate == path and path.is_file() and tools.get(name + '_sha256') == sha(path)
            check(checks, errors, 'manifest_tool_pins', ok)
        for name, path in paths.items():
            report[name + '_sha256'] = sha(path) if path.is_file() else None
        report['status'] = 'PASS_FABLE5_G2_UNMODIFIED9_GPU_STATIC_AUDIT' if not errors else 'FAIL_FABLE5_G2_UNMODIFIED9_GPU_STATIC_AUDIT'
    except Exception as exc:
        errors.append(f'exception:{type(exc).__name__}:{exc}')
        report['status'] = 'FAIL_FABLE5_G2_UNMODIFIED9_GPU_STATIC_AUDIT'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(report['status'])
    return 0 if not errors else 2


if __name__ == '__main__':
    raise SystemExit(main())
