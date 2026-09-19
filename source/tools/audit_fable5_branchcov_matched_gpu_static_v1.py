#!/usr/bin/env python3
"""Static-only audit for the Fable5 constructed branch-coverage GPU guard.

This audit is intentionally fail-closed and performs no compiler, CUDA binary,
GPU, NVML, timing, or workload action. It verifies only the immutable artifact
chain for a correctness micro-witness that is prohibited from performance or
real-workload claims.
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
VARIANT = 'constructed_branch_coverage_micro_witness_v1'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check(checks: dict[str, bool], errors: list[str], name: str, result: bool) -> None:
    checks[name] = result
    if not result:
        errors.append('failed:' + name)


def need_text(checks: dict[str, bool], errors: list[str], text: str, name: str, token: str) -> None:
    check(checks, errors, name, token in text)


def forbid_text(checks: dict[str, bool], errors: list[str], text: str, name: str, token: str) -> None:
    check(checks, errors, name, token not in text)


def path_under(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return None
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('json root is not an object')
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    report: dict[str, Any] = {
        'schema': 'fable5-branchcov-matched-gpu-static-audit-v1',
        'root': str(root),
        'execution_boundary': {
            'cpu_static_only': True,
            'compiler_invoked': False,
            'cuda_binary_executed': False,
            'gpu_used': False,
            'nvml_used': False,
            'nvidia_smi_called': False,
        },
        'scope': 'Static branch-coverage micro-witness audit only; no runtime correctness or performance result.',
        'checks': {},
        'errors': [],
    }
    checks: dict[str, bool] = report['checks']
    errors: list[str] = report['errors']
    try:
        check(checks, errors, 'isolated_root', root == ROOT)
        runner = root / 'src/fable5_matched_gpu_runner_v1.cu'
        core = root / 'src/safe_c1_dynamic_gts.cu'
        binary = root / 'build/GTS_fable5_matched_v1'
        receipt = root / 'provenance/fable5_matched_gpu_runner_v1_cpu_build_20260730.json'
        guard = root / 'tools/run_fable5_branchcov_matched_gpu1_guarded_v1.sh'
        preflight = root / 'tools/preflight_fable5_branchcov_matched_trace_v1.py'
        validator = root / 'tools/validate_fable5_matched_gpu_receipts_v1.py'
        contract = root / 'manifests/fable5_branchcov_matched_trace_preflight_v1.json'
        manifest = root / 'manifests/fable5_branchcov_matched_gpu_runner_v1.json'
        paths = {'runner': runner, 'core': core, 'binary': binary, 'receipt': receipt, 'guard': guard,
                 'preflight': preflight, 'validator': validator, 'contract': contract, 'manifest': manifest}
        for name, path in paths.items():
            check(checks, errors, 'exists_' + name, path.is_file())
        if core.is_file():
            report['core_sha256'] = sha(core)
            check(checks, errors, 'pinned_core_unchanged', report['core_sha256'] == CORE_SHA)
        if runner.is_file():
            report['runner_sha256'] = sha(runner)
            check(checks, errors, 'pinned_runner_unchanged', report['runner_sha256'] == RUNNER_SHA)
            text = runner.read_text(encoding='utf-8')
            for name, token in {
                'native_certificate': 'frozen.snapshot.certify_leaf(pool, id)',
                'native_receipt': 'run_gts_base_topk_with_receipt',
                'receipt_sidecar_scan': 'state.sidecar_candidates_for(base.visited_leaf_ids)',
                'exact_delta_scan': 'state.delta_ids().begin()',
                'full_active_oracle': 'exact_active_oracle(pool, query, state.active()',
                'rebuild_barrier': 'base deletion requires immediate REBUILD before ',
                'no_performance_scope': 'no timing, throughput, latency, C2, C3, or deployment claim',
            }.items():
                need_text(checks, errors, text, 'runner_' + name, token)
            for name, token in {
                'legacy_incremental': 'incremental_insert.cuh',
                'legacy_update': '#include "update.cuh"',
            }.items():
                forbid_text(checks, errors, text, 'runner_' + name, token)
        receipt_doc: dict[str, Any] = {}
        if receipt.is_file():
            receipt_doc = load_json(receipt)
            check(checks, errors, 'receipt_cpu_only',
                  receipt_doc.get('schema') == 'fable5-matched-gpu-runner-v1-cpu-build-receipt'
                  and receipt_doc.get('status') == 'PASS_CPU_BUILD_ONLY_NOT_EXECUTED'
                  and receipt_doc.get('source_sha256') == RUNNER_SHA
                  and receipt_doc.get('archived_core_sha256') == CORE_SHA
                  and receipt_doc.get('binary_sha256') == (sha(binary) if binary.is_file() else None)
                  and receipt_doc.get('cuda_binary_executed') is False
                  and receipt_doc.get('gpu_used') is False
                  and receipt_doc.get('nvml_used') is False)
            log = Path(receipt_doc.get('compile_log', '__missing__'))
            check(checks, errors, 'receipt_compile_log_pin', log.is_file() and receipt_doc.get('compile_log_sha256') == sha(log))
        if preflight.is_file():
            ptext = preflight.read_text(encoding='utf-8')
            try:
                ast.parse(ptext, filename=str(preflight))
                check(checks, errors, 'preflight_ast', True)
            except SyntaxError:
                check(checks, errors, 'preflight_ast', False)
            for name, token in {
                'branch_contract_name': 'fable5_branchcov_matched_trace_preflight_v1.json',
                'branch_variant': VARIANT,
                'candidate_contract_binding': 'candidate_branchcov_preflight_contract',
                'constructed_scope': 'constructed branch-coverage micro-witness',
                'static_certificate_limit': 'NOT_ESTABLISHED_STATICALLY',
            }.items():
                need_text(checks, errors, ptext, 'preflight_' + name, token)
            for name, token in {
                'nvml': 'pynvml', 'subprocess': 'subprocess', 'visibility': 'CUDA_VISIBLE_DEVICES',
                'legacy_gpu_tool': 'nvidia' + '-smi',
            }.items():
                forbid_text(checks, errors, ptext, 'preflight_forbid_' + name, token)
        contract_doc: dict[str, Any] = {}
        if contract.is_file():
            contract_doc = load_json(contract)
            variant = contract_doc.get('variant', {})
            check(checks, errors, 'contract_variant',
                  contract_doc.get('schema') == 'fable5-matched-trace-preflight-contract-v1'
                  and variant.get('id') == VARIANT
                  and variant.get('source_bundle') == 'fable5_branch_coverage_e1_20260730'
                  and 'performance' in variant.get('prohibited_claims', [])
                  and len(contract_doc.get('event_sequence', [])) == 12)
            witness = path_under(root, variant.get('witness'))
            check(checks, errors, 'contract_witness_pin', witness is not None and witness.is_file() and variant.get('witness_sha256') == sha(witness))
        if validator.is_file():
            text = validator.read_text(encoding='utf-8')
            for name, token in {
                'shared_trace': 'trace_sha256', 'certificate_receipt': 'native_certificate_receipt_mismatch',
                'traversal_receipt': 'native_traversal_receipt_mismatch', 'oracle': 'oracle_result_mismatch',
                'capacity': 'safe_missing_capacity_same_leaf_proof', 'visibility': 'safe_direct_never_visible_in_exact_result',
            }.items():
                need_text(checks, errors, text, 'validator_' + name, token)
            forbid_text(checks, errors, text, 'validator_legacy_gpu_tool', 'nvidia' + '-smi')
        if guard.is_file():
            text = guard.read_text(encoding='utf-8')
            for name, token in {
                'execute_gate': '[[ "$EXECUTE" == 1 ]] ||',
                'gpu_uuid': 'GPU-CONFIGURE-ARCHIVE-DEVICE',
                'branch_preflight': 'preflight_fable5_branchcov_matched_trace_v1.py',
                'branch_contract': 'fable5_branchcov_matched_trace_preflight_v1.json',
                'pinned_nvml_python': "readonly NVML_PYTHON_BIN='/workspace/project/livsyn/.venv/bin/python'",
                'binding_import_gate': "-c 'import pynvml'",
                'safe_policy': '--policy safe_c1', 'buffer_policy': '--policy buffer_only',
                'validator': '"$VALIDATOR" --safe-results',
                'constructed_scope': 'constructed branch-coverage matched correctness only',
            }.items():
                need_text(checks, errors, text, 'guard_' + name, token)
            forbid_text(checks, errors, text, 'guard_legacy_gpu_tool', 'nvidia' + '-smi')
            execute = text.find('[[ "$EXECUTE" == 1 ]] ||')
            preflight_pos = text.find('"$PYTHON_BIN" "$TRACE_PREFLIGHT"')
            binding = text.find("-c 'import pynvml'")
            run_dir = text.find('RUN_DIR="$ROOT/runs/$RUN_NAME"')
            audit = text.find('"$PYTHON_BIN" "$STATIC_AUDIT"')
            nvml = text.find('try:\n    import pynvml')
            check(checks, errors, 'guard_order_execute_preflight', execute >= 0 and preflight_pos >= 0 and execute < preflight_pos)
            check(checks, errors, 'guard_order_preflight_binding_run_dir', preflight_pos >= 0 and binding >= 0 and run_dir >= 0 and preflight_pos < binding < run_dir)
            check(checks, errors, 'guard_order_static_audit_before_nvml', audit >= 0 and nvml >= 0 and audit < nvml)
        if manifest.is_file():
            m = load_json(manifest)
            check(checks, errors, 'manifest_schema_boundary',
                  m.get('schema') == 'fable5-branchcov-matched-gpu-runner-v1'
                  and m.get('status') == 'CPU_BUILT_PRE_GPU'
                  and m.get('execution_boundary') == {'compiled': True, 'cuda_binary_executed': False, 'gpu_used': False, 'nvml_queried': False})
            v = m.get('variant', {})
            check(checks, errors, 'manifest_variant',
                  v.get('id') == VARIANT and 'performance' in v.get('prohibited_claims', [])
                  and 'real workload' in m.get('non_claims', []))
            r = m.get('runner', {})
            check(checks, errors, 'manifest_runner_pin',
                  r.get('source_sha256') == RUNNER_SHA and r.get('binary_sha256') == (sha(binary) if binary.is_file() else None)
                  and r.get('cpu_build_receipt_sha256') == (sha(receipt) if receipt.is_file() else None))
            c = m.get('candidate', {})
            bundle = path_under(root, c.get('bundle')); trace = path_under(root, c.get('trace')); candidate = path_under(root, c.get('candidate_contract')); ccontract = path_under(root, c.get('branch_preflight_contract'))
            check(checks, errors, 'manifest_candidate_pins',
                  bundle is not None and bundle.is_dir() and trace is not None and trace.is_file() and candidate is not None and candidate.is_file() and ccontract == contract
                  and c.get('trace_sha256') == (sha(trace) if trace is not None and trace.is_file() else None)
                  and c.get('candidate_contract_sha256') == (sha(candidate) if candidate is not None and candidate.is_file() else None)
                  and c.get('branch_preflight_contract_sha256') == (sha(contract) if contract.is_file() else None))
            tools = m.get('tools', {})
            check(checks, errors, 'manifest_tool_pins',
                  tools.get('guard_sha256') == (sha(guard) if guard.is_file() else None)
                  and tools.get('branch_preflight_sha256') == (sha(preflight) if preflight.is_file() else None)
                  and tools.get('receipt_validator_sha256') == (sha(validator) if validator.is_file() else None))
        for name, path in paths.items():
            report[name + '_sha256'] = sha(path) if path.is_file() else None
        report['status'] = 'PASS_FABLE5_BRANCHCOV_MATCHED_GPU_STATIC_AUDIT' if not errors else 'FAIL_FABLE5_BRANCHCOV_MATCHED_GPU_STATIC_AUDIT'
    except Exception as exc:  # noqa: BLE001
        errors.append(f'exception:{type(exc).__name__}:{exc}')
        report['status'] = 'FAIL_FABLE5_BRANCHCOV_MATCHED_GPU_STATIC_AUDIT'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(report['status'])
    return 0 if not errors else 2


if __name__ == '__main__':
    raise SystemExit(main())
