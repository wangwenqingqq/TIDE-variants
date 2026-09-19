#!/usr/bin/env python3
"""Fail-closed CPU-only artifact pinning for the isolated G2 v5 witness.

This module never invokes CUDA, a CUDA binary, nvidia-smi, or subprocesses.
It pins the exact source quote-include closure, linked binary, guard/finalizer
scripts, CPU bundle payload, and frozen v2/v3 external inputs that a future
one-shot guarded execution would consume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v5_g2_artifact_pinned')
SOURCE = ROOT / 'src' / 'safe_c1_dynamic_gts.cu'
INCLUDE_ROOT = ROOT / 'include'
BINARY = ROOT / 'bin' / 'GTS_safe_c1_g2_v5_rebuild_witness'
BUNDLES_ROOT = ROOT / 'bundles'
PIN_NAME = 'execution_artifact_pin_v5.json'

V2_ROOT = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v2_search_native')
V2_SELECTION = V2_ROOT / 'bundles' / 'g1a_v2_identity_oracle_selection_20260728T110122Z' / 'candidate_selection_v2.json'
V3_ROOT = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v3_capacity_witness')
V3_BUNDLE = V3_ROOT / 'bundles' / 'g1b_capacity_witness_20260728T114307Z_cpu'
V3_RESULT = V3_BUNDLE / 'g1b_capacity_witness_v3.json'
V3_MANIFEST = V3_BUNDLE / 'manifest.json'
V3_PAYLOADS = (
    'pool.i16', 'queries.i16', 'stable_id_to_pool_row.i32', 'initial_base_stable_ids.i32',
)

CRITICAL_TOOL_RELS = (
    'tools/artifact_pinning_v5.py',
    'tools/audit_g2_static_contract_v5.py',
    'tools/preflight_g2_rebuild_witness_v5.py',
    'tools/finalize_g2_rebuild_witness_v5.py',
    'tools/run_g2_rebuild_witness_guarded_v5.sh',
    'tools/safe_c1_v5_rebuild_bundle_contract.py',
    'tools/make_g2_rebuild_witness_bundle_v5.py',
    'tools/test_g2_rebuild_witness_pipeline_v5.py',
)

PIN_SCHEMA = 'safe-c1-g2-v5-execution-artifact-pin'
VERIFY_SCHEMA = 'safe-c1-g2-v5-execution-artifact-pin-verification'
PIN_STATUS = 'PREPARED_CPU_ONLY_EXECUTION_ARTIFACT_PIN'
VERIFY_PASS = 'PASS_CPU_ONLY_EXECUTION_ARTIFACT_PIN_VERIFICATION'
VERIFY_FAIL = 'FAIL_CPU_ONLY_EXECUTION_ARTIFACT_PIN_VERIFICATION'
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"\\\n]+)"', re.MULTILINE)


class PinError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    ensure_regular_file(path)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def ensure_no_symlink_component(path: Path) -> None:
    """Reject a symlink in the supplied absolute path, including a parent."""
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            # The caller will produce the stronger missing-path error when needed.
            break
        if stat.S_ISLNK(mode):
            raise PinError(f'symlink_component_forbidden:{current}')


def ensure_regular_file(path: Path) -> Path:
    path = _absolute(path)
    ensure_no_symlink_component(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise PinError(f'missing_file:{path}') from exc
    if not stat.S_ISREG(mode):
        raise PinError(f'non_regular_file_forbidden:{path}')
    return path


def ensure_directory(path: Path) -> Path:
    path = _absolute(path)
    ensure_no_symlink_component(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise PinError(f'missing_directory:{path}') from exc
    if not stat.S_ISDIR(mode):
        raise PinError(f'not_directory:{path}')
    return path


def assert_under(path: Path, parent: Path, label: str) -> None:
    try:
        _absolute(path).relative_to(_absolute(parent))
    except ValueError as exc:
        raise PinError(f'{label}_outside_allowed_root:{path}') from exc


def canonical_root(root: Path) -> Path:
    root = ensure_directory(root)
    if root != ROOT:
        raise PinError(f'root_must_equal_canonical_v5_root:{root}')
    return root


def canonical_bundle(root: Path, bundle: Path) -> Path:
    bundle = ensure_directory(bundle)
    bundles = ensure_directory(root / 'bundles')
    if bundle.parent != bundles:
        raise PinError(f'bundle_must_be_direct_child_of_v5_bundles_root:{bundle}')
    return bundle


def canonical_binary(root: Path, binary: Path) -> Path:
    binary = ensure_regular_file(binary)
    expected = _absolute(root / 'bin' / BINARY.name)
    if binary != expected:
        raise PinError(f'binary_must_equal_canonical_v5_binary:{binary}')
    if not os.access(binary, os.X_OK):
        raise PinError(f'binary_not_executable:{binary}')
    return binary


def canonical_pin_path(bundle: Path, pin: Path) -> Path:
    pin = _absolute(pin)
    expected = _absolute(bundle / PIN_NAME)
    if pin != expected:
        raise PinError(f'pin_must_be_bundle_local_canonical_name:{pin}')
    return pin


def load_json(path: Path) -> dict[str, Any]:
    ensure_regular_file(path)
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PinError(f'invalid_json:{path}:{exc}') from exc
    if not isinstance(value, dict):
        raise PinError(f'json_not_object:{path}')
    return value


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path = _absolute(path)
    ensure_directory(path.parent)
    ensure_no_symlink_component(path.parent)
    raw = json.dumps(value, indent=2, sort_keys=True) + '\n'
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise PinError(f'refuse_overwrite_existing_pin:{path}') from exc
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        handle.write(raw)


def write_json_report(path: Path, value: dict[str, Any]) -> None:
    path = _absolute(path)
    ensure_directory(path.parent)
    ensure_no_symlink_component(path.parent)
    if path.exists():
        ensure_regular_file(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def logical_local_path(path: Path) -> str:
    path = _absolute(path)
    for root in (ROOT / 'src', ROOT / 'include'):
        try:
            return str(path.relative_to(_absolute(root.parent)))
        except ValueError:
            continue
    raise PinError(f'include_closure_path_outside_v5_src_or_include:{path}')


def resolve_quote_include(current: Path, include_name: str) -> Path:
    candidate_rel = Path(include_name)
    if candidate_rel.is_absolute() or '..' in candidate_rel.parts:
        raise PinError(f'unsafe_quote_include_path:{current}:{include_name}')
    candidates = (current.parent / candidate_rel, INCLUDE_ROOT / candidate_rel)
    for candidate in candidates:
        candidate = _absolute(candidate)
        if candidate.exists() or candidate.is_symlink():
            ensure_regular_file(candidate)
            allowed = False
            for allowed_root in (ROOT / 'src', ROOT / 'include'):
                try:
                    candidate.relative_to(_absolute(allowed_root))
                    allowed = True
                    break
                except ValueError:
                    pass
            if not allowed:
                raise PinError(f'quote_include_outside_v5_closure_roots:{current}:{candidate}')
            return candidate
    raise PinError(f'unresolved_quote_include:{current}:{include_name}')


def source_include_closure() -> dict[str, Any]:
    root = canonical_root(ROOT)
    source = ensure_regular_file(SOURCE)
    assert_under(source, root / 'src', 'source')
    pending = [source]
    seen: dict[Path, str] = {}
    while pending:
        current = _absolute(pending.pop())
        if current in seen:
            continue
        ensure_regular_file(current)
        try:
            text = current.read_text(encoding='utf-8')
        except UnicodeDecodeError as exc:
            raise PinError(f'non_utf8_source_or_header:{current}') from exc
        seen[current] = sha256(current)
        for match in _INCLUDE_RE.finditer(text):
            pending.append(resolve_quote_include(current, match.group(1)))
    files = {logical_local_path(path): digest for path, digest in sorted(seen.items(), key=lambda item: logical_local_path(item[0]))}
    closure_descriptor = {
        'include_search_roots': [str(_absolute(INCLUDE_ROOT))],
        'source': logical_local_path(source),
        'files_sha256': files,
    }
    return {
        **closure_descriptor,
        'sha256': sha256_bytes(canonical_json_bytes(closure_descriptor)),
    }


def expected_artifact_paths(bundle: Path) -> dict[str, Any]:
    return {
        'root': str(_absolute(ROOT)),
        'source': str(_absolute(SOURCE)),
        'include_search_roots': [str(_absolute(INCLUDE_ROOT))],
        'binary': str(_absolute(BINARY)),
        'bundle': str(_absolute(bundle)),
        'pin': str(_absolute(bundle / PIN_NAME)),
        'external_v2_selection': str(_absolute(V2_SELECTION)),
        'external_v3_result': str(_absolute(V3_RESULT)),
        'external_v3_bundle_manifest': str(_absolute(V3_MANIFEST)),
        'external_v3_bundle': str(_absolute(V3_BUNDLE)),
    }


def collect_actual_hashes(root: Path, bundle: Path, binary: Path) -> dict[str, Any]:
    canonical_root(root)
    bundle = canonical_bundle(ROOT, bundle)
    binary = canonical_binary(ROOT, binary)
    manifest_path = bundle / 'manifest.json'
    manifest = load_json(manifest_path)
    if manifest.get('schema') != 'safe-c1-g2-rebuild-witness-v5-bundle-manifest':
        raise PinError('wrong_v5_bundle_manifest_schema')
    if manifest.get('status') != 'PREPARED_CPU_ONLY_G2_REBUILD_WITNESS':
        raise PinError('bundle_not_cpu_prepared')
    if manifest.get('gpu_used') is not False or manifest.get('cuda_binary_executed') is not False:
        raise PinError('bundle_claims_gpu_or_cuda_execution')
    payload = manifest.get('files_sha256')
    if not isinstance(payload, dict) or not payload:
        raise PinError('bundle_payload_hash_map_missing')
    payload_actual: dict[str, str] = {}
    for name, declared in sorted(payload.items()):
        if not isinstance(name, str) or not isinstance(declared, str):
            raise PinError('bundle_payload_hash_map_invalid')
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or len(rel.parts) != 1:
            raise PinError(f'unsafe_bundle_payload_name:{name}')
        payload_file = bundle / rel
        digest = sha256(payload_file)
        if digest != declared:
            raise PinError(f'bundle_manifest_payload_mismatch:{name}')
        payload_actual[name] = digest
    tool_hashes: dict[str, str] = {}
    for rel in CRITICAL_TOOL_RELS:
        path = ROOT / rel
        assert_under(path, ROOT / 'tools', 'critical_tool')
        tool_hashes[rel] = sha256(path)
    v3_payloads = {name: sha256(V3_BUNDLE / name) for name in V3_PAYLOADS}
    return {
        'source_include_closure': source_include_closure(),
        'binary': {'path': str(binary), 'sha256': sha256(binary)},
        'critical_tools': tool_hashes,
        'bundle': {
            'manifest_path': str(_absolute(manifest_path)),
            'manifest_sha256': sha256(manifest_path),
            'payload_sha256': payload_actual,
        },
        'external_v2': {
            'selection_path': str(_absolute(V2_SELECTION)),
            'selection_sha256': sha256(V2_SELECTION),
        },
        'external_v3': {
            'result_path': str(_absolute(V3_RESULT)),
            'result_sha256': sha256(V3_RESULT),
            'bundle_manifest_path': str(_absolute(V3_MANIFEST)),
            'bundle_manifest_sha256': sha256(V3_MANIFEST),
            'immutable_payload_sha256': v3_payloads,
        },
    }


def nested_differences(expected: Any, actual: Any, prefix: str = '') -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        errors: list[str] = []
        keys = set(expected) | set(actual)
        for key in sorted(keys):
            child = f'{prefix}.{key}' if prefix else str(key)
            if key not in expected:
                errors.append(f'unexpected_actual_hash_field:{child}')
            elif key not in actual:
                errors.append(f'missing_actual_hash_field:{child}')
            else:
                errors.extend(nested_differences(expected[key], actual[key], child))
        return errors
    if expected != actual:
        return [f'hash_mismatch:{prefix}']
    return []


def verify_pin_payload(root: Path, bundle: Path, binary: Path, pin_path: Path, pin: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    root = canonical_root(root)
    bundle = canonical_bundle(root, bundle)
    binary = canonical_binary(root, binary)
    pin_path = canonical_pin_path(bundle, pin_path)
    expected_paths = expected_artifact_paths(bundle)
    if pin.get('schema') != PIN_SCHEMA:
        errors.append('pin_schema_invalid')
    if pin.get('status') != PIN_STATUS:
        errors.append('pin_status_invalid')
    paths = pin.get('artifact_paths')
    if not isinstance(paths, dict):
        errors.append('pin_artifact_paths_missing')
    else:
        errors.extend('pinned_path_mismatch:' + item for item in nested_differences(expected_paths, paths, 'artifact_paths'))
    actual = collect_actual_hashes(root, bundle, binary)
    pinned = pin.get('pinned_hashes')
    if not isinstance(pinned, dict):
        errors.append('pin_hashes_missing')
    else:
        errors.extend(nested_differences(pinned, actual, 'pinned_hashes'))
    return actual, errors


def create_execution_artifact_pin(root: Path, bundle: Path, binary: Path, out: Path) -> dict[str, Any]:
    root = canonical_root(root)
    bundle = canonical_bundle(root, bundle)
    binary = canonical_binary(root, binary)
    out = canonical_pin_path(bundle, out)
    if out.exists() or out.is_symlink():
        raise PinError(f'refuse_overwrite_existing_pin:{out}')
    actual = collect_actual_hashes(root, bundle, binary)
    pin = {
        'schema': PIN_SCHEMA,
        'status': PIN_STATUS,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'gpu_used': False,
        'cuda_binary_executed': False,
        'nvidia_smi_called': False,
        'scope': ('CPU-only execution-artifact identity pin; it authorizes neither a GPU run nor any performance/general correctness claim'),
        'artifact_paths': expected_artifact_paths(bundle),
        'pinned_hashes': actual,
        'external_policy': 'v2/v3 are read-only external hashes; no v2/v3 result is copied into the v5 bundle',
    }
    write_json_exclusive(out, pin)
    return pin


def verify_execution_artifact_pin(root: Path, bundle: Path, binary: Path, pin_path: Path) -> dict[str, Any]:
    try:
        root = canonical_root(root)
        bundle = canonical_bundle(root, bundle)
        binary = canonical_binary(root, binary)
        pin_path = canonical_pin_path(bundle, pin_path)
        pin = load_json(pin_path)
        actual, errors = verify_pin_payload(root, bundle, binary, pin_path, pin)
        report = {
            'schema': VERIFY_SCHEMA,
            'status': VERIFY_PASS if not errors else VERIFY_FAIL,
            'gpu_used': False,
            'cuda_binary_executed': False,
            'nvidia_smi_called': False,
            'root': str(root),
            'bundle': str(bundle),
            'binary': str(binary),
            'pin_path': str(pin_path),
            'pin_sha256': sha256(pin_path),
            'actual_hashes': actual,
            'errors': errors,
        }
    except Exception as exc:
        report = {
            'schema': VERIFY_SCHEMA,
            'status': VERIFY_FAIL,
            'gpu_used': False,
            'cuda_binary_executed': False,
            'nvidia_smi_called': False,
            'root': str(_absolute(root)),
            'bundle': str(_absolute(bundle)),
            'binary': str(_absolute(binary)),
            'pin_path': str(_absolute(pin_path)),
            'actual_hashes': {},
            'errors': [f'artifact_pin_verification_exception:{exc}'],
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('create', 'verify'):
        item = sub.add_parser(name)
        item.add_argument('--root', type=Path, required=True)
        item.add_argument('--bundle', type=Path, required=True)
        item.add_argument('--binary', type=Path, required=True)
        item.add_argument('--pin', type=Path, required=True)
        if name == 'verify':
            item.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'create':
        try:
            pin = create_execution_artifact_pin(args.root, args.bundle, args.binary, args.pin)
            print(f'{PIN_STATUS} pin={args.pin} pin_payload_sha256={sha256_bytes(canonical_json_bytes(pin))}')
            return 0
        except Exception as exc:
            print(f'FAIL_CPU_ONLY_EXECUTION_ARTIFACT_PIN_CREATION:{exc}')
            return 2
    report = verify_execution_artifact_pin(args.root, args.bundle, args.binary, args.pin)
    try:
        write_json_report(args.out, report)
    except Exception as exc:
        print(f'FAIL_CPU_ONLY_EXECUTION_ARTIFACT_PIN_VERIFICATION_OUTPUT:{exc}')
        return 2
    print(report['status'])
    return 0 if report['status'] == VERIFY_PASS else 2


if __name__ == '__main__':
    raise SystemExit(main())
