#!/usr/bin/env python3
"""Read-only pre-NVML contract gate for the unmodified G2 9-op Safe-C1 witness.

This tool never imports NVML, executes CUDA, creates a run directory, or edits
an input. It byte-pins the original 9-event witness, independently replays its
exact integer-L2 oracle, and binds it to the native E0 snapshot. Passing this
gate is not a native-certificate, traversal, workload, or performance result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

CANONICAL_ROOT = Path('/workspace/experiments/tide_safe_c1_20260730/safe_c1_fable5_matched_v1')
CONTRACT_NAME = 'fable5_g2_unmodified9_trace_preflight_v1.json'
VARIANT_ID = 'unmodified_g2_l2_correctness_witness_v1'
SOURCE_BUNDLE = 'g2_rebuild_witness_v5_20260728T125440Z_cpu'
SCHEMA = 'fable5-g2-unmodified9-trace-preflight-contract-v1'
REPORT_SCHEMA = 'fable5-g2-unmodified9-trace-preflight-report-v1'
PASS = 'PASS_FABLE5_G2_UNMODIFIED9_TRACE_STATIC_PREFLIGHT'
FAIL = 'FAIL_FABLE5_G2_UNMODIFIED9_TRACE_STATIC_PREFLIGHT'
HEADER = struct.Struct('<8sIIIIIIIfQ')
EVENT = struct.Struct('<IB3si')
OP_INSERT, OP_DELETE, OP_KNN, OP_REBUILD = 1, 2, 3, 5
OP_NAMES = {OP_INSERT: 'INSERT', OP_DELETE: 'DELETE', OP_KNN: 'KNN', OP_REBUILD: 'REBUILD'}
EXPECTED_HEADER = {
    'magic': 'E1GTRC02', 'version': 2, 'dimension': 32, 'base_n': 4096,
    'reservoir_n': 2048, 'pool_n': 6144, 'query_n': 3, 'k': 10,
    'radius': 300.0, 'event_count': 9,
}
EXPECTED_SEQUENCE = [
    {'op_index': 0, 'op': 'INSERT', 'slot': 'A'},
    {'op_index': 1, 'op': 'INSERT', 'slot': 'B'},
    {'op_index': 2, 'op': 'KNN', 'slot': 'query_before'},
    {'op_index': 3, 'op': 'DELETE', 'slot': 'A'},
    {'op_index': 4, 'op': 'INSERT', 'slot': 'C'},
    {'op_index': 5, 'op': 'KNN', 'slot': 'query_middle'},
    {'op_index': 6, 'op': 'DELETE', 'slot': 'base_delete'},
    {'op_index': 7, 'op': 'REBUILD', 'slot': 'rebuild'},
    {'op_index': 8, 'op': 'KNN', 'slot': 'query_after'},
]


class PreflightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in '0123456789abcdef' for ch in value)


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise PreflightError(f'invalid_{label}_json:{type(exc).__name__}') from exc
    require(isinstance(value, dict), f'invalid_{label}_root')
    return value


def under(root: Path, relative: Any, label: str) -> Path:
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), f'{label}_relative_path_invalid')
    target = (root / relative).resolve(strict=True)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise PreflightError(f'{label}_path_escapes_root') from exc
    return target


def parse_trace(path: Path) -> tuple[dict[str, Any], list[tuple[int, int, int]]]:
    raw = path.read_bytes()
    require(len(raw) >= HEADER.size, 'truncated_trace_header')
    magic, version, dimension, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = HEADER.unpack_from(raw)
    try:
        magic_text = magic.decode('ascii')
    except UnicodeDecodeError as exc:
        raise PreflightError('trace_magic_non_ascii') from exc
    header = {
        'magic': magic_text, 'version': int(version), 'dimension': int(dimension),
        'base_n': int(base_n), 'reservoir_n': int(reservoir_n), 'pool_n': int(pool_n),
        'query_n': int(query_n), 'k': int(k), 'radius': float(radius), 'event_count': int(event_count),
    }
    require(header == EXPECTED_HEADER, 'trace_header_not_exact_unmodified_g2_9op_abi')
    require(len(raw) == HEADER.size + EVENT.size * event_count, 'trace_byte_count_mismatch')
    events: list[tuple[int, int, int]] = []
    for index in range(event_count):
        op_index, op, reserved, argument = EVENT.unpack_from(raw, HEADER.size + EVENT.size * index)
        require(op_index == index and reserved == b'\x00\x00\x00', f'bad_event_encoding:{index}')
        require(op in OP_NAMES, f'unknown_event_opcode:{index}')
        events.append((int(op_index), int(op), int(argument)))
    return header, events


def slots_from_trace(events: list[tuple[int, int, int]]) -> dict[str, int]:
    require(len(events) == len(EXPECTED_SEQUENCE), 'trace_event_count_not_9')
    slots: dict[str, int] = {}
    for expected, actual in zip(EXPECTED_SEQUENCE, events):
        index, op, argument = actual
        require(index == expected['op_index'], f'event_index_mismatch:{index}')
        require(OP_NAMES[op] == expected['op'], f'event_opcode_mismatch:{index}')
        slot = expected['slot']
        if slot == 'rebuild':
            require(argument == 0, 'rebuild_argument_not_zero')
        else:
            slots[slot] = argument
    require(events[3][2] == slots['A'], 'delete_A_does_not_match_insert_A')
    require(events[6][2] == slots['base_delete'], 'base_delete_slot_mismatch')
    require(len({slots['A'], slots['B'], slots['C'], slots['base_delete']}) == 4, 'dynamic_and_base_ids_not_distinct')
    return slots


def validate_contract(root: Path, contract_path: Path, bundle: Path, trace: Path, trace_sha: str,
                      header: dict[str, Any], slots: dict[str, int]) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json_object(contract_path, 'contract')
    require(contract.get('schema') == SCHEMA and contract.get('status') == 'STATIC_PRE_NVML_CONTRACT_ONLY', 'contract_schema_or_status')
    require(contract.get('root') == str(CANONICAL_ROOT), 'contract_root_mismatch')
    require(contract.get('execution_boundary') == {'cuda_binary_executed': False, 'gpu_used': False, 'nvml_queried': False}, 'contract_execution_boundary')
    variant = contract.get('variant')
    require(isinstance(variant, dict) and variant.get('id') == VARIANT_ID and variant.get('source_bundle') == SOURCE_BUNDLE, 'contract_variant_mismatch')
    prohibited = variant.get('prohibited_claims')
    require(isinstance(prohibited, list) and all(item in prohibited for item in ('performance', 'C2', 'C3', 'deployment')), 'contract_nonclaim_boundary')
    require(contract.get('trace_abi', {}).get('header') == EXPECTED_HEADER, 'contract_header_mismatch')
    trace_abi = contract.get('trace_abi', {})
    require(trace_abi.get('header_bytes') == HEADER.size and trace_abi.get('event_bytes') == EVENT.size, 'contract_trace_abi_size')
    require(trace_abi.get('trace_filename') == 'trace.g2trc' and trace_abi.get('trace_sha256') == trace_sha, 'contract_trace_identity')
    require(contract.get('event_sequence') == EXPECTED_SEQUENCE, 'contract_event_sequence')
    require(contract.get('slot_ids') == slots, 'contract_slot_ids')
    require(bundle == root / 'bundles' / SOURCE_BUNDLE, 'bundle_not_canonical_original_g2')
    require(trace == bundle / 'trace.g2trc', 'trace_not_canonical_original_g2')
    layout = contract.get('bundle_layout')
    require(isinstance(layout, dict) and layout.get('stable_id_layout') == 'identity_full_immutable_pool_seeded_stable_ids_v5', 'contract_layout')
    required_files = layout.get('required_files')
    require(isinstance(required_files, dict) and set(required_files) == {'pool.i16', 'queries.i16', 'stable_id_to_pool_row.i32', 'initial_base_stable_ids.i32'}, 'contract_required_files')
    for name, detail in required_files.items():
        require(isinstance(detail, dict) and isinstance(detail.get('bytes'), int) and is_sha256(detail.get('sha256')), f'contract_file_detail:{name}')
        payload = bundle / name
        require(payload.is_file() and payload.stat().st_size == detail['bytes'] and sha256_file(payload) == detail['sha256'], f'payload_pin_mismatch:{name}')
    source_hashes = contract.get('root_source_hashes')
    require(isinstance(source_hashes, dict) and set(source_hashes) == {
        'src/safe_c1_dynamic_gts.cu', 'include/safe_c1_search_native_routing_certificate.hpp',
        'src/fable5_matched_gpu_runner_v1.cu', 'src/fable5_native_snapshot_exporter_v1.cu',
    }, 'contract_source_hash_set')
    for relative, digest in source_hashes.items():
        require(is_sha256(digest), f'contract_bad_source_hash:{relative}')
        source = under(root, relative, 'contract_source')
        require(source.is_file() and sha256_file(source) == digest, f'root_source_hash_mismatch:{relative}')
    snapshot_spec = contract.get('native_e0_snapshot')
    require(isinstance(snapshot_spec, dict) and is_sha256(snapshot_spec.get('sha256')), 'contract_snapshot_spec')
    snapshot_path = under(root, snapshot_spec.get('relative_path'), 'snapshot')
    require(snapshot_path.is_file() and sha256_file(snapshot_path) == snapshot_spec['sha256'], 'native_snapshot_hash_mismatch')
    snapshot = read_json_object(snapshot_path, 'native_snapshot')
    fields = snapshot_spec.get('required_fields')
    require(isinstance(fields, dict), 'snapshot_required_fields')
    require(snapshot.get('schema') == fields.get('schema') and snapshot.get('status') == fields.get('status'), 'native_snapshot_schema_or_status')
    require(snapshot.get('c2_used') is fields.get('c2_used') and snapshot.get('c3_used') is fields.get('c3_used'), 'native_snapshot_c2_c3_boundary')
    require(snapshot.get('residual_pruning', {}).get('mode') == fields.get('residual_pruning_mode'), 'native_snapshot_residual_mode')
    immutable = snapshot.get('immutable_input')
    require(isinstance(immutable, dict), 'native_snapshot_immutable_input')
    expected_immutable = {
        'header_trace_sha256': trace_sha,
        'pool_i16_sha256': required_files['pool.i16']['sha256'],
        'queries_i16_sha256': required_files['queries.i16']['sha256'],
        'initial_base_stable_ids_i32_sha256': required_files['initial_base_stable_ids.i32']['sha256'],
        'stable_id_to_pool_row_i32_sha256': required_files['stable_id_to_pool_row.i32']['sha256'],
        'core_source_sha256': source_hashes['src/safe_c1_dynamic_gts.cu'],
        'exporter_source_sha256': source_hashes['src/fable5_native_snapshot_exporter_v1.cu'],
        'matched_runner_source_sha256': source_hashes['src/fable5_matched_gpu_runner_v1.cu'],
    }
    require(all(immutable.get(key) == value for key, value in expected_immutable.items()), 'native_snapshot_input_binding')
    runtime = contract.get('runtime_requirements')
    expected_runtime = {
        'requires_A_native_certificate_accept_and_direct_sidecar': True,
        'requires_B_same_native_certificate_leaf_and_capacity_delta': True,
        'requires_C_direct_reuse_after_delete_A': True,
        'requires_native_traversal_receipt_to_make_A_and_C_visible': True,
        'requires_global_delta_B_visible_when_exact_topk': True,
        'requires_same_trace_and_full_active_set_exact_oracle_for_both_policies': True,
        'requires_base_delete_immediate_metadata_only_rebuild': True,
        'requires_immutable_pool_and_active_set_preservation_across_rebuild': True,
    }
    require(runtime == expected_runtime, 'contract_runtime_requirements')
    return contract, snapshot


def read_i16(path: Path, count: int) -> list[int]:
    raw = path.read_bytes()
    require(len(raw) == count * 2, f'i16_size:{path.name}')
    return list(struct.unpack(f'<{count}h', raw))


def read_i32(path: Path, count: int) -> list[int]:
    raw = path.read_bytes()
    require(len(raw) == count * 4, f'i32_size:{path.name}')
    return list(struct.unpack(f'<{count}i', raw))


def squared_l2(pool: list[int], queries: list[int], dim: int, ident: int, query_id: int) -> int:
    begin = ident * dim
    qbegin = query_id * dim
    return sum((pool[begin + axis] - queries[qbegin + axis]) ** 2 for axis in range(dim))


def exact_topk(pool: list[int], queries: list[int], dim: int, active: set[int], query_id: int, k: int) -> list[tuple[int, float]]:
    ordered = sorted((squared_l2(pool, queries, dim, ident, query_id), ident) for ident in active)
    require(len(ordered) >= k, 'active_below_k')
    require(len(ordered) == k or ordered[k - 1][0] != ordered[k][0], f'dynamic_k_boundary_tie:q{query_id}')
    return [(ident, math.sqrt(distance)) for distance, ident in ordered[:k]]


def replay_exact(bundle: Path, header: dict[str, Any], events: list[tuple[int, int, int]], slots: dict[str, int]) -> dict[str, Any]:
    dim, pool_n, base_n, query_n, k = header['dimension'], header['pool_n'], header['base_n'], header['query_n'], header['k']
    pool = read_i16(bundle / 'pool.i16', pool_n * dim)
    queries = read_i16(bundle / 'queries.i16', query_n * dim)
    mapping = read_i32(bundle / 'stable_id_to_pool_row.i32', pool_n)
    initial = read_i32(bundle / 'initial_base_stable_ids.i32', base_n)
    require(mapping == list(range(pool_n)) and initial == list(range(base_n)), 'identity_layout_mismatch')
    active = set(initial)
    query_rows: dict[int, list[tuple[int, float]]] = {}
    rebuild_pending = False
    active_before_rebuild: set[int] | None = None
    for index, op, argument in events:
        require((op == OP_REBUILD and argument == 0) or (0 <= argument < (query_n if op == OP_KNN else pool_n)), f'event_argument_range:{index}')
        if op == OP_INSERT:
            require(not rebuild_pending and argument not in active and base_n <= argument < pool_n, f'invalid_insert:{index}')
            active.add(argument)
        elif op == OP_DELETE:
            require(not rebuild_pending and argument in active, f'invalid_delete:{index}')
            active.remove(argument)
            if argument < base_n:
                rebuild_pending = True
                active_before_rebuild = set(active)
        elif op == OP_KNN:
            require(not rebuild_pending, f'query_while_rebuild_pending:{index}')
            query_rows[index] = exact_topk(pool, queries, dim, active, argument, k)
        elif op == OP_REBUILD:
            require(rebuild_pending and active_before_rebuild == active, 'invalid_immediate_rebuild')
            rebuild_pending = False
        else:
            raise PreflightError(f'unsupported_event:{index}')
    require(not rebuild_pending and active_before_rebuild is not None, 'missing_or_incomplete_rebuild')
    ids = {op: [item[0] for item in rows] for op, rows in query_rows.items()}
    require(slots['A'] in ids[2] and slots['B'] in ids[2], 'before_query_missing_A_or_B_exact_topk')
    require(slots['C'] in ids[5] and slots['C'] in ids[8], 'middle_or_after_query_missing_C_exact_topk')
    return {
        'query_exact_topk': {str(op): [[ident, distance] for ident, distance in rows] for op, rows in query_rows.items()},
        'named_dynamic_exact_visibility': {'op2_A': slots['A'] in ids[2], 'op2_B': slots['B'] in ids[2], 'op5_C': slots['C'] in ids[5], 'op8_C': slots['C'] in ids[8]},
        'active_count_after_rebuild': len(active),
        'active_after_rebuild_matches_pre_rebuild': True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    bundle = args.bundle.resolve(strict=True)
    trace = args.trace.resolve(strict=True)
    contract_path = args.contract.resolve(strict=True)
    require(root == CANONICAL_ROOT, 'root_not_canonical')
    require(contract_path == root / 'manifests' / CONTRACT_NAME, 'contract_not_canonical')
    require(is_sha256(args.trace_sha256), 'trace_sha256_format')
    require(sha256_file(trace) == args.trace_sha256, 'trace_sha256_mismatch')
    header, events = parse_trace(trace)
    slots = slots_from_trace(events)
    contract, snapshot = validate_contract(root, contract_path, bundle, trace, args.trace_sha256, header, slots)
    replay = replay_exact(bundle, header, events, slots)
    return {
        'schema': REPORT_SCHEMA,
        'status': PASS,
        'execution_boundary': {'gpu_used': False, 'cuda_binary_executed': False, 'nvml_queried': False},
        'scope': 'Read-only byte/ABI/CPU-exact-oracle gate for an unmodified selected G2 L2 correctness witness; not native certificate/traversal, workload, or performance evidence.',
        'root': str(root), 'bundle': str(bundle), 'trace': str(trace), 'trace_sha256': args.trace_sha256,
        'contract_sha256': sha256_file(contract_path), 'header': header, 'slot_ids': slots,
        'native_e0_snapshot': {'path': contract['native_e0_snapshot']['relative_path'], 'sha256': contract['native_e0_snapshot']['sha256'], 'c2_used': snapshot['c2_used'], 'c3_used': snapshot['c3_used']},
        'cpu_exact_replay': replay,
        'certificate_outcome': {'status': 'NOT_ESTABLISHED_STATICALLY', 'reason': 'Native runner must recompute strict certificates and traversal receipts at runtime.'},
        'errors': [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--trace-sha256', required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        report = {
            'schema': REPORT_SCHEMA, 'status': FAIL,
            'execution_boundary': {'gpu_used': False, 'cuda_binary_executed': False, 'nvml_queried': False},
            'scope': 'Read-only fail-closed pre-NVML gate; no native certificate, traversal, workload, or performance result.',
            'errors': [f'{type(exc).__name__}:{exc}'],
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding='utf-8')
    print(json.dumps(report, sort_keys=True, separators=(',', ':')))
    return 0 if report['status'] == PASS else 2


if __name__ == '__main__':
    raise SystemExit(main())
