#!/usr/bin/env python3
"""Independent CPU post-run validator for the unmodified G2 9-op GPU receipts.

It validates native runtime receipts from the existing no-timing matched runner
against a separately replayed full-active-set integer-L2 oracle. It does not
measure time or make workload, performance, C2, C3, or deployment claims.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

HEADER = struct.Struct('<8sIIIIIIIfQ')
EVENT = struct.Struct('<IB3si')
OP_INSERT, OP_DELETE, OP_KNN, OP_REBUILD = 1, 2, 3, 5
EXPECTED = [
    ('update', 'insert'), ('update', 'insert'), ('query', None),
    ('update', 'delete'), ('update', 'insert'), ('query', None),
    ('update', 'delete'), ('rebuild', None), ('query', None),
]
ROLE_IDS = {'A': 4330, 'B': 4357, 'C': 4399, 'base_delete': 0}
QUERY_IDS = {2: 0, 5: 2, 8: 2}


def fail_if(errors: list[str], condition: bool, label: str) -> None:
    if condition:
        errors.append(label)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f'non_object_jsonl:{path.name}:{line_no}')
        result.append(item)
    if not result:
        raise ValueError(f'empty_jsonl:{path.name}')
    return result


def as_ints(value: Any) -> list[int] | None:
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        return None
    return list(value)


def result_rows(value: Any) -> list[tuple[int, float]] | None:
    if not isinstance(value, list):
        return None
    result: list[tuple[int, float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], int):
            return None
        try:
            distance = float(row[1])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(distance):
            return None
        result.append((row[0], distance))
    return result


def equal_results(left: Any, right: Any) -> bool:
    a, b = result_rows(left), result_rows(right)
    if a is None or b is None or len(a) != len(b):
        return False
    return all(xid == yid and math.isclose(xd, yd, rel_tol=1e-8, abs_tol=1e-9)
               for (xid, xd), (yid, yd) in zip(a, b))


def decode_trace(path: Path, expected_sha: str) -> tuple[dict[str, int | float], list[tuple[int, int, int]]]:
    import hashlib
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha:
        raise ValueError('trace_sha_mismatch')
    if len(raw) < HEADER.size:
        raise ValueError('truncated_trace')
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, radius, count = HEADER.unpack_from(raw)
    if magic != b'E1GTRC02' or version != 2 or (dim, base_n, reservoir_n, pool_n, query_n, k, count) != (32, 4096, 2048, 6144, 3, 10, 9) or float(radius) != 300.0:
        raise ValueError('unexpected_trace_header')
    if len(raw) != HEADER.size + EVENT.size * count:
        raise ValueError('trace_size')
    events: list[tuple[int, int, int]] = []
    expected_ops = [(OP_INSERT, 4330), (OP_INSERT, 4357), (OP_KNN, 0), (OP_DELETE, 4330),
                    (OP_INSERT, 4399), (OP_KNN, 2), (OP_DELETE, 0), (OP_REBUILD, 0), (OP_KNN, 2)]
    for i, wanted in enumerate(expected_ops):
        index, op, reserved, arg = EVENT.unpack_from(raw, HEADER.size + i * EVENT.size)
        if index != i or reserved != b'\x00\x00\x00' or (op, arg) != wanted:
            raise ValueError(f'unexpected_event:{i}')
        events.append((int(index), int(op), int(arg)))
    return {'dimension': dim, 'base_n': base_n, 'pool_n': pool_n, 'query_n': query_n, 'k': k}, events


def read_i16(path: Path, count: int) -> list[int]:
    raw = path.read_bytes()
    if len(raw) != count * 2:
        raise ValueError(f'i16_size:{path.name}')
    return list(struct.unpack(f'<{count}h', raw))


def read_i32(path: Path, count: int) -> list[int]:
    raw = path.read_bytes()
    if len(raw) != count * 4:
        raise ValueError(f'i32_size:{path.name}')
    return list(struct.unpack(f'<{count}i', raw))


def squared_l2(pool: list[int], queries: list[int], dim: int, ident: int, query_id: int) -> int:
    return sum((pool[ident * dim + j] - queries[query_id * dim + j]) ** 2 for j in range(dim))


def exact_rows(pool: list[int], queries: list[int], dim: int, active: set[int], query_id: int, k: int) -> list[tuple[int, float]]:
    ordered = sorted((squared_l2(pool, queries, dim, ident, query_id), ident) for ident in active)
    if len(ordered) < k or (len(ordered) > k and ordered[k - 1][0] == ordered[k][0]):
        raise ValueError(f'nonunique_or_short_oracle:q{query_id}')
    return [(ident, math.sqrt(distance)) for distance, ident in ordered[:k]]


def active_sets(bundle: Path, header: dict[str, int | float], events: list[tuple[int, int, int]]) -> tuple[dict[int, set[int]], list[int], list[int]]:
    pool_n, base_n = int(header['pool_n']), int(header['base_n'])
    mapping = read_i32(bundle / 'stable_id_to_pool_row.i32', pool_n)
    initial = read_i32(bundle / 'initial_base_stable_ids.i32', base_n)
    if mapping != list(range(pool_n)) or initial != list(range(base_n)):
        raise ValueError('identity_layout')
    active = set(initial)
    by_query: dict[int, set[int]] = {}
    before_rebuild: list[int] = []
    after_rebuild: list[int] = []
    barrier = False
    for index, op, arg in events:
        if op == OP_INSERT:
            if barrier or arg in active:
                raise ValueError(f'bad_insert:{index}')
            active.add(arg)
        elif op == OP_DELETE:
            if barrier or arg not in active:
                raise ValueError(f'bad_delete:{index}')
            active.remove(arg)
            if arg < base_n:
                barrier = True
                before_rebuild = sorted(active)
        elif op == OP_KNN:
            if barrier:
                raise ValueError(f'query_before_rebuild:{index}')
            by_query[index] = set(active)
        elif op == OP_REBUILD:
            if not barrier:
                raise ValueError('rebuild_without_base_delete')
            barrier = False
            after_rebuild = sorted(active)
    if barrier or before_rebuild != after_rebuild:
        raise ValueError('rebuild_active_set_not_preserved')
    return by_query, before_rebuild, after_rebuild


def indexed(rows: list[dict[str, Any]], label: str, errors: list[str]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get('record') not in {'update', 'query', 'rebuild'}:
            continue
        op = row.get('op_index')
        if not isinstance(op, int) or op in records:
            errors.append(f'{label}_invalid_or_duplicate_op')
        else:
            records[op] = row
    return records


def expected_sidecars(dynamics: dict[int, tuple[str, int]], visited: list[int]) -> list[int]:
    leaves = set(visited)
    return sorted(ident for ident, (kind, leaf) in dynamics.items() if kind == 'direct' and leaf in leaves)


def expected_delta(dynamics: dict[int, tuple[str, int]]) -> list[int]:
    return sorted(ident for ident, (kind, _leaf) in dynamics.items() if kind == 'delta')


def validate(args: argparse.Namespace) -> dict[str, Any]:
    safe_rows = load_jsonl(args.safe_results)
    buffer_rows = load_jsonl(args.buffer_results)
    header, events = decode_trace(args.trace, args.trace_sha256)
    query_active, before_rebuild, after_rebuild = active_sets(args.bundle, header, events)
    pool = read_i16(args.bundle / 'pool.i16', int(header['pool_n']) * int(header['dimension']))
    queries = read_i16(args.bundle / 'queries.i16', int(header['query_n']) * int(header['dimension']))
    errors: list[str] = []

    for label, rows, policy in (('safe', safe_rows, 'safe_c1'), ('buffer', buffer_rows, 'buffer_only')):
        meta = rows[0]
        fail_if(errors, meta.get('record') != 'meta' or meta.get('schema') != 'fable5-matched-gpu-runner-v1', f'{label}_meta_schema')
        fail_if(errors, meta.get('policy') != policy, f'{label}_policy')
        fail_if(errors, meta.get('trace_sha256') != args.trace_sha256, f'{label}_trace_sha')
        fail_if(errors, meta.get('legacy_incremental_updater_used') is not False, f'{label}_legacy_updater')
        fail_if(errors, meta.get('rebuild_policy') != 'base_delete_immediate', f'{label}_rebuild_policy')
        fail_if(errors, 'timing' not in str(meta.get('scope', '')) or 'C2' not in str(meta.get('scope', '')) or 'C3' not in str(meta.get('scope', '')), f'{label}_scope_boundary')

    safe_ops = indexed(safe_rows, 'safe', errors)
    buffer_ops = indexed(buffer_rows, 'buffer', errors)
    expected_set = set(range(9))
    fail_if(errors, set(safe_ops) != expected_set or set(buffer_ops) != expected_set, 'nine_op_record_set')
    for op, (record, update) in enumerate(EXPECTED):
        for label, items in (('safe', safe_ops), ('buffer', buffer_ops)):
            row = items.get(op)
            if row is None:
                continue
            fail_if(errors, row.get('record') != record or (update is not None and row.get('op') != update), f'{label}_pattern:{op}')

    for op in range(9):
        s, b = safe_ops.get(op), buffer_ops.get(op)
        if s is None or b is None:
            continue
        fail_if(errors, s.get('record') != b.get('record'), f'record_kind_mismatch:{op}')
        if s.get('record') == 'update':
            fail_if(errors, s.get('op') != b.get('op') or s.get('stable_id') != b.get('stable_id'), f'update_trace_mismatch:{op}')
        elif s.get('record') == 'query':
            fail_if(errors, s.get('query_id') != b.get('query_id') or s.get('query_id') != QUERY_IDS[op], f'query_trace_mismatch:{op}')

    # Runtime route proof, not a CPU-mirror assertion.
    s0, s1, s3, s4, s6 = (safe_ops.get(i, {}) for i in (0, 1, 3, 4, 6))
    b0, b1, b3, b4, b6 = (buffer_ops.get(i, {}) for i in (0, 1, 3, 4, 6))
    leaf_a = s0.get('native_certificate_leaf')
    for op, s, b, ident in ((0, s0, b0, ROLE_IDS['A']), (1, s1, b1, ROLE_IDS['B']), (4, s4, b4, ROLE_IDS['C'])):
        cert = s.get('native_certificate_leaf')
        fail_if(errors, not isinstance(cert, int) or cert != b.get('native_certificate_leaf'), f'certificate_mismatch:{op}')
        fail_if(errors, s.get('stable_id') != ident or b.get('stable_id') != ident, f'route_id_mismatch:{op}')
        fail_if(errors, b.get('placement') != 'delta' or b.get('fallback_reason') != 'buffer_only_policy' or b.get('sidecar_leaf_id') != -1, f'buffer_route_mismatch:{op}')
    fail_if(errors, not isinstance(leaf_a, int) or leaf_a < 0, 'A_not_native_direct_certificate')
    fail_if(errors, s0.get('placement') != 'direct' or s0.get('sidecar_leaf_id') != leaf_a or s0.get('fallback_reason') != 'none', 'A_not_direct')
    fail_if(errors, s1.get('placement') != 'delta' or s1.get('fallback_reason') != 'capacity_full' or s1.get('sidecar_leaf_id') != -1 or s1.get('native_certificate_leaf') != leaf_a, 'B_not_same_leaf_capacity_delta')
    fail_if(errors, s3.get('stable_id') != ROLE_IDS['A'] or s3.get('prior_placement') != 'direct' or b3.get('prior_placement') != 'delta', 'delete_A_route')
    fail_if(errors, s4.get('placement') != 'direct' or s4.get('sidecar_leaf_id') != leaf_a or s4.get('fallback_reason') != 'none' or s4.get('native_certificate_leaf') != leaf_a, 'C_not_direct_slot_reuse')
    fail_if(errors, s6.get('stable_id') != ROLE_IDS['base_delete'] or s6.get('prior_placement') != 'base' or b6.get('prior_placement') != 'base', 'base_delete_route')

    safe_dynamic: dict[int, tuple[str, int]] = {}
    buffer_dynamic: set[int] = set()
    query_required = {
        2: {'sidecar': {ROLE_IDS['A']}, 'delta': {ROLE_IDS['B']}, 'results': {ROLE_IDS['A'], ROLE_IDS['B']}},
        5: {'sidecar': {ROLE_IDS['C']}, 'delta': {ROLE_IDS['B']}, 'results': {ROLE_IDS['C']}},
        8: {'sidecar': set(), 'delta': set(), 'results': {ROLE_IDS['C']}},
    }
    for index, op, argument in events:
        s, b = safe_ops.get(index), buffer_ops.get(index)
        if s is None or b is None:
            continue
        if op == OP_INSERT:
            if index == 0:
                safe_dynamic[argument] = ('direct', int(leaf_a) if isinstance(leaf_a, int) else -1)
            elif index == 1:
                safe_dynamic[argument] = ('delta', -1)
            elif index == 4:
                safe_dynamic[argument] = ('direct', int(leaf_a) if isinstance(leaf_a, int) else -1)
            buffer_dynamic.add(argument)
        elif op == OP_DELETE:
            if argument in safe_dynamic:
                safe_dynamic.pop(argument)
                buffer_dynamic.discard(argument)
        elif op == OP_REBUILD:
            for key in ('trigger', 'pre_rebuild_active_hash', 'post_rebuild_active_hash', 'immutable_pool_hash_before', 'immutable_pool_hash_after'):
                fail_if(errors, s.get(key) != b.get(key), f'rebuild_pair_mismatch:{key}')
            fail_if(errors, s.get('trigger') != 'base_delete_immediate', 'rebuild_trigger')
            fail_if(errors, s.get('pre_rebuild_active_hash') != s.get('post_rebuild_active_hash'), 'rebuild_active_hash_changed')
            fail_if(errors, s.get('immutable_pool_hash_before') != s.get('immutable_pool_hash_after'), 'rebuild_pool_hash_changed')
            safe_dynamic.clear()
            buffer_dynamic.clear()
        elif op == OP_KNN:
            visited_s, visited_b = as_ints(s.get('gts_visited_leaf_ids')), as_ints(b.get('gts_visited_leaf_ids'))
            side_s, side_b = as_ints(s.get('sidecar_ids')), as_ints(b.get('sidecar_ids'))
            delta_s, delta_b = as_ints(s.get('delta_ids')), as_ints(b.get('delta_ids'))
            fail_if(errors, visited_s is None or visited_b is None or visited_s != visited_b, f'visited_receipt_mismatch:{index}')
            fail_if(errors, side_s is None or side_b is None or delta_s is None or delta_b is None, f'candidate_list_malformed:{index}')
            if visited_s is None or side_s is None or side_b is None or delta_s is None or delta_b is None:
                continue
            fail_if(errors, side_s != expected_sidecars(safe_dynamic, visited_s), f'sidecar_receipt_selection:{index}')
            fail_if(errors, delta_s != expected_delta(safe_dynamic), f'safe_delta_selection:{index}')
            fail_if(errors, side_b != [], f'buffer_sidecar_nonempty:{index}')
            fail_if(errors, delta_b != sorted(buffer_dynamic), f'buffer_delta_selection:{index}')
            fail_if(errors, s.get('oracle_full_active_set_checked') is not True or b.get('oracle_full_active_set_checked') is not True, f'runner_oracle_flag:{index}')
            fail_if(errors, s.get('active_hash') != b.get('active_hash'), f'active_hash_pair:{index}')
            fail_if(errors, not equal_results(s.get('results'), b.get('results')), f'policy_result_mismatch:{index}')
            exact = exact_rows(pool, queries, int(header['dimension']), query_active[index], argument, int(header['k']))
            observed = result_rows(s.get('results'))
            fail_if(errors, observed is None or len(observed) != len(exact) or any(oid != eid or not math.isclose(od, ed, rel_tol=1e-8, abs_tol=1e-9) for (oid, od), (eid, ed) in zip(observed or [], exact)), f'independent_exact_oracle_mismatch:{index}')
            result_ids = {ident for ident, _distance in (observed or [])}
            required = query_required[index]
            fail_if(errors, not required['sidecar'].issubset(set(side_s)), f'named_sidecar_not_visible:{index}')
            fail_if(errors, not required['delta'].issubset(set(delta_s)), f'named_delta_not_visible:{index}')
            fail_if(errors, not required['results'].issubset(result_ids), f'named_exact_result_not_visible:{index}')

    return {
        'schema': 'fable5-g2-unmodified9-gpu-receipt-validator-v1',
        'status': 'PASS_FABLE5_G2_UNMODIFIED9_RECEIPT_VALIDATION' if not errors else 'FAIL_FABLE5_G2_UNMODIFIED9_RECEIPT_VALIDATION',
        'execution_boundary': {'gpu_used': False, 'cuda_binary_executed': False, 'nvml_queried': False},
        'scope': 'CPU-only post-run validation of the unmodified selected G2 L2 correctness witness; no timing/workload/performance/C2/C3/deployment interpretation.',
        'trace_sha256': args.trace_sha256,
        'safe_results': str(args.safe_results.resolve()),
        'buffer_results': str(args.buffer_results.resolve()),
        'independent_oracle_query_ops': [2, 5, 8],
        'rebuild_active_ids_match': before_rebuild == after_rebuild,
        'errors': errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--trace-sha256', required=True)
    parser.add_argument('--safe-results', type=Path, required=True)
    parser.add_argument('--buffer-results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate(args)
    except Exception as exc:
        report = {
            'schema': 'fable5-g2-unmodified9-gpu-receipt-validator-v1',
            'status': 'FAIL_FABLE5_G2_UNMODIFIED9_RECEIPT_VALIDATION',
            'execution_boundary': {'gpu_used': False, 'cuda_binary_executed': False, 'nvml_queried': False},
            'scope': 'CPU-only fail-closed post-run receipt validation; no performance interpretation.',
            'errors': [f'{type(exc).__name__}:{exc}'],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(report['status'])
    return 0 if not report['errors'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
