#!/usr/bin/env python3
"""CPU-only generator for the isolated Safe-C1 G1B capacity-two witness.

The only accepted provenance input is the frozen, READY v2 selection receipt.
It is read in place and never copied into the v3 bundle.  The generated trace
is exactly I(a),Q(a),I(b),Q(b),I(c),Q(c),D(c),D(b),D(a), where the externally
validated a/b/c share one search-native certificate leaf.  The v3 runner must
still independently prove that c reaches exact delta due to *capacity*, not a
certificate rejection.
"""
from __future__ import annotations

from array import array
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from safe_c1_v3_bundle_contract import (
    BOUNDED_TIE_DOMAIN,
    CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
    DYNAMIC_K_BOUNDARY_POLICY,
    STATIC_BASE_POLICY,
    BundleContractError,
    OP_DELETE,
    OP_INSERT,
    OP_KNN,
    TraceEvent,
    TraceHeader,
    load_current_runner_identity_stable_id_layout,
    load_i16,
    parse_trace,
    sha256,
    validate_tie_free_witness,
    write_i16,
    write_trace,
)

ROOT = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v3_capacity_witness')
EXPECTED_V2_ARTIFACT = Path(
    '/workspace/experiments/tide_safe_c1_20260728/'
    'safe_c1_dynamic_gts_v2_search_native/bundles/'
    'g1a_v2_identity_oracle_selection_20260728T110122Z/candidate_selection_v2.json'
)
EXPECTED_LEAF = 133
EXPECTED_IDS = (4330, 4357, 4399)
V2_SCHEMA = 'safe-c1-g1-v2-candidate-selection'
V2_ENGINE_SCHEMA = 'safe-c1-g1-topk-v2-search-native'
V2_CERTIFICATE_SCHEMA = 'safe-c1-search-native-sibling-boundary-v2'
V3_BUNDLE_SCHEMA = 'safe-c1-g1b-v3-capacity-witness-bundle-manifest'
V3_CONTRACT_SCHEMA = 'safe-c1-g1b-capacity-witness-contract-v3'
EXTERNAL_SELECTION_PAYLOAD_NAMES = (
    'pool.i16', 'queries.i16', 'trace.e1gtrc',
    'initial_base_stable_ids.i32', 'stable_id_to_pool_row.i32',
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleContractError(f'{label}_parse_failed:{exc}') from exc
    if not isinstance(value, dict):
        raise BundleContractError(f'{label}_not_json_object')
    return value


def _expect_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise BundleContractError(f'{label}_not_int')
    return value


def _validate_external_ready_selection(path: Path) -> tuple[dict[str, Any], tuple[int, int, int], tuple[int, int, int]]:
    """Return selected IDs and their original query IDs after strict binding checks."""
    if path.resolve() != EXPECTED_V2_ARTIFACT.resolve():
        raise BundleContractError('wrong_external_v2_selection_artifact_path')
    doc = _json_object(path, 'v2_selection_artifact')
    if doc.get('schema') != V2_SCHEMA or doc.get('status') != 'READY_FOR_V2_WITNESS':
        raise BundleContractError('external_v2_selection_not_ready_or_wrong_schema')
    if doc.get('engine_schema') != V2_ENGINE_SCHEMA or doc.get('certificate_schema') != V2_CERTIFICATE_SCHEMA:
        raise BundleContractError('external_v2_selection_engine_or_certificate_mismatch')
    if doc.get('v1_evidence_used') is not False or doc.get('gpu_used') is not True:
        raise BundleContractError('external_v2_selection_provenance_policy_mismatch')
    if doc.get('selection_trace_exactly_isolated') is not True:
        raise BundleContractError('external_v2_selection_not_isolated')
    # The READY JSON alone is insufficient provenance: bind every external
    # payload byte used to derive v3 back to the hashes recorded by that READY
    # selection artifact before copying any input into v3.
    recorded_payload_hashes = doc.get('bundle_input_files_sha256')
    if not isinstance(recorded_payload_hashes, dict):
        raise BundleContractError('external_v2_selection_missing_input_hashes')
    for name in EXTERNAL_SELECTION_PAYLOAD_NAMES:
        expected = recorded_payload_hashes.get(name)
        source = path.parent / name
        if not isinstance(expected, str) or not source.is_file() or sha256(source) != expected:
            raise BundleContractError(f'external_v2_selection_payload_sha_mismatch:{name}')
    oracle = doc.get('independent_full_active_exact_topk_oracle')
    if not isinstance(oracle, dict) or oracle.get('status') != 'PASS':
        raise BundleContractError('external_v2_selection_independent_oracle_not_pass')
    group = doc.get('g1b_same_leaf_direct_group')
    if not isinstance(group, dict):
        raise BundleContractError('external_v2_selection_missing_g1b_group')
    if _expect_int(group.get('sidecar_leaf_id'), 'group_leaf') != EXPECTED_LEAF:
        raise BundleContractError('external_v2_selection_wrong_same_leaf')
    if _expect_int(group.get('required_group_size'), 'group_size') != 3:
        raise BundleContractError('external_v2_selection_wrong_group_size')
    if group.get('stable_ids') != list(EXPECTED_IDS):
        raise BundleContractError('external_v2_selection_wrong_selected_ids')
    sequence = group.get('g1b_capacity_two_sequence')
    if not isinstance(sequence, dict) or sequence.get('leaf_capacity') != 2 or \
            sequence.get('insert_stable_ids_first') != list(EXPECTED_IDS[:2]) or \
            sequence.get('third_candidate_stable_id') != EXPECTED_IDS[2] or \
            sequence.get('result_claimed_here') is not False:
        raise BundleContractError('external_v2_selection_wrong_g1b_followup_contract')
    rows = group.get('direct_receipts')
    if not isinstance(rows, list) or len(rows) != 3:
        raise BundleContractError('external_v2_selection_wrong_direct_receipt_count')
    ids: list[int] = []
    original_query_ops: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BundleContractError(f'external_v2_selection_direct_receipt_not_object:{index}')
        stable_id = _expect_int(row.get('stable_id'), f'external_stable_id:{index}')
        leaf = _expect_int(row.get('sidecar_leaf_id'), f'external_leaf:{index}')
        query_op = _expect_int(row.get('query_op_index'), f'external_query_op:{index}')
        if stable_id != EXPECTED_IDS[index] or leaf != EXPECTED_LEAF:
            raise BundleContractError(f'external_v2_selection_direct_receipt_mismatch:{index}')
        visited = row.get('gts_visited_leaf_ids')
        if not isinstance(visited, list) or EXPECTED_LEAF not in visited:
            raise BundleContractError(f'external_v2_selection_missing_real_traversal_leaf:{index}')
        ids.append(stable_id)
        original_query_ops.append(query_op)
    return doc, tuple(ids), tuple(original_query_ops)


def _query_row(values: array, query_id: int, dimension: int) -> array:
    output = array('h')
    begin = query_id * dimension
    output.extend(values[begin:begin + dimension])
    return output


def _candidate_payload_matches(pool: array, queries: array, stable_id: int, query_id: int, dimension: int) -> bool:
    return all(pool[stable_id * dimension + axis] == queries[query_id * dimension + axis]
               for axis in range(dimension))


def _write_contract(path: Path, *, selection_sha: str, ids: tuple[int, int, int]) -> None:
    # This is a v3-local *derived constraint*, not a copied selection artifact.
    contents = '\n'.join((
        f'schema={V3_CONTRACT_SCHEMA}',
        'leaf_capacity=2',
        f'sidecar_leaf_id={EXPECTED_LEAF}',
        f'stable_id_a={ids[0]}',
        f'stable_id_b={ids[1]}',
        f'stable_id_c={ids[2]}',
        'query_id_a=0',
        'query_id_b=1',
        'query_id_c=2',
        f'v2_selection_sha256={selection_sha}',
        '',
    ))
    path.write_text(contents, encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--v2-selection', type=Path, required=True,
                        help='external read-only READY candidate_selection_v2.json')
    parser.add_argument('--out', type=Path, required=True,
                        help='new v3 bundle path under v3/bundles')
    args = parser.parse_args()
    external = args.v2_selection.resolve()
    out = args.out.resolve()
    try:
        out.relative_to((ROOT / 'bundles').resolve())
    except ValueError as exc:
        raise SystemExit('v3 bundle output must be under this v3 root\'s bundles directory') from exc
    if out.exists():
        raise SystemExit(f'refusing to overwrite v3 capacity witness bundle: {out}')

    artifact, ids, original_query_ops = _validate_external_ready_selection(external)
    source_bundle = external.parent
    source_header, source_events = parse_trace(source_bundle / 'trace.e1gtrc')
    pool = load_i16(source_bundle / 'pool.i16', source_header.pool_n * source_header.dimension)
    source_queries = load_i16(source_bundle / 'queries.i16', source_header.query_n * source_header.dimension)
    stable_to_row, initial_base_ids = load_current_runner_identity_stable_id_layout(source_bundle, source_header)
    if stable_to_row != list(range(source_header.pool_n)) or initial_base_ids != list(range(source_header.base_n)):
        raise BundleContractError('external_v2_identity_layout_recheck_failed')

    source_query_ids: list[int] = []
    for expected_id, query_op in zip(ids, original_query_ops):
        if query_op < 0 or query_op >= len(source_events):
            raise BundleContractError('external_v2_query_op_out_of_range')
        event = source_events[query_op]
        if event.op != OP_KNN:
            raise BundleContractError('external_v2_direct_receipt_query_op_not_knn')
        query_id = event.argument
        if not 0 <= query_id < source_header.query_n:
            raise BundleContractError('external_v2_query_id_out_of_range')
        if not _candidate_payload_matches(pool, source_queries, expected_id, query_id, source_header.dimension):
            raise BundleContractError('external_v2_direct_receipt_not_self_query_payload')
        source_query_ids.append(query_id)

    queries = array('h')
    for query_id in source_query_ids:
        queries.extend(_query_row(source_queries, query_id, source_header.dimension))
    events = [
        TraceEvent(0, OP_INSERT, ids[0]), TraceEvent(1, OP_KNN, 0),
        TraceEvent(2, OP_INSERT, ids[1]), TraceEvent(3, OP_KNN, 1),
        TraceEvent(4, OP_INSERT, ids[2]), TraceEvent(5, OP_KNN, 2),
        TraceEvent(6, OP_DELETE, ids[2]), TraceEvent(7, OP_DELETE, ids[1]),
        TraceEvent(8, OP_DELETE, ids[0]),
    ]
    header = TraceHeader(source_header.dimension, source_header.base_n,
                         source_header.reservoir_n, source_header.pool_n,
                         3, source_header.k, source_header.radius, len(events))
    tie_preflight = validate_tie_free_witness(pool, queries, header, events)
    selection_sha = sha256(external)

    temporary = out.with_name(out.name + '.tmp')
    if temporary.exists():
        raise SystemExit(f'refusing stale temporary v3 bundle path: {temporary}')
    temporary.mkdir(parents=True)
    try:
        write_i16(temporary / 'pool.i16', pool)
        write_i16(temporary / 'queries.i16', queries)
        write_trace(temporary / 'trace.e1gtrc', header, events)
        for name in ('initial_base_stable_ids.i32', 'stable_id_to_pool_row.i32'):
            shutil.copyfile(source_bundle / name, temporary / name)
        _write_contract(temporary / 'g1b_capacity_contract.txt', selection_sha=selection_sha, ids=ids)
        bundle_spec = {
            'schema': 'safe-c1-g1b-v3-capacity-witness-bundle-spec',
            'scope': ('strict capacity-two G1B witness only: no timing, throughput, rebuild, range, '
                      'base-delete, all-input tie, or complete-C3 claim'),
            'trace': ['I(a)', 'Q(a)', 'I(b)', 'Q(b)', 'I(c)', 'Q(c)', 'D(c)', 'D(b)', 'D(a)'],
            'leaf_capacity': 2,
            'sidecar_leaf_id_expected_from_external_v2_selection': EXPECTED_LEAF,
            'stable_ids': {'a': ids[0], 'b': ids[1], 'c': ids[2]},
            'query_source': {
                'mode': 'copied_bytes_from_external_v2_self-query rows',
                'external_v2_query_ids': source_query_ids,
                'new_v3_query_ids': [0, 1, 2],
                'self_query_payload_verified': True,
            },
            'placement_required_at_runtime': {
                'a': {'placement': 'direct', 'certified_leaf_id': EXPECTED_LEAF, 'fallback_reason': 'none'},
                'b': {'placement': 'direct', 'certified_leaf_id': EXPECTED_LEAF, 'fallback_reason': 'none'},
                'c': {'placement': 'delta', 'certified_leaf_id': EXPECTED_LEAF, 'fallback_reason': 'capacity_full'},
            },
            'stable_id_layout': CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
        }
        write_json(temporary / 'bundle_spec.json', bundle_spec)
        write_json(temporary / 'tie_free_preflight.json', tie_preflight)
        files = {file.name: sha256(file) for file in sorted(temporary.iterdir()) if file.is_file()}
        source_payload_names = EXTERNAL_SELECTION_PAYLOAD_NAMES
        manifest = {
            'schema': V3_BUNDLE_SCHEMA,
            'status': 'PREPARED_CPU_ONLY_READY_FOR_G1B_CAPACITY_WITNESS',
            'bundle_kind': 'g1b_capacity_witness',
            'gpu_used': False,
            'cuda_binary_executed': False,
            'engine_schema_required': 'safe-c1-g1b-capacity-witness-v3-search-native',
            'stable_id_layout': CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
            'scope': ('fresh CPU-prepared v3 capacity-two witness; source v2 selection artifact is external '
                      'read-only provenance only; no performance/rebuild/range/base-delete/complete-C3 claim'),
            'external_v2_selection_provenance': {
                'path': str(external), 'sha256': selection_sha, 'schema': artifact['schema'],
                'status': artifact['status'], 'copied_into_v3_bundle': False,
                'required_same_leaf_id': EXPECTED_LEAF, 'required_stable_ids': list(ids),
            },
            'external_v2_payload_source': {
                'path': str(source_bundle),
                'sha256': {name: sha256(source_bundle / name) for name in source_payload_names},
                'role': 'immutable input bytes only; no v2 run output is copied or treated as v3 result evidence',
            },
            'header': {'magic': 'E1GTRC01', 'version': 1, 'dimension': header.dimension,
                       'base_n': header.base_n, 'reservoir_n': header.reservoir_n,
                       'pool_n': header.pool_n, 'query_n': header.query_n, 'k': header.k,
                       'radius': header.radius, 'event_count': header.event_count},
            'capacity_contract': {
                'path': 'g1b_capacity_contract.txt', 'sha256': files['g1b_capacity_contract.txt'],
                'schema': V3_CONTRACT_SCHEMA, 'leaf_capacity': 2,
                'sidecar_leaf_id': EXPECTED_LEAF, 'stable_ids': list(ids),
                'runtime_requirement': 'a/b direct; c certificate-success then capacity_full exact-delta',
            },
            'tie_free_witness_domain': tie_preflight,
            'files_sha256': files,
        }
        write_json(temporary / 'manifest.json', manifest)
        os.replace(temporary, out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({'status': 'PREPARED_CPU_ONLY_READY_FOR_G1B_CAPACITY_WITNESS',
                      'gpu_used': False, 'out': str(out), 'stable_ids': list(ids)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except BundleContractError as exc:
        raise SystemExit(str(exc))
