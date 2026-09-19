#!/usr/bin/env python3
"""Create one fresh CPU-prepared G2 stable-ID seeded rebuild witness bundle.

The v3 G1B PASS receipt and v2 selection are external read-only bindings.  This
script copies only immutable input bytes into a new v4 bundle; it never copies a
v2/v3 engine result as v4 evidence and never invokes CUDA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from safe_c1_v4_rebuild_bundle_contract import (
    BOUNDED_TIE_ABORT,
    CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
    DYNAMIC_POLICY,
    OP_DELETE,
    OP_INSERT,
    OP_KNN,
    OP_REBUILD,
    SCHEMA,
    STATIC_POLICY,
    TraceEvent,
    TraceHeader,
    cint,
    load_i16,
    load_identity_layout,
    parse_trace as parse_g2_trace,
    read_contract,
    require_dynamic_k_boundary,
    require_static_epoch_tie_free,
    sha256,
    topk_exact,
    validate_strict_g2_trace,
    write_trace,
)

ROOT = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v4_rebuild_witness')
V3_ROOT = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v3_capacity_witness')
V3_BUNDLE = V3_ROOT / 'bundles/g1b_capacity_witness_20260728T114307Z_cpu'
V3_RESULT = V3_BUNDLE / 'g1b_capacity_witness_v3.json'
V2_ROOT = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v2_search_native')
V2_SELECTION = V2_ROOT / 'bundles/g1a_v2_identity_oracle_selection_20260728T110122Z/candidate_selection_v2.json'


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        fail(f'not_json_object:{path}')
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def assert_canonical_root(root: Path) -> None:
    if root.resolve() != ROOT.resolve():
        fail('root_must_equal_canonical_v4_root')


def verify_external_bindings() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    g1b = read_json(V3_RESULT)
    selection = read_json(V2_SELECTION)
    v3_manifest = read_json(V3_BUNDLE / 'manifest.json')
    if g1b.get('schema') != 'safe-c1-g1b-v3-capacity-witness-result' or \
            g1b.get('status') != 'PASS_G1B_CAPACITY_WITNESS' or g1b.get('error_count') != 0:
        fail('external_g1b_pass_artifact_not_pass')
    external = g1b.get('external_v2_selection_provenance')
    if not isinstance(external, dict) or external.get('sha256') != sha256(V2_SELECTION):
        fail('external_g1b_result_not_bound_to_v2_selection')
    group = selection.get('g1b_same_leaf_direct_group')
    if not isinstance(group, dict):
        fail('v2_selection_missing_same_leaf_direct_group')
    ids = group.get('stable_ids')
    leaf = group.get('sidecar_leaf_id')
    if ids != [4330, 4357, 4399] or leaf != 133:
        fail('external_selection_group_not_expected_final_v3_group')
    if external.get('required_stable_ids') != ids or external.get('required_same_leaf_id') != leaf:
        fail('v3_pass_group_disagrees_with_v2_selection')
    capacity = v3_manifest.get('capacity_contract')
    if not isinstance(capacity, dict) or capacity.get('stable_ids') != ids or capacity.get('sidecar_leaf_id') != leaf:
        fail('v3_bundle_capacity_contract_disagrees')
    files = ('pool.i16', 'queries.i16', 'stable_id_to_pool_row.i32', 'initial_base_stable_ids.i32')
    source_hashes = {name: sha256(V3_BUNDLE / name) for name in files}
    manifest_hashes = v3_manifest.get('files_sha256')
    if not isinstance(manifest_hashes, dict) or any(manifest_hashes.get(name) != source_hashes[name] for name in files):
        fail('v3_immutable_payload_hash_mismatch')
    return g1b, selection, v3_manifest, source_hashes


def choose_base_delete(bundle: Path, g1b_contract: dict[str, str]) -> tuple[int, dict[str, Any]]:
    # Reads only v4-local immutable bytes after they were copied, so this
    # selection can be fully reproduced from the new bundle.
    # The v3 trace has a distinct ABI magic; the generator persisted the copied
    # population fields in source_v3_header.json before calling this selector.
    # all population fields/payloads are copied, while v4 writes a new trace.
    import struct
    raw = (bundle / 'source_v3_header.json').read_text(encoding='utf-8')
    h = json.loads(raw)
    header = TraceHeader(**h)
    pool = load_i16(bundle / 'pool.i16', header.pool_n * header.dimension)
    queries = load_i16(bundle / 'queries.i16', header.query_n * header.dimension)
    _mapping, initial = load_identity_layout(bundle, header)
    a = int(g1b_contract['stable_id_a']); b = int(g1b_contract['stable_id_b']); c = int(g1b_contract['stable_id_c'])
    qa = int(g1b_contract['query_id_a']); qc = int(g1b_contract['query_id_c'])
    all_queries = list(range(header.query_n))
    initial_set = set(initial)
    # E0 static gate does not depend on the deletion choice; still execute it
    # here so bundle generation aborts before producing a misleading contract.
    static_e0 = require_static_epoch_tie_free(pool, queries, initial_set, header, all_queries, 'E0')
    failures: list[str] = []
    for base_delete in initial:
        e1 = (initial_set - {base_delete}) | {b, c}
        try:
            static_e1 = require_static_epoch_tie_free(pool, queries, e1, header, all_queries, 'E1')
            active_before = initial_set | {a, b}
            active_middle = initial_set | {b, c}
            require_dynamic_k_boundary(pool, queries, active_before, header, qa, 2)
            require_dynamic_k_boundary(pool, queries, active_middle, header, qc, 5)
            require_dynamic_k_boundary(pool, queries, e1, header, qc, 8)
            if topk_exact(pool, queries, active_before, header, qa)[0][1] != a:
                raise RuntimeError('a_not_unique_top1_before')
            if topk_exact(pool, queries, active_middle, header, qc)[0][1] != c:
                raise RuntimeError('c_not_unique_top1_middle')
            if topk_exact(pool, queries, e1, header, qc)[0][1] != c:
                raise RuntimeError('c_not_unique_top1_after')
            return base_delete, {'static_e0': static_e0, 'static_e1': static_e1}
        except Exception as exc:  # only diagnostic; candidate scan is deterministic
            if len(failures) < 12:
                failures.append(f'{base_delete}:{exc}')
    fail('no_base_delete_id_satisfies_g2_tie_and_self_top1_contract:' + ';'.join(failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(); out = args.out.resolve()
    assert_canonical_root(root)
    try:
        out.relative_to((root / 'bundles').resolve())
    except ValueError as exc:
        raise SystemExit('out_must_be_under_new_v4_bundles_root') from exc
    if out.exists():
        raise SystemExit(f'out_must_be_fresh:{out}')
    g1b, selection, _v3_manifest, source_hashes = verify_external_bindings()
    out.mkdir(parents=True, exist_ok=False)
    try:
        # Only immutable input payload bytes are copied.  v3/v2 result artifacts
        # remain external read-only provenance and are never copied as v4 evidence.
        for name in ('pool.i16', 'queries.i16', 'stable_id_to_pool_row.i32', 'initial_base_stable_ids.i32'):
            shutil.copyfile(V3_BUNDLE / name, out / name)
        # Need the original v3 header fields to create v4's new trace.  Parse
        # the known packed ABI without importing a v3 module into v4.
        import struct
        raw = (V3_BUNDLE / 'trace.e1gtrc').read_bytes()
        old_header = struct.Struct('<8sI6IfQ').unpack_from(raw, 0)
        _magic, _version, dimension, base_n, reservoir_n, pool_n, query_n, k, radius, _event_count = old_header
        header = TraceHeader(dimension, base_n, reservoir_n, pool_n, query_n, k, radius, 9)
        write_json(out / 'source_v3_header.json', {
            'dimension': header.dimension, 'base_n': header.base_n, 'reservoir_n': header.reservoir_n,
            'pool_n': header.pool_n, 'query_n': header.query_n, 'k': header.k,
            'radius': header.radius, 'event_count': header.event_count,
        })
        _mapping, initial = load_identity_layout(out, header)
        # Derive fields from the actual externally passed v3 G1B contract.
        values = {}
        for raw_line in (V3_BUNDLE / 'g1b_capacity_contract.txt').read_text(encoding='utf-8').splitlines():
            if '=' in raw_line:
                key, value = raw_line.strip().split('=', 1); values[key] = value
        expected = ('stable_id_a', 'stable_id_b', 'stable_id_c', 'query_id_a', 'query_id_c', 'sidecar_leaf_id')
        if any(key not in values for key in expected):
            fail('external_v3_capacity_contract_missing_required_field')
        # Seed temporary contract-independent header file before deterministic
        # base-delete selection.  choose_base_delete reads these new bytes only.
        base_delete, tie_data = choose_base_delete(out, values)
        contract_lines = [
            f'schema={SCHEMA}', 'leaf_capacity=1', f'sidecar_leaf_id={values["sidecar_leaf_id"]}',
            f'stable_id_a={values["stable_id_a"]}', f'stable_id_b={values["stable_id_b"]}',
            f'stable_id_c={values["stable_id_c"]}', f'query_id_before={values["query_id_a"]}',
            f'query_id_middle={values["query_id_c"]}', f'query_id_after={values["query_id_c"]}',
            f'base_delete_id={base_delete}', f'v3_g1b_result_sha256={sha256(V3_RESULT)}',
            f'v2_selection_sha256={sha256(V2_SELECTION)}',
        ]
        (out / 'g2_rebuild_contract.txt').write_text('\n'.join(contract_lines) + '\n', encoding='utf-8')
        contract = read_contract(out / 'g2_rebuild_contract.txt')
        events = [
            TraceEvent(0, OP_INSERT, cint(contract, 'stable_id_a')),
            TraceEvent(1, OP_INSERT, cint(contract, 'stable_id_b')),
            TraceEvent(2, OP_KNN, cint(contract, 'query_id_before')),
            TraceEvent(3, OP_DELETE, cint(contract, 'stable_id_a')),
            TraceEvent(4, OP_INSERT, cint(contract, 'stable_id_c')),
            TraceEvent(5, OP_KNN, cint(contract, 'query_id_middle')),
            TraceEvent(6, OP_DELETE, cint(contract, 'base_delete_id')),
            TraceEvent(7, OP_REBUILD, 0),
            TraceEvent(8, OP_KNN, cint(contract, 'query_id_after')),
        ]
        write_trace(out / 'trace.g2trc', header, events)
        parsed_header, parsed_events = parse_g2_trace(out / 'trace.g2trc')
        validate_strict_g2_trace(parsed_header, parsed_events, contract)
        # G2 uses the original base prefix only for epoch E0.  E1 is the
        # non-contiguous stable-ID seed set computed by the contract.
        spec = {
            'schema': 'safe-c1-g2-rebuild-witness-v4-bundle-spec',
            'scope': ('strict one-rebuild stable-ID witness: seeded full immutable pool; direct + capacity delta + '
                      'direct-slot reuse + base delete + immediate rebuild + post-rebuild query; no performance or general rebuild claim'),
            'trace': 'trace.g2trc', 'contract': 'g2_rebuild_contract.txt',
            'stable_id_layout': CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
            'static_base_policy': STATIC_POLICY,
            'dynamic_k_boundary_policy': DYNAMIC_POLICY,
            'base_delete_selection': {'stable_id': base_delete, 'method': 'first ascending base ID satisfying E0/E1 bounded tie and self-top1 gates'},
            'v3_g1b_external_read_only': str(V3_RESULT),
            'v2_selection_external_read_only': str(V2_SELECTION),
        }
        write_json(out / 'bundle_spec.json', spec)
        files = ('pool.i16', 'queries.i16', 'stable_id_to_pool_row.i32', 'initial_base_stable_ids.i32',
                 'source_v3_header.json', 'g2_rebuild_contract.txt', 'trace.g2trc', 'bundle_spec.json')
        manifest = {
            'schema': 'safe-c1-g2-rebuild-witness-v4-bundle-manifest',
            'status': 'PREPARED_CPU_ONLY_G2_REBUILD_WITNESS',
            'bundle_kind': 'g2_seeded_stable_id_rebuild_witness',
            'gpu_used': False, 'cuda_binary_executed': False,
            'engine_schema_required': 'safe-c1-g2-rebuild-witness-v4-search-native',
            'scope': spec['scope'], 'stable_id_layout': CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
            'header': {'magic': 'E1GTRC02', 'version': 2, **{key: getattr(header, key) for key in ('dimension','base_n','reservoir_n','pool_n','query_n','k','radius','event_count')}},
            'contract': {'path': 'g2_rebuild_contract.txt', 'sha256': sha256(out / 'g2_rebuild_contract.txt'),
                         'leaf_capacity': 1, 'base_delete_id': base_delete},
            'external_g1b_pass_artifact': {
                'path': str(V3_RESULT), 'sha256': sha256(V3_RESULT), 'status': g1b.get('status'),
                'copied_into_v4_bundle': False,
                'role': 'read-only proof that a/b/c have the required real GTS same-leaf capacity provenance'},
            'external_v2_selection': {
                'path': str(V2_SELECTION), 'sha256': sha256(V2_SELECTION), 'status': selection.get('status'),
                'copied_into_v4_bundle': False,
                'role': 'read-only selection provenance only; never a v4 result'},
            'external_v3_immutable_payload_source': {
                'path': str(V3_BUNDLE), 'role': 'immutable input bytes only; no v3 runner output copied as v4 evidence',
                'sha256': source_hashes},
            'files_sha256': {name: sha256(out / name) for name in files},
            'cpu_generation': {'static_epoch_checks': tie_data, 'bounded_tie_abort': BOUNDED_TIE_ABORT},
        }
        write_json(out / 'manifest.json', manifest)
        print(f'PREPARED_CPU_ONLY_G2_REBUILD_BUNDLE out={out}')
        return 0
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
