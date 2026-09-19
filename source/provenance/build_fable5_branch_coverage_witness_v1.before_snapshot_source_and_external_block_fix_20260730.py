#!/usr/bin/env python3
"""Build a clearly labeled CPU-only Safe-C1 certificate-reject micro-witness.

This tool creates an isolated *branch-coverage input*, not a workload or a
performance dataset. It keeps the 4096-object base and all queries byte-identical
and replaces exactly one unused reservoir row with an integer vector whose
root-pivot radius equals a real frozen E0 sibling boundary. Under the CPU mirror,
that vector fails the strict certificate but remains query-visible. A fresh native
E0 export is still mandatory before any runtime trace or GPU claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path('/workspace/experiments/tide_safe_c1_20260730/safe_c1_fable5_matched_v1')
SELECTOR_PATH = ROOT / 'tools' / 'select_fable5_matched_trace_from_native_snapshot_v1.py'
EXPECTED = {'dimension': 32, 'base_n': 4096, 'reservoir_n': 2048, 'pool_n': 6144, 'query_n': 3, 'k': 10, 'radius': 300.0}
REPLACEMENT_ID = 6143
ROOT_BOUNDARY_SLOT = 1
QUERY_ANCHOR = 2
TAIL_START = 28
WITNESS_NAME = 'fable5_branch_coverage_witness_v1.json'


class Blocked(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Blocked(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_selector():
    spec = importlib.util.spec_from_file_location('fable5_selector_for_branch_witness', SELECTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nearest_four_square(total: int, desired: list[int]) -> list[int]:
    """Return a closest signed four-square representation of ``total``.

    The first 28 scaled coordinates are fixed. Enumerating the final four is
    deterministic and small here (sqrt(total) is about 150), and makes the
    root-pivot squared distance exactly equal to the real base-boundary value.
    """
    require(total >= 0 and len(desired) == 4, 'invalid_four_square_request')
    maximum = math.isqrt(total)
    best_tail: dict[int, tuple[int, int, int]] = {}
    for third in range(-maximum, maximum + 1):
        third_sq = third * third
        for fourth in range(-maximum, maximum + 1):
            partial = third_sq + fourth * fourth
            if partial > total:
                continue
            score = (third - desired[2]) ** 2 + (fourth - desired[3]) ** 2
            previous = best_tail.get(partial)
            if previous is None or score < previous[0]:
                best_tail[partial] = (score, third, fourth)
    best: tuple[int, list[int]] | None = None
    for first in range(-maximum, maximum + 1):
        first_sq = first * first
        for second in range(-maximum, maximum + 1):
            partial = first_sq + second * second
            tail = best_tail.get(total - partial)
            if tail is None:
                continue
            score = (first - desired[0]) ** 2 + (second - desired[1]) ** 2 + tail[0]
            candidate = (score, [first, second, tail[1], tail[2]])
            if best is None or candidate < best:
                best = candidate
    require(best is not None, 'four_square_solver_failed')
    return best[1]


def pool_row(pool: list[int], dimension: int, ident: int) -> list[int]:
    offset = ident * dimension
    return pool[offset:offset + dimension]


def make_vector(selector, pool: list[int], queries: list[int], header: dict[str, Any], tree: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    dimension = header['dimension']
    root_first_child = selector.child_id(0, 0)
    root_boundary_child = selector.child_id(0, ROOT_BOUNDARY_SLOT)
    root_pivot = tree['nodes'][root_first_child]['pid']
    require(root_pivot >= 0, 'invalid_root_pivot')
    boundary = tree['nodes'][root_boundary_child]['min']
    pivot_row = pool_row(pool, dimension, root_pivot)

    boundary_witnesses: list[int] = []
    for ident in range(header['base_n']):
        if selector.host_l2(pool, dimension, root_pivot, ident) == boundary:
            boundary_witnesses.append(ident)
    require(boundary_witnesses, 'no_exact_base_witness_for_selected_root_boundary')
    boundary_witness = min(boundary_witnesses)
    target_squared = sum((pivot_row[axis] - pool_row(pool, dimension, boundary_witness)[axis]) ** 2 for axis in range(dimension))
    require(selector.f32(math.sqrt(target_squared)) == boundary, 'integer_boundary_radius_does_not_match_native_f32_boundary')

    query = pool_row(queries, dimension, QUERY_ANCHOR)
    direction = [query[axis] - pivot_row[axis] for axis in range(dimension)]
    direction_sq = sum(value * value for value in direction)
    require(direction_sq > 0, 'query_equals_root_pivot')
    scale = math.sqrt(target_squared / direction_sq)
    scaled = [round(scale * value) for value in direction]
    residual = target_squared - sum(value * value for value in scaled[:TAIL_START])
    tail = nearest_four_square(residual, scaled[TAIL_START:])
    offset = scaled[:TAIL_START] + tail
    vector = [pivot_row[axis] + offset[axis] for axis in range(dimension)]
    require(all(-32768 <= value <= 32767 for value in vector), 'constructed_vector_outside_i16')
    require(sum((vector[axis] - pivot_row[axis]) ** 2 for axis in range(dimension)) == target_squared, 'constructed_vector_radius_not_exact')
    return vector, {
        'root_pivot_stable_id': root_pivot,
        'root_boundary_slot': ROOT_BOUNDARY_SLOT,
        'root_boundary_min_f32': boundary,
        'boundary_witness_base_id': boundary_witness,
        'root_boundary_squared_distance': target_squared,
        'query_anchor': QUERY_ANCHOR,
        'query_anchor_squared_distance': sum((vector[axis] - query[axis]) ** 2 for axis in range(dimension)),
        'construction_tail_start': TAIL_START,
    }


def copy_and_mutate(source: Path, target: Path, pool: list[int], vector: list[int], header: dict[str, Any]) -> None:
    stage = Path(tempfile.mkdtemp(prefix=f'.{target.name}.tmp-', dir=str(target.parent)))
    try:
        for filename in ('trace.g2trc', 'queries.i16', 'initial_base_stable_ids.i32', 'stable_id_to_pool_row.i32'):
            shutil.copyfile(source / filename, stage / filename)
        raw = bytearray((source / 'pool.i16').read_bytes())
        offset = REPLACEMENT_ID * header['dimension'] * 2
        import struct
        raw[offset:offset + header['dimension'] * 2] = struct.pack(f'<{header["dimension"]}h', *vector)
        (stage / 'pool.i16').write_bytes(raw)
        os.replace(stage, target)
        stage = None  # type: ignore[assignment]
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bundle', type=Path, required=True)
    parser.add_argument('--native-snapshot', type=Path, required=True)
    parser.add_argument('--out-bundle', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, Any] = {
        'schema': 'fable5-branch-coverage-witness-build-v1',
        'execution_boundary': {'cpu_only': True, 'cuda_binary_executed': False, 'gpu_used': False, 'nvml_used': False},
        'scope': 'Constructed adversarial branch-coverage input only; not a real workload, performance result, native runtime receipt, or paper claim.',
    }
    try:
        selector = load_selector()
        source = args.source_bundle.resolve(strict=True)
        target = args.out_bundle.resolve(strict=False)
        require(source.parent == (ROOT / 'bundles').resolve(strict=True), 'source_bundle_must_be_direct_child_of_root_bundles')
        require(target.parent == (ROOT / 'bundles').resolve(strict=True) and not target.exists() and target.name and not target.name.startswith('.'), 'out_bundle_must_be_new_direct_child_of_root_bundles')
        require(not args.report.exists(), 'report_path_already_exists')
        header = selector.read_header(source / 'trace.g2trc')
        require({key: header[key] for key in EXPECTED} == EXPECTED, 'unexpected_g2_header')
        initial = selector.read_i32(source / 'initial_base_stable_ids.i32', header['base_n'])
        mapping = selector.read_i32(source / 'stable_id_to_pool_row.i32', header['pool_n'])
        require(initial == list(range(header['base_n'])) and mapping == list(range(header['pool_n'])), 'identity_layout_failure')
    except Blocked as error:
        report.update({'status': 'BLOCKED_NO_BRANCH_COVERAGE_BUNDLE_WRITTEN', 'blocker': str(error)})
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(report['status'], error)
        return 2

    # The actual construction is kept below in a second guarded block so all
    # inputs/layout checks above occur before the target bundle is created.
    try:
        # Re-read after the explicit identity gate, retaining simple types.
        pool = selector.read_i16(source / 'pool.i16', header['pool_n'] * header['dimension'])
        queries = selector.read_i16(source / 'queries.i16', header['query_n'] * header['dimension'])
        snapshot, observed = selector.parse_snapshot(
            args.native_snapshot, source, source / 'trace.g2trc',
            ROOT / 'src/safe_c1_dynamic_gts.cu', SELECTOR_PATH,
            ROOT / 'src/fable5_matched_gpu_runner_v1.cu',
        )
        tree = selector.normalize_snapshot(snapshot, header, initial)
        vector, design = make_vector(selector, pool, queries, header, tree)
        original_vector = pool_row(pool, header['dimension'], REPLACEMENT_ID)
        mutated = pool[:]
        mutated[REPLACEMENT_ID * header['dimension']:(REPLACEMENT_ID + 1) * header['dimension']] = vector
        certificate = selector.certify(tree, mutated, header['dimension'], REPLACEMENT_ID)
        require(certificate['leaf'] == -1 and str(certificate['reason']).startswith('no_strict_child:0:'), 'cpu_mirror_did_not_reject_at_root')
        ranker = selector.PrefixRanker(set(initial), mutated, queries, header['dimension'], header['k'])
        visible, reason, topk = ranker.dynamic((REPLACEMENT_ID,), QUERY_ANCHOR)
        require(visible and REPLACEMENT_ID in topk, 'constructed_reject_not_query_visible_under_cpu_exact_merge')
        contract = selector.load_preflight_contract(source, header)
        search: dict[str, Any] = {}
        slots, trace = selector.candidate_selection(tree, header, mutated, queries, initial, 1, 'auto', contract, search)
        require(slots['R'] == REPLACEMENT_ID and len(trace) == 192, 'cpu_mirror_12op_plan_not_available')
        copy_and_mutate(source, target, pool, vector, header)
        source_pool = (source / 'pool.i16').read_bytes()
        target_pool = (target / 'pool.i16').read_bytes()
        offset = REPLACEMENT_ID * header['dimension'] * 2
        require(source_pool[:offset] == target_pool[:offset] and source_pool[offset + header['dimension'] * 2:] == target_pool[offset + header['dimension'] * 2:], 'mutation_changed_more_than_one_reservoir_row')
        witness = {
            'schema': 'fable5-branch-coverage-witness-v1',
            'status': 'CPU_CONSTRUCTED_BRANCH_COVERAGE_INPUT_PENDING_NATIVE_SNAPSHOT',
            'scope': 'One constructed certificate-reject branch-coverage row; not a workload/performance dataset and not a native runtime correctness result.',
            'source_bundle': str(source),
            'source_snapshot_for_planning_only': {'path': str(args.native_snapshot), 'sha256': sha256(args.native_snapshot)},
            'input_header': header,
            'base_and_queries_unchanged': True,
            'replacement': {
                'stable_id': REPLACEMENT_ID,
                'row': REPLACEMENT_ID,
                'original_i16': original_vector,
                'constructed_i16': vector,
                **design,
            },
            'cpu_mirror_planning_only': {
                'certificate': {'leaf': certificate['leaf'], 'reason': certificate['reason']},
                'query_visible': {'query_id': QUERY_ANCHOR, 'topk': topk},
                'planned_slots': slots,
                'selector_search': search,
                'static_certificate_outcomes': 'NOT_ESTABLISHED_FOR_NEW_INPUT_UNTIL_FRESH_NATIVE_SNAPSHOT_AND_RUNTIME_RECEIPT',
            },
            'source_hashes': observed,
            'output_file_sha256': {name: sha256(target / name) for name in ('trace.g2trc', 'pool.i16', 'queries.i16', 'initial_base_stable_ids.i32', 'stable_id_to_pool_row.i32')},
            'builder_source': {'relative_path': 'tools/build_fable5_branch_coverage_witness_v1.py', 'sha256': sha256(Path(__file__).resolve())},
        }
        (target / WITNESS_NAME).write_text(json.dumps(witness, indent=2, sort_keys=True) + '\n')
        report.update({
            'status': 'PASS_CPU_CONSTRUCTED_BRANCH_COVERAGE_INPUT_PENDING_NATIVE_SNAPSHOT',
            'output_bundle': str(target),
            'witness': str(target / WITNESS_NAME),
            'witness_sha256': sha256(target / WITNESS_NAME),
            'replacement_id': REPLACEMENT_ID,
            'planned_slots_cpu_mirror_only': slots,
            'selector_search': search,
        })
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(report['status'], target)
        return 0
    except Blocked as error:
        if target.exists():
            raise RuntimeError('construction failed after target creation; preserve for manual audit') from error
        report.update({'status': 'BLOCKED_NO_BRANCH_COVERAGE_BUNDLE_WRITTEN', 'blocker': str(error)})
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(report['status'], error)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
