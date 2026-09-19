#!/usr/bin/env python3
"""CPU-only regressions for the v2 certificate-selection readiness pipeline.

The tests never invoke the guarded runner, nvidia-smi, CUDA, or a CUDA binary.
They bind certificate selection to a current-runner identity stable-ID layout,
a bounded tie-policy literal shared with the C++ source, and an independent
full-active-set exact top-k replay of every synthetic selection query.
"""
from __future__ import annotations

from array import array
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from safe_c1_v2_bundle_contract import (
    OP_DELETE,
    OP_INSERT,
    OP_KNN,
    BOUNDED_TIE_ABORT,
    BOUNDED_TIE_DOMAIN,
    CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE,
    DYNAMIC_K_BOUNDARY_POLICY,
    STATIC_BASE_POLICY,
    TraceEvent,
    TraceHeader,
    load_i32_le,
    load_i16,
    parse_trace,
    validate_tie_free_witness,
    write_i16,
    write_i32_le,
    write_trace,
)


FIXTURE_VALUES = [0, 5, 12, 2, 8, 16, 20]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_manifest_file_hashes(bundle: Path) -> None:
    """Refresh only a synthetic fixture manifest after a deliberate mutation."""
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"] = {
        path.name: sha(path)
        for path in sorted(bundle.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def swap_first_two_mapping_entries(path: Path) -> None:
    values = load_i32_le(path)
    if len(values) < 2:
        raise RuntimeError("mapping fixture is too short to create a permutation")
    values[0], values[1] = values[1], values[0]
    write_i32_le(path, values)


def expected_fixture_top2(stable_id: int) -> list[list[int | float]]:
    ranked = sorted(
        ((FIXTURE_VALUES[candidate] - FIXTURE_VALUES[stable_id]) ** 2, candidate)
        for candidate in (0, 1, 2, stable_id)
    )[:2]
    return [[candidate, math.sqrt(distance)] for distance, candidate in ranked]


def make_source(work: Path) -> Path:
    source = work / "source"
    source.mkdir()
    # Base [0,5,12], reservoir [2,8,16,20], D=1, K=2. The first three
    # reservoir IDs are bounded-tie direct-group fixtures; the fourth can model
    # a certificate reject without becoming a required gate.
    pool = array("h", FIXTURE_VALUES)
    write_i16(source / "pool.i16", pool)
    write_i32_le(source / "initial_base_stable_ids.i32", [0, 1, 2])
    write_i32_le(source / "stable_id_to_pool_row.i32", list(range(len(FIXTURE_VALUES))))
    write_trace(
        source / "trace.e1gtrc",
        TraceHeader(1, 3, 4, 7, 1, 2, 0.0, 1),
        [TraceEvent(0, OP_KNN, 0)],
    )
    return source


def assert_engine_contract_literals_and_guard_order(root: Path) -> None:
    source = (root / "src/safe_c1_dynamic_gts.cu").read_text(encoding="utf-8")
    if source.count(STATIC_BASE_POLICY) != 2 or source.count(DYNAMIC_K_BOUNDARY_POLICY) != 2:
        raise RuntimeError("C++ engine bounded tie-policy literals drifted from the shared contract")
    if source.count(CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE) != 2:
        raise RuntimeError("C++ engine stable-ID layout receipt literal drifted")
    guard = source.find("require_identity_stable_id_layout(args.bundle, header)")
    tree_build = source.find("indexConstru(runtime.data_d")
    first_main_cuda = source.find("GI_CUDA(cudaMallocManaged", guard)
    if guard < 0 or tree_build < 0 or first_main_cuda < 0 or not (guard < tree_build and guard < first_main_cuda):
        raise RuntimeError("C++ runner identity-layout guard is not before tree construction and first main CUDA allocation")


def assert_nonidentity_source_layout_rejected(
    root: Path, source: Path, *, selection_generator: Path, work: Path
) -> None:
    mutated_source = work / "source_nonidentity"
    shutil.copytree(source, mutated_source)
    swap_first_two_mapping_entries(mutated_source / "stable_id_to_pool_row.i32")
    bundle = root / "bundles" / f".cpu_selection_nonidentity_source_{uuid.uuid4().hex}"
    try:
        generated = subprocess.run(
            [sys.executable, str(selection_generator), "--source-bundle", str(mutated_source),
             "--out", str(bundle), "--count", "3"], text=True, capture_output=True,
        )
        token = "stable_id_layout_not_identity_for_current_runner:stable_id_to_pool_row.i32"
        if generated.returncode == 0 or token not in (generated.stdout + generated.stderr):
            raise RuntimeError(f"nonidentity source layout was accepted: {generated.stdout} {generated.stderr}")
    finally:
        shutil.rmtree(bundle, ignore_errors=True)


def assert_preflight_rejects_identity_layout_drift(
    root: Path, source: Path, *, selection_generator: Path, preflight: Path, work: Path
) -> None:
    """Refresh hashes after a map permutation: preflight must still reject semantics."""
    bundle = root / "bundles" / f".cpu_selection_nonidentity_preflight_{uuid.uuid4().hex}"
    try:
        generated = subprocess.run(
            [sys.executable, str(selection_generator), "--source-bundle", str(source),
             "--out", str(bundle), "--count", "3"], text=True, capture_output=True,
        )
        if generated.returncode != 0:
            raise RuntimeError(f"preflight drift fixture generation failed: {generated.stderr}")
        swap_first_two_mapping_entries(bundle / "stable_id_to_pool_row.i32")
        refresh_manifest_file_hashes(bundle)
        out = work / "nonidentity_layout_preflight.json"
        check = subprocess.run(
            [sys.executable, str(preflight), "--root", str(root), "--bundle", str(bundle),
             "--mode", "certificate-selection", "--out", str(out)], text=True, capture_output=True,
        )
        document = json.loads(out.read_text())
        token = "bundle_cpu_revalidation_failed:stable_id_layout_not_identity_for_current_runner:stable_id_to_pool_row.i32"
        if check.returncode == 0 or document.get("status") != "FAIL_CPU_ONLY_CERTIFICATE_SELECTION_BUNDLE_PREFLIGHT" or \
                token not in document.get("errors", []):
            raise RuntimeError(f"preflight accepted semantic mapping drift: {check.stdout} {check.stderr} {document}")
    finally:
        shutil.rmtree(bundle, ignore_errors=True)


def assert_nonisolated_trace_rejected(
    root: Path,
    source: Path,
    *,
    selection_generator: Path,
    preflight: Path,
    work: Path,
) -> None:
    """A valid bounded-tie trace with extra events must not authorize selection."""
    token = uuid.uuid4().hex
    bundle = root / "bundles" / f".cpu_selection_fixture_nonisolated_{token}"
    try:
        generated = subprocess.run(
            [sys.executable, str(selection_generator), "--source-bundle", str(source),
             "--out", str(bundle), "--count", "3"], text=True, capture_output=True,
        )
        if generated.returncode != 0:
            raise RuntimeError(f"nonisolated: selection bundle generation failed: {generated.stderr}")
        header, events = parse_trace(bundle / "trace.e1gtrc")
        # ID 6 is a reservoir object not among the first three selected IDs.
        events.extend([
            TraceEvent(len(events), OP_INSERT, 6),
            TraceEvent(len(events) + 1, OP_DELETE, 6),
        ])
        header = TraceHeader(
            header.dimension, header.base_n, header.reservoir_n, header.pool_n,
            header.query_n, header.k, header.radius, len(events),
        )
        write_trace(bundle / "trace.e1gtrc", header, events)
        pool = load_i16(bundle / "pool.i16", header.pool_n * header.dimension)
        queries = load_i16(bundle / "queries.i16", header.query_n * header.dimension)
        tie = validate_tie_free_witness(pool, queries, header, events)
        (bundle / "tie_free_preflight.json").write_text(json.dumps(tie, indent=2, sort_keys=True) + "\n")
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["header"]["event_count"] = header.event_count
        manifest["tie_free_witness_domain"] = tie
        (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        refresh_manifest_file_hashes(bundle)
        out = work / "nonisolated_selection_preflight.json"
        check = subprocess.run(
            [sys.executable, str(preflight), "--root", str(root), "--bundle", str(bundle),
             "--mode", "certificate-selection", "--out", str(out)],
            text=True, capture_output=True,
        )
        document = json.loads(out.read_text())
        if check.returncode == 0 or document.get("status") != "FAIL_CPU_ONLY_CERTIFICATE_SELECTION_BUNDLE_PREFLIGHT" or \
                "certificate_selection_trace_not_exactly_isolated" not in document.get("errors", []):
            raise RuntimeError(f"nonisolated trace was not rejected: {check.stdout} {check.stderr} {document}")
    finally:
        shutil.rmtree(bundle, ignore_errors=True)


def execute_case(
    root: Path,
    source: Path,
    *,
    name: str,
    count: int,
    direct_count: int,
    direct_leaf_ids: list[int],
    expect_ready: bool,
    selection_generator: Path,
    preflight: Path,
    finalizer: Path,
    work: Path,
    dynamic_policy_override: str | None = None,
    wrong_topk_tail: bool = False,
    identity_layout_drift_after_preflight: bool = False,
    expected_failure_tokens: set[str] | None = None,
) -> dict:
    token = uuid.uuid4().hex
    bundle = root / "bundles" / f".cpu_selection_fixture_{name}_{token}"
    run_root = root.parent / "runs" / f".cpu_selection_fixture_{name}_{token}"
    try:
        generated = subprocess.run(
            [sys.executable, str(selection_generator), "--source-bundle", str(source),
             "--out", str(bundle), "--count", str(count)], text=True, capture_output=True,
        )
        if generated.returncode != 0:
            raise RuntimeError(f"{name}: selection bundle generation failed: {generated.stderr}")

        selection_preflight = work / f"{name}_selection_preflight.json"
        check = subprocess.run(
            [sys.executable, str(preflight), "--root", str(root), "--bundle", str(bundle),
             "--mode", "certificate-selection", "--out", str(selection_preflight)],
            text=True, capture_output=True,
        )
        document = json.loads(selection_preflight.read_text())
        if check.returncode != 0 or document.get("status") != "PASS_CPU_ONLY_CERTIFICATE_SELECTION_BUNDLE_PREFLIGHT":
            raise RuntimeError(f"{name}: selection preflight failed: {check.stdout} {check.stderr} {document}")
        if document.get("current_runner_identity_stable_id_layout_validated") is not True:
            raise RuntimeError(f"{name}: selection preflight did not validate the identity stable-ID layout")

        if identity_layout_drift_after_preflight:
            # Bypass preflight only in this synthetic finalizer regression. Refresh
            # hashes so the finalizer reaches its own semantic identity check.
            swap_first_two_mapping_entries(bundle / "stable_id_to_pool_row.i32")
            refresh_manifest_file_hashes(bundle)

        contract = json.loads((bundle / "selection_candidates.json").read_text())
        candidates = contract.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != count:
            raise RuntimeError(f"{name}: malformed generated candidates")
        if direct_count < 3 or direct_count > count or len(direct_leaf_ids) != direct_count:
            raise RuntimeError(f"{name}: invalid synthetic direct count/leaf fixture")

        run_root.mkdir(parents=True)
        fake_binary = work / f"GTS_safe_c1_g1_v2_search_native_{name}"
        fake_binary.write_bytes(f"synthetic-cpu-only-not-executed:{name}".encode())
        static = {
            "schema": "safe-c1-g1-static-contract-audit-v2-search-native-tie-free",
            "status": "PASS_CPU_ONLY_STATIC_CONTRACT",
            "gpu_used": False,
            "cuda_binary_executed": False,
            "files": {"bin/GTS_safe_c1_g1_v2_search_native": sha(fake_binary)},
        }
        static_path = run_root / "preflight/static_contract_v2.json"
        static_path.parent.mkdir()
        static_path.write_text(json.dumps(static), encoding="utf-8")

        engine_dynamic_policy = dynamic_policy_override or DYNAMIC_K_BOUNDARY_POLICY
        engine = run_root / "runner_output/engine_results.jsonl"
        engine.parent.mkdir()
        rows: list[dict] = [
            {
                "record": "meta",
                "schema": "safe-c1-g1-topk-v2-search-native",
                "gts_static_probe": "BOUNDED_TIE_WITNESS_VALIDATED",
                "stable_id_layout": {"mode": CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE, "validated": True},
                "tie_semantics": {
                    "domain": BOUNDED_TIE_DOMAIN,
                    "status": "BOUNDED_TIE_WITNESS_PRECONDITION_SATISFIED",
                    "static_base_oracle_policy": STATIC_BASE_POLICY,
                    "dynamic_k_boundary_policy": engine_dynamic_policy,
                    "on_violation": BOUNDED_TIE_ABORT,
                    "global_canonical_distance_stable_id_proof": False,
                },
            }
        ]
        for ordinal, candidate in enumerate(candidates):
            stable_id = candidate["stable_id"]
            is_direct = ordinal < direct_count
            leaf = direct_leaf_ids[ordinal] if is_direct else -1
            result_rows = expected_fixture_top2(stable_id)
            if wrong_topk_tail and ordinal == 0:
                # Query 3 has exact top-2 [3,0], but [3,1] keeps self/top-1
                # and every reported distance valid while making the full top-k wrong.
                alternate_id = 1 if result_rows[1][0] != 1 else 0
                result_rows[1] = [
                    alternate_id,
                    math.sqrt((FIXTURE_VALUES[alternate_id] - FIXTURE_VALUES[stable_id]) ** 2),
                ]
            rows.append({
                "record": "update", "op_index": candidate["insert_op_index"], "op": "insert",
                "stable_id": stable_id, "placement": "direct" if is_direct else "delta",
                "sidecar_leaf_id": leaf,
            })
            rows.append({
                "record": "query", "op_index": candidate["query_op_index"], "kind": "knn",
                "query_id": candidate["query_id"], "results": result_rows,
                "gts_visited_leaf_ids": [leaf] if is_direct else [],
                "sidecar_candidate_count": 1 if is_direct else 0,
                "delta_candidate_count": 0 if is_direct else 1,
            })
            rows.append({
                "record": "update", "op_index": candidate["delete_op_index"], "op": "delete",
                "stable_id": stable_id, "placement": "deleted", "sidecar_leaf_id": leaf,
            })
        engine.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        summary = run_root / "runner_output/engine_summary.json"
        summary.write_text(json.dumps({
            "schema": "safe-c1-g1-topk-v2-search-native",
            "status": "PASS_G1_BOUNDED_TIE_WITNESS_PENDING_INDEPENDENT_VALIDATOR",
            "stable_id_layout": {"mode": CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE, "validated": True},
            "tie_semantics": {
                "domain": BOUNDED_TIE_DOMAIN,
                "status": "BOUNDED_TIE_WITNESS_PRECONDITION_SATISFIED",
                "static_base_oracle_policy": STATIC_BASE_POLICY,
                "dynamic_k_boundary_policy": engine_dynamic_policy,
                "on_violation": BOUNDED_TIE_ABORT,
                "global_canonical_distance_stable_id_proof": False,
            },
        }), encoding="utf-8")

        finalized = subprocess.run(
            [sys.executable, str(finalizer), "--root", str(root), "--bundle", str(bundle),
             "--run-root", str(run_root), "--engine-jsonl", str(engine), "--engine-summary", str(summary),
             "--static-contract", str(static_path), "--binary", str(fake_binary),
             "--out", str(bundle / "candidate_selection_v2.json")], text=True, capture_output=True,
        )
        selection = json.loads((bundle / "candidate_selection_v2.json").read_text())
        manifest = json.loads((bundle / "manifest.json").read_text())
        if not expect_ready:
            required = expected_failure_tokens or {"insufficient_new_v2_same_leaf_direct_group"}
            if finalized.returncode == 0 or selection.get("status") != "FAIL" or \
                    not required <= set(selection.get("errors", [])) or \
                    manifest.get("status") != "PREPARED_CPU_ONLY_READY_FOR_V2_CERTIFICATE_SELECTION_RUN":
                raise RuntimeError(
                    f"{name}: finalizer failure did not preserve the pending bundle or required tokens "
                    f"{required}: {finalized.stdout} {finalized.stderr} {selection}"
                )
            return selection
        if finalized.returncode != 0:
            raise RuntimeError(f"{name}: selection finalizer failed: {finalized.stdout} {finalized.stderr}")
        group = selection.get("g1b_same_leaf_direct_group")
        oracle = selection.get("independent_full_active_exact_topk_oracle")
        if selection.get("status") != "READY_FOR_V2_WITNESS" or selection.get("v1_evidence_used") is not False or \
                len(selection.get("direct_candidates", [])) != direct_count or \
                not isinstance(group, dict) or group.get("sidecar_leaf_id") != direct_leaf_ids[0] or \
                len(group.get("stable_ids", [])) != 3 or manifest.get("status") != "READY_FOR_V2_WITNESS" or \
                not isinstance(oracle, dict) or oracle.get("status") != "PASS" or \
                oracle.get("queries_checked") != count or oracle.get("topk_id_order_mismatch_count") != 0 or \
                oracle.get("topk_distance_mismatch_count") != 0 or \
                oracle.get("query_payload_mismatch_count") != 0 or \
                oracle.get("unique_candidate_top1_mismatch_count") != 0 or \
                oracle.get("stable_id_layout", {}).get("mode") != CURRENT_RUNNER_STABLE_ID_LAYOUT_MODE:
            raise RuntimeError(f"{name}: finalizer did not bind the full-active exact top-k oracle")

        witness_preflight = work / f"{name}_witness_preflight.json"
        ready = subprocess.run(
            [sys.executable, str(preflight), "--root", str(root), "--bundle", str(bundle),
             "--mode", "witness", "--out", str(witness_preflight)], text=True, capture_output=True,
        )
        ready_doc = json.loads(witness_preflight.read_text())
        if ready.returncode != 0 or ready_doc.get("status") != "PASS_CPU_ONLY_BUNDLE_PREFLIGHT_READY_FOR_V2_GUARDED_RUN":
            raise RuntimeError(f"{name}: ready witness preflight failed: {ready.stdout} {ready.stderr} {ready_doc}")
        return selection
    finally:
        shutil.rmtree(bundle, ignore_errors=True)
        shutil.rmtree(run_root, ignore_errors=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    selection_generator = root / "tools/make_g1_v2_certificate_selection_bundle.py"
    preflight = root / "tools/preflight_g1_v2_bundle.py"
    finalizer = root / "tools/finalize_g1_v2_candidate_selection.py"
    work = Path(tempfile.mkdtemp(prefix="safe_c1_v2_selection_fixture_"))
    try:
        source = make_source(work)
        assert_engine_contract_literals_and_guard_order(root)
        assert_nonidentity_source_layout_rejected(
            root, source, selection_generator=selection_generator, work=work,
        )
        assert_preflight_rejects_identity_layout_drift(
            root, source, selection_generator=selection_generator, preflight=preflight, work=work,
        )
        assert_nonisolated_trace_rejected(
            root, source, selection_generator=selection_generator, preflight=preflight, work=work,
        )
        split_direct = execute_case(
            root, source, name="split_direct", count=3, direct_count=3,
            direct_leaf_ids=[11, 12, 13], expect_ready=False,
            selection_generator=selection_generator, preflight=preflight, finalizer=finalizer, work=work,
        )
        if "insufficient_new_v2_same_leaf_direct_group" not in split_direct.get("errors", []):
            raise RuntimeError("split direct fixture did not exercise the same-leaf hard gate")

        all_direct = execute_case(
            root, source, name="all_direct", count=3, direct_count=3,
            direct_leaf_ids=[11, 11, 11], expect_ready=True,
            selection_generator=selection_generator, preflight=preflight, finalizer=finalizer, work=work,
        )
        if all_direct.get("certificate_delta_rejection_count") != 0 or all_direct.get("delta_rejections") != []:
            raise RuntimeError("all-direct fixture did not prove certificate-delta is non-gating")

        mixed = execute_case(
            root, source, name="mixed", count=4, direct_count=3,
            direct_leaf_ids=[11, 11, 11], expect_ready=True,
            selection_generator=selection_generator, preflight=preflight, finalizer=finalizer, work=work,
        )
        if mixed.get("certificate_delta_rejection_count") != 1 or len(mixed.get("delta_rejections", [])) != 1:
            raise RuntimeError("mixed fixture did not record certificate-delta statistic")

        drift = execute_case(
            root, source, name="dynamic_policy_drift", count=3, direct_count=3,
            direct_leaf_ids=[11, 11, 11], expect_ready=False,
            dynamic_policy_override="exact squared-L2 ranks k and k+1 distinct at every trace kNN state",
            expected_failure_tokens={
                "engine_summary_tie_contract_mismatch", "engine_meta_tie_contract_mismatch",
            },
            selection_generator=selection_generator, preflight=preflight, finalizer=finalizer, work=work,
        )
        drift_oracle = drift.get("independent_full_active_exact_topk_oracle")
        if not isinstance(drift_oracle, dict) or drift_oracle.get("status") != "PASS" or \
                set(drift.get("errors", [])) != {
                    "engine_summary_tie_contract_mismatch", "engine_meta_tie_contract_mismatch",
                }:
            raise RuntimeError("dynamic tie-policy drift fixture did not isolate the two metadata contract errors")

        topk_wrong = execute_case(
            root, source, name="topk_wrong_tail", count=3, direct_count=3,
            direct_leaf_ids=[11, 11, 11], expect_ready=False, wrong_topk_tail=True,
            expected_failure_tokens={"full_active_exact_topk_id_order_mismatch:count=1"},
            selection_generator=selection_generator, preflight=preflight, finalizer=finalizer, work=work,
        )
        if "candidate_not_top1:0" in topk_wrong.get("errors", []):
            raise RuntimeError("top-k tail fixture did not preserve its correct top-1 receipt")

        nonidentity_finalizer = execute_case(
            root, source, name="nonidentity_finalizer", count=3, direct_count=3,
            direct_leaf_ids=[11, 11, 11], expect_ready=False,
            identity_layout_drift_after_preflight=True,
            expected_failure_tokens={
                "oracle_bundle_or_identity_layout_parse_failed:stable_id_layout_not_identity_for_current_runner:stable_id_to_pool_row.i32",
            },
            selection_generator=selection_generator, preflight=preflight, finalizer=finalizer, work=work,
        )
        if nonidentity_finalizer.get("independent_full_active_exact_topk_oracle", {}).get("status") != "FAIL":
            raise RuntimeError("finalizer did not fail closed on a post-preflight nonidentity layout")

        print("PASS v2_certificate_selection_bundle_preflight")
        print("PASS v2_certificate_selection_rejects_nonisolated_trace")
        print("PASS v2_candidate_selection_rejects_split_direct_leaves")
        print("PASS v2_candidate_selection_same_leaf_direct_triple_hard_gate")
        print("PASS v2_candidate_selection_all_direct_without_delta_gate")
        print("PASS v2_candidate_selection_records_delta_statistic_only")
        print("PASS v2_engine_tie_policy_literal_contract")
        print("PASS v2_identity_stable_id_layout_generator_and_preflight_guard")
        print("PASS v2_candidate_selection_rejects_nonidentity_layout_after_preflight")
        print("PASS v2_candidate_selection_rejects_dynamic_policy_drift")
        print("PASS v2_candidate_selection_rejects_top1_correct_but_topk_wrong")
        print("PASS safe_c1_v2_certificate_selection_pipeline CPU-only")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
