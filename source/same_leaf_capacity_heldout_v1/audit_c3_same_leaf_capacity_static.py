#!/workspace/legacy_workspace/GTS/bench_env/bin/python
"""CPU-only source/build/guard contract audit for C3 same-leaf capacity probe.

It never invokes the CUDA runner, nvidia-smi, or a GPU.  PASS only means the
pre-registered source/build/launcher wiring satisfies the narrow static
contract; runtime evidence still requires the guarded runner and independent
CPU validator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.MULTILINE | re.DOTALL) is not None


def function_body(text: str, signature_pattern: str) -> str | None:
    match = re.search(signature_pattern, text, re.MULTILINE)
    if match is None:
        return None
    start = text.find("{", match.end())
    if start < 0:
        return None
    level = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            level += 1
        elif text[pos] == "}":
            level -= 1
            if level == 0:
                return text[start:pos + 1]
    return None


def line_number(text: str, needle: str) -> int | None:
    at = text.find(needle)
    return None if at < 0 else text.count("\n", 0, at) + 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--selection", required=True, type=Path)
    p.add_argument("--runner", required=True, type=Path)
    p.add_argument("--validator", required=True, type=Path)
    p.add_argument("--tree-header", required=True, type=Path)
    p.add_argument("--search-header", required=True, type=Path)
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--build-manifest", required=True, type=Path)
    p.add_argument("--launcher", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    errors: list[dict[str, Any]] = []
    protocol = json.loads(a.protocol.read_text())
    selection = json.loads(a.selection.read_text())
    build = json.loads(a.build_manifest.read_text())
    runner = a.runner.read_text()
    validator = a.validator.read_text()
    tree = a.tree_header.read_text()
    search = a.search_header.read_text()
    launcher = a.launcher.read_text()

    def check(ok: bool, tag: str, **extra: Any) -> None:
        if not ok:
            errors.append({"check": tag, **extra})

    # Immutable protocol/selection/build binding.
    check(protocol.get("schema") == "c3-same-leaf-capacity-heldout-protocol-v1", "protocol_schema")
    check(protocol.get("status") == "PRE_REGISTERED_AFTER_CPU_SELECTION_AND_COMPILE_ONLY_BEFORE_GPU_EXECUTION", "protocol_preregistration")
    check(protocol.get("candidate_trace", {}).get("target_leaf") == 221, "protocol_target_leaf")
    check(protocol.get("candidate_trace", {}).get("initial_target_occupancy") == 4, "protocol_initial_occupancy")
    check(protocol.get("candidate_trace", {}).get("accepted_count") == 16, "protocol_accept_count")
    check(protocol.get("candidate_trace", {}).get("boundary_stable_id") == 5430, "protocol_boundary_id")
    check(protocol.get("candidate_trace", {}).get("boundary_required_outcome", {}).get("reason") == "static_scan_limit", "protocol_boundary_reason")
    check(protocol.get("heldout_nonself_knn", {}).get("query_ids") == [0, 1, 2], "protocol_heldout_queries")
    check(protocol.get("heldout_nonself_knn", {}).get("k_values") == [1, 10, 20], "protocol_heldout_ks")
    check(protocol.get("selection_witness", {}).get("path") == str(a.selection.resolve()), "protocol_selection_path")
    check(protocol.get("selection_witness", {}).get("sha256") == sha(a.selection.resolve()), "protocol_selection_hash")
    check(protocol.get("selection_witness", {}).get("full_structure_fingerprint") == "fnv1a64:a4a1881cd82237fb", "protocol_fingerprint")
    check(selection.get("schema") == "c3-same-leaf-capacity-selection-v1", "selection_schema")
    check(selection.get("status") == "CPU_ONLY_SELECTION_PASS_NO_CUDA", "selection_cpu_only")
    check(selection.get("selection_geometry_structural_fingerprint") == "fnv1a64:a4a1881cd82237fb", "selection_fingerprint")
    check(selection.get("candidate_selection_rule", {}).get("target_leaf") == 221, "selection_target")
    check(selection.get("candidate_selection_rule", {}).get("accepted_stable_ids") == protocol.get("candidate_trace", {}).get("accepted_stable_ids"), "selection_accepted_trace")
    check(selection.get("candidate_selection_rule", {}).get("boundary_stable_id") == 5430, "selection_boundary")
    check(selection.get("heldout_nonself_query_rule", {}).get("query_ids") == [0, 1, 2], "selection_heldout_queries")

    check(build.get("schema") == "c3-same-leaf-capacity-compile-manifest-v1", "build_manifest_schema")
    check(build.get("status") == "COMPILE_ONLY_SUCCESS_NO_CUDA_BINARY_EXECUTION", "build_manifest_compile_only")
    check(build.get("source", {}).get("path") == str(a.runner.resolve()), "build_source_path")
    check(build.get("source", {}).get("sha256") == sha(a.runner.resolve()), "build_source_hash")
    check(build.get("binary", {}).get("path") == str(a.binary.resolve()), "build_binary_path")
    check(build.get("binary", {}).get("sha256") == sha(a.binary.resolve()), "build_binary_hash")
    check(build.get("protocol", {}).get("path") == str(a.protocol.resolve()), "build_protocol_path")
    check(build.get("protocol", {}).get("sha256") == sha(a.protocol.resolve()), "build_protocol_hash")
    check(build.get("selection_witness", {}).get("path") == str(a.selection.resolve()), "build_selection_path")
    check(build.get("selection_witness", {}).get("sha256") == sha(a.selection.resolve()), "build_selection_hash")
    check(a.binary.is_file() and a.binary.stat().st_size > 0, "compiled_binary_exists")

    # Read-only static implementation audit.
    forbidden_includes = [r'^\s*#\s*include\s+[<"]incremental_insert\.cuh[>"]', r'^\s*#\s*include\s+[<"]update\.cuh[>"]']
    for idx, pattern in enumerate(forbidden_includes):
        check(not has(runner, pattern), f"no_forbidden_include_{idx}")
    for token in (r'\bincrementalInsert\s*\(', r'\bfindTargetLeaf\s*\(', r'\bupdateTree\s*\(', r'\bmergeBuffer\s*\('):
        check(not has(runner, token), "no_legacy_call", token=token)
    check(has(runner, r'constexpr\s+int\s+kTargetLeaf\s*=\s*221'), "runner_target_leaf_constant")
    check(has(runner, r'constexpr\s+int\s+kBoundaryStableId\s*=\s*5430'), "runner_boundary_constant")
    check(has(runner, r'kAcceptedStableIds\[kMaxAccepted\]\s*=\s*\{\s*4462\s*,\s*4507'), "runner_accepted_prefix")
    check(has(runner, r'4826\s*,\s*4942\s*,\s*5085\s*\}'), "runner_accepted_suffix")
    check(has(runner, r'kHeldoutQueryIds\[kHeldoutQueryCount\]\s*=\s*\{\s*0\s*,\s*1\s*,\s*2\s*\}'), "runner_queries_constant")
    check(has(runner, r'kHeldoutKValues\[kHeldoutKCount\]\s*=\s*\{\s*1\s*,\s*10\s*,\s*20\s*\}'), "runner_ks_constant")
    check(has(runner, r'kSelectionGeometryFNV1a64\s*=\s*0xa4a1881cd82237fbULL'), "runner_fingerprint_constant")
    check(has(runner, r'structural_geometry_fingerprint\s*\('), "runner_structural_fingerprint")
    check(has(runner, r'observed_fingerprint\s*!=\s*kSelectionGeometryFNV1a64'), "runner_blocks_geometry_mismatch")
    check(has(runner, r'require_target_leaf_and_external_nonself\s*\('), "runner_target_and_nonself_gate")
    check(has(runner, r'void\s+native_append\s*\('), "runner_native_append")
    check(has(runner, r'cudaMemcpy\(runtime\.id_list\s*\+\s*slot'), "runner_real_id_write")
    check(has(runner, r'cudaMemcpy\(runtime\.node_list\s*\+\s*leaf_id'), "runner_real_leaf_write")
    check(has(runner, r'leaf\.size\s*>=\s*MAX_SIZE'), "runner_native_prewrite_static_gate")
    check(has(runner, r'pre_leaf_size\s*>=\s*frozen\.max_size'), "runner_certificate_static_gate")
    check(has(runner, r'boundary\.reason\s*!=\s*"static_scan_limit"'), "runner_boundary_reason_gate")
    check(has(runner, r'const\s+Snapshot\s+boundary_before'), "runner_boundary_before_snapshot")
    check(has(runner, r'const\s+Snapshot\s+boundary_after'), "runner_boundary_after_snapshot")
    check(has(runner, r'assert_snapshot_identical\(boundary_before,\s*boundary_after'), "runner_boundary_no_mutation_assert")
    check(has(runner, r'emit_capacity_boundary\('), "runner_boundary_record")
    check(has(runner, r'"native_append_called\\\":false'), "runner_boundary_no_append_record")
    check(has(runner, r'"write_attempted\\\":false'), "runner_boundary_no_write_record")
    check(has(runner, r'"buffer_merge_used\\\":false'), "runner_boundary_no_merge_record")
    check(runner.count('native_append(runtime, kTargetLeaf') == 1, "runner_one_native_append_call_site", count=runner.count('native_append(runtime, kTargetLeaf'))
    check(has(runner, r'run_real_static_topk_external\s*\('), "runner_external_static_api")
    check(has(runner, r'exact_topk_external\s*\('), "runner_external_exact_oracle")
    check(has(runner, r'validate_heldout_query\s*\('), "runner_external_compare")
    check(has(runner, r'for\s*\(int\s+qidx\s*=\s*0;\s*qidx\s*<\s*kHeldoutQueryCount'), "runner_query_loop")
    check(has(runner, r'for\s*\(int\s+kidx\s*=\s*0;\s*kidx\s*<\s*kHeldoutKCount'), "runner_k_loop")
    check(has(runner, r'k\s*>\s*MAX_SIZE'), "runner_k_static_capacity_gate")
    main_body = function_body(runner, r'int\s+main\s*\(int\s+argc')
    check(main_body is not None, "runner_main_found")
    if main_body is not None:
        snap = main_body.find('const Snapshot initial = capture_snapshot(runtime)')
        gate = main_body.find('require_target_leaf_and_external_nonself')
        append = main_body.find('native_append(runtime, kTargetLeaf')
        external = main_body.find('run_real_static_topk_external')
        boundary = main_body.find('const Snapshot boundary_before')
        check(snap >= 0 and gate > snap and append > gate and external > append and boundary > external, "runner_runtime_gate_order")
        check(main_body.find('native_append(runtime, kTargetLeaf', boundary) < 0, "runner_no_append_after_boundary_phase")

    # Static source fact responsible for boundary gate: real vector leaf scan.
    leaf_body = function_body(search, r'__global__\s+void\s+dataProcessKnnVec\s*\(')
    check(leaf_body is not None, "archived_vector_leaf_kernel_found")
    if leaf_body is not None:
        check(has(leaf_body, r'for\s*\(\s*int\s+did\s*=\s*tid\s*;\s*did\s*<\s*MAX_SIZE\s*;\s*did\s*\+=\s*THREAD_NUM\s*\)'), "archived_leaf_scan_did_lt_max_size")
        check(has(leaf_body, r'if\s*\(\s*did\s*<\s*node\.size\s*&&\s*node\.is_leaf\s*==\s*1\s*\)'), "archived_leaf_payload_gate")
    check(has(tree, r'__managed__\s+int\s+MAX_SIZE\s*=\s*20'), "archived_max_size_20")
    check(has(tree, r'#define\s+LEAF_PAD_SLOTS\s+64'), "archived_leaf_pad_64")
    knn_body = function_body(search, r'__global__\s+void\s+nodeProcessKnn\s*\(')
    check(knn_body is not None, "archived_node_knn_found")
    if knn_body is not None:
        check(has(knn_body, r'if\s*\(\s*nid\s*%\s*TREE_ORDER\s*!=\s*0\s*\)'), "archived_successor_branch")
        check(has(knn_body, r'dis_q\s*-\s*node_list\[nid\s*\+\s*1\]\s*\.\s*min_dis'), "archived_successor_min_bound")
        check('max_dis_d' not in knn_body, "archived_knn_no_direct_max_distance_read")

    # Independent CPU validator must recompute rather than trust runner output.
    check(not has(validator, r'\b(nvidia-smi|subprocess|os\.system|cudaMemcpy|cudaDevice)\b'), "validator_cpu_only_source")
    check(has(validator, r'def\s+exact_external_topk\s*\('), "validator_external_oracle")
    check(has(validator, r'pool\[ids\]\.astype\(np\.int64'), "validator_int64_oracle")
    check(has(validator, r'runtime_initial_geometry_differs_from_witness'), "validator_full_witness_compare")
    check(has(validator, r'fnv1a64_geometry\(initial\)\s*!=\s*EXPECTED_FINGERPRINT'), "validator_runtime_fingerprint")
    check(has(validator, r'heldout_query_not_nonself'), "validator_nonself_check")
    check(has(validator, r'capacity_boundary_certificate'), "validator_boundary_certificate")
    check(has(validator, r'boundary_candidate_written'), "validator_boundary_not_written")
    check(has(validator, r'heldout_checks\s*!=\s*144'), "validator_all_heldout_checks")

    # Launcher must be inert by default and fail-closed before GPU access.
    check(has(launcher, r'default/no argument does not run CUDA or call nvidia-smi'), "launcher_inert_default_documented")
    check(has(launcher, r'GPU0_MAX_MEMORY_USED_MIB=256'), "launcher_memory_ceiling")
    check(has(launcher, r'verify_frozen_input_provenance\s*\('), "launcher_provenance_function")
    check(has(launcher, r'input_provenance_verdict=PASS'), "launcher_hash_provenance")
    check(has(launcher, r'if\s*!\s*apps="\$\(nvidia-smi\s+--id=0\s+--query-compute-apps'), "launcher_compute_query_fail_closed")
    check('query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 || true' not in launcher, "launcher_no_compute_query_failopen")
    check(has(launcher, r'\[\[\s*"\$apps"\s*!=\s*\*"No devices were found"\*\s*\]\]'), "launcher_rejects_no_device")
    check(has(launcher, r'\[\[\s*-z\s*"\$apps_clean"\s*\]\]'), "launcher_no_compute_processes")
    check(has(launcher, r'\[\[\s*"\$mem"\s*-le\s*"\$GPU0_MAX_MEMORY_USED_MIB"\s*\]\]'), "launcher_memory_gate")
    check(has(launcher, r'\[\[\s*"\$util"\s*-eq\s*0\s*\]\]'), "launcher_utilization_gate")
    check(has(launcher, r'CUDA_VISIBLE_DEVICES="\$GPU0_UUID"'), "launcher_uuid_mapping")
    check('CUDA_VISIBLE_DEVICES=0' not in launcher, "launcher_no_ordinal_mapping")
    check(has(launcher, r'gpu0_snapshot_and_require_idle\s+before_lock'), "launcher_prelock_idle")
    check(has(launcher, r'gpu0_snapshot_and_require_idle\s+immediately_pre_run'), "launcher_final_idle")
    check(has(launcher, r'setsid\s+timeout\s+--foreground\s+-k\s+30s\s+35m'), "launcher_isolated_timeout")
    check(has(launcher, r'reap_active_runner_before_unlock'), "launcher_reap_before_unlock")
    check(has(launcher, r'trap\s+cleanup_guard\s+EXIT'), "launcher_exit_trap")
    check(has(launcher, r"trap 'on_guard_signal INT' INT"), "launcher_int_trap")
    check(has(launcher, r"trap 'on_guard_signal TERM' TERM"), "launcher_term_trap")
    check(has(launcher, r'mkdir\s+"\$OUT"\s*\|\|\s*die\s+"refusing to overwrite existing run directory'), "launcher_no_run_overwrite")
    check(has(launcher, r'run_validator'), "launcher_runs_cpu_validator")
    execute_start = launcher.find('PRE="$ROOT/logs/preflight_')
    audit_at = launcher.find('static_audit "$PRE/static_audit.json"', execute_start)
    provenance_at = launcher.find('verify_frozen_input_provenance "$PRE/input_provenance.env"', execute_start)
    gpu_at = launcher.find('GPU0_UUID="$(nvidia-smi', execute_start)
    check(execute_start >= 0 and audit_at > execute_start and provenance_at > audit_at and gpu_at > provenance_at, "launcher_cpu_checks_before_first_gpu_query")

    result = {
        "schema": "c3-same-leaf-capacity-static-audit-v1",
        "status": "PASS" if not errors else "FAIL",
        "gpu_used": False,
        "scope": "static source/build/guard audit only; no CUDA runner or GPU command executed",
        "error_count": len(errors), "errors": errors,
        "static_leaf_scan_evidence": {
            "kernel": "dataProcessKnnVec",
            "scan": "did < MAX_SIZE",
            "max_size": 20,
            "leaf_payload_gate": "did < node.size && node.is_leaf == 1",
            "implication": "safe direct tier must not write when pre_leaf_size >= 20 even if padded id_list has 64 slots",
            "source_line": line_number(search, '__global__ void dataProcessKnnVec'),
        },
        "inputs": {str(path.resolve()): sha(path.resolve()) for path in (a.protocol, a.selection, a.runner, a.validator, a.tree_header, a.search_header, a.binary, a.build_manifest, a.launcher)},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "errors": len(errors), "out": str(a.out.resolve())}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == '__main__':
    raise SystemExit(main())
