#!/workspace/legacy_workspace/GTS/bench_env/bin/python
"""CPU-only source/contract audit for the bounded C3 native direct-insert probe.

This is deliberately a *static* audit: it executes no CUDA binary and does not
query a GPU.  Its search-upper checks bind the runtime certificate to the
actual archived `search_v2::nodeProcessKnn` implementation, which uses a
non-last child's immediate next sibling `min_dis` rather than `max_dis_d`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.MULTILINE | re.DOTALL) is not None


def function_body(text: str, signature_pattern: str) -> str | None:
    """Return the balanced-brace body for one C++/CUDA function signature."""
    match = re.search(signature_pattern, text, re.MULTILINE)
    if match is None:
        return None
    start = text.find("{", match.end())
    if start < 0:
        return None
    level = 0
    for pos in range(start, len(text)):
        char = text[pos]
        if char == "{":
            level += 1
        elif char == "}":
            level -= 1
            if level == 0:
                return text[start : pos + 1]
    return None


def line_number(text: str, needle: str) -> int | None:
    pos = text.find(needle)
    return None if pos < 0 else text.count("\n", 0, pos) + 1


def vector_static_api_body(search: str) -> str | None:
    """Find the vector-query overload that launches nodeProcessKnn + leaf scan."""
    pattern = r"void\s+searchIndexKnnV2\s*\(\s*short\s*\*\s*data_d"
    for match in re.finditer(pattern, search, re.MULTILINE):
        brace = search.find("{", match.end())
        signature = search[match.start() : brace] if brace >= 0 else ""
        body = function_body(search[match.start() :], pattern)
        if body is not None and re.search(r"float\s*\*\s*query_data", signature):
            return body
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--validator", required=True, type=Path)
    parser.add_argument("--tree-header", required=True, type=Path)
    parser.add_argument("--search-header", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--build-manifest", required=True, type=Path,
                        help="compile-only manifest binding this source/binary pair")
    parser.add_argument("--launcher", required=True, type=Path,
                        help="manual GPU0 guard; inspected only, never executed here")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    errors: list[dict[str, object]] = []
    protocol = json.loads(args.protocol.read_text())
    build_manifest = json.loads(args.build_manifest.read_text())
    runner = args.runner.read_text()
    launcher = args.launcher.read_text()
    validator = args.validator.read_text()
    tree = args.tree_header.read_text()
    search = args.search_header.read_text()

    def check(condition: bool, tag: str, **extra: object) -> None:
        if not condition:
            errors.append({"check": tag, **extra})

    # Frozen pre-registration/implementation identity.  The build manifest
    # prevents a later source edit or stale executable from silently passing a
    # text-only audit.
    check(protocol.get("schema") == "c3-native-certified-direct-insert-protocol-v2-search-upper", "protocol_schema")
    check(protocol.get("status") == "PRE_REGISTERED_AFTER_STATIC_SEARCH_UPPER_AUDIT_BEFORE_GPU_EXECUTION", "protocol_preregistration_status")
    check(build_manifest.get("schema") == "c3-native-direct-insert-compile-manifest-v1", "build_manifest_schema")
    check(build_manifest.get("status") == "COMPILE_ONLY_SUCCESS_NO_CUDA_BINARY_EXECUTION", "build_manifest_compile_only_status")
    runner_path = str(args.runner.resolve())
    binary_path = str(args.binary.resolve())
    check(build_manifest.get("source", {}).get("path") == runner_path, "build_manifest_source_path")
    check(build_manifest.get("binary", {}).get("path") == binary_path, "build_manifest_binary_path")
    check(build_manifest.get("source", {}).get("sha256") == sha(args.runner.resolve()), "build_manifest_source_hash")
    check(build_manifest.get("binary", {}).get("sha256") == sha(args.binary.resolve()), "build_manifest_binary_hash")

    # The native write must remain narrowly scoped and cannot call legacy C3 paths.
    check(not has(runner, r'^\s*#\s*include\s+[<"]incremental_insert\.cuh[>"]'), "no_incremental_insert_include")
    check(not has(runner, r'^\s*#\s*include\s+[<"]update\.cuh[>"]'), "no_update_include")
    check(not has(runner, r'\bincrementalInsert\s*\('), "no_legacy_incremental_insert_call")
    check(not has(runner, r'\bfindTargetLeaf\s*\('), "no_legacy_target_leaf_call")
    check(has(runner, r'void\s+native_append\s*\('), "native_append_function")
    check(has(runner, r'cudaMemcpy\(runtime\.id_list\s*\+\s*slot'), "native_real_id_list_write")
    check(has(runner, r'cudaMemcpy\(runtime\.node_list\s*\+\s*leaf_id'), "native_real_leaf_size_write")
    check(has(runner, r'leaf\.size\s*>=\s*MAX_SIZE'), "static_leaf_scan_capacity_gate")
    check(has(runner, r'void\s+searchIndexKnnV2|searchIndexKnnV2\(runtime\.data_d'), "real_static_topk_api")

    # The C3 probe must run the actual static API in uncalibrated baseline mode.
    check(has(runner, r'#define\s+RP_DEFINE_CONSTANTS'), "runner_defines_residual_constants")
    check(has(runner, r'#include\s+"residual_pruning\.cuh"'), "runner_includes_residual_header")
    check(has(runner, r'void\s+upload_rp_constants\s*\('), "runner_defines_residual_uploader")
    check(has(runner, r'install_baseline_residual_mode_zero\s*\('), "runner_installs_residual_mode_zero")
    check(has(runner, r'upload_rp_constants\([^;]*nullptr\s*,\s*nullptr\s*,\s*nullptr\s*,\s*0\s*,\s*0\s*\)'), "runner_uploads_mode_zero")
    check(has(runner, r'cudaMemcpyFromSymbol\s*\(\s*&mode\s*,\s*c_rp_mode'), "runner_reads_back_residual_mode")
    check(has(runner, r'read_residual_mode\s*\(\s*\)\s*!=\s*0'), "runner_requires_verified_residual_mode_zero")
    check(has(validator, r'residual_pruning_mode'), "validator_checks_residual_mode_metadata")
    check(has(validator, r'residual_pruning_runtime_verified'), "validator_checks_residual_runtime_metadata")

    # The manual launcher must be fail-closed on physical GPU0.  This auditor
    # never invokes it: it verifies source-level presence and ordering of the
    # two independent idle checks (pre-lock and immediately pre-timeout).
    check(has(launcher, r'GPU0_MAX_MEMORY_USED_MIB=256'), "launcher_gpu0_memory_ceiling")
    check(has(launcher, r'default/no argument does not run CUDA or call nvidia-smi'), "launcher_inert_default_documented")
    check(has(launcher, r'gpu0_snapshot_and_require_idle\s*\(\)'), "launcher_idle_check_function")
    check(has(launcher, r'--query-gpu=uuid,memory\.used,utilization\.gpu\s+--format=csv,noheader,nounits'), "launcher_machine_readable_gpu_query")
    check(has(launcher, r'\[\[\s*"\$memory_used"\s*-le\s*"\$GPU0_MAX_MEMORY_USED_MIB"\s*\]\]'), "launcher_requires_memory_ceiling")
    check(has(launcher, r'\[\[\s*"\$utilization"\s*-eq\s*0\s*\]\]'), "launcher_requires_zero_utilization")
    check(has(launcher, r'\[\[\s*-z\s*"\$apps_clean"\s*\]\]'), "launcher_requires_no_compute_processes")
    check(has(launcher, r'gpu0_snapshot_and_require_idle\s+"before_lock"'), "launcher_checks_before_lock")
    check(has(launcher, r'gpu0_snapshot_and_require_idle\s+"immediately_pre_run"'), "launcher_checks_immediately_pre_run")
    lock_pos = launcher.find('if ! mkdir "$LOCK"')
    final_check_pos = launcher.find('gpu0_snapshot_and_require_idle "immediately_pre_run"')
    timeout_pos = launcher.find('timeout --foreground 20m "$RUNNER"')
    check(lock_pos >= 0 and final_check_pos > lock_pos and timeout_pos > final_check_pos,
          "launcher_final_idle_check_order")
    check(has(launcher, r'CUDA_VISIBLE_DEVICES="\$GPU0_UUID"'), "launcher_maps_physical_gpu0_by_uuid")
    check(has(launcher, r'--launcher\s+"\$0"'), "launcher_passes_self_to_static_audit")

    # Bind the certificate to what *static vector KNN* really launches.
    knn_body = function_body(search, r'__global__\s+void\s+nodeProcessKnn\s*\(')
    vector_body = vector_static_api_body(search)
    check(knn_body is not None, "archived_node_process_knn_found")
    if knn_body is not None:
        check(has(knn_body, r'dis_lb\s*=\s*node\.min_dis\s*-\s*dis_q'), "archived_knn_lower_from_min_dis")
        check(has(knn_body, r'if\s*\(\s*nid\s*%\s*TREE_ORDER\s*!=\s*0\s*\)'), "archived_knn_nonlast_condition")
        check(has(knn_body, r'dis_q\s*-\s*node_list\[nid\s*\+\s*1\]\s*\.\s*min_dis'), "archived_knn_next_sibling_min_upper")
        check("max_dis_d" not in knn_body, "archived_knn_does_not_read_max_dis_d")
    check(vector_body is not None, "archived_vector_static_api_found")
    if vector_body is not None:
        check("nodeProcessKnn<<<" in vector_body, "archived_vector_api_launches_node_process_knn")
        check("dataProcessKnnVec<<<" in vector_body, "archived_vector_api_launches_leaf_scan")

    # Therefore this is an additional certificate condition, not a source-comment assumption.
    check(has(runner, r'verify_search_upper_invariant'), "runner_search_upper_audit")
    check(has(runner, r'max_upper\s*<=\s*next_min'), "runner_max_le_next_min_check")
    check(has(runner, r'actual_search_upper\s*=\s*frozen\.nodes\[next\]\.min_dis'), "runner_actual_next_sibling_upper")
    check(has(runner, r'distance\s*<\s*actual_search_upper\s*-\s*kStrictEpsilon'), "runner_candidate_below_actual_search_upper")
    check(has(validator, r'def\s+search_upper_invariant\s*\('), "validator_search_upper_audit")
    check(has(validator, r'max_upper\s*<=\s*next_min'), "validator_max_le_next_min_check")
    check(has(validator, r'def\s+exact_topk\s*\('), "validator_independent_exact_oracle")
    check(has(validator, r'pool\[ids\]\.astype\(np\.int64'), "validator_int64_distance")
    check(has(tree, r'#define\s+LEAF_PAD_SLOTS\s+64'), "archived_leaf_padding_64_present")
    check(has(tree, r'__managed__\s+int\s+MAX_SIZE\s*=\s*20'), "archived_static_max_size_20_present")
    check(args.binary.is_file() and args.binary.stat().st_size > 0, "compiled_binary_exists")

    result = {
        "schema": "c3-native-direct-insert-static-audit-v2",
        "status": "PASS" if not errors else "FAIL",
        "gpu_used": False,
        "scope": "source/build contract audit only; no CUDA runner execution",
        "error_count": len(errors),
        "errors": errors,
        "actual_search_upper_audit": {
            "static_vector_api": "searchIndexKnnV2(float query_data) launches nodeProcessKnn and dataProcessKnnVec",
            "node_process_upper": "non-last child uses dis_q - node_list[nid+1].min_dis; nodeProcessKnn does not read max_dis_d",
            "certificate_response": "do not assume a tree-construction theorem; frozen geometry must explicitly satisfy max_dis_d[child] <= next_sibling.min_dis and each candidate must be strictly below both max_dis_d and the actual next-sibling min_dis",
            "node_process_knn_line": line_number(search, "__global__ void nodeProcessKnn"),
            "vector_static_api_line": line_number(search, "void searchIndexKnnV2(short *data_d, TN *node_list, int *id_list, int *max_node_num, float *query_data"),
        },
        "inputs": {str(path.resolve()): sha(path.resolve()) for path in (args.protocol, args.build_manifest, args.runner, args.launcher, args.validator, args.tree_header, args.search_header, args.binary)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "errors": len(errors), "out": str(args.out.resolve())}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
