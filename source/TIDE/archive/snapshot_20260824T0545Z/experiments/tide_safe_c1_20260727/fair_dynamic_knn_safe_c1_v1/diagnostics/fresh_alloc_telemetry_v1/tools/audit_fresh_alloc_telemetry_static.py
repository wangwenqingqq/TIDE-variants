#!/usr/bin/env python3
"""Static-only provenance audit for the Stage-0 fresh-allocation variant.

It reads only source/text files.  It never invokes nvcc, a CUDA runner, NVML,
nvidia-smi, or any GPU API.  On a clean PASS, ``--write`` emits a reviewed
unified diff and a one-way source manifest.  The manifest excludes itself,
this audit script, build receipts, object files, and binaries, avoiding a
self/hash cycle.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/fair_dynamic_knn_safe_c1_v1")
DIAG = ROOT / "diagnostics/fresh_alloc_telemetry_v1"
PROVENANCE = DIAG / "provenance"

CONTROL_HEADER = ROOT / "src/g3_safe_search_v2.cuh"
CONTROL_MATRIX = ROOT / "src/g3_safe_c1_native_matrix.cu"
CONTROL_RUNNER = ROOT / "runner/fair_safe_c1_latency_tradeoff_e1_runner.cu"
CANDIDATE_HEADER = DIAG / "src/g3_safe_search_v2_fresh_alloc_telemetry.cuh"
CANDIDATE_MATRIX = DIAG / "src/g3_safe_c1_native_matrix_fresh_alloc_telemetry.cu"
CANDIDATE_RUNNER = DIAG / "runner/fair_safe_c1_fresh_alloc_telemetry_e1_runner.cu"
COMPILE_HELPER = DIAG / "tools/compile_safe_c1_fresh_alloc_telemetry.sh"
VALIDATOR = DIAG / "tools/validate_safe_c1_fresh_alloc_telemetry_output.py"
GUARD = DIAG / "tools/run_safe_c1_fresh_alloc_telemetry_guarded.py"
REVIEWED_DIFF = PROVENANCE / "fresh_alloc_telemetry_reviewed_diff.patch"
MANIFEST = PROVENANCE / "fresh_alloc_telemetry_source_manifest.json"

# Fixed v1b source closure: never derive these expected values from a mutable
# provenance file during the audit.
EXPECTED_CONTROL_SHA256 = {
    CONTROL_HEADER: "65688fedcbb05a1fff680555e3139b6e640bfdd72288b93dfccddcec90dfbf01",
    CONTROL_MATRIX: "b2a334711339c7dbd6d23e15d0d9c6ade6572da6aa31932846c36ab48e11d74a",
    CONTROL_RUNNER: "3a283f322ef2675b7e6d64062fb219b228375c4d15ebda1003b8f9f63766f9fb",
    ROOT / "reference/include/tree.cuh": "c1324bef173358c31a8371e1f050d4372cd1c92cd98c71af3832b2d19c769fd6",
    ROOT / "reference/include/file.cuh": "b8be03246ec82cadd3228b75a6dfd91c552ccfa3124d2ca9470e28e0031ffd8a",
    ROOT / "reference/include/config.cuh": "622d0977e49d80d5791364bc12463de9bce5aaca324d6a681512004c20d100cb",
    ROOT / "reference/include/mlp_constant.cuh": "cf6545624c0cbc51744b978298e6d138591521c7b600ece8812b6195f735e559",
    ROOT / "reference/include/residual_pruning.cuh": "745a4bd5564b8085c40a48abac625f24dcbacb8a996d29e4cba311dd936e33b2",
}
CONTROL_RUN = ROOT / "runs/safe-c1-tradeoff-pilot-v1b-20260730"
CONTROL_RUN_SHA256 = {
    "engine.jsonl": "4a32720f49ac49bed6bd6f2755532231ea8a04e5d3c797e342e56ef22cb35f11",
    "summary.json": "5d9f197cf7297cb970a69f8a5fdb7255efad9d06c37adac343377af5a185c103",
    "guard_receipt.json": "1400941c7d898c2f8daa4884c85c8fa8e1a027f1ffdcc2f88af4a574070b63f8",
    "independent_validation.json": "2719fcffa018e352d4c7b8da480c889545a57d5a39605bff424511fb8acb042e",
    "admission.env": "a979ba830a2f39bf12ddace378b1be7106835233cd2e0f8e1bd8917175c75721",
}

VARIANT = "fresh_allocation_telemetry_control"
MANIFEST_SCHEMA = "fair-safe-c1-fresh-allocation-telemetry-source-manifest-v1"
EVENT_SCHEMA = "fresh-allocation-telemetry-e1-v1"
SUMMARY_SCHEMA = "fresh-allocation-telemetry-e1-v1"
CLAIM_SCOPE = "semantic_fresh_allocation_control_only"

# These are the sealed pure-semantic runner/validator/guard bytes reviewed by
# this Stage-0 audit.  The manifest carries the complete closure digest below;
# pinning the three executable/semantic boundary sources here prevents an old
# timing-oriented implementation from being silently substituted before a
# manifest can be generated.
EXPECTED_SEMANTIC_SOURCE_SHA256 = {
    CANDIDATE_RUNNER: "20c39ba38c83a51062658a9e915b88f7af8aedbcc7b79d78da357b0544ad70e9",
    VALIDATOR: "0aa64cd2807c662c710bd72a6717b08894464cd8d89fb99fd5f8b80f8ffab4cc",
    GUARD: "b5eb4d275b52cf4cef3c85a52aee6eb94f11573042be86b7a3ba9ea83d8fd42b",
}
BANNED_ADDED_TOKENS = (
    "cudaMallocAsync", "cudaFreeAsync", "cudaMallocFromPoolAsync",
    "cudaMemcpyAsync", "cudaMemsetAsync", "cudaStream", "cudaEvent",
    "cudaGraph", "cudaMemPool", "cudaDeviceSetMemPool",
    "cudaLaunchHostFunc", "cudaStreamAttachMemAsync", "cudaStreamBeginCapture",
    "cudaStreamEndCapture", "thrust::cuda::par", "std::async", "std::thread",
    "<thread>", "<future>", "TraversalScratch", "scratch_cache",
    "persistent_p_list", "reuse_p_list", "p_list_cache",
)

class AuditError(RuntimeError):
    pass

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def private_dir(path: Path, label: str) -> None:
    try:
        entry = path.lstat()
    except OSError as exc:
        raise AuditError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        raise AuditError(f"unsafe {label}: {path}")
    if entry.st_uid != 0 or entry.st_gid != 0 or stat.S_IMODE(entry.st_mode) != 0o700:
        raise AuditError(f"{label} must be root-private 0700: {path}")

def private_file(path: Path, label: str, executable: bool = False) -> None:
    try:
        entry = path.lstat()
    except OSError as exc:
        raise AuditError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise AuditError(f"unsafe {label}: {path}")
    if entry.st_uid != 0 or entry.st_gid != 0 or stat.S_IMODE(entry.st_mode) & 0o077:
        raise AuditError(f"{label} is not root-private: {path}")
    if executable and not stat.S_IMODE(entry.st_mode) & 0o100:
        raise AuditError(f"{label} is not owner-executable: {path}")

def text_and_hash(path: Path, label: str, executable: bool = False) -> tuple[str, str]:
    private_file(path, label, executable)
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
    except UnicodeDecodeError as exc:
        raise AuditError(f"{label} is not utf-8: {path}") from exc

def added_lines(patch: str) -> str:
    return "\n".join(line[1:] for line in patch.splitlines()
                     if line.startswith("+") and not line.startswith("+++"))

def cxx_matching_brace(text: str, first_brace: int) -> int:
    """Brace matcher that skips C++ comments and quoted literals."""
    depth, index, state, escaped = 0, first_brace, "normal", False
    while index < len(text):
        ch = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if state == "normal":
            if ch == "/" and nxt == "/":
                state = "line"; index += 2; continue
            if ch == "/" and nxt == "*":
                state = "block"; index += 2; continue
            if ch == '"':
                state = "string"; escaped = False
            elif ch == "'":
                state = "char"; escaped = False
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return index
        elif state == "line":
            if ch == "\n": state = "normal"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "normal"; index += 2; continue
        else:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif (state == "string" and ch == '"') or (state == "char" and ch == "'"):
                state = "normal"
        index += 1
    raise AuditError("unterminated C++ function body")

def cxx_function(text: str, marker: str, label: str) -> str:
    pos = text.find(marker)
    if pos < 0: raise AuditError(f"missing {label} marker: {marker}")
    if text.find(marker, pos + 1) >= 0: raise AuditError(f"ambiguous {label} marker")
    brace = text.find("{", pos)
    if brace < 0: raise AuditError(f"missing {label} opening brace")
    return text[pos:cxx_matching_brace(text, brace) + 1]

def cuda_calls(text: str) -> list[str]:
    return re.findall(r"\b(cuda[A-Za-z0-9_]+)\s*\(", text)

def require_token(issues: list[str], text: str, token: str, label: str) -> None:
    if token not in text: issues.append(f"missing {label}: {token}")

def active_vector_knn(text: str) -> str:
    return cxx_function(text, "// knn query (return top-k ids and distances) - vector queries",
                        "active vector KNN")

def topk_adapter(text: str) -> str:
    return cxx_function(text, "inline TraversalReceipt run_gts_base_topk_with_receipt(",
                        "top-k adapter")

def python_function(text: str, name: str, label: str) -> str:
    """Return one top-level Python function without executing its module."""
    pattern = re.compile(r"^def " + re.escape(name) + r"\s*\(", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise AuditError(f"expected one {label} function, got {len(matches)}")
    start = matches[0].start()
    next_def = re.search(r"^def [A-Za-z_][A-Za-z0-9_]*\s*\(",
                         text[matches[0].end():], re.MULTILINE)
    end = len(text) if next_def is None else matches[0].end() + next_def.start()
    return text[start:end]


def python_set_assignment(text: str, name: str, label: str) -> tuple[str, int, int]:
    """Return a top-level ``NAME = { ... }`` set assignment and its offsets."""
    match = re.search(r"^" + re.escape(name) + r"\s*=\s*\{", text, re.MULTILINE)
    if match is None:
        raise AuditError(f"missing {label} set assignment: {name}")
    if len(re.findall(r"^" + re.escape(name) + r"\s*=\s*\{", text, re.MULTILINE)) != 1:
        raise AuditError(f"ambiguous {label} set assignment: {name}")
    brace = text.find("{", match.start())
    end = cxx_matching_brace(text, brace) + 1
    return text[match.start():end], match.start(), end

def check_control(issues: list[str]) -> None:
    for path, expected in EXPECTED_CONTROL_SHA256.items():
        try:
            private_file(path, "immutable v1b source")
            actual = sha256_file(path)
            if actual != expected:
                issues.append(f"immutable v1b source hash drift: {path}")
        except AuditError as exc:
            issues.append(str(exc))
    for name, expected in CONTROL_RUN_SHA256.items():
        path = CONTROL_RUN / name
        try:
            private_file(path, "immutable v1b run artifact")
            if sha256_file(path) != expected:
                issues.append(f"immutable v1b run hash drift: {path}")
        except AuditError as exc:
            issues.append(str(exc))

def check_header(issues: list[str], control: str, candidate: str) -> None:
    try:
        old, new = active_vector_knn(control), active_vector_knn(candidate)
    except AuditError as exc:
        issues.append(str(exc)); return
    anchors = (
        'CHECK(cudaMallocManaged((void **)&res_dis, qnum * k * sizeof(float)));',
        'CHECK(cudaMallocManaged((void **)&size_list, (tree_h + 1) * sizeof(int)));',
        'CHECK(cudaMalloc((void **)&disk, qnum * sizeof(float)));',
        'safe_c1_traversal_require_cuda_success(cudaMemGetInfo(&avail, &total), "cudaMemGetInfo");',
        'CHECK(cudaMalloc((void **)&p_list_k, size_a * sizeof(double)));',
        'safe_c1_traversal_require_cuda_success(cudaFree(p_list_k), "cudaFree p_list_k");',
        'safe_c1_traversal_require_cuda_success(cudaFree(size_list), "cudaFree size_list");',
        'safe_c1_traversal_require_cuda_success(cudaFree(disk), "cudaFree disk");',
        'int end = min((int)(i + qnum_l_low), qnum_l);',
    )
    for anchor in anchors:
        if old.count(anchor) != 1 or new.count(anchor) != 1:
            issues.append(f"header fresh-allocation anchor changed/duplicated: {anchor}")
    if cuda_calls(old) != cuda_calls(new):
        issues.append("header active vector CUDA call sequence differs from v1b")
    if old.count("st.push(") != new.count("st.push(") or old.count("st.pop()") != new.count("st.pop()"):
        issues.append("header active vector push/pop count differs from v1b")
    order = anchors[:5]
    positions = [new.find(x) for x in order]
    if any(x < 0 for x in positions) or positions != sorted(positions):
        issues.append("header fresh alloc -> meminfo -> p_list ordering changed")
    for token in (
        "SafeC1FreshAllocTelemetry", "safe_c1_last_fresh_alloc_telemetry",
        "raw_cuda_mem_avail_bytes", "policy_available_bytes", "p_list_elements",
        "stack_events", "size_list_writes", "traversal_steps", "ordered_leaf_pairs",
        "res_dis_allocated", "p_list_freed", "exit_stack_empty",
    ):
        require_token(issues, candidate, token, "header telemetry")

def _next_cuda_call(text: str, start: int) -> int:
    match = re.search(r"\bcuda[A-Za-z0-9_]+\s*\(", text[start:])
    return -1 if match is None else start + match.start()


def _direct_ledger_event_after(issues: list[str], scope: str, anchor: str,
                               slot: str, operation: str, bytes_token: str,
                               label: str, before: int = -1) -> int:
    """Bind one ledger append to the preceding real CUDA success macro.

    CHECK/G3_CUDA/safe_c1_traversal_require_cuda_success only continue after
    a successful CUDA result.  Therefore an append between that macro's
    terminating semicolon and the next CUDA call is a source-level proof that
    the normal-path ledger is observed after—not fabricated before/after—the
    actual direct runtime operation.
    """
    count = scope.count(anchor)
    if count != 1:
        issues.append(f"direct ledger anchor missing/ambiguous ({label}): {anchor}")
        return -1
    call = scope.find(anchor)
    success_end = scope.find(";", call)
    if success_end < 0:
        issues.append(f"direct ledger CUDA success statement unterminated: {label}")
        return -1
    event = scope.find("safe_c1_fresh_alloc_record_direct_cuda_event(", success_end + 1)
    next_cuda = _next_cuda_call(scope, success_end + 1)
    if event < 0 or (next_cuda >= 0 and event > next_cuda):
        issues.append(f"direct ledger event is not after successful CUDA call and before next CUDA call: {label}")
        return -1
    if before >= 0 and event > before:
        issues.append(f"direct ledger event occurs after its runtime boundary: {label}")
        return -1
    event_end = scope.find(";", event)
    if event_end < 0:
        issues.append(f"direct ledger event unterminated: {label}")
        return -1
    event_text = scope[event:event_end + 1]
    for token in (slot, operation, bytes_token):
        if token not in event_text:
            issues.append(f"direct ledger event lacks {token}: {label}")
    return event


def check_direct_cuda_event_ledger(issues: list[str], header: str, matrix: str) -> None:
    """Static source audit for the ten direct CUDA alloc/free runtime events."""
    try:
        header_scope = active_vector_knn(header)
        matrix_scope = topk_adapter(matrix)
        begin_scope = cxx_function(header, "inline void safe_c1_fresh_alloc_begin_epoch(",
                                   "fresh-allocation epoch initializer")
    except AuditError as exc:
        issues.append(str(exc)); return

    for token in (
        "SafeC1FreshAllocDirectCudaEvent", "direct_allocation_events",
        "SAFE_C1_FRESH_ALLOC_LOCAL_RESULT_IDS", "SAFE_C1_FRESH_ALLOC_RES_DIS",
        "SAFE_C1_FRESH_ALLOC_SIZE_LIST", "SAFE_C1_FRESH_ALLOC_DISK",
        "SAFE_C1_FRESH_ALLOC_P_LIST_K", "SAFE_C1_FRESH_ALLOC_DIRECT_ALLOC",
        "SAFE_C1_FRESH_ALLOC_DIRECT_FREE",
    ):
        require_token(issues, header, token, "direct CUDA ledger definition")
    if header.count("direct_allocation_events.push_back(") != 1:
        issues.append("direct CUDA ledger must have one canonical append helper")
    if (header.count("safe_c1_fresh_alloc_record_direct_cuda_event(") +
            matrix.count("safe_c1_fresh_alloc_record_direct_cuda_event(")) != 11:
        issues.append("direct CUDA event helper is invoked outside the ten named lifecycle sites")
    for forbidden in (
        "direct_allocation_events.clear(", "direct_allocation_events.assign(",
        "direct_allocation_events.resize(", "direct_allocation_events =",
    ):
        if forbidden in header or forbidden in matrix:
            issues.append("direct CUDA ledger can be overwritten/fabricated: " + forbidden)
    for token in (
        "const bool stack_empty = st.empty();",
        "res_dis == nullptr && size_list == nullptr && disk == nullptr && p_list_k == nullptr",
        "if (!stack_empty || !named_globals_null || update_disk)",
        "safe_c1_last_visited_leaf_pairs.clear();",
        "safe_c1_last_fresh_alloc_telemetry = SafeC1FreshAllocTelemetry{};",
        "allocation_epoch = ++safe_c1_fresh_alloc_epoch_counter;",
        "entry_stack_empty = stack_empty;", "entry_named_global_scratch_ptrs_null = named_globals_null;",
        "receipt_cleared = receipt_cleared;", "update_disk_at_header_entry = update_disk;",
        "qnum = qnum_value;", "k = k_value;", "tree_h = tree_h_value;",
    ):
        require_token(issues, begin_scope, token, "epoch reset initializer")
    if "safe_c1_last_fresh_alloc_telemetry = SafeC1FreshAllocTelemetry{};" in header_scope:
        issues.append("header resets telemetry after local_result_ids allocation boundary")

    begin = matrix_scope.find("safe_c1_fresh_alloc_begin_epoch(qnum, k, runtime.tree_height);")
    local_anchor = "G3_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&local_result_ids),"
    local_call = matrix_scope.find(local_anchor)
    if begin < 0 or local_call < 0 or begin >= local_call:
        issues.append("epoch/reset is not before the real local_result_ids allocation")
    # The adapter must not pre-clear this mutable bit: begin_epoch is the
    # stale-state gate.  Normal header exit / exception cleanup own its reset.
    if "update_disk = false;" in matrix_scope[:begin]:
        issues.append("matrix masks stale update_disk before the epoch gate")
    traversal_call = matrix_scope.find("searchIndexKnnV2(", local_call)

    matrix_events = [
        (local_anchor, "SAFE_C1_FRESH_ALLOC_LOCAL_RESULT_IDS",
         "SAFE_C1_FRESH_ALLOC_DIRECT_ALLOC", "local_result_ids_bytes",
         "local_result_ids allocation", traversal_call),
        ("G3_CUDA(cudaFree(res_dis));", "SAFE_C1_FRESH_ALLOC_RES_DIS",
         "SAFE_C1_FRESH_ALLOC_DIRECT_FREE", "res_dis_requested_bytes",
         "res_dis free", -1),
        ("G3_CUDA(cudaFree(local_result_ids));", "SAFE_C1_FRESH_ALLOC_LOCAL_RESULT_IDS",
         "SAFE_C1_FRESH_ALLOC_DIRECT_FREE", "local_result_ids_requested_bytes",
         "local_result_ids free", -1),
    ]
    header_events = [
        ("CHECK(cudaMallocManaged((void **)&res_dis, qnum * k * sizeof(float)));",
         "SAFE_C1_FRESH_ALLOC_RES_DIS", "SAFE_C1_FRESH_ALLOC_DIRECT_ALLOC",
         "res_dis_requested_bytes", "res_dis allocation"),
        ("CHECK(cudaMallocManaged((void **)&size_list, (tree_h + 1) * sizeof(int)));",
         "SAFE_C1_FRESH_ALLOC_SIZE_LIST", "SAFE_C1_FRESH_ALLOC_DIRECT_ALLOC",
         "size_list_requested_bytes", "size_list allocation"),
        ("CHECK(cudaMalloc((void **)&disk, qnum * sizeof(float)));",
         "SAFE_C1_FRESH_ALLOC_DISK", "SAFE_C1_FRESH_ALLOC_DIRECT_ALLOC",
         "disk_requested_bytes", "disk allocation"),
        ("CHECK(cudaMalloc((void **)&p_list_k, size_a * sizeof(double)));",
         "SAFE_C1_FRESH_ALLOC_P_LIST_K", "SAFE_C1_FRESH_ALLOC_DIRECT_ALLOC",
         "p_list_requested_bytes", "p_list_k allocation"),
        ("safe_c1_traversal_require_cuda_success(cudaFree(p_list_k), \"cudaFree p_list_k\");",
         "SAFE_C1_FRESH_ALLOC_P_LIST_K", "SAFE_C1_FRESH_ALLOC_DIRECT_FREE",
         "p_list_requested_bytes", "p_list_k free"),
        ("safe_c1_traversal_require_cuda_success(cudaFree(size_list), \"cudaFree size_list\");",
         "SAFE_C1_FRESH_ALLOC_SIZE_LIST", "SAFE_C1_FRESH_ALLOC_DIRECT_FREE",
         "size_list_requested_bytes", "size_list free"),
        ("safe_c1_traversal_require_cuda_success(cudaFree(disk), \"cudaFree disk\");",
         "SAFE_C1_FRESH_ALLOC_DISK", "SAFE_C1_FRESH_ALLOC_DIRECT_FREE",
         "disk_requested_bytes", "disk free"),
    ]
    if matrix_scope.count("safe_c1_fresh_alloc_record_direct_cuda_event(") != len(matrix_events):
        issues.append("matrix does not contain exactly the three direct CUDA ledger events")
    if header_scope.count("safe_c1_fresh_alloc_record_direct_cuda_event(") != len(header_events):
        issues.append("header does not contain exactly the seven direct CUDA ledger events")
    matrix_positions: list[int] = []
    for anchor, slot, operation, bytes_token, label, boundary in matrix_events:
        matrix_positions.append(_direct_ledger_event_after(
            issues, matrix_scope, anchor, slot, operation, bytes_token, label, boundary))
    header_positions: list[int] = []
    for anchor, slot, operation, bytes_token, label in header_events:
        header_positions.append(_direct_ledger_event_after(
            issues, header_scope, anchor, slot, operation, bytes_token, label))
    if any(position < 0 for position in matrix_positions + header_positions):
        return
    if header_positions != sorted(header_positions):
        issues.append("header direct CUDA ledger events are not in lifecycle order")
    if not (matrix_positions[0] < traversal_call < matrix_positions[1] < matrix_positions[2]):
        issues.append("adapter direct CUDA ledger events do not bracket the real traversal in lifecycle order")
    _require_null_after_success(issues, header_scope, header_events[4][0],
                                "p_list_k", header_positions[4], "p_list_k free")
    _require_null_after_success(issues, header_scope, header_events[5][0],
                                "size_list", header_positions[5], "size_list free")
    _require_null_after_success(issues, header_scope, header_events[6][0],
                                "disk", header_positions[6], "disk free")
    _require_null_after_success(issues, matrix_scope, matrix_events[1][0],
                                "res_dis", matrix_positions[1], "res_dis free")
    _require_null_after_success(issues, matrix_scope, matrix_events[2][0],
                                "local_result_ids", matrix_positions[2], "local_result_ids free")
    receipt_copy = matrix_scope.find("output.fresh_allocation_telemetry =")
    if receipt_copy < matrix_positions[2]:
        issues.append("receipt copies direct CUDA ledger before final normal-path free event")
    # Runtime order is thereby fixed: adapter local-result allocation, the
    # synchronous header's 4 alloc+3 free events, then adapter res_dis/local
    # result frees.  The expected tags are checked on each individual append.


def _require_null_after_success(issues: list[str], scope: str, anchor: str,
                                pointer: str, event_position: int, label: str) -> None:
    call = scope.find(anchor)
    end = scope.find(";", call)
    null = scope.find(pointer + " = nullptr;", end + 1)
    if call < 0 or end < 0 or null < 0 or null > event_position:
        issues.append(f"direct CUDA free does not null its named scratch pointer before ledger event: {label}")


def check_fresh_alloc_exit_and_failstop(issues: list[str], header: str, matrix: str) -> None:
    """Audit fail-stop cleanup and normal-return state, separately from events."""
    try:
        header_scope = active_vector_knn(header)
        matrix_scope = topk_adapter(matrix)
        cleanup_scope = cxx_function(header, "struct SafeC1FreshAllocGlobalScratchCleanup",
                                     "fresh-allocation exceptional cleanup guard")
        adapter_ledger = cxx_function(matrix, "inline void require_complete_fresh_allocation_lifecycle(",
                                      "adapter complete direct-lifecycle gate")
    except AuditError as exc:
        issues.append(str(exc)); return
    for token in (
        "while (!st.empty())", "st.pop();",
        "if (p_list_k != nullptr) { (void)cudaFree(p_list_k); p_list_k = nullptr; }",
        "if (size_list != nullptr) { (void)cudaFree(size_list); size_list = nullptr; }",
        "if (disk != nullptr) { (void)cudaFree(disk); disk = nullptr; }",
        "if (res_dis != nullptr) { (void)cudaFree(res_dis); res_dis = nullptr; }",
        "update_disk = false;",
    ):
        require_token(issues, cleanup_scope, token, "exceptional fail-stop cleanup")
    for token in (
        "SafeC1FreshAllocGlobalScratchCleanup fresh_alloc_exception_cleanup;",
        "exit_stack_empty = st.empty();",
        "incomplete Stage-0 vector KNN exit state", "update_disk = false;",
        "normal_return = true;", "fresh_alloc_exception_cleanup.dismiss();",
    ):
        require_token(issues, header_scope, token, "normal header exit")
    final_header_free = header_scope.find("safe_c1_traversal_require_cuda_success(cudaFree(disk)")
    normal_reset = header_scope.rfind("update_disk = false;")
    normal_return = header_scope.rfind("normal_return = true;")
    if final_header_free < 0 or not (final_header_free < normal_reset < normal_return):
        issues.append("normal header exit does not reset update_disk only after its direct lifecycle")
    for pointer in ("p_list_k", "size_list", "disk"):
        require_token(issues, header_scope, pointer + " = nullptr;", "normal header free")
    for token in (
        "exit_stack_empty", "res_dis = nullptr;", "local_result_ids = nullptr;",
        "require_complete_fresh_allocation_lifecycle(completed);",
        "Stage-0 direct CUDA lifecycle left live traversal scratch",
    ):
        require_token(issues, matrix_scope, token, "normal adapter exit")
    for token in (
        "std::array<ExpectedEvent, 10>", "direct_allocation_events.size() != expected.size()",
        "entry_named_global_scratch_ptrs_null", "exit_stack_empty",
        "Stage-0 direct CUDA event ledger order/value mismatch",
    ):
        require_token(issues, adapter_ledger, token, "adapter complete direct-lifecycle gate")


def check_matrix(issues: list[str], control: str, candidate: str) -> None:
    try:
        old, new = topk_adapter(control), topk_adapter(candidate)
    except AuditError as exc:
        issues.append(str(exc)); return
    for token in (
        "cudaMallocManaged(reinterpret_cast<void**>(&local_result_ids)",
        "searchIndexKnnV2(runtime.data_d", "cudaDeviceSynchronize()",
        "cudaFree(res_dis)", "cudaFree(local_result_ids)",
    ):
        require_token(issues, new, token, "matrix lifecycle")
    if cuda_calls(old) != cuda_calls(new):
        issues.append("matrix top-k adapter CUDA call sequence differs from v1b")
    for token in (
        "native_final_res_distances", "ordered_raw_leaf_pairs",
        "fresh_allocation_telemetry", "ordered_leaf_pairs",
    ):
        require_token(issues, candidate, token, "matrix telemetry propagation")
    required_order = ("receipt_leaf_spans(", "materialize_receipt_leaf_rows(",
                      "scan_local_rows_exact(runtime, query_device, receipt_locals)")
    indices = [new.find(x) for x in required_order]
    if any(x < 0 for x in indices) or indices != sorted(indices):
        issues.append("matrix no longer uses receipt-first candidate materialization")

def check_topk_receipt_completeness(issues: list[str], matrix: str, runner: str) -> None:
    """Require receipt-derived KNN sufficiency; legacy res_ids stay attributes only.

    A real GTS receipt must be nonempty for a nonempty base and must materialize
    enough rows for an exact receipt-leaf scan.  The archive-owned res_ids/
    res_dis slots are preserved as bounded, receipt-attributable diagnostics;
    their non-sentinel cardinality must never become a second candidate/result
    completeness contract.
    """
    try:
        scope = topk_adapter(matrix)
        traversal_witness = cxx_function(
            runner, "WitnessJson witness_traversal(const FreshWitnessData& witness, bool native)",
            "native-result diagnostic witness serializer")
        slot_witness = cxx_function(
            runner, "WitnessJson witness_native_result_slots(",
            "native-result slot serializer")
    except AuditError as exc:
        issues.append(str(exc)); return
    primary = {
        "required-base contract": "const int required_base_results = std::min(k, runtime.base_count);",
        "raw required-count export": "output.native_result_required_count = required_base_results;",
        "nonempty-base/empty-receipt rejection":
            "if (required_base_results > 0 && fresh_header.ordered_leaf_pairs.empty())",
        "empty-receipt failure":
            "native GTS top-k receipt selected no leaf for a nonempty immutable base",
        "receipt export": "output.ordered_raw_leaf_pairs = fresh_header.ordered_leaf_pairs;",
        "receipt-row materialization":
            "materialize_receipt_leaf_rows(runtime, snapshot, output.visited_leaf_ids);",
        "receipt-row lower bound":
            "if (static_cast<int>(output.base_receipt_rows.size()) < required_base_results)",
        "receipt-row lower-bound failure":
            "native GTS receipt materialized fewer base rows than the requested top-k contract",
        "exact receipt scan":
            "output.base_results = scan_local_rows_exact(runtime, query_device, receipt_locals);",
        "exact-result lower bound":
            "if (static_cast<int>(output.base_results.size()) < required_base_results)",
        "exact-result lower-bound failure":
            "native GTS receipt exact scan cannot supply the requested base top-k",
    }
    attributes = {
        "raw slots are non-authoritative": "res_ids are deliberately not an input to this exact scan.",
        "no hidden raw completeness":
            "this buffer's non-sentinel count into a hidden completeness contract;",
        "unique native result set": "std::set<LocalRow> native_result_locals;",
        "native result loop": "for (int index = 0; index < qnum * k; ++index)",
        "raw id recording": "output.native_final_res_ids.push_back(local);",
        "raw distance recording": "output.native_final_res_distances.push_back(res_dis[index]);",
        "native sentinel filtering": "if (local == -1) continue;",
        "native result bounds": "if (local < 0 || local >= runtime.base_count)",
        "native bounds failure": "native top-k res_ids contains an invalid non-sentinel local row",
        "native receipt attribution":
            "if (receipt_local_set.find(local) == receipt_local_set.end())",
        "native attribution failure":
            "native top-k res_ids is not attributable to a receipt leaf payload",
        "native uniqueness insertion": "if (!native_result_locals.insert(local).second)",
        "native uniqueness failure":
            "native top-k res_ids contains a duplicate non-sentinel local row",
        "native completion diagnostic": "output.native_result_complete =",
        "native completion comparison":
            "static_cast<int>(native_result_locals.size()) >= required_base_results;",
        "native result cross-check": "output.native_final_res_ids_cross_checked = true;",
    }
    for label, token in {**primary, **attributes}.items():
        require_token(issues, scope, token, "top-k receipt/attribute contract " + label)
    positions = {label: scope.find(token) for label, token in {**primary, **attributes}.items()}
    if any(position < 0 for position in positions.values()):
        return
    if len(re.findall(r"\brequired_base_results\s*=", scope)) != 1:
        issues.append("top-k receipt contract does not define exactly one min(k, base_count) requirement")
    if not (positions["required-base contract"] <
            positions["raw required-count export"] <
            positions["nonempty-base/empty-receipt rejection"] <
            positions["empty-receipt failure"] <
            positions["receipt export"]):
        issues.append("top-k adapter can export a receipt before rejecting an empty nonempty-base receipt")
    if not (positions["receipt-row materialization"] <
            positions["receipt-row lower bound"] <
            positions["receipt-row lower-bound failure"] <
            positions["exact receipt scan"] <
            positions["exact-result lower bound"] <
            positions["exact-result lower-bound failure"] <
            positions["unique native result set"]):
        issues.append("top-k adapter does not gate receipt rows and exact receipt results before raw res_ids")
    native_loop = scope.find(attributes["native result loop"], positions["unique native result set"])
    if native_loop < 0 or not (positions["unique native result set"] < native_loop <
                               positions["raw id recording"] <
                               positions["raw distance recording"] <
                               positions["native sentinel filtering"] <
                               positions["native result bounds"] <
                               positions["native bounds failure"] <
                               positions["native receipt attribution"] <
                               positions["native attribution failure"] <
                               positions["native uniqueness insertion"] <
                               positions["native uniqueness failure"] <
                               positions["native completion diagnostic"] <
                               positions["native completion comparison"] <
                               positions["native result cross-check"]):
        issues.append("top-k adapter does not retain raw res_ids as checked receipt-attributable attributes")
    for forbidden in (
        "if (static_cast<int>(native_result_locals.size()) < required_base_results)",
        "native top-k res_ids cannot witness the requested immutable-base top-k",
    ):
        if forbidden in scope:
            issues.append("raw res_ids were promoted back into a top-k completeness gate: " + forbidden)
    if scope.count(primary["exact receipt scan"]) != 1 or scope.count("output.base_results =") != 1:
        issues.append("raw res_ids can no longer be proven absent from the sole receipt-derived base result")
    completion_at = positions["native completion diagnostic"]
    cross_at = positions["native result cross-check"]
    if "fail(" in scope[completion_at:cross_at]:
        issues.append("native_result_complete is treated as a fail-stop rather than an attribute-only diagnostic")
    for token in (
        "output.native_result_required_count = base.native_result_required_count;",
        "output.native_result_complete = base.native_result_complete;",
    ):
        require_token(issues, matrix, token, "native res_ids diagnostic export propagation")

    # The runner must serialize, rather than derive correctness from, the raw
    # slots.  The non-sentinel count is a visible diagnostic field, not a gate.
    for token in (
        "witness.native_result_ids.size() == witness.native_result_distances.size()",
        "std::size_t non_sentinel = 0;",
        "for (const LocalRow row : witness.native_result_ids) if (row != -1) ++non_sentinel;",
        '{"native_result_slot_count", witness_uint(witness.native_result_ids.size())}',
        '{"native_result_non_sentinel_count", witness_uint(non_sentinel)}',
        '{"native_result_required_count", witness_null()}',
        '{"native_result_complete", witness_null()}',
        '{"native_result_required_count", witness_int(witness.native_result_required_count)}',
        '{"native_result_complete", witness_bool(witness.native_result_complete)}',
        "witness.native_result_required_count >= 0",
        "static_cast<int>(witness.candidate_rows.size()) >= witness.native_result_required_count",
        "require(witness.native_result_complete ==",
        "(static_cast<int>(non_sentinel) >= witness.native_result_required_count),",
        "legacy native res_ids completeness diagnostic drift",
        '{"native_result_slots_sha256",', "hash_native_result_slots(witness.native_result_ids,",
        '{"native_result_slots",', "witness_native_result_slots(witness.native_result_ids,",
        '{"native_final_res_ids_cross_checked",',
    ):
        require_token(issues, traversal_witness, token, "runner raw-res_ids diagnostic serialization")
    for token in (
        "require(ids.size() == distances.size(), \"Stage-0 native result slot vector mismatch\")",
        '{"index", witness_uint(index)}, {"local_row", witness_int(ids[index])}',
        '{"distance_f32_bits", witness_string(f32_bits_hex(distances[index]))}',
    ):
        require_token(issues, slot_witness, token, "runner raw-res_ids slot serialization")
    if re.search(r"\bnon_sentinel\s*(?:<|<=|!=)\s*(?:t\.k|witness\.telemetry\.k)", traversal_witness):
        issues.append("runner promotes raw non-sentinel res_ids count into a primary KNN gate")
    compact_witness = re.sub(r"\s+", " ", traversal_witness)
    if (re.search(r"require\(\s*witness\.native_result_complete\s*(?:,|\)|&&|\|\|)", compact_witness)
            or re.search(r"witness\.native_result_complete\s*==\s*true", compact_witness)):
        issues.append("runner requires native_result_complete to be true instead of serializing its diagnostic value")
    for token in (
        "output.fresh_witness.native_result_required_count =",
        "export_data.native_result_required_count;",
        "output.fresh_witness.native_result_complete = export_data.native_result_complete;",
    ):
        require_token(issues, runner, token, "runner native res_ids diagnostic propagation")


def check_semantic_output_contract(issues: list[str], runner: str,
                                   validator: str, guard: str) -> None:
    """Bind candidate output to semantic-only fields, never v1b durations.

    The fixed v1b engine is retained solely as a hash-bound historical
    reference.  This check intentionally compares no candidate serializer with
    its timing-oriented predecessor; instead it statically verifies the new
    event/summary contract and the validator/guard rejection boundary.
    """
    try:
        event_writer = cxx_function(runner, "void write_semantic_observation(",
                                    "semantic engine event writer")
        dry_writer = cxx_function(runner, "void write_dry_run(",
                                  "semantic dry-run writer")
        semantic_query = cxx_function(runner, "Observation execute_semantic_query(",
                                      "semantic query executor")
        semantic_copy = cxx_function(runner, "Observation semantic_only_observation(",
                                     "semantic observation projection")
        run_scope = cxx_function(runner, "int run(const Args& args)",
                                 "semantic runner")
        semantic_records = python_function(validator, "validate_semantic_engine_records",
                                           "validator semantic record")
        semantic_summary = python_function(validator, "validate_semantic_engine_summary",
                                           "validator semantic summary")
        semantic_binding = python_function(validator, "validate_semantic_engine_binding",
                                           "validator semantic witness binding")
        comparison = python_function(validator, "compare_fixed_reference",
                                     "historical semantic comparator")
        validator_main = python_function(validator, "validate", "validator main")
        legacy_records = python_function(validator, "validate_legacy_control_records",
                                         "historical reference validator")
        legacy_keys, legacy_start, legacy_end = python_set_assignment(
            validator, "LEGACY_CONTROL_ENGINE_MEASUREMENT_KEYS", "historical reference")
        guard_output = python_function(guard, "verify_runner_outputs",
                                       "guard semantic output verifier")
        guard_command = python_function(guard, "runner_command", "guard runner command")
        guard_validator = python_function(guard, "run_validator", "guard validator runner")
    except AuditError as exc:
        issues.append(str(exc))
        return

    for label, scope, tokens in (
        ("semantic event writer", event_writer, (
            "fresh-allocation-telemetry-e1-v1", "semantic_observation",
            "semantic_pass", "schedule_phase", "schedule_phase_pass",
            "schedule_phase_slot", "merged_result_sha256", "merged",
        )),
        ("semantic dry-run writer", dry_writer, (
            "fresh-allocation-telemetry-e1-v1", "dry_run_semantic_plan",
            "PASS_CPU_ONLY_SEMANTIC_PLAN", "claim_scope",
            "semantic_fresh_allocation_control_only",
        )),
        ("semantic runner summary", run_scope, (
            "PASS_SEMANTIC_CONTROL", "claim_scope",
            "semantic_fresh_allocation_control_only", "initialization",
            "semantic", "no_performance_conclusion",
        )),
        ("validator semantic records", semantic_records, (
            "SEMANTIC_ENGINE_EVENT_KEYS", "SEMANTIC_ENGINE_SCHEMA",
            "SEMANTIC_ENGINE_RECORD", "semantic_pass", "schedule_phase",
        )),
        ("validator semantic summary", semantic_summary, (
            "SEMANTIC_ENGINE_SUMMARY_KEYS", "SEMANTIC_CLAIM_SCOPE",
            "no_performance_conclusion",
        )),
        ("validator semantic witness binding", semantic_binding, (
            "semantic_pass", "schedule_phase", "merged_result_sha256",
        )),
        ("guard semantic output verifier", guard_output, (
            "EVENT_KEYS", "SUMMARY_KEYS", "semantic_observation",
            "semantic-control", "PASS_SEMANTIC_CONTROL", "claim_scope",
            "api_host_ns", "timing_claim", "latency",
        )),
        ("guard semantic runner command", guard_command, ("--mode", "semantic-control")),
        ("guard semantic validator result", guard_validator, (
            "VALIDATOR_PASS", "claim_scope", "fixed_reference",
        )),
    ):
        for token in tokens:
            require_token(issues, scope, token, label)

    # The candidate serializers/projection/query path and every candidate-side
    # validator scope must remain free of duration fields.  The guard is
    # intentionally exempt: it contains the explicit runtime rejection test.
    forbidden = ("api_host_ns", "timing_claim", "latency")
    for label, scope in (
        ("semantic event writer", event_writer),
        ("semantic dry-run writer", dry_writer),
        ("semantic runner", run_scope),
        ("semantic query executor", semantic_query),
        ("semantic observation projection", semantic_copy),
        ("validator semantic records", semantic_records),
        ("validator semantic summary", semantic_summary),
        ("validator semantic witness binding", semantic_binding),
        ("historical semantic comparator", comparison),
        ("validator candidate acceptance", validator_main),
    ):
        for token in forbidden:
            if token in scope:
                issues.append(f"{label} retains prohibited timing field: {token}")

    # The guard must reject the fields as serialized output, not merely name
    # them in a comment or a dead key list.
    rejection = ('require("api_host_ns" not in serialized and "timing_claim" not in serialized and'
                 '\n            "latency" not in serialized,')
    if rejection not in guard_output:
        issues.append("guard does not reject all prohibited semantic-output timing fields")

    # Historical v1b fields are permitted only in the explicit legacy input
    # schema.  They cannot flow through the semantic candidate checks or the
    # field-by-field comparator used for the fixed reference.
    for token in ("api_host_ns", "timing_claim", "post_api_external_delta_merge_excluded_from_api_host_ns"):
        require_token(issues, legacy_keys, token, "historical v1b-only key schema")
    require_token(issues, legacy_records, "LEGACY_CONTROL_ENGINE_MEASUREMENT_KEYS",
                  "historical v1b-only record validator")
    residual = validator[:legacy_start] + validator[legacy_end:]
    for token in ("api_host_ns", "timing_claim"):
        if token in residual:
            issues.append(f"historical timing field escapes legacy key schema: {token}")
    for token in ("api_host_ns", "timing_claim", "latency"):
        if token in comparison:
            issues.append(f"historical comparator derives candidate claim from timing field: {token}")
    for token in (
        "hash-bound historical reference",
        "duration fields are never aggregated or used by candidate acceptance.",
        "validate_semantic_engine_records", "validate_semantic_engine_summary",
        "validate_legacy_control_records", "compare_fixed_reference",
    ):
        require_token(issues, validator, token, "semantic-only historical reference boundary")
    candidate_at = validator_main.find("validate_semantic_engine_records")
    summary_at = validator_main.find("validate_semantic_engine_summary")
    legacy_at = validator_main.find("validate_legacy_control_records")
    compare_at = validator_main.find("compare_fixed_reference")
    if min(candidate_at, summary_at, legacy_at, compare_at) < 0 or not (
            candidate_at < summary_at < legacy_at < compare_at):
        issues.append("validator does not validate candidate semantics before historical comparison")

def check_runner(issues: list[str], candidate: str) -> None:
    for token in (
        "fresh_allocation_semantic_control", "--mode", "semantic-control", "dry-run",
        "write_semantic_observation", "execute_semantic_query", "semantic_only_observation",
        "fresh-allocation-telemetry-e1-v1", "semantic_observation",
        "PASS_SEMANTIC_CONTROL", "PASS_CPU_ONLY_SEMANTIC_PLAN",
        "semantic_fresh_allocation_control_only", "initialization", "no_performance_conclusion",
    ):
        require_token(issues, candidate, token, "pure-semantic runner contract")
    # The runner has no legitimate reason to retain the historical duration
    # fields anywhere: the witness still proves lifecycle/schedule semantics,
    # while candidate engine JSON is intentionally timing-free.
    for token in ("api_host_ns", "timing_claim", "latency"):
        if token in candidate:
            issues.append(f"runner retains prohibited timing output token: {token}")
    for token in (
        "--witness", "--source-manifest", "--run-id", "--fresh-alloc-guard-nonce",
        "SAFE_C1_FRESH_ALLOC_TELEMETRY_GUARD_NONCE",
        "fair-safe-c1-stage0-fresh-allocation-witness-v1",
        "run_start", "query_witness", "run_end", "prev_record_sha256", "record_sha256",
        "source_manifest_sha256", "full_source_closure_sha256",
        "fresh_alloc", "native_result_slots", "candidate_rows", "ordered_leaf_pairs", "traversal_steps",
    ):
        require_token(issues, candidate, token, "Stage-0 runner contract")
    try:
        target_check = cxx_function(candidate, "void validate_output_targets_before_cuda(const Args& args)",
                                    "candidate output target validation")
        if "args.witness" not in target_check:
            issues.append("witness target is not checked before CUDA")
        if "read_text(args.source_manifest)" not in target_check:
            issues.append("source manifest is not read in the pre-CUDA target validation")
    except AuditError as exc:
        issues.append(str(exc))
    for token in (
        "? engine->query_knn(item.query_id, query, k)",
        ": engine->query_range(item.query_id, query, std::numeric_limits<DistanceSq>::max());",
    ):
        require_token(issues, candidate, token, "v1b public API execution")
    try:
        phase = cxx_function(candidate, "void run_phase(NativeSafeC1Matrix* engine,",
                             "global ABBA phase scheduler")
        parse = cxx_function(candidate, "Args parse_args(int argc, char** argv)",
                             "runner CLI parser")
        unique_json = cxx_function(candidate,
            "std::optional<std::size_t> unique_json_member_value_offset(",
            "strict unique JSON-member parser")
        json_field = cxx_function(candidate,
            "std::optional<std::string> json_string_field(",
            "strict JSON-string field parser")
        artifact_parser = cxx_function(candidate,
            "std::string manifest_artifact_sha_or_fail(",
            "strict manifest artifact parser")
    except AuditError as exc:
        issues.append(str(exc)); return
    for token in (
        "const int global_pass = global_pass_base + pass;",
        "const int pass_slot_base = global_slot_base + pass * slots_per_pass;",
        "const bool native_first = ((global_pass + item.ordinal) % 2) == 0;",
        "first.phase_pass = global_pass;", "second.phase_pass = global_pass;",
        "first.phase_slot = pass_slot_base + item.ordinal * 2;",
        "second.phase_slot = pass_slot_base + item.ordinal * 2 + 1;",
    ):
        require_token(issues, phase, token, "global ABBA scheduler")
    if "((pass + item.ordinal) % 2)" in phase:
        issues.append("runner retains a phase-local ABBA schedule instead of the global ABBA schedule")
    require_token(issues, candidate,
                  "ABBA_native_first_if_(global_phase_pass_plus_case_ordinal)_mod_2_is_0",
                  "witness global ABBA ordering declaration")
    for token in (
        "std::set<std::string> seen_flags;",
        'require(seen_flags.insert(flag).second, "duplicate option: " + flag);',
    ):
        require_token(issues, parse, token, "runner duplicate CLI rejection")
    for token in (
        "if (text.find('\\\\') != std::string::npos) return std::nullopt;",
        "if (value_offset.has_value()) return std::nullopt;",
        "return value_offset;",
    ):
        require_token(issues, unique_json, token, "strict unique JSON-member parser")
    require_token(issues, json_field, "unique_json_member_value_offset(text, key);",
                  "strict JSON-string field parser")
    for token in (
        "unique_json_member_value_offset(text, artifact);",
        "source manifest omits or duplicates artifact object: ",
        "json_object_end_without_escapes(text, *object_begin);",
    ):
        require_token(issues, artifact_parser, token, "strict manifest artifact parser")
    if "json_string_field(text, artifact)" in artifact_parser:
        issues.append("manifest artifact parser reverted to first-match string parsing")

def check_receipt_leaf_pairs_single_source(issues: list[str], matrix: str) -> None:
    """Require adapter leaf-pair export to use one post-traversal telemetry copy."""
    try:
        scope = topk_adapter(matrix)
    except AuditError as exc:
        issues.append(str(exc)); return
    if "safe_c1_last_visited_leaf_pairs" in scope:
        issues.append("adapter exports leaf pairs from global receipt vector instead of telemetry snapshot")
    snapshots = list(re.finditer(
        r"\b(?:const\s+)?SafeC1FreshAllocTelemetry\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        r"safe_c1_last_fresh_alloc_telemetry\s*;", scope))
    if len(snapshots) != 1:
        issues.append("adapter must take exactly one value telemetry snapshot after traversal")
        return
    snapshot = snapshots[0]
    name = snapshot.group(1)
    traversal = scope.find("searchIndexKnnV2(")
    sync = scope.find("G3_CUDA(cudaGetLastError());", traversal)
    if traversal < 0 or sync < 0 or snapshot.start() < sync:
        issues.append("telemetry snapshot is not taken after the real traversal completes")
    for token in (
        f"output.ordered_raw_leaf_pairs = {name}.ordered_leaf_pairs;",
        f"static_cast<int>({name}.ordered_leaf_pairs.size())",
        f"{name}.ordered_leaf_pairs",
    ):
        require_token(issues, scope, token, "telemetry-snapshot leaf-pair export")
    # The adapter may append its final res_dis/local_result_ids free events to
    # mutable telemetry after this leaf-only snapshot; leaf exports above must
    # remain exclusively sourced from the immutable value copy.


def check_direct_event_serialization(issues: list[str], runner: str, validator: str) -> None:
    """Ensure the witness serializes the observed runtime vector, not a summary."""
    try:
        runner_ledger = cxx_function(runner, "void require_direct_event_ledger(",
                                     "runner direct-event ledger assertion")
        direct_witness = cxx_function(runner, "WitnessJson witness_direct_events(",
                                      "runner direct-event witness serializer")
        fresh_witness = cxx_function(runner, "WitnessJson witness_fresh_alloc(",
                                     "runner fresh-allocation witness serializer")
        rank_helper = cxx_function(runner, "int direct_event_rank(",
                                   "runner direct-event rank helper")
    except AuditError as exc:
        issues.append(str(exc)); return
    for token in (
        "std::array<ExpectedEvent, 10>", "t.direct_allocation_events.size()",
        "SAFE_C1_FRESH_ALLOC_LOCAL_RESULT_IDS", "SAFE_C1_FRESH_ALLOC_RES_DIS",
        "SAFE_C1_FRESH_ALLOC_SIZE_LIST", "SAFE_C1_FRESH_ALLOC_DISK",
        "SAFE_C1_FRESH_ALLOC_P_LIST_K", "SAFE_C1_FRESH_ALLOC_DIRECT_ALLOC",
        "SAFE_C1_FRESH_ALLOC_DIRECT_FREE", "actual.requested_bytes == expected[index].bytes",
    ):
        require_token(issues, runner_ledger, token, "runner direct-event ledger assertion")
    for token in (
        "require_direct_event_ledger(t);", "t.direct_allocation_events.size()",
        "for (std::size_t index = 0; index < t.direct_allocation_events.size(); ++index)",
        "const SafeC1FreshAllocDirectCudaEvent& event", "fresh_direct_slot_name(event.slot)",
        "fresh_direct_operation_name(event.operation)", "requested_bytes",
    ):
        require_token(issues, direct_witness, token, "runner direct-event serializer")
    if "direct_allocation_events" not in rank_helper or "require_direct_event_ledger(t);" not in rank_helper:
        issues.append("runner allocation index summary is not derived from direct event ledger")
    direct_emit = fresh_witness.find('{"direct_events", witness_direct_events(t)}')
    native_guard = fresh_witness.find("if (!native)")
    if direct_emit < 0 or native_guard < 0 or direct_emit < native_guard:
        issues.append("native fresh-allocation witness does not emit observed direct event timeline")
    scratch_key = '"entry_named_global_scratch_ptrs_null"'
    native_data = fresh_witness.find("const SafeC1FreshAllocTelemetry& t", native_guard)
    fallback_scratch = fresh_witness.find(scratch_key, native_guard, native_data)
    fallback_null = fresh_witness.find("witness_null()", fallback_scratch, native_data)
    native_scratch = fresh_witness.find(scratch_key, native_data)
    native_true = fresh_witness.find("witness_bool(t.entry_named_global_scratch_ptrs_null)", native_scratch)
    if (native_data < 0 or fallback_scratch < native_guard or fallback_null < fallback_scratch or
            native_scratch < native_data or native_true < native_scratch):
        issues.append("fresh-allocation witness does not bind named scratch-pointer freshness (native true/fallback null)")

    for token in (
        "DIRECT_ALLOCATION_EVENT_KEYS", '"index", "slot", "operation", "requested_bytes"',
        '"direct_events"', "entry_named_global_scratch_ptrs_null",
        'fresh.get("entry_named_global_scratch_ptrs_null") is True',
        '"entry_named_global_scratch_ptrs_null": None', "immediate direct allocation event count",
        "expected_events", "free_order = (\"p_list_k\", \"size_list\", \"disk\", \"res_dis\", \"local_result_ids\")",
        "direct event/ledger byte binding", "complete immediate direct allocation lifecycle",
    ):
        if token and token not in validator:
            issues.append(f"validator lacks direct-event evidence check: {token}")
    for token in ("import pynvml", "import torch", "import cupy", "nvidia-smi", "subprocess.run("):
        if token in validator:
            issues.append(f"validator contains prohibited runtime/GPU token: {token}")


def check_added_policy(issues: list[str], patches: list[str]) -> None:
    added = "\n".join(added_lines(patch) for patch in patches).lower()
    for token in BANNED_ADDED_TOKENS:
        if token.lower() in added:
            issues.append(f"forbidden async/stream/event/reuse token added: {token}")

def check_compile_helper(issues: list[str], text: str) -> None:
    for token in (
        "fresh_alloc_telemetry_e1_runner.cu", "-I\"$DIAG/src\"",
        "-I\"$ROOT/reference/include\"", "-c", "-arch=$ARCH", "nvcc",
        "set -euo pipefail",
    ):
        require_token(issues, text, token, "compile helper")
    for token in ("--run", "nvidia-smi", "cuda-memcheck", "nohup", "CUDA_VISIBLE_DEVICES"):
        if token in text:
            issues.append(f"compile helper contains launch-like token: {token}")

def check_validator(issues: list[str], text: str) -> None:
    for token in (
        'SEMANTIC_ENGINE_SCHEMA = "fresh-allocation-telemetry-e1-v1"',
        'SEMANTIC_ENGINE_RECORD = "semantic_observation"',
        'SEMANTIC_ENGINE_MODE = "semantic-control"',
        'SEMANTIC_ENGINE_STATUS = "PASS_SEMANTIC_CONTROL"',
        'SEMANTIC_CLAIM_SCOPE = "semantic_fresh_allocation_control_only"',
        'MERGED_HASH_DOMAIN = "fair-safe-c1-semantic-control-e1-merged-result-v1"',
        "validate_semantic_engine_records", "validate_semantic_engine_summary",
        "validate_semantic_engine_binding", "compare_fixed_reference",
        "semantic control makes no performance conclusion",
    ):
        require_token(issues, text, token, "validator semantic contract")
    try:
        event_keys, _, _ = python_set_assignment(text, "SEMANTIC_ENGINE_EVENT_KEYS",
                                                  "semantic event")
        summary_keys, _, _ = python_set_assignment(text, "SEMANTIC_ENGINE_SUMMARY_KEYS",
                                                    "semantic summary")
    except AuditError as exc:
        issues.append(str(exc))
    else:
        for token in ("semantic_pass", "schedule_phase", "merged_result_sha256", "merged"):
            require_token(issues, event_keys, token, "validator semantic event keys")
        for token in ("claim_scope", "limitations", "fresh_witness"):
            require_token(issues, summary_keys, token, "validator semantic summary keys")
        for token in ("api_host_ns", "timing_claim", "latency"):
            if token in event_keys or token in summary_keys:
                issues.append(f"validator semantic output key set retains timing field: {token}")
    for token in (
        "fair-safe-c1-stage0-fresh-allocation-witness-v1", "fresh_alloc",
        "prev_record_sha256", "record_sha256", "traversal_steps", "cudaMalloc", "cudaMemGetInfo",
    ):
        require_token(issues, text, token, "validator")
    for token in ("import pynvml", "import torch", "import cupy", "nvidia-smi", "subprocess.run("):
        if token in text:
            issues.append(f"validator contains prohibited runtime/GPU token: {token}")

def check_guard_manifest_binding(issues: list[str], guard: str) -> None:
    """Require the launcher to bind its own source bytes into provenance.

    The guard is executable source that controls eventual GPU launch.  It must
    be both an artifact and a required closure member.  The static auditor is
    deliberately not included, because including it would make the manifest
    depend on the post-publication audit-script hash.
    """
    for token in (
        'EVENT_SCHEMA = "fresh-allocation-telemetry-e1-v1"',
        'SUMMARY_SCHEMA = "fresh-allocation-telemetry-e1-v1"',
        'CLAIM_SCOPE = "semantic_fresh_allocation_control_only"',
        'VALIDATOR_PASS = "PASS_SEMANTIC_CONTROL_VALIDATED"',
        "GUARD = Path(__file__).resolve()",
        '"guard": (GUARD, True),',
        '"guard_sha256": "guard",',
        "def verify_manifest()",
        "for name, (path, executable) in SOURCE_ARTIFACTS.items()",
        "str(GUARD)",
        '"validator", "guard"):',
        'build.get(name + "_sha256") == source_hashes[name]',
        "required_sources.issubset(closure_paths)",
    ):
        require_token(issues, guard, token, "guard manifest self-binding")
    source_map = re.search(r"SOURCE_ARTIFACTS[^=]*=\s*\{(.*?)^\}", guard,
                           flags=re.MULTILINE | re.DOTALL)
    if source_map is None or not re.search(r'"guard"\s*:\s*\(\s*GUARD\s*,\s*True\s*\)',
                                            source_map.group(1)):
        issues.append("guard SOURCE_ARTIFACTS does not declare executable guard artifact")
    verify_at = guard.find("def verify_manifest()")
    next_def = re.search(r"^def [A-Za-z_][A-Za-z0-9_]*\(", guard[verify_at + 1:], re.MULTILINE)
    verify_scope = guard[verify_at: verify_at + 1 + next_def.start()] if verify_at >= 0 and next_def else ""
    if not verify_scope or "str(GUARD)" not in verify_scope:
        issues.append("guard verify_manifest does not require the guard in source_closure")
    else:
        for token in ("event_schema", "summary_schema", "claim_scope", "CLAIM_SCOPE"):
            require_token(issues, verify_scope, token, "guard semantic manifest binding")
    try:
        command_scope = python_function(guard, "runner_command", "guard runner command")
        output_scope = python_function(guard, "verify_runner_outputs", "guard output verifier")
        validator_scope = python_function(guard, "run_validator", "guard validator runner")
    except AuditError as exc:
        issues.append(str(exc))
        return
    for token in ("--mode", "semantic-control"):
        require_token(issues, command_scope, token, "guard semantic launch mode")
    for token in (
        "EVENT_KEYS", "SUMMARY_KEYS", "semantic_observation", "PASS_SEMANTIC_CONTROL",
        "claim_scope", "api_host_ns", "timing_claim", "latency",
    ):
        require_token(issues, output_scope, token, "guard semantic output boundary")
    for token in ("VALIDATOR_PASS", "claim_scope", "fixed_reference"):
        require_token(issues, validator_scope, token, "guard semantic validator boundary")


def candidate_patch(control_path: Path, candidate_path: Path, control: str, candidate: str) -> str:
    return "".join(difflib.unified_diff(
        control.splitlines(),
        candidate.splitlines(),
        fromfile="a/" + str(control_path.relative_to(ROOT)),
        tofile="b/" + str(candidate_path.relative_to(ROOT)),
        lineterm="\n",
    ))


def closure_members() -> tuple[Path, ...]:
    return (
        CANDIDATE_RUNNER,
        CANDIDATE_HEADER,
        CANDIDATE_MATRIX,
        COMPILE_HELPER,
        VALIDATOR,
        GUARD,
        ROOT / "reference/include/tree.cuh",
        ROOT / "reference/include/file.cuh",
        ROOT / "reference/include/config.cuh",
        ROOT / "reference/include/mlp_constant.cuh",
        ROOT / "reference/include/residual_pruning.cuh",
    )


def check_semantic_source_closure(issues: list[str], hashes: dict[Path, str]) -> None:
    """Require the semantic runner/validator/guard in the reviewed closure.

    This is deliberately static: it authenticates source bytes and closure
    membership without compiling, importing, or launching any CUDA-facing
    component.
    """
    closure = closure_members()
    if len(set(closure)) != len(closure):
        issues.append("source closure has duplicate members")
    if Path(__file__).resolve() in closure:
        issues.append("static auditor must not be included in its source closure")
    for path, expected in EXPECTED_SEMANTIC_SOURCE_SHA256.items():
        if path not in closure:
            issues.append("pure-semantic source missing from closure: " + str(path))
        actual = hashes.get(path)
        if actual != expected:
            issues.append("pure-semantic source hash drift: " + str(path))


def source_closure_digest(entries: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for entry in entries:
        path = Path(entry["path"])
        lines.append(str(path.relative_to(ROOT)) + "\t" + entry["sha256"])
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode("ascii")).hexdigest()


def atomic_write(path: Path, payload: bytes, replace: bool) -> None:
    private_dir(path.parent, "provenance directory")
    if path.exists() and not replace:
        raise AuditError("refuse overwrite existing artifact without --replace: " + str(path))
    if path.exists():
        private_file(path, "existing provenance artifact")
    temp = path.with_name("." + path.name + ".audit-tmp")
    if temp.exists() or temp.is_symlink():
        raise AuditError("stale/unsafe provenance temporary: " + str(temp))
    fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except Exception:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def render_manifest(hashes: dict[Path, str], reviewed_diff_sha256: str) -> dict[str, object]:
    artifacts = {
        "runner_source": {"path": str(CANDIDATE_RUNNER), "sha256": hashes[CANDIDATE_RUNNER]},
        "header_source": {"path": str(CANDIDATE_HEADER), "sha256": hashes[CANDIDATE_HEADER]},
        "matrix_source": {"path": str(CANDIDATE_MATRIX), "sha256": hashes[CANDIDATE_MATRIX]},
        "compile_helper": {"path": str(COMPILE_HELPER), "sha256": hashes[COMPILE_HELPER]},
        "validator": {"path": str(VALIDATOR), "sha256": hashes[VALIDATOR]},
        "guard": {"path": str(GUARD), "sha256": hashes[GUARD]},
        "reviewed_diff": {"path": str(REVIEWED_DIFF), "sha256": reviewed_diff_sha256},
    }
    closure = [{"path": str(path), "sha256": hashes[path]} for path in closure_members()]
    return {
        "schema": MANIFEST_SCHEMA,
        "diagnostic_variant": VARIANT,
        "publication_eligible": False,
        "claim_scope": CLAIM_SCOPE,
        "event_schema": EVENT_SCHEMA,
        "summary_schema": SUMMARY_SCHEMA,
        "control_artifacts": {
            "run_dir": str(CONTROL_RUN),
            **{name + "_sha256": digest for name, digest in CONTROL_RUN_SHA256.items()},
        },
        "artifacts": artifacts,
        "source_closure": closure,
        "full_source_closure_sha256": source_closure_digest(closure),
    }


def audit() -> tuple[dict[str, object], str, dict[Path, str]]:
    issues: list[str] = []
    try:
        for directory, label in (
            (DIAG, "diagnostic root"),
            (DIAG / "src", "diagnostic source directory"),
            (DIAG / "runner", "diagnostic runner directory"),
            (DIAG / "tools", "diagnostic tools directory"),
            (PROVENANCE, "diagnostic provenance directory"),
        ):
            private_dir(directory, label)
        check_control(issues)
        texts: dict[Path, str] = {}
        hashes: dict[Path, str] = {}
        for path, label, executable in (
            (CANDIDATE_HEADER, "candidate header", False),
            (CANDIDATE_MATRIX, "candidate matrix", False),
            (CANDIDATE_RUNNER, "candidate runner", False),
            (COMPILE_HELPER, "compile helper", True),
            (VALIDATOR, "CPU-only validator", True),
            (GUARD, "launch guard", True),
        ):
            text, digest = text_and_hash(path, label, executable)
            texts[path], hashes[path] = text, digest
        for path in closure_members()[6:]:
            private_file(path, "immutable reference include")
            hashes[path] = sha256_file(path)

        controls: dict[Path, str] = {}
        for path in (CONTROL_HEADER, CONTROL_MATRIX, CONTROL_RUNNER):
            controls[path], _ = text_and_hash(path, "immutable v1b source")
        patches = [
            candidate_patch(CONTROL_HEADER, CANDIDATE_HEADER,
                            controls[CONTROL_HEADER], texts[CANDIDATE_HEADER]),
            candidate_patch(CONTROL_MATRIX, CANDIDATE_MATRIX,
                            controls[CONTROL_MATRIX], texts[CANDIDATE_MATRIX]),
            candidate_patch(CONTROL_RUNNER, CANDIDATE_RUNNER,
                            controls[CONTROL_RUNNER], texts[CANDIDATE_RUNNER]),
        ]
        check_added_policy(issues, patches)
        check_header(issues, controls[CONTROL_HEADER], texts[CANDIDATE_HEADER])
        check_matrix(issues, controls[CONTROL_MATRIX], texts[CANDIDATE_MATRIX])
        check_direct_cuda_event_ledger(issues, texts[CANDIDATE_HEADER], texts[CANDIDATE_MATRIX])
        check_fresh_alloc_exit_and_failstop(issues, texts[CANDIDATE_HEADER], texts[CANDIDATE_MATRIX])
        check_receipt_leaf_pairs_single_source(issues, texts[CANDIDATE_MATRIX])
        check_topk_receipt_completeness(issues, texts[CANDIDATE_MATRIX], texts[CANDIDATE_RUNNER])
        check_semantic_source_closure(issues, hashes)
        check_semantic_output_contract(issues, texts[CANDIDATE_RUNNER], texts[VALIDATOR], texts[GUARD])
        check_runner(issues, texts[CANDIDATE_RUNNER])
        check_compile_helper(issues, texts[COMPILE_HELPER])
        check_validator(issues, texts[VALIDATOR])
        check_guard_manifest_binding(issues, texts[GUARD])
        check_direct_event_serialization(issues, texts[CANDIDATE_RUNNER], texts[VALIDATOR])
        for token in ("#include \"g3_safe_search_v2_fresh_alloc_telemetry.cuh\"",):
            require_token(issues, texts[CANDIDATE_MATRIX], token, "candidate include wiring")
        for token in ("g3_safe_c1_native_matrix_fresh_alloc_telemetry.cu",):
            require_token(issues, texts[CANDIDATE_RUNNER], token, "candidate include wiring")

        # Detect concurrent editing: the review must bind exactly the bytes read.
        for path, digest in hashes.items():
            if sha256_file(path) != digest:
                issues.append("source changed during static audit: " + str(path))
        control_after: list[str] = []
        check_control(control_after)
        issues.extend("control changed during static audit: " + issue for issue in control_after)
    except (AuditError, OSError) as exc:
        issues.append(str(exc))
        hashes = {}
        patches = []

    reviewed = "".join(patches)
    closure_entries = ([{"path": str(path), "sha256": hashes[path]}
                        for path in closure_members()]
                       if all(path in hashes for path in closure_members()) else [])
    result: dict[str, object] = {
        "schema": "fair-safe-c1-fresh-allocation-telemetry-source-static-audit-v1",
        "diagnostic_variant": VARIANT,
        "static_only": True,
        "event_schema": EVENT_SCHEMA,
        "summary_schema": SUMMARY_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "historical_reference_scope": "hash_bound_semantic_fields_only",
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "reviewed_diff_sha256": hashlib.sha256(reviewed.encode("utf-8")).hexdigest() if reviewed else None,
        "full_source_closure_sha256": (source_closure_digest(closure_entries)
                                        if closure_entries else None),
        "semantic_source_sha256": {
            "runner": hashes.get(CANDIDATE_RUNNER),
            "validator": hashes.get(VALIDATOR),
            "guard": hashes.get(GUARD),
        },
        "reviewed_paths": [str(CANDIDATE_HEADER), str(CANDIDATE_MATRIX), str(CANDIDATE_RUNNER)],
        "source_closure_excludes": [
            str(MANIFEST), str(REVIEWED_DIFF), str(DIAG / "provenance/fresh_alloc_telemetry_static_audit.json"),
            str(DIAG / "build"), str(DIAG / "bin"),
            str(DIAG / "tools/audit_fresh_alloc_telemetry_static.py"),
        ],
    }
    return result, reviewed, hashes


def main() -> int:
    parser = argparse.ArgumentParser(description="Static-only fresh-allocation provenance audit")
    parser.add_argument("--write", action="store_true", help="write reviewed diff and source manifest only on PASS")
    parser.add_argument("--replace", action="store_true", help="replace existing generated provenance artifacts (requires --write)")
    args = parser.parse_args()
    if args.replace and not args.write:
        parser.error("--replace requires --write")

    result, reviewed, hashes = audit()
    if result["status"] != "PASS":
        print(json.dumps(result, sort_keys=True, indent=2))
        return 1
    if args.write:
        reviewed_bytes = reviewed.encode("utf-8")
        manifest = render_manifest(hashes, hashlib.sha256(reviewed_bytes).hexdigest())
        manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        # Recheck once immediately before publication to make the two provenance
        # artifacts a binding of the exact reviewed source closure.
        for path, digest in hashes.items():
            if sha256_file(path) != digest:
                print(json.dumps({**result, "status": "FAIL", "issues": ["source changed before provenance write: " + str(path)]}, sort_keys=True, indent=2))
                return 1
        control_before_write: list[str] = []
        check_control(control_before_write)
        if control_before_write:
            print(json.dumps({**result, "status": "FAIL", "issues": ["control changed before provenance write: " + issue for issue in control_before_write]}, sort_keys=True, indent=2))
            return 1
        try:
            atomic_write(REVIEWED_DIFF, reviewed_bytes, args.replace)
            atomic_write(MANIFEST, manifest_bytes, args.replace)
        except AuditError as exc:
            print(json.dumps({**result, "status": "FAIL", "issues": [str(exc)]}, sort_keys=True, indent=2))
            return 1
        result["reviewed_diff"] = str(REVIEWED_DIFF)
        result["source_manifest"] = str(MANIFEST)
        result["source_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
