#!/usr/bin/python3.12
"""CPU-only validator for the Safe-C1 Stage-0 fresh-allocation telemetry.

This validator intentionally imports neither CUDA nor the runner.  It validates:

* the sealed E1 bundle and the original latency runner's v1b engine.jsonl/summary.json;
* the full warmup+measured ABBA invocation schedule in fresh_alloc_witness.jsonl;
* the named direct cudaMalloc/cudaMallocManaged lifecycle of the five GTS
  vector-top-k scratch allocations, including its inline immediate-event ledger;
* the current 4-GiB-or-half-free-memory p_list_k policy;
* witness-chain integrity plus full raw-array hash-preimage and structural checks; and
* measured-witness binding to the unchanged v1b engine output.

The sidecar deliberately inlines the bounded per-query traversal, receipt, and
native-result diagnostic arrays.  This lets the CPU-only validator recompute
all Stage-0 commitments and check their structural relation to the sealed E1
bundle.  It still cannot prove physical page uniqueness or observe allocations
inside the CUDA driver/Thrust implementation; those limits are explicit in the
output.

Wire contract (v1)
==================

The sidecar has exactly:

    run_start
    544 x query_witness    # 1 warmup + 1 measured, 136 cases, N/F per case
    run_end

All records use schema ``fair-safe-c1-stage0-fresh-allocation-witness-v1``.
Record-chain hashes use the tagged canonical value encoding implemented by
``canonical_value`` below, not an implementation-dependent JSON pretty-printer.
Per-array hashes use the fixed domain-separated line encodings documented in
the source manifest.  The runner and this validator must agree exactly on this
contract; changing a field name or a hash domain requires a schema bump.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import re
import stat
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/fair_dynamic_knn_safe_c1_v1")
DIAG = ROOT / "diagnostics/fresh_alloc_telemetry_v1"
BUNDLE_PATH = ROOT / "inputs/e1_frozen_base_knn_projection_v1"
DEFAULT_MANIFEST = DIAG / "provenance/fresh_alloc_telemetry_source_manifest.json"

# Candidate semantic engine-output contract.
SEMANTIC_ENGINE_SCHEMA = "fresh-allocation-telemetry-e1-v1"
SEMANTIC_ENGINE_RECORD = "semantic_observation"
SEMANTIC_ENGINE_MODE = "semantic-control"
SEMANTIC_ENGINE_STATUS = "PASS_SEMANTIC_CONTROL"
SEMANTIC_CLAIM_SCOPE = "semantic_fresh_allocation_control_only"
NATIVE_PATH = "real_gts_vector_topk_receipt_all_native_leaf_rows"
FALLBACK_PATH = "exact_full_immutable_base_range_fallback_no_gts_receipt"
MERGED_HASH_DOMAIN = "fair-safe-c1-semantic-control-e1-merged-result-v1"

# The immutable v1b run is a hash-bound historical reference.  Its old
# duration fields are never aggregated or used by candidate acceptance.
LEGACY_CONTROL_ENGINE_SCHEMA = "fair-safe-c1-latency-tradeoff-e1-v1"
LEGACY_CONTROL_ENGINE_RECORD = "measurement"
LEGACY_MERGED_HASH_DOMAIN = "fair-safe-c1-latency-tradeoff-e1-merged-result-v1"

# New Stage-0 sidecar contract.
WITNESS_SCHEMA = "fair-safe-c1-stage0-fresh-allocation-witness-v1"
VARIANT = "fresh_allocation_telemetry_control"
MANIFEST_SCHEMA = "fair-safe-c1-fresh-allocation-telemetry-source-manifest-v1"
RUN_END_STATUS = "PASS_STAGE0_FRESH_ALLOCATION_WITNESS"
VALIDATOR_PASS_STATUS = "PASS_SEMANTIC_CONTROL_VALIDATED"
VALIDATOR_SELFCHECK_STATUS = "PASS_CPU_ONLY_STATIC_SELFCHECK"
CONTROL_RUN = ROOT / "runs/safe-c1-tradeoff-pilot-v1b-20260730"
CONTROL_HASHES = {
    "engine.jsonl": "4a32720f49ac49bed6bd6f2755532231ea8a04e5d3c797e342e56ef22cb35f11",
    "summary.json": "5d9f197cf7297cb970a69f8a5fdb7255efad9d06c37adac343377af5a185c103",
}
GUARD_RECEIPT_SCHEMA = "fresh-allocation-telemetry-guard-v1"
GUARD_PENDING_STATUSES = {"RUNNER_EXITED_PENDING_VALIDATION"}
GUARD_SCRIPT = DIAG / "tools/run_safe_c1_fresh_alloc_telemetry_guarded.py"
CHAIN_DOMAIN = b"fair-safe-c1-stage0/witness-record/v1\n"
FIRST_WITNESSED_NATIVE_ALLOCATION_EPOCH = 1

# Hash domains for the inline raw arrays.  They are declared here so the runner,
# source manifest, comparator, and validator cannot silently use generic hashes.
HASH_DOMAINS = {
    "stack_schedule": "fair-safe-c1-stage0/stack-schedule/v1",
    "size_list_write": "fair-safe-c1-stage0/size-list-writes/v1",
    "traversal_steps": "fair-safe-c1-stage0/traversal-steps/v1",
    "ordered_leaf_pair": "fair-safe-c1-stage0/ordered-leaf-pairs/v1",
    "unique_leaf_set": "fair-safe-c1-stage0/unique-leaf-set/v1",
    "receipt_span": "fair-safe-c1-stage0/receipt-spans/v1",
    "ordered_candidate_row": "fair-safe-c1-stage0/receipt-candidate-rows/v1",
    "candidate_stable_set": "fair-safe-c1-stage0/candidate-stable-set/v1",
    "native_result_slot": "fair-safe-c1-stage0/native-result-slots/v1",
    "exact_candidate_stable_distance": "fair-safe-c1-stage0/exact-candidate-stable-distance/v1",
    "api_result_stable_distance": "fair-safe-c1-stage0/api-result-stable-distance/v1",
}

# Sealed E1 input contract copied from the independent v1b validator.
ADMISSION_SCHEMA = "e1-frozen-base-knn-projection-admission-v2"
TRACE_MAGIC = b"E1GTRC01"
TRACE_HEADER = struct.Struct("<8sI6IfQ")
TRACE_EVENT = struct.Struct("<IB3xi")
INSERT = 1
KNN = 3
INPUT_HASHES = {
    "manifest.json": "68dbf15788793a828c8a9503d11304da576acbdca2f609dd0f8799ae39cd9a4c",
    "metadata.json": "2f515a5cf3bef61d6084f4c0d90075ee16cb4e365971b75102a24bdbbc588798",
    "trace.e1gtrc": "9402c609710fc9076f46653dd5bf30527013e158c63b4576dd665c27f9973ae5",
    "pool.i16": "899adaa59b265ee788841f1a48b667b7da39df166972ba9568c4e94727b31170",
    "queries.i16": "18e0ebbe8ddcdcf6e2312e1e48310111a4d96c1fcb752622d1e9ec9508c21c1e",
    "stable_id_to_pool_row.i32": "93710cce11c994b6b1934713842c93cfcec76a3563fc47574abf419137f4c5c8",
    "initial_base_stable_ids.i32": "6b0751ba5e64fc9c13ddfb44778fa7d6a1f7d7aa9d6a5e38a1f0a1502c3fb9e3",
}

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

SEMANTIC_ENGINE_EVENT_KEYS = {
    "schema", "record", "diagnostic_variant", "publication_eligible",
    "condition", "semantic_pass", "schedule_phase", "schedule_phase_pass",
    "schedule_phase_slot", "case_ordinal", "op_index", "query_id",
    "external_global_delta_live", "api_result_count_before_adapter_truncation",
    "base_candidate_count", "visited_leaf_count",
    "full_immutable_base_candidate_count", "exact_full_immutable_base_fallback",
    "base_path", "merged_overlap_at_k", "merged_exact_match",
    "merged_result_sha256", "merged",
}
SEMANTIC_ENGINE_SUMMARY_KEYS = {
    "schema", "mode", "status", "diagnostic_variant", "publication_eligible",
    "claim_scope", "scope", "schedule", "conditions", "quality",
    "fresh_witness", "limitations",
}
LEGACY_CONTROL_ENGINE_MEASUREMENT_KEYS = {
    "schema", "record", "not_same_api", "timing_claim", "condition", "pass",
    "case_ordinal", "op_index", "query_id", "external_global_delta_live",
    "api_host_ns", "post_api_external_delta_merge_excluded_from_api_host_ns",
    "api_result_count_before_adapter_truncation", "base_candidate_count",
    "visited_leaf_count", "full_immutable_base_candidate_count",
    "exact_full_immutable_base_fallback", "base_path", "merged_overlap_at_k",
    "merged_exact_match", "merged_result_sha256", "merged",
}

# Stage-0 witness exact record shapes.  Keep these central: any source change
# must change this validator and the source manifest together.
RUN_START_KEYS = {
    "schema", "record", "diagnostic_variant", "publication_eligible", "run_id",
    "source_manifest_sha256", "artifact_hashes", "control_engine_schema",
    "bundle_path", "bundle_input_hashes", "admission_sha256", "tree_payload_sha256",
    "schedule", "hash_contract", "record_index", "prev_record_sha256", "record_sha256",
}
QUERY_KEYS = {
    "schema", "record", "diagnostic_variant", "publication_eligible", "record_index",
    "phase", "phase_pass", "phase_slot", "condition", "case_ordinal", "op_index",
    "query_id", "external_global_delta_live", "tree_payload_sha256", "base_path",
    "fresh_alloc", "traversal", "results", "query_status", "engine_mode_before",
    "engine_mode_after", "exception_stage", "prev_record_sha256", "record_sha256",
}
RUN_END_KEYS = {
    "schema", "record", "diagnostic_variant", "publication_eligible", "record_index",
    "status", "query_witness_count", "native_query_count", "fallback_query_count",
    "exception_count", "last_record_sha256", "prev_record_sha256", "record_sha256",
}
FRESH_ALLOC_NATIVE_KEYS = {
    "applicable", "scope", "allocation_epoch", "entry_named_global_scratch_ptrs_null",
    "entry", "memory", "allocations", "direct_events", "exit",
}
FRESH_ALLOC_FALLBACK_KEYS = {
    "applicable", "reason", "entry_named_global_scratch_ptrs_null", "allocations", "direct_events",
}
FRESH_ENTRY_KEYS = {"stack_empty", "receipt_cleared", "update_disk_set_false"}
FRESH_MEMORY_KEYS = {
    "qnum", "k", "tree_height", "free_bytes_after_fixed_allocs", "total_bytes",
    "capacity_policy", "policy_bytes", "p_list_elements", "p_list_requested_bytes",
}
FRESH_EXIT_KEYS = {"stack_empty", "all_tracked_freed"}
ALLOCATION_KEYS = {
    "slot", "allocator", "requested_bytes", "alloc_call_index", "alloc_status",
    "free_call_index", "free_status",
}
# This inline timeline is emitted immediately after each successful named direct
# cudaMalloc/cudaMallocManaged/cudaFree call, not reconstructed during sidecar
# serialization.  It is chain-covered together with the summary allocation ledger.
DIRECT_ALLOCATION_EVENT_KEYS = {"index", "slot", "operation", "requested_bytes"}
# Arrays are deliberately inline and indexed.  The item schemas and exact
# ASCII digest encodings live in validate_traversal().  Do not add an optional
# summary-only representation: a v1 record either supplies every preimage or
# fails closed.
TRAVERSAL_KEYS = {
    "stack_push_count", "stack_pop_count", "max_stack_depth",
    "stack_events", "stack_schedule_sha256",
    "size_list_write_count", "size_list_writes", "size_list_write_sha256",
    "traversal_step_count", "traversal_steps", "traversal_steps_sha256",
    "raw_leaf_pair_count", "ordered_leaf_pairs", "ordered_leaf_pair_sha256",
    "unique_leaf_count", "visited_leaf_ids", "unique_leaf_set_sha256",
    "id_list_capacity", "receipt_span_count", "receipt_leaf_spans", "receipt_span_sha256",
    "candidate_row_count", "candidate_rows", "ordered_candidate_row_sha256",
    "candidate_stable_set_sha256", "native_result_slot_count",
    "native_result_non_sentinel_count", "native_result_required_count",
    "native_result_complete", "native_result_slots", "native_result_slots_sha256",
    "native_final_res_ids_cross_checked",
}
STACK_EVENT_KEYS = {
    "index", "kind", "qs", "qe", "cur_level", "qnum_up", "offset_n", "qs_up",
    "size_a", "depth_after",
}
SIZE_LIST_WRITE_KEYS = {"index", "level", "value"}
TRAVERSAL_STEP_KEYS = {
    "index", "qs", "qe", "cur_level", "qnum_up", "offset_n", "qs_up", "size_a",
    "qnum_l", "offset_p", "offset_up_p", "nnum_l", "pnum_level", "pnum_level_total",
    "leaf_lnum", "receipt_stride", "update_disk_before", "used_label_cnode",
    "update_disk_after",
}
ORDERED_LEAF_PAIR_KEYS = {"index", "query_id", "leaf_id"}
RECEIPT_SPAN_KEYS = {"index", "leaf_id", "id_list_lid", "size"}
CANDIDATE_ROW_KEYS = {"index", "leaf_id", "id_list_slot", "local_row", "stable_id"}
NATIVE_RESULT_SLOT_KEYS = {"index", "local_row", "distance_f32_bits"}
F32_BITS_RE = re.compile(r"^[0-9a-f]{8}$")
RESULT_KEYS = {
    "base_candidate_count", "exact_candidate_stable_distance_sha256",
    "api_result_count_before_adapter_truncation", "api_result_stable_distance_sha256",
    "merged", "merged_result_sha256", "merged_overlap_at_k", "merged_exact_match",
}


class Fail(RuntimeError):
    """A closed-fail validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Fail(message)


def plain_int(value: Any, lower: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= lower


def require_sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA_RE.fullmatch(value) is not None,
            f"invalid SHA-256: {label}")
    return value


def require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} is not an object")
    require(set(value) == expected,
            f"{label} key set differs; missing={sorted(expected-set(value))}, "
            f"extra={sorted(set(value)-expected)}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lstat_regular(path: Path, label: str, private: bool) -> os.stat_result:
    try:
        status = path.lstat()
    except FileNotFoundError as exc:
        raise Fail(f"missing {label}: {path}") from exc
    require(stat.S_ISREG(status.st_mode) and not stat.S_ISLNK(status.st_mode),
            f"unsafe {label}: not a regular non-symlink")
    require(status.st_uid == 0 and status.st_gid == 0, f"{label} is not root-owned")
    if private:
        require(stat.S_IMODE(status.st_mode) & 0o077 == 0, f"{label} is not root-private")
    else:
        require(stat.S_IMODE(status.st_mode) & 0o022 == 0, f"{label} is writable by non-root")
    return status


def lstat_dir(path: Path, label: str, private: bool = True) -> os.stat_result:
    try:
        status = path.lstat()
    except FileNotFoundError as exc:
        raise Fail(f"missing {label}: {path}") from exc
    require(stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode),
            f"unsafe {label}: not a directory/non-symlink")
    require(status.st_uid == 0 and status.st_gid == 0, f"{label} is not root-owned")
    if private:
        require(stat.S_IMODE(status.st_mode) == 0o700, f"{label} is not root-private 0700")
    else:
        require(stat.S_IMODE(status.st_mode) & 0o022 == 0, f"{label} is writable by non-root")
    return status


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise Fail(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> Any:
    raise Fail(f"forbidden non-finite JSON constant: {value}")


def strict_json_loads(text: str, label: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_object,
                          parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, Fail) as exc:
        if isinstance(exc, Fail):
            raise
        raise Fail(f"invalid {label} JSON: {exc}") from exc


def load_private_json(path: Path, label: str) -> dict[str, Any]:
    lstat_regular(path, label, private=True)
    value = strict_json_loads(path.read_text(encoding="utf-8"), label)
    require(isinstance(value, dict), f"{label} root is not an object")
    return value


def load_private_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    lstat_regular(path, label, private=True)
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            require(line.endswith("\n") and line != "\n",
                    f"{label} line {line_number} is not a nonempty LF JSON line")
            value = strict_json_loads(line, f"{label} line {line_number}")
            require(isinstance(value, dict), f"{label} line {line_number} is not an object")
            output.append(value)
    return output


def read_env(path: Path) -> dict[str, str]:
    lstat_regular(path, "admission", private=True)
    output: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        require(line.count("=") == 1 and line.index("=") > 0,
                f"malformed admission line {line_number}")
        key, value = line.split("=", 1)
        require(key not in output and value and not any(ch.isspace() for ch in value),
                f"unsafe admission line {line_number}")
        output[key] = value
    return output


def env_need(env: dict[str, str], key: str) -> str:
    require(key in env, f"admission omits {key}")
    return env[key]


def env_int(env: dict[str, str], key: str) -> int:
    value = env_need(env, key)
    require(value.isdecimal(), f"admission {key} is not a decimal")
    return int(value)


def read_le_array(path: Path, typecode: str, width: int, label: str) -> array.array:
    lstat_regular(path, label, private=False)
    raw = path.read_bytes()
    require(len(raw) % width == 0, f"{label} byte length is not aligned")
    output = array.array(typecode)
    require(output.itemsize == width, f"host array width drift for {label}")
    output.frombytes(raw)
    if sys.byteorder != "little":
        output.byteswap()
    return output


# ----- Sealed input/oracle -------------------------------------------------

@dataclass(frozen=True)
class Case:
    ordinal: int
    op_index: int
    query_id: int
    external_delta_live: int
    active: bytes
    exact_topk: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Bundle:
    dimension: int
    base_n: int
    pool_n: int
    query_n: int
    k: int
    events: tuple[tuple[int, int, int], ...]
    pool: array.array
    queries: array.array
    stable_to_pool_row: array.array
    base_ids: array.array


def parse_bundle(path: Path) -> Bundle:
    require(path.resolve() == BUNDLE_PATH and not path.is_symlink(), "wrong sealed bundle path")
    lstat_dir(path, "sealed bundle", private=False)
    for name, expected_sha in INPUT_HASHES.items():
        member = path / name
        lstat_regular(member, f"sealed input {name}", private=False)
        require(sha256_file(member) == expected_sha, f"sealed input hash drift: {name}")

    raw = (path / "trace.e1gtrc").read_bytes()
    require(len(raw) >= TRACE_HEADER.size, "short trace header")
    magic, version, dimension, base_n, reservoir, pool_n, query_n, k, _radius, count = \
        TRACE_HEADER.unpack_from(raw)
    require(magic == TRACE_MAGIC and version == 1 and dimension > 0 and base_n >= k > 0,
            "trace header contract")
    require(pool_n == base_n + reservoir and query_n > 0,
            "trace population contract")
    require(len(raw) == TRACE_HEADER.size + count * TRACE_EVENT.size,
            "trace byte length")
    events: list[tuple[int, int, int]] = []
    for index in range(count):
        op_index, opcode, argument = TRACE_EVENT.unpack_from(raw, TRACE_HEADER.size + index * TRACE_EVENT.size)
        require(op_index == index and opcode in (INSERT, KNN), f"trace op {index}")
        require((opcode == INSERT and base_n <= argument < pool_n) or
                (opcode == KNN and 0 <= argument < query_n), f"trace argument {index}")
        events.append((op_index, opcode, argument))

    pool = read_le_array(path / "pool.i16", "h", 2, "pool")
    queries = read_le_array(path / "queries.i16", "h", 2, "queries")
    mapping = read_le_array(path / "stable_id_to_pool_row.i32", "i", 4, "stable mapping")
    base = read_le_array(path / "initial_base_stable_ids.i32", "i", 4, "base IDs")
    require(len(pool) == pool_n * dimension and len(queries) == query_n * dimension,
            "input matrix shape")
    require(len(mapping) == pool_n and len(base) == base_n and len(set(mapping)) == pool_n,
            "stable mapping shape")
    require(len(set(base)) == base_n and min(base) >= 0 and max(base) < pool_n,
            "base IDs shape")
    return Bundle(dimension, base_n, pool_n, query_n, k, tuple(events), pool, queries, mapping, base)


def exact_distance(bundle: Bundle, stable: int, query_id: int) -> int:
    pool_row = int(bundle.stable_to_pool_row[stable])
    left = pool_row * bundle.dimension
    right = query_id * bundle.dimension
    total = 0
    for dimension in range(bundle.dimension):
        delta = int(bundle.pool[left + dimension]) - int(bundle.queries[right + dimension])
        total += delta * delta
    return total


def exact_topk(bundle: Bundle, active: bytearray, query_id: int) -> tuple[tuple[int, int], ...]:
    values = [(exact_distance(bundle, stable, query_id), stable)
              for stable, enabled in enumerate(active) if enabled]
    values.sort()
    require(len(values) >= bundle.k, "fewer than k active IDs")
    return tuple((stable, distance) for distance, stable in values[:bundle.k])


def active_hash(active: bytearray) -> str:
    return hashlib.sha256(
        "".join(f"{stable}\n" for stable, enabled in enumerate(active) if enabled).encode("ascii")
    ).hexdigest()


def build_cases(bundle: Bundle) -> tuple[tuple[Case, ...], int, str]:
    active = bytearray(bundle.pool_n)
    for stable in bundle.base_ids:
        active[int(stable)] = 1
    delta_live = 0
    output: list[Case] = []
    for op_index, opcode, argument in bundle.events:
        if opcode == INSERT:
            require(not active[argument], f"duplicate insertion at op {op_index}")
            active[argument] = 1
            delta_live += 1
            continue
        output.append(Case(len(output), op_index, argument, delta_live,
                           bytes(active), exact_topk(bundle, active, argument)))
    require(len(output) == 136 and delta_live == 169, "sealed trace cardinality")
    return tuple(output), delta_live, active_hash(active)


def validate_admission(bundle: Bundle, path: Path) -> dict[str, str]:
    env = read_env(path)
    require(env_need(env, "schema") == ADMISSION_SCHEMA and env_need(env, "status") == "PASS",
            "admission schema/status")
    require(Path(env_need(env, "bundle_realpath")).resolve() == BUNDLE_PATH,
            "admission bundle path")
    require(env_need(env, "ops") == "insert,knn" and env_need(env, "excluded_ops") == "delete,range",
            "admission operation projection")
    require(env_need(env, "base_immutable") == "true" and
            env_need(env, "direct_sidecar_allowed") == "false" and
            env_need(env, "legacy_routing_allowed") == "false",
            "admission Safe-C1 policy")
    expected_ints = {
        "dimension": bundle.dimension, "base_n": bundle.base_n, "pool_n": bundle.pool_n,
        "query_n": bundle.query_n, "k": bundle.k, "event_count": len(bundle.events),
        "insert_count": 169, "knn_count": 136, "final_active_count": 4265,
    }
    for key, expected in expected_ints.items():
        require(env_int(env, key) == expected, f"admission mismatch: {key}")
    input_to_env = {
        "manifest.json": "manifest_sha256", "metadata.json": "metadata_sha256",
        "trace.e1gtrc": "trace_sha256", "pool.i16": "pool_sha256",
        "queries.i16": "queries_sha256", "stable_id_to_pool_row.i32": "mapping_sha256",
        "initial_base_stable_ids.i32": "base_ids_sha256",
    }
    for name, expected in INPUT_HASHES.items():
        require(env_need(env, input_to_env[name]) == expected, f"admission input hash: {name}")
    require(env_need(env, "projection_event_stream_sha256") ==
            "0d7267e098445723c7c065e9937206e37eca040b27025ad98b979c9a9af16b64",
            "admission projected event SHA")
    require(env_need(env, "final_active_set_sha256") ==
            "281d47954a4bbb2a85bafb09e150dd72e2ce4f9086568cf1a1a77d5657fa8e99",
            "admission final active SHA")
    return env


# ----- Existing v1b engine semantics --------------------------------------

def parse_result_rows(value: Any, label: str, bundle: Bundle, case: Case) -> tuple[tuple[int, int], ...]:
    require(isinstance(value, list) and len(value) == bundle.k,
            f"{label} must have exactly k rows")
    prior: tuple[int, int] | None = None
    seen: set[int] = set()
    rows: list[tuple[int, int]] = []
    for index, row in enumerate(value):
        require(isinstance(row, list) and len(row) == 2, f"{label}[{index}] is not a pair")
        stable, distance = row
        require(plain_int(stable) and stable < bundle.pool_n, f"{label}[{index}] stable ID")
        require(plain_int(distance), f"{label}[{index}] distance")
        require(stable not in seen and case.active[stable] == 1,
                f"{label}[{index}] inactive/duplicate stable ID")
        require(distance == exact_distance(bundle, stable, case.query_id),
                f"{label}[{index}] non-exact squared-L2")
        key = (distance, stable)
        require(prior is None or prior < key, f"{label} noncanonical ordering")
        seen.add(stable)
        prior = key
        rows.append((stable, distance))
    return tuple(rows)


def merged_hash(rows: tuple[tuple[int, int], ...]) -> str:
    payload = MERGED_HASH_DOMAIN + "\n" + "".join(f"{stable}:{distance}\n" for stable, distance in rows)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def overlap(rows: tuple[tuple[int, int], ...], oracle: tuple[tuple[int, int], ...]) -> int:
    return len({stable for stable, _ in rows} & {stable for stable, _ in oracle})


def _condition_order(global_pass: int, case: Case) -> tuple[str, str]:
    native_first = ((global_pass + case.ordinal) % 2) == 0
    return (("native_query_knn_candidate", "exact_query_range_full_base_fallback")
            if native_first else
            ("exact_query_range_full_base_fallback", "native_query_knn_candidate"))


def validate_semantic_engine_records(
        bundle: Bundle, cases: tuple[Case, ...], records: list[dict[str, Any]]
) -> tuple[dict[tuple[int, int, str], dict[str, Any]], dict[str, int]]:
    require(len(records) == len(cases) * 2, "wrong semantic engine record count")
    mapping: dict[tuple[int, int, str], dict[str, Any]] = {}
    native_exact = native_overlap = fallback_exact = 0
    cursor = 0
    global_pass = 1
    phase_slot_base = len(cases) * 2
    for case in cases:
        for within_case, condition in enumerate(_condition_order(global_pass, case)):
            row = records[cursor]
            cursor += 1
            require_exact_keys(row, SEMANTIC_ENGINE_EVENT_KEYS,
                               f"semantic engine record {cursor}")
            require(row.get("schema") == SEMANTIC_ENGINE_SCHEMA and
                    row.get("record") == SEMANTIC_ENGINE_RECORD and
                    row.get("diagnostic_variant") == VARIANT and
                    row.get("publication_eligible") is False,
                    f"semantic engine identity {cursor}")
            require(row.get("condition") == condition and row.get("semantic_pass") == 0 and
                    row.get("schedule_phase") == "semantic" and
                    row.get("schedule_phase_pass") == global_pass and
                    row.get("schedule_phase_slot") == phase_slot_base + case.ordinal * 2 + within_case and
                    row.get("case_ordinal") == case.ordinal and
                    row.get("op_index") == case.op_index and
                    row.get("query_id") == case.query_id and
                    row.get("external_global_delta_live") == case.external_delta_live,
                    f"semantic engine schedule binding {cursor}")
            rows = parse_result_rows(row.get("merged"), f"semantic engine merged {cursor}",
                                     bundle, case)
            require(row.get("merged_result_sha256") == merged_hash(rows),
                    f"semantic engine merged hash {cursor}")
            actual_overlap = overlap(rows, case.exact_topk)
            exact = rows == case.exact_topk
            require(row.get("merged_overlap_at_k") == actual_overlap and
                    row.get("merged_exact_match") is exact,
                    f"semantic engine quality {cursor}")
            if condition == "native_query_knn_candidate":
                require(row.get("api_result_count_before_adapter_truncation") == bundle.k and
                        plain_int(row.get("base_candidate_count"), bundle.k) and
                        row["base_candidate_count"] <= bundle.base_n and
                        plain_int(row.get("visited_leaf_count"), 1) and
                        row.get("full_immutable_base_candidate_count") == 0 and
                        row.get("exact_full_immutable_base_fallback") is False and
                        row.get("base_path") == NATIVE_PATH,
                        f"semantic native contract {cursor}")
                native_exact += int(exact)
                native_overlap += actual_overlap
            else:
                require(row.get("api_result_count_before_adapter_truncation") == bundle.base_n and
                        row.get("base_candidate_count") == bundle.base_n and
                        plain_int(row.get("visited_leaf_count")) and
                        row.get("full_immutable_base_candidate_count") == bundle.base_n and
                        row.get("exact_full_immutable_base_fallback") is True and
                        row.get("base_path") == FALLBACK_PATH and exact,
                        f"semantic fallback contract {cursor}")
                fallback_exact += 1
            key = (0, case.ordinal, condition)
            require(key not in mapping, f"duplicate semantic engine key {key}")
            mapping[key] = row
    return mapping, {
        "native_exact_set_count": native_exact,
        "native_overlap_sum": native_overlap,
        "fallback_exact_set_count": fallback_exact,
    }


def validate_semantic_engine_summary(summary: dict[str, Any],
                                     cases: tuple[Case, ...],
                                     metrics: dict[str, int]) -> None:
    require_exact_keys(summary, SEMANTIC_ENGINE_SUMMARY_KEYS, "semantic engine summary")
    require(summary.get("schema") == SEMANTIC_ENGINE_SCHEMA and
            summary.get("mode") == SEMANTIC_ENGINE_MODE and
            summary.get("status") == SEMANTIC_ENGINE_STATUS and
            summary.get("diagnostic_variant") == VARIANT and
            summary.get("publication_eligible") is False and
            summary.get("claim_scope") == SEMANTIC_CLAIM_SCOPE,
            "semantic summary identity")
    require(summary.get("scope") ==
            "sealed-input semantic/correctness control with fresh-allocation witness sidecar",
            "semantic summary scope")
    require(summary.get("schedule") == {
        "initialization_passes": 1,
        "semantic_passes": 1,
        "semantic_records": len(cases) * 2,
        "ordering": "ABBA_native_first_if_(global_phase_pass_plus_case_ordinal)_mod_2_is_0",
    }, "semantic summary schedule")
    require(summary.get("conditions") == {
        "native_query_knn_candidate": len(cases),
        "exact_query_range_full_base_fallback": len(cases),
    }, "semantic summary condition counts")
    require(summary.get("quality") == {
        "native_exact_set_count": metrics["native_exact_set_count"],
        "native_overlap_sum": metrics["native_overlap_sum"],
        "fallback_exact_set_count": metrics["fallback_exact_set_count"],
        "fallback_expected_exact_set_count": len(cases),
    }, "semantic summary quality")
    require(summary.get("fresh_witness") == {
        "query_records": len(cases) * 4,
        "native_query_records": len(cases) * 2,
        "fallback_query_records": len(cases) * 2,
    }, "semantic summary witness counts")
    require(summary.get("limitations") == [
        "semantic_control_only",
        "conditions_have_distinct_semantic_contracts",
        "fresh_allocation_lifecycle_is_checked_in_sealed_witness",
        "no_delete_range_workload_rebuild_or_direct_sidecar_claim",
        "no_performance_conclusion",
    ], "semantic summary limitations")


def legacy_merged_hash(rows: tuple[tuple[int, int], ...]) -> str:
    payload = LEGACY_MERGED_HASH_DOMAIN + "\n" + "".join(
        f"{stable}:{distance}\n" for stable, distance in rows
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def validate_legacy_control_records(
        bundle: Bundle, cases: tuple[Case, ...], records: list[dict[str, Any]]
) -> dict[tuple[int, int, str], dict[str, Any]]:
    require(len(records) == len(cases) * 2, "wrong historical reference record count")
    mapping: dict[tuple[int, int, str], dict[str, Any]] = {}
    cursor = 0
    for pass_number in range(1):
        for case in cases:
            for condition in _condition_order(pass_number, case):
                row = records[cursor]
                cursor += 1
                require_exact_keys(row, LEGACY_CONTROL_ENGINE_MEASUREMENT_KEYS,
                                   f"historical reference record {cursor}")
                require(row.get("schema") == LEGACY_CONTROL_ENGINE_SCHEMA and
                        row.get("record") == LEGACY_CONTROL_ENGINE_RECORD,
                        f"historical reference identity {cursor}")
                require(row.get("condition") == condition and row.get("pass") == pass_number and
                        row.get("case_ordinal") == case.ordinal and
                        row.get("op_index") == case.op_index and
                        row.get("query_id") == case.query_id and
                        row.get("external_global_delta_live") == case.external_delta_live,
                        f"historical reference schedule binding {cursor}")
                rows = parse_result_rows(row.get("merged"),
                                         f"historical reference merged {cursor}", bundle, case)
                require(row.get("merged_result_sha256") == legacy_merged_hash(rows),
                        f"historical reference merged hash {cursor}")
                actual_overlap = overlap(rows, case.exact_topk)
                exact = rows == case.exact_topk
                require(row.get("merged_overlap_at_k") == actual_overlap and
                        row.get("merged_exact_match") is exact,
                        f"historical reference quality {cursor}")
                if condition == "native_query_knn_candidate":
                    require(row.get("api_result_count_before_adapter_truncation") == bundle.k and
                            plain_int(row.get("base_candidate_count"), bundle.k) and
                            row["base_candidate_count"] <= bundle.base_n and
                            plain_int(row.get("visited_leaf_count"), 1) and
                            row.get("full_immutable_base_candidate_count") == 0 and
                            row.get("exact_full_immutable_base_fallback") is False and
                            row.get("base_path") == NATIVE_PATH,
                            f"historical native contract {cursor}")
                else:
                    require(row.get("api_result_count_before_adapter_truncation") == bundle.base_n and
                            row.get("base_candidate_count") == bundle.base_n and
                            plain_int(row.get("visited_leaf_count")) and
                            row.get("full_immutable_base_candidate_count") == bundle.base_n and
                            row.get("exact_full_immutable_base_fallback") is True and
                            row.get("base_path") == FALLBACK_PATH and exact,
                            f"historical fallback contract {cursor}")
                key = (pass_number, case.ordinal, condition)
                require(key not in mapping, f"duplicate historical reference key {key}")
                mapping[key] = row
    return mapping


# ----- Canonical witness chain --------------------------------------------

def canonical_value(value: Any) -> bytes:
    """Tagged, length-delimited, recursive JSON-value canonicalization.

    It supports the JSON types permitted by the witness.  Object keys are
    sorted by UTF-8 bytes, integers use Python's canonical decimal conversion,
    and no floating value is permitted.  This prevents whitespace/key-order
    ambiguity while avoiding language-specific JSON formatting.
    """
    if value is None:
        return b"N;"
    if value is True:
        return b"B1;"
    if value is False:
        return b"B0;"
    if isinstance(value, int) and not isinstance(value, bool):
        return b"I" + str(value).encode("ascii") + b";"
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"S" + str(len(raw)).encode("ascii") + b":" + raw + b";"
    if isinstance(value, list):
        return (b"A" + str(len(value)).encode("ascii") + b"[" +
                b"".join(canonical_value(item) for item in value) + b"]")
    if isinstance(value, dict):
        pairs: list[tuple[bytes, str, Any]] = []
        for key, item in value.items():
            require(isinstance(key, str), "non-string JSON object key in witness")
            pairs.append((key.encode("utf-8"), key, item))
        pairs.sort(key=lambda item: item[0])
        return (b"O" + str(len(pairs)).encode("ascii") + b"{" +
                b"".join(canonical_value(key) + canonical_value(item) for _, key, item in pairs) + b"}")
    raise Fail(f"unsupported witness JSON value type: {type(value).__name__}")


def record_digest(record: dict[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_sha256", None)
    return hashlib.sha256(CHAIN_DOMAIN + canonical_value(payload)).hexdigest()


def canonical_line_hash(domain: str, lines: Iterable[str]) -> str:
    raw = domain.encode("ascii") + b"\n"
    for line in lines:
        require("\n" not in line and "\r" not in line, "noncanonical raw hash line")
        raw += line.encode("ascii") + b"\n"
    return hashlib.sha256(raw).hexdigest()


# ----- Static diagnostic manifest / guard stage receipt ------------------

def validate_manifest(path: Path) -> dict[str, Any]:
    """Validate the one-way source manifest shared with the guarded launcher.

    Build receipt/static audit intentionally bind this manifest later; they are
    not listed in its own closure, avoiding a circular hash dependency.
    """
    manifest = load_private_json(path, "source manifest")
    require(manifest.get("schema") == MANIFEST_SCHEMA and
            manifest.get("diagnostic_variant") == VARIANT and
            manifest.get("publication_eligible") is False,
            "source manifest identity")
    require(manifest.get("event_schema") == SEMANTIC_ENGINE_SCHEMA and
            manifest.get("summary_schema") == SEMANTIC_ENGINE_SCHEMA and
            manifest.get("claim_scope") == SEMANTIC_CLAIM_SCOPE,
            "source manifest semantic/schema boundary")
    control = manifest.get("control_artifacts")
    require(isinstance(control, dict) and control.get("run_dir") == str(CONTROL_RUN),
            "source manifest fixed-control binding")
    for name, expected in CONTROL_HASHES.items():
        require(control.get(name + "_sha256") == expected,
                f"source manifest fixed-control hash: {name}")

    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), "source manifest artifacts")
    for name, spec in artifacts.items():
        require(isinstance(name, str) and isinstance(spec, dict) and
                set(spec) == {"path", "sha256"}, f"source artifact {name}")
        artifact = Path(spec["path"])
        require(artifact.is_absolute() and artifact.is_relative_to(ROOT),
                f"source artifact escapes root: {name}")
        lstat_regular(artifact, f"source artifact {name}", private=artifact.is_relative_to(DIAG))
        require(sha256_file(artifact) == require_sha(spec["sha256"], f"source artifact {name}"),
                f"source artifact hash drift: {name}")

    closure = manifest.get("source_closure")
    require(isinstance(closure, list) and closure, "source manifest closure")
    closure_lines: list[str] = []
    closure_paths: set[str] = set()
    for index, entry in enumerate(closure):
        require(isinstance(entry, dict) and set(entry) == {"path", "sha256"},
                f"source closure entry {index}")
        raw_path = entry["path"]
        require(isinstance(raw_path, str), f"source closure path {index}")
        member = Path(raw_path)
        require(member.is_absolute() and member.is_relative_to(ROOT),
                f"source closure escapes root {index}")
        require(raw_path not in closure_paths, f"duplicate source closure path {index}")
        closure_paths.add(raw_path)
        lstat_regular(member, f"source closure member {index}", private=member.is_relative_to(DIAG))
        digest = require_sha(entry["sha256"], f"source closure hash {index}")
        require(sha256_file(member) == digest, f"source closure hash drift {index}")
        closure_lines.append(str(member.relative_to(ROOT)) + "\t" + digest)
    expected_closure = hashlib.sha256(("\n".join(sorted(closure_lines)) + "\n").encode("ascii")).hexdigest()
    require(manifest.get("full_source_closure_sha256") == expected_closure,
            "source manifest closure digest")
    return manifest

def validate_guard_receipt(path: Path, args: argparse.Namespace, manifest_sha: str) -> dict[str, Any]:
    """Validate the immutable pre-validator stage receipt.

    It is write-once at the post-run/pre-validator boundary.  Its detailed
    GPU/NVML provenance may evolve, but its identity, artifact hashes, runner
    completion, and canonical output paths are a strict P1 input contract.
    """
    receipt = load_private_json(path, "guard stage receipt")
    require(receipt.get("schema") == GUARD_RECEIPT_SCHEMA, "guard receipt schema")
    require(receipt.get("status") in GUARD_PENDING_STATUSES, "guard receipt not staged for validation")
    inputs = receipt.get("inputs")
    runner = receipt.get("runner")
    outputs = receipt.get("outputs")
    require(isinstance(inputs, dict) and isinstance(runner, dict) and isinstance(outputs, dict),
            "guard receipt blocks")
    require(inputs.get("source_manifest_sha256") == manifest_sha,
            "guard receipt source manifest binding")
    require(inputs.get("validator_sha256") == sha256_file(Path(__file__).resolve()),
            "guard receipt executing validator hash")
    require_sha(inputs.get("runner_source_sha256"), "guard receipt runner source")
    require_sha(inputs.get("binary_sha256"), "guard receipt runner binary")
    lstat_regular(GUARD_SCRIPT, "fresh-allocation guarded launcher", private=True)
    require(inputs.get("guard_sha256") == sha256_file(GUARD_SCRIPT),
            "guard receipt guarded-launcher hash")
    require(plain_int(runner.get("exit_code")) and runner.get("exit_code") == 0 and
            runner.get("timed_out") is False and runner.get("finished_within_ttl") is True,
            "guard runner completion")
    require(runner.get("events_sha256") == sha256_file(args.events), "guard events hash")
    require(runner.get("summary_sha256") == sha256_file(args.summary), "guard summary hash")
    require(runner.get("witness_sha256") == sha256_file(args.witness), "guard witness hash")
    require(outputs.get("events") == str(args.events.resolve()) and
            outputs.get("summary") == str(args.summary.resolve()) and
            outputs.get("witness") == str(args.witness.resolve()) and
            outputs.get("validator") == str(args.out.resolve()),
            "guard output path binding")
    return receipt


# ----- Stage-0 witness record validation ----------------------------------

def validate_hash_contract(value: Any) -> None:
    require(value == HASH_DOMAINS, "witness header hash contract")


def validate_run_start(row: dict[str, Any], args: argparse.Namespace, manifest_sha: str,
                       manifest_source_closure_sha: str, manifest_runner_sha: str,
                       guard_receipt: dict[str, Any], expected_query_count: int) -> None:
    require_exact_keys(row, RUN_START_KEYS, "witness run_start")
    require(row.get("schema") == WITNESS_SCHEMA and row.get("record") == "run_start" and
            row.get("diagnostic_variant") == VARIANT and row.get("publication_eligible") is False,
            "witness run_start identity")
    require(row.get("record_index") == 0 and row.get("prev_record_sha256") is None,
            "witness run_start chain root")
    require(isinstance(row.get("run_id"), str) and RUN_RE.fullmatch(row["run_id"]) is not None,
            "witness run ID")
    require(row.get("source_manifest_sha256") == manifest_sha,
            "witness source manifest binding")
    require(row.get("control_engine_schema") == SEMANTIC_ENGINE_SCHEMA,
            "witness semantic-engine schema binding")
    require(row.get("bundle_path") == str(args.bundle.resolve()) and
            row.get("bundle_input_hashes") == INPUT_HASHES and
            row.get("admission_sha256") == sha256_file(args.admission),
            "witness sealed input binding")
    require_sha(row.get("tree_payload_sha256"), "witness tree payload")
    artifacts = row.get("artifact_hashes")
    require(isinstance(artifacts, dict) and set(artifacts) == {
        "source_closure_sha256", "runner_sha256", "binary_sha256",
    }, "witness artifact hash map")
    for name, digest in artifacts.items():
        require_sha(digest, f"witness artifact {name}")
    require(artifacts["source_closure_sha256"] == manifest_source_closure_sha and
            artifacts["runner_sha256"] == manifest_runner_sha,
            "witness source-closure/runner binding")
    receipt_inputs = guard_receipt.get("inputs")
    require(isinstance(receipt_inputs, dict) and
            artifacts["runner_sha256"] == receipt_inputs.get("runner_source_sha256") and
            artifacts["binary_sha256"] == receipt_inputs.get("binary_sha256"),
            "witness runner/binary binding to immutable pending receipt")
    schedule = row.get("schedule")
    require(isinstance(schedule, dict) and set(schedule) == {
        "initialization_passes", "semantic_passes", "case_count", "query_witness_count",
        "native_query_count", "fallback_query_count", "ordering",
    }, "witness schedule shape")
    require(schedule.get("initialization_passes") == 1 and schedule.get("semantic_passes") == 1 and
            schedule.get("case_count") == expected_query_count and
            schedule.get("query_witness_count") == expected_query_count * 4 and
            schedule.get("native_query_count") == expected_query_count * 2 and
            schedule.get("fallback_query_count") == expected_query_count * 2 and
            schedule.get("ordering") == "ABBA_native_first_if_(global_phase_pass_plus_case_ordinal)_mod_2_is_0",
            "witness schedule declaration")
    validate_hash_contract(row.get("hash_contract"))
    require(row.get("record_sha256") == record_digest(row), "run_start record chain hash")


def expected_workload(global_pass: int, phase: str, phase_slot: int,
                      cases: tuple[Case, ...]) -> tuple[Case, str]:
    """Reconstruct the one global ABBA witness stream.

    The run_start contract fixes exactly one warmup and one measured pass.  In
    contrast to the fixed-v1b engine's measured ``pass=0`` compatibility
    field, witness ``phase_pass`` and ``phase_slot`` are global across both
    stages: warmup is pass 0 / slots [0, 272), measured is pass 1 / slots
    [272, 544).  Both values are independently checked so a serializer cannot
    reset them at the phase boundary and silently repeat the same ABBA order.
    """
    slots_per_pass = len(cases) * 2
    require(slots_per_pass > 0 and global_pass in (0, 1),
            "invalid global witness pass")
    expected_phase = "initialization" if global_pass == 0 else "semantic"
    require(phase == expected_phase, "global witness pass/phase mismatch")
    slot_start = global_pass * slots_per_pass
    require(slot_start <= phase_slot < slot_start + slots_per_pass,
            "witness global phase slot outside pass")
    local_slot = phase_slot - slot_start
    case_index = local_slot // 2
    within_case = local_slot % 2
    case = cases[case_index]
    native_first = ((global_pass + case.ordinal) % 2) == 0
    conditions = (("native_query_knn_candidate", "exact_query_range_full_base_fallback")
                  if native_first else
                  ("exact_query_range_full_base_fallback", "native_query_knn_candidate"))
    return case, conditions[within_case]


def validate_allocations(value: Any, *, epoch: int, condition: str) -> tuple[int, int, int]:
    """Validate the native direct allocation ledger and return qnum/k/tree_h."""
    fresh = require_exact_keys(value, FRESH_ALLOC_NATIVE_KEYS, "native fresh_alloc")
    require(fresh.get("applicable") is True and
            fresh.get("scope") == "named_direct_cuda_allocation_lifecycle_v1" and
            fresh.get("allocation_epoch") == epoch and
            fresh.get("entry_named_global_scratch_ptrs_null") is True,
            "native fresh-allocation declaration")
    entry = require_exact_keys(fresh.get("entry"), FRESH_ENTRY_KEYS, "fresh allocation entry")
    require(entry == {
        "stack_empty": True, "receipt_cleared": True, "update_disk_set_false": True,
    }, "native reset witness")
    memory = require_exact_keys(fresh.get("memory"), FRESH_MEMORY_KEYS, "fresh allocation memory")
    qnum = memory.get("qnum")
    k = memory.get("k")
    tree_height = memory.get("tree_height")
    free_bytes = memory.get("free_bytes_after_fixed_allocs")
    total_bytes = memory.get("total_bytes")
    require(plain_int(qnum, 1) and plain_int(k, 1) and plain_int(tree_height, 1),
            "native qnum/k/tree_height")
    require(plain_int(free_bytes) and plain_int(total_bytes, free_bytes),
            "native cudaMemGetInfo values")
    require(memory.get("capacity_policy") == "cap_4GiB_if_free_gt_8GiB_else_half",
            "native p_list capacity policy label")
    limit = 4 * 1024 * 1024 * 1024
    policy_bytes = limit if free_bytes > 2 * limit else free_bytes // 2
    require(memory.get("policy_bytes") == policy_bytes and
            memory.get("p_list_elements") == policy_bytes // 8 and
            memory.get("p_list_requested_bytes") == (policy_bytes // 8) * 8,
            "native p_list capacity arithmetic")

    # The compact ledger and the inline immediate-event timeline must describe
    # the same five named direct objects.  The latter is required evidence: a
    # serializer may not fabricate a successful lifecycle solely from these
    # summary call/free indices after the fact.
    allocation_specs = [
        ("local_result_ids", "cudaMallocManaged", qnum * k * 4),
        ("res_dis", "cudaMallocManaged", qnum * k * 4),
        ("size_list", "cudaMallocManaged", (tree_height + 1) * 4),
        ("disk", "cudaMalloc", qnum * 4),
        ("p_list_k", "cudaMalloc", (policy_bytes // 8) * 8),
    ]
    allocations = fresh.get("allocations")
    require(isinstance(allocations, list) and len(allocations) == len(allocation_specs),
            "native direct allocation inventory")
    ledger: dict[str, dict[str, Any]] = {}
    for index, (actual, wanted) in enumerate(zip(allocations, allocation_specs), 1):
        item = require_exact_keys(actual, ALLOCATION_KEYS, f"native allocation {index}")
        slot, allocator, bytes_needed = wanted
        require(item.get("slot") == slot and item.get("allocator") == allocator and
                item.get("requested_bytes") == bytes_needed and
                plain_int(item.get("alloc_call_index"), 1) and
                plain_int(item.get("free_call_index"), 1) and
                item.get("alloc_status") == "success" and item.get("free_status") == "success",
                f"native allocation ledger {slot}")
        ledger[slot] = item

    events = fresh.get("direct_events")
    require(isinstance(events, list) and len(events) == len(allocation_specs) * 2,
            "immediate direct allocation event count")
    expected_events = [(slot, "alloc", bytes_needed) for slot, _allocator, bytes_needed in allocation_specs]
    # The adapter frees p_list before the two traversal buffers; this is not a
    # reverse-inventory convenience order.  It must match the inline source
    # event sequence emitted immediately after each successful cudaFree.
    free_order = ("p_list_k", "size_list", "disk", "res_dis", "local_result_ids")
    bytes_by_slot = {slot: bytes_needed for slot, _allocator, bytes_needed in allocation_specs}
    expected_events.extend((slot, "free", bytes_by_slot[slot]) for slot in free_order)
    alloc_ordinal = free_ordinal = 0
    for index, (raw, expected_event) in enumerate(zip(events, expected_events)):
        event = require_exact_keys(raw, DIRECT_ALLOCATION_EVENT_KEYS,
                                   f"immediate direct allocation event {index}")
        slot, operation, bytes_needed = expected_event
        require(plain_int(event.get("index")) and event.get("index") == index and
                event.get("slot") == slot and event.get("operation") == operation and
                plain_int(event.get("requested_bytes"), 1) and
                event.get("requested_bytes") == bytes_needed,
                f"immediate direct allocation event sequence {index}")
        item = ledger[slot]
        require(item["requested_bytes"] == event["requested_bytes"],
                f"direct event/ledger byte binding {slot}")
        if operation == "alloc":
            alloc_ordinal += 1
            require(item["alloc_call_index"] == alloc_ordinal,
                    f"direct allocation event/ledger order {slot}")
        else:
            free_ordinal += 1
            require(item["free_call_index"] == free_ordinal,
                    f"direct free event/ledger order {slot}")
    require(alloc_ordinal == len(allocation_specs) and free_ordinal == len(allocation_specs),
            "complete immediate direct allocation lifecycle")
    exit_state = require_exact_keys(fresh.get("exit"), FRESH_EXIT_KEYS, "fresh allocation exit")
    require(exit_state == {"stack_empty": True, "all_tracked_freed": True},
            "native fresh-allocation exit")
    return qnum, k, tree_height


def validate_fallback_fresh_alloc(value: Any) -> None:
    fresh = require_exact_keys(value, FRESH_ALLOC_FALLBACK_KEYS, "fallback fresh_alloc")
    require(fresh == {
        "applicable": False,
        "reason": "exact_full_immutable_base_range_fallback",
        # The copied native header is never entered on this full-range fallback,
        # so a null value is an explicit not-applicable witness rather than a
        # claim about the process-global scratch pointers.
        "entry_named_global_scratch_ptrs_null": None,
        "allocations": [],
        "direct_events": [],
    }, "fallback fresh-allocation non-applicability")


def signed_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def stable_distance_hash(rows: tuple[tuple[int, int], ...], domain: str,
                         label: str) -> str:
    """Reproduce safe_c1_g3::stable_distance_vector_sha256 exactly."""
    seen: set[int] = set()
    prior: tuple[int, int] | None = None
    lines: list[str] = []
    for index, (stable, distance) in enumerate(rows):
        require(plain_int(stable) and plain_int(distance), f"{label} stable/distance {index}")
        require(stable not in seen, f"{label} duplicate stable ID {stable}")
        key = (distance, stable)
        require(prior is None or prior < key, f"{label} noncanonical stable-distance order")
        prior = key
        seen.add(stable)
        lines.append(f"{stable}:{distance}")
    return canonical_line_hash(domain, lines)


def exact_rows_for_stables(bundle: Bundle, case: Case,
                            stables: Iterable[int], label: str) -> tuple[tuple[int, int], ...]:
    seen: set[int] = set()
    rows: list[tuple[int, int]] = []
    for stable in stables:
        require(plain_int(stable) and stable < bundle.pool_n, f"{label} stable ID")
        require(stable not in seen, f"{label} duplicate stable ID")
        seen.add(stable)
        rows.append((stable, exact_distance(bundle, stable, case.query_id)))
    rows.sort(key=lambda row: (row[1], row[0]))
    return tuple(rows)


def external_delta_topk(bundle: Bundle, case: Case) -> tuple[tuple[int, int], ...]:
    """Recreate the runner's exact_topk_for_ids() external-delta operand."""
    immutable = {int(stable) for stable in bundle.base_ids}
    delta = [stable for stable, enabled in enumerate(case.active)
             if enabled and stable not in immutable]
    require(len(delta) == case.external_delta_live, "sealed external-delta cardinality")
    return exact_rows_for_stables(bundle, case, delta, "external delta")[:bundle.k]


def expected_merged_from_base(bundle: Bundle, case: Case,
                               base_topk: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    combined = list(base_topk) + list(external_delta_topk(bundle, case))
    stable_ids = [stable for stable, _distance in combined]
    require(len(stable_ids) == len(set(stable_ids)), "base/external-delta overlap")
    combined.sort(key=lambda row: (row[1], row[0]))
    return tuple(combined[:bundle.k])


def validate_stack_events(value: Any, traversal: dict[str, Any]) -> list[tuple[int, int, int, int, int, int, int]]:
    require(isinstance(value, list), "stack_events is not an array")
    require(traversal.get("stack_push_count") + traversal.get("stack_pop_count") == len(value),
            "stack event count differs from push/pop counters")
    logical_stack: list[tuple[int, int, int, int, int, int, int]] = []
    popped: list[tuple[int, int, int, int, int, int, int]] = []
    depth = 0
    maximum = 0
    push_count = pop_count = 0
    lines = [f"count={len(value)}"]
    for index, raw in enumerate(value):
        event = require_exact_keys(raw, STACK_EVENT_KEYS, f"stack event {index}")
        require(event.get("index") == index and event.get("kind") in (0, 1),
                f"stack event index/kind {index}")
        for key in ("qs", "qe", "cur_level", "qnum_up", "offset_n", "qs_up", "size_a"):
            require(signed_int(event.get(key)), f"stack event {index} integer {key}")
        require(plain_int(event.get("depth_after")), f"stack event {index} depth")
        require(event["qs"] >= 0 and event["qe"] >= event["qs"] and
                event["cur_level"] >= 1 and event["qnum_up"] >= 1 and
                event["offset_n"] >= 0 and event["qs_up"] >= 0 and event["size_a"] >= 0,
                f"stack event {index} traversal tuple")
        item = (event["qs"], event["qe"], event["cur_level"], event["qnum_up"],
                event["offset_n"], event["qs_up"], event["size_a"])
        if event["kind"] == 0:
            logical_stack.append(item)
            depth += 7
            push_count += 1
        else:
            require(logical_stack, f"stack pop without matching push at {index}")
            require(logical_stack.pop() == item, f"stack LIFO tuple mismatch at {index}")
            depth -= 7
            pop_count += 1
            popped.append(item)
        require(event["depth_after"] == depth, f"stack depth accounting at {index}")
        maximum = max(maximum, depth)
        lines.append(
            f"i={index};kind={event['kind']};qs={event['qs']};qe={event['qe']}"
            f";level={event['cur_level']};qnum_up={event['qnum_up']}"
            f";offset_n={event['offset_n']};qs_up={event['qs_up']}"
            f";size_a={event['size_a']};depth={event['depth_after']}"
        )
    require(not logical_stack and depth == 0, "native stack did not drain")
    require(push_count == traversal.get("stack_push_count") and
            pop_count == traversal.get("stack_pop_count") and
            maximum == traversal.get("max_stack_depth") and push_count > 0,
            "stack summary counters")
    require(traversal.get("stack_schedule_sha256") ==
            canonical_line_hash(HASH_DOMAINS["stack_schedule"], lines),
            "stack schedule hash preimage")
    return popped


def validate_size_list_writes(value: Any, traversal: dict[str, Any], *, qnum: int,
                              tree_height: int) -> None:
    require(isinstance(value, list), "size_list_writes is not an array")
    require(traversal.get("size_list_write_count") == len(value) and len(value) >= 2,
            "size-list write count")
    lines = [f"count={len(value)}"]
    for index, raw in enumerate(value):
        item = require_exact_keys(raw, SIZE_LIST_WRITE_KEYS, f"size-list write {index}")
        require(item.get("index") == index and signed_int(item.get("level")) and
                signed_int(item.get("value")), f"size-list write fields {index}")
        require(0 <= item["level"] <= tree_height and item["value"] >= 0,
                f"size-list write bounds {index}")
        lines.append(f"i={index};level={item['level']};value={item['value']}")
    first = value[0]
    second = value[1]
    require(first["level"] == 0 and first["value"] == qnum and
            second["level"] == 1 and second["value"] > 0,
            "size-list initialization writes")
    require(traversal.get("size_list_write_sha256") ==
            canonical_line_hash(HASH_DOMAINS["size_list_write"], lines),
            "size-list write hash preimage")


def validate_traversal_steps(value: Any, traversal: dict[str, Any], *, tree_height: int,
                             popped: list[tuple[int, int, int, int, int, int, int]]) -> None:
    require(isinstance(value, list), "traversal_steps is not an array")
    require(traversal.get("traversal_step_count") == len(value) == len(popped),
            "traversal-step / stack-pop count")
    lines = [f"count={len(value)}"]
    prior_update_after = False
    for index, raw in enumerate(value):
        item = require_exact_keys(raw, TRAVERSAL_STEP_KEYS, f"traversal step {index}")
        require(item.get("index") == index, f"traversal step index {index}")
        for key in (
            "qs", "qe", "cur_level", "qnum_up", "offset_n", "qs_up", "size_a", "qnum_l",
            "offset_p", "offset_up_p", "nnum_l", "pnum_level", "pnum_level_total",
            "leaf_lnum", "receipt_stride",
        ):
            require(signed_int(item.get(key)), f"traversal step {index} integer {key}")
        for key in ("update_disk_before", "used_label_cnode", "update_disk_after"):
            require(isinstance(item.get(key), bool), f"traversal step {index} boolean {key}")
        require(item["qs"] >= 0 and item["qe"] >= item["qs"] and
                1 <= item["cur_level"] <= tree_height and item["qnum_up"] >= 1 and
                item["offset_n"] >= 0 and item["qs_up"] >= 0 and item["size_a"] >= 0 and
                item["qnum_l"] == item["qe"] - item["qs"] + 1 and item["qnum_l"] >= 1 and
                item["offset_p"] >= 0 and item["offset_up_p"] >= 0 and item["nnum_l"] >= 1,
                f"traversal step bounds {index}")
        tuple_from_step = (item["qs"], item["qe"], item["cur_level"], item["qnum_up"],
                           item["offset_n"], item["qs_up"], item["size_a"])
        require(tuple_from_step == popped[index], f"traversal step/pop tuple mismatch {index}")
        require(item["update_disk_before"] is prior_update_after and
                (not item["update_disk_before"] or item["update_disk_after"]),
                f"update_disk monotonicity at traversal step {index}")
        prior_update_after = item["update_disk_after"]
        if item["cur_level"] < tree_height:
            require(item["pnum_level"] >= 0 and item["pnum_level_total"] >= item["pnum_level"] and
                    item["leaf_lnum"] == -1 and item["receipt_stride"] == -1,
                    f"internal traversal-step shape {index}")
        else:
            require(item["pnum_level"] == -1 and item["pnum_level_total"] == -1 and
                    item["leaf_lnum"] >= 0 and
                    (item["receipt_stride"] >= 0 if item["leaf_lnum"] > 0
                     else item["receipt_stride"] == -1),
                    f"leaf traversal-step shape {index}")
        lines.append(
            f"i={index};qs={item['qs']};qe={item['qe']};level={item['cur_level']}"
            f";qnum_up={item['qnum_up']};offset_n={item['offset_n']};qs_up={item['qs_up']}"
            f";size_a={item['size_a']};qnum_l={item['qnum_l']};offset_p={item['offset_p']}"
            f";offset_up_p={item['offset_up_p']};nnum_l={item['nnum_l']}"
            f";pnum={item['pnum_level']};pnum_total={item['pnum_level_total']}"
            f";lnum={item['leaf_lnum']};receipt_stride={item['receipt_stride']}"
            f";update_before={1 if item['update_disk_before'] else 0}"
            f";label={1 if item['used_label_cnode'] else 0}"
            f";update_after={1 if item['update_disk_after'] else 0}"
        )
    require(traversal.get("traversal_steps_sha256") ==
            canonical_line_hash(HASH_DOMAINS["traversal_steps"], lines),
            "traversal-step hash preimage")


def validate_native_traversal(value: Any, *, qnum: int, k: int, tree_height: int,
                              base_candidate_count: Any, bundle: Bundle, case: Case,
                              tree_payload_sha256: str) -> tuple[tuple[int, int], ...]:
    traversal = require_exact_keys(value, TRAVERSAL_KEYS, "native traversal witness")
    for key in (
        "stack_push_count", "stack_pop_count", "max_stack_depth", "size_list_write_count",
        "traversal_step_count", "raw_leaf_pair_count", "unique_leaf_count", "id_list_capacity",
        "receipt_span_count", "candidate_row_count", "native_result_slot_count",
        "native_result_non_sentinel_count", "native_result_required_count",
    ):
        require(plain_int(traversal.get(key)), f"native traversal integer {key}")
    for key in (
        "stack_schedule_sha256", "size_list_write_sha256", "traversal_steps_sha256",
        "ordered_leaf_pair_sha256", "unique_leaf_set_sha256", "receipt_span_sha256",
        "ordered_candidate_row_sha256", "candidate_stable_set_sha256", "native_result_slots_sha256",
    ):
        require_sha(traversal.get(key), f"native traversal commitment {key}")
    # This is attribution-only: it proves each non-sentinel archive res_ids
    # slot was checked against a receipt row, not that all required KNN slots
    # were populated or that this legacy buffer defines the Safe-C1 result.
    require(isinstance(traversal.get("native_final_res_ids_cross_checked"), bool) and
            traversal["native_final_res_ids_cross_checked"] is True,
            "native result ID attribution flag")
    require(plain_int(base_candidate_count, k) and
            traversal["candidate_row_count"] == base_candidate_count and
            traversal["candidate_row_count"] >= k,
            "native candidate count relation")
    require(traversal["native_result_required_count"] == min(k, base_candidate_count),
            "native diagnostic required-result count")
    require(isinstance(traversal.get("native_result_complete"), bool),
            "native diagnostic result-completeness type")

    popped = validate_stack_events(traversal["stack_events"], traversal)
    validate_size_list_writes(traversal["size_list_writes"], traversal,
                              qnum=qnum, tree_height=tree_height)
    validate_traversal_steps(traversal["traversal_steps"], traversal,
                             tree_height=tree_height, popped=popped)

    pairs_raw = traversal["ordered_leaf_pairs"]
    require(isinstance(pairs_raw, list) and len(pairs_raw) == traversal["raw_leaf_pair_count"],
            "ordered leaf-pair count")
    pairs: list[tuple[int, int]] = []
    pair_lines = [f"tree={tree_payload_sha256}", f"qnum={qnum}", f"count={len(pairs_raw)}"]
    for index, raw in enumerate(pairs_raw):
        item = require_exact_keys(raw, ORDERED_LEAF_PAIR_KEYS, f"ordered leaf pair {index}")
        require(item.get("index") == index and signed_int(item.get("query_id")) and
                signed_int(item.get("leaf_id")), f"ordered leaf pair fields {index}")
        require(0 <= item["query_id"] < qnum and item["leaf_id"] >= 0,
                f"ordered leaf pair bounds {index}")
        pairs.append((item["query_id"], item["leaf_id"]))
        pair_lines.append(f"i={index};q={item['query_id']};leaf={item['leaf_id']}")
    require(traversal["raw_leaf_pair_count"] >= traversal["unique_leaf_count"] > 0,
            "raw/unique leaf count relation")
    require(traversal["ordered_leaf_pair_sha256"] ==
            canonical_line_hash(HASH_DOMAINS["ordered_leaf_pair"], pair_lines),
            "ordered leaf-pair hash preimage")

    leaves_raw = traversal["visited_leaf_ids"]
    require(isinstance(leaves_raw, list), "visited_leaf_ids is not an array")
    leaves: list[int] = []
    for index, leaf in enumerate(leaves_raw):
        require(signed_int(leaf) and leaf >= 0, f"visited leaf {index}")
        leaves.append(leaf)
    expected_leaves = sorted({leaf for _query, leaf in pairs})
    require(leaves == expected_leaves and len(leaves) == traversal["unique_leaf_count"],
            "visited leaves must be sorted unique raw receipt leaves")
    leaf_lines = [f"tree={tree_payload_sha256}", f"count={len(leaves)}"]
    leaf_lines.extend(f"i={index};leaf={leaf}" for index, leaf in enumerate(leaves))
    require(traversal["unique_leaf_set_sha256"] ==
            canonical_line_hash(HASH_DOMAINS["unique_leaf_set"], leaf_lines),
            "unique leaf-set hash preimage")

    capacity = traversal["id_list_capacity"]
    require(capacity > 0, "native physical id_list capacity")
    spans_raw = traversal["receipt_leaf_spans"]
    require(isinstance(spans_raw, list) and len(spans_raw) == traversal["receipt_span_count"] == len(leaves),
            "receipt-span count")
    spans: list[tuple[int, int, int]] = []
    span_lines = [f"tree={tree_payload_sha256}", f"id_list_capacity={capacity}",
                  f"count={len(spans_raw)}"]
    for index, raw in enumerate(spans_raw):
        item = require_exact_keys(raw, RECEIPT_SPAN_KEYS, f"receipt span {index}")
        require(item.get("index") == index and signed_int(item.get("leaf_id")) and
                signed_int(item.get("id_list_lid")) and signed_int(item.get("size")),
                f"receipt span fields {index}")
        require(item["leaf_id"] == leaves[index] and item["id_list_lid"] >= 0 and item["size"] >= 0 and
                item["id_list_lid"] + item["size"] <= capacity,
                f"receipt span bounds/order {index}")
        spans.append((item["leaf_id"], item["id_list_lid"], item["size"]))
        span_lines.append(f"i={index};leaf={item['leaf_id']};lid={item['id_list_lid']};size={item['size']}")
    for index, (_leaf_a, lid_a, size_a) in enumerate(spans):
        for _leaf_b, lid_b, size_b in spans[index + 1:]:
            require(lid_a + size_a <= lid_b or lid_b + size_b <= lid_a,
                    "receipt leaf spans physically overlap")
    require(traversal["receipt_span_sha256"] ==
            canonical_line_hash(HASH_DOMAINS["receipt_span"], span_lines),
            "receipt-span hash preimage")

    candidates_raw = traversal["candidate_rows"]
    expected_candidate_count = sum(size for _leaf, _lid, size in spans)
    require(isinstance(candidates_raw, list) and len(candidates_raw) == expected_candidate_count ==
            traversal["candidate_row_count"], "candidate-row count / span coverage")
    candidate_stables: list[int] = []
    candidate_locals: set[int] = set()
    candidate_lines = [f"tree={tree_payload_sha256}", f"id_list_capacity={capacity}",
                       f"count={len(candidates_raw)}"]
    cursor = 0
    for leaf, lid, size in spans:
        for offset in range(size):
            item = require_exact_keys(candidates_raw[cursor], CANDIDATE_ROW_KEYS,
                                      f"candidate row {cursor}")
            require(item.get("index") == cursor and signed_int(item.get("leaf_id")) and
                    signed_int(item.get("id_list_slot")) and signed_int(item.get("local_row")) and
                    signed_int(item.get("stable_id")), f"candidate row fields {cursor}")
            require(item["leaf_id"] == leaf and item["id_list_slot"] == lid + offset and
                    0 <= item["local_row"] < bundle.base_n and item["local_row"] not in candidate_locals and
                    item["stable_id"] == int(bundle.base_ids[item["local_row"]]),
                    f"candidate row provenance {cursor}")
            candidate_locals.add(item["local_row"])
            candidate_stables.append(item["stable_id"])
            candidate_lines.append(
                f"i={cursor};leaf={item['leaf_id']};slot={item['id_list_slot']}"
                f";local={item['local_row']};stable={item['stable_id']}"
            )
            cursor += 1
    require(traversal["ordered_candidate_row_sha256"] ==
            canonical_line_hash(HASH_DOMAINS["ordered_candidate_row"], candidate_lines),
            "candidate-row hash preimage")
    stable_lines = [f"count={len(candidate_stables)}"]
    stable_lines.extend(f"i={index};stable={stable}"
                        for index, stable in enumerate(sorted(candidate_stables)))
    require(traversal["candidate_stable_set_sha256"] ==
            canonical_line_hash(HASH_DOMAINS["candidate_stable_set"], stable_lines),
            "candidate stable-set hash preimage")

    slots_raw = traversal["native_result_slots"]
    require(isinstance(slots_raw, list) and len(slots_raw) == traversal["native_result_slot_count"] == qnum * k,
            "native result-slot count")
    slot_lines = [f"qnum={qnum}", f"k={k}", f"count={len(slots_raw)}"]
    non_sentinel = 0
    seen_non_sentinel_locals: set[int] = set()
    for index, raw in enumerate(slots_raw):
        item = require_exact_keys(raw, NATIVE_RESULT_SLOT_KEYS, f"native result slot {index}")
        require(item.get("index") == index and signed_int(item.get("local_row")) and
                isinstance(item.get("distance_f32_bits"), str) and
                F32_BITS_RE.fullmatch(item["distance_f32_bits"]) is not None,
                f"native result slot fields {index}")
        local = item["local_row"]
        require(local == -1 or (0 <= local < bundle.base_n and local in candidate_locals),
                f"native result slot receipt attribution {index}")
        if local != -1:
            require(local not in seen_non_sentinel_locals,
                    f"native result slot duplicate receipt local {index}")
            seen_non_sentinel_locals.add(local)
            non_sentinel += 1
        slot_lines.append(f"i={index};local={local};distance_f32_bits={item['distance_f32_bits']}")
    require(traversal["native_result_non_sentinel_count"] == non_sentinel,
            "native non-sentinel slot count")
    require(traversal["native_result_complete"] ==
            (non_sentinel >= traversal["native_result_required_count"]),
            "native diagnostic result-completeness value")
    require(traversal["native_result_slots_sha256"] ==
            canonical_line_hash(HASH_DOMAINS["native_result_slot"], slot_lines),
            "native result-slot hash preimage")

    # The sealed exact scan of all receipt rows remains the sole native
    # candidate/result preimage, irrespective of legacy res_ids completeness.
    return exact_rows_for_stables(bundle, case, candidate_stables, "receipt candidate")


def validate_fallback_traversal(value: Any) -> None:
    traversal = require_exact_keys(value, TRAVERSAL_KEYS, "fallback traversal witness")
    expected = {
        "stack_push_count": 0, "stack_pop_count": 0, "max_stack_depth": 0,
        "stack_events": [], "stack_schedule_sha256": None,
        "size_list_write_count": 0, "size_list_writes": [], "size_list_write_sha256": None,
        "traversal_step_count": 0, "traversal_steps": [], "traversal_steps_sha256": None,
        "raw_leaf_pair_count": 0, "ordered_leaf_pairs": [], "ordered_leaf_pair_sha256": None,
        "unique_leaf_count": 0, "visited_leaf_ids": [], "unique_leaf_set_sha256": None,
        "id_list_capacity": 0, "receipt_span_count": 0, "receipt_leaf_spans": [],
        "receipt_span_sha256": None, "candidate_row_count": 0, "candidate_rows": [],
        "ordered_candidate_row_sha256": None, "candidate_stable_set_sha256": None,
        "native_result_slot_count": 0, "native_result_non_sentinel_count": 0,
        "native_result_required_count": None, "native_result_complete": None,
        "native_result_slots": [], "native_result_slots_sha256": None,
        "native_final_res_ids_cross_checked": False,
    }
    require(traversal == expected, "fallback must have no native GTS traversal witness")


def validate_results(value: Any, *, bundle: Bundle, case: Case, native: bool,
                     expected_candidate: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    results = require_exact_keys(value, RESULT_KEYS, "query results")
    base_count = results.get("base_candidate_count")
    api_count = results.get("api_result_count_before_adapter_truncation")
    require(plain_int(base_count) and plain_int(api_count), "result count fields")
    require(base_count == len(expected_candidate), "result candidate count preimage")
    expected_api = expected_candidate[:bundle.k] if native else expected_candidate
    require(api_count == len(expected_api), "result API count preimage")
    require(results.get("exact_candidate_stable_distance_sha256") ==
            stable_distance_hash(expected_candidate, HASH_DOMAINS["exact_candidate_stable_distance"],
                                 "exact candidate"),
            "exact candidate stable-distance hash preimage")
    require(results.get("api_result_stable_distance_sha256") ==
            stable_distance_hash(expected_api, HASH_DOMAINS["api_result_stable_distance"],
                                 "API result"),
            "API stable-distance hash preimage")
    rows = parse_result_rows(results.get("merged"), "witness merged", bundle, case)
    require(results.get("merged_result_sha256") == merged_hash(rows), "witness merged hash")
    expected_merged = expected_merged_from_base(bundle, case, expected_candidate[:bundle.k])
    require(rows == expected_merged, "merged result does not follow witnessed base + sealed external delta")
    actual_overlap = overlap(rows, case.exact_topk)
    exact = rows == case.exact_topk
    require(results.get("merged_overlap_at_k") == actual_overlap and
            results.get("merged_exact_match") is exact,
            "witness merged quality")
    if native:
        require(api_count == bundle.k and bundle.k <= base_count <= bundle.base_n,
                "native result count contract")
    else:
        require(api_count == bundle.base_n and base_count == bundle.base_n and exact,
                "fallback result count/exactness contract")
    return rows


def validate_semantic_engine_binding(witness: dict[str, Any], engine: dict[str, Any]) -> None:
    results = witness["results"]
    traversal = witness["traversal"]
    same = {
        "condition": witness["condition"],
        "semantic_pass": 0,
        "schedule_phase": "semantic",
        "schedule_phase_pass": 1,
        "schedule_phase_slot": witness["phase_slot"],
        "case_ordinal": witness["case_ordinal"],
        "op_index": witness["op_index"],
        "query_id": witness["query_id"],
        "external_global_delta_live": witness["external_global_delta_live"],
        "base_path": witness["base_path"],
        "api_result_count_before_adapter_truncation":
            results["api_result_count_before_adapter_truncation"],
        "base_candidate_count": results["base_candidate_count"],
        "visited_leaf_count": traversal["unique_leaf_count"],
        "merged": results["merged"],
        "merged_result_sha256": results["merged_result_sha256"],
        "merged_overlap_at_k": results["merged_overlap_at_k"],
        "merged_exact_match": results["merged_exact_match"],
    }
    for key, expected_value in same.items():
        require(engine.get(key) == expected_value, f"semantic witness/engine mismatch: {key}")


def validate_query(row: dict[str, Any], *, expected_index: int, phase: str,
                   phase_pass: int, phase_slot: int, case: Case, expected_condition: str,
                   header_tree_sha: str, native_epoch: int,
                   engine_records: dict[tuple[int, int, str], dict[str, Any]],
                   bundle: Bundle) -> tuple[bool, str]:
    require_exact_keys(row, QUERY_KEYS, f"query witness {expected_index}")
    require(row.get("schema") == WITNESS_SCHEMA and row.get("record") == "query_witness" and
            row.get("diagnostic_variant") == VARIANT and row.get("publication_eligible") is False,
            f"query witness identity {expected_index}")
    require(row.get("record_index") == expected_index and row.get("phase") == phase and
            row.get("phase_pass") == phase_pass and row.get("phase_slot") == phase_slot and
            row.get("condition") == expected_condition and
            row.get("case_ordinal") == case.ordinal and row.get("op_index") == case.op_index and
            row.get("query_id") == case.query_id and
            row.get("external_global_delta_live") == case.external_delta_live,
            f"query witness schedule {expected_index}")
    require(row.get("tree_payload_sha256") == header_tree_sha,
            f"query witness tree binding {expected_index}")
    native = expected_condition == "native_query_knn_candidate"
    require(row.get("base_path") == (NATIVE_PATH if native else FALLBACK_PATH),
            f"query witness base path {expected_index}")
    require(row.get("query_status") == "PASS" and row.get("engine_mode_before") == "ready" and
            row.get("engine_mode_after") == "ready" and row.get("exception_stage") is None,
            f"query witness success/fail-stop state {expected_index}")

    results = row["results"]
    base_count = results.get("base_candidate_count") if isinstance(results, dict) else None
    if native:
        qnum, k, tree_height = validate_allocations(row["fresh_alloc"], epoch=native_epoch,
                                                     condition=expected_condition)
        require(qnum == 1 and k == bundle.k, f"native qnum/k contract {expected_index}")
        expected_candidate = validate_native_traversal(
            row["traversal"], qnum=qnum, k=k, tree_height=tree_height,
            base_candidate_count=base_count, bundle=bundle, case=case,
            tree_payload_sha256=header_tree_sha,
        )
    else:
        validate_fallback_fresh_alloc(row["fresh_alloc"])
        validate_fallback_traversal(row["traversal"])
        expected_candidate = exact_rows_for_stables(
            bundle, case, (int(stable) for stable in bundle.base_ids), "full immutable base"
        )
    validate_results(row["results"], bundle=bundle, case=case,
                     native=native, expected_candidate=expected_candidate)

    if phase == "semantic":
        key = (0, case.ordinal, expected_condition)
        require(key in engine_records, f"missing matching semantic engine record {key}")
        validate_semantic_engine_binding(row, engine_records[key])
    return native, row["record_sha256"]


def validate_run_end(row: dict[str, Any], *, record_index: int, previous_digest: str,
                     query_count: int, native_count: int, fallback_count: int) -> None:
    require_exact_keys(row, RUN_END_KEYS, "witness run_end")
    require(row.get("schema") == WITNESS_SCHEMA and row.get("record") == "run_end" and
            row.get("diagnostic_variant") == VARIANT and row.get("publication_eligible") is False,
            "witness run_end identity")
    require(row.get("record_index") == record_index and row.get("status") == RUN_END_STATUS and
            row.get("query_witness_count") == query_count and
            row.get("native_query_count") == native_count and
            row.get("fallback_query_count") == fallback_count and
            row.get("exception_count") == 0 and row.get("last_record_sha256") == previous_digest and
            row.get("prev_record_sha256") == previous_digest,
            "witness run_end fields")
    require(row.get("record_sha256") == record_digest(row), "witness run_end chain hash")


def validate_witness(records: list[dict[str, Any]], *, args: argparse.Namespace,
                     bundle: Bundle, cases: tuple[Case, ...], manifest_sha: str,
                     manifest_source_closure_sha: str, manifest_runner_sha: str,
                     guard_receipt: dict[str, Any],
                     engine_records: dict[tuple[int, int, str], dict[str, Any]]) -> dict[str, Any]:
    expected_queries = len(cases) * 4
    require(len(records) == expected_queries + 2,
            "witness must be run_start + 544 query records + run_end")
    header = records[0]
    validate_run_start(header, args, manifest_sha, manifest_source_closure_sha,
                       manifest_runner_sha, guard_receipt, len(cases))
    previous_digest = header["record_sha256"]
    native_count = fallback_count = 0
    record_index = 1
    # The runner resets only the copied-header forensic counter after bootstrap
    # and before ABBA, so the first witnessed native header epoch is one.
    native_epoch = FIRST_WITNESSED_NATIVE_ALLOCATION_EPOCH
    slots_per_pass = len(cases) * 2
    # Witness pass/slot values form one stream across warmup then measured;
    # they deliberately do not reset as the fixed-v1b engine's measured
    # compatibility records retain their separate pass=0 convention.
    phase_order = (("initialization", 0), ("semantic", 1))
    for phase, global_pass in phase_order:
        for local_slot in range(slots_per_pass):
            phase_slot = global_pass * slots_per_pass + local_slot
            row = records[record_index]
            require(row.get("prev_record_sha256") == previous_digest,
                    f"witness chain predecessor at record {record_index}")
            case, condition = expected_workload(global_pass, phase, phase_slot, cases)
            native, digest = validate_query(
                row, expected_index=record_index, phase=phase, phase_pass=global_pass,
                phase_slot=phase_slot, case=case, expected_condition=condition,
                header_tree_sha=header["tree_payload_sha256"], native_epoch=native_epoch,
                engine_records=engine_records, bundle=bundle,
            )
            require(digest == record_digest(row), f"witness record chain hash {record_index}")
            if native:
                native_count += 1
                native_epoch += 1
            else:
                fallback_count += 1
            previous_digest = digest
            record_index += 1
    end = records[record_index]
    require(end.get("prev_record_sha256") == previous_digest,
            "witness run_end chain predecessor")
    validate_run_end(end, record_index=record_index, previous_digest=previous_digest,
                     query_count=expected_queries, native_count=native_count,
                     fallback_count=fallback_count)
    return {
        "run_id": header["run_id"],
        "tree_payload_sha256": header["tree_payload_sha256"],
        "witness_chain_sha256": end["record_sha256"],
        "witness_queries": expected_queries,
        "native_queries": native_count,
        "fallback_queries": fallback_count,
    }


# ----- Main / output -------------------------------------------------------

def write_once(path: Path, value: dict[str, Any]) -> None:
    require(path.parent == path.parent.resolve() and path.parent.is_dir(),
            "validator output parent")
    lstat_dir(path.parent, "validator output parent", private=True)
    require(not path.exists() and not path.is_symlink(), "validator refuses output overwrite")
    fd, temporary = tempfile.mkstemp(prefix=".fresh-alloc-validator.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def compare_fixed_reference(candidate: dict[tuple[int, int, str], dict[str, Any]],
                            reference: dict[tuple[int, int, str], dict[str, Any]]) -> dict[str, Any]:
    require(set(candidate) == set(reference), "candidate/reference semantic key set")
    compared = (
        "condition", "case_ordinal", "op_index", "query_id",
        "external_global_delta_live", "api_result_count_before_adapter_truncation",
        "full_immutable_base_candidate_count", "exact_full_immutable_base_fallback",
        "base_path", "merged", "merged_overlap_at_k", "merged_exact_match",
    )
    field_agreement = {field: 0 for field in compared}
    record_agreement = 0
    sequence = hashlib.sha256()
    for key in sorted(candidate):
        left = candidate[key]
        right = reference[key]
        all_equal = True
        for field in compared:
            equal = left.get(field) == right.get(field)
            field_agreement[field] += int(equal)
            all_equal = all_equal and equal
        record_agreement += int(all_equal)
        sequence.update((str(key) + "\n").encode("ascii"))
        sequence.update(left["merged_result_sha256"].encode("ascii") + b"\n")
    return {
        "reference_records": len(candidate),
        "compared_fields": list(compared),
        "record_agreement_count": record_agreement,
        "field_agreement_counts": field_agreement,
        "semantic_fingerprint_sha256": sequence.hexdigest(),
    }


def canonical_paths(args: argparse.Namespace) -> None:
    for name in (
        "bundle", "admission", "events", "summary", "witness", "guard_receipt",
        "control_events", "control_summary", "variant_manifest", "out",
    ):
        setattr(args, name, Path(getattr(args, name)).absolute())


def validate_paths(args: argparse.Namespace) -> Path:
    require(args.bundle.resolve() == BUNDLE_PATH and not args.bundle.is_symlink(),
            "validator accepts only sealed bundle")
    run = args.events.parent
    require(args.events.name == "engine.jsonl" and args.summary.name == "summary.json" and
            args.witness.name == "fresh_alloc_witness.jsonl" and
            args.guard_receipt.name == "receipt_for_validator.json" and
            args.admission.name == "admission.env" and args.out.name == "independent_validation.json",
            "canonical candidate run artifact names")
    require(args.events.parent == args.summary.parent == args.witness.parent == args.guard_receipt.parent ==
            args.admission.parent == args.out.parent == run,
            "candidate run artifact parents")
    require(run.parent == ROOT / "runs" and RUN_RE.fullmatch(run.name) is not None and
            run != CONTROL_RUN,
            "canonical candidate run directory")
    lstat_dir(run, "candidate run directory", private=True)
    require(args.variant_manifest.resolve() == DEFAULT_MANIFEST.resolve(),
            "canonical source manifest path")
    require(args.control_events.resolve() == (CONTROL_RUN / "engine.jsonl").resolve() and
            args.control_summary.resolve() == (CONTROL_RUN / "summary.json").resolve(),
            "fixed control paths")
    return run


def validate(args: argparse.Namespace) -> dict[str, Any]:
    canonical_paths(args)
    validate_paths(args)
    bundle = parse_bundle(args.bundle)
    admission = validate_admission(bundle, args.admission)
    cases, final_delta, final_active = build_cases(bundle)
    require(env_need(admission, "final_active_set_sha256") == final_active and final_delta == 169,
            "admission final active binding")

    manifest = validate_manifest(args.variant_manifest)
    manifest_sha = sha256_file(args.variant_manifest)
    runner_source = manifest["artifacts"].get("runner_source")
    require(isinstance(runner_source, dict), "source manifest omits runner_source artifact")
    manifest_runner_sha = require_sha(runner_source.get("sha256"), "manifest runner source")

    guard_receipt = validate_guard_receipt(args.guard_receipt, args, manifest_sha)
    pending_receipt_sha256 = sha256_file(args.guard_receipt)

    engine = load_private_jsonl(args.events, "candidate semantic engine JSONL")
    summary = load_private_json(args.summary, "candidate semantic summary")
    engine_records, metrics = validate_semantic_engine_records(bundle, cases, engine)
    validate_semantic_engine_summary(summary, cases, metrics)

    for name, expected_hash in CONTROL_HASHES.items():
        control_path = CONTROL_RUN / name
        lstat_regular(control_path, "fixed historical reference " + name, private=True)
        require(sha256_file(control_path) == expected_hash,
                "fixed historical reference hash drift: " + name)
    control_engine = load_private_jsonl(args.control_events, "fixed historical reference JSONL")
    control_records = validate_legacy_control_records(bundle, cases, control_engine)
    comparison = compare_fixed_reference(engine_records, control_records)

    witness = load_private_jsonl(args.witness, "fresh allocation witness")
    witness_metrics = validate_witness(
        witness, args=args, bundle=bundle, cases=cases, manifest_sha=manifest_sha,
        manifest_source_closure_sha=manifest["full_source_closure_sha256"],
        manifest_runner_sha=manifest_runner_sha, guard_receipt=guard_receipt,
        engine_records=engine_records,
    )

    return {
        "schema": "fair-safe-c1-fresh-allocation-telemetry-output-validator-v2",
        "status": VALIDATOR_PASS_STATUS,
        "diagnostic_variant": VARIANT,
        "publication_eligible": False,
        "gpu_workload_launched_by_validator": False,
        "claim_scope": SEMANTIC_CLAIM_SCOPE,
        "scope": (
            "CPU-only sealed-input/oracle/ABBA/engine-binding/fresh-direct-allocation "
            "lifecycle validation; no performance conclusion or physical-device allocation "
            "proof beyond the inline raw-array preimages"
        ),
        "variant_manifest_sha256": manifest_sha,
        "manifest_full_source_closure_sha256": manifest["full_source_closure_sha256"],
        "pending_receipt_sha256": pending_receipt_sha256,
        "events_sha256": sha256_file(args.events),
        "summary_sha256": sha256_file(args.summary),
        "witness_sha256": sha256_file(args.witness),
        "fixed_reference": {
            "run_dir": str(CONTROL_RUN),
            "events_sha256": CONTROL_HASHES["engine.jsonl"],
            "summary_sha256": CONTROL_HASHES["summary.json"],
            "semantic_fingerprint_sha256": comparison["semantic_fingerprint_sha256"],
        },
        "comparison": comparison,
        "validated_semantic_records": len(engine),
        "validated_witness_queries": witness_metrics["witness_queries"],
        "validated_native_queries": witness_metrics["native_queries"],
        "validated_fallback_queries": witness_metrics["fallback_queries"],
        "witness_chain_sha256": witness_metrics["witness_chain_sha256"],
        "known_limits": [
            "inline raw arrays are checked against fixed domain-separated hashes and sealed-input structural preimages, but cannot prove physical device-page identity",
            "named direct cudaMalloc/cudaMallocManaged lifecycle does not observe allocations inside Thrust, CUDA runtime, or the driver",
            "hash-bound historical reference is not an acceptance oracle for repaired native leaf or candidate counts",
            "semantic control makes no performance conclusion",
        ],
    }


def self_check() -> dict[str, Any]:
    bundle = parse_bundle(BUNDLE_PATH)
    cases, final_delta, final_active = build_cases(bundle)
    return {
        "schema": "fair-safe-c1-fresh-allocation-telemetry-output-validator-v2",
        "status": VALIDATOR_SELFCHECK_STATUS,
        "diagnostic_variant": VARIANT,
        "publication_eligible": False,
        "claim_scope": SEMANTIC_CLAIM_SCOPE,
        "gpu_workload_launched": False,
        "knn_cases": len(cases),
        "insertions": final_delta,
        "final_active_set_sha256": final_active,
        "expected_witness_query_records": len(cases) * 4,
        "expected_native_fresh_lifecycles": len(cases) * 2,
        "first_witnessed_native_allocation_epoch": FIRST_WITNESSED_NATIVE_ALLOCATION_EPOCH,
        "last_witnessed_native_allocation_epoch": (
            FIRST_WITNESSED_NATIVE_ALLOCATION_EPOCH + len(cases) * 2 - 1
        ),
    }


def reject_duplicate_cli_options(argv: list[str]) -> None:
    """Reject repeated long options before argparse discards earlier values.

    Both spelling aliases for the immutable source manifest have one semantic
    destination, so ``--manifest`` plus ``--variant-manifest`` is also a
    duplicate.  ``--option=value`` is normalized to the same option token.
    """
    aliases = {"--manifest": "--variant-manifest"}
    seen: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("--") or token == "--":
            continue
        option = token.split("=", 1)[0]
        canonical = aliases.get(option, option)
        require(canonical not in seen, "duplicate CLI option: " + canonical)
        seen.add(canonical)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--bundle")
    parser.add_argument("--admission")
    parser.add_argument("--events")
    parser.add_argument("--summary")
    parser.add_argument("--witness")
    parser.add_argument("--guard-receipt")
    parser.add_argument("--control-events")
    parser.add_argument("--control-summary")
    parser.add_argument("--variant-manifest", dest="variant_manifest", default=str(DEFAULT_MANIFEST))
    # Kept as a local compatibility alias while callers migrate to the guard's
    # explicit spelling.  It has no effect on the fixed canonical path check.
    parser.add_argument("--manifest", dest="variant_manifest", default=argparse.SUPPRESS)
    parser.add_argument("--out")
    try:
        reject_duplicate_cli_options(sys.argv[1:])
    except Fail as exc:
        parser.error(str(exc))
    args = parser.parse_args()
    all_names = (
        "bundle", "admission", "events", "summary", "witness", "guard_receipt",
        "control_events", "control_summary", "out",
    )
    try:
        if args.self_check:
            require(all(getattr(args, name) is None for name in all_names),
                    "--self-check cannot accept run artifact arguments")
            print(json.dumps(self_check(), sort_keys=True))
            return 0
        require(all(getattr(args, name) is not None for name in all_names),
                "all run artifact arguments are required")
        result = validate(args)
        write_once(Path(args.out).absolute(), result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Fail as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
