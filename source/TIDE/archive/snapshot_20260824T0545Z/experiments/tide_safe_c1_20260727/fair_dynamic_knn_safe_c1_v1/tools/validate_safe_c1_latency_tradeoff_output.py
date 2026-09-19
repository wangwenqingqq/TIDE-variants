#!/usr/bin/python3.12
"""Independent CPU validator for guarded Safe-C1 latency--quality output.

This module deliberately does not import the CUDA runner or archived GTS code.
It replays the sealed frozen-base/insertion-only/KNN-only trace with an exact
integer squared-L2 oracle and validates the guard envelope, output hashes,
ABBA schedule, per-record API labels, summary aggregates, and result hashes.

It validates a latency--quality tradeoff only.  It must never be used to label
the two APIs as a fair same-API comparison or a speedup result.
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
from typing import Any

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/fair_dynamic_knn_safe_c1_v1")
ADMISSION_SCHEMA = "e1-frozen-base-knn-projection-admission-v2"
TRACE_MAGIC = b"E1GTRC01"
TRACE_HEADER = struct.Struct("<8sI6IfQ")
TRACE_EVENT = struct.Struct("<IB3xi")
INSERT = 1
KNN = 3
EXPECTED_GPU_ORDINAL = 1
EXPECTED_GPU_UUID = "GPU-CONFIGURE-ARCHIVE-DEVICE"
EXPECTED_SCHEMA = "fair-safe-c1-latency-tradeoff-e1-v1"
EXPECTED_GUARD_SCHEMA = "fair-safe-c1-latency-tradeoff-guard-v1"
EXPECTED_LABEL = "latency_quality_tradeoff_only"
EXPECTED_SCOPE = (
    "native query_knn candidate API versus current exact query_range(UINT64_MAX) "
    "full-result immutable-base fallback; not a fair same-API speedup"
)
EXPECTED_FALLBACK_PATH = "exact_full_immutable_base_range_fallback_no_gts_receipt"
EXPECTED_NATIVE_PATH = "real_gts_vector_topk_receipt_all_native_leaf_rows"
RESULT_HASH_DOMAIN = "fair-safe-c1-latency-tradeoff-e1-merged-result-v1"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
UUID_RE = re.compile(r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
PCI_RE = re.compile(r"^[0-9a-fA-F]{8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")

EXPECTED_ARTIFACT_HASHES = {
    "runner_source": "3a283f322ef2675b7e6d64062fb219b228375c4d15ebda1003b8f9f63766f9fb",
    "compile_helper": "63067e078cc9349dc5060d27c848d72ed696b29c4e37dd088ea66497576df382",
    "object": "ada205321f6abde0c7be6b354ec93bc59196b25cdf6001afb5558604b6862e1f",
    "binary": "af77e4dd6881e946f128f0908bcf003ea807698f7540a2101241035b4a25cc01",
    "preflight_tool": "62258f1461055744bfc7f74bb8825acb135291873f0289845038eb3ff0037b7d",
    "nvml_helper": "7d039a4317bf0f0ebed6d7fd1cb363d536b541ea1444caf3d73ee8098c755f34",
    "source_closure": "b528800f0a476ac70d7fedf7b7234470a975149d212ab29b31bdac4e9f7c864a",
}
EXPECTED_INPUT_HASHES = {
    "manifest.json": "68dbf15788793a828c8a9503d11304da576acbdca2f609dd0f8799ae39cd9a4c",
    "metadata.json": "2f515a5cf3bef61d6084f4c0d90075ee16cb4e365971b75102a24bdbbc588798",
    "trace.e1gtrc": "9402c609710fc9076f46653dd5bf30527013e158c63b4576dd665c27f9973ae5",
    "pool.i16": "899adaa59b265ee788841f1a48b667b7da39df166972ba9568c4e94727b31170",
    "queries.i16": "18e0ebbe8ddcdcf6e2312e1e48310111a4d96c1fcb752622d1e9ec9508c21c1e",
    "stable_id_to_pool_row.i32": "93710cce11c994b6b1934713842c93cfcec76a3563fc47574abf419137f4c5c8",
    "initial_base_stable_ids.i32": "6b0751ba5e64fc9c13ddfb44778fa7d6a1f7d7aa9d6a5e38a1f0a1502c3fb9e3",
}
SNAPSHOT_KEYS = (
    "schema", "gpu_ordinal", "gpu_uuid", "gpu_pci_bus_id", "compute_process_count",
    "graphics_process_count", "python_realpath", "python_sha256", "nvml_library_realpath",
    "nvml_library_sha256", "nvml_driver_version",
)
MEASUREMENT_KEYS = {
    "schema", "record", "not_same_api", "timing_claim", "condition", "pass",
    "case_ordinal", "op_index", "query_id", "external_global_delta_live",
    "api_host_ns", "post_api_external_delta_merge_excluded_from_api_host_ns",
    "api_result_count_before_adapter_truncation", "base_candidate_count",
    "visited_leaf_count", "full_immutable_base_candidate_count",
    "exact_full_immutable_base_fallback", "base_path", "merged_overlap_at_k",
    "merged_exact_match", "merged_result_sha256", "merged",
}


class Fail(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Fail(message)


def is_plain_int(value: Any, lower: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= lower


def require_sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA_RE.fullmatch(value) is not None, f"invalid SHA-256: {label}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_private_dir(path: Path, label: str) -> None:
    st = path.lstat()
    require(stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode), f"unsafe {label} directory")
    require(st.st_uid == 0 and st.st_gid == 0 and stat.S_IMODE(st.st_mode) == 0o700,
            f"{label} is not root-private 0700")


def require_private_regular(path: Path, label: str, executable: bool = False) -> None:
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode), f"unsafe/missing {label}: {path}")
    require(st.st_uid == 0 and st.st_gid == 0 and (stat.S_IMODE(st.st_mode) & 0o077) == 0,
            f"{label} is not root-private")
    if executable:
        require((stat.S_IMODE(st.st_mode) & 0o100) != 0, f"{label} is not owner executable")


def require_trusted_root_regular(path: Path, label: str) -> None:
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode), f"unsafe/missing trusted {label}: {path}")
    require(st.st_uid == 0 and st.st_gid == 0 and (stat.S_IMODE(st.st_mode) & 0o022) == 0,
            f"trusted {label} is writable by a non-root principal")


def safe_bundle_bytes(path: Path, label: str) -> bytes:
    require_trusted_root_regular(path, label)
    return path.read_bytes()


def safe_bytes(path: Path, label: str) -> bytes:
    require_private_regular(path, label)
    return path.read_bytes()


def read_env(path: Path) -> dict[str, str]:
    raw = safe_bytes(path, "admission").decode("ascii")
    values: dict[str, str] = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        require(line.count("=") == 1 and line.index("=") > 0, "malformed admission line")
        key, value = line.split("=", 1)
        require(key and value and key not in values and not any(ch.isspace() for ch in value),
                f"invalid admission field {key!r}")
        values[key] = value
    return values


def need(env: dict[str, str], key: str) -> str:
    require(key in env, f"admission omits {key}")
    return env[key]


def admission_int(env: dict[str, str], key: str) -> int:
    value = need(env, key)
    require(value.isdecimal(), f"non-decimal admission integer {key}")
    return int(value)


def read_le_array(path: Path, typecode: str, width: int, label: str) -> array.array:
    raw = safe_bundle_bytes(path, label)
    require(len(raw) % width == 0, f"{label} byte size is invalid")
    values = array.array(typecode)
    require(values.itemsize == width, f"host type width mismatch for {label}")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


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
    mapping: array.array
    base_ids: array.array


def parse_bundle(bundle: Path) -> Bundle:
    require_private_dir(bundle, "bundle")
    for filename, expected in EXPECTED_INPUT_HASHES.items():
        path = bundle / filename
        require_trusted_root_regular(path, "sealed input")
        require(sha256_file(path) == expected, f"sealed bundle drift: {filename}")
    raw = safe_bundle_bytes(bundle / "trace.e1gtrc", "trace")
    require(len(raw) >= TRACE_HEADER.size, "truncated trace")
    magic, version, dimension, base_n, reservoir_n, pool_n, query_n, k, _radius, event_count = (
        TRACE_HEADER.unpack_from(raw)
    )
    require(magic == TRACE_MAGIC and version == 1, "trace magic/version mismatch")
    require(dimension > 0 and base_n > 0 and pool_n >= base_n and reservoir_n == pool_n - base_n,
            "trace population shape mismatch")
    require(query_n > 0 and k > 0 and base_n >= k, "trace query shape mismatch")
    require(len(raw) == TRACE_HEADER.size + event_count * TRACE_EVENT.size, "trace byte size mismatch")
    events: list[tuple[int, int, int]] = []
    for position in range(event_count):
        op_index, opcode, argument = TRACE_EVENT.unpack_from(raw, TRACE_HEADER.size + position * TRACE_EVENT.size)
        require(op_index == position and opcode in (INSERT, KNN), f"illegal projected operation {position}")
        if opcode == INSERT:
            require(base_n <= argument < pool_n, f"illegal delta insertion {position}")
        else:
            require(0 <= argument < query_n, f"illegal KNN query ID {position}")
        events.append((op_index, opcode, argument))
    pool = read_le_array(bundle / "pool.i16", "h", 2, "pool")
    queries = read_le_array(bundle / "queries.i16", "h", 2, "queries")
    mapping = read_le_array(bundle / "stable_id_to_pool_row.i32", "i", 4, "mapping")
    base_ids = read_le_array(bundle / "initial_base_stable_ids.i32", "i", 4, "initial base IDs")
    require(len(pool) == pool_n * dimension and len(queries) == query_n * dimension,
            "matrix shape mismatches trace")
    require(len(mapping) == pool_n and len(base_ids) == base_n, "ID map/base IDs shape mismatch")
    require(len(set(mapping)) == pool_n and min(mapping) >= 0 and max(mapping) < pool_n,
            "stable-to-row map is not bijective")
    require(len(set(base_ids)) == base_n and min(base_ids) >= 0 and max(base_ids) < pool_n,
            "initial base IDs are invalid")
    return Bundle(dimension, base_n, pool_n, query_n, k, tuple(events), pool, queries, mapping, base_ids)


def exact_distance(bundle: Bundle, stable: int, query_id: int) -> int:
    require(0 <= stable < bundle.pool_n and 0 <= query_id < bundle.query_n, "distance ID outside bundle")
    row = int(bundle.mapping[stable])
    p = row * bundle.dimension
    q = query_id * bundle.dimension
    total = 0
    for coordinate in range(bundle.dimension):
        delta = int(bundle.pool[p + coordinate]) - int(bundle.queries[q + coordinate])
        total += delta * delta
    return total


def exact_topk(bundle: Bundle, active: bytearray, query_id: int) -> tuple[tuple[int, int], ...]:
    best: list[tuple[int, int]] = []
    for stable, present in enumerate(active):
        if not present:
            continue
        item = (exact_distance(bundle, stable, query_id), stable)
        if len(best) < bundle.k:
            best.append(item)
            if len(best) == bundle.k:
                best.sort()
        elif item < best[-1]:
            best[-1] = item
            best.sort()
    require(len(best) == bundle.k, "active population smaller than k")
    return tuple((stable, distance) for distance, stable in best)


def active_hash(active: bytearray) -> str:
    return hashlib.sha256(
        "".join(f"{stable}\n" for stable, present in enumerate(active) if present).encode("ascii")
    ).hexdigest()


def build_cases(bundle: Bundle) -> tuple[tuple[Case, ...], int, str]:
    active = bytearray(bundle.pool_n)
    for stable in bundle.base_ids:
        active[int(stable)] = 1
    delta_live = 0
    cases: list[Case] = []
    for op_index, opcode, argument in bundle.events:
        if opcode == INSERT:
            require(not active[argument], f"duplicate insertion at {op_index}")
            active[argument] = 1
            delta_live += 1
            continue
        cases.append(Case(
            ordinal=len(cases),
            op_index=op_index,
            query_id=argument,
            external_delta_live=delta_live,
            active=bytes(active),
            exact_topk=exact_topk(bundle, active, argument),
        ))
    return tuple(cases), delta_live, active_hash(active)


def parse_result_rows(value: Any, label: str, bundle: Bundle, case: Case) -> tuple[tuple[int, int], ...]:
    require(isinstance(value, list) and len(value) == bundle.k, f"{label} must contain exactly k rows")
    prior: tuple[int, int] | None = None
    seen: set[int] = set()
    rows: list[tuple[int, int]] = []
    for position, row in enumerate(value):
        require(isinstance(row, list) and len(row) == 2, f"{label}[{position}] is not a pair")
        stable, distance = row
        require(is_plain_int(stable) and stable < bundle.pool_n, f"{label}[{position}] invalid stable ID")
        require(is_plain_int(distance), f"{label}[{position}] invalid distance")
        require(stable not in seen and case.active[stable] == 1, f"{label}[{position}] not active/unique")
        expected_distance = exact_distance(bundle, stable, case.query_id)
        require(distance == expected_distance, f"{label}[{position}] distance is not exact int64 L2")
        key = (distance, stable)
        require(prior is None or prior < key, f"{label} is not canonical distance/stable order")
        seen.add(stable)
        prior = key
        rows.append((stable, distance))
    return tuple(rows)


def result_hash(rows: tuple[tuple[int, int], ...]) -> str:
    payload = RESULT_HASH_DOMAIN + "\n" + "".join(f"{stable}:{distance}\n" for stable, distance in rows)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def overlap(rows: tuple[tuple[int, int], ...], oracle: tuple[tuple[int, int], ...]) -> int:
    return len({stable for stable, _ in rows} & {stable for stable, _ in oracle})


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require_private_regular(path, "engine JSONL")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            require(line.endswith("\n"), f"JSONL record {line_number} lacks LF")
            require(line != "\n", f"JSONL record {line_number} is blank")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Fail(f"invalid JSONL at {line_number}: {exc}") from exc
            require(isinstance(row, dict), f"JSONL record {line_number} is not an object")
            records.append(row)
    return records


def snapshot_ok(snapshot: Any, label: str, expected_pci: str | None = None) -> str:
    require(isinstance(snapshot, dict) and set(snapshot) == set(SNAPSHOT_KEYS), f"{label} snapshot field set")
    require(snapshot["schema"] == "safe-c1-g3-nvml-snapshot-v2", f"{label} snapshot schema")
    require(snapshot["gpu_ordinal"] == str(EXPECTED_GPU_ORDINAL), f"{label} GPU ordinal")
    require(snapshot["gpu_uuid"] == EXPECTED_GPU_UUID and UUID_RE.fullmatch(snapshot["gpu_uuid"]) is not None,
            f"{label} GPU UUID")
    require(PCI_RE.fullmatch(snapshot["gpu_pci_bus_id"]) is not None, f"{label} PCI ID")
    if expected_pci is not None:
        require(snapshot["gpu_pci_bus_id"] == expected_pci, f"{label} PCI identity drift")
    require(snapshot["compute_process_count"] == "0" and snapshot["graphics_process_count"] == "0",
            f"{label} observes a GPU process")
    for key in ("python_realpath", "python_sha256", "nvml_library_realpath", "nvml_library_sha256",
                "nvml_driver_version"):
        require(isinstance(snapshot[key], str) and snapshot[key] and not any(ch.isspace() for ch in snapshot[key]),
                f"{label} invalid {key}")
    require_sha(snapshot["python_sha256"], label + " python sha")
    require_sha(snapshot["nvml_library_sha256"], label + " NVML sha")
    return snapshot["gpu_pci_bus_id"]


def read_json_private(path: Path, label: str) -> dict[str, Any]:
    require_private_regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Fail(f"invalid {label} JSON: {exc}") from exc
    require(isinstance(value, dict), f"{label} root is not object")
    return value


def validate_guard_receipt(args: argparse.Namespace, bundle: Bundle) -> dict[str, Any]:
    receipt = read_json_private(args.guard_receipt, "guard receipt")
    require(receipt.get("schema") == EXPECTED_GUARD_SCHEMA, "wrong guard receipt schema")
    require(receipt.get("status") == "RUNNER_EXITED_PENDING_VALIDATION",
            "receipt must be sealed after runner and before validation")
    require(receipt.get("scope") == EXPECTED_SCOPE, "receipt scope is not tradeoff-only")
    run_name = receipt.get("run_name")
    require(isinstance(run_name, str) and RUN_NAME_RE.fullmatch(run_name) is not None,
            "unsafe receipt run name")
    require(isinstance(receipt.get("approval_id"), str) and receipt["approval_id"], "missing approval ID")
    token = receipt.get("token")
    require(isinstance(token, dict) and set(token) == {
        "nonce_sha256", "issued_boottime_ns", "expires_boottime_ns", "ttl_seconds"
    }, "receipt token fields")
    require_sha(token["nonce_sha256"], "receipt nonce digest")
    require(is_plain_int(token["ttl_seconds"], 1) and token["ttl_seconds"] <= 120, "receipt TTL")
    require(is_plain_int(token["issued_boottime_ns"], 1) and is_plain_int(token["expires_boottime_ns"], 1),
            "receipt token clock fields")
    require(token["expires_boottime_ns"] - token["issued_boottime_ns"] ==
            token["ttl_seconds"] * 1_000_000_000, "receipt TTL envelope")

    gpu = receipt.get("gpu")
    require(isinstance(gpu, dict), "receipt GPU block missing")
    require(gpu.get("requested_ordinal") == EXPECTED_GPU_ORDINAL, "receipt GPU ordinal")
    require(gpu.get("expected_uuid") == EXPECTED_GPU_UUID and
            gpu.get("launch_cuda_visible_devices") == EXPECTED_GPU_UUID, "receipt GPU UUID binding")
    samples = gpu.get("prelaunch_idle_samples")
    require(isinstance(samples, list) and len(samples) == 3, "receipt must hold three prelaunch samples")
    pci: str | None = None
    last_time = -1
    for number, sample in enumerate(samples, 1):
        require(isinstance(sample, dict) and set(sample) == {
            "sample", "boottime_ns", "fields", "raw_sha256"
        }, "prelaunch sample envelope")
        require(sample["sample"] == number and is_plain_int(sample["boottime_ns"], 1) and
                sample["boottime_ns"] > last_time, "prelaunch sample ordering")
        last_time = sample["boottime_ns"]
        require_sha(sample["raw_sha256"], "prelaunch raw SHA")
        pci = snapshot_ok(sample["fields"], f"prelaunch-{number}", pci)
    postrun = gpu.get("postrun_idle_snapshot")
    require(isinstance(postrun, dict) and set(postrun) == {"boottime_ns", "fields", "raw_sha256"},
            "postrun snapshot envelope")
    require(is_plain_int(postrun["boottime_ns"], 1) and postrun["boottime_ns"] >= last_time,
            "postrun snapshot clock")
    require_sha(postrun["raw_sha256"], "postrun raw SHA")
    snapshot_ok(postrun["fields"], "postrun", pci)

    inputs = receipt.get("inputs")
    require(isinstance(inputs, dict), "receipt inputs missing")
    expected_paths = {
        "bundle_path": str(args.bundle.resolve()),
        "admission_path": str(args.admission.resolve()),
        "runner_source_path": str(ROOT / "runner/fair_safe_c1_latency_tradeoff_e1_runner.cu"),
        "compile_helper_path": str(ROOT / "tools/compile_safe_c1_latency_tradeoff.sh"),
        "object_path": str(ROOT / "build/fair_safe_c1_latency_tradeoff_e1_runner.sm_80.o"),
        "binary_path": str(ROOT / "bin/fair_safe_c1_latency_tradeoff_e1_runner.sm_80"),
        "preflight_tool_path": str(ROOT / "tools/preflight_frozen_base_knn_bundle.py"),
        "nvml_helper_path": str(ROOT / "tools/.nvml_idle_snapshot.py"),
        "source_closure_path": str(ROOT / "provenance/v24_unmodified_source_closure.sha256"),
    }
    for key, value in expected_paths.items():
        require(inputs.get(key) == value, f"receipt input path mismatch: {key}")
    for key, expected in EXPECTED_ARTIFACT_HASHES.items():
        receipt_key = key + "_sha256"
        require(inputs.get(receipt_key) == expected, f"receipt static artifact hash mismatch: {key}")
    require(inputs.get("input_files_sha256") == EXPECTED_INPUT_HASHES, "receipt bundle hash map mismatch")
    require(inputs.get("admission_sha256") == sha256_file(args.admission), "receipt admission hash mismatch")
    require(inputs.get("validator_sha256") == sha256_file(Path(__file__).resolve()),
            "receipt validator hash mismatches executing validator")
    require_sha(inputs.get("guard_sha256"), "receipt guard SHA")

    runner = receipt.get("runner")
    require(isinstance(runner, dict), "receipt runner block missing")
    require(runner.get("exit_code") == 0 and runner.get("timed_out") is False and
            runner.get("finished_within_ttl") is True, "runner status is not an admitted completion")
    require(is_plain_int(runner.get("started_boottime_ns"), 1) and
            is_plain_int(runner.get("finished_boottime_ns"), 1) and
            runner["finished_boottime_ns"] >= runner["started_boottime_ns"], "runner clock envelope")
    require(runner["finished_boottime_ns"] <= token["expires_boottime_ns"], "runner exceeded TTL")
    require(runner.get("cuda_visible_devices") == EXPECTED_GPU_UUID and
            runner.get("nvidia_visible_devices") == EXPECTED_GPU_UUID, "runner visibility binding")
    require(runner.get("warmup_passes") == 1 and runner.get("measured_passes") == 1,
            "unapproved pilot pass count")
    require_sha(runner.get("events_sha256"), "receipt events SHA")
    require_sha(runner.get("summary_sha256"), "receipt summary SHA")
    require(runner["events_sha256"] == sha256_file(args.events) and
            runner["summary_sha256"] == sha256_file(args.summary), "output hash does not match receipt")

    outputs = receipt.get("outputs")
    require(isinstance(outputs, dict) and outputs.get("events") == str(args.events.resolve()) and
            outputs.get("summary") == str(args.summary.resolve()) and
            outputs.get("validator") == str(args.out.resolve()), "receipt output paths")
    return receipt


def validate_admission(bundle: Bundle, admission: Path) -> dict[str, str]:
    env = read_env(admission)
    require(need(env, "schema") == ADMISSION_SCHEMA and need(env, "status") == "PASS",
            "admission schema/status")
    require(Path(need(env, "bundle_realpath")).resolve() == ROOT / "inputs/e1_frozen_base_knn_projection_v1",
            "admission bundle path")
    require(need(env, "ops") == "insert,knn" and need(env, "excluded_ops") == "delete,range",
            "admission operation projection")
    require(need(env, "base_immutable") == "true" and
            need(env, "direct_sidecar_allowed") == "false" and
            need(env, "legacy_routing_allowed") == "false", "admission Safe-C1 policy")
    checks = {
        "dimension": bundle.dimension,
        "base_n": bundle.base_n,
        "pool_n": bundle.pool_n,
        "query_n": bundle.query_n,
        "k": bundle.k,
        "event_count": len(bundle.events),
        "insert_count": 169,
        "knn_count": 136,
        "final_active_count": 4265,
    }
    for key, expected in checks.items():
        require(admission_int(env, key) == expected, f"admission mismatch: {key}")
    for filename, expected in EXPECTED_INPUT_HASHES.items():
        key = {
            "manifest.json": "manifest_sha256", "metadata.json": "metadata_sha256",
            "trace.e1gtrc": "trace_sha256", "pool.i16": "pool_sha256",
            "queries.i16": "queries_sha256", "stable_id_to_pool_row.i32": "mapping_sha256",
            "initial_base_stable_ids.i32": "base_ids_sha256",
        }[filename]
        require(need(env, key) == expected, f"admission hash mismatch: {key}")
    require(need(env, "projection_event_stream_sha256") ==
            "0d7267e098445723c7c065e9937206e37eca040b27025ad98b979c9a9af16b64",
            "admission projected event hash")
    require(need(env, "final_active_set_sha256") ==
            "281d47954a4bbb2a85bafb09e150dd72e2ce4f9086568cf1a1a77d5657fa8e99",
            "admission final active hash")
    return env


def distribution(values: list[int]) -> dict[str, int]:
    require(values, "empty timing distribution")
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min_ns": ordered[0],
        "p50_ns": ordered[(50 * (len(ordered) - 1)) // 100],
        "p95_ns": ordered[(95 * (len(ordered) - 1)) // 100],
        "max_ns": ordered[-1],
        "mean_ns": sum(ordered) // len(ordered),
    }


def validate_records(bundle: Bundle, cases: tuple[Case, ...], records: list[dict[str, Any]],
                     measured_passes: int) -> dict[str, Any]:
    require(len(records) == measured_passes * len(cases) * 2, "wrong measurement record count")
    native_values: list[int] = []
    fallback_values: list[int] = []
    native_exact = 0
    native_overlap = 0
    fallback_exact = 0
    sequence = hashlib.sha256()
    cursor = 0
    for pass_number in range(measured_passes):
        for case in cases:
            first_native = ((pass_number + case.ordinal) % 2) == 0
            expected_conditions = (
                ("native_query_knn_candidate", "exact_query_range_full_base_fallback")
                if first_native else
                ("exact_query_range_full_base_fallback", "native_query_knn_candidate")
            )
            for condition in expected_conditions:
                record = records[cursor]
                cursor += 1
                require(set(record) == MEASUREMENT_KEYS, f"measurement field set at record {cursor}")
                require(record.get("schema") == EXPECTED_SCHEMA and record.get("record") == "measurement",
                        f"measurement schema at record {cursor}")
                require(record.get("not_same_api") is True and record.get("timing_claim") == EXPECTED_LABEL,
                        f"measurement claim label at record {cursor}")
                require(record.get("condition") == condition and record.get("pass") == pass_number and
                        record.get("case_ordinal") == case.ordinal and record.get("op_index") == case.op_index and
                        record.get("query_id") == case.query_id and
                        record.get("external_global_delta_live") == case.external_delta_live,
                        f"measurement schedule/trace binding at record {cursor}")
                require(is_plain_int(record.get("api_host_ns")) and
                        record.get("post_api_external_delta_merge_excluded_from_api_host_ns") is True,
                        f"API timing field at record {cursor}")
                rows = parse_result_rows(record.get("merged"), f"merged record {cursor}", bundle, case)
                require(record.get("merged_result_sha256") == result_hash(rows),
                        f"merged result hash at record {cursor}")
                actual_overlap = overlap(rows, case.exact_topk)
                exact = rows == case.exact_topk
                require(record.get("merged_overlap_at_k") == actual_overlap and
                        record.get("merged_exact_match") is exact,
                        f"merged quality accounting at record {cursor}")
                if condition == "native_query_knn_candidate":
                    require(record.get("api_result_count_before_adapter_truncation") == bundle.k and
                            is_plain_int(record.get("base_candidate_count"), bundle.k) and
                            record["base_candidate_count"] <= bundle.base_n and
                            is_plain_int(record.get("visited_leaf_count")) and
                            record.get("full_immutable_base_candidate_count") == 0 and
                            record.get("exact_full_immutable_base_fallback") is False and
                            record.get("base_path") == EXPECTED_NATIVE_PATH,
                            f"native API contract at record {cursor}")
                    native_values.append(record["api_host_ns"])
                    native_exact += int(exact)
                    native_overlap += actual_overlap
                else:
                    require(record.get("api_result_count_before_adapter_truncation") == bundle.base_n and
                            record.get("base_candidate_count") == bundle.base_n and
                            is_plain_int(record.get("visited_leaf_count")) and
                            record.get("full_immutable_base_candidate_count") == bundle.base_n and
                            record.get("exact_full_immutable_base_fallback") is True and
                            record.get("base_path") == EXPECTED_FALLBACK_PATH and exact,
                            f"exact fallback API contract at record {cursor}")
                    fallback_values.append(record["api_host_ns"])
                    fallback_exact += 1
                sequence.update(
                    (f"{pass_number}:{case.ordinal}:{condition}:{case.op_index}:{case.query_id}:").encode("ascii")
                )
                for stable, distance in rows:
                    sequence.update(f"{stable}:{distance},".encode("ascii"))
                sequence.update(b"\n")
    return {
        "native_values": native_values,
        "fallback_values": fallback_values,
        "native_exact_set_count": native_exact,
        "native_overlap_sum": native_overlap,
        "fallback_exact_set_count": fallback_exact,
        "sequence_sha256": sequence.hexdigest(),
    }


def validate_summary(summary: dict[str, Any], measured_passes: int, cases: tuple[Case, ...],
                     metrics: dict[str, Any]) -> None:
    expected_keys = {
        "schema", "mode", "status", "not_same_api", "timing_claim", "scope",
        "warmup_passes_discarded", "measured_passes", "measurement_records",
        "post_api_external_delta_merge_excluded_from_api_host_ns",
        "json_and_oracle_excluded_from_api_host_ns", "native_api_host_ns",
        "fallback_api_host_ns", "quality", "limitations",
    }
    require(set(summary) == expected_keys, "summary field set")
    require(summary.get("schema") == EXPECTED_SCHEMA and summary.get("mode") == "timing" and
            summary.get("status") == "PASS_TRADEOFF_TIMING", "summary schema/status")
    require(summary.get("not_same_api") is True and summary.get("timing_claim") == EXPECTED_LABEL and
            summary.get("scope") == EXPECTED_SCOPE, "summary claim/scope")
    require(summary.get("warmup_passes_discarded") == 1 and
            summary.get("measured_passes") == measured_passes and
            summary.get("measurement_records") == measured_passes * len(cases) * 2,
            "summary pass/count fields")
    require(summary.get("post_api_external_delta_merge_excluded_from_api_host_ns") is True and
            summary.get("json_and_oracle_excluded_from_api_host_ns") is True,
            "summary timing envelope")
    require(summary.get("native_api_host_ns") == distribution(metrics["native_values"]) and
            summary.get("fallback_api_host_ns") == distribution(metrics["fallback_values"]),
            "summary distributions")
    quality = summary.get("quality")
    expected_quality = {
        "native_exact_set_count": metrics["native_exact_set_count"],
        "native_overlap_sum": metrics["native_overlap_sum"],
        "fallback_exact_set_count": metrics["fallback_exact_set_count"],
        "fallback_expected_exact_set_count": measured_passes * len(cases),
    }
    require(quality == expected_quality, "summary quality aggregate")
    require(summary.get("limitations") == [
        "not_same_api", "no CUDA event/kernel timing", "external delta is CPU wrapper work",
        "no delete/range-workload/rebuild/direct-sidecar claim",
    ], "summary limitations")


def write_once(path: Path, value: dict[str, Any]) -> None:
    require(path.parent == path.parent.resolve() and path.parent.is_dir(), "validator output parent")
    require_private_dir(path.parent, "validator output parent")
    require(not path.exists() and not path.is_symlink(), "validator refuses overwrite")
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".latency-validator.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
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


def validate(args: argparse.Namespace) -> dict[str, Any]:
    args.bundle = args.bundle.resolve()
    args.admission = args.admission.resolve()
    args.events = args.events.resolve()
    args.summary = args.summary.resolve()
    args.guard_receipt = args.guard_receipt.resolve()
    args.out = args.out.resolve()
    require(args.bundle == ROOT / "inputs/e1_frozen_base_knn_projection_v1", "wrong sealed bundle path")
    require(args.events.parent == args.summary.parent == args.guard_receipt.parent == args.out.parent,
            "outputs/receipt must share one run directory")
    require(args.events.name == "engine.jsonl" and args.summary.name == "summary.json" and
            args.guard_receipt.name == "guard_receipt.json" and args.out.name == "independent_validation.json",
            "noncanonical guarded output names")
    require_private_dir(args.events.parent, "run")
    bundle = parse_bundle(args.bundle)
    admission = validate_admission(bundle, args.admission)
    cases, inserts, final_hash = build_cases(bundle)
    require(inserts == admission_int(admission, "insert_count") and
            len(cases) == admission_int(admission, "knn_count") and
            final_hash == need(admission, "final_active_set_sha256"), "replayed trace witness")
    receipt = validate_guard_receipt(args, bundle)
    records = load_jsonl(args.events)
    summary = read_json_private(args.summary, "timing summary")
    measured = receipt["runner"]["measured_passes"]
    metrics = validate_records(bundle, cases, records, measured)
    validate_summary(summary, measured, cases, metrics)
    return {
        "schema": "fair-safe-c1-latency-tradeoff-output-validator-v1",
        "status": "PASS",
        "scope": "independent exact-oracle validation of a non-equivalent latency-quality tradeoff; not a same-API speedup",
        "guard_receipt_sha256_at_validation": sha256_file(args.guard_receipt),
        "events_sha256": sha256_file(args.events),
        "summary_sha256": sha256_file(args.summary),
        "trace_sha256": EXPECTED_INPUT_HASHES["trace.e1gtrc"],
        "validated_measurements": len(records),
        "measured_passes": measured,
        "knn_cases_per_pass": len(cases),
        "native_exact_set_count": metrics["native_exact_set_count"],
        "native_overlap_sum": metrics["native_overlap_sum"],
        "fallback_exact_set_count": metrics["fallback_exact_set_count"],
        "fallback_expected_exact_set_count": measured * len(cases),
        "measurement_sequence_sha256": metrics["sequence_sha256"],
        "oracle": "pure-stdlib exact integer squared L2; canonical order=(distance_sq,stable_id); validates output hashes, receipt envelope, ABBA order, result hashes, and summary aggregates",
    }


def validator_self_check() -> int:
    bundle = parse_bundle(ROOT / "inputs/e1_frozen_base_knn_projection_v1")
    cases, inserts, final_hash = build_cases(bundle)
    require(len(bundle.events) == 305 and inserts == 169 and len(cases) == 136,
            "sealed trace cardinality drift")
    require(final_hash == "281d47954a4bbb2a85bafb09e150dd72e2ce4f9086568cf1a1a77d5657fa8e99",
            "sealed final active-set witness drift")
    print(json.dumps({
        "schema": "fair-safe-c1-latency-tradeoff-output-validator-v1",
        "status": "PASS_CPU_ONLY_ORACLE_SELFCHECK",
        "gpu_workload_launched": False,
        "validated_trace_events": len(bundle.events),
        "insert_count": inserts,
        "knn_cases": len(cases),
        "final_active_set_sha256": final_hash,
        "scope": "sealed-bundle exact-oracle replay only; no timing artifact accepted and no CUDA path invoked",
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--guard-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        if args.self_check:
            require(all(value is None for value in (
                args.bundle, args.admission, args.events, args.summary, args.guard_receipt, args.out
            )), "--self-check cannot accept timing-output arguments")
            return validator_self_check()
        require(all(value is not None for value in (
            args.bundle, args.admission, args.events, args.summary, args.guard_receipt, args.out
        )), "bundle/admission/events/summary/guard-receipt/out are all required")
        result = validate(args)
        write_once(args.out, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Fail as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
