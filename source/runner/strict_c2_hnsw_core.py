#!/usr/bin/env python3
"""Shared CPU-only strict-C2 HNSW partition runner core for frozen-base E1.

This is intentionally a *separate* comparator from the historical HNSW timing
adapter.  It has the same Safe-C1 replay semantics as the native-GTS recall
probe:

* HNSW receives exactly the initial immutable base (``max_elements=base_n``).
* Projected insertions are retained only in an external exact ``global_delta``;
  the trace never calls ``Index.add_items``, delete, resize, or any other HNSW
  mutation API.
* All projected insertions are replayed in order.  Only presealed partition
  KNN ordinals obtain HNSW base candidates; skipped-partition KNNs never read a
  query vector, invoke HNSW, invoke an oracle, or emit a record.
* Each selected answer is re-scored with int64 squared L2 in canonical
  ``(distance_sq, stable_id)`` order, merged with the exact delta, and compared
  with exhaustive immutable-base/full-active int64 oracles.  This is a
  partition-local recall/quality diagnostic, not an exactness, dynamic-index,
  GPU, or timing claim.

The module imports hnswlib/numpy only after validating the sealed input bundle
and after checking that both packages come from a root-private venv supplied by
``--runtime-root``.  Run it with that venv's interpreter in isolated mode
(``python -I``).  No user-owned ``bench_env`` is ever accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import stat
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


MAGIC = b"E1GTRC01"
TRACE_VERSION = 1
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3si")
INSERT = 1
KNN = 3

PROJECTION_SCHEMA = "e1-frozen-base-knn-projection-manifest-v1"
METADATA_SCHEMA = "e1-frozen-base-knn-projection-bundle-v1"
ADMISSION_SCHEMA = "e1-frozen-base-knn-projection-admission-v2"
RUN_SCHEMA = "fair-hnsw-strict-c2-partition-quality-probe-v1"
PARTITION_SEAL_SCHEMA = "safe-c1-hnsw-strict-c2-partition-seal-v1"
LOCK_SCHEMA = "safe-c1-hnsw-strict-c2-heldout-lock-v1"
SOURCE_SPLIT_SCHEMA = "safe-c1-hnsw-calibration-test-split-v1"
SOURCE_SPLIT_FILE_SHA256 = "92394b5aa7cee61d6f2583b08061d30a1a287ef5776c7dbe5c9a6a79ddadd585"
SOURCE_SPLIT_CANONICAL_SHA256 = "c825f1e29f80fcb93b0d302e8a3da3273de0de578a839bd72d86eceb5b532355"

HNSW_VERSION = "0.8.0"
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_RANDOM_SEED = 20260727
HNSW_THREADS = 1

REQUIRED_PAYLOADS = {
    "metadata.json",
    "trace.e1gtrc",
    "pool.i16",
    "queries.i16",
    "stable_id_to_pool_row.i32",
    "initial_base_stable_ids.i32",
}


class Fail(RuntimeError):
    """Fail closed before emitting a result that could be mistaken for evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Fail(message)


def _lstat(path: Path, label: str) -> os.stat_result:
    try:
        return path.lstat()
    except FileNotFoundError as exc:
        raise Fail(f"missing {label}: {path}") from exc


def require_real_directory(path: Path, label: str) -> Path:
    st = _lstat(path, label)
    require(not stat.S_ISLNK(st.st_mode) and stat.S_ISDIR(st.st_mode),
            f"{label} must be a real non-symlink directory: {path}")
    return path.resolve(strict=True)


def require_real_regular_file(path: Path, label: str) -> Path:
    st = _lstat(path, label)
    require(not stat.S_ISLNK(st.st_mode) and stat.S_ISREG(st.st_mode),
            f"{label} must be a real non-symlink regular file: {path}")
    return path.resolve(strict=True)


def read_regular_bytes(path: Path, label: str) -> bytes:
    require_real_regular_file(path, label)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise Fail(f"cannot read {label}: {path}: {exc}") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, label: str) -> str:
    require_real_regular_file(path, label)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Fail(f"cannot hash {label}: {path}: {exc}") from exc
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(ch in "0123456789abcdef" for ch in value),
            f"{label} must be a lowercase SHA-256 digest")
    return value


def require_uint(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{label} must be a nonnegative integer")
    return int(value)


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = read_regular_bytes(path, label)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Fail(f"invalid JSON in {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def parse_env(path: Path) -> dict[str, str]:
    payload = read_regular_bytes(path, "preflight admission")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Fail("preflight admission is not UTF-8") from exc
    result: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        require("=" in line, f"malformed admission line {line_number}")
        key, value = line.split("=", 1)
        require(key and value and key not in result,
                f"empty or duplicate admission key at line {line_number}")
        result[key] = value
    return result


def env_need(env: dict[str, str], key: str) -> str:
    value = env.get(key)
    require(value is not None and value != "", f"preflight admission omits {key}")
    return value


def env_uint(env: dict[str, str], key: str) -> int:
    value = env_need(env, key)
    require(value.isdecimal(), f"preflight {key} is not a decimal integer")
    return int(value)


def env_sha(env: dict[str, str], key: str) -> str:
    return require_sha256(env_need(env, key), f"preflight {key}")


def simple_filename(name: Any, label: str) -> str:
    require(isinstance(name, str) and name not in {"", ".", ".."} and
            "/" not in name and "\\" not in name and Path(name).name == name,
            f"unsafe {label}: {name!r}")
    return name


def canonical_active_hash(active: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for stable_id in sorted(active):
        digest.update(f"{stable_id}\n".encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TraceEvent:
    op_index: int
    opcode: int
    argument: int


@dataclass(frozen=True)
class Header:
    dimension: int
    base_n: int
    reservoir_n: int
    pool_n: int
    query_n: int
    k: int
    radius: float
    event_count: int


@dataclass(frozen=True)
class Bundle:
    path: Path
    header: Header
    events: tuple[TraceEvent, ...]
    pool_bytes: bytes
    queries_bytes: bytes
    stable_to_pool_row: tuple[int, ...]
    base_ids: tuple[int, ...]
    metadata_sha256: str
    manifest_sha256: str
    trace_sha256: str
    projection_event_stream_sha256: str
    source_trace_sha256: str
    source_event_stream_sha256: str
    final_active_count: int
    final_active_set_sha256: str
    insert_count: int
    knn_count: int



@dataclass(frozen=True)
class RunArgs:
    bundle: Path
    preflight: Path
    out: Path
    summary: Path
    runtime_root: Path
    partition_seal: Path
    ef_search: int
    partition: str
    entrypoint: Path
    lock_receipt: Path | None


@dataclass(frozen=True)
class Runtime:
    np: Any
    hnswlib: Any
    runtime_root: Path
    hnswlib_version: str
    numpy_version: str
    hnswlib_module_path: Path
    hnswlib_module_sha256: str
    numpy_module_path: Path



def validate_new_output(path: Path, label: str) -> None:
    require(not os.path.lexists(path), f"{label} already exists; refusing overwrite: {path}")
    parent = path.parent if path.parent != Path("") else Path(".")
    parent = require_real_directory(parent, f"{label} parent")
    parent_st = _lstat(parent, f"{label} parent")
    require(parent_st.st_uid == 0 and (parent_st.st_mode & 0o077) == 0,
            f"{label} parent must be root-owned and private")


def validate_output_targets(args: RunArgs) -> None:
    validate_new_output(args.out, "JSONL output")
    validate_new_output(args.summary, "summary output")
    require(args.out.resolve(strict=False) != args.summary.resolve(strict=False),
            "JSONL output and summary output alias")


def parse_trace(trace_bytes: bytes) -> tuple[Header, tuple[TraceEvent, ...], str]:
    require(len(trace_bytes) >= HEADER.size, "trace is shorter than its fixed header")
    magic, version, dimension, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = \
        HEADER.unpack_from(trace_bytes, 0)
    require(magic == MAGIC and version == TRACE_VERSION,
            "trace magic/version differs from frozen E1 projection contract")
    require(dimension > 0 and base_n > 0 and reservoir_n >= 0 and
            pool_n == base_n + reservoir_n and query_n > 0 and
            k > 0 and base_n >= k,
            "invalid frozen-base trace header")
    require(len(trace_bytes) == HEADER.size + EVENT.size * event_count,
            "trace byte count disagrees with event_count")
    events: list[TraceEvent] = []
    for position in range(event_count):
        op_index, opcode, reserved, argument = EVENT.unpack_from(
            trace_bytes, HEADER.size + position * EVENT.size)
        require(op_index == position and reserved == b"\0\0\0",
                f"trace event {position} has noncanonical encoding")
        require(opcode in {INSERT, KNN},
                f"trace event {position} is forbidden opcode={opcode}; delete/range must be absent")
        events.append(TraceEvent(int(op_index), int(opcode), int(argument)))
    header = Header(int(dimension), int(base_n), int(reservoir_n), int(pool_n),
                    int(query_n), int(k), float(radius), int(event_count))
    return header, tuple(events), sha256_bytes(trace_bytes[HEADER.size:])


def parse_i32(raw: bytes, count: int, label: str) -> tuple[int, ...]:
    require(len(raw) == 4 * count, f"{label} byte count is invalid")
    return tuple(struct.unpack(f"<{count}i", raw))


def validate_admission(args: RunArgs) -> dict[str, str]:
    bundle = require_real_directory(args.bundle, "bundle")
    admission = require_real_regular_file(args.preflight, "preflight admission")
    env = parse_env(admission)
    required_strings = {
        "schema": ADMISSION_SCHEMA,
        "status": "PASS",
        "projection_manifest_schema": PROJECTION_SCHEMA,
        "projection_metadata_schema": METADATA_SCHEMA,
        "ops": "insert,knn",
        "excluded_ops": "delete,range",
        "base_immutable": "true",
        "direct_sidecar_allowed": "false",
        "legacy_routing_allowed": "false",
    }
    for key, wanted in required_strings.items():
        require(env_need(env, key) == wanted, f"preflight policy drift for {key}")
    require(Path(env_need(env, "bundle_realpath")).resolve(strict=True) == bundle,
            "preflight bundle_realpath does not match --bundle")
    for key in ("dimension", "base_n", "pool_n", "query_n", "k", "event_count",
                "insert_count", "knn_count", "final_active_count", "source_event_count"):
        env_uint(env, key)
    require(env_uint(env, "source_event_count") > env_uint(env, "event_count"),
            "projection must have fewer events than its source trace")
    for key in ("manifest_sha256", "metadata_sha256", "trace_sha256", "pool_sha256",
                "queries_sha256", "mapping_sha256", "base_ids_sha256",
                "projection_event_stream_sha256", "final_active_set_sha256",
                "source_trace_sha256", "source_event_stream_sha256"):
        env_sha(env, key)
    return env


def validate_bundle(args: RunArgs, env: dict[str, str]) -> Bundle:
    bundle = require_real_directory(args.bundle, "bundle")
    manifest_path = bundle / "manifest.json"
    metadata_path = bundle / "metadata.json"
    manifest = read_json_object(manifest_path, "manifest.json")
    metadata = read_json_object(metadata_path, "metadata.json")
    require(manifest.get("schema") == PROJECTION_SCHEMA,
            "manifest schema is not the frozen-base KNN projection schema")
    require(metadata.get("schema") == METADATA_SCHEMA,
            "metadata schema is not the frozen-base KNN projection bundle schema")

    manifest_sha = sha256_file(manifest_path, "manifest.json")
    metadata_sha = sha256_file(metadata_path, "metadata.json")
    require(manifest_sha == env_sha(env, "manifest_sha256"), "manifest hash drifted after admission")
    require(metadata_sha == env_sha(env, "metadata_sha256"), "metadata hash drifted after admission")

    declared = manifest.get("files_sha256")
    require(isinstance(declared, dict) and declared,
            "manifest.files_sha256 must be a nonempty object")
    declared_names: set[str] = set()
    observed_payloads: dict[str, str] = {}
    for name, wanted_hash in declared.items():
        name = simple_filename(name, "manifest.files_sha256 key")
        require(name not in declared_names, f"duplicate manifest file name: {name}")
        declared_names.add(name)
        wanted_hash = require_sha256(wanted_hash, f"manifest hash for {name}")
        actual = sha256_file(bundle / name, f"manifest payload {name}")
        require(actual == wanted_hash, f"manifest payload hash mismatch: {name}")
        observed_payloads[name] = actual
    missing = REQUIRED_PAYLOADS - declared_names
    require(not missing, f"manifest omits required payloads: {sorted(missing)}")

    trace_path = bundle / "trace.e1gtrc"
    trace_bytes = read_regular_bytes(trace_path, "trace.e1gtrc")
    header, events, event_stream_sha = parse_trace(trace_bytes)
    trace_sha = sha256_bytes(trace_bytes)
    require(trace_sha == env_sha(env, "trace_sha256"), "trace hash drifted after admission")
    require(observed_payloads["trace.e1gtrc"] == trace_sha,
            "manifest and observed trace hashes disagree")
    require(manifest.get("projection_trace_sha256") == trace_sha and
            manifest.get("source_trace_sha256") == env_sha(env, "source_trace_sha256"),
            "manifest source/projection trace provenance drift")

    for env_key, filename in {
        "pool_sha256": "pool.i16",
        "queries_sha256": "queries.i16",
        "mapping_sha256": "stable_id_to_pool_row.i32",
        "base_ids_sha256": "initial_base_stable_ids.i32",
    }.items():
        require(observed_payloads.get(filename) == env_sha(env, env_key),
                f"manifest/preflight disagreement for {filename}")

    pool_bytes = read_regular_bytes(bundle / "pool.i16", "pool.i16")
    queries_bytes = read_regular_bytes(bundle / "queries.i16", "queries.i16")
    mapping_bytes = read_regular_bytes(bundle / "stable_id_to_pool_row.i32",
                                       "stable_id_to_pool_row.i32")
    base_bytes = read_regular_bytes(bundle / "initial_base_stable_ids.i32",
                                    "initial_base_stable_ids.i32")
    require(len(pool_bytes) == 2 * header.pool_n * header.dimension,
            "pool.i16 shape disagrees with trace header")
    require(len(queries_bytes) == 2 * header.query_n * header.dimension,
            "queries.i16 shape disagrees with trace header")
    mapping = parse_i32(mapping_bytes, header.pool_n, "stable_id_to_pool_row.i32")
    base_ids = parse_i32(base_bytes, header.base_n, "initial_base_stable_ids.i32")
    require(all(0 <= row < header.pool_n for row in mapping) and
            len(set(mapping)) == header.pool_n,
            "stable-ID to pool-row mapping is not a bijection")
    require(all(0 <= stable_id < header.pool_n for stable_id in base_ids) and
            len(set(base_ids)) == header.base_n,
            "initial immutable base has invalid or duplicate stable IDs")

    # Independently replay only the explicitly admitted projection.  This does
    # not consume the copied original quantized oracle; that file is hashed as a
    # manifest payload only and never parsed as a projection oracle.
    active: set[int] = set(base_ids)
    inserts = 0
    knns = 0
    for event in events:
        if event.opcode == INSERT:
            require(0 <= event.argument < header.pool_n and event.argument not in active,
                    f"op {event.op_index}: invalid/repeated projected insertion")
            active.add(event.argument)
            inserts += 1
        else:
            require(0 <= event.argument < header.query_n,
                    f"op {event.op_index}: KNN query ID outside query matrix")
            knns += 1

    projection = metadata.get("projection")
    require(isinstance(projection, dict), "metadata.projection must be an object")
    require(projection.get("allowed_ops") == ["insert", "knn"] and
            projection.get("excluded_ops") == ["delete", "range"],
            "metadata projection operation contract drift")
    require(projection.get("source_quantized_oracle_role") ==
            "source provenance only; never a projection oracle",
            "metadata must explicitly forbid use of source quantized oracle")
    require(projection.get("projection_trace_sha256") == trace_sha and
            projection.get("projection_event_stream_sha256") == event_stream_sha,
            "metadata projection trace/event stream provenance drift")
    require(projection.get("stable_id_mapping_sha256") == env_sha(env, "mapping_sha256") and
            projection.get("initial_base_stable_ids_sha256") == env_sha(env, "base_ids_sha256"),
            "metadata stable-ID provenance drift")
    require(require_uint(projection.get("source_indices_count"),
                         "metadata.projection.source_indices_count") == len(events),
            "metadata projection source index count drift")
    for key in ("source_indices_sha256",):
        require_sha256(projection.get(key), f"metadata.projection.{key}")
    for key in ("first_source_index", "last_source_index"):
        value = require_uint(projection.get(key), f"metadata.projection.{key}")
        require(value < env_uint(env, "source_event_count"),
                f"metadata.projection.{key} lies outside declared source trace")
    require(projection["first_source_index"] <= projection["last_source_index"],
            "metadata projected source interval is reversed")
    final_hash = canonical_active_hash(active)
    require(require_uint(projection.get("final_active_count"),
                         "metadata.projection.final_active_count") == len(active) and
            require_sha256(projection.get("final_active_set_sha256"),
                           "metadata.projection.final_active_set_sha256") == final_hash,
            "metadata final active-state witness drift")

    meta_header = metadata.get("header")
    source_header = metadata.get("source_header")
    require(isinstance(meta_header, dict) and isinstance(source_header, dict),
            "metadata must contain projected and source headers")
    header_fields = {
        "magic": "E1GTRC01", "version": TRACE_VERSION,
        "dimension": header.dimension, "base_n": header.base_n,
        "reservoir_n": header.reservoir_n, "pool_n": header.pool_n,
        "query_n": header.query_n, "k": header.k, "event_count": header.event_count,
    }
    for key, wanted in header_fields.items():
        require(meta_header.get(key) == wanted, f"metadata.header.{key} drift")
    require(meta_header.get("radius") == header.radius, "metadata.header.radius drift")
    for key, wanted in header_fields.items():
        if key == "event_count":
            continue
        require(source_header.get(key) == wanted, f"metadata.source_header.{key} drift")
    require(source_header.get("radius") == header.radius and
            require_uint(source_header.get("event_count"), "metadata.source_header.event_count") ==
            env_uint(env, "source_event_count"),
            "metadata source header drift")

    trace_counts = metadata.get("trace_counts")
    require(isinstance(trace_counts, dict) and isinstance(trace_counts.get("source"), dict) and
            isinstance(trace_counts.get("projection"), dict),
            "metadata.trace_counts must contain source and projection objects")
    source_counts = trace_counts["source"]
    projection_counts = trace_counts["projection"]
    for name in ("insert", "delete", "knn", "range"):
        require_uint(source_counts.get(name), f"metadata.trace_counts.source.{name}")
    require(sum(int(source_counts[name]) for name in ("insert", "delete", "knn", "range")) ==
            env_uint(env, "source_event_count") and int(source_counts["delete"]) > 0 and
            int(source_counts["range"]) > 0,
            "metadata source trace counts are incompatible with projection scope")
    require(set(projection_counts) == {"insert", "knn"} and
            projection_counts.get("insert") == inserts and projection_counts.get("knn") == knns,
            "metadata projected trace counts drift")

    require(metadata.get("source_trace_sha256") == env_sha(env, "source_trace_sha256") and
            metadata.get("source_event_stream_sha256") == env_sha(env, "source_event_stream_sha256"),
            "metadata source provenance hashes drift")
    for env_key, actual in {
        "dimension": header.dimension, "base_n": header.base_n, "pool_n": header.pool_n,
        "query_n": header.query_n, "k": header.k, "event_count": header.event_count,
        "insert_count": inserts, "knn_count": knns, "final_active_count": len(active),
    }.items():
        require(env_uint(env, env_key) == actual, f"preflight/header replay mismatch for {env_key}")
    require(env_sha(env, "projection_event_stream_sha256") == event_stream_sha and
            env_sha(env, "final_active_set_sha256") == final_hash,
            "preflight event/state witness drift")

    return Bundle(
        path=bundle,
        header=header,
        events=events,
        pool_bytes=pool_bytes,
        queries_bytes=queries_bytes,
        stable_to_pool_row=mapping,
        base_ids=base_ids,
        metadata_sha256=metadata_sha,
        manifest_sha256=manifest_sha,
        trace_sha256=trace_sha,
        projection_event_stream_sha256=event_stream_sha,
        source_trace_sha256=env_sha(env, "source_trace_sha256"),
        source_event_stream_sha256=env_sha(env, "source_event_stream_sha256"),
        final_active_count=len(active),
        final_active_set_sha256=final_hash,
        insert_count=inserts,
        knn_count=knns,
    )


def require_root_private_file_under(path: Path, runtime_root: Path, label: str) -> Path:
    real = path.resolve(strict=True)
    require(real.is_relative_to(runtime_root),
            f"{label} is outside root-private runtime root: {real}")
    current = runtime_root
    root_st = _lstat(current, "runtime root")
    require(not stat.S_ISLNK(root_st.st_mode) and stat.S_ISDIR(root_st.st_mode) and
            root_st.st_uid == 0 and (root_st.st_mode & 0o077) == 0,
            "runtime root must be a root-owned mode-0700 non-symlink directory")
    for part in real.relative_to(runtime_root).parts:
        current = current / part
        st = _lstat(current, f"runtime package path {current}")
        require(not stat.S_ISLNK(st.st_mode) and st.st_uid == 0 and
                (st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)) == 0,
                f"{label} has a non-root-owned or group/other-writable component: {current}")
    return real


def load_root_private_runtime(args: RunArgs) -> Runtime:
    # Isolated mode prevents user-site and PYTHONPATH injection before this
    # runner imports native extensions.  The guard is expected to invoke the
    # venv's interpreter with -I and an otherwise minimal environment.
    require(sys.flags.isolated == 1, "runner must be invoked with the root-private venv using python -I")
    require(os.geteuid() == 0, "root-private runtime must be executed as root")
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        require(not os.environ.get(key), f"{key} must be empty for a trusted runtime")
    runtime_root_input = args.runtime_root
    runtime_st = _lstat(runtime_root_input, "runtime root")
    require(not stat.S_ISLNK(runtime_st.st_mode) and stat.S_ISDIR(runtime_st.st_mode) and
            runtime_st.st_uid == 0 and (runtime_st.st_mode & 0o077) == 0,
            "--runtime-root must be a root-owned mode-0700 non-symlink directory")
    runtime_root = runtime_root_input.resolve(strict=True)
    require(Path(sys.prefix).resolve(strict=True) == runtime_root,
            "--runtime-root must resolve exactly to this interpreter's sys.prefix")
    require(os.environ.get("VIRTUAL_ENV", str(runtime_root)) == str(runtime_root),
            "VIRTUAL_ENV, when set, must identify --runtime-root")

    # This program has no CUDA API/import.  Pin ordinary CPU thread pools before
    # importing NumPy/HNSW so a later quality comparison has a stable CPU-only
    # configuration; no elapsed times are measured or emitted.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"

    try:
        import numpy as np  # type: ignore[import-not-found]
        import hnswlib  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment diagnostic
        raise Fail(f"cannot import hnswlib/numpy from root-private runtime: {exc}") from exc

    try:
        hnsw_dist = importlib.metadata.distribution("hnswlib")
        numpy_dist = importlib.metadata.distribution("numpy")
        hnsw_version = hnsw_dist.version
        numpy_version = numpy_dist.version
    except importlib.metadata.PackageNotFoundError as exc:
        raise Fail("root-private runtime omits hnswlib or numpy distribution metadata") from exc
    require(hnsw_version == HNSW_VERSION,
            f"hnswlib must be pinned to {HNSW_VERSION}, found {hnsw_version!r}")
    require(getattr(np, "__version__", None) == numpy_version,
            "numpy module/distribution version mismatch")

    hnsw_module_path = require_root_private_file_under(
        Path(hnswlib.__file__), runtime_root, "hnswlib module")
    numpy_module_path = require_root_private_file_under(
        Path(np.__file__), runtime_root, "numpy module")
    require_root_private_file_under(Path(hnsw_dist.locate_file("")), runtime_root,
                                    "hnswlib distribution")
    require_root_private_file_under(Path(numpy_dist.locate_file("")), runtime_root,
                                    "numpy distribution")
    return Runtime(
        np=np,
        hnswlib=hnswlib,
        runtime_root=runtime_root,
        hnswlib_version=hnsw_version,
        numpy_version=numpy_version,
        hnswlib_module_path=hnsw_module_path,
        hnswlib_module_sha256=sha256_file(hnsw_module_path, "hnswlib module"),
        numpy_module_path=numpy_module_path,
    )


def make_numeric_views(runtime: Runtime, bundle: Bundle) -> tuple[Any, Any, Any, Any]:
    np = runtime.np
    pool = np.frombuffer(bundle.pool_bytes, dtype=np.dtype("<i2")).reshape(
        bundle.header.pool_n, bundle.header.dimension)
    queries = np.frombuffer(bundle.queries_bytes, dtype=np.dtype("<i2")).reshape(
        bundle.header.query_n, bundle.header.dimension)
    mapping = np.asarray(bundle.stable_to_pool_row, dtype=np.int64)
    base_ids = np.asarray(bundle.base_ids, dtype=np.int64)
    require(int(pool.shape[0]) == bundle.header.pool_n and int(pool.shape[1]) == bundle.header.dimension and
            int(queries.shape[0]) == bundle.header.query_n and int(queries.shape[1]) == bundle.header.dimension and
            int(mapping.size) == bundle.header.pool_n and int(base_ids.size) == bundle.header.base_n,
            "numeric view shape drift")
    return pool, queries, mapping, base_ids


def score_ids_int64(runtime: Runtime, pool: Any, queries: Any, mapping: Any,
                    query_id: int, stable_ids: Sequence[int], k: int) -> list[list[int]]:
    """Exact canonical top-k over the explicitly supplied stable IDs only."""
    np = runtime.np
    require(0 <= query_id < int(queries.shape[0]), "query ID outside numeric query matrix")
    if not stable_ids:
        return []
    ids = np.asarray(list(stable_ids), dtype=np.int64)
    require(ids.ndim == 1 and int(ids.size) == len(stable_ids) and
            bool(np.all((ids >= 0) & (ids < int(mapping.size)))),
            "candidate stable ID lies outside mapping")
    require(int(np.unique(ids).size) == int(ids.size), "candidate stable IDs are not unique")
    rows = mapping[ids]
    query = queries[query_id].astype(np.int64, copy=False)
    values = pool[rows].astype(np.int64, copy=False)
    delta = values - query
    squared = np.sum(delta * delta, axis=1, dtype=np.int64)
    order = np.lexsort((ids, squared))
    count = min(k, int(ids.size))
    return [[int(ids[index]), int(squared[index])] for index in order[:count]]


def merge_topk(rows_a: Sequence[Sequence[int]], rows_b: Sequence[Sequence[int]], k: int) -> list[list[int]]:
    merged = [[int(row[0]), int(row[1])] for row in rows_a] + \
             [[int(row[0]), int(row[1])] for row in rows_b]
    require(len({row[0] for row in merged}) == len(merged),
            "immutable-base and global-delta result IDs overlap")
    merged.sort(key=lambda row: (row[1], row[0]))
    return merged[:k]


def overlap_count(actual: Sequence[Sequence[int]], expected: Sequence[Sequence[int]]) -> int:
    return len({int(row[0]) for row in actual} & {int(row[0]) for row in expected})


def canonical_sequence_update(digest: Any, op_index: int, query_id: int,
                              active: Iterable[int], candidate_base: Sequence[Sequence[int]],
                              merged: Sequence[Sequence[int]], exact_base: Sequence[Sequence[int]],
                              exact_full: Sequence[Sequence[int]]) -> None:
    digest.update(f"{op_index}:{query_id}:{canonical_active_hash(active)}:".encode("ascii"))
    for label, rows in (("candidate_base", candidate_base), ("candidate_merged", merged),
                        ("oracle_base", exact_base), ("oracle_merged", exact_full)):
        digest.update((label + ":").encode("ascii"))
        for stable_id, distance_sq in rows:
            digest.update(f"{int(stable_id)}:{int(distance_sq)},".encode("ascii"))
    digest.update(b"\n")


def write_new(path: Path, payload: str) -> None:
    """Create a root-private output exactly once, refusing symlink overwrite races."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        encoded = payload.encode("utf-8")
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            require(written > 0, f"short write to {path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)




@dataclass(frozen=True)
class PartitionSeal:
    partition: str
    members_by_op: dict[int, dict[str, int]]
    member_ordinals: frozenset[int]
    member_sha256: str
    seal_sha256: str
    source_split_file_sha256: str
    source_split_canonical_sha256: str
    target: dict[str, int] | None


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def member_digest(members: Sequence[dict[str, int]]) -> str:
    # Official sealed-split membership invariant; alternate digest encodings are
    # deliberately not accepted by strict-C2.
    return hashlib.sha256("".join(
        f"{item['knn_ordinal']}\t{item['op_index']}\t{item['query_id']}\n" for item in members
    ).encode("ascii")).hexdigest()


def trace_knn_members(bundle: Bundle) -> list[dict[str, int]]:
    delta_live = 0
    members: list[dict[str, int]] = []
    for event in bundle.events:
        if event.opcode == INSERT:
            delta_live += 1
        else:
            members.append({"knn_ordinal": len(members), "op_index": event.op_index,
                            "query_id": event.argument, "global_delta_live": delta_live})
    require(delta_live == bundle.insert_count and len(members) == bundle.knn_count,
            "trace KNN membership replay drift")
    return members


def require_uint_fields(value: Any, keys: Sequence[str], label: str) -> dict[str, int]:
    require(isinstance(value, dict) and set(value) == set(keys),
            f"{label} shape drift")
    output: dict[str, int] = {}
    for key in keys:
        output[key] = require_uint(value.get(key), f"{label}.{key}")
    return output


def validate_partition_seal(args: RunArgs, bundle: Bundle, env: dict[str, str]) -> PartitionSeal:
    path = require_real_regular_file(args.partition_seal, "partition seal")
    raw = read_regular_bytes(path, "partition seal")
    seal = read_json_object(path, "partition seal")
    require(seal.get("schema") == PARTITION_SEAL_SCHEMA,
            "partition seal schema mismatch")
    require(seal.get("partition") == args.partition and args.partition in {"calibration", "held_out"},
            "partition seal / requested partition mismatch")
    supplied_self = require_sha256(seal.get("seal_sha256"), "partition seal self hash")
    unchecked = dict(seal)
    unchecked.pop("seal_sha256", None)
    require(hashlib.sha256(canonical_json(unchecked)).hexdigest() == supplied_self,
            "partition seal self-hash mismatch")
    require(seal.get("source_split_schema") == SOURCE_SPLIT_SCHEMA and
            seal.get("source_split_file_sha256") == SOURCE_SPLIT_FILE_SHA256 and
            seal.get("source_split_canonical_sha256") == SOURCE_SPLIT_CANONICAL_SHA256,
            "partition seal does not bind the official sealed split")
    require(seal.get("cross_partition_content_absent") is True,
            "partition seal must declare cross-partition content absent")
    require(seal.get("member_sha256_scope") ==
            "SHA-256 of LF-delimited ASCII lines in KNN-ordinal order: knn_ordinal<TAB>op_index<TAB>query_id<LF>.",
            "partition seal official member hash scope drift")
    require(seal.get("seal_sha256_scope") ==
            "SHA-256 of canonical UTF-8 JSON for this object with seal_sha256 omitted; sort_keys=true, separators=(',', ':'), trailing LF.",
            "partition seal self-hash scope drift")
    # SHA/path binding is kept distinct so no target/foreign partition object can
    # hide in a permissive projection parser.
    raw_projection = seal.get("projection")
    require(isinstance(raw_projection, dict) and set(raw_projection) ==
            {"bundle_path", "manifest_sha256", "metadata_sha256", "trace_sha256",
             "source_trace_sha256", "event_count", "insert_count", "knn_count", "k"},
            "partition seal projection keys drift")
    require(isinstance(raw_projection.get("bundle_path"), str) and
            Path(raw_projection["bundle_path"]).resolve(strict=True) == bundle.path,
            "partition seal bundle path mismatch")
    projection = {key: require_uint(raw_projection.get(key), f"partition seal projection.{key}")
                  for key in ("event_count", "insert_count", "knn_count", "k")}
    expected_hashes = {
        "manifest_sha256": bundle.manifest_sha256,
        "metadata_sha256": bundle.metadata_sha256,
        "trace_sha256": bundle.trace_sha256,
        "source_trace_sha256": bundle.source_trace_sha256,
    }
    for key, expected in expected_hashes.items():
        require(raw_projection.get(key) == expected,
                f"partition seal projection binding mismatch for {key}")
    require(projection == {"event_count": bundle.header.event_count,
                           "insert_count": bundle.insert_count,
                           "knn_count": bundle.knn_count,
                           "k": bundle.header.k},
            "partition seal projection cardinality mismatch")
    raw_members = seal.get("members")
    require(isinstance(raw_members, list) and seal.get("member_count") == len(raw_members),
            "partition seal members/cardinality mismatch")
    expected_all = trace_knn_members(bundle)
    parity = 0 if args.partition == "calibration" else 1
    expected = [item for item in expected_all if item["knn_ordinal"] % 2 == parity]
    require(len(raw_members) == len(expected) == bundle.knn_count // 2,
            "partition seal expected member count drift")
    members: list[dict[str, int]] = []
    for number, (raw_member, desired) in enumerate(zip(raw_members, expected)):
        member = require_uint_fields(raw_member,
                                     ("knn_ordinal", "op_index", "query_id", "global_delta_live"),
                                     f"partition seal member {number}")
        require(member == desired, f"partition seal member differs from presealed trace at {number}")
        members.append(member)
    digest = member_digest(members)
    require(seal.get("member_sha256") == digest,
            "partition seal member hash mismatch")
    require(digest == ("eae0763680a0f71862a3b63e3c4429ffa467e5f9e7df2a03cc5607cbd552658d"
                       if args.partition == "calibration" else
                       "e96b8f45d71d3aa566e09c7b92a4844d664b30774c0f04850d7d0f8cea4a7aba"),
            "strict-C2 official tab/ordinal member hash mismatch")
    target: dict[str, int] | None
    if args.partition == "calibration":
        require("held_out_quality_target_embedded" not in seal,
                "calibration seal illegally carries held-out target flag")
        target = require_uint_fields(
            seal.get("calibration_native_gts_target"),
            ("merged_overlap_numerator", "merged_overlap_denominator",
             "merged_exact_match_numerator", "merged_exact_match_denominator",
             "knn_count", "k"),
            "calibration-only Native-GTS target")
        require(tuple(target.values()) == (651, 680, 53, 68, 68, bundle.header.k),
                "calibration target is not the established calibration-only target")
        allowed = {"schema", "scope", "partition", "source_split_schema", "source_split_file_sha256", "source_split_canonical_sha256",
                   "projection", "member_sha256_scope", "member_count", "member_sha256", "members",
                   "cross_partition_content_absent", "seal_sha256_scope", "calibration_native_gts_target",
                   "seal_sha256"}
    else:
        require(seal.get("held_out_quality_target_embedded") is False and
                "calibration_native_gts_target" not in seal,
                "held-out seal must not carry any calibration/held-out quality target")
        target = None
        allowed = {"schema", "scope", "partition", "source_split_schema", "source_split_file_sha256", "source_split_canonical_sha256",
                   "projection", "member_sha256_scope", "member_count", "member_sha256", "members",
                   "cross_partition_content_absent", "seal_sha256_scope", "held_out_quality_target_embedded",
                   "seal_sha256"}
    require(set(seal) == allowed, "partition seal has unrecognized/cross-partition fields")
    require(sha256_bytes(raw) == sha256_file(path, "partition seal"),
            "partition seal raw/hash read inconsistency")
    return PartitionSeal(args.partition, {item["op_index"]: item for item in members},
                         frozenset(item["knn_ordinal"] for item in members), digest,
                         supplied_self, SOURCE_SPLIT_FILE_SHA256, SOURCE_SPLIT_CANONICAL_SHA256, target)


def validate_heldout_lock(args: RunArgs, bundle: Bundle, seal: PartitionSeal) -> str:
    require(args.partition == "held_out" and args.lock_receipt is not None,
            "held-out partition requires a lock receipt")
    path = require_real_regular_file(args.lock_receipt, "held-out lock receipt")
    lock = read_json_object(path, "held-out lock receipt")
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("status") == "LOCKED",
            "held-out lock receipt schema/status mismatch")
    expected_lock_keys = {"schema", "status", "scope", "partition", "selected_on_partition",
                          "selection_policy", "candidate_ef_grid", "locked_ef_search",
                          "calibration_selection_sha256", "calibration_seal_sha256", "heldout_seal_sha256",
                          "source_split_file_sha256", "source_split_canonical_sha256",
                          "manifest_sha256", "metadata_sha256", "trace_sha256",
                          "lock_sha256_scope", "lock_sha256"}
    require(set(lock) == expected_lock_keys,
            "held-out lock receipt key set mismatch")
    require(lock.get("partition") == "held_out" and lock.get("selected_on_partition") == "calibration",
            "held-out lock receipt partition binding mismatch")
    require(lock.get("selection_policy") ==
            "smallest pre-registered EF satisfying calibration merged_overlap_sum >= 651/680 and merged_exact_match_count >= 53/68" and
            lock.get("candidate_ef_grid") == [17, 18, 19, 20, 21, 22, 23],
            "held-out lock selection policy/grid mismatch")
    locked_ef = require_uint(lock.get("locked_ef_search"), "lock locked_ef_search")
    require(locked_ef in lock["candidate_ef_grid"], "held-out lock EF lies outside candidate grid")
    require(locked_ef == args.ef_search and locked_ef >= bundle.header.k,
            "held-out runner EF must be exactly the locked calibration-selected EF")
    require(lock.get("heldout_seal_sha256") == seal.seal_sha256 and
            lock.get("source_split_file_sha256") == seal.source_split_file_sha256 and
            lock.get("source_split_canonical_sha256") == seal.source_split_canonical_sha256 and
            lock.get("manifest_sha256") == bundle.manifest_sha256 and
            lock.get("metadata_sha256") == bundle.metadata_sha256 and
            lock.get("trace_sha256") == bundle.trace_sha256,
            "held-out lock does not bind current seal/bundle")
    require_sha256(lock.get("calibration_selection_sha256"),
                   "held-out lock calibration selection hash")
    require_sha256(lock.get("calibration_seal_sha256"),
                   "held-out lock calibration seal hash")
    require(lock.get("lock_sha256_scope") ==
            "SHA-256 of canonical UTF-8 JSON for this object with lock_sha256 omitted; sort_keys=true, separators=(',', ':'), trailing LF.",
            "held-out lock self-hash scope drift")
    raw_self = require_sha256(lock.get("lock_sha256"), "lock self hash")
    unchecked = dict(lock)
    unchecked.pop("lock_sha256", None)
    require(hashlib.sha256(canonical_json(unchecked)).hexdigest() == raw_self,
            "held-out lock self-hash mismatch")
    return raw_self


def source_sha(path: Path, label: str) -> str:
    real = require_real_regular_file(path, label)
    info = _lstat(real, label)
    require(info.st_uid == 0 and (info.st_mode & 0o077) == 0,
            f"{label} must be root-owned and private")
    return sha256_file(real, label)



def run_partition(args: RunArgs) -> int:
    """Run exactly one presealed partition, never querying skipped KNN events."""
    require(args.partition in {"calibration", "held_out"}, "unknown strict-C2 partition")
    require(args.ef_search > 0, "EF must be positive")
    # Everything through seal/lock validation is stdlib-only.  In particular,
    # historical combined split files are never opened by this strict runner.
    validate_output_targets(args)
    admission = validate_admission(args)
    bundle = validate_bundle(args, admission)
    require(args.ef_search >= bundle.header.k,
            "EF must be at least the trace k")
    seal = validate_partition_seal(args, bundle, admission)
    lock_sha = validate_heldout_lock(args, bundle, seal) if args.partition == "held_out" else None
    entrypoint_sha = source_sha(args.entrypoint, "strict-C2 entrypoint source")
    core_sha = source_sha(Path(__file__), "strict-C2 shared core source")

    # No third-party import or HNSW construction occurs before all roots, input
    # hashes, local membership, and (for test) the immutable EF lock pass.
    runtime = load_root_private_runtime(args)
    pool, queries, mapping, base_ids = make_numeric_views(runtime, bundle)
    np = runtime.np
    immutable_base = tuple(int(item) for item in bundle.base_ids)
    immutable_base_set = set(immutable_base)
    require(len(immutable_base_set) == bundle.header.base_n,
            "immutable base cardinality drift")
    base_vectors_f32 = pool[mapping[base_ids]].astype(np.float32, copy=True)
    require(tuple(base_vectors_f32.shape) == (bundle.header.base_n, bundle.header.dimension),
            "immutable base matrix shape drift")
    index = runtime.hnswlib.Index(space="l2", dim=bundle.header.dimension)
    index.init_index(max_elements=bundle.header.base_n, M=HNSW_M,
                     ef_construction=HNSW_EF_CONSTRUCTION,
                     random_seed=HNSW_RANDOM_SEED, allow_replace_deleted=False)
    index.set_num_threads(HNSW_THREADS)
    index.add_items(base_vectors_f32, base_ids, num_threads=HNSW_THREADS)
    require(int(index.get_current_count()) == bundle.header.base_n and
            int(index.get_max_elements()) == bundle.header.base_n,
            "HNSW must contain exactly the immutable initial base")
    index.set_ef(args.ef_search)

    active: set[int] = set(immutable_base)
    global_delta: list[int] = []
    records: list[dict[str, Any]] = []
    sequence = hashlib.sha256()
    base_exact_match_count = merged_exact_match_count = 0
    base_overlap_sum = merged_overlap_sum = 0
    inserts = knns_seen = evaluated_knns = skipped_knns = 0
    hnsw_query_invocations = oracle_evaluations = 0

    for event in bundle.events:
        if event.opcode == INSERT:
            stable_id = event.argument
            require(stable_id not in active and stable_id not in immutable_base_set,
                    f"op {event.op_index}: insertion attempts to mutate immutable base")
            active.add(stable_id)
            global_delta.append(stable_id)
            inserts += 1
            # There is intentionally no insertion JSONL record: strict-C2
            # output contains only evaluated partition KNN records.
            continue

        knn_ordinal = knns_seen
        knns_seen += 1
        selected = seal.members_by_op.get(event.op_index)
        if selected is None:
            # This branch deliberately reads no query vector and invokes neither
            # HNSW nor any exact oracle.  It emits no output record.
            require(knn_ordinal not in seal.member_ordinals,
                    f"op {event.op_index}: selected ordinal absent from seal map")
            skipped_knns += 1
            continue
        require(selected == {"knn_ordinal": knn_ordinal, "op_index": event.op_index,
                             "query_id": event.argument, "global_delta_live": len(global_delta)},
                f"op {event.op_index}: presealed partition membership/time-state mismatch")
        query_id = event.argument
        query_f32 = queries[query_id].astype(np.float32, copy=True).reshape(1, -1)
        labels, returned_distances = index.knn_query(query_f32, k=bundle.header.k,
                                                      num_threads=HNSW_THREADS)
        hnsw_query_invocations += 1
        require(tuple(labels.shape) == (1, bundle.header.k) and
                tuple(returned_distances.shape) == (1, bundle.header.k),
                f"op {event.op_index}: HNSW did not return exactly k base candidates")
        raw_labels = [int(value) for value in labels[0].tolist()]
        require(len(raw_labels) == bundle.header.k and len(set(raw_labels)) == bundle.header.k and
                all(stable_id in immutable_base_set for stable_id in raw_labels),
                f"op {event.op_index}: HNSW returned invalid immutable-base labels")
        candidate_base = score_ids_int64(runtime, pool, queries, mapping, query_id,
                                         raw_labels, bundle.header.k)
        exact_delta = score_ids_int64(runtime, pool, queries, mapping, query_id,
                                      global_delta, bundle.header.k)
        merged = merge_topk(candidate_base, exact_delta, bundle.header.k)
        exact_base = score_ids_int64(runtime, pool, queries, mapping, query_id,
                                     immutable_base, bundle.header.k)
        exact_full = score_ids_int64(runtime, pool, queries, mapping, query_id,
                                     tuple(active), bundle.header.k)
        oracle_evaluations += 1
        require(len(candidate_base) == bundle.header.k and len(merged) == bundle.header.k and
                len(exact_base) == bundle.header.k and len(exact_full) == bundle.header.k,
                f"op {event.op_index}: top-k cardinality drift")
        base_overlap = overlap_count(candidate_base, exact_base)
        merged_overlap = overlap_count(merged, exact_full)
        base_exact = candidate_base == exact_base
        merged_exact = merged == exact_full
        base_exact_match_count += int(base_exact)
        merged_exact_match_count += int(merged_exact)
        base_overlap_sum += base_overlap
        merged_overlap_sum += merged_overlap
        evaluated_knns += 1
        canonical_sequence_update(sequence, event.op_index, query_id, active,
                                  candidate_base, merged, exact_base, exact_full)
        records.append({
            "record": "knn", "partition": args.partition,
            "knn_ordinal": knn_ordinal, "op_index": event.op_index,
            "query_id": query_id, "k": bundle.header.k,
            "global_delta_live": len(global_delta),
            "hnsw_base_candidate_path": "hnswlib_knn_query_frozen_immutable_base",
            "hnsw_base_candidate_count": len(candidate_base),
            "hnsw_returned_labels": raw_labels,
            "hnsw_returned_distance_count": len(returned_distances[0]),
            "hnsw_ef_search": args.ef_search,
            "hnsw_base_candidate_path_used": True,
            "hnsw_trace_index_mutated": False,
            "hnsw_add_items_called_for_trace": False,
            "base_immutable": True, "direct_sidecar_used": False,
            "candidate_rescored_with_int64": True,
            "float_hnsw_distances_used_for_ranking": False,
            "full_active_int64_oracle_used": True,
            "base_overlap_count": base_overlap, "merged_overlap_count": merged_overlap,
            "base_exact_match": base_exact, "merged_exact_match": merged_exact,
            "base_candidate_topk": candidate_base,
            "exact_global_delta_topk": exact_delta, "merged": merged,
        })

    require(inserts == bundle.insert_count and knns_seen == bundle.knn_count and
            evaluated_knns == len(seal.members_by_op) and
            skipped_knns == bundle.knn_count - len(seal.members_by_op),
            "strict-C2 replay/evaluation cardinality drift")
    require(hnsw_query_invocations == oracle_evaluations == evaluated_knns,
            "strict-C2 selected-query/oracle counter drift")
    require(len(records) == evaluated_knns and len({record["op_index"] for record in records}) == evaluated_knns,
            "strict-C2 output record cardinality/uniqueness drift")
    require(len(active) == bundle.final_active_count and
            canonical_active_hash(active) == bundle.final_active_set_sha256,
            "final active-state witness drift after full insertion replay")
    require(int(index.get_current_count()) == bundle.header.base_n and
            int(index.get_max_elements()) == bundle.header.base_n,
            "HNSW immutable-base structural counter drift")

    other = "held_out" if args.partition == "calibration" else "calibration"
    summary: dict[str, Any] = {
        "schema": RUN_SCHEMA, "status": "PASS_PARTITION_PROBE",
        "mode": "strict_c2_partition_quality_probe", "partition": args.partition,
        "scope": ("CPU-only/no-timing strict-C2 partition-local HNSW frozen immutable-base "
                  "candidate-recall probe with external exact global delta; not a performance claim"),
        "input": {"manifest_sha256": bundle.manifest_sha256,
                  "metadata_sha256": bundle.metadata_sha256, "trace_sha256": bundle.trace_sha256,
                  "projection_event_stream_sha256": bundle.projection_event_stream_sha256,
                  "source_trace_sha256": bundle.source_trace_sha256,
                  "source_event_stream_sha256": bundle.source_event_stream_sha256},
        "source": {"entrypoint_path": str(args.entrypoint.resolve()), "entrypoint_sha256": entrypoint_sha,
                   "core_path": str(Path(__file__).resolve()), "core_sha256": core_sha},
        "partition_seal": {"sha256": seal.seal_sha256, "member_sha256": seal.member_sha256,
                           "member_count": evaluated_knns,
                           "source_split_file_sha256": seal.source_split_file_sha256,
                           "source_split_canonical_sha256": seal.source_split_canonical_sha256},
        "heldout_lock_sha256": lock_sha,
        "event_replay": {"event_count": len(bundle.events), "insert_replayed_count": inserts,
                         "knn_seen_count": knns_seen, "evaluated_partition_knn_count": evaluated_knns,
                         "skipped_other_partition_knn_count": skipped_knns,
                         "all_trace_inserts_replayed_in_order": True},
        "partition_metrics": {"knn_count": evaluated_knns, "k": bundle.header.k,
                              "base_exact_match_count": base_exact_match_count,
                              "merged_exact_match_count": merged_exact_match_count,
                              "base_overlap_sum": base_overlap_sum,
                              "merged_overlap_sum": merged_overlap_sum,
                              "canonical_int64_knn_sequence_sha256": sequence.hexdigest()},
        "no_cross_partition_metrics": {
            "evaluated_partition_only": True, "other_partition_name": other,
            "other_partition_hnsw_query_invocations": 0,
            "other_partition_oracle_evaluations": 0,
            "other_partition_records_written": 0,
            "other_partition_metric_aggregates_written": 0,
            "other_partition_quality_target_read": False,
            "skipped_other_partition_knn_count": skipped_knns,
        },
        "hnsw": {"library": "hnswlib", "library_version": runtime.hnswlib_version,
                 "library_module_path": str(runtime.hnswlib_module_path),
                 "library_module_sha256": runtime.hnswlib_module_sha256,
                 "numpy_version": runtime.numpy_version,
                 "numpy_module_path": str(runtime.numpy_module_path),
                 "runtime_root": str(runtime.runtime_root), "space": "l2", "M": HNSW_M,
                 "ef_construction": HNSW_EF_CONSTRUCTION, "ef_search": args.ef_search,
                 "random_seed": HNSW_RANDOM_SEED, "threads": HNSW_THREADS,
                 "max_elements": bundle.header.base_n,
                 "current_count_after_trace": int(index.get_current_count())},
        "structural_counters": {"hnsw_initial_base_add_items_calls": 1,
                                "hnsw_trace_add_items_calls": 0,
                                "hnsw_trace_mark_deleted_calls": 0,
                                "hnsw_trace_resize_index_calls": 0,
                                "hnsw_trace_index_mutation_calls": 0,
                                "external_global_delta_insert_count": len(global_delta),
                                "selected_partition_hnsw_query_invocations": hnsw_query_invocations,
                                "selected_partition_oracle_evaluations": oracle_evaluations,
                                "skipped_partition_hnsw_query_invocations": 0,
                                "skipped_partition_oracle_evaluations": 0,
                                "skipped_partition_records_written": 0},
        "base_immutable": True, "direct_sidecar_used": False,
        "legacy_routing_used": False, "provided_full_trace_oracle_used": False,
        "gpu_used": False, "timing_claim": False,
        "limitations": [
            "Only the presealed partition KNN ordinals are evaluated; all 169 trace inserts are replayed, while the other 68 KNNs are deliberately skipped without HNSW or oracle invocation.",
            "This is a frozen immutable-base, insertion-only, KNN-only projection, not evidence for delete, range, rebuild, compaction, concurrency, or full dynamic-index correctness.",
            "Trace inserts live only in the exact external global delta; HNSW receives no trace add_items call and has max_elements=base_n.",
            "No elapsed time or throughput is measured or emitted; this run cannot support a performance claim.",
        ],
    }
    jsonl = "".join(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                    for record in records)
    write_new(args.out, jsonl)
    write_new(args.summary, json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return 0

