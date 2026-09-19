#!/usr/bin/python3.12
"""Independent pure-stdlib validator for the frozen-base HNSW recall probe.

This program never imports or executes hnswlib, NumPy, CUDA, GTS, NVML, or a
user-owned runtime.  It replays the sealed 305-event Safe-C1 projection with
an exact Python integer squared-L2 oracle, validates the HNSW candidate IDs
and their independent integer re-score, and checks the no-trace-mutation / no-
timing / CPU-only structural contract recorded by the runner.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import stat
import struct
import sys
from pathlib import Path
from typing import Any, Iterable

MAGIC = b"E1GTRC01"
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
INSERT, KNN = 1, 3

ADMISSION_SCHEMA = "e1-frozen-base-knn-projection-admission-v2"
MANIFEST_SCHEMA = "e1-frozen-base-knn-projection-manifest-v1"
METADATA_SCHEMA = "e1-frozen-base-knn-projection-bundle-v1"
SPLIT_SCHEMA = "safe-c1-hnsw-calibration-test-split-v1"
RUN_SCHEMA = "fair-hnsw-frozen-base-candidate-recall-probe-v1"
VALIDATOR_SCHEMA = "fair-hnsw-frozen-base-recall-output-validator-v1"

# This validator is intentionally bound to the sealed E1 Safe-C1 projection,
# rather than accepting a superficially similar arbitrary trace.
EXPECTED_EVENT_COUNT = 305
EXPECTED_BASE_N = 4096
EXPECTED_K = 10
EXPECTED_INSERTS = 169
EXPECTED_KNNS = 136


class Fail(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Fail(message)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def need_int(value: Any, label: str, minimum: int = 0) -> int:
    require(is_int(value) and value >= minimum, f"{label} must be an integer >= {minimum}")
    return int(value)


def need_bool(value: Any, label: str) -> bool:
    require(isinstance(value, bool), f"{label} must be a boolean")
    return value


def need_sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(ch in "0123456789abcdef" for ch in value),
            f"{label} must be a lowercase SHA-256 digest")
    return value


def reject_json_constant(value: str) -> None:
    raise Fail(f"non-finite JSON constant is forbidden: {value}")


def lstat(path: Path, label: str) -> os.stat_result:
    try:
        return path.lstat()
    except FileNotFoundError as exc:
        raise Fail(f"missing {label}: {path}") from exc


def real_directory(path: Path, label: str) -> Path:
    st = lstat(path, label)
    require(stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode),
            f"{label} must be a real non-symlink directory: {path}")
    return path.resolve(strict=True)


def regular_file(path: Path, label: str) -> Path:
    st = lstat(path, label)
    require(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode),
            f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def read_bytes(path: Path, label: str) -> bytes:
    regular_file(path, label)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise Fail(f"cannot read {label}: {path}: {exc}") from exc


def sha256_file(path: Path, label: str) -> str:
    regular_file(path, label)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise Fail(f"cannot hash {label}: {path}: {exc}") from exc
    return digest.hexdigest()


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    raw = read_bytes(path, label)
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, Fail) as exc:
        raise Fail(f"invalid {label} JSON: {exc}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def read_env(path: Path) -> dict[str, str]:
    try:
        raw = read_bytes(path, "admission").decode("ascii")
    except UnicodeDecodeError as exc:
        raise Fail("admission must be ASCII") from exc
    result: dict[str, str] = {}
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        require("=" in line and line.index("=") > 0,
                f"malformed admission line {line_number}")
        key, value = line.split("=", 1)
        require(key not in result and value != "",
                f"duplicate/empty admission key at line {line_number}")
        result[key] = value
    return result


def env_need(env: dict[str, str], key: str) -> str:
    value = env.get(key)
    require(value is not None and value != "", f"admission omits {key}")
    return value


def env_int(env: dict[str, str], key: str) -> int:
    value = env_need(env, key)
    require(value.isdecimal(), f"admission {key} is not a decimal integer")
    return int(value)


def env_sha(env: dict[str, str], key: str) -> str:
    return need_sha(env_need(env, key), f"admission {key}")


def active_hash(active: bytearray) -> str:
    digest = hashlib.sha256()
    for stable_id, present in enumerate(active):
        if present:
            digest.update(f"{stable_id}\n".encode("ascii"))
    return digest.hexdigest()


def active_count(active: bytearray) -> int:
    return active.count(1)


def read_le_array(path: Path, typecode: str, width: int, label: str) -> array.array:
    raw = read_bytes(path, label)
    require(len(raw) % width == 0, f"{label} byte count is not a multiple of {width}")
    values = array.array(typecode)
    require(values.itemsize == width, f"host {typecode!r} width is not {width}")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def parse_bundle(bundle: Path, env: dict[str, str]) -> tuple[
        dict[str, int], list[tuple[int, int, int]], array.array, array.array,
        array.array, array.array, str, str]:
    bundle = real_directory(bundle, "bundle")
    manifest_path = bundle / "manifest.json"
    metadata_path = bundle / "metadata.json"
    manifest = read_json_object(manifest_path, "manifest.json")
    metadata = read_json_object(metadata_path, "metadata.json")
    require(manifest.get("schema") == MANIFEST_SCHEMA,
            "projection manifest schema mismatch")
    require(metadata.get("schema") == METADATA_SCHEMA,
            "projection metadata schema mismatch")
    require(sha256_file(manifest_path, "manifest.json") == env_sha(env, "manifest_sha256"),
            "manifest changed after admission")
    require(sha256_file(metadata_path, "metadata.json") == env_sha(env, "metadata_sha256"),
            "metadata changed after admission")

    files = manifest.get("files_sha256")
    require(isinstance(files, dict), "manifest.files_sha256 must be an object")
    required = {
        "metadata.json": "metadata_sha256",
        "trace.e1gtrc": "trace_sha256",
        "pool.i16": "pool_sha256",
        "queries.i16": "queries_sha256",
        "stable_id_to_pool_row.i32": "mapping_sha256",
        "initial_base_stable_ids.i32": "base_ids_sha256",
    }
    for filename, admission_key in required.items():
        wanted = need_sha(files.get(filename), f"manifest.files_sha256[{filename!r}]")
        actual = sha256_file(bundle / filename, f"bundle payload {filename}")
        require(actual == wanted == env_sha(env, admission_key),
                f"manifest/admission payload mismatch for {filename}")

    raw_trace = read_bytes(bundle / "trace.e1gtrc", "trace.e1gtrc")
    require(len(raw_trace) >= HEADER.size, "truncated projection trace")
    magic, version, dimension, base_n, reservoir_n, pool_n, query_n, k, _radius, event_count = \
        HEADER.unpack_from(raw_trace)
    require(magic == MAGIC and version == 1, "projection trace magic/version mismatch")
    require(dimension > 0 and base_n > 0 and reservoir_n == pool_n - base_n and
            pool_n > base_n and query_n > 0 and k > 0 and base_n >= k,
            "invalid projection trace header")
    require(len(raw_trace) == HEADER.size + event_count * EVENT.size,
            "projection trace byte count/event_count mismatch")
    require(event_count == EXPECTED_EVENT_COUNT and base_n == EXPECTED_BASE_N and k == EXPECTED_K,
            "validator is sealed for the 305-event / base-4096 / k=10 E1 projection")
    events: list[tuple[int, int, int]] = []
    for position in range(event_count):
        op_index, opcode, argument = EVENT.unpack_from(raw_trace, HEADER.size + position * EVENT.size)
        require(op_index == position and opcode in (INSERT, KNN),
                f"illegal/noncanonical projection event at {position}")
        if opcode == INSERT:
            require(base_n <= argument < pool_n,
                    f"projection insertion outside mutable reservoir at op {position}")
        else:
            require(0 <= argument < query_n,
                    f"projection KNN query outside matrix at op {position}")
        events.append((int(op_index), int(opcode), int(argument)))

    event_stream_sha = hashlib.sha256(raw_trace[HEADER.size:]).hexdigest()
    require(event_stream_sha == env_sha(env, "projection_event_stream_sha256"),
            "projection event-stream hash mismatch")
    require(manifest.get("projection_trace_sha256") == env_sha(env, "trace_sha256") and
            manifest.get("source_trace_sha256") == env_sha(env, "source_trace_sha256"),
            "manifest source/projection trace provenance mismatch")

    projection = metadata.get("projection")
    require(isinstance(projection, dict), "metadata.projection must be an object")
    require(metadata.get("scope") ==
            "Safe-C1 immutable-base KNN gate only; not the full E1 workload, not a base-delete, range, rebuild, direct-sidecar, or performance claim.",
            "projection scope drift")
    require(projection.get("allowed_ops") == ["insert", "knn"] and
            projection.get("excluded_ops") == ["delete", "range"] and
            projection.get("source_quantized_oracle_role") ==
            "source provenance only; never a projection oracle",
            "projection operation/oracle contract mismatch")
    require(projection.get("projection_trace_sha256") == env_sha(env, "trace_sha256") and
            projection.get("projection_event_stream_sha256") == event_stream_sha and
            projection.get("stable_id_mapping_sha256") == env_sha(env, "mapping_sha256") and
            projection.get("initial_base_stable_ids_sha256") == env_sha(env, "base_ids_sha256"),
            "metadata projection provenance mismatch")

    pool = read_le_array(bundle / "pool.i16", "h", 2, "pool.i16")
    queries = read_le_array(bundle / "queries.i16", "h", 2, "queries.i16")
    mapping = read_le_array(bundle / "stable_id_to_pool_row.i32", "i", 4,
                            "stable_id_to_pool_row.i32")
    base_ids = read_le_array(bundle / "initial_base_stable_ids.i32", "i", 4,
                             "initial_base_stable_ids.i32")
    require(len(pool) == pool_n * dimension and len(queries) == query_n * dimension and
            len(mapping) == pool_n and len(base_ids) == base_n,
            "projection matrix shape mismatch")
    require(len(set(mapping)) == pool_n and min(mapping) >= 0 and max(mapping) < pool_n,
            "stable-ID to pool-row mapping is not bijective")
    require(len(set(base_ids)) == base_n and min(base_ids) >= 0 and max(base_ids) < pool_n,
            "initial immutable base IDs are invalid")

    active = bytearray(pool_n)
    for stable_id in base_ids:
        active[int(stable_id)] = 1
    insert_count = 0
    knn_count = 0
    for op_index, opcode, argument in events:
        if opcode == INSERT:
            require(not active[argument], f"projection repeats insertion at op {op_index}")
            active[argument] = 1
            insert_count += 1
        else:
            knn_count += 1
    require(insert_count == EXPECTED_INSERTS and knn_count == EXPECTED_KNNS,
            "projection operation counts differ from sealed E1 projection")
    require(active_count(active) == env_int(env, "final_active_count") and
            active_hash(active) == env_sha(env, "final_active_set_sha256"),
            "projection final active-state witness mismatch")
    for key, actual in {
        "dimension": dimension, "base_n": base_n, "pool_n": pool_n,
        "query_n": query_n, "k": k, "event_count": event_count,
        "insert_count": insert_count, "knn_count": knn_count,
    }.items():
        require(env_int(env, key) == actual, f"admission/header mismatch for {key}")
    return ({"dimension": int(dimension), "base_n": int(base_n), "pool_n": int(pool_n),
             "query_n": int(query_n), "k": int(k), "event_count": int(event_count),
             "insert_count": insert_count, "knn_count": knn_count},
            events, pool, queries, mapping, base_ids, event_stream_sha,
            sha256_file(metadata_path, "metadata.json"))


def validate_admission(bundle: Path, admission: Path) -> dict[str, str]:
    bundle = real_directory(bundle, "bundle")
    env = read_env(admission)
    required = {
        "schema": ADMISSION_SCHEMA,
        "status": "PASS",
        "projection_manifest_schema": MANIFEST_SCHEMA,
        "projection_metadata_schema": METADATA_SCHEMA,
        "ops": "insert,knn",
        "excluded_ops": "delete,range",
        "base_immutable": "true",
        "direct_sidecar_allowed": "false",
        "legacy_routing_allowed": "false",
    }
    for key, expected in required.items():
        require(env_need(env, key) == expected, f"admission policy drift for {key}")
    require(Path(env_need(env, "bundle_realpath")).resolve(strict=True) == bundle,
            "admission bundle_realpath does not bind --bundle")
    for key in ("dimension", "base_n", "pool_n", "query_n", "k", "event_count",
                "insert_count", "knn_count", "final_active_count", "source_event_count"):
        env_int(env, key)
    require(env_int(env, "source_event_count") == 512 and
            env_int(env, "event_count") == EXPECTED_EVENT_COUNT,
            "admission is not for the sealed E1 projection")
    for key in ("manifest_sha256", "metadata_sha256", "trace_sha256", "pool_sha256",
                "queries_sha256", "mapping_sha256", "base_ids_sha256",
                "projection_event_stream_sha256", "final_active_set_sha256",
                "source_trace_sha256", "source_event_stream_sha256"):
        env_sha(env, key)
    return env


def parse_rows(value: Any, label: str, expected_count: int) -> list[list[int]]:
    require(isinstance(value, list) and len(value) == expected_count,
            f"{label} must contain exactly {expected_count} rows")
    rows: list[list[int]] = []
    seen: set[int] = set()
    previous: tuple[int, int] | None = None
    for index, row in enumerate(value):
        require(isinstance(row, list) and len(row) == 2,
                f"{label}[{index}] must be [stable_id,distance_sq]")
        stable_id, distance_sq = row
        stable_id = need_int(stable_id, f"{label}[{index}].stable_id")
        distance_sq = need_int(distance_sq, f"{label}[{index}].distance_sq")
        require(stable_id not in seen, f"{label} has duplicate stable ID {stable_id}")
        seen.add(stable_id)
        key = (distance_sq, stable_id)
        require(previous is None or previous < key,
                f"{label} is not strictly canonical (distance_sq,stable_id) order")
        previous = key
        rows.append([stable_id, distance_sq])
    return rows


def parse_labels(value: Any, label: str, expected_count: int, base_set: set[int]) -> list[int]:
    require(isinstance(value, list) and len(value) == expected_count,
            f"{label} must contain exactly {expected_count} labels")
    labels: list[int] = []
    seen: set[int] = set()
    for index, raw in enumerate(value):
        stable_id = need_int(raw, f"{label}[{index}]")
        require(stable_id in base_set, f"{label}[{index}] is not an immutable-base stable ID")
        require(stable_id not in seen, f"{label} contains duplicate stable ID {stable_id}")
        seen.add(stable_id)
        labels.append(stable_id)
    return labels


def topk_int64(pool: array.array, queries: array.array, mapping: array.array,
               query_id: int, candidate_ids: Iterable[int], dimension: int, k: int) -> list[list[int]]:
    require(0 <= query_id and (query_id + 1) * dimension <= len(queries),
            "query ID lies outside query matrix")
    q_offset = query_id * dimension
    best: list[tuple[int, int]] = []  # (distance_sq, stable_id)
    seen: set[int] = set()
    for stable_id in candidate_ids:
        require(is_int(stable_id) and 0 <= stable_id < len(mapping),
                "oracle candidate stable ID is out of mapping range")
        require(stable_id not in seen, "oracle candidate stable IDs are not unique")
        seen.add(stable_id)
        p_offset = int(mapping[stable_id]) * dimension
        distance_sq = 0
        for coordinate in range(dimension):
            delta = int(pool[p_offset + coordinate]) - int(queries[q_offset + coordinate])
            distance_sq += delta * delta
        item = (distance_sq, stable_id)
        if len(best) < k:
            best.append(item)
            if len(best) == k:
                best.sort()
        elif item < best[-1]:
            best[-1] = item
            best.sort()
    # When candidate_ids has fewer than k rows (notably early global_delta),
    # it never reached the in-loop full-k sort.  Canonicalize unconditionally.
    best.sort()
    return [[stable_id, distance_sq] for distance_sq, stable_id in best]


def merge_topk(rows_a: list[list[int]], rows_b: list[list[int]], k: int) -> list[list[int]]:
    merged = [[int(stable_id), int(distance_sq)] for stable_id, distance_sq in rows_a]
    merged.extend([int(stable_id), int(distance_sq)] for stable_id, distance_sq in rows_b)
    require(len({stable_id for stable_id, _ in merged}) == len(merged),
            "candidate base and global delta overlap")
    merged.sort(key=lambda row: (row[1], row[0]))
    return merged[:k]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    regular_file(path, "HNSW runner JSONL")
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                require(line.endswith("\n"), f"JSONL line {line_number} lacks LF terminator")
                try:
                    record = json.loads(line, parse_constant=reject_json_constant)
                except (json.JSONDecodeError, Fail) as exc:
                    raise Fail(f"invalid JSONL at line {line_number}: {exc}") from exc
                require(isinstance(record, dict), f"JSONL line {line_number} is not an object")
                records.append(record)
    except UnicodeDecodeError as exc:
        raise Fail(f"HNSW runner JSONL is not UTF-8: {exc}") from exc
    return records


def canonical_split_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def trace_knn_members(events: list[tuple[int, int, int]]) -> list[dict[str, int]]:
    delta_live = 0
    members: list[dict[str, int]] = []
    for op_index, opcode, argument in events:
        if opcode == INSERT:
            delta_live += 1
        else:
            members.append({"knn_ordinal": len(members), "op_index": op_index,
                            "query_id": argument, "global_delta_live": delta_live})
    require(len(members) == EXPECTED_KNNS and delta_live == EXPECTED_INSERTS,
            "projection cannot produce expected calibration membership")
    return members


def member_hash(members: list[dict[str, int]]) -> str:
    payload = "".join(
        f"{member['knn_ordinal']}\t{member['op_index']}\t{member['query_id']}\n"
        for member in members
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_split(split_path: Path, bundle: Path, env: dict[str, str],
                   header: dict[str, int], events: list[tuple[int, int, int]]) -> dict[int, str]:
    split = read_json_object(split_path, "calibration/test split")
    require(split.get("schema") == SPLIT_SCHEMA, "unexpected calibration/test split schema")
    supplied_hash = need_sha(split.get("split_sha256"), "split_sha256")
    unchecked = dict(split)
    unchecked.pop("split_sha256", None)
    require(hashlib.sha256(canonical_split_bytes(unchecked)).hexdigest() == supplied_hash,
            "calibration/test split self-hash mismatch")
    input_paths = split.get("input_paths")
    input_hashes = split.get("input_sha256")
    require(isinstance(input_paths, dict) and isinstance(input_hashes, dict),
            "split omits input binding objects")
    require(isinstance(input_paths.get("bundle"), str) and
            Path(input_paths["bundle"]).resolve(strict=True) == bundle,
            "split bundle path does not bind --bundle")
    expected_hashes = {
        "bundle_manifest_json_sha256": env_sha(env, "manifest_sha256"),
        "bundle_metadata_json_sha256": env_sha(env, "metadata_sha256"),
        "projection_trace_sha256": env_sha(env, "trace_sha256"),
        "source_trace_sha256": env_sha(env, "source_trace_sha256"),
    }
    for key, expected in expected_hashes.items():
        require(input_hashes.get(key) == expected, f"split input hash mismatch for {key}")
    projection = split.get("projection")
    require(isinstance(projection, dict), "split projection must be an object")
    for key, expected in (("event_count", header["event_count"]), ("k", header["k"]),
                          ("knn_count", EXPECTED_KNNS),
                          ("metadata_sha256", env_sha(env, "metadata_sha256")),
                          ("trace_sha256", env_sha(env, "trace_sha256"))):
        require(projection.get(key) == expected, f"split projection mismatch for {key}")
    policy = split.get("selection_policy")
    require(isinstance(policy, dict) and
            policy.get("rule") == "zero-based KNN ordinal in projected trace order: even ordinal => calibration; odd ordinal => held_out" and
            policy.get("membership_uses_only") ==
            "KNN ordinal; neither HNSW output nor Native-GTS overlap/exactness/candidate-count fields participate in membership allocation" and
            policy.get("quality_optimization") == "none",
            "split allocation policy drift")

    expected_members = trace_knn_members(events)
    expected_by_name = {
        "calibration": [item for item in expected_members if item["knn_ordinal"] % 2 == 0],
        "held_out": [item for item in expected_members if item["knn_ordinal"] % 2 == 1],
    }
    splits = split.get("splits")
    require(isinstance(splits, dict) and set(splits) == set(expected_by_name),
            "split partition names mismatch")
    membership: dict[int, str] = {}
    for name, expected in expected_by_name.items():
        partition = splits.get(name)
        require(isinstance(partition, dict), f"split partition {name} must be an object")
        supplied_members = partition.get("members")
        require(isinstance(supplied_members, list) and
                partition.get("member_count") == len(expected) == 68,
                f"split partition {name} cardinality mismatch")
        require(partition.get("member_sha256") == member_hash(expected),
                f"split partition {name} member hash mismatch")
        require(len(supplied_members) == len(expected),
                f"split partition {name} member list length mismatch")
        for ordinal, (observed, wanted) in enumerate(zip(supplied_members, expected)):
            require(isinstance(observed, dict) and set(observed) == set(wanted) and observed == wanted,
                    f"split partition {name} member mismatch at ordinal {ordinal}")
            op_index = wanted["op_index"]
            require(op_index not in membership, "split has an overlapping KNN member")
            membership[op_index] = name
    require(len(membership) == EXPECTED_KNNS, "split does not cover all KNN events")
    return membership


def require_root_private_module(path_value: Any, runtime_root_value: Any, label: str) -> Path:
    require(isinstance(path_value, str) and isinstance(runtime_root_value, str),
            f"{label} path/runtime root must be strings")
    root_raw = Path(runtime_root_value)
    root = real_directory(root_raw, f"{label} runtime root")
    root_st = lstat(root_raw, f"{label} runtime root")
    require(root_st.st_uid == 0 and (root_st.st_mode & 0o077) == 0,
            f"{label} runtime root is not root-owned and mode-0700")
    raw = Path(path_value)
    require(raw.is_absolute() and ".." not in raw.parts and raw.is_relative_to(root),
            f"{label} module path is not lexically beneath its runtime root")
    relative = raw.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        st = lstat(current, f"{label} component {current}")
        require(not stat.S_ISLNK(st.st_mode) and st.st_uid == 0 and
                (st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)) == 0,
                f"{label} has a symlink/non-root-owned/writable component: {current}")
    final = regular_file(raw, f"{label} module")
    require(final.is_relative_to(root), f"{label} resolved outside runtime root")
    return final


def counter_value(counters: dict[str, Any], canonical: str, aliases: tuple[str, ...]) -> int:
    present = [key for key in (canonical,) + aliases if key in counters]
    require(present, f"summary structural_counters omits {canonical}")
    value = need_int(counters[present[0]], f"structural_counters.{present[0]}")
    for key in present[1:]:
        require(need_int(counters[key], f"structural_counters.{key}") == value,
                f"inconsistent aliases for structural counter {canonical}")
    return value


def validate_summary(summary_path: Path, env: dict[str, str], header: dict[str, int],
                     event_stream_sha: str, final_hash: str, final_count: int,
                     inserts: int, knns: int, base_exact: int, merged_exact: int,
                     base_overlap: int, merged_overlap: int, sequence_sha: str,
                     observed_ef: set[int]) -> None:
    summary = read_json_object(summary_path, "HNSW runner summary")
    require(summary.get("schema") == RUN_SCHEMA and summary.get("mode") == "recall_probe" and
            summary.get("status") == "PASS_PROBE", "summary schema/mode/status mismatch")
    require(isinstance(summary.get("scope"), str) and "CPU-only" in summary["scope"] and
            "no timing claim" in summary["scope"], "summary scope lacks CPU-only/no-timing boundary")
    expected = {
        "event_count": header["event_count"],
        "insert_count": inserts,
        "knn_count": knns,
        "global_delta_live": inserts,
        "final_active_count": final_count,
        "final_active_set_sha256": final_hash,
        "manifest_sha256": env_sha(env, "manifest_sha256"),
        "metadata_sha256": env_sha(env, "metadata_sha256"),
        "trace_sha256": env_sha(env, "trace_sha256"),
        "projection_event_stream_sha256": event_stream_sha,
        "source_trace_sha256": env_sha(env, "source_trace_sha256"),
        "source_event_stream_sha256": env_sha(env, "source_event_stream_sha256"),
        "base_exact_match_count": base_exact,
        "merged_exact_match_count": merged_exact,
        "base_overlap_sum": base_overlap,
        "merged_overlap_sum": merged_overlap,
        "canonical_int64_knn_sequence_sha256": sequence_sha,
    }
    for key, wanted in expected.items():
        require(summary.get(key) == wanted, f"summary mismatch for {key}")
    require(summary.get("base_immutable") is True and
            summary.get("hnsw_base_candidate_path_used_for_trace") is True and
            summary.get("direct_sidecar_used") is False and
            summary.get("legacy_routing_used") is False and
            summary.get("provided_full_trace_oracle_used") is False and
            summary.get("full_active_int64_oracle_used") is True and
            summary.get("candidate_rescored_with_int64") is True and
            summary.get("float_hnsw_distances_used_for_ranking") is False and
            summary.get("gpu_used") is False and summary.get("timing_claim") is False,
            "summary Safe-C1 / CPU-only / no-timing flags mismatch")
    limitations = summary.get("limitations")
    require(isinstance(limitations, list) and all(isinstance(item, str) for item in limitations) and
            any("not a universal exactness claim" in item for item in limitations) and
            any("No elapsed time" in item for item in limitations),
            "summary limitations omit exactness/no-timing boundary")

    counters = summary.get("structural_counters")
    require(isinstance(counters, dict), "summary structural_counters must be an object")
    require(counter_value(counters, "hnsw_initial_base_add_items_calls",
                          ("base_add_items_calls",)) == 1,
            "initial immutable-base add_items call count must be exactly one")
    require(counter_value(counters, "hnsw_trace_add_items_calls",
                          ("trace_add_items_calls",)) == 0,
            "trace add_items call count must be zero")
    for key in ("hnsw_trace_mark_deleted_calls", "hnsw_trace_resize_index_calls",
                "hnsw_trace_index_mutation_calls"):
        require(need_int(counters.get(key), f"structural_counters.{key}") == 0,
                f"{key} must be zero")
    require(need_int(counters.get("external_global_delta_insert_count"),
                     "structural_counters.external_global_delta_insert_count") == inserts,
            "external global delta insertion count mismatch")

    hnsw = summary.get("hnsw")
    require(isinstance(hnsw, dict), "summary.hnsw must be an object")
    require(hnsw.get("library") == "hnswlib" and hnsw.get("library_version") == "0.8.0" and
            hnsw.get("space") == "l2" and hnsw.get("M") == 16 and
            hnsw.get("ef_construction") == 200 and hnsw.get("random_seed") == 20260727 and
            hnsw.get("threads") == 1 and hnsw.get("max_elements") == header["base_n"] and
            hnsw.get("current_count_after_trace") == header["base_n"],
            "summary HNSW immutable-base configuration mismatch")
    ef_search = need_int(hnsw.get("ef_search"), "summary.hnsw.ef_search", header["k"])
    require(observed_ef == {ef_search}, "record-level HNSW efSearch disagrees with summary")
    lib_module = require_root_private_module(hnsw.get("library_module_path"),
                                             hnsw.get("runtime_root"), "hnswlib")
    numpy_module = require_root_private_module(hnsw.get("numpy_module_path"),
                                               hnsw.get("runtime_root"), "numpy")
    del numpy_module
    require(sha256_file(lib_module, "hnswlib module") ==
            need_sha(hnsw.get("library_module_sha256"), "summary.hnsw.library_module_sha256"),
            "summary HNSW module hash mismatch")
    require(isinstance(hnsw.get("numpy_version"), str) and hnsw["numpy_version"],
            "summary.hnsw.numpy_version must be a nonempty string")


def rate_counts(base_exact: int, merged_exact: int, base_overlap: int,
                merged_overlap: int, knn: int, k: int) -> dict[str, int]:
    return {
        "knn_count": knn,
        "k": k,
        "base_exact_match_count": base_exact,
        "merged_exact_match_count": merged_exact,
        "base_overlap_sum": base_overlap,
        "merged_overlap_sum": merged_overlap,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    bundle = real_directory(args.bundle, "bundle")
    env = validate_admission(bundle, args.admission)
    header, events, pool, queries, mapping, base_ids, event_stream_sha, metadata_sha = \
        parse_bundle(bundle, env)
    require(metadata_sha == env_sha(env, "metadata_sha256"), "metadata SHA drift")
    split_membership = validate_split(args.split, bundle, env, header, events)
    records = load_jsonl(args.output)
    require(len(records) == len(events) == EXPECTED_EVENT_COUNT,
            "HNSW JSONL must contain exactly one record for all 305 projected events")

    active = bytearray(header["pool_n"])
    immutable_base = tuple(int(stable_id) for stable_id in base_ids)
    immutable_base_set = set(immutable_base)
    for stable_id in immutable_base:
        active[stable_id] = 1
    global_delta: list[int] = []
    inserts = 0
    knns = 0
    base_exact_count = 0
    merged_exact_count = 0
    base_overlap_sum = 0
    merged_overlap_sum = 0
    observed_ef: set[int] = set()
    sequence = hashlib.sha256()
    split_totals = {
        "calibration": {"base_exact": 0, "merged_exact": 0, "base_overlap": 0,
                        "merged_overlap": 0, "knn": 0},
        "held_out": {"base_exact": 0, "merged_exact": 0, "base_overlap": 0,
                       "merged_overlap": 0, "knn": 0},
    }

    for event, record in zip(events, records):
        op_index, opcode, argument = event
        require(record.get("op_index") == op_index, f"output op_index mismatch at {op_index}")
        if opcode == INSERT:
            require(record.get("record") == "update" and record.get("op") == "insert" and
                    record.get("stable_id") == argument and
                    record.get("placement") == "global_delta" and
                    record.get("base_mutated") is False and
                    record.get("hnsw_trace_index_mutated") is False and
                    record.get("hnsw_trace_add_items_called") is False,
                    f"frozen-base/global-delta update contract mismatch at op {op_index}")
            require(not active[argument] and argument not in immutable_base_set,
                    f"invalid/repeated projected insertion at op {op_index}")
            active[argument] = 1
            global_delta.append(argument)
            inserts += 1
            continue

        require(record.get("record") == "knn" and record.get("query_id") == argument and
                record.get("k") == header["k"], f"KNN record shape mismatch at op {op_index}")
        require(record.get("hnsw_base_candidate_path") ==
                "hnswlib_knn_query_frozen_immutable_base" and
                record.get("hnsw_base_candidate_path_used_for_trace") is True and
                record.get("hnsw_trace_index_mutated") is False and
                record.get("hnsw_add_items_called_for_trace") is False and
                record.get("base_immutable") is True and
                record.get("direct_sidecar_used") is False and
                record.get("candidate_rescored_with_int64") is True and
                record.get("float_hnsw_distances_used_for_ranking") is False and
                record.get("full_active_int64_oracle_used") is True and
                record.get("global_delta_live") == inserts,
                f"HNSW Safe-C1 candidate-path declaration mismatch at op {op_index}")
        require(need_int(record.get("hnsw_base_candidate_count"),
                         f"op {op_index} hnsw_base_candidate_count") == header["k"] and
                need_int(record.get("hnsw_returned_distance_count"),
                         f"op {op_index} hnsw_returned_distance_count") == header["k"],
                f"HNSW must return exactly {header['k']} base candidates at op {op_index}")
        ef_search = need_int(record.get("hnsw_ef_search"), f"op {op_index} hnsw_ef_search",
                             header["k"])
        observed_ef.add(ef_search)
        labels = parse_labels(record.get("hnsw_returned_labels"),
                              f"op {op_index} hnsw_returned_labels", header["k"],
                              immutable_base_set)
        candidate_base = parse_rows(record.get("base_candidate_topk"),
                                    f"op {op_index} base_candidate_topk", header["k"])
        require({stable_id for stable_id, _ in candidate_base} == set(labels),
                f"op {op_index} rescored candidate IDs differ from raw HNSW labels")
        expected_candidate = topk_int64(pool, queries, mapping, argument, labels,
                                        header["dimension"], header["k"])
        require(candidate_base == expected_candidate,
                f"op {op_index} candidate int64 re-score/canonical order mismatch")
        expected_delta = topk_int64(pool, queries, mapping, argument, global_delta,
                                    header["dimension"], header["k"])
        exact_delta = parse_rows(record.get("exact_global_delta_topk"),
                                 f"op {op_index} exact_global_delta_topk",
                                 min(header["k"], len(global_delta)))
        require(exact_delta == expected_delta,
                f"op {op_index} global delta is not exact int64 canonical top-k")
        merged = parse_rows(record.get("merged"), f"op {op_index} merged", header["k"])
        expected_merged = merge_topk(candidate_base, expected_delta, header["k"])
        require(merged == expected_merged,
                f"op {op_index} merged candidate/delta top-k mismatch")
        expected_base = topk_int64(pool, queries, mapping, argument, immutable_base,
                                   header["dimension"], header["k"])
        expected_full = topk_int64(pool, queries, mapping, argument,
                                   (stable_id for stable_id, present in enumerate(active) if present),
                                   header["dimension"], header["k"])
        require(len(expected_base) == header["k"] and len(expected_full) == header["k"],
                f"op {op_index} exact oracle cardinality mismatch")
        base_overlap = len({stable_id for stable_id, _ in candidate_base} &
                           {stable_id for stable_id, _ in expected_base})
        merged_overlap = len({stable_id for stable_id, _ in merged} &
                             {stable_id for stable_id, _ in expected_full})
        base_exact = candidate_base == expected_base
        merged_exact = merged == expected_full
        require(record.get("base_overlap_count") == base_overlap and
                record.get("merged_overlap_count") == merged_overlap and
                record.get("base_exact_match") is base_exact and
                record.get("merged_exact_match") is merged_exact,
                f"op {op_index} overlap/exactness fields disagree with independent oracle")
        sequence.update(f"{op_index}:{argument}:{active_hash(active)}:".encode("ascii"))
        for label, rows in (("candidate_base", candidate_base), ("candidate_merged", merged),
                            ("oracle_base", expected_base), ("oracle_merged", expected_full)):
            sequence.update((label + ":").encode("ascii"))
            for stable_id, distance_sq in rows:
                sequence.update(f"{stable_id}:{distance_sq},".encode("ascii"))
        sequence.update(b"\n")
        base_exact_count += int(base_exact)
        merged_exact_count += int(merged_exact)
        base_overlap_sum += base_overlap
        merged_overlap_sum += merged_overlap
        knns += 1
        split_name = split_membership.get(op_index)
        require(split_name in split_totals, f"op {op_index} has no sealed split membership")
        totals = split_totals[split_name]
        totals["base_exact"] += int(base_exact)
        totals["merged_exact"] += int(merged_exact)
        totals["base_overlap"] += base_overlap
        totals["merged_overlap"] += merged_overlap
        totals["knn"] += 1

    final_hash = active_hash(active)
    final_count = active_count(active)
    require(inserts == header["insert_count"] == EXPECTED_INSERTS and
            knns == header["knn_count"] == EXPECTED_KNNS,
            "output operation counts do not equal sealed projection")
    require(final_count == env_int(env, "final_active_count") and
            final_hash == env_sha(env, "final_active_set_sha256"),
            "output final active-state witness differs from sealed projection")
    validate_summary(args.summary, env, header, event_stream_sha, final_hash, final_count,
                     inserts, knns, base_exact_count, merged_exact_count, base_overlap_sum,
                     merged_overlap_sum, sequence.hexdigest(), observed_ef)
    return {
        "schema": VALIDATOR_SCHEMA,
        "status": "PASS",
        "bundle": str(bundle),
        "admission": str(args.admission.resolve()),
        "split": str(args.split.resolve()),
        "output": str(args.output.resolve()),
        "summary": str(args.summary.resolve()),
        "trace_sha256": env_sha(env, "trace_sha256"),
        "metadata_sha256": env_sha(env, "metadata_sha256"),
        "validated_events": len(events),
        "insert": inserts,
        "knn": knns,
        "hnsw_ef_search": next(iter(observed_ef)),
        "base_exact_match_count": base_exact_count,
        "merged_exact_match_count": merged_exact_count,
        "base_overlap_sum": base_overlap_sum,
        "merged_overlap_sum": merged_overlap_sum,
        "final_active_count": final_count,
        "final_active_set_sha256": final_hash,
        "canonical_int64_knn_sequence_sha256": sequence.hexdigest(),
        "split_metrics": {
            name: rate_counts(values["base_exact"], values["merged_exact"],
                              values["base_overlap"], values["merged_overlap"],
                              values["knn"], header["k"])
            for name, values in split_totals.items()
        },
        "oracle": "pure-stdlib exact integer squared L2; canonical order=(distance_sq,stable_id); independently validates HNSW candidate re-score, exact external delta, full-active overlap, and immutable-base state",
        "scope": "CPU-only/no-timing frozen immutable-base HNSW candidate-recall probe; not a full dynamic, deletion, range, rebuild, direct-sidecar, GPU, or universal exactness result",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Pure-stdlib independent HNSW recall-probe validator")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = validate(args)
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except Fail as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
