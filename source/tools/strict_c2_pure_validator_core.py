#!/usr/bin/python3.12
"""Independent pure-stdlib validator for strict-C2 partition-local HNSW output.

This program never imports or executes hnswlib, NumPy, CUDA, GTS, NVML, or a
user-owned runtime.  It replays all 305 projected events to preserve insertion
state, but invokes its exact Python integer squared-L2 oracle only for the
presealed partition's 68 KNNs.  It validates candidate re-scoring, exact delta,
full-active overlap, immutable-base state, and the recorded no-cross-partition /
no-timing / CPU-only structural contract.
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
PARTITION_SEAL_SCHEMA = "safe-c1-hnsw-strict-c2-partition-seal-v1"
LOCK_SCHEMA = "safe-c1-hnsw-strict-c2-heldout-lock-v1"
SOURCE_SPLIT_SCHEMA = "safe-c1-hnsw-calibration-test-split-v1"
SOURCE_SPLIT_FILE_SHA256 = "92394b5aa7cee61d6f2583b08061d30a1a287ef5776c7dbe5c9a6a79ddadd585"
SOURCE_SPLIT_CANONICAL_SHA256 = "c825f1e29f80fcb93b0d302e8a3da3273de0de578a839bd72d86eceb5b532355"
RUN_SCHEMA = "fair-hnsw-strict-c2-partition-quality-probe-v1"
VALIDATOR_SCHEMA = "fair-hnsw-strict-c2-partition-output-validator-v1"

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




def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def official_member_hash(members: list[dict[str, int]]) -> str:
    # Exact official sealed-split invariant. Do not replace with a diagnostic
    # ordering or alternate serialization.
    return hashlib.sha256("".join(
        f"{item['knn_ordinal']}\t{item['op_index']}\t{item['query_id']}\n" for item in members
    ).encode("ascii")).hexdigest()


def trace_partition_members(events: list[tuple[int, int, int]], partition: str) -> list[dict[str, int]]:
    require(partition in {"calibration", "held_out"}, "unknown strict-C2 partition")
    parity = 0 if partition == "calibration" else 1
    delta_live = 0
    output: list[dict[str, int]] = []
    ordinal = 0
    for op_index, opcode, argument in events:
        if opcode == INSERT:
            delta_live += 1
        else:
            if ordinal % 2 == parity:
                output.append({"knn_ordinal": ordinal, "op_index": op_index,
                               "query_id": argument, "global_delta_live": delta_live})
            ordinal += 1
    require(delta_live == EXPECTED_INSERTS and ordinal == EXPECTED_KNNS and len(output) == 68,
            "sealed trace cannot produce strict-C2 partition membership")
    return output


def validate_partition_seal(path: Path, bundle: Path, env: dict[str, str],
                            header: dict[str, int], events: list[tuple[int, int, int]],
                            partition: str) -> tuple[dict[int, dict[str, int]], str, str]:
    seal = read_json_object(path, "strict-C2 partition seal")
    require(seal.get("schema") == PARTITION_SEAL_SCHEMA and seal.get("partition") == partition,
            "strict-C2 partition seal schema/partition mismatch")
    self_sha = need_sha(seal.get("seal_sha256"), "partition seal SHA")
    unchecked = dict(seal)
    unchecked.pop("seal_sha256", None)
    require(hashlib.sha256(canonical_json(unchecked)).hexdigest() == self_sha,
            "partition seal self-hash mismatch")
    require(seal.get("source_split_schema") == SOURCE_SPLIT_SCHEMA and
            seal.get("source_split_file_sha256") == SOURCE_SPLIT_FILE_SHA256 and
            seal.get("source_split_canonical_sha256") == SOURCE_SPLIT_CANONICAL_SHA256,
            "partition seal source split provenance mismatch")
    require(seal.get("cross_partition_content_absent") is True and
            seal.get("member_sha256_scope") ==
            "SHA-256 of LF-delimited ASCII lines in KNN-ordinal order: knn_ordinal<TAB>op_index<TAB>query_id<LF>.",
            "partition seal cross-partition/member-hash declaration mismatch")
    require(seal.get("seal_sha256_scope") ==
            "SHA-256 of canonical UTF-8 JSON for this object with seal_sha256 omitted; sort_keys=true, separators=(',', ':'), trailing LF.",
            "partition seal self-hash scope mismatch")
    projection = seal.get("projection")
    require(isinstance(projection, dict) and set(projection) ==
            {"bundle_path", "manifest_sha256", "metadata_sha256", "trace_sha256",
             "source_trace_sha256", "event_count", "insert_count", "knn_count", "k"},
            "partition seal projection shape mismatch")
    require(isinstance(projection.get("bundle_path"), str) and
            Path(projection["bundle_path"]).resolve(strict=True) == bundle,
            "partition seal bundle path mismatch")
    for key, expected in {
        "manifest_sha256": env_sha(env, "manifest_sha256"),
        "metadata_sha256": env_sha(env, "metadata_sha256"),
        "trace_sha256": env_sha(env, "trace_sha256"),
        "source_trace_sha256": env_sha(env, "source_trace_sha256"),
        "event_count": header["event_count"], "insert_count": EXPECTED_INSERTS,
        "knn_count": EXPECTED_KNNS, "k": header["k"],
    }.items():
        require(projection.get(key) == expected, f"partition seal projection mismatch for {key}")
    members = seal.get("members")
    expected_members = trace_partition_members(events, partition)
    require(isinstance(members, list) and seal.get("member_count") == len(members) == len(expected_members),
            "partition seal member cardinality mismatch")
    normalized: list[dict[str, int]] = []
    for number, (raw, expected) in enumerate(zip(members, expected_members)):
        require(isinstance(raw, dict) and set(raw) ==
                {"knn_ordinal", "op_index", "query_id", "global_delta_live"},
                f"partition seal member {number} shape mismatch")
        actual = {key: need_int(raw.get(key), f"partition seal member {number}.{key}")
                  for key in ("knn_ordinal", "op_index", "query_id", "global_delta_live")}
        require(actual == expected, f"partition seal member {number} differs from trace parity assignment")
        normalized.append(actual)
    digest = official_member_hash(normalized)
    expected_digest = ("eae0763680a0f71862a3b63e3c4429ffa467e5f9e7df2a03cc5607cbd552658d"
                       if partition == "calibration" else
                       "e96b8f45d71d3aa566e09c7b92a4844d664b30774c0f04850d7d0f8cea4a7aba")
    require(digest == expected_digest and seal.get("member_sha256") == digest,
            "strict-C2 official tab/ordinal member hash mismatch")
    common = {"schema", "scope", "partition", "source_split_schema", "source_split_file_sha256", "source_split_canonical_sha256",
              "projection", "member_sha256_scope", "member_count", "member_sha256", "members",
              "cross_partition_content_absent", "seal_sha256_scope", "seal_sha256"}
    if partition == "calibration":
        target = seal.get("calibration_native_gts_target")
        require(isinstance(target, dict) and set(target) ==
                {"merged_overlap_numerator", "merged_overlap_denominator",
                 "merged_exact_match_numerator", "merged_exact_match_denominator", "knn_count", "k"},
                "calibration seal target shape mismatch")
        wanted = (651, 680, 53, 68, 68, 10)
        got = tuple(need_int(target.get(key), f"calibration target.{key}") for key in
                    ("merged_overlap_numerator", "merged_overlap_denominator",
                     "merged_exact_match_numerator", "merged_exact_match_denominator", "knn_count", "k"))
        require(got == wanted, "calibration target mismatch")
        common.add("calibration_native_gts_target")
    else:
        require(seal.get("held_out_quality_target_embedded") is False and
                "calibration_native_gts_target" not in seal,
                "held-out seal carries forbidden quality target")
        common.add("held_out_quality_target_embedded")
    require(set(seal) == common, "partition seal has unrecognized/cross-partition fields")
    return {item["op_index"]: item for item in normalized}, digest, self_sha


def root_private_regular(path_value: Any, label: str) -> Path:
    require(isinstance(path_value, str) and Path(path_value).is_absolute(),
            f"{label} must be an absolute path")
    path = Path(path_value)
    file = regular_file(path, label)
    info = lstat(path, label)
    require(info.st_uid == 0 and (info.st_mode & 0o077) == 0,
            f"{label} must be root-owned/private")
    parent = path.parent
    parent_info = lstat(parent, f"{label} parent")
    require(stat.S_ISDIR(parent_info.st_mode) and not stat.S_ISLNK(parent_info.st_mode) and
            parent_info.st_uid == 0 and (parent_info.st_mode & 0o077) == 0,
            f"{label} parent must be root-owned/private")
    return file


def require_root_private_module(path_value: Any, runtime_root_value: Any, label: str) -> Path:
    require(isinstance(path_value, str) and isinstance(runtime_root_value, str),
            f"{label} path/runtime root must be strings")
    root_raw = Path(runtime_root_value)
    root = real_directory(root_raw, f"{label} runtime root")
    root_st = lstat(root_raw, f"{label} runtime root")
    require(root_st.st_uid == 0 and (root_st.st_mode & 0o077) == 0,
            f"{label} runtime root is not root-owned/private")
    raw = Path(path_value)
    require(raw.is_absolute() and ".." not in raw.parts and raw.is_relative_to(root),
            f"{label} module path is not lexically beneath runtime root")
    current = root
    for part in raw.relative_to(root).parts:
        current = current / part
        st = lstat(current, f"{label} component {current}")
        require(not stat.S_ISLNK(st.st_mode) and st.st_uid == 0 and
                (st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)) == 0,
                f"{label} has a symlink/non-root-owned/writable component")
    return regular_file(raw, f"{label} module")


def validate_lock(path: Path, bundle: Path, seal_sha: str, partition: str) -> str | None:
    if partition == "calibration":
        require(path is None, "calibration validator must not receive a held-out lock")
        return None
    lock = read_json_object(path, "held-out lock receipt")
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("status") == "LOCKED" and
            lock.get("partition") == "held_out" and lock.get("selected_on_partition") == "calibration",
            "held-out lock schema/status/partition mismatch")
    expected_keys = {"schema", "status", "scope", "partition", "selected_on_partition",
                     "selection_policy", "candidate_ef_grid", "locked_ef_search",
                     "calibration_selection_sha256", "calibration_seal_sha256", "heldout_seal_sha256",
                     "source_split_file_sha256", "source_split_canonical_sha256", "manifest_sha256", "metadata_sha256", "trace_sha256",
                     "lock_sha256_scope", "lock_sha256"}
    require(set(lock) == expected_keys, "held-out lock key set mismatch")
    require(lock.get("selection_policy") ==
            "smallest pre-registered EF satisfying calibration merged_overlap_sum >= 651/680 and merged_exact_match_count >= 53/68",
            "held-out lock selection policy drift")
    require(lock.get("candidate_ef_grid") == [17, 18, 19, 20, 21, 22, 23],
            "held-out lock candidate grid drift")
    locked = need_int(lock.get("locked_ef_search"), "held-out lock EF", EXPECTED_K)
    require(locked in lock["candidate_ef_grid"], "held-out lock EF not in candidate grid")
    for key, expected in {
        "heldout_seal_sha256": seal_sha,
        "source_split_file_sha256": SOURCE_SPLIT_FILE_SHA256,
        "source_split_canonical_sha256": SOURCE_SPLIT_CANONICAL_SHA256,
        "manifest_sha256": sha256_file(bundle / "manifest.json", "manifest.json"),
        "metadata_sha256": sha256_file(bundle / "metadata.json", "metadata.json"),
        "trace_sha256": sha256_file(bundle / "trace.e1gtrc", "trace.e1gtrc"),
    }.items():
        require(lock.get(key) == expected, f"held-out lock binding mismatch for {key}")
    need_sha(lock.get("calibration_selection_sha256"), "held-out lock calibration selection SHA")
    need_sha(lock.get("calibration_seal_sha256"), "held-out lock calibration seal SHA")
    require(lock.get("lock_sha256_scope") ==
            "SHA-256 of canonical UTF-8 JSON for this object with lock_sha256 omitted; sort_keys=true, separators=(',', ':'), trailing LF.",
            "held-out lock self-hash scope drift")
    supplied = need_sha(lock.get("lock_sha256"), "held-out lock self hash")
    raw = dict(lock)
    raw.pop("lock_sha256", None)
    require(hashlib.sha256(canonical_json(raw)).hexdigest() == supplied,
            "held-out lock self-hash mismatch")
    return supplied


def rows_for_record(record: dict[str, Any], pool: array.array, queries: array.array,
                    mapping: array.array, immutable_base: tuple[int, ...],
                    immutable_base_set: set[int], active: bytearray, global_delta: list[int],
                    header: dict[str, int], expected_member: dict[str, int], partition: str,
                    observed_ef: set[int]) -> tuple[int, int, bool, bool, list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    op_index = expected_member["op_index"]
    query_id = expected_member["query_id"]
    expected_keys = {"record", "partition", "knn_ordinal", "op_index", "query_id", "k",
                     "global_delta_live", "hnsw_base_candidate_path", "hnsw_base_candidate_count",
                     "hnsw_returned_labels", "hnsw_returned_distance_count", "hnsw_ef_search",
                     "hnsw_base_candidate_path_used", "hnsw_trace_index_mutated",
                     "hnsw_add_items_called_for_trace", "base_immutable", "direct_sidecar_used",
                     "candidate_rescored_with_int64", "float_hnsw_distances_used_for_ranking",
                     "full_active_int64_oracle_used", "base_overlap_count", "merged_overlap_count",
                     "base_exact_match", "merged_exact_match", "base_candidate_topk",
                     "exact_global_delta_topk", "merged"}
    require(set(record) == expected_keys, f"op {op_index}: record contains foreign/omitted fields")
    require(record.get("record") == "knn" and record.get("partition") == partition and
            record.get("knn_ordinal") == expected_member["knn_ordinal"] and
            record.get("op_index") == op_index and record.get("query_id") == query_id and
            record.get("k") == header["k"] and record.get("global_delta_live") == len(global_delta),
            f"op {op_index}: selected partition record identity/time-state mismatch")
    require(record.get("hnsw_base_candidate_path") ==
            "hnswlib_knn_query_frozen_immutable_base" and
            record.get("hnsw_base_candidate_path_used") is True and
            record.get("hnsw_trace_index_mutated") is False and
            record.get("hnsw_add_items_called_for_trace") is False and
            record.get("base_immutable") is True and record.get("direct_sidecar_used") is False and
            record.get("candidate_rescored_with_int64") is True and
            record.get("float_hnsw_distances_used_for_ranking") is False and
            record.get("full_active_int64_oracle_used") is True,
            f"op {op_index}: HNSW Safe-C1 declarations drift")
    require(need_int(record.get("hnsw_base_candidate_count"), f"op {op_index} candidate count") == header["k"] and
            need_int(record.get("hnsw_returned_distance_count"), f"op {op_index} distance count") == header["k"],
            f"op {op_index}: HNSW candidate cardinality drift")
    ef = need_int(record.get("hnsw_ef_search"), f"op {op_index} EF", header["k"])
    observed_ef.add(ef)
    labels = parse_labels(record.get("hnsw_returned_labels"), f"op {op_index} labels", header["k"], immutable_base_set)
    candidate = parse_rows(record.get("base_candidate_topk"), f"op {op_index} candidate", header["k"])
    require({stable_id for stable_id, _ in candidate} == set(labels),
            f"op {op_index}: int64 candidate rescore IDs mismatch labels")
    expected_candidate = topk_int64(pool, queries, mapping, query_id, labels, header["dimension"], header["k"])
    require(candidate == expected_candidate, f"op {op_index}: candidate int64 re-score mismatch")
    expected_delta = topk_int64(pool, queries, mapping, query_id, global_delta, header["dimension"], header["k"])
    delta = parse_rows(record.get("exact_global_delta_topk"), f"op {op_index} delta", min(header["k"], len(global_delta)))
    require(delta == expected_delta, f"op {op_index}: external delta exactness mismatch")
    merged = parse_rows(record.get("merged"), f"op {op_index} merged", header["k"])
    require(merged == merge_topk(candidate, expected_delta, header["k"]),
            f"op {op_index}: candidate/delta merge mismatch")
    # Pure validator invokes its exhaustive oracles only for the selected
    # partition.  The caller's skipped branch never calls this function.
    exact_base = topk_int64(pool, queries, mapping, query_id, immutable_base, header["dimension"], header["k"])
    exact_full = topk_int64(pool, queries, mapping, query_id,
                            (stable_id for stable_id, present in enumerate(active) if present),
                            header["dimension"], header["k"])
    base_overlap = len({stable_id for stable_id, _ in candidate} & {stable_id for stable_id, _ in exact_base})
    merged_overlap = len({stable_id for stable_id, _ in merged} & {stable_id for stable_id, _ in exact_full})
    base_exact = candidate == exact_base
    merged_exact = merged == exact_full
    require(record.get("base_overlap_count") == base_overlap and
            record.get("merged_overlap_count") == merged_overlap and
            record.get("base_exact_match") is base_exact and record.get("merged_exact_match") is merged_exact,
            f"op {op_index}: record metric fields disagree with pure oracle")
    return base_overlap, merged_overlap, base_exact, merged_exact, candidate, merged, exact_base, exact_full


def canonical_sequence(sequence: Any, op_index: int, query_id: int, active: bytearray,
                       candidate: list[list[int]], merged: list[list[int]],
                       exact_base: list[list[int]], exact_full: list[list[int]]) -> None:
    sequence.update(f"{op_index}:{query_id}:{active_hash(active)}:".encode("ascii"))
    for label, rows in (("candidate_base", candidate), ("candidate_merged", merged),
                        ("oracle_base", exact_base), ("oracle_merged", exact_full)):
        sequence.update((label + ":").encode("ascii"))
        for stable_id, distance_sq in rows:
            sequence.update(f"{stable_id}:{distance_sq},".encode("ascii"))
    sequence.update(b"\n")


def validate_summary(summary_path: Path, bundle: Path, env: dict[str, str], header: dict[str, int],
                     event_stream_sha: str, partition: str, seal_sha: str, member_sha: str,
                     lock_sha: str | None, metrics: dict[str, Any], expected_ef: int,
                     runner_entrypoint: Path) -> None:
    summary = read_json_object(summary_path, "strict-C2 runner summary")
    other = "held_out" if partition == "calibration" else "calibration"
    expected_keys = {"schema", "status", "mode", "partition", "scope", "input", "source",
                     "partition_seal", "heldout_lock_sha256", "event_replay", "partition_metrics",
                     "no_cross_partition_metrics", "hnsw", "structural_counters", "base_immutable",
                     "direct_sidecar_used", "legacy_routing_used", "provided_full_trace_oracle_used",
                     "gpu_used", "timing_claim", "limitations"}
    require(set(summary) == expected_keys, "summary has foreign/omitted (possibly cross-partition) fields")
    require(summary.get("schema") == RUN_SCHEMA and summary.get("status") == "PASS_PARTITION_PROBE" and
            summary.get("mode") == "strict_c2_partition_quality_probe" and summary.get("partition") == partition,
            "summary schema/status/mode/partition mismatch")
    require(isinstance(summary.get("scope"), str) and "CPU-only/no-timing" in summary["scope"] and
            "not a performance claim" in summary["scope"], "summary scope boundary missing")
    require(summary.get("heldout_lock_sha256") == lock_sha,
            "summary held-out lock hash mismatch")
    expected_input = {"manifest_sha256": env_sha(env, "manifest_sha256"),
                      "metadata_sha256": env_sha(env, "metadata_sha256"),
                      "trace_sha256": env_sha(env, "trace_sha256"),
                      "projection_event_stream_sha256": event_stream_sha,
                      "source_trace_sha256": env_sha(env, "source_trace_sha256"),
                      "source_event_stream_sha256": env_sha(env, "source_event_stream_sha256")}
    require(summary.get("input") == expected_input, "summary immutable input binding mismatch")
    source = summary.get("source")
    require(isinstance(source, dict) and set(source) == {"entrypoint_path", "entrypoint_sha256", "core_path", "core_sha256"},
            "summary source binding shape mismatch")
    expected_runner = root_private_regular(str(runner_entrypoint.resolve()), "expected runner entrypoint")
    observed_runner = root_private_regular(source.get("entrypoint_path"), "summary runner entrypoint")
    require(observed_runner == expected_runner and source.get("entrypoint_sha256") ==
            sha256_file(expected_runner, "expected runner entrypoint"), "summary runner source binding mismatch")
    core = root_private_regular(source.get("core_path"), "summary runner core")
    require(core.name == "strict_c2_hnsw_core.py" and source.get("core_sha256") ==
            sha256_file(core, "summary runner core"), "summary runner core source binding mismatch")
    require(summary.get("partition_seal") == {"sha256": seal_sha, "member_sha256": member_sha,
                                               "member_count": 68,
                                               "source_split_file_sha256": SOURCE_SPLIT_FILE_SHA256,
                                               "source_split_canonical_sha256": SOURCE_SPLIT_CANONICAL_SHA256},
            "summary partition seal binding mismatch")
    require(summary.get("event_replay") == {"event_count": EXPECTED_EVENT_COUNT,
                                             "insert_replayed_count": EXPECTED_INSERTS,
                                             "knn_seen_count": EXPECTED_KNNS,
                                             "evaluated_partition_knn_count": 68,
                                             "skipped_other_partition_knn_count": 68,
                                             "all_trace_inserts_replayed_in_order": True},
            "summary full insertion replay / partition count mismatch")
    require(summary.get("partition_metrics") == metrics,
            "summary selected partition metrics mismatch")
    require(summary.get("no_cross_partition_metrics") == {
        "evaluated_partition_only": True, "other_partition_name": other,
        "other_partition_hnsw_query_invocations": 0,
        "other_partition_oracle_evaluations": 0,
        "other_partition_records_written": 0,
        "other_partition_metric_aggregates_written": 0,
        "other_partition_quality_target_read": False,
        "skipped_other_partition_knn_count": 68,
    }, "summary does not prove no cross-partition metric computation/output")
    require(summary.get("base_immutable") is True and summary.get("direct_sidecar_used") is False and
            summary.get("legacy_routing_used") is False and summary.get("provided_full_trace_oracle_used") is False and
            summary.get("gpu_used") is False and summary.get("timing_claim") is False,
            "summary Safe-C1 / CPU/no-timing flags mismatch")
    limits = summary.get("limitations")
    require(isinstance(limits, list) and len(limits) == 4 and all(isinstance(item, str) for item in limits) and
            any("other 68 KNNs" in item and "without HNSW or oracle invocation" in item for item in limits) and
            any("No elapsed time" in item for item in limits),
            "summary limitations omit partition/no-timing boundary")
    counters = summary.get("structural_counters")
    require(isinstance(counters, dict) and counters == {
        "hnsw_initial_base_add_items_calls": 1, "hnsw_trace_add_items_calls": 0,
        "hnsw_trace_mark_deleted_calls": 0, "hnsw_trace_resize_index_calls": 0,
        "hnsw_trace_index_mutation_calls": 0, "external_global_delta_insert_count": EXPECTED_INSERTS,
        "selected_partition_hnsw_query_invocations": 68, "selected_partition_oracle_evaluations": 68,
        "skipped_partition_hnsw_query_invocations": 0, "skipped_partition_oracle_evaluations": 0,
        "skipped_partition_records_written": 0,
    }, "summary HNSW/partition structural counter mismatch")
    hnsw = summary.get("hnsw")
    require(isinstance(hnsw, dict) and hnsw.get("library") == "hnswlib" and
            hnsw.get("library_version") == "0.8.0" and hnsw.get("space") == "l2" and
            hnsw.get("M") == 16 and hnsw.get("ef_construction") == 200 and
            hnsw.get("random_seed") == 20260727 and hnsw.get("threads") == 1 and
            hnsw.get("max_elements") == header["base_n"] and hnsw.get("current_count_after_trace") == header["base_n"],
            "summary HNSW immutable-base configuration mismatch")
    require(hnsw.get("ef_search") == expected_ef, "summary HNSW EF mismatch")
    lib = require_root_private_module(hnsw.get("library_module_path"), hnsw.get("runtime_root"), "hnswlib")
    numpy = require_root_private_module(hnsw.get("numpy_module_path"), hnsw.get("runtime_root"), "numpy")
    del numpy
    require(sha256_file(lib, "hnswlib module") == need_sha(hnsw.get("library_module_sha256"), "summary hnsw module SHA") and
            isinstance(hnsw.get("numpy_version"), str) and bool(hnsw["numpy_version"]),
            "summary trusted runtime binding mismatch")


def validate_partition(bundle_path: Path, admission_path: Path, output_path: Path, summary_path: Path,
                       seal_path: Path, partition: str, runner_entrypoint: Path,
                       lock_path: Path | None = None) -> dict[str, Any]:
    bundle = real_directory(bundle_path, "bundle")
    env = validate_admission(bundle, admission_path)
    header, events, pool, queries, mapping, base_ids, event_stream_sha, metadata_sha = parse_bundle(bundle, env)
    require(metadata_sha == env_sha(env, "metadata_sha256"), "metadata SHA drift")
    selected_by_op, member_sha, seal_sha = validate_partition_seal(seal_path, bundle, env, header, events, partition)
    lock_sha = validate_lock(lock_path, bundle, seal_sha, partition)
    records = load_jsonl(output_path)
    require(len(records) == 68, "strict-C2 JSONL must contain only 68 selected KNN records")
    active = bytearray(header["pool_n"])
    immutable_base = tuple(int(item) for item in base_ids)
    immutable_base_set = set(immutable_base)
    for stable_id in immutable_base:
        active[stable_id] = 1
    global_delta: list[int] = []
    record_index = 0
    inserts = knns_seen = evaluated = skipped = 0
    base_exact = merged_exact = base_overlap = merged_overlap = 0
    observed_ef: set[int] = set()
    sequence = hashlib.sha256()
    for op_index, opcode, argument in events:
        if opcode == INSERT:
            require(not active[argument] and argument not in immutable_base_set,
                    f"invalid/repeated projected insertion at op {op_index}")
            active[argument] = 1
            global_delta.append(argument)
            inserts += 1
            continue
        ordinal = knns_seen
        knns_seen += 1
        member = selected_by_op.get(op_index)
        if member is None:
            # Do not call rows_for_record or the integer oracle on skipped KNNs.
            skipped += 1
            continue
        require(member["knn_ordinal"] == ordinal and member["query_id"] == argument and
                member["global_delta_live"] == len(global_delta),
                f"selected member trace/time state mismatch at op {op_index}")
        require(record_index < len(records), "selected KNN output ends early")
        result = rows_for_record(records[record_index], pool, queries, mapping,
                                 immutable_base, immutable_base_set, active, global_delta,
                                 header, member, partition, observed_ef)
        record_index += 1
        bo, mo, be, me, candidate, merged, exact_base, exact_full = result
        base_overlap += bo
        merged_overlap += mo
        base_exact += int(be)
        merged_exact += int(me)
        canonical_sequence(sequence, op_index, argument, active, candidate, merged, exact_base, exact_full)
        evaluated += 1
    require(record_index == len(records) and inserts == EXPECTED_INSERTS and knns_seen == EXPECTED_KNNS and
            evaluated == 68 and skipped == 68, "strict-C2 selected/skipped replay cardinality mismatch")
    require(len(observed_ef) == 1, "selected output has inconsistent EF")
    ef = next(iter(observed_ef))
    final_hash = active_hash(active)
    require(active_count(active) == env_int(env, "final_active_count") and
            final_hash == env_sha(env, "final_active_set_sha256"),
            "full insertion replay final active-state witness mismatch")
    metrics = {"knn_count": evaluated, "k": header["k"],
               "base_exact_match_count": base_exact, "merged_exact_match_count": merged_exact,
               "base_overlap_sum": base_overlap, "merged_overlap_sum": merged_overlap,
               "canonical_int64_knn_sequence_sha256": sequence.hexdigest(),
               "hnsw_ef_search": ef}
    # Runner summary deliberately keeps EF in its HNSW configuration rather than
    # in partition_metrics; pass it separately to the summary check.
    summary_metrics = dict(metrics)
    summary_metrics.pop("hnsw_ef_search")
    validate_summary(summary_path, bundle, env, header, event_stream_sha, partition, seal_sha,
                     member_sha, lock_sha, summary_metrics, ef, runner_entrypoint)
    return {
        "schema": VALIDATOR_SCHEMA, "status": "PASS", "partition": partition,
        "bundle": str(bundle), "admission": str(admission_path.resolve()),
        "partition_seal": str(seal_path.resolve()), "partition_seal_sha256": seal_sha,
        "output": str(output_path.resolve()), "summary": str(summary_path.resolve()),
        "heldout_lock": None if lock_path is None else str(lock_path.resolve()),
        "heldout_lock_sha256": lock_sha,
        "trace_sha256": env_sha(env, "trace_sha256"), "metadata_sha256": env_sha(env, "metadata_sha256"),
        "validated_events": EXPECTED_EVENT_COUNT, "insert_replayed_count": inserts,
        "knn_seen_count": knns_seen, "evaluated_partition_knn_count": evaluated,
        "skipped_other_partition_knn_count": skipped, "hnsw_ef_search": ef,
        "partition_metrics": summary_metrics,
        "no_cross_partition_metrics": {
            "other_partition_name": "held_out" if partition == "calibration" else "calibration",
            "other_partition_hnsw_query_invocations": 0,
            "other_partition_oracle_evaluations": 0,
            "other_partition_records_written": 0,
            "other_partition_metric_aggregates_written": 0,
            "validated_only_presealed_partition": True,
        },
        "oracle": "pure-stdlib exact integer squared L2 only on selected partition; skipped KNNs are not queried/oracled/written",
        "scope": "CPU-only/no-timing strict-C2 partition validator; not a full dynamic, deletion, range, rebuild, direct-sidecar, GPU, or universal exactness result",
    }
