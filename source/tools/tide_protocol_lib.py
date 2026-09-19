#!/usr/bin/env python3
"""Fail-closed, stdlib-only helpers for the TIDE Gates 2–3 protocol.

This module contains no host runner, CUDA, networking, subprocess calls, or
external data access. It parses files explicitly supplied by a caller.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

HEX32 = re.compile(r"^[0-9a-f]{32}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DISTANCE = re.compile(r"^(0|[1-9][0-9]*)$")
IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]+$")
FORBIDDEN_FIELD_PARTS = (
    "authorization", "credential", "password", "private_seed", "secret",
    "token", "ledger", "reservation",
)
REQUIRED_CASES = frozenset({
    "direct_admission",
    "sidecar_capacity_fallback",
    "certificate_rejection",
    "overlay_delete",
    "base_delete_rebuild_barrier",
    "first_query_after_rebuild",
    "nontrivial_receipt_witness",
})
FAULT_STAGES = frozenset({
    "traversal", "receipt", "receipt_transfer", "sidecar_enumeration",
    "delta_enumeration", "candidate_transfer", "distance_key", "final_merge",
})


class ProtocolError(ValueError):
    """Expected fail-closed validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{where}: expected JSON object")
    return value


def require_list(value: Any, where: str) -> Sequence[Any]:
    require(isinstance(value, list), f"{where}: expected JSON array")
    return value


def require_string(value: Any, where: str, *, nonempty: bool = True) -> str:
    require(isinstance(value, str), f"{where}: expected string")
    if nonempty:
        require(bool(value), f"{where}: empty string")
    return value


def require_identifier(value: Any, where: str) -> str:
    value = require_string(value, where)
    require(IDENTIFIER.fullmatch(value) is not None, f"{where}: invalid identifier")
    return value


def require_hex32(value: Any, where: str) -> str:
    value = require_string(value, where)
    require(HEX32.fullmatch(value) is not None, f"{where}: expected 32 lowercase hex")
    return value


def require_sha256(value: Any, where: str) -> str:
    value = require_string(value, where)
    require(HEX64.fullmatch(value) is not None, f"{where}: expected SHA-256 lowercase hex")
    return value


def require_nonnegative_int(value: Any, where: str, *, positive: bool = False) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{where}: expected integer")
    require(value > 0 if positive else value >= 0, f"{where}: out of range")
    return value


def require_distance_key(value: Any, where: str) -> str:
    value = require_string(value, where)
    require(DISTANCE.fullmatch(value) is not None, f"{where}: non-canonical nonnegative decimal")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"{path}: invalid JSON: {error}") from error
    return require_mapping(value, str(path))


def load_jsonl(path: Path) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                require(line.endswith("\n"), f"{path}:{line_no}: missing terminal newline")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ProtocolError(f"{path}:{line_no}: invalid JSON: {error}") from error
                rows.append(require_mapping(value, f"{path}:{line_no}"))
    except OSError as error:
        raise ProtocolError(f"{path}: cannot read: {error}") from error
    require(rows, f"{path}: empty JSONL")
    return rows


def write_canonical_json(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))


def print_result(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def ensure_no_private_fields(value: Any, where: str = "$") -> None:
    """Block clear secret-bearing fields in machine-readable protocol artifacts."""
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = require_string(key, f"{where}.<key>")
            lowered = key_text.lower()
            require(not any(part in lowered for part in FORBIDDEN_FIELD_PARTS),
                    f"{where}.{key_text}: forbidden private/control field")
            ensure_no_private_fields(child, f"{where}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            ensure_no_private_fields(child, f"{where}[{index}]")


def active_set_digest(stable_ids: Iterable[str]) -> str:
    ordered = sorted(stable_ids)
    for value in ordered:
        require_identifier(value, "stable_id")
    return sha256_bytes("".join(f"{value}\n" for value in ordered).encode("ascii"))


def validate_distance_pairs(rows: Any, where: str) -> List[Tuple[int, str]]:
    rows = require_list(rows, where)
    parsed: List[Tuple[int, str]] = []
    seen = set()
    for index, row in enumerate(rows):
        row = require_mapping(row, f"{where}[{index}]")
        stable_id = require_identifier(row.get("stable_id"), f"{where}[{index}].stable_id")
        distance = require_distance_key(row.get("distance_key"), f"{where}[{index}].distance_key")
        require(stable_id not in seen, f"{where}[{index}]: duplicate stable_id")
        seen.add(stable_id)
        parsed.append((int(distance), stable_id))
    require(parsed == sorted(parsed), f"{where}: not strictly ordered by (distance_key,stable_id)")
    return parsed


def load_catalog(path: Path) -> Dict[str, Any]:
    rows = load_jsonl(path)
    objects: Dict[str, Dict[str, Any]] = {}
    queries: Dict[str, Dict[str, Any]] = {}
    vector_dim = None
    for row_no, row in enumerate(rows, 1):
        ensure_no_private_fields(row, f"catalog[{row_no}]")
        kind = require_string(row.get("record_type"), f"catalog[{row_no}].record_type")
        if kind == "object":
            stable_id = require_identifier(row.get("stable_id"), f"catalog[{row_no}].stable_id")
            require(stable_id not in objects, f"catalog[{row_no}]: duplicate stable_id")
            membership = require_string(row.get("initial_membership"),
                                        f"catalog[{row_no}].initial_membership")
            require(membership in {"base", "arrival"},
                    f"catalog[{row_no}].initial_membership: expected base|arrival")
            vector = require_list(row.get("vector"), f"catalog[{row_no}].vector")
            require(vector, f"catalog[{row_no}].vector: empty")
            for dimension, component in enumerate(vector):
                require(isinstance(component, int) and not isinstance(component, bool),
                        f"catalog[{row_no}].vector[{dimension}]: expected signed integer")
            if vector_dim is None:
                vector_dim = len(vector)
            require(len(vector) == vector_dim, f"catalog[{row_no}].vector: dimension mismatch")
            objects[stable_id] = dict(row)
        elif kind == "query":
            query_id = require_identifier(row.get("query_id"), f"catalog[{row_no}].query_id")
            require(query_id not in queries, f"catalog[{row_no}]: duplicate query_id")
            partition = require_string(row.get("partition"), f"catalog[{row_no}].partition")
            require(partition in {"qualification", "calibration", "heldout"},
                    f"catalog[{row_no}].partition: invalid partition")
            vector = require_list(row.get("vector"), f"catalog[{row_no}].vector")
            require(vector, f"catalog[{row_no}].vector: empty")
            for dimension, component in enumerate(vector):
                require(isinstance(component, int) and not isinstance(component, bool),
                        f"catalog[{row_no}].vector[{dimension}]: expected signed integer")
            if vector_dim is None:
                vector_dim = len(vector)
            require(len(vector) == vector_dim, f"catalog[{row_no}].vector: dimension mismatch")
            queries[query_id] = dict(row)
        else:
            raise ProtocolError(f"catalog[{row_no}].record_type: expected object|query")
    require(objects and queries, "catalog: needs objects and queries")
    require(vector_dim is not None, "catalog: missing vector dimension")
    return {"objects": objects, "queries": queries, "dimension": vector_dim,
            "sha256": sha256_file(path)}


def validate_trace_rows(rows: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any]) -> Dict[str, Any]:
    header = require_mapping(rows[0], "trace[1]")
    ensure_no_private_fields(header, "trace[1]")
    require(header.get("record_type") == "trace_header", "trace[1].record_type: expected trace_header")
    require(header.get("schema") == "tide.lifecycle-trace.v1", "trace[1].schema mismatch")
    trace_id = require_hex32(header.get("trace_id"), "trace[1].trace_id")
    require(header.get("metric_contract") == "integer_l2_squared_v1",
            "trace[1].metric_contract must be integer_l2_squared_v1")
    k = require_nonnegative_int(header.get("k"), "trace[1].k", positive=True)
    require(header.get("tie_order") == ["distance_key", "stable_id"],
            "trace[1].tie_order mismatch")
    require(header.get("sealed") is True, "trace[1].sealed must be true")
    cases = set(require_list(header.get("required_cases"), "trace[1].required_cases"))
    require(cases == REQUIRED_CASES, "trace[1].required_cases must be the exact E1 set")
    require_sha256(header.get("catalog_sha256"), "trace[1].catalog_sha256")
    require(header["catalog_sha256"] == catalog["sha256"], "trace[1].catalog_sha256 mismatch")
    base = {stable_id for stable_id, row in catalog["objects"].items()
            if row["initial_membership"] == "base"}
    live = set(base)
    overlay = set()
    all_cases = set()
    need_first_query = False
    rebuild_count = 0
    operations: List[Dict[str, Any]] = []
    for line_no, raw in enumerate(rows[1:], 2):
        row = require_mapping(raw, f"trace[{line_no}]")
        ensure_no_private_fields(row, f"trace[{line_no}]")
        seq = require_nonnegative_int(row.get("seq"), f"trace[{line_no}].seq", positive=True)
        require(seq == len(operations) + 1, f"trace[{line_no}].seq: not contiguous")
        operation = require_string(row.get("op"), f"trace[{line_no}].op")
        tags = set(require_list(row.get("required_cases"), f"trace[{line_no}].required_cases"))
        require(tags <= REQUIRED_CASES, f"trace[{line_no}].required_cases: unknown tag")
        all_cases |= tags
        if operation == "insert":
            stable_id = require_identifier(row.get("stable_id"), f"trace[{line_no}].stable_id")
            require(stable_id in catalog["objects"], f"trace[{line_no}]: unknown object")
            require(catalog["objects"][stable_id]["initial_membership"] == "arrival",
                    f"trace[{line_no}]: insert may reference arrival object only")
            require(stable_id not in live, f"trace[{line_no}]: insertion of live object")
            expected = require_string(row.get("expected_insert_outcome"),
                                      f"trace[{line_no}].expected_insert_outcome")
            tag_outcome = {
                "direct": "direct_admission",
                "capacity_fallback": "sidecar_capacity_fallback",
                "certificate_reject": "certificate_rejection",
            }
            require(expected in tag_outcome, f"trace[{line_no}]: invalid expected_insert_outcome")
            require(tag_outcome[expected] in tags,
                    f"trace[{line_no}]: expected insert outcome must have matching required_cases tag")
            live.add(stable_id)
            overlay.add(stable_id)
            causes_rebuild = row.get("post_op_rebuild", False)
            require(isinstance(causes_rebuild, bool), f"trace[{line_no}].post_op_rebuild: expected bool")
            if causes_rebuild:
                rebuild_count += 1
                base = set(live)
                overlay.clear()
                need_first_query = True
        elif operation == "delete":
            stable_id = require_identifier(row.get("stable_id"), f"trace[{line_no}].stable_id")
            require(stable_id in live, f"trace[{line_no}]: deletion of non-live object")
            if stable_id in base:
                require("base_delete_rebuild_barrier" in tags,
                        f"trace[{line_no}]: base delete requires barrier tag")
                live.remove(stable_id)
                rebuild_count += 1
                base = set(live)
                overlay.clear()
                need_first_query = True
            else:
                require("overlay_delete" in tags, f"trace[{line_no}]: overlay delete requires tag")
                live.remove(stable_id)
                overlay.remove(stable_id)
        elif operation == "query":
            query_id = require_identifier(row.get("query_id"), f"trace[{line_no}].query_id")
            require(query_id in catalog["queries"], f"trace[{line_no}]: unknown query")
            require(row.get("k", k) == k, f"trace[{line_no}].k mismatch")
            if need_first_query:
                require("first_query_after_rebuild" in tags,
                        f"trace[{line_no}]: missing first_query_after_rebuild tag")
                need_first_query = False
            if "nontrivial_receipt_witness" in tags:
                require_identifier(row.get("witness_id"), f"trace[{line_no}].witness_id")
        else:
            raise ProtocolError(f"trace[{line_no}].op: expected insert|delete|query")
        operations.append(dict(row))
    require(operations, "trace: no operations")
    require(all_cases == REQUIRED_CASES, "trace: required E1 cases not all exercised")
    require(not need_first_query, "trace: no first query after final rebuild")
    return {"header": dict(header), "operations": operations, "initial_base": base,
            "final_live": live, "rebuild_count": rebuild_count, "k": k,
            "trace_id": trace_id}


def load_and_validate_trace(trace_path: Path, catalog_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    catalog = load_catalog(catalog_path)
    rows = load_jsonl(trace_path)
    trace = validate_trace_rows(rows, catalog)
    trace["sha256"] = sha256_file(trace_path)
    return catalog, trace


def squared_l2(left: Sequence[int], right: Sequence[int]) -> int:
    require(len(left) == len(right), "distance: dimension mismatch")
    return sum((a - b) * (a - b) for a, b in zip(left, right))


def require_relative_path(value: Any, where: str) -> str:
    value = require_string(value, where)
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts and value != ".",
            f"{where}: path must be a nonempty relative non-traversing path")
    return value
