#!/usr/bin/env python3
"""Generate the standalone CPU bootstrap oracle for the G3 4K pilot.

This program uses only Python standard-library CPU code. It does not import
CUDA, start an engine, inspect a GPU, or read engine state. The output is not
an engine/fixture trace record and must remain outside the fixture directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

SCHEMA = "safe-c1-g3-bootstrap-oracle-v1"
BUNDLE_REL = Path("inputs/g3_sift4096_branchstress_l2_d3")
OUT_REL = Path("preflight/bootstrap_oracles/g3_sift4096_branchstress_l2_d3/bootstrap_oracle.json")
K = 10
KNN_QUERY_ID = 1
RANGE_QUERY_ID = 2
RANGE_RADIUS_SQ = 427_400_000
REQUIRED_FIXTURE_FILES = (
    "initial_base_stable_ids.i32",
    "metadata.json",
    "oracle_expected.jsonl",
    "pool.i16",
    "queries.i16",
    "selection_receipt.json",
    "stable_id_to_pool_row.i32",
    "trace.jsonl",
)
INT64_MAX = (1 << 63) - 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_set_sha256(ids: list[int]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in sorted(set(ids))).encode("ascii")).hexdigest()


def read_ints(path: Path, code: str, expected_count: int) -> tuple[int, ...]:
    raw = path.read_bytes()
    width = struct.calcsize("<" + code)
    if len(raw) != width * expected_count:
        raise ValueError(f"{path.name}: bytes={len(raw)} expected={width * expected_count}")
    return tuple(value[0] for value in struct.iter_unpack("<" + code, raw))


def checked_fixture(bundle: Path) -> tuple[dict[str, Any], dict[str, str], str]:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("files_sha256")
    if not isinstance(declared, dict) or set(declared) != set(REQUIRED_FIXTURE_FILES):
        raise ValueError("fixture manifest does not bind exactly the expected fixture files")
    measured: dict[str, str] = {}
    for name in REQUIRED_FIXTURE_FILES:
        path = bundle / name
        if not path.is_file():
            raise ValueError(f"missing fixture input: {name}")
        measured[name] = sha256_file(path)
        if measured[name] != declared[name]:
            raise ValueError(f"fixture hash mismatch: {name}")
    if manifest.get("status") != "CPU_PREPARED_NOT_NATIVE_EXECUTED" or manifest.get("gpu_used") is not False:
        raise ValueError("fixture does not declare CPU_PREPARED_NOT_NATIVE_EXECUTED")
    return manifest, measured, sha256_file(manifest_path)


def load_inputs(bundle: Path) -> tuple[dict[str, Any], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    header = metadata.get("header", {})
    dim = header.get("dim")
    pool_n = header.get("pool_n")
    query_n = header.get("query_n")
    base_n = header.get("base_n")
    if not all(isinstance(value, int) and value > 0 for value in (dim, pool_n, query_n, base_n)):
        raise ValueError("invalid fixture dimensions")
    if header.get("k") != K or base_n != 4096:
        raise ValueError("fixture is not the fixed 4K/K=10 bootstrap target")
    metric = metadata.get("metric_contract", {})
    if metric.get("coordinate_type") != "int16" or metric.get("oracle_distance") != "int64 squared L2":
        raise ValueError("fixture metric contract is not int16/int64-L2")
    if metric.get("knn_order") != "(distance_sq, stable_id)" or metric.get("range_contract") != "inclusive distance_sq <= radius_sq":
        raise ValueError("fixture tie/range contract mismatch")
    pool = read_ints(bundle / "pool.i16", "h", pool_n * dim)
    queries = read_ints(bundle / "queries.i16", "h", query_n * dim)
    stable_to_pool_row = read_ints(bundle / "stable_id_to_pool_row.i32", "i", pool_n)
    base_ids = read_ints(bundle / "initial_base_stable_ids.i32", "i", base_n)
    if len(set(base_ids)) != base_n or any(sid < 0 or sid >= pool_n for sid in base_ids):
        raise ValueError("initial base is not a valid unique stable-ID set")
    if sorted(stable_to_pool_row) != list(range(pool_n)):
        raise ValueError("stable_id_to_pool_row is not a bijection")
    return metadata, pool, queries, stable_to_pool_row, base_ids


def l2sq_int64(pool: tuple[int, ...], queries: tuple[int, ...], stable_to_pool_row: tuple[int, ...], dim: int, query_id: int, stable_id: int) -> int:
    query_offset = query_id * dim
    pool_offset = stable_to_pool_row[stable_id] * dim
    total = 0
    for column in range(dim):
        delta = int(pool[pool_offset + column]) - int(queries[query_offset + column])
        total += delta * delta
        if total > INT64_MAX:
            raise OverflowError("squared L2 exceeds signed int64")
    return total


def all_distances(pool: tuple[int, ...], queries: tuple[int, ...], stable_to_pool_row: tuple[int, ...], dim: int, query_id: int, active_ids: tuple[int, ...]) -> list[tuple[int, int]]:
    return sorted((l2sq_int64(pool, queries, stable_to_pool_row, dim, query_id, stable_id), stable_id) for stable_id in active_ids)


def result_rows(pairs: list[tuple[int, int]]) -> list[list[int]]:
    return [[stable_id, distance_sq] for distance_sq, stable_id in pairs]


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def build(root: Path, bundle: Path, output: Path) -> dict[str, Any]:
    manifest, file_hashes, manifest_sha256 = checked_fixture(bundle)
    metadata, pool, queries, stable_to_pool_row, initial_base = load_inputs(bundle)
    header = metadata["header"]
    dim = int(header["dim"])
    pool_n = int(header["pool_n"])
    query_n = int(header["query_n"])
    if max(KNN_QUERY_ID, RANGE_QUERY_ID) >= query_n:
        raise ValueError("fixed bootstrap query id outside fixture query array")
    active_ids = tuple(sorted(int(stable_id) for stable_id in initial_base))
    if len(active_ids) != 4096:
        raise ValueError("bootstrap active set must contain exactly 4096 IDs")

    knn_pairs = all_distances(pool, queries, stable_to_pool_row, dim, KNN_QUERY_ID, active_ids)
    knn_result = knn_pairs[:K]
    if len(knn_result) != K:
        raise ValueError("initial active set has fewer than K rows")
    kth_distance = knn_result[-1][0]
    next_distance = knn_pairs[K][0] if len(knn_pairs) > K else None
    knn_tie_count_at_kth = sum(distance_sq == kth_distance for distance_sq, _ in knn_pairs)
    knn_boundary_tie = next_distance == kth_distance if next_distance is not None else False

    range_pairs_all = all_distances(pool, queries, stable_to_pool_row, dim, RANGE_QUERY_ID, active_ids)
    range_rows = [(distance_sq, stable_id) for distance_sq, stable_id in range_pairs_all if distance_sq <= RANGE_RADIUS_SQ]
    range_boundary_count = sum(distance_sq == RANGE_RADIUS_SQ for distance_sq, _ in range_pairs_all)
    if not range_rows or range_boundary_count != 1:
        raise ValueError("fixed range radius is not a one-row inclusive boundary contract")
    if range_rows[0][0] != RANGE_RADIUS_SQ:
        raise ValueError("fixed range radius is not exactly the nearest active distance")

    generator = Path(__file__).resolve()
    validator = generator.with_name("validate_g3_bootstrap_oracle.py")
    output_relative = relative(output, root)
    bundle_relative = relative(bundle, root)
    try:
        output.resolve().relative_to(bundle.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("bootstrap oracle must not be written inside the fixture bundle")

    fixture_hashes = {"manifest.json": manifest_sha256, **file_hashes}
    query_bytes = (bundle / "queries.i16").read_bytes()
    active_hash = stable_set_sha256(list(active_ids))
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "record": "bootstrap_oracle",
        "status": "CPU_ONLY_NOT_RUN",
        "scope": "Independent CPU bootstrap oracle for the initial 4096-ID base only; no engine state, CUDA/native execution, GPU inspection, or performance result.",
        "execution": {
            "cpu_only": True,
            "gpu_used": False,
            "native_engine_executed": False,
            "binary_launch_count": 0,
            "cuda_imported": False,
            "not_engine_trace": True,
        },
        "trace_separation": {
            "fixture_trace": f"{bundle_relative}/trace.jsonl",
            "fixture_oracle": f"{bundle_relative}/oracle_expected.jsonl",
            "future_engine_trace": "engine.jsonl",
            "bootstrap_output": output_relative,
            "must_not_append_to_fixture_or_engine_trace": True,
            "bootstrap_has_op_index": False,
            "reason": "The fixture engine validator consumes exactly trace op_index 0..20 and rejects extra records.",
        },
        "fixture": {
            "bundle_relative_path": bundle_relative,
            "manifest_schema": manifest.get("schema"),
            "manifest_status": manifest.get("status"),
            "all_fixture_inputs_sha256": fixture_hashes,
            "manifest_declared_files_sha256": manifest["files_sha256"],
            "all_manifest_hashes_verified": True,
            "header": {
                "base_n": int(header["base_n"]),
                "pool_n": pool_n,
                "query_n": query_n,
                "dim": dim,
                "k": K,
            },
        },
        "active_set": {
            "source": "initial_base_stable_ids.i32",
            "engine_state_read": False,
            "count": len(active_ids),
            "stable_ids_sha256": active_hash,
            "source_file_sha256": file_hashes["initial_base_stable_ids.i32"],
            "membership_rule": "exactly the unique stable IDs decoded from initial_base_stable_ids.i32",
        },
        "metric_contract": {
            "coordinate_encoding": "little-endian int16",
            "stable_id_mapping": "stable_id_to_pool_row.i32 maps stable ID to physical pool row",
            "distance": "signed int64 squared L2 accumulated exactly over dim coordinates",
            "knn_k": K,
            "knn_tie_order": "ascending (distance_sq, stable_id)",
            "range_inclusion": "distance_sq <= radius_sq",
            "range_result_order": "ascending (distance_sq, stable_id)",
        },
        "queries": [
            {
                "bootstrap_id": "initial_base_knn",
                "kind": "knn",
                "query_id": KNN_QUERY_ID,
                "query_vector_sha256": hashlib.sha256(query_bytes[KNN_QUERY_ID * dim * 2:(KNN_QUERY_ID + 1) * dim * 2]).hexdigest(),
                "k": K,
                "radius_sq": None,
                "active_ids_sha256": active_hash,
                "results": result_rows(knn_result),
                "kth_distance_sq": kth_distance,
                "next_distance_sq": next_distance,
                "kth_distance_tie_count": knn_tie_count_at_kth,
                "knn_boundary_tie": knn_boundary_tie,
            },
            {
                "bootstrap_id": "initial_base_range",
                "kind": "range",
                "query_id": RANGE_QUERY_ID,
                "query_vector_sha256": hashlib.sha256(query_bytes[RANGE_QUERY_ID * dim * 2:(RANGE_QUERY_ID + 1) * dim * 2]).hexdigest(),
                "k": None,
                "radius_sq": RANGE_RADIUS_SQ,
                "radius_selection": "fixed radius equal to the unique nearest active-base squared distance",
                "active_ids_sha256": active_hash,
                "results": result_rows(range_rows),
                "distance_equal_to_radius_count": range_boundary_count,
                "result_count": len(range_rows),
            },
        ],
        "tool_sha256": {
            "generator": sha256_file(generator),
            "validator": sha256_file(validator) if validator.is_file() else None,
        },
    }
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bundle-rel", type=Path, default=BUNDLE_REL)
    parser.add_argument("--out-rel", type=Path, default=OUT_REL)
    args = parser.parse_args()
    root = args.root.resolve()
    bundle = (root / args.bundle_rel).resolve()
    output = (root / args.out_rel).resolve()
    artifact = build(root, bundle, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "gpu_used": False, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
