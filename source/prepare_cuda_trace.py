#!/usr/bin/env python3
"""Prepare versioned compact E1-G0 CUDA Safe-C1 trace bundles.

This tool is an *input-preparation* layer for an independent CUDA reference
executor (E1-G0).  It neither invokes CUDA nor modifies a source bundle, and
it must never be cited as a GTS integration result.

Binary file layout (little endian, version 1):
  header: <8s I I I I I I I f Q
          magic, version, dim, base_n, reservoir_n, pool_n, query_n, k,
          radius, event_count
  event : <I B 3x i
          op_index, op_code (1 insert/2 delete/3 knn/4 range), argument
          (stable_id for update; query_id for query)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

MAGIC = b"E1GTRC01"
VERSION = 1
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
OP_TO_CODE = {"insert": 1, "delete": 2, "knn": 3, "range": 4}
CODE_TO_OP = {value: key for key, value in OP_TO_CODE.items()}


@dataclass(frozen=True)
class SourceBundle:
    path: Path
    kind: str  # raw_safe_c1_run | stable_contract_bundle
    family: str
    scenario: str
    seed: int | None
    pool: np.ndarray
    queries: np.ndarray
    trace: list[dict[str, Any]]
    dim: int
    base_n: int
    reservoir_n: int
    radius: float
    k: int
    source_hashes: dict[str, str]
    source_metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{lineno}: expected object")
            out.append(item)
    return out


def raw_family(path: Path) -> str:
    parts = set(path.parts)
    if "final_safe_c1_sift_slice_20260727" in parts:
        return "sift128_slice"
    if "final_safe_c1_oracle_v2_20260727" in parts:
        return "synthetic_l2"
    return "unknown"


def normalized_raw_trace(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in rows:
        op = str(event["op"])
        rec: dict[str, Any] = {"op_index": int(event["op_index"]), "op": op}
        if op in {"insert", "delete"}:
            rec["argument"] = int(event["id"])
        elif op in {"knn", "range"}:
            rec["argument"] = int(event["query_id"])
        else:
            raise ValueError(f"unsupported trace op {op!r}")
        if "label" in event:
            rec["label"] = str(event["label"])
        out.append(rec)
    return out


def normalized_contract_trace(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in rows:
        op = str(event["op"])
        rec: dict[str, Any] = {"op_index": int(event["op_index"]), "op": op}
        if op in {"insert", "delete"}:
            rec["argument"] = int(event["stable_id"])
        elif op in {"knn", "range"}:
            rec["argument"] = int(event["query_id"])
        else:
            raise ValueError(f"unsupported stable-trace op {op!r}")
        if "label" in event:
            rec["label"] = str(event["label"])
        out.append(rec)
    return out


def load_raw_run(path: Path, default_radius: float, default_k: int) -> SourceBundle:
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data = summary["data"]
    pool = np.asarray(np.load(path / "pool.npy"), dtype=np.float32, order="C")
    queries = np.asarray(np.load(path / "queries.npy"), dtype=np.float32, order="C")
    qcontract = summary.get("query_contract", {})
    # Legacy synthetic suite relied on command defaults; SIFT persists these values explicitly.
    radius = float(qcontract.get("range_radius", default_radius))
    k = int(qcontract.get("k", default_k))
    trace = normalized_raw_trace(jsonl(path / "trace.jsonl"))
    source_files = ["summary.json", "trace.jsonl", "pool.npy", "queries.npy"]
    return SourceBundle(
        path=path,
        kind="raw_safe_c1_run",
        family=raw_family(path),
        scenario=str(summary.get("scenario", "unknown")),
        seed=int(summary["seed"]) if "seed" in summary else None,
        pool=pool,
        queries=queries,
        trace=trace,
        dim=int(data["dimension"]),
        base_n=int(data["base_n"]),
        reservoir_n=int(data["reservoir_n"]),
        radius=radius,
        k=k,
        source_hashes={name: sha256_file(path / name) for name in source_files},
        source_metadata={
            "summary_schema": summary.get("schema"),
            "oracle_pass": summary.get("oracle_pass"),
            "same_final_active_set": summary.get("same_final_active_set"),
            "query_parameter_provenance": "summary.query_contract" if qcontract else "script_default_for_original_synthetic_suite",
        },
    )


def load_contract_bundle(path: Path, default_radius: float, default_k: int) -> SourceBundle:
    del default_radius, default_k
    contract_path = path / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    dim = int(contract["dimension"])
    pool = np.fromfile(path / "pool.f32", dtype="<f4").reshape(int(contract["pool_n"]), dim)
    queries = np.fromfile(path / "queries.f32", dtype="<f4").reshape(int(contract["query_n"]), dim)
    trace = normalized_contract_trace(jsonl(path / "stable_trace.jsonl"))
    source_files = ["contract.json", "stable_trace.jsonl", "pool.f32", "queries.f32"]
    return SourceBundle(
        path=path,
        kind="stable_contract_bundle",
        family="stable_contract",
        scenario="contract_trace",
        seed=None,
        pool=np.asarray(pool, dtype=np.float32, order="C"),
        queries=np.asarray(queries, dtype=np.float32, order="C"),
        trace=trace,
        dim=dim,
        base_n=int(contract["base_n"]),
        reservoir_n=int(contract["reservoir_n"]),
        radius=float(contract["radius"]),
        k=int(contract["k"]),
        source_hashes={name: sha256_file(path / name) for name in source_files},
        source_metadata={
            "contract_schema": contract.get("schema"),
            "purpose": contract.get("purpose"),
            "query_parameter_provenance": "contract.json",
        },
    )


def load_source(path: Path, default_radius: float, default_k: int) -> SourceBundle:
    path = path.resolve()
    if (path / "contract.json").is_file():
        return load_contract_bundle(path, default_radius, default_k)
    needed = [path / x for x in ("summary.json", "trace.jsonl", "pool.npy", "queries.npy")]
    if all(x.is_file() for x in needed):
        return load_raw_run(path, default_radius, default_k)
    raise ValueError(f"not a recognized stable run/contract bundle: {path}")


def validate(bundle: SourceBundle) -> dict[str, int]:
    if bundle.pool.ndim != 2 or bundle.pool.shape != (bundle.base_n + bundle.reservoir_n, bundle.dim):
        raise ValueError(f"{bundle.path}: pool shape {bundle.pool.shape} violates base/reservoir/dim metadata")
    if bundle.queries.ndim != 2 or bundle.queries.shape[1] != bundle.dim:
        raise ValueError(f"{bundle.path}: query shape {bundle.queries.shape} violates dim={bundle.dim}")
    if bundle.k <= 0 or bundle.radius < 0:
        raise ValueError(f"{bundle.path}: invalid k/radius")
    active = set(range(bundle.base_n))
    ever_inserted: set[int] = set()
    counts = {op: 0 for op in OP_TO_CODE}
    for expected_index, rec in enumerate(bundle.trace):
        oi, op, arg = int(rec["op_index"]), str(rec["op"]), int(rec["argument"])
        if oi != expected_index:
            raise ValueError(f"{bundle.path}: non-contiguous op_index at trace row {expected_index}: {oi}")
        if op not in OP_TO_CODE:
            raise ValueError(f"{bundle.path}: unsupported op {op}")
        if op == "insert":
            if not (bundle.base_n <= arg < bundle.base_n + bundle.reservoir_n):
                raise ValueError(f"{bundle.path}: insert {arg} is outside disjoint reservoir")
            if arg in active or arg in ever_inserted:
                raise ValueError(f"{bundle.path}: duplicate/reinserted stable ID {arg}")
            active.add(arg)
            ever_inserted.add(arg)
        elif op == "delete":
            if arg not in active:
                raise ValueError(f"{bundle.path}: delete {arg} is not stable/live at op {oi}")
            active.remove(arg)
        else:
            if not (0 <= arg < len(bundle.queries)):
                raise ValueError(f"{bundle.path}: query ID {arg} out of range at op {oi}")
        counts[op] += 1
    return counts


def output_name(bundle: SourceBundle) -> str:
    seed = f"seed_{bundle.seed}" if bundle.seed is not None else "no_seed"
    return "__".join([bundle.family, bundle.scenario, seed]).replace("/", "_").replace(" ", "_")


def write_trace(bundle: SourceBundle, out: Path, *, source_script_sha256: str) -> dict[str, Any]:
    counts = validate(bundle)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite trace directory: {out}")
    out.mkdir(parents=True)
    trace_path = out / "trace.e1gtrc"
    with trace_path.open("wb") as fh:
        fh.write(HEADER.pack(
            MAGIC, VERSION, bundle.dim, bundle.base_n, bundle.reservoir_n,
            int(bundle.pool.shape[0]), int(bundle.queries.shape[0]), bundle.k,
            np.float32(bundle.radius), len(bundle.trace),
        ))
        for rec in bundle.trace:
            fh.write(EVENT.pack(int(rec["op_index"]), OP_TO_CODE[str(rec["op"])], int(rec["argument"])))
    # Self-contained immutable input data for the independent CUDA reference executor.
    (out / "pool.f32").write_bytes(np.asarray(bundle.pool, dtype="<f4", order="C").tobytes(order="C"))
    (out / "queries.f32").write_bytes(np.asarray(bundle.queries, dtype="<f4", order="C").tobytes(order="C"))
    np.arange(bundle.base_n, dtype="<i4").tofile(out / "initial_base_stable_ids.i32")
    np.arange(bundle.pool.shape[0], dtype="<i4").tofile(out / "stable_id_to_pool_row.i32")

    check = verify_binary(trace_path, bundle)
    metadata: dict[str, Any] = {
        "schema": "e1g-cuda-trace-metadata-v1",
        "scope": "E1-G0 independent CUDA Safe-C1 reference executor input; NOT GTS integration evidence",
        "gpu_used": False,
        "binary_format": {
            "endianness": "little",
            "header_struct": "<8s I I I I I I I f Q",
            "header_bytes": HEADER.size,
            "magic_ascii": MAGIC.decode("ascii"),
            "version": VERSION,
            "event_struct": "<I B 3x i",
            "event_bytes": EVENT.size,
            "op_codes": {str(code): op for op, code in OP_TO_CODE.items()},
            "argument": "stable_id for insert/delete; query_id for knn/range",
        },
        "header": {
            "magic": MAGIC.decode("ascii"), "version": VERSION, "dim": bundle.dim,
            "base_n": bundle.base_n, "reservoir_n": bundle.reservoir_n,
            "pool_n": int(bundle.pool.shape[0]), "query_n": int(bundle.queries.shape[0]),
            "k": bundle.k, "radius": float(np.float32(bundle.radius)), "event_count": len(bundle.trace),
        },
        "trace_counts": counts,
        "source": {
            "path": str(bundle.path), "kind": bundle.kind, "family": bundle.family,
            "scenario": bundle.scenario, "seed": bundle.seed, "files_sha256": bundle.source_hashes,
            "metadata": bundle.source_metadata,
        },
        "stable_id_contract": {
            "initial_base_ids": "[0, base_n)",
            "reservoir_ids": "[base_n, base_n + reservoir_n)",
            "mapping": "identity stable_id_to_pool_row.i32",
            "delete_semantics": "stable ID, never logical/live rank",
            "query_semantics": "external query_id into queries.f32",
        },
        "payload_files": {
            "trace": "trace.e1gtrc", "pool": "pool.f32", "queries": "queries.f32",
            "initial_base_ids": "initial_base_stable_ids.i32", "stable_id_mapping": "stable_id_to_pool_row.i32",
        },
        "prepare_cuda_trace_py_sha256": source_script_sha256,
        "binary_self_check": check,
    }
    metadata_path = out / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    all_files = sorted(p for p in out.iterdir() if p.is_file())
    manifest = {
        "schema": "e1g-cuda-trace-manifest-v1",
        "status": "prepared_for_E1_G0_cuda_reference_not_GTS_result",
        "gpu_used": False,
        "files_sha256": {p.name: sha256_file(p) for p in all_files},
        "metadata_sha256": sha256_file(metadata_path),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"out": str(out), "metadata": metadata, "manifest": manifest}


def verify_binary(path: Path, bundle: SourceBundle) -> dict[str, Any]:
    blob = path.read_bytes()
    expected_size = HEADER.size + EVENT.size * len(bundle.trace)
    if len(blob) != expected_size:
        raise ValueError(f"binary size mismatch: {len(blob)} != {expected_size}")
    raw_header = HEADER.unpack_from(blob, 0)
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = raw_header
    expected_header = (MAGIC, VERSION, bundle.dim, bundle.base_n, bundle.reservoir_n, bundle.pool.shape[0], bundle.queries.shape[0], bundle.k, len(bundle.trace))
    observed_header = (magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, event_count)
    if observed_header != expected_header:
        raise ValueError(f"binary header mismatch: observed={observed_header}, expected={expected_header}")
    if not np.isclose(radius, np.float32(bundle.radius), rtol=0.0, atol=0.0):
        raise ValueError(f"binary radius mismatch: {radius} != {np.float32(bundle.radius)}")
    for i, rec in enumerate(bundle.trace):
        oi, code, arg = EVENT.unpack_from(blob, HEADER.size + i * EVENT.size)
        expected = (int(rec["op_index"]), OP_TO_CODE[str(rec["op"])], int(rec["argument"]))
        if (oi, code, arg) != expected:
            raise ValueError(f"event {i} mismatch: got={(oi, code, arg)} expected={expected}")
    return {"pass": True, "header_bytes": HEADER.size, "event_bytes": EVENT.size, "byte_length": len(blob), "events_verified": len(bundle.trace)}


def discover_current(experiment_root: Path) -> list[Path]:
    roots = [
        experiment_root / "runs" / "final_safe_c1_oracle_v2_20260727" / "output",
        experiment_root / "runs" / "final_safe_c1_sift_slice_20260727" / "output",
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for summary in sorted(root.rglob("summary.json")):
            # `suite_summary.json` is intentionally not selected by this pattern.
            if summary.name != "summary.json":
                continue
            run_dir = summary.parent
            if all((run_dir / name).is_file() for name in ("pool.npy", "queries.npy", "trace.jsonl")):
                candidates.append(run_dir)
    return candidates


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", type=Path, help="raw stable run or stable contract bundle; repeatable")
    p.add_argument("--discover-current", action="store_true", help="discover final synthetic-v2 (6) and SIFT slice (3) runs")
    p.add_argument("--experiment-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--default-radius", type=float, default=3.0, help="only for legacy synthetic raw summaries without query_contract")
    p.add_argument("--default-k", type=int, default=10, help="only for raw summaries without query_contract")
    args = p.parse_args()
    selected: list[Path] = list(args.input or [])
    if args.discover_current:
        selected.extend(discover_current(args.experiment_root.resolve()))
    # deterministic de-duplication
    selected = list(dict.fromkeys(x.resolve() for x in selected))
    if not selected:
        raise SystemExit("provide --input or --discover-current")
    out_root = args.out.resolve()
    if out_root.exists():
        raise SystemExit(f"refusing to overwrite output root: {out_root}")
    out_root.mkdir(parents=True)
    script_sha = sha256_file(Path(__file__).resolve())
    results: list[dict[str, Any]] = []
    try:
        for source in selected:
            bundle = load_source(source, args.default_radius, args.default_k)
            result = write_trace(bundle, out_root / output_name(bundle), source_script_sha256=script_sha)
            results.append({
                "source": str(source), "output": result["out"], "family": bundle.family,
                "scenario": bundle.scenario, "seed": bundle.seed,
                "events": len(bundle.trace), "self_check": result["metadata"]["binary_self_check"],
            })
    except Exception:
        # A partial result is never a valid batch. Preserve it for forensic inspection rather
        # than deleting it automatically, but make the failure explicit.
        (out_root / "FAILED.txt").write_text("batch preparation failed; inspect existing partial outputs and rerun into a new output root\n", encoding="utf-8")
        raise
    batch = {
        "schema": "e1g-cuda-trace-batch-v1",
        "scope": "prepared inputs for independent E1-G0 CUDA Safe-C1 reference; NOT GTS integration",
        "gpu_used": False,
        "selected_count": len(selected),
        "prepared_count": len(results),
        "expected_current_final_count": 9 if args.discover_current else None,
        "results": results,
        "prepare_cuda_trace_py_sha256": script_sha,
    }
    (out_root / "batch_metadata.json").write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_root), "prepared_count": len(results), "gpu_used": False, "scope": "E1-G0-input-preparation-not-GTS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
