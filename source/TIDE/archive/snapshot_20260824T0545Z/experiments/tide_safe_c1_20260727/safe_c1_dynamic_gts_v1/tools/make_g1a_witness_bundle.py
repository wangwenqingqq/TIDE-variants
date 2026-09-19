#!/usr/bin/env python3
"""CPU-only generator for the first Safe-C1/GTS dynamic top-k visibility gate.

It derives immutable quantized vectors from the archived adversarial input but
creates a new, explicitly top-k-only witness trace.  Every insert/delete has a
follow-up query equal to the affected object's vector.  No GPU is used.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np

MAGIC = b"E1GTRC01"
VERSION = 1
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
ROOT = Path("/workspace/experiments/tide_safe_c1_20260727")
SOURCE = ROOT / "runs/e1gi_b_quantized_gts_bundles_v1_20260727/synthetic_l2__adversarial__seed_20260727"
OUT = ROOT / "safe_c1_dynamic_gts_v1/bundles/g1a_witness_pre_rebuild_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def exact_rows(pool: np.ndarray, query: np.ndarray, active: set[int], k: int) -> list[list[float | int]]:
    ids = np.array(sorted(active), dtype=np.int64)
    d = pool[ids].astype(np.int64) - query.astype(np.int64)
    distances = np.sqrt(np.sum(d * d, axis=1, dtype=np.int64).astype(np.float64))
    choice = np.lexsort((ids, distances))[:k]
    return [[int(ids[i]), float(distances[i])] for i in choice]


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing bundle: {OUT}")
    source_meta = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))
    raw = (SOURCE / "trace.e1gtrc").read_bytes()
    magic, version, dim, base_n, reservoir_n, pool_n, _query_n, k, radius, _events = HEADER.unpack_from(raw, 0)
    if (magic, version) != (MAGIC, VERSION):
        raise SystemExit("unexpected source trace header")
    pool = np.fromfile(SOURCE / "pool.i16", dtype="<i2").reshape(pool_n, dim)

    # These IDs already have a frozen-base Safe-C1 placement history in the old
    # reference runner.  This new trace does not trust that history: the G1
    # engine recertifies them with L=2 and records actual traversal receipts.
    a, b, c, d, e, f = 4097, 5529, 5585, 4096, 4105, 4112
    q_rows = [a, b, c, a, c, d, e, f]
    queries = np.asarray(pool[q_rows], dtype="<i2", order="C")
    events = [
        (1, a), (3, 0),
        (1, b), (3, 1),
        (1, c), (3, 2),
        (2, a), (3, 3),
        (2, c), (3, 4),
        (1, d), (3, 5),
        (1, e), (3, 6),
        (1, f), (3, 7),
    ]
    active = set(range(base_n))
    seen: set[int] = set()
    for code, arg in events:
        if code == 1:
            if not (base_n <= arg < base_n + reservoir_n <= pool_n) or arg in active or arg in seen:
                raise SystemExit(f"invalid insert {arg}")
            active.add(arg); seen.add(arg)
        elif code == 2:
            if arg not in active:
                raise SystemExit(f"invalid delete {arg}")
            active.remove(arg)
        elif code != 3:
            raise SystemExit("G1A is top-k-only")

    # Independent CPU oracle is only used to reject a witness tie before any
    # GPU activity.  G1's later validator recomputes all answers independently.
    active = set(range(base_n))
    witnesses: list[dict[str, object]] = []
    for oi, (code, arg) in enumerate(events):
        if code == 1:
            active.add(arg)
        elif code == 2:
            active.remove(arg)
        else:
            target = q_rows[arg]
            answer = exact_rows(pool, queries[arg], active, k)
            if target in {a, b, c, d, e, f}:
                # Inserts must be unique top-1; deletes must exclude their target.
                witnesses.append({"query_op_index": oi, "query_id": arg, "target_stable_id": target,
                                  "top1": answer[0][0], "target_in_topk": any(row[0] == target for row in answer)})
    # Positive inserted-object witnesses must be unique zero-distance rank-1.
    for entry in witnesses:
        oi, target, top1 = int(entry["query_op_index"]), int(entry["target_stable_id"]), int(entry["top1"])
        if oi in {1, 3, 5, 11, 13, 15} and top1 != target:
            raise SystemExit(f"non-unique insertion witness at op {oi}: target={target}, top1={top1}")
        if oi == 7 and bool(entry["target_in_topk"]):
            raise SystemExit("deleted direct witness still appears in CPU top-k")
        if oi == 9 and bool(entry["target_in_topk"]):
            raise SystemExit("deleted delta witness still appears in CPU top-k")

    OUT.mkdir(parents=True)
    (OUT / "pool.i16").write_bytes(pool.astype("<i2", copy=False).tobytes(order="C"))
    (OUT / "queries.i16").write_bytes(queries.tobytes(order="C"))
    trace = HEADER.pack(MAGIC, VERSION, dim, base_n, reservoir_n, pool_n, len(queries), k,
                        np.float32(radius), len(events))
    trace += b"".join(EVENT.pack(i, code, arg) for i, (code, arg) in enumerate(events))
    (OUT / "trace.e1gtrc").write_bytes(trace)
    for name in ("initial_base_stable_ids.i32", "stable_id_to_pool_row.i32"):
        shutil.copyfile(SOURCE / name, OUT / name)
    source_hashes = {name: sha256(SOURCE / name) for name in ("pool.i16", "trace.e1gtrc", "metadata.json")}
    metadata = {
        "schema": "e1gi-b-quantized-gts-integration-bundle-v1",
        "scope": "CPU-prepared G1A Safe-C1/GTS top-k witness input; no GPU used; no range/rebuild/base-delete claims",
        "gpu_used": False,
        "header": {"magic": MAGIC.decode(), "version": VERSION, "dim": dim, "base_n": base_n,
                   "reservoir_n": reservoir_n, "pool_n": pool_n, "query_n": len(queries), "k": k,
                   "radius": float(radius), "event_count": len(events)},
        "quantization": {"scale": 100.0, "formula": "inherited immutable int16 source payload",
                           "coordinate_type": "little-endian signed int16", "overflow_policy": "source already rejected clipping"},
        "metric_contract": {"metric": "L2", "distance_arithmetic": "int64 squared-difference accumulation then float64 sqrt in independent oracle",
                            "topk_tie_break": "(distance, stable_id)", "range_in_scope": False},
        "stable_id_contract": {"initial_base_ids": "[0, base_n)", "reservoir_ids": "[base_n, base_n + reservoir_n)",
                               "delete_semantics": "stable ID, never logical rank", "query_semantics": "external witness query ID"},
        "g1_scope": {"leaf_capacity": 2, "rebuild": False, "base_delete": False, "range": False,
                    "required_engine_behavior": "real GTS traversal receipt; scan sidecars only in returned visited leaves; scan delta globally"},
        "source_payload": {"path": str(SOURCE), "sha256": source_hashes},
    }
    write_json(OUT / "metadata.json", metadata)
    write_json(OUT / "witness_contract.json", {"schema": "safe-c1-g1a-witness-contract-v1", "events": [
        {"op_index": 1, "mode": "inserted_top1", "stable_id": a, "expected_placement": "runtime"},
        {"op_index": 3, "mode": "inserted_top1", "stable_id": b, "expected_placement": "runtime"},
        {"op_index": 5, "mode": "inserted_top1", "stable_id": c, "expected_placement": "runtime"},
        {"op_index": 7, "mode": "deleted_absent", "stable_id": a},
        {"op_index": 9, "mode": "deleted_absent", "stable_id": c},
        {"op_index": 11, "mode": "inserted_top1", "stable_id": d, "expected_placement": "runtime"},
        {"op_index": 13, "mode": "inserted_top1", "stable_id": e, "expected_placement": "runtime"},
        {"op_index": 15, "mode": "inserted_top1", "stable_id": f, "expected_placement": "runtime"},
    ], "cpu_preflight": witnesses})
    manifest = {"schema": "safe-c1-g1a-bundle-manifest-v1", "status": "prepared_cpu_only",
                "files_sha256": {x.name: sha256(x) for x in sorted(OUT.iterdir()) if x.is_file()}}
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"status": "PASS_CPU_ONLY_G1A_BUNDLE", "out": str(OUT), "events": len(events), "queries": len(queries), "gpu_used": False}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
