#!/usr/bin/env python3
"""Independent exact validator for E1-GI-B quantized runner exports.

This verifier reads only the prepared quantized payload and engine JSONL.  It
replays stable-ID updates from trace.e1gtrc and recomputes exact L2 answers from
pool.i16 / queries.i16 using int64 accumulation.  It intentionally does not
read or trust `quantized_oracle_expected.jsonl`, engine summaries, or engine
final-state claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"E1GTRC01"
VERSION = 1
HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
OP_BY_CODE = {1: "insert", 2: "delete", 3: "knn", 4: "range"}


def stable_set_sha256(active: set[int]) -> str:
    return hashlib.sha256("".join(f"{sid}\n" for sid in sorted(active)).encode("ascii")).hexdigest()


def make_error(kind: str, **values: Any) -> dict[str, Any]:
    return {"type": kind, **values}


def int_field(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be integer")
    return int(value)


def load_engine(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, errors = [], []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(make_error("invalid_json", line=line_no, reason=str(exc)))
                continue
            if not isinstance(obj, dict):
                errors.append(make_error("invalid_engine_record", line=line_no, reason="must be JSON object"))
                continue
            # Summary/meta have no evidentiary value and cannot fulfill an event.
            if obj.get("record") in {"summary", "meta"}:
                continue
            rows.append(obj)
    return rows, errors


def read_payload(bundle: Path) -> tuple[dict[str, Any], list[tuple[int, str, int]], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    errors: list[dict[str, Any]] = []
    if metadata.get("schema") != "e1gi-b-quantized-gts-integration-bundle-v1":
        errors.append(make_error("unexpected_metadata_schema", observed=metadata.get("schema")))
    q = metadata.get("quantization", {})
    if q.get("scale") != 100.0 or q.get("coordinate_type") != "little-endian signed int16":
        errors.append(make_error("unexpected_quantization_contract", observed=q))
    blob = (bundle / "trace.e1gtrc").read_bytes()
    if len(blob) < HEADER.size:
        raise ValueError("truncated trace header")
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = HEADER.unpack_from(blob, 0)
    header = {"magic": magic.decode("ascii", errors="replace"), "version": version, "dim": dim, "base_n": base_n, "reservoir_n": reservoir_n, "pool_n": pool_n, "query_n": query_n, "k": k, "radius": float(radius), "event_count": event_count, "header_bytes": HEADER.size, "event_bytes": EVENT.size}
    if magic != MAGIC or version != VERSION:
        errors.append(make_error("unsupported_trace_header", observed=header))
    for name in ("dim", "base_n", "reservoir_n", "pool_n", "query_n", "k", "event_count"):
        if name in metadata.get("header", {}) and int(metadata["header"][name]) != int(header[name]):
            errors.append(make_error("metadata_header_mismatch", field=name, metadata=metadata["header"][name], binary=header[name]))
    if "radius" in metadata.get("header", {}) and not np.isclose(np.float32(metadata["header"]["radius"]), np.float32(radius), rtol=0.0, atol=0.0):
        errors.append(make_error("metadata_header_mismatch", field="radius", metadata=metadata["header"]["radius"], binary=float(radius)))
    expected_bytes = HEADER.size + EVENT.size * event_count
    if len(blob) != expected_bytes:
        errors.append(make_error("trace_size_mismatch", actual_bytes=len(blob), expected_bytes=expected_bytes))
    events: list[tuple[int, str, int]] = []
    if len(blob) >= expected_bytes:
        for pos in range(event_count):
            op_index, code, argument = EVENT.unpack_from(blob, HEADER.size + pos * EVENT.size)
            op = OP_BY_CODE.get(code)
            if op is None:
                errors.append(make_error("unknown_opcode", event_position=pos, op_index=op_index, code=code))
                op = f"unknown_{code}"
            if op_index != pos:
                errors.append(make_error("noncontiguous_op_index", event_position=pos, op_index=op_index))
            events.append((int(op_index), op, int(argument)))
    raw_pool = np.fromfile(bundle / "pool.i16", dtype="<i2")
    raw_queries = np.fromfile(bundle / "queries.i16", dtype="<i2")
    if len(raw_pool) != pool_n * dim:
        errors.append(make_error("pool_size_mismatch", values=len(raw_pool), expected=pool_n * dim)); pool = np.empty((0, dim), dtype=np.int16)
    else: pool = raw_pool.reshape(pool_n, dim)
    if len(raw_queries) != query_n * dim:
        errors.append(make_error("queries_size_mismatch", values=len(raw_queries), expected=query_n * dim)); queries = np.empty((0, dim), dtype=np.int16)
    else: queries = raw_queries.reshape(query_n, dim)
    return header, events, pool, queries, errors


def exact(pool: np.ndarray, query: np.ndarray, active: set[int], kind: str, radius: float, k: int) -> list[tuple[int, float]]:
    ids = np.fromiter(sorted(active), dtype=np.int64)
    delta = pool[ids].astype(np.int64, copy=False) - query.astype(np.int64, copy=False)
    distance = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.int64).astype(np.float64))
    if kind == "knn":
        chosen = np.lexsort((ids, distance))[: min(k, len(ids))]
    elif kind == "range":
        keep = np.flatnonzero(distance <= radius + 1e-12)
        chosen = keep[np.lexsort((ids[keep], distance[keep]))] if len(keep) else np.empty(0, dtype=np.int64)
    else:
        raise ValueError(f"unknown kind {kind}")
    return [(int(ids[i]), float(distance[i])) for i in chosen]


def parse_results(row: dict[str, Any]) -> list[tuple[int, float]]:
    raw = row.get("results")
    if not isinstance(raw, list): raise ValueError("results must be a list")
    answer: list[tuple[int, float]] = []
    for i, item in enumerate(raw):
        if isinstance(item, dict): sid, distance = item.get("stable_id"), item.get("distance")
        elif isinstance(item, list) and len(item) == 2: sid, distance = item
        else: raise ValueError(f"results[{i}] must be [stable_id,distance] or object")
        stable_id = int_field(sid, f"results[{i}].stable_id")
        d = float(distance)
        if not np.isfinite(d): raise ValueError(f"results[{i}].distance nonfinite")
        answer.append((stable_id, d))
    if len({sid for sid, _ in answer}) != len(answer): raise ValueError("duplicate stable ID")
    return answer


def verify(bundle: Path, engine: Path, atol: float, rtol: float) -> dict[str, Any]:
    header, events, pool, queries, errors = read_payload(bundle)
    records, parse_errors = load_engine(engine)
    errors.extend(parse_errors)
    indexed: dict[int, dict[str, Any]] = {}
    for record_no, row in enumerate(records, 1):
        try: oi = int_field(row.get("op_index"), "op_index")
        except ValueError as exc:
            errors.append(make_error("bad_engine_record", record_number=record_no, reason=str(exc))); continue
        if oi in indexed: errors.append(make_error("duplicate_engine_record", op_index=oi))
        else: indexed[oi] = row
    base_n, reservoir_n, pool_n, query_n, k = (int(header[n]) for n in ("base_n", "reservoir_n", "pool_n", "query_n", "k"))
    radius = float(header["radius"])
    active=set(range(base_n)); checked={"update":0,"knn":0,"range":0}; compared={"knn":0,"range":0}
    for oi, op, arg in events:
        row=indexed.pop(oi, None)
        if row is None:
            errors.append(make_error("missing_engine_record", op_index=oi, op=op))
        elif op in {"insert","delete"}:
            try:
                if row.get("record") != "update": raise ValueError("record must be update")
                if row.get("op") != op: raise ValueError(f"op mismatch expected={op} actual={row.get('op')}")
                if int_field(row.get("stable_id"), "stable_id") != arg: raise ValueError(f"stable ID mismatch expected={arg}")
            except ValueError as exc: errors.append(make_error("update_metadata_mismatch",op_index=oi,reason=str(exc)))
        elif op in {"knn","range"}:
            try:
                if row.get("record") != "query": raise ValueError("record must be query")
                if row.get("kind") != op: raise ValueError(f"kind mismatch expected={op} actual={row.get('kind')}")
                if int_field(row.get("query_id"), "query_id") != arg: raise ValueError(f"query ID mismatch expected={arg}")
                observed=parse_results(row); wanted=exact(pool,queries[arg],active,op,radius,k); compared[op]+=1
                if op == "knn":
                    if len(observed)!=len(wanted): errors.append(make_error("knn_length_mismatch",op_index=oi,expected=len(wanted),observed=len(observed)))
                    for pos,(got,want) in enumerate(zip(observed,wanted)):
                        if got[0]!=want[0] or not np.isclose(got[1],want[1],atol=atol,rtol=rtol):
                            errors.append(make_error("knn_mismatch",op_index=oi,position=pos,expected=[want[0],want[1]],observed=[got[0],got[1]])); break
                else:
                    got={sid:d for sid,d in observed}; want={sid:d for sid,d in wanted}
                    missing=sorted(set(want)-set(got)); extra=sorted(set(got)-set(want)); bad=[sid for sid in sorted(set(got)&set(want)) if not np.isclose(got[sid],want[sid],atol=atol,rtol=rtol)]
                    if missing or extra or bad: errors.append(make_error("range_mismatch",op_index=oi,missing=missing[:20],extra=extra[:20],bad_distance_ids=bad[:20],counts={"missing":len(missing),"extra":len(extra),"bad_distance":len(bad)}))
            except (ValueError, IndexError) as exc: errors.append(make_error("query_metadata_or_result_error",op_index=oi,reason=str(exc)))
        else: errors.append(make_error("unsupported_binary_op",op_index=oi,op=op))
        # Independently replay the trace, not any engine state report.
        if op == "insert":
            if not (base_n<=arg<base_n+reservoir_n<=pool_n): errors.append(make_error("invalid_insert_in_binary_trace",op_index=oi,stable_id=arg))
            elif arg in active: errors.append(make_error("duplicate_insert_in_binary_trace",op_index=oi,stable_id=arg))
            else: active.add(arg); checked['update']+=1
        elif op == "delete":
            if arg not in active: errors.append(make_error("delete_nonlive_in_binary_trace",op_index=oi,stable_id=arg))
            else: active.remove(arg); checked['update']+=1
        elif op in {'knn','range'}: checked[op]+=1
    if indexed: errors.append(make_error("unexpected_engine_records",count=len(indexed),op_indices=sorted(indexed)[:50]))
    return {"schema":"e1gi-b-quantized-gts-integration-verification-v1","scope":"independent NumPy exact verifier for quantized E1-GI-B runner output; NOT GTS result evidence","gpu_used":False,"prepared_bundle":str(bundle),"engine_jsonl":str(engine),"header":header,"quantization":{"scale":100.0,"coordinate_type":"int16","distance_units":"quantized-coordinate units"},"pass":not errors,"checked_events":checked,"exact_query_comparisons":compared,"final_active_count":len(active),"verifier_active_set_sha256":stable_set_sha256(active),"error_count":len(errors),"errors":errors[:200],"ignored_engine_summary_lines":True,"tolerance":{"atol":atol,"rtol":rtol}}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--prepared-bundle',required=True,type=Path);p.add_argument('--engine-jsonl',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--atol',type=float,default=1e-3);p.add_argument('--rtol',type=float,default=1e-5);a=p.parse_args()
    result=verify(a.prepared_bundle.resolve(),a.engine_jsonl.resolve(),a.atol,a.rtol)
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'pass':result['pass'],'error_count':result['error_count'],'out':str(a.out.resolve()),'gpu_used':False},sort_keys=True))
    return 0 if result['pass'] else 2
if __name__=='__main__': raise SystemExit(main())
