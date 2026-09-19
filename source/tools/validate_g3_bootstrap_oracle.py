#!/usr/bin/env python3
"""Validate standalone G3 4K bootstrap-oracle bytes with CPU/Python stdlib only."""
from __future__ import annotations
import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

SCHEMA = "safe-c1-g3-bootstrap-oracle-v1"
VALIDATION_SCHEMA = "safe-c1-g3-bootstrap-oracle-validation-v1"
BUNDLE_REL = Path("inputs/g3_sift4096_branchstress_l2_d3")
ORACLE_REL = Path("preflight/bootstrap_oracles/g3_sift4096_branchstress_l2_d3/bootstrap_oracle.json")
OUT_REL = Path("preflight/bootstrap_oracles/g3_sift4096_branchstress_l2_d3/bootstrap_oracle_validation.json")
K, KNN_QUERY_ID, RANGE_QUERY_ID, RANGE_RADIUS_SQ = 10, 1, 2, 427_400_000
FILES = ("initial_base_stable_ids.i32", "metadata.json", "oracle_expected.jsonl", "pool.i16", "queries.i16", "selection_receipt.json", "stable_id_to_pool_row.i32", "trace.jsonl")
INT64_MAX = (1 << 63) - 1


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def set_sha(ids: tuple[int, ...]) -> str:
    return hashlib.sha256("".join(f"{i}\n" for i in sorted(set(ids))).encode("ascii")).hexdigest()


def ints(path: Path, code: str, n: int) -> tuple[int, ...]:
    raw = path.read_bytes(); width = struct.calcsize("<" + code)
    if len(raw) != n * width:
        raise ValueError(f"bad payload bytes: {path.name}")
    return tuple(x[0] for x in struct.iter_unpack("<" + code, raw))


def under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve()); return True
    except ValueError:
        return False


def d2(pool: tuple[int, ...], queries: tuple[int, ...], mapping: tuple[int, ...], dim: int, qid: int, sid: int) -> int:
    total, qoff, poff = 0, qid * dim, mapping[sid] * dim
    for j in range(dim):
        delta = int(pool[poff + j]) - int(queries[qoff + j]); total += delta * delta
        if total > INT64_MAX: raise OverflowError("int64 L2 overflow")
    return total


def ranked(pool: tuple[int, ...], queries: tuple[int, ...], mapping: tuple[int, ...], dim: int, qid: int, active: tuple[int, ...]) -> list[tuple[int, int]]:
    return sorted((d2(pool, queries, mapping, dim, qid, sid), sid) for sid in active)


def rows(pairs: list[tuple[int, int]]) -> list[list[int]]:
    return [[sid, distance] for distance, sid in pairs]


def fail(errors: list[dict[str, Any]], name: str, expected: Any, actual: Any) -> None:
    if actual != expected: errors.append({"check": name, "expected": expected, "actual": actual})


def load(bundle: Path) -> tuple[dict[str, Any], dict[str, str], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    manifest = json.loads((bundle / "manifest.json").read_text())
    declared = manifest.get("files_sha256")
    if not isinstance(declared, dict) or set(declared) != set(FILES): raise ValueError("bad fixture file set")
    actual = {name: sha(bundle / name) for name in FILES}
    if actual != declared: raise ValueError("fixture manifest hash mismatch")
    if manifest.get("status") != "CPU_PREPARED_NOT_NATIVE_EXECUTED" or manifest.get("gpu_used") is not False: raise ValueError("fixture status")
    meta = json.loads((bundle / "metadata.json").read_text())
    h = meta.get("header", {})
    if (h.get("base_n"), h.get("pool_n"), h.get("query_n"), h.get("dim"), h.get("k")) != (4096,6144,131,128,K): raise ValueError("fixture header")
    m = meta.get("metric_contract", {})
    if (m.get("coordinate_type"),m.get("oracle_distance"),m.get("knn_order"),m.get("range_contract")) != ("int16","int64 squared L2","(distance_sq, stable_id)","inclusive distance_sq <= radius_sq"): raise ValueError("metric contract")
    pool = ints(bundle / "pool.i16", "h", 6144*128)
    queries = ints(bundle / "queries.i16", "h", 131*128)
    mapping = ints(bundle / "stable_id_to_pool_row.i32", "i", 6144)
    active = tuple(sorted(ints(bundle / "initial_base_stable_ids.i32", "i", 4096)))
    if len(set(active)) != 4096 or min(active) < 0 or max(active) >= 6144: raise ValueError("active IDs")
    if sorted(mapping) != list(range(6144)): raise ValueError("mapping")
    return meta, {"manifest.json": sha(bundle / "manifest.json"), **actual}, pool, queries, mapping, active


def validate(root: Path, bundle: Path, oracle: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    try:
        data = json.loads(oracle.read_text())
        meta, hashes, pool, queries, mapping, active = load(bundle)
    except Exception as e:
        return {"schema": VALIDATION_SCHEMA, "status": "FAIL_CPU_ONLY_NOT_RUN", "execution": {"cpu_only":True,"gpu_used":False,"native_engine_executed":False,"binary_launch_count":0}, "error_count":1, "errors":[{"check":"load", "actual":str(e)}]}
    fail(errors,"oracle_outside_fixture",False,under(oracle,bundle))
    fail(errors,"schema",SCHEMA,data.get("schema")); fail(errors,"record","bootstrap_oracle",data.get("record")); fail(errors,"status","CPU_ONLY_NOT_RUN",data.get("status")); fail(errors,"op_index_absent",False,"op_index" in data)
    for k,v in {"cpu_only":True,"gpu_used":False,"native_engine_executed":False,"binary_launch_count":0,"cuda_imported":False,"not_engine_trace":True}.items(): fail(errors,"execution."+k,v,data.get("execution",{}).get(k))
    sep=data.get("trace_separation",{}); fail(errors,"separation.no_append",True,sep.get("must_not_append_to_fixture_or_engine_trace")); fail(errors,"separation.no_op_index",False,sep.get("bootstrap_has_op_index")); fail(errors,"separation.trace",f"{BUNDLE_REL.as_posix()}/trace.jsonl",sep.get("fixture_trace"))
    f=data.get("fixture",{}); fail(errors,"fixture.path",BUNDLE_REL.as_posix(),f.get("bundle_relative_path")); fail(errors,"fixture.hashes",hashes,f.get("all_fixture_inputs_sha256")); fail(errors,"fixture.declared",{k:v for k,v in hashes.items() if k!="manifest.json"},f.get("manifest_declared_files_sha256")); fail(errors,"fixture.verified",True,f.get("all_manifest_hashes_verified")); fail(errors,"fixture.header",{"base_n":4096,"pool_n":6144,"query_n":131,"dim":128,"k":K},f.get("header"))
    active_hash=set_sha(active); a=data.get("active_set",{}); expected_a={"source":"initial_base_stable_ids.i32","engine_state_read":False,"count":4096,"stable_ids_sha256":active_hash,"source_file_sha256":hashes["initial_base_stable_ids.i32"],"membership_rule":"exactly the unique stable IDs decoded from initial_base_stable_ids.i32"}
    for k,v in expected_a.items(): fail(errors,"active."+k,v,a.get(k))
    expected_metric={"coordinate_encoding":"little-endian int16","stable_id_mapping":"stable_id_to_pool_row.i32 maps stable ID to physical pool row","distance":"signed int64 squared L2 accumulated exactly over dim coordinates","knn_k":K,"knn_tie_order":"ascending (distance_sq, stable_id)","range_inclusion":"distance_sq <= radius_sq","range_result_order":"ascending (distance_sq, stable_id)"}
    for k,v in expected_metric.items(): fail(errors,"metric."+k,v,data.get("metric_contract",{}).get(k))
    dim=128; qs=(bundle/"queries.i16").read_bytes(); kn=ranked(pool,queries,mapping,dim,KNN_QUERY_ID,active); rg=ranked(pool,queries,mapping,dim,RANGE_QUERY_ID,active); kth=kn[K-1][0]; nxt=kn[K][0]; rr=[p for p in rg if p[0]<=RANGE_RADIUS_SQ]; boundary=sum(p[0]==RANGE_RADIUS_SQ for p in rg)
    expected_q=[{"bootstrap_id":"initial_base_knn","kind":"knn","query_id":KNN_QUERY_ID,"query_vector_sha256":hashlib.sha256(qs[KNN_QUERY_ID*dim*2:(KNN_QUERY_ID+1)*dim*2]).hexdigest(),"k":K,"radius_sq":None,"active_ids_sha256":active_hash,"results":rows(kn[:K]),"kth_distance_sq":kth,"next_distance_sq":nxt,"kth_distance_tie_count":sum(p[0]==kth for p in kn),"knn_boundary_tie":nxt==kth},{"bootstrap_id":"initial_base_range","kind":"range","query_id":RANGE_QUERY_ID,"query_vector_sha256":hashlib.sha256(qs[RANGE_QUERY_ID*dim*2:(RANGE_QUERY_ID+1)*dim*2]).hexdigest(),"k":None,"radius_sq":RANGE_RADIUS_SQ,"radius_selection":"fixed radius equal to the unique nearest active-base squared distance","active_ids_sha256":active_hash,"results":rows(rr),"distance_equal_to_radius_count":boundary,"result_count":len(rr)}]
    fail(errors,"queries",expected_q,data.get("queries"));
    if not (boundary==1 and len(rr)==1 and rr[0][0]==RANGE_RADIUS_SQ): errors.append({"check":"inclusive_range_boundary","expected":"one exact boundary row","actual":{"boundary":boundary,"rows":rows(rr)}})
    tb=data.get("tool_sha256",{}); fail(errors,"tool.generator",sha(root/"tools/generate_g3_bootstrap_oracle.py"),tb.get("generator")); fail(errors,"tool.validator",sha(root/"tools/validate_g3_bootstrap_oracle.py"),tb.get("validator"))
    return {"schema":VALIDATION_SCHEMA,"status":"PASS_CPU_ONLY_NOT_RUN" if not errors else "FAIL_CPU_ONLY_NOT_RUN","scope":"Independent CPU validation only; no CUDA/native execution, engine-state read, GPU inspection, or performance result.","execution":{"cpu_only":True,"gpu_used":False,"native_engine_executed":False,"binary_launch_count":0,"cuda_imported":False,"not_engine_trace":True},"oracle_path":str(oracle),"oracle_sha256":sha(oracle),"fixture_manifest_sha256":hashes["manifest.json"],"active_set_count":4096,"active_set_sha256":active_hash,"fixed_knn_query_id":KNN_QUERY_ID,"fixed_range_query_id":RANGE_QUERY_ID,"fixed_range_radius_sq":RANGE_RADIUS_SQ,"error_count":len(errors),"errors":errors}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument("--bundle-rel",type=Path,default=BUNDLE_REL); p.add_argument("--oracle-rel",type=Path,default=ORACLE_REL); p.add_argument("--out-rel",type=Path,default=OUT_REL); a=p.parse_args(); root=a.root.resolve(); bundle=(root/a.bundle_rel).resolve(); oracle=(root/a.oracle_rel).resolve(); out=(root/a.out_rel).resolve()
    if under(out,bundle): raise ValueError("report must remain outside fixture bundle")
    result=validate(root,bundle,oracle); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps({"status":result["status"],"gpu_used":False,"error_count":result["error_count"]},sort_keys=True)); return 0 if result["status"]=="PASS_CPU_ONLY_NOT_RUN" else 2
if __name__=="__main__": raise SystemExit(main())
