#!/usr/bin/env python3
"""Generate Safe-C2 v2 tie-free IDs from frozen v1 memberships without GPU use.

Admission requires agreement between exact int64 squared-L2 and the explicit
source-level C++ raw_l2 fp32 recurrence (ordered accumulation then sqrt).
"""
import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

D = 128
W = 100
FVEC_DTYPE = np.dtype([("dim", "<i4"), ("v", "<f4", (D,))])
IVEC_DTYPE = np.dtype([("dim", "<i4"), ("ids", "<i4", (W,))])
PARTS = ("calibration", "validation", "test")

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()
def read_ids(path: Path) -> list[int]:
    x=[int(v) for v in path.read_text().split()]
    if len(x)!=len(set(x)): raise RuntimeError(f"duplicate IDs in {path}")
    return x
def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",dir=path.parent,delete=False) as f:
        f.write(text); temp=Path(f.name)
    os.replace(temp,path)
def ids_payload(ids:list[int])->str: return "".join(f"{q}\n" for q in ids)
def cxx_raw_l2_fp32_rows(a:np.ndarray,b:np.ndarray)->np.ndarray:
    total=np.zeros(a.shape[0],dtype=np.float32)
    for dim in range(D):
        delta=np.subtract(a[:,dim],b[dim],dtype=np.float32)
        term=np.multiply(delta,delta,dtype=np.float32)
        total=np.add(total,term,dtype=np.float32)
    return np.sqrt(total,dtype=np.float32)
def classify(base,queries,gt,q:int)->dict:
    candidates=gt["ids"][q]; b=base["v"][candidates]; qq=queries["v"][q]
    if np.any(candidates<0) or np.any(candidates>=len(base)): raise RuntimeError(f"invalid public GT ID at q={q}")
    if not np.all(b==np.floor(b)) or not np.all(qq==np.floor(qq)): raise RuntimeError(f"non-integral coordinate violates v2 exact-L2 premise at q={q}")
    diff=b.astype(np.int64)-qq.astype(np.int64); d2=np.sum(diff*diff,axis=1,dtype=np.int64); eo=np.lexsort((candidates,d2))
    f=cxx_raw_l2_fp32_rows(b,qq); fo=np.lexsort((candidates,f)); stored={int(x) for x in candidates[:10]}
    buckets={}
    for dist,sq in zip(f,d2): buckets.setdefault(int(np.asarray(dist,dtype=np.float32).view(np.uint32)),set()).add(int(sq))
    collapse=sum(len(v)*(len(v)-1)//2 for v in buckets.values() if len(v)>1)
    return {"exact_boundary_tie":bool(d2[eo[9]]==d2[eo[10]]),"exact_stored_top10_mismatch":stored!={int(x) for x in candidates[eo[:10]]},"fp32_boundary_tie":bool(f[fo[9]]==f[fo[10]]),"fp32_stored_top10_mismatch":stored!={int(x) for x in candidates[fo[:10]]},"fp32_equal_distance_distinct_d2_pairs":int(collapse),"max_d2":int(np.max(d2))}
def ambiguous(x:dict)->bool:
    return any((x["exact_boundary_tie"],x["exact_stored_top10_mismatch"],x["fp32_boundary_tie"],x["fp32_stored_top10_mismatch"],x["fp32_equal_distance_distinct_d2_pairs"]!=0))
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--v1-protocol",type=Path,required=True); ap.add_argument("--v2-preflight-dir",type=Path,required=True); args=ap.parse_args()
    p1=json.loads(args.v1_protocol.read_text())
    if p1.get("schema")!="safe-c2-corrected-after-submit-protocol-v1": raise RuntimeError("not the frozen Safe-C2 v1 protocol")
    data=p1["data"]; base_path,query_path,gt_path=(Path(data[x]["path"]) for x in ("base","query","groundtruth")); parts=p1["split"]["files"]
    original={part:read_ids(Path(parts[part]["path"])) for part in PARTS}; all_original=sum((original[k] for k in PARTS),[])
    if len(all_original)!=10000 or set(all_original)!=set(range(10000)): raise RuntimeError("v1 split is not disjoint/exhaustive 0..9999")
    base=np.memmap(base_path,dtype=FVEC_DTYPE,mode="r"); queries=np.memmap(query_path,dtype=FVEC_DTYPE,mode="r"); gt=np.memmap(gt_path,dtype=IVEC_DTYPE,mode="r")
    if len(base)!=1000000 or len(queries)!=10000 or len(gt)!=10000: raise RuntimeError("unexpected SIFT1M record counts")
    if not np.all(base["dim"]==D) or not np.all(queries["dim"]==D) or not np.all(gt["dim"]==W): raise RuntimeError("unexpected fvecs/ivecs row headers")
    classes={q:classify(base,queries,gt,q) for q in range(10000)}; out_dir=args.v2_preflight_dir; out_dir.mkdir(parents=True,exist_ok=True); selected={}; excluded={}; entries={}
    for part in PARTS:
        selected[part]=[q for q in original[part] if not ambiguous(classes[q])]; excluded[part]=[q for q in original[part] if ambiguous(classes[q])]
        sp=out_dir/f"{part}.ids"; ep=out_dir/f"{part}.excluded_ambiguous.ids"; atomic_text(sp,ids_payload(selected[part])); atomic_text(ep,ids_payload(excluded[part]))
        entries[part]={"original":{"path":str(Path(parts[part]["path"])),"count":len(original[part]),"sha256":sha(Path(parts[part]["path"]))},"selected":{"path":str(sp),"count":len(selected[part]),"sha256":sha(sp),"first_ids":selected[part][:8]},"excluded_ambiguous":{"path":str(ep),"count":len(excluded[part]),"sha256":sha(ep),"ids":excluded[part]}}
    all_excluded=[q for part in PARTS for q in excluded[part]]; allp=out_dir/"all.excluded_ambiguous.ids"; atomic_text(allp,ids_payload(all_excluded))
    def fpstats(ids):
        cs=[classes[q] for q in ids]
        return {"fp32_boundary_ties":sum(x["fp32_boundary_tie"] for x in cs),"fp32_stored_top10_mismatches":sum(x["fp32_stored_top10_mismatch"] for x in cs),"fp32_equal_distance_distinct_d2_pairs":sum(x["fp32_equal_distance_distinct_d2_pairs"] for x in cs),"max_d2":max(x["max_d2"] for x in cs)}
    audit={"schema":"safe-c2-tiefree-selection-v2","created_utc":datetime.now(timezone.utc).isoformat(),"cuda_used":False,"implementation_identity":"Safe-C2 v2 tie-free corrected-after-submit; not submitted legacy C2","source_v1_protocol":{"path":str(args.v1_protocol),"sha256":sha(args.v1_protocol)},"raw_data":{key:{"path":data[key]["path"],"protocol_sha256":data[key]["sha256"],"actual_sha256":sha(Path(data[key]["path"]))} for key in ("base","query","groundtruth")},"split_seed_unchanged":p1["split"]["seed"],"membership_rule":"Preserve frozen v1 membership and order within each stage; remove only IDs that violate exact or source-level-fp32 public-GT tie-free admission.","eligibility_rule":{"distance":"exact int64 squared L2 plus source-level raw_l2 fp32 recurrence on integral raw SIFT128 values","stable_order":"ascending (distance, ID) within stored public top-100","selected_id_requirements":["exact and fp32 rank-10/rank-11 are unequal","stored public GT first-10 ID set equals exact and fp32 stable top-10 ID set","no fp32 equal-distance collision among distinct squared distances in public GT100"]},"partitions":entries,"all_excluded_ambiguous":{"path":str(allp),"count":len(all_excluded),"sha256":sha(allp),"ids":all_excluded},"selected_total":sum(len(v) for v in selected.values()),"excluded_total":len(all_excluded),"selected_disjoint":len(set().union(*map(set,selected.values())))==sum(len(v) for v in selected.values()),"selected_subset_of_v1":all(set(selected[k]).issubset(original[k]) for k in selected),"selected_plus_excluded_partition_original_10000":set(sum((selected[p] for p in PARTS),[]))|set(all_excluded)==set(all_original),"fp32_source_semantics":"ordered volatile float32 accumulation of delta*delta, then float32 sqrt; matches raw_l2 source-level recurrence","fp32_selected_checks":{p:fpstats(selected[p]) for p in PARTS}}
    amb={q for q,x in classes.items() if ambiguous(x)}
    if audit["excluded_total"]!=len(amb) or set(all_excluded)!=amb: raise RuntimeError("per-stage exclusions do not exactly cover full exact+fp32 ambiguity set")
    if audit["selected_total"]!=9863 or audit["excluded_total"]!=137: raise RuntimeError(f"unexpected final tie-free split selected={audit['selected_total']} excluded={audit['excluded_total']}")
    for x in audit["fp32_selected_checks"].values():
        if x["fp32_boundary_ties"] or x["fp32_stored_top10_mismatches"] or x["fp32_equal_distance_distinct_d2_pairs"]: raise RuntimeError("selected split violates fp32 admission")
    apath=out_dir/"selection_audit.json"; atomic_text(apath,json.dumps(audit,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"PASS","cuda_used":False,"selection_audit":str(apath),"selected_counts":{k:len(v) for k,v in selected.items()},"excluded_counts":{k:len(v) for k,v in excluded.items()},"selected_total":audit["selected_total"],"excluded_total":audit["excluded_total"],"fp32_selected_checks":audit["fp32_selected_checks"]},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
