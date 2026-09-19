#!/usr/bin/env python3
"""Build a canary-only strict workload manifest without GPU/CUDA actions."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT=Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
FORMAL=ROOT/"inputs/final_workload_v2/workload_manifest.json"
CANARY_SOURCE=ROOT/"inputs/final_workload_v2/qualification_canary.ids"
CANARY_MAP_SOURCE=ROOT/"inputs/final_workload_v2/qualification_canary.mapping.tsv"
PROTOCOL=ROOT/"protocols/c2_v4_canary_runtime_protocol_v1.json"
PRELEDGER=ROOT/"provenance/c2_v4_preledger_gate_v2_independent_audit.json"
OUT=ROOT/"inputs/canary_workload_v1"
FIELDS={
"schema","status","selection_pre_gt_path","selection_pre_gt_sha256","v3_forensic_record_path","v3_forensic_record_sha256","formal_protocol_path","formal_protocol_sha256","sealed_v2_test_forbidden","v2_sealed_test_ids_sha256_forbidden","base_fvecs_path","base_fvecs_sha256","query_fvecs_path","query_fvecs_sha256","query_fvecs_count","groundtruth_ivecs_path","groundtruth_ivecs_sha256","groundtruth_width","mapping_path","mapping_sha256","calibration_ids_path","calibration_ids_sha256","calibration_query_count","validation_ids_path","validation_ids_sha256","validation_query_count","sealed_test_ids_path","sealed_test_ids_sha256","sealed_test_query_count"}

def fail(x:str)->None: raise RuntimeError(x)
def direct_file(p:Path,label:str)->Path:
 if not p.is_absolute() or p.is_symlink() or not p.is_file() or p.resolve()!=p:fail(label+": not direct file "+str(p))
 return p
def direct_dir(p:Path,label:str)->Path:
 if not p.is_absolute() or p.is_symlink() or not p.is_dir() or p.resolve()!=p:fail(label+": not direct directory "+str(p))
 return p
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def loads(p:Path)->dict:
 direct_file(p,str(p))
 def hook(rows):
  d={}
  for k,v in rows:
   if k in d:fail("duplicate JSON key "+k)
   d[k]=v
  return d
 x=json.loads(p.read_text(encoding="utf-8"),object_pairs_hook=hook)
 if not isinstance(x,dict):fail("nonobject JSON")
 return x
def canon_ids(p:Path,n:int)->list[int]:
 direct_file(p,"canary source IDs")
 rows=p.read_text(encoding="ascii").splitlines()
 if len(rows)!=n:fail("canary count")
 ans=[]
 for i,x in enumerate(rows):
  if not x or not x.isdecimal() or (len(x)>1 and x[0]=="0"):fail("bad canary ID")
  ans.append(int(x))
 if len(set(ans))!=n or any(x<0 or x>=12524 for x in ans):fail("bad canary uniqueness/range")
 return ans
def write_text(p:Path,text:str)->None:
 with p.open("x",encoding="ascii",newline="\n") as f:
  f.write(text);f.flush();os.fsync(f.fileno())
def write_json(p:Path,x:dict)->None:
 with p.open("x",encoding="utf-8",newline="\n") as f:
  json.dump(x,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())

def main()->int:
 if OUT.exists() or OUT.is_symlink():fail("refuse existing canary workload")
 direct_dir(OUT.parent,"inputs parent")
 if (ROOT/"runs").exists() or (ROOT/".locks").exists():fail("normal C2 run/lock exists")
 for p in (FORMAL,CANARY_SOURCE,CANARY_MAP_SOURCE,PROTOCOL,PRELEDGER):direct_file(p,str(p))
 formal=loads(FORMAL); protocol=loads(PROTOCOL); pre=loads(PRELEDGER)
 if set(formal)!=FIELDS or formal.get("status")!="READY_FOR_C2_V4_FORMAL_AFTER_PRELEDGER_PARSER_GATE":fail("formal manifest contract")
 if protocol.get("status")!="FROZEN_CANARY_MANIFEST_BUILD_PENDING_NO_GPU":fail("canary protocol state")
 if pre.get("status")!="PASS_PRELEDGER_GATE_V2_STRICT_RECEIPT_VERIFIED":fail("formal preledger not verified")
 if protocol["formal_workload_v2"]["sha256"]!=sha(FORMAL) or protocol["canary_selection"]["sha256"]!=sha(CANARY_SOURCE):fail("protocol binding")
 ids=canon_ids(CANARY_SOURCE,1010)
 maps=CANARY_MAP_SOURCE.read_text(encoding="ascii").splitlines()
 if len(maps)!=len(ids):fail("canary map count")
 for q,row in zip(ids,maps):
  a=row.split("\t")
  if len(a)!=3 or a[0]!=str(q):fail("canary map binding")
 temp=OUT.parent/(OUT.name+".tmp")
 if temp.exists() or temp.is_symlink():fail("existing canary temp")
 temp.mkdir(mode=0o750)
 try:
  canary_ids=temp/"qualification_canary.ids"
  canary_map=temp/"qualification_canary.mapping.tsv"
  write_text(canary_ids,"".join(str(x)+"\n" for x in ids))
  write_text(canary_map,"\n".join(maps)+"\n")
  manifest=dict(formal)
  manifest["formal_protocol_path"]=str(PROTOCOL)
  manifest["formal_protocol_sha256"]=sha(PROTOCOL)
  for stage in ("calibration","validation","sealed_test"):
   manifest[stage+"_ids_path"]=str(OUT/"qualification_canary.ids")
   manifest[stage+"_ids_sha256"]=sha(canary_ids)
   manifest[stage+"_query_count"]=len(ids)
  if set(manifest)!=FIELDS:fail("canary manifest fields")
  write_json(temp/"workload_manifest.json",manifest)
  receipt={
   "schema":"safe-c2-v4-canary-workload-build-receipt-v1",
   "status":"PASS_CPU_ONLY_CANARY_MANIFEST_BUILT_NO_GPU",
   "scope":"Canary-only manifest. The C++ binary calibration field is isolated to canary IDs; duplicate validation/test bindings are schema adapters and not formal IDs.",
   "formal_workload_v2":{"path":str(FORMAL),"sha256":sha(FORMAL)},
   "canary_protocol":{"path":str(PROTOCOL),"sha256":sha(PROTOCOL)},
   "canary_ids_source":{"path":str(CANARY_SOURCE),"sha256":sha(CANARY_SOURCE)},
   "canary_manifest_future_path":str(OUT/"workload_manifest.json"),
   "canary_count":len(ids),
   "formal_id_paths_visible_to_canary_binary":False,
   "gpu_work_executed":False,
   "formal_claim_eligible":False,
  }
  write_json(temp/"canary_workload_receipt.json",receipt)
  hashes={p.name:sha(p) for p in temp.iterdir() if p.is_file()}
  write_text(temp/"canary_artifacts.sha256","".join(d+"  "+n+"\n" for n,d in sorted(hashes.items())))
  os.replace(temp,OUT)
 except Exception:
  shutil.rmtree(temp,ignore_errors=True)
  raise
 print(json.dumps({"status":"PASS_C2_V4_CANARY_WORKLOAD_CPU_ONLY","manifest":str(OUT/"workload_manifest.json"),"manifest_sha256":sha(OUT/"workload_manifest.json"),"canary_count":len(ids),"gpu_work_executed":False,"formal_claim_eligible":False},sort_keys=True))
 return 0
if __name__=="__main__":
 try:raise SystemExit(main())
 except Exception as e:
  print("BUILD_C2_V4_CANARY_WORKLOAD_FAIL: "+str(e),file=sys.stderr);raise SystemExit(2)
