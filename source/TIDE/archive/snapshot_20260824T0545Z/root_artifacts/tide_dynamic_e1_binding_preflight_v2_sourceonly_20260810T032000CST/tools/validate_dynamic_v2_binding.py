#!/usr/bin/env python3
"""Validate a v2 dynamic-host E1 binding declaration; never executes runtime."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
HEX=re.compile(r"^[0-9a-f]{64}$")
ALLOWED_SCOPE="DYNAMIC_STANDALONE_NONLEGACY_GTSQ_DEVELOPMENT_HOST"
PRIOR_ROOT="/workspace/tide_e1_synthetic_lifecycle_evidence_v2_20260810T030500CST_8a7b6c5d4e3f2910fedcba9876543210"
CAPS=("native_receipt","direct_sidecar","delta_fallback","overlay_delete","base_delete_rebuild","first_post_rebuild_query","tree_checksum","stable_to_local_map","capacity_fault_injection","event_serialization")
class Error(ValueError): pass
def req(ok,msg):
 if not ok: raise Error(msg)
def text(x,n): req(isinstance(x,str) and bool(x),n+" must be nonempty string"); return x
def pin(x,n): x=text(x,n); req(HEX.fullmatch(x) is not None,n+" must be 64 lowercase hex"); return x
def main():
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--binding",required=True,type=Path); args=ap.parse_args()
 try:
  b=json.loads(args.binding.read_text(encoding="utf-8")); req(isinstance(b,dict),"binding must be object")
  req(b.get("schema")=="tide.dynamic-e1-runtime-binding.v2","schema mismatch"); req(b.get("template_only") is not True,"template cannot bind runtime")
  req(b.get("runtime_kind")=="gtsq_lifecycle_host_v2_dynamic_protocol_events","wrong runtime kind")
  req(b.get("runtime_api_version")=="v2_full_protocol_e1_events","dynamic v1 or incomplete event API rejected")
  path=text(b.get("runtime_path"),"runtime_path"); req(Path(path).is_absolute(),"runtime_path must be absolute"); req("tide_e1_synthetic_lifecycle_evidence" not in path,"prior synthetic evidence path rejected")
  req(b.get("qualification_scope")==ALLOWED_SCOPE,"scope must be exact allowed dynamic standalone scope")
  nc=b.get("non_native_non_gpu_nonclaim"); req(isinstance(nc,dict),"nonclaim must be object")
  for k in ("archived_or_native_gts_qualified","gpu_executed_or_qualified","full_paper_gate1_completed"): req(nc.get(k) is False,"nonclaim must explicitly set false: "+k)
  for k in ("runtime_source_manifest_sha256","runtime_binary_sha256","event_emitter_source_sha256","event_schema_sha256","adapter_header_sha256","adapter_source_sha256"): pin(b.get(k),k)
  metric=text(b.get("metric_contract"),"metric_contract")
  if metric!="integer_l2_squared_v1": pin(b.get("numerical_bridge_source_sha256"),"numerical_bridge_source_sha256")
  oracles=b.get("oracles"); req(isinstance(oracles,dict),"oracles must be object"); vals=[]
  for name in ("base","full","mirror"):
   o=oracles.get(name); req(isinstance(o,dict),"missing oracle "+name); text(o.get("id"),"oracle id"); vals.append(pin(o.get("implementation_sha256"),"oracle implementation hash"))
  req(len(set(vals))==3,"oracle implementation hashes collide")
  caps=b.get("actual_capabilities"); req(isinstance(caps,dict),"actual_capabilities must be object")
  for k in CAPS: req(caps.get(k) is True,"actual v2 capability not true: "+k)
  req(b.get("prior_static_synthetic_fixture_forbidden") is True,"prior fixture prohibition missing")
  req(b.get("prohibited_prior_root")==PRIOR_ROOT,"prior fixture root must be explicit and exact")
  req(b.get("separate_evidence_root_required") is True,"fresh root requirement missing")
  print(json.dumps({"status":"PASS_DYNAMIC_V2_BINDING_PREPARATION_ONLY","runtime_kind":b["runtime_kind"],"scope":ALLOWED_SCOPE,"nonclaim":"No runtime, trace, root, GTS, or GPU execution occurred."},sort_keys=True,separators=(",",":"))); return 0
 except (OSError,json.JSONDecodeError,Error) as e:
  print(json.dumps({"status":"FAIL_CLOSED","error":str(e)},sort_keys=True,separators=(",",":"))); return 2
if __name__=="__main__":sys.exit(main())
