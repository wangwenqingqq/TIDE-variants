#!/usr/bin/env python3
"""Validate a final dynamic-runtime binding declaration; no runtime execution."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
HEX=re.compile(r"^[0-9a-f]{64}$")
CAPS=("native_receipt","direct_sidecar","delta_fallback","overlay_delete","base_delete_rebuild","first_post_rebuild_query","tree_checksum","stable_to_local_map","capacity_fault_injection","event_serialization")
class Error(ValueError): pass
def req(ok,msg):
 if not ok: raise Error(msg)
def sha(x,name): req(isinstance(x,str) and HEX.fullmatch(x) is not None,name+" must be 64 lowercase hex"); return x
def text(x,name): req(isinstance(x,str) and x,name+" must be nonempty string"); return x
def main():
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--binding",required=True,type=Path); a=ap.parse_args()
 try:
  b=json.loads(a.binding.read_text(encoding="utf-8")); req(isinstance(b,dict),"binding must be object")
  req(b.get("schema")=="tide.dynamic-e1-runtime-binding.v1","schema mismatch"); req(b.get("template_only") is not True,"template cannot bind runtime")
  req(b.get("runtime_kind")=="gtsq_lifecycle_host_v1_dynamic_qualified_core","runtime_kind mismatch")
  p=text(b.get("runtime_path"),"runtime_path"); req(Path(p).is_absolute(),"runtime_path must be absolute")
  for k in ("runtime_source_manifest_sha256","runtime_binary_sha256","event_emitter_source_sha256","event_schema_sha256","adapter_header_sha256","adapter_source_sha256"): sha(b.get(k),k)
  scope=text(b.get("qualification_scope"),"qualification_scope").lower(); req(not any(x in scope for x in ("synthetic","standalone","static")),"qualification scope binds forbidden non-dynamic subject")
  metric=text(b.get("metric_contract"),"metric_contract")
  if metric != "integer_l2_squared_v1": sha(b.get("numerical_bridge_source_sha256"),"numerical_bridge_source_sha256")
  oracles=b.get("oracles"); req(isinstance(oracles,dict),"oracles must be object"); pins=[]
  for name in ("base","full","mirror"):
   r=oracles.get(name); req(isinstance(r,dict),"missing oracle "+name); text(r.get("id"),"oracle id"); pins.append(sha(r.get("implementation_sha256"),"oracle hash"))
  req(len(set(pins))==3,"oracle implementation hashes collide")
  caps=b.get("actual_capabilities"); req(isinstance(caps,dict),"actual_capabilities must be object")
  for c in CAPS: req(caps.get(c) is True,"capability not true: "+c)
  req(b.get("prior_synthetic_evidence_forbidden") is True,"must forbid prior synthetic evidence")
  req(b.get("separate_evidence_root_required") is True,"must require separate evidence root")
  print(json.dumps({"status":"PASS_DYNAMIC_BINDING_PREPARATION_ONLY","runtime_kind":b["runtime_kind"],"metric_contract":metric,"nonclaim":"No runtime, trace, root, or E1 execution occurred."},sort_keys=True,separators=(",",":")))
  return 0
 except (OSError,json.JSONDecodeError,Error) as e:
  print(json.dumps({"status":"FAIL_CLOSED","error":str(e)},sort_keys=True,separators=(",",":"))); return 2
if __name__=="__main__": sys.exit(main())
