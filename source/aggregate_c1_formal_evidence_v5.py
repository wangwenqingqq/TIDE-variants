#!/usr/bin/env python3
"""CPU-only cross-root gate: measured C1 evidence plus separate Nsight profile evidence."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,re
from datetime import datetime,timezone
from typing import Any

ROOT_DEFAULT="/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v5_formal"
RUNS_ROOT=pathlib.Path(ROOT_DEFAULT)/"runs"
PROFILES_ROOT=pathlib.Path(ROOT_DEFAULT)/"profiles"
PINS_PATH=pathlib.Path(ROOT_DEFAULT)/"hardened_static_pins_v5.json"
SELF_PATH=pathlib.Path(ROOT_DEFAULT)/"aggregate_c1_formal_evidence_v5.py"

def sha(p:pathlib.Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def raw_file(value:str,label:str)->pathlib.Path:
    p=pathlib.Path(value)
    if not p.is_absolute() or p.is_symlink() or not p.is_file():raise SystemExit(f"{label}: absolute regular non-symlink file required: {p}")
    q=p.resolve(strict=True)
    if q!=p:raise SystemExit(f"{label}: noncanonical/symlink traversal: {p} -> {q}")
    return p
def raw_dir(value:str,label:str)->pathlib.Path:
    p=pathlib.Path(value)
    if not p.is_absolute() or p.is_symlink() or not p.is_dir():raise SystemExit(f"{label}: absolute non-symlink directory required: {p}")
    q=p.resolve(strict=True)
    if q!=p:raise SystemExit(f"{label}: noncanonical/symlink traversal: {p} -> {q}")
    return p
def load(p:pathlib.Path,label:str)->dict[str,Any]:
    raw_file(str(p),label)
    try:x=json.loads(p.read_text())
    except Exception as e:raise SystemExit(f"{label}: invalid JSON: {e}")
    if not isinstance(x,dict):raise SystemExit(f"{label}: JSON object required")
    return x
def norm_pci(value:Any)->str|None:
    m=re.fullmatch(r"(?:(?:[0-9a-f]{4}|[0-9a-f]{8}):)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])",str(value).strip().lower())
    return m.group(1) if m else None
def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--measured-root",type=pathlib.Path,required=True)
    ap.add_argument("--profile-root",type=pathlib.Path,required=True)
    ap.add_argument("--pins",type=pathlib.Path,required=True)
    a=ap.parse_args()
    pins_path=raw_file(str(a.pins),"pins")
    if pins_path!=PINS_PATH:raise SystemExit("pins must be canonical v5 hardened pin file")
    self_path=raw_file(str(pathlib.Path(__file__)),"aggregate self")
    if self_path!=SELF_PATH:raise SystemExit("aggregate must execute from canonical v5 self path")
    mroot=raw_dir(str(a.measured_root),"measured root")
    proot=raw_dir(str(a.profile_root),"profile root")
    if mroot.parent!=RUNS_ROOT or not mroot.name.startswith("c1_v5_measured_"):raise SystemExit("measured root boundary")
    if proot.parent!=PROFILES_ROOT or not proot.name.startswith("c1_v5_profile_"):raise SystemExit("profile root boundary")
    pins=load(pins_path,"pins")
    if pins.get("schema")!="gtspp-c1-v5-pins-v1":raise SystemExit("pins schema")
    self_pin=pins.get("runtime_helpers",{}).get("formal_aggregator")
    if not isinstance(self_pin,dict) or self_pin.get("path")!=str(self_path) or self_pin.get("sha256")!=sha(self_path):raise SystemExit("aggregate self pin/provenance mismatch")
    m=load(mroot/"final_evidence_v5.json","measured final evidence")
    p=load(proot/"profile_verification_v5.json","profile verification")
    ps=sha(pins_path)
    if m.get("schema")!="gtspp-c1-v5-measured-evidence-v2" or m.get("pass") is not True or m.get("formal_claim_eligible") is not False:raise SystemExit("measured evidence not in required pre-formal state")
    if m.get("run_root")!=str(mroot):raise SystemExit("measured evidence root mismatch")
    if p.get("schema")!="gtspp-c1-v5-profile-verification-v3" or p.get("pass") is not True or p.get("formal_profile_eligible") is not True:raise SystemExit("profile verification not pass")
    if p.get("profile_root")!=str(proot):raise SystemExit("profile verification root mismatch")
    if m.get("pins_sha256")!=ps or p.get("pins_sha256")!=ps:raise SystemExit("cross-root pins disagreement")
    if m.get("gpu_uuid")!=p.get("gpu_uuid") or not str(m.get("gpu_uuid","")).startswith("GPU-"):raise SystemExit("cross-root GPU UUID disagreement")
    mpci=norm_pci(m.get("gpu_pci_bus_id_normalized"));ppci=norm_pci(p.get("gpu_pci_bus_id_normalized"))
    if mpci is None or ppci is None or mpci!=ppci:raise SystemExit("cross-root GPU PCI disagreement")
    out={
        "schema":"gtspp-c1-v5-formal-evidence-aggregate-v3","created_utc":datetime.now(timezone.utc).isoformat(),"pins_sha256":ps,
        "aggregate_script":{"path":str(self_path),"sha256":sha(self_path)},"aggregate_script_sha256":sha(self_path),
        "measured_evidence":{"path":str(mroot/"final_evidence_v5.json"),"sha256":sha(mroot/"final_evidence_v5.json")},
        "profile_verification":{"path":str(proot/"profile_verification_v5.json"),"sha256":sha(proot/"profile_verification_v5.json")},
        "gpu_uuid":m["gpu_uuid"],"gpu_pci_bus_id_normalized":mpci,"pass":True,"formal_claim_eligible":True,
        "scope":"Only the frozen clean C1 SIFT1M query-only protocol. It supports a bounded C1 timing/allocation statement after both roots pass; it does not validate C2/C3, end-to-end updates, recall, generality, or absolute range-search correctness.",
        "correctness_scope":"cross-variant differential equivalence only; no independent absolute range-search oracle is included.",
    }
    target=proot/"formal_evidence_aggregate_v5.json"
    if target.is_symlink():raise SystemExit("refuse symlink formal aggregate output")
    tmp=target.with_name(target.name+".tmp");tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");tmp.replace(target)
    print(json.dumps({"pass":True,"output":str(target)}));return 0
if __name__=="__main__":raise SystemExit(main())
