#!/usr/bin/env python3
"""Final CPU-only attestation written only after the v5 guard reaches COMPLETE."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, re
from datetime import datetime, timezone

def sha(p:pathlib.Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def regular(p:pathlib.Path,label:str)->pathlib.Path:
    if not p.is_absolute() or p.is_symlink() or not p.is_file(): raise SystemExit(f"{label}: not absolute regular non-symlink {p}")
    if p.resolve(strict=True)!=p: raise SystemExit(f"{label}: noncanonical/symlink traversal {p}")
    return p
def obj(p:pathlib.Path,label:str)->dict:
    regular(p,label)
    try:
        x=json.loads(p.read_text())
    except Exception as e: raise SystemExit(f"{label}: invalid JSON {e}")
    if not isinstance(x,dict):raise SystemExit(f"{label}: JSON object required")
    return x
def need(x:bool,message:str)->None:
    if not x: raise SystemExit(message)
def norm_pci(value:object)->str|None:
    m=re.fullmatch(r"(?:(?:[0-9a-f]{4}|[0-9a-f]{8}):)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])",str(value).strip().lower())
    return m.group(1) if m else None
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--run-root",type=pathlib.Path,required=True);ap.add_argument("--pins",type=pathlib.Path,required=True);a=ap.parse_args()
    root=a.run_root
    need(root.is_absolute() and root.is_dir() and not root.is_symlink() and root.resolve()==root,"unsafe run root")
    pins_path=regular(a.pins,"pins"); pins=obj(pins_path,"pins"); pin_sha=sha(pins_path)
    manifest=obj(root/"run_manifest_v5.json","manifest"); card=obj(root/"guard_run_card_v5.json","guard card")
    semantic_path=root/"semantic_verification_v5.json"; analysis_path=root/"analysis_summary_v5.json"
    semantic=obj(semantic_path,"semantic report"); analysis=obj(analysis_path,"analysis report")
    need(manifest.get("schema")=="gtspp-c1-v5-run-manifest-v2" and manifest.get("execution_mode")=="MEASURED","manifest mode/schema")
    need(manifest.get("status")=="COMPLETE" and card.get("status")=="COMPLETE","manifest/guard not COMPLETE")
    need(manifest.get("pins",{}).get("sha256")==pin_sha and semantic.get("pins_sha256")==pin_sha and analysis.get("pins_sha256")==pin_sha,"pins SHA disagreement")
    need(semantic.get("pass") is True and semantic.get("failure_count")==0,"semantic gate not pass")
    need(analysis.get("semantic_report_sha256")==sha(semantic_path),"analysis not bound to semantic report")
    gpu=manifest.get("gpu",{}); uuid=gpu.get("uuid")
    need(gpu.get("physical_index")==0 and isinstance(uuid,str) and uuid.startswith("GPU-") and card.get("physical_gpu_uuid")==uuid,"GPU provenance disagreement")
    telemetry=regular(pathlib.Path(str(gpu.get("telemetry_snapshot",""))),"manifest outer telemetry")
    need(str(telemetry).startswith(str(root/"gpu_snapshots")+"/"),"outer telemetry boundary")
    telemetry_lines=[x.strip() for x in telemetry.read_text(errors="replace").splitlines() if x.strip()]
    need(len(telemetry_lines)==1,"outer telemetry row count")
    telemetry_fields=[x.strip() for x in telemetry_lines[0].split(",")]
    need(len(telemetry_fields)>=5 and telemetry_fields[1]=="0" and telemetry_fields[2]==uuid,"outer telemetry UUID identity")
    gpu_pci=norm_pci(telemetry_fields[4])
    need(gpu_pci is not None,"outer telemetry malformed PCI")
    need(not (root/".c1_guard_session_v5.json").exists(),"guard session residue")
    plan=manifest.get("execution_plan",{}); plan_path=regular(pathlib.Path(str(plan.get("path",""))),"execution plan")
    need(sha(plan_path)==plan.get("sha256"),"execution plan hash mismatch")
    plan_json=obj(plan_path,"execution plan"); schedule=plan_json.get("schedule")
    need(isinstance(schedule,list) and len(schedule)==20 and plan.get("schedule")==schedule,"schedule binding invalid")
    children=manifest.get("children",[])
    need(isinstance(children,list) and len(children)==20,"exactly 20 child provenance records required")
    expected=[(x.get("replicate"),x.get("variant")) for x in schedule]
    got=[(x.get("replicate"),x.get("variant")) for x in children]
    need(got==expected,"child launch order differs from fixed schedule")
    pins_by_variant={x.get("name"):x for x in pins.get("variant_binaries",[])}
    for child in children:
        v=child["variant"]; b=child.get("binary",{})
        need(v in pins_by_variant and b.get("path")==pins_by_variant[v].get("path") and b.get("sha256")==pins_by_variant[v].get("sha256"),f"child binary provenance {v}")
        env=child.get("environment",{})
        need(child.get("env_i") is True and env.get("CUDA_VISIBLE_DEVICES")==uuid and env.get("NVIDIA_VISIBLE_DEVICES")==uuid and env.get("CUDA_DEVICE_ORDER")=="PCI_BUS_ID" and env.get("C1_EXECUTION_MODE")=="MEASURED",f"child visibility/env {v}")
    events=manifest.get("events",[])
    need(sum(1 for x in events if x.get("kind")=="variant_completion_contract_pass")==20,"not all child completion events recorded")
    need(any(x.get("kind")=="execution_schedule_verified" for x in events),"schedule verification event missing")
    snap=root/"gpu_snapshots"; need(snap.is_dir() and not snap.is_symlink(),"snapshot directory")
    hashes=[]
    for p in sorted(snap.iterdir()):
        if p.is_file():
            regular(p,"snapshot"); hashes.append({"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size})
    need(any(x["path"].endswith("outer_postrun_telemetry_gpu0.csv") for x in hashes) and any(x["path"].endswith("outer_exit_telemetry_gpu0.csv") for x in hashes),"final telemetry absent")
    result={"schema":"gtspp-c1-v5-measured-evidence-v2","attested_utc":datetime.now(timezone.utc).isoformat(),"run_root":str(root),"pins_sha256":pin_sha,
      "manifest_sha256":sha(root/"run_manifest_v5.json"),"guard_card_sha256":sha(root/"guard_run_card_v5.json"),
      "semantic_sha256":sha(semantic_path),"analysis_sha256":sha(analysis_path),"execution_plan_sha256":sha(plan_path),
      "gpu_uuid":uuid,"gpu_pci_bus_id_normalized":gpu_pci,"children":len(children),"telemetry_artifacts":hashes,"pass":True,"formal_claim_eligible":False,"status":"MEASURED_EVIDENCE_ONLY_PROFILE_REQUIRED","warning":"This attests only the measured timing run. A separately guarded E_G/P_F Nsight profile and cross-root aggregate with exact UUID and PCI binding are still required before any formal C1 statement."}
    out=root/"final_evidence_v5.json";tmp=out.with_suffix(".tmp");tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");tmp.replace(out)
    print(json.dumps({"pass":True,"output":str(out),"children":20,"telemetry_files":len(hashes)}))
    return 0
if __name__=="__main__":raise SystemExit(main())

