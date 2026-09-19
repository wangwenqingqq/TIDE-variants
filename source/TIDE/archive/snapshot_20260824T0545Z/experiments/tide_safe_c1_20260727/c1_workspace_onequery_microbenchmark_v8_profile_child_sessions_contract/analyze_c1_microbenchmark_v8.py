#!/usr/bin/env python3
"""CPU-only measured-only analysis after strict C1 v8 semantic verification."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, importlib.util, json, math, pathlib, random, statistics
from datetime import datetime, timezone

VARIANTS=("E_G_c1_off_reference","P_G_workspace_only","E_F_fastpath_only","P_F_full_C1")
ROOT_DEFAULT=pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v8_profile_child_sessions_contract")
_contract_spec=importlib.util.spec_from_file_location("gtspp_c1_v8_capacity_snapshot_contract",ROOT_DEFAULT/"capacity_snapshot_contract_v8.py")
if _contract_spec is None or _contract_spec.loader is None: raise RuntimeError("cannot load v8 capacity snapshot contract")
_contract=importlib.util.module_from_spec(_contract_spec); _contract_spec.loader.exec_module(_contract)
capacity_snapshot_violations=_contract.capacity_snapshot_violations
def sha(p:pathlib.Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def q(xs:list[float],x:float)->float:
    ys=sorted(xs); z=(len(ys)-1)*x; lo=int(z); hi=min(lo+1,len(ys)-1); return ys[lo]*(hi-z)+ys[hi]*(z-lo)
def stats(xs:list[float])->dict:
    if not xs: raise ValueError("empty distribution")
    return {"n":len(xs),"mean":statistics.fmean(xs),"median":q(xs,.5),"p95":q(xs,.95),"p99":q(xs,.99)}
def ci(xs:list[float],seed:int=20260727,draws:int=10000)->dict:
    if not xs: raise ValueError("empty CI")
    r=random.Random(seed); n=len(xs); means=sorted(statistics.fmean(xs[r.randrange(n)] for _ in range(n)) for _ in range(draws))
    return {"seed":seed,"draws":draws,"mean":statistics.fmean(xs),"ci95_low":q(means,.025),"ci95_high":q(means,.975)}
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--run-root",type=pathlib.Path,required=True);ap.add_argument("--pins",type=pathlib.Path,required=True);a=ap.parse_args()
    root=a.run_root.resolve(); pins=a.pins.resolve()
    semantic=root/"semantic_verification_v8.json"
    if not semantic.is_file() or semantic.is_symlink():raise SystemExit("strict semantic report missing/symlink")
    s=json.loads(semantic.read_text())
    if s.get("schema")!="gtspp-c1-v8-semantic-verification-v1" or s.get("pass") is not True:raise SystemExit("refusing statistics: strict semantic gate did not pass")
    if s.get("pins_sha256")!=sha(pins):raise SystemExit("refusing statistics: pins drifted after semantic gate")
    manifest=json.loads((root/"run_manifest_v8.json").read_text())
    if manifest.get("execution_mode")!="MEASURED":raise SystemExit("refusing non-measured/profile run")
    aggregate={}; names=[]
    for rep in range(1,6):
        names.append(f"rep{rep}")
        for v in VARIANTS:
            out=root/f"rep{rep}"/v
            card=json.loads((out/"run_card.json").read_text())
            if card.get("run_mode")!="MEASURED_PROTOCOL" or card.get("profile_only") is not False:raise SystemExit(f"{out}: non-measured data")
            rec=[json.loads(x) for x in (out/"phase_steady_ops.jsonl").read_text().splitlines() if x.strip()]
            if len(rec)!=1024:raise SystemExit(f"{out}: invalid steady count")
            for ordinal,x in enumerate(rec):
                violations=capacity_snapshot_violations(x)
                if violations: raise SystemExit(f"{out}: invalid capacity snapshot at steady/{ordinal}: {violations}")
            wall=[float(x["end_to_end_wall_ms"]) for x in rec]; gpu=[float(x["gpu_event_ms"]) for x in rec]
            if not all(math.isfinite(x) and x>=0 for x in wall+gpu):raise SystemExit(f"{out}: nonfinite timing")
            phase=json.loads((out/"allocation_by_phase.json").read_text())["phases"]
            aggregate.setdefault(v,[]).append({"replicate":f"rep{rep}","steady_e2e_ms":stats(wall),"steady_gpu_event_ms":stats(gpu),"allocation_by_phase":phase})
    result={"schema":"gtspp-c1-v8-analysis-v1","generated_utc":datetime.now(timezone.utc).isoformat(),
      "scope":"Measured C1 SIFT1M query-only microbenchmark only; profile/NSight outputs excluded; no C2/C3/end-to-end claim.",
      "pins_sha256":sha(pins),"semantic_report":str(semantic),"semantic_report_sha256":sha(semantic),"variants":{},
      "warning":"Descriptive timing only after strict canonical/tree/CUDA/capacity/allocator/telemetry gate; do not generalize beyond this fixed protocol."}
    for v,reps in aggregate.items():
        med=[x["steady_e2e_ms"]["median"] for x in reps]
        result["variants"][v]={"replicate_median_e2e_ms":med,"bootstrap_mean_of_replicate_medians":ci(med),"per_replicate":reps}
    result["primary_paired_E_G_over_P_F_median_ratio"]=[{"replicate":names[i],"ratio":result["variants"]["E_G_c1_off_reference"]["replicate_median_e2e_ms"][i]/result["variants"]["P_F_full_C1"]["replicate_median_e2e_ms"][i] if result["variants"]["P_F_full_C1"]["replicate_median_e2e_ms"][i]>0 else None} for i in range(5)]
    out=root/"analysis_summary_v8.json";tmp=out.with_suffix(".tmp");tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");tmp.replace(out)
    print(json.dumps({"pass":True,"output":str(out),"replicates":5}))
    return 0
if __name__=="__main__":raise SystemExit(main())

