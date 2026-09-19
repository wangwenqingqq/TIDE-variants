#!/usr/bin/env python3
"""Read-only independent audit of the corrected formal calibration v2 output."""
import hashlib,json,pathlib,sys
ROOT=pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
OUT=ROOT/"formal_runs_v2/c2_v4_calibration_v2"
def fail(x):raise SystemExit("FAIL_METADATAFIX_FORMAL_CALIBRATION_POST_AUDIT: "+x)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load(p):
 if not p.is_file() or p.is_symlink():fail("not direct file "+str(p))
 return json.load(open(p))
state=load(OUT/"run_state.json")
if state.get("status")!="COMPLETE_METADATAFIX_FORMAL_CALIBRATION_PASS" or state.get("performance_claim_eligible") is not False or state.get("failed_v1_output_is_not_an_input") is not True:fail("state")
summary=load(OUT/"summary.json")
if summary.get("schema")!="safe-c2-speculative-fallback-v4-run-v1" or summary.get("status")!="PASS_V4_CALIBRATION_GUARDED":fail("summary")
if summary.get("scope")!="Safe-C2 v4 speculative gamma traversal with gamma-only fallback on a strict fresh manifest-bound workload; calibration selects a fixed gamma vector only and held-out evaluation is separately gated":fail("scope")
if summary.get("implementation_identity")!="v4 speculative-fallback skeleton; not v2 and not the submitted legacy C2 kernel" or summary.get("v4_speculative_fallback") is not True or "v3_speculative_fallback" in summary:fail("identity")
wb=summary.get("workload_binding",{})
if wb.get("manifest")!=str(ROOT/"inputs/final_workload_v2/workload_manifest.json") or wb.get("schema")!="gts-v4-fresh-sift-learn-workload-v1" or not wb.get("sha256_verified_in_runner"):fail("workload")
if summary.get("input_paths",{}).get("stage_query_ids")!=str(ROOT/"inputs/final_workload_v2/calibration.ids"):fail("IDs")
if summary.get("timing_comparison",{}).get("eligible_for_speed_comparison") is not False:fail("speed eligibility")
gate=summary.get("final_guarded_per_query_no_regression_gate",{})
if not gate.get("passed") or gate.get("query_count")!=2458 or gate.get("max_violating_queries")!=0 or gate.get("repetitions_checked")!=1:fail("gate")
handoff=load(OUT/"calibration_handoff.json")
if handoff.get("status")!="PASS_METADATAFIX_FORMAL_CALIBRATION_FIXED_VECTOR_HANDOFF" or handoff.get("performance_claim_eligible") is not False or handoff.get("failed_v1_output_used_as_input") is not False:fail("handoff")
for label,path in (("summary",OUT/"summary.json"),("gamma_vector",OUT/"final_v4_speculative_gamma_vector.txt")):
 r=handoff.get(label,{})
 if r.get("path")!=str(path) or r.get("sha256")!=sha(path):fail("handoff "+label)
if handoff.get("calibration_ids",{}).get("count")!=2458:fail("handoff count")
art=OUT/"output_artifacts.sha256"
rows=[line.split("  ",1) for line in art.read_text().splitlines() if line]
if len(rows)<3 or len({name for _,name in rows})!=len(rows):fail("artifact list")
for digest,name in rows:
 p=OUT/name
 if not p.is_file() or p.is_symlink() or sha(p)!=digest:fail("artifact "+name)
if any(p.is_symlink() for p in OUT.iterdir()):fail("symlink output")
tele=(OUT/"gpu_telemetry.csv").read_text().splitlines()
if not any(",pre_launch,0, GPU-CONFIGURE-ARCHIVE-DEVICE, 00000000:16:00.0," in line for line in tele) or not any(",post_run,0, GPU-CONFIGURE-ARCHIVE-DEVICE, 00000000:16:00.0," in line for line in tele):fail("telemetry")
print(json.dumps({"schema":"safe-c2-v4-metadatafix-formal-calibration-result-independent-audit-v2","status":"PASS_METADATAFIX_FORMAL_CALIBRATION_OUTPUT_AUDIT","scope":"formal calibration fixed-vector handoff only; no heldout/test metric or performance claim","output_root":str(OUT),"binary_sha256":state["binary"]["sha256"],"manifest_sha256":state["manifest"]["sha256"],"artifact_records":len(rows),"gamma_vector_sha256":handoff["gamma_vector"]["sha256"],"gamma_vector":handoff["gamma_vector"]["values"],"final_guarded_calibration_gate":gate,"timing_comparison_eligible":False,"failed_v1_output_used_as_input":False,"performance_claim_eligible":False},sort_keys=True))
