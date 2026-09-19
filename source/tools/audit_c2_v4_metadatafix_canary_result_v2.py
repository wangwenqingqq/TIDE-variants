#!/usr/bin/env python3
"""Read-only independent audit of the metadatafix qualification canary output."""
import hashlib,json,pathlib,sys
ROOT=pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
OUT=ROOT/"canary_runs/c2_v4_metadatafix_canary_v2"
def fail(x):raise SystemExit("FAIL_METADATAFIX_CANARY_POST_AUDIT: "+x)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load(p):
 if not p.is_file() or p.is_symlink():fail("not direct file "+str(p))
 return json.load(open(p))
state=load(OUT/"run_state.json")
if state.get("status")!="COMPLETE_METADATAFIX_CANARY_QUALIFICATION_ONLY_PASS" or state.get("formal_claim_eligible") is not False:fail("state")
summary=load(OUT/"summary.json")
if summary.get("schema")!="safe-c2-speculative-fallback-v4-run-v1" or summary.get("status")!="PASS_V4_CALIBRATION_GUARDED":fail("summary")
if summary.get("implementation_identity")!="v4 speculative-fallback skeleton; not v2 and not the submitted legacy C2 kernel" or summary.get("v4_speculative_fallback") is not True or "v3_speculative_fallback" in summary:fail("identity")
if summary.get("scope")!="Safe-C2 v4 speculative gamma traversal with gamma-only fallback on a strict fresh manifest-bound workload; calibration selects a fixed gamma vector only and held-out evaluation is separately gated":fail("scope")
wb=summary.get("workload_binding",{})
if wb.get("schema")!="gts-v4-fresh-sift-learn-workload-v1" or not wb.get("sha256_verified_in_runner"):fail("workload")
if summary.get("input_paths",{}).get("stage_query_ids")!=str(ROOT/"inputs/canary_workload_v1/qualification_canary.ids"):fail("ids")
if summary.get("timing_comparison",{}).get("eligible_for_speed_comparison") is not False:fail("speed eligibility")
gate=summary.get("final_guarded_per_query_no_regression_gate",{})
if not gate.get("passed") or gate.get("query_count")!=1010 or gate.get("max_violating_queries")!=0:fail("gate")
art=OUT/"output_artifacts.sha256"
if not art.is_file() or art.is_symlink():fail("artifacts")
rows=[line.split("  ",1) for line in art.read_text().splitlines() if line]
if not rows or len({n for _,n in rows})!=len(rows):fail("artifact records")
for digest,name in rows:
 p=OUT/name
 if not p.is_file() or p.is_symlink() or sha(p)!=digest:fail("artifact hash "+name)
if any(p.is_symlink() for p in OUT.iterdir()):fail("symlink output")
tele=(OUT/"gpu_telemetry.csv").read_text().splitlines()
if len(tele)<4 or not any(",pre_launch,0, GPU-CONFIGURE-ARCHIVE-DEVICE, 00000000:16:00.0," in x for x in tele) or not any(",post_run,0, GPU-CONFIGURE-ARCHIVE-DEVICE, 00000000:16:00.0," in x for x in tele):fail("telemetry identity")
print(json.dumps({"schema":"safe-c2-v4-metadatafix-canary-result-independent-audit-v2","status":"PASS_METADATAFIX_CANARY_QUALIFICATION_ONLY_OUTPUT_AUDIT","scope":"exact-binary qualification only; output gamma/metrics are explicitly excluded from formal work","output_root":str(OUT),"binary_sha256":state["binary"]["sha256"],"manifest_sha256":state["manifest"]["sha256"],"artifact_records":len(rows),"final_guarded_gate":gate,"timing_comparison_eligible":False,"formal_claim_eligible":False},sort_keys=True))
