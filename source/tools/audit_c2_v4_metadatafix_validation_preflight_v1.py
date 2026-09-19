#!/usr/bin/env python3
"""CPU-only preflight for fixed-vector Safe-C2 v4 heldout validation."""
import hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
PINS=ROOT/"provenance/c2_v4_metadatafix_validation_gpu_guard_pins_v1.json"
PARENT=ROOT/"formal_runs_v2";OUT=PARENT/"c2_v4_validation_v1"
def fail(x):raise SystemExit("FAIL_METADATAFIX_VALIDATION_PREFLIGHT: "+x)
def reg(p):
 p=pathlib.Path(p)
 if p.is_symlink() or not p.is_file():fail("not direct regular "+str(p))
 return p
def sha(p):
 h=hashlib.sha256()
 with open(reg(p),"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load(p):
 with open(reg(p),encoding="utf-8") as f:return json.load(f)
def ids(p,n,label):
 a=[int(x) for x in reg(p).read_text(encoding="ascii").splitlines() if x]
 if len(a)!=n or len(set(a))!=n or min(a,default=-1)<0 or max(a,default=12524)>=12524:fail(label+" IDs")
 return set(a)
pins=load(PINS)
if pins.get("schema")!="safe-c2-v4-metadatafix-validation-gpu-guard-pins-v1" or pins.get("status")!="PINNED_READY_FOR_METADATAFIX_VALIDATION_GPU_PREFLIGHT":fail("pins")
for name,r in pins.get("bindings",{}).items():
 if not isinstance(r,dict) or set(r)!={"path","sha256"} or sha(r["path"])!=r["sha256"]:fail("pin "+name)
proto=load(pins["bindings"]["protocol"]["path"])
if proto.get("schema")!="safe-c2-v4-metadatafix-validation-execution-protocol-v1" or proto.get("status")!="FROZEN_PRE_METADATAFIX_VALIDATION_GPU_EXECUTION":fail("protocol")
if proto.get("canonical_root")!=str(ROOT) or proto.get("write_boundary")!=str(OUT):fail("boundary")
if proto.get("runtime")!={"mode":"evaluate","stage_argument":"validation","warmup_reps":1,"timed_reps":7}:fail("runtime")
g=proto.get("gpu",{})
if (g.get("physical_index"),g.get("cuda_visible_devices"),g.get("uuid"),g.get("pci_bus_id"))!=(0,"0","GPU-CONFIGURE-ARCHIVE-DEVICE","00000000:16:00.0"):fail("GPU")
if not PARENT.is_dir() or PARENT.is_symlink() or PARENT.resolve()!=PARENT:fail("formal parent")
if OUT.exists() or OUT.is_symlink() or (ROOT/"runs").exists() or (ROOT/".locks").exists():fail("output/lock exists")
manifest=load(pins["bindings"]["workload_manifest"]["path"])
if manifest.get("schema")!="gts-v4-fresh-sift-learn-workload-v1":fail("manifest schema")
splits={}
for stage,n in (("calibration",2458),("validation",2463),("sealed_test",6403)):
 if sha(manifest[stage+"_ids_path"])!=manifest[stage+"_ids_sha256"]:fail(stage+" checksum")
 splits[stage]=ids(manifest[stage+"_ids_path"],n,stage)
if splits["calibration"]&splits["validation"] or splits["validation"]&splits["sealed_test"] or splits["calibration"]&splits["sealed_test"]:fail("formal split overlap")
canary=ids(ROOT/"inputs/canary_workload_v1/qualification_canary.ids",1010,"canary")
if any(canary&s for s in splits.values()):fail("canary overlap")
if proto.get("validation_ids")!={"path":manifest["validation_ids_path"],"sha256":manifest["validation_ids_sha256"],"count":2463}:fail("validation binding")
handoff=load(pins["bindings"]["calibration_handoff"]["path"])
if handoff.get("status")!="PASS_METADATAFIX_FORMAL_CALIBRATION_FIXED_VECTOR_HANDOFF" or handoff.get("performance_claim_eligible") is not False or handoff.get("failed_v1_output_used_as_input") is not False:fail("calibration handoff")
gamma=handoff.get("gamma_vector",{})
if gamma.get("path")!=proto.get("gamma_vector",{}).get("path") or gamma.get("sha256")!=proto.get("gamma_vector",{}).get("sha256"):fail("gamma protocol")
if sha(gamma["path"])!=gamma["sha256"]:fail("gamma hash")
cal_audit=load(pins["bindings"]["calibration_audit"]["path"])
if cal_audit.get("status")!="PASS_METADATAFIX_FORMAL_CALIBRATION_OUTPUT_AUDIT" or cal_audit.get("binary_sha256")!=pins["bindings"]["binary"]["sha256"]:fail("calibration audit")
for name,status in (("source_audit_result","PASS_V4_METADATA_AND_PATH_SAFETY_SOURCE_AUDIT"),("build_card","COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION"),("canary_audit","PASS_METADATAFIX_CANARY_QUALIFICATION_ONLY_OUTPUT_AUDIT"),("preledger_audit","PASS_PRELEDGER_RECEIPT_CPU_ONLY")):
 x=load(pins["bindings"][name]["path"])
 if x.get("status")!=status:fail(name+" status")
if load(pins["bindings"]["canary_audit"]["path"]).get("binary_sha256")!=pins["bindings"]["binary"]["sha256"]:fail("canary audit binary")
receipt=load(pins["bindings"]["preledger_receipt"]["path"])
binary=pins["bindings"]["binary"]
for r,label in ((load(pins["bindings"]["build_card"]["path"]).get("binary"),"build"),(receipt.get("binary"),"receipt"),(proto.get("binary"),"protocol")):
 if not isinstance(r,dict) or r.get("path")!=binary["path"] or r.get("sha256")!=binary["sha256"]:fail(label+" binary")
expected_gate={"minimum_final_guarded_speedup":1.05,"metric":"final guarded query_pipeline_wall_ms only; raw speculative timing is diagnostic","requires_validation_gate":True,"default":"sealed test remains locked unless an independent postrun decision records PASS"}
if proto.get("sealed_test_gate")!=expected_gate:fail("test gate precommit")
decision_binding=proto.get("sealed_test_decision_protocol")
if not isinstance(decision_binding,dict) or set(decision_binding)!={"path","sha256"} or sha(decision_binding["path"])!=decision_binding["sha256"]:fail("decision protocol binding")
decision=load(decision_binding["path"])
formula={"numerator":"paired_same_split_baseline.guarded_timing.query_pipeline_wall_ms_sum","denominator":"timing.stage.guarded_timing.query_pipeline_wall_ms_sum","aggregation":"aggregate sum over identical validation queries and timed ABBA repetitions","raw_speculative_timing_forbidden":True,"requires":{"query_count":2463,"timed_reps":7,"shared_warmup_pairs":1,"final_guarded_gate_pass":True,"denominator_positive_finite":True},"pass_condition":"speedup >= 1.05","on_failure":"sealed test remains locked; no retry"}
if decision.get("schema")!="safe-c2-v4-metadatafix-validation-sealed-decision-protocol-v1" or decision.get("status")!="FROZEN_BEFORE_METADATAFIX_VALIDATION_GPU_EXECUTION" or decision.get("formula")!=formula:fail("decision protocol contract")
tool=decision.get("tool")
if not isinstance(tool,dict) or set(tool)!={"path","sha256"} or sha(tool["path"])!=tool["sha256"]:fail("decision tool binding")
if pathlib.Path(decision.get("receipt_path","")).exists() or pathlib.Path(decision.get("receipt_path","")).is_symlink():fail("sealed decision receipt exists before validation")
if subprocess.run([sys.executable,tool["path"],"--self-test"]).returncode!=0:fail("decision tool self test")
if subprocess.run(["bash","-n",pins["bindings"]["guard"]["path"]]).returncode!=0:fail("guard syntax")
print("PASS_METADATAFIX_VALIDATION_CPU_ONLY_PREFLIGHT")
