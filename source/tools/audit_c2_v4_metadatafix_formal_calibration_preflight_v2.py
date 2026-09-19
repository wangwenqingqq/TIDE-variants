#!/usr/bin/env python3
"""CPU-only fail-closed preflight for one replacement Safe-C2 v4 formal calibration."""
import hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
PINS=ROOT/"provenance/c2_v4_metadatafix_formal_calibration_gpu_guard_pins_v2.json"
PARENT=ROOT/"formal_runs_v2"
OUT=PARENT/"c2_v4_calibration_v2"
def fail(x):raise SystemExit("FAIL_METADATAFIX_FORMAL_CALIBRATION_PREFLIGHT: "+x)
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
 vals=[int(v) for v in reg(p).read_text(encoding="ascii").splitlines() if v]
 if len(vals)!=n or len(set(vals))!=n or min(vals,default=-1)<0 or max(vals,default=12524)>=12524:fail(label+" IDs")
 return set(vals)
pins=load(PINS)
if pins.get("schema")!="safe-c2-v4-metadatafix-formal-calibration-gpu-guard-pins-v2" or pins.get("status")!="PINNED_READY_FOR_METADATAFIX_FORMAL_CALIBRATION_GPU_PREFLIGHT":fail("pins")
for name,r in pins.get("bindings",{}).items():
 if not isinstance(r,dict) or set(r)!={"path","sha256"} or sha(r["path"])!=r["sha256"]:fail("pin "+name)
proto=load(pins["bindings"]["protocol"]["path"])
if proto.get("schema")!="safe-c2-v4-metadatafix-formal-calibration-execution-protocol-v2" or proto.get("status")!="FROZEN_PRE_METADATAFIX_FORMAL_CALIBRATION_GPU_EXECUTION":fail("protocol")
if proto.get("canonical_root")!=str(ROOT) or proto.get("write_boundary")!=str(OUT):fail("protocol boundary")
if proto.get("runtime")!={"mode":"calibrate","stage_argument":"calibration","warmup_reps":0,"timed_reps":1}:fail("runtime")
g=proto.get("gpu",{})
if (g.get("physical_index"),g.get("cuda_visible_devices"),g.get("uuid"),g.get("pci_bus_id"))!=(0,"0","GPU-CONFIGURE-ARCHIVE-DEVICE","00000000:16:00.0"):fail("GPU")
for p in (PARENT,OUT,ROOT/"runs",ROOT/".locks"):
 if p.exists() or p.is_symlink():fail("fresh output/lock boundary "+str(p))
manifest=load(pins["bindings"]["workload_manifest"]["path"])
if manifest.get("schema")!="gts-v4-fresh-sift-learn-workload-v1" or manifest.get("status")!="READY_FOR_C2_V4_FORMAL_AFTER_PRELEDGER_PARSER_GATE":fail("manifest")
splits={}
for stage,n in (("calibration",2458),("validation",2463),("sealed_test",6403)):
 p=manifest[stage+"_ids_path"]
 if sha(p)!=manifest[stage+"_ids_sha256"]:fail(stage+" checksum")
 splits[stage]=ids(p,n,stage)
if splits["calibration"]&splits["validation"] or splits["calibration"]&splits["sealed_test"] or splits["validation"]&splits["sealed_test"]:fail("formal split overlap")
canary=ids(ROOT/"inputs/canary_workload_v1/qualification_canary.ids",1010,"canary")
if any(canary&s for s in splits.values()):fail("canary/formal overlap")
if proto.get("calibration_ids")!={"path":manifest["calibration_ids_path"],"sha256":manifest["calibration_ids_sha256"],"count":2458}:fail("protocol calibration binding")
for name,expect in (("source_audit_result","PASS_V4_METADATA_AND_PATH_SAFETY_SOURCE_AUDIT"),("build_card","COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION"),("canary_audit","PASS_METADATAFIX_CANARY_QUALIFICATION_ONLY_OUTPUT_AUDIT"),("failed_v1_boundary","FAILED_RUN_FROZEN_NOT_USABLE")):
 x=load(pins["bindings"][name]["path"])
 if x.get("status")!=expect:fail(name+" status")
receipt=load(pins["bindings"]["preledger_receipt"]["path"])
if receipt.get("status")!="PASS_NO_CUDA_OR_DATASET_LOAD":fail("receipt")
receipt_audit=load(pins["bindings"]["preledger_audit"]["path"])
if receipt_audit.get("status")!="PASS_PRELEDGER_RECEIPT_CPU_ONLY":fail("receipt audit")
binary=pins["bindings"]["binary"]
build=load(pins["bindings"]["build_card"]["path"])
def same_binary(record,label):
 if not isinstance(record,dict) or record.get("path")!=binary["path"] or record.get("sha256")!=binary["sha256"]:
  fail(label+" binary binding")
same_binary(build.get("binary"),"build")
same_binary(receipt.get("binary"),"receipt")
if receipt_audit.get("binary_sha256")!=binary["sha256"]:fail("receipt audit/binary binding")
same_binary(proto.get("binary"),"protocol")
guard=pathlib.Path(pins["bindings"]["guard"]["path"])
if subprocess.run(["bash","-n",str(guard)]).returncode!=0:fail("guard syntax")
print("PASS_METADATAFIX_FORMAL_CALIBRATION_CPU_ONLY_PREFLIGHT")
