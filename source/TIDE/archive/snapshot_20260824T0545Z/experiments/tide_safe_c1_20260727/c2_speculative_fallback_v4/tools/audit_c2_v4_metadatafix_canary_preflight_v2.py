#!/usr/bin/env python3
"""CPU-only static preflight for the exact-binary metadatafix canary v2."""
import hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
PINS=ROOT/"provenance/c2_v4_metadatafix_canary_gpu_guard_pins_v2.json"
PARENT=ROOT/"canary_runs"
OUT=PARENT/"c2_v4_metadatafix_canary_v2"
def fail(x):raise SystemExit("FAIL_METADATAFIX_CANARY_PREFLIGHT: "+x)
def regular(p):
 p=pathlib.Path(p)
 if p.is_symlink() or not p.is_file():fail("not direct regular "+str(p))
 return p
def sha(p):
 h=hashlib.sha256()
 with open(regular(p),"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load(p):
 with open(regular(p),encoding="utf-8") as f:return json.load(f)
def ids(p,n,label):
 x=[int(v) for v in regular(p).read_text(encoding="ascii").splitlines() if v]
 if len(x)!=n or len(set(x))!=n or min(x,default=-1)<0 or max(x,default=12524)>=12524:fail(label+" IDs")
 return set(x)
pins=load(PINS)
if pins.get("schema")!="safe-c2-v4-metadatafix-canary-gpu-guard-pins-v2" or pins.get("status")!="PINNED_READY_FOR_METADATAFIX_CANARY_GPU_PREFLIGHT":fail("pins identity")
for name,record in pins.get("bindings",{}).items():
 if not isinstance(record,dict) or set(record)!={"path","sha256"} or sha(record["path"])!=record["sha256"]:fail("pin mismatch "+name)
protocol=load(pins["bindings"]["protocol"]["path"])
if protocol.get("schema")!="safe-c2-v4-metadatafix-canary-gpu-execution-protocol-v2" or protocol.get("status")!="FROZEN_PRE_METADATAFIX_CANARY_GPU_EXECUTION":fail("protocol identity")
if protocol.get("canonical_root")!=str(ROOT) or protocol.get("write_boundary")!=str(OUT):fail("protocol root/boundary")
if protocol.get("runtime")!={"mode":"calibrate","stage_argument":"calibration","warmup_reps":0,"timed_reps":1}:fail("runtime")
g=protocol.get("gpu",{})
if (g.get("physical_index"),g.get("cuda_visible_devices"),g.get("uuid"),g.get("pci_bus_id"))!=(0,"0","GPU-CONFIGURE-ARCHIVE-DEVICE","00000000:16:00.0"):fail("GPU contract")
if protocol.get("prior_canary_output_is_forbidden_input") is not True or protocol.get("formal_gamma_or_metric_use_forbidden") is not True:fail("isolation flags")
if not PARENT.is_dir() or PARENT.is_symlink() or PARENT.resolve()!=PARENT:fail("canary parent directness")
if OUT.exists() or OUT.is_symlink():fail("new canary output exists")
for p in (ROOT/"formal_runs_v2",ROOT/"runs",ROOT/".locks"):
 if p.exists() or p.is_symlink():fail("prohibited competing state "+str(p))
manifest=load(pins["bindings"]["canary_manifest"]["path"])
if manifest.get("schema")!="gts-v4-fresh-sift-learn-workload-v1" or manifest.get("status")!="READY_FOR_C2_V4_FORMAL_AFTER_PRELEDGER_PARSER_GATE":fail("canary manifest identity")
canary_path=ROOT/"inputs/canary_workload_v1/qualification_canary.ids"
c=ids(canary_path,1010,"canary")
for stage in ("calibration","validation","sealed_test"):
 if manifest.get(stage+"_ids_path")!=str(canary_path) or manifest.get(stage+"_ids_sha256")!=sha(canary_path) or manifest.get(stage+"_query_count")!=1010:fail("canary manifest stage adapter "+stage)
formal=load(ROOT/"inputs/final_workload_v2/workload_manifest.json")
for stage,n in (("calibration",2458),("validation",2463),("sealed_test",6403)):
 if c & ids(formal[stage+"_ids_path"],n,"formal "+stage):fail("canary/formal overlap "+stage)
src_audit=load(pins["bindings"]["source_audit_result"]["path"])
if src_audit.get("status")!="PASS_V4_METADATA_AND_PATH_SAFETY_SOURCE_AUDIT":fail("source audit status")
build=load(pins["bindings"]["build_card"]["path"])
if build.get("status")!="COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION":fail("build status")
receipt=load(pins["bindings"]["canary_preledger_receipt"]["path"])
if receipt.get("status")!="PASS_NO_CUDA_OR_DATASET_LOAD":fail("canary receipt status")
receipt_audit=load(pins["bindings"]["canary_preledger_audit"]["path"])
if receipt_audit.get("status")!="PASS_PRELEDGER_RECEIPT_CPU_ONLY":fail("canary receipt independent audit")
guard=pathlib.Path(pins["bindings"]["guard"]["path"])
if subprocess.run(["bash","-n",str(guard)]).returncode!=0:fail("guard syntax")
print("PASS_METADATAFIX_CANARY_CPU_ONLY_PREFLIGHT")
