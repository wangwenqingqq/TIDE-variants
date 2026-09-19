#!/usr/bin/env python3
"""CPU-only static audit for the telemetry-first held-out performance wrapper."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
RUNNER=ROOT/"tools/run_static_aabb_heldout_perf_evidence_v2.py"
FREEZE=ROOT/"provenance/static_aabb_heldout_perf_evidence_runner_freeze_v2.json"
PREFLIGHT=ROOT/"provenance/static_aabb_heldout_perf_evidence_runner_preflight_v2.json"
PROTOCOL=ROOT/"provenance/static_aabb_heldout_perf_evidence_protocol_v1.json"
EXPECTED={
 "runner":"3d604e91afac4dd611ccb5341c30c21a69d5297c7ef0be96f9262d666e7810c1",
 "freeze":"eb76a0fc07f033252b36500a7cd7f7686fe1584583cbc7494f37dc1d227d3fbd",
 "protocol":"b52bbf64087f6509546e52d9970c1025bf5f82f1a5120f3c2ef6e2b2b7397386",
}
def sha(p:Path)->str:
 if not p.is_file() or p.is_symlink():
  raise RuntimeError("invalid "+str(p))
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):
   h.update(b)
 return h.hexdigest()
def main()->int:
 got={k:sha(v) for k,v in {"runner":RUNNER,"freeze":FREEZE,"protocol":PROTOCOL}.items()}
 for k,v in EXPECTED.items():
  if got[k]!=v:
   raise RuntimeError("hash "+k)
 s=RUNNER.read_text()
 required=[
  "--preflight-only","--allow-unfrozen-audit","CUDA_VISIBLE_DEVICES",
  "GPU_UUID","gpu_telemetry.csv","gpu_process_samples.jsonl",
  "gpu_state_before.xml","gpu_state_after.xml","environment.json",
  "runner_freeze_reference.json","launch.json","timing_pairs.json",
  "telemetry_summary.json","COMPLETE_CONDITIONAL_DVFS_UNCONTROLLED",
  "EXPLORATORY_TELEMETRY_INSUFFICIENT","INVALID_FOREIGN_COMPUTE_INTERFERENCE",
  "target_sample_times","max_inter_sample_gap_ms","target",
  "\"warmups\":WARMUPS","--reps\",str(REPS)",
  "runner_attempted_clock_lock\":False","runner_attempted_application_clocks\":False",
  "runner_attempted_power_limit_change\":False","runner_attempted_compute_mode_change\":False",
  "exit_code\":0",
 ]
 for token in required:
  if token not in s:
   raise RuntimeError("missing runner token "+token)
 for forbidden in ("--lock-gpu-clocks","-lgc","-rgc","os.kill(","pkill","killall","c2_speculative_fallback_v4/validation","c2_speculative_fallback_v4/sealed"):
  if forbidden in s:
   raise RuntimeError("forbidden runner token "+forbidden)
 preflight_branch=s.find("if args.preflight_only:")
 gpu_branch=s.find("nv=nvsmi()")
 if preflight_branch<0 or gpu_branch<0 or preflight_branch>gpu_branch:
  raise RuntimeError("preflight-only GPU boundary")
 freeze=json.loads(FREEZE.read_text())
 if freeze.get("schema")!="safe-c2-static-aabb-heldout-perf-runner-freeze-v2" or freeze.get("status")!="FROZEN_CPU_ONLY":
  raise RuntimeError("freeze state")
 if freeze.get("runner",{}).get("sha256")!=got["runner"] or freeze.get("protocol_sha256")!=EXPECTED["protocol"]:
  raise RuntimeError("freeze bindings")
 pre=json.loads(PREFLIGHT.read_text())
 if pre.get("status")!="PASS_CPU_ONLY_READY_FOR_GPU_GUARD" or pre.get("nvidia_smi_called") is not False:
  raise RuntimeError("runner preflight")
 protocol=json.loads(PROTOCOL.read_text())
 req=protocol.get("required_run_artifacts",[])
 for artifact in ["preflight.json","runner_freeze_reference.json","launch.json","environment.json","gpu_state_before.xml","gpu_state_after.xml","gpu_telemetry.csv","gpu_process_samples.jsonl","telemetry_summary.json","timing_pairs.json","stdout.log","stderr.log","result.json","snapshot_sha256.txt","terminal.json","manifest.json"]:
  if artifact not in req:
   raise RuntimeError("protocol artifact "+artifact)
 out={
  "schema":"safe-c2-static-aabb-heldout-perf-runner-audit-v2",
  "status":"PASS_CPU_ONLY_RUNNER_AUDITED",
  "scope":"static wrapper audit only; no nvidia-smi or CUDA binary execution",
  "runner_sha256":got["runner"],"freeze_sha256":got["freeze"],"protocol_sha256":got["protocol"],
  "formal_claim_eligible":False,"gpu_binary_executed":False,"nvidia_smi_called":False,
  "evidence_ceiling":"CONDITIONAL_PAIRED_DVFS_UNCONTROLLED only; no hardware-controlled or stable absolute-latency claim",
 }
 print(json.dumps(out,sort_keys=True))
 return 0
if __name__=="__main__":
 try:
  raise SystemExit(main())
 except Exception as e:
  print("FAIL_STATIC_AABB_HELDOUT_PERF_RUNNER_AUDIT_V2: "+str(e),file=sys.stderr)
  raise SystemExit(2)
