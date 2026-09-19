#!/usr/bin/env python3
"""
Telemetry-first successor for the frozen SIFT static-AABB held-out paired performance protocol.

Default execution is one five-process batch (5 warmups + 30 alternating paired reps each).
It never changes GPU clocks/power/compute mode, never kills a process, and fails closed if
GPU telemetry/process coverage is insufficient. --preflight-only is CPU-only.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, json, os
from pathlib import Path
import secrets, shutil, statistics, subprocess, sys, threading, time, traceback
from typing import Any

ROOT=Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
RUNS=ROOT/"runs"
BINARY=ROOT/"bin/static_aabb_perf_probe_v2_1_diskguard"
SOURCE=ROOT/"src/static_aabb_perf_probe_v2_1_diskguard.cu"
IDS=ROOT/"inputs/standard_sift_static_aabb_heldout256_v1.ids"
BASE=Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs")
QUERY=Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs")
GT=Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs")
PROTOCOL=ROOT/"provenance/static_aabb_heldout_perf_evidence_protocol_v1.json"
PARENT_PROTOCOL=ROOT/"provenance/static_aabb_heldout_protocol_v1.json"
PARENT_PREFLIGHT=ROOT/"provenance/static_aabb_heldout_preflight_v1.json"
SOURCE_AUDIT=ROOT/"tools/audit_diskguard_v2_1_sources.py"
RUNNER_FREEZE=ROOT/"provenance/static_aabb_heldout_perf_evidence_runner_freeze_v1.json"
GPU_INDEX=0
GPU_UUID="GPU-CONFIGURE-ARCHIVE-DEVICE"
WARMUPS=5
REPS=30
INVOCATIONS=5
SAMPLE_PERIOD_S=0.100
MAX_GAP_MS=500.0
EXPECTED={
 "binary":"e587ad822de007880e1caae8ea89072c07a5b786dcb5d15357ecde61ce577cde",
 "source":"5dc92a9660bc168e1325edae7a1c9d41f59a93e036e83f57f2ccd9d55fa6819f",
 "ids":"fdfce62ae706d76f4c3a87b6dc199ca0b7524da821ebae46516a99fb99140932",
 "base":"21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816",
 "query":"f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc",
 "gt":"2b71de0a8d5a83e6a84eec3e23fb8b611d8801dd9b3a6cd62f070ab65ea65f4f",
 "protocol":"b52bbf64087f6509546e52d9970c1025bf5f82f1a5120f3c2ef6e2b2b7397386",
 "parent_protocol":"3621befa31a0c6e9a3b424cc6429f848bc6b45f1704f7047c8992b76ca1214a1",
 "parent_preflight":"d309ee0195a816d97f216481fa5d940f8bc85ac1b7a5387540d09128e1c60ca7",
 "source_audit":"eca5d8932fe0dd14a91778be87a13f961852d297074434fba2767ad071affa7a",
}
GPU_FIELDS=["index","uuid","pci.bus_id","name","driver_version","compute_mode","persistence_mode","power.limit","pstate","clocks.sm","clocks.current.graphics","clocks.current.memory","power.draw","temperature.gpu","utilization.gpu","utilization.memory","memory.used"]
CSV_HEADER=["wall_utc_ns","monotonic_ns","sample_seq","phase","gpu_index","gpu_uuid","pstate","clock_sm_mhz","clock_graphics_mhz","clock_memory_mhz","power_draw_w","power_limit_w","temperature_gpu_c","utilization_gpu_pct","utilization_memory_pct","memory_used_mib"]

def utc() -> str:
 return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
def sha(path:Path)->str:
 if not path.is_file() or path.is_symlink(): raise RuntimeError("invalid regular file: "+str(path))
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def art(path:Path)->dict[str,Any]:
 return {"path":str(path),"bytes":path.stat().st_size,"sha256":sha(path)}
def atomic(path:Path,obj:dict[str,Any])->None:
 tmp=path.with_name(path.name+".tmp")
 tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
 os.replace(tmp,path)
def run_capture(cmd:list[str],path:Path,check:bool=True)->subprocess.CompletedProcess[str]:
 p=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 path.write_text(p.stdout)
 if check and p.returncode: raise RuntimeError("command failed: "+" ".join(cmd))
 return p
def nvsmi()->str:
 x=shutil.which("nvidia-smi")
 if not x: raise RuntimeError("nvidia-smi unavailable")
 return x
def physical_gpu_info(nv:str)->dict[str,str]:
 q=[nv,"--query-gpu=index,uuid,pci.bus_id,name,driver_version,compute_mode,persistence_mode,power.limit","--format=csv,noheader,nounits"]
 p=subprocess.run(q,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 if p.returncode: raise RuntimeError("physical GPU query failed")
 for line in p.stdout.splitlines():
  vals=[x.strip() for x in next(csv.reader([line]))]
  if vals and vals[0]==str(GPU_INDEX):
   if len(vals)!=8 or vals[1]!=GPU_UUID: raise RuntimeError("physical GPU identity")
   return {"index":vals[0],"uuid":vals[1],"pci_bus_id":vals[2],"name":vals[3],"driver_version":vals[4],"compute_mode":vals[5],"persistence_mode":vals[6],"power_limit_w":vals[7]}
 raise RuntimeError("physical GPU row absent")
def fnum(value:str)->float|None:
 try:return float(value.strip().replace("W","").replace("MiB","").replace("MHz",""))
 except Exception:return None

class Collector:
 def __init__(self,nv:str,out:Path):
  self.nv,self.out=nv,out
  self.stop_event=threading.Event();self.lock=threading.Lock()
  self.phase="prelaunch";self.target_pid: int|None=None;self.seq=0
  self.samples:list[dict[str,Any]]=[];self.foreign:set[int]=set();self.target_seen:set[int]=set();self.target_sample_times:dict[int,list[int]]={};self.errors:list[str]=[]
  self.csv_path=out/"gpu_telemetry.csv";self.proc_path=out/"gpu_process_samples.jsonl"
  self.thread=threading.Thread(target=self.loop,daemon=True)
 def set_target(self,pid:int|None,phase:str)->None:
  with self.lock:self.target_pid=pid;self.phase=phase
 def start(self)->None:self.thread.start()
 def stop(self)->None:
  self.stop_event.set();self.thread.join(timeout=15)
  if self.thread.is_alive():raise RuntimeError("telemetry collector failed to stop")
 def current(self)->tuple[int|None,str]:
  with self.lock:return self.target_pid,self.phase
 def sample(self,cw:csv.writer,pf)->None:
  target,phase=self.current();mono=time.monotonic_ns();wall=time.time_ns()
  q=[self.nv,"--query-gpu="+",".join(GPU_FIELDS),"--format=csv,noheader,nounits"]
  p=subprocess.run(q,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  lines=[x.strip() for x in p.stdout.splitlines() if x.strip()]
  row=None
  for line in lines:
   vals=next(csv.reader([line]))
   if vals and vals[0].strip()==str(GPU_INDEX):row=[x.strip() for x in vals];break
  if p.returncode or row is None or len(row)!=len(GPU_FIELDS) or row[1]!=GPU_UUID:
   self.errors.append("gpu query invalid at sample "+str(self.seq))
   row=[""]*len(GPU_FIELDS);row[0]=str(GPU_INDEX);row[1]=GPU_UUID
  ap=subprocess.run([self.nv,"--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  raw=ap.stdout
  procs=[]
  if ap.returncode:self.errors.append("process query failed at sample "+str(self.seq))
  for line in raw.splitlines():
   if not line.strip():continue
   vals=next(csv.reader([line]))
   if len(vals)>=2 and vals[0].strip()==GPU_UUID:
    try:procs.append({"pid":int(vals[1].strip()),"process_name":vals[2].strip() if len(vals)>2 else "","used_memory":vals[3].strip() if len(vals)>3 else ""})
    except ValueError:self.errors.append("bad process row at sample "+str(self.seq))
  pids={x["pid"] for x in procs}
  if target is None:self.foreign.update(pids)
  else:
   if target in pids:
    self.target_seen.add(target)
    self.target_sample_times.setdefault(target,[]).append(mono)
   self.foreign.update(pids-{target})
  values={"wall_utc_ns":wall,"monotonic_ns":mono,"sample_seq":self.seq,"phase":phase,"gpu_index":row[0],"gpu_uuid":row[1],"pstate":row[8],"clock_sm_mhz":row[9],"clock_graphics_mhz":row[10],"clock_memory_mhz":row[11],"power_draw_w":row[12],"power_limit_w":row[7],"temperature_gpu_c":row[13],"utilization_gpu_pct":row[14],"utilization_memory_pct":row[15],"memory_used_mib":row[16]}
  cw.writerow([values[k] for k in CSV_HEADER]);pf.write(json.dumps({"wall_utc_ns":wall,"monotonic_ns":mono,"sample_seq":self.seq,"phase":phase,"gpu_uuid":GPU_UUID,"target_pid":target,"compute_processes":procs,"raw_query_output":raw},sort_keys=True)+"\n");pf.flush()
  self.samples.append(values);self.seq+=1
 def loop(self)->None:
  with self.csv_path.open("w",newline="") as f,self.proc_path.open("w") as pf:
   cw=csv.writer(f);cw.writerow(CSV_HEADER);f.flush()
   next_t=time.monotonic()
   while not self.stop_event.is_set():
    self.sample(cw,pf);f.flush()
    next_t+=SAMPLE_PERIOD_S
    self.stop_event.wait(max(0.0,next_t-time.monotonic()))
 def wait_samples(self,n:int,timeout:float=15.0)->None:
  end=time.monotonic()+timeout
  while time.monotonic()<end:
   if len(self.samples)>=n:return
   time.sleep(.02)
  raise RuntimeError("telemetry startup samples timeout")
 def summary(self)->dict[str,Any]:
  if not self.samples:raise RuntimeError("no telemetry samples")
  ms=[x["monotonic_ns"] for x in self.samples]
  gaps=[(b-a)/1e6 for a,b in zip(ms,ms[1:])]
  numeric=["clock_sm_mhz","clock_graphics_mhz","clock_memory_mhz","power_draw_w","power_limit_w","temperature_gpu_c","utilization_gpu_pct","utilization_memory_pct","memory_used_mib"]
  stats={}
  complete=True
  for k in numeric:
   vals=[fnum(str(x[k])) for x in self.samples]
   if any(v is None for v in vals):complete=False
   vv=[v for v in vals if v is not None]
   stats[k]={"min":min(vv) if vv else None,"max":max(vv) if vv else None,"median":statistics.median(vv) if vv else None}
  return {"sample_count":len(self.samples),"first_monotonic_ns":ms[0],"last_monotonic_ns":ms[-1],"max_inter_sample_gap_ms":max(gaps) if gaps else 0.0,"pstates":sorted({x["pstate"] for x in self.samples if x["pstate"]}),"observed_target_pids":sorted(self.target_seen),"target_sample_times":{str(k):v for k,v in self.target_sample_times.items()},"foreign_compute_pids":sorted(self.foreign),"query_errors":self.errors,"numeric_field_complete":complete,"stats":stats}

def exact_ids()->list[int]:
 x=[int(v) for v in IDS.read_text().splitlines()]
 if len(x)!=256 or x!=sorted(x) or len(set(x))!=256 or x[0]!=5000 or x[-1]!=5259:raise RuntimeError("frozen heldout256 IDs")
 return x
def cpu_preflight(require_freeze:bool)->dict[str,Any]:
 paths={"binary":BINARY,"source":SOURCE,"ids":IDS,"base":BASE,"query":QUERY,"gt":GT,"protocol":PROTOCOL,"parent_protocol":PARENT_PROTOCOL,"parent_preflight":PARENT_PREFLIGHT,"source_audit":SOURCE_AUDIT}
 out={k:art(v) for k,v in paths.items()}
 for k,v in EXPECTED.items():
  if out[k]["sha256"]!=v:raise RuntimeError("SHA mismatch: "+k)
 p=json.loads(PROTOCOL.read_text())
 if p.get("status")!="FROZEN_CPU_ONLY_PREPARED":raise RuntimeError("perf protocol state")
 if p["heldout_binding"]["ids_file"]["sha256"]!=EXPECTED["ids"]:raise RuntimeError("protocol IDs")
 parent=json.loads(PARENT_PROTOCOL.read_text())
 if parent.get("status")!="FROZEN_CPU_ONLY_PREPARED":raise RuntimeError("parent protocol")
 old=json.loads(PARENT_PREFLIGHT.read_text())
 if old.get("status")!="PASS_CPU_ONLY_FROZEN_EXECUTION_PENDING":raise RuntimeError("parent preflight")
 audit=subprocess.run([str(SOURCE_AUDIT)],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if audit.returncode:raise RuntimeError("source audit failed: "+audit.stderr.strip())
 auditj=json.loads(audit.stdout)
 if auditj.get("status")!="PASS":raise RuntimeError("source audit status")
 qids=exact_ids()
 freeze_info=None
 if require_freeze:
  if not RUNNER_FREEZE.is_file() or RUNNER_FREEZE.is_symlink():raise RuntimeError("runner freeze absent")
  freeze=json.loads(RUNNER_FREEZE.read_text())
  if freeze.get("status")!="FROZEN_CPU_ONLY":raise RuntimeError("runner freeze state")
  if freeze.get("runner",{}).get("sha256")!=sha(Path(__file__).resolve()):raise RuntimeError("runner freeze SHA")
  if freeze.get("protocol_sha256")!=EXPECTED["protocol"] or freeze.get("protocol",{}).get("sha256")!=EXPECTED["protocol"]:raise RuntimeError("runner freeze protocol")
  freeze_info=art(RUNNER_FREEZE)
 return {"schema":"safe-c2-static-aabb-heldout-perf-runner-preflight-v1","status":"PASS_CPU_ONLY_READY_FOR_GPU_GUARD" if require_freeze else "PASS_CPU_ONLY_UNFROZEN_RUNNER_AUDIT","scope":"artifact/input audit only; no nvidia-smi or CUDA binary execution","nvidia_smi_called":False,"gpu_binary_executed":False,"formal_claim_eligible":False,"target_gpu":{"index":GPU_INDEX,"uuid":GPU_UUID},"measurement":{"warmups":WARMUPS,"paired_repetitions":REPS,"independent_process_invocations":INVOCATIONS},"artifacts":out,"runner":art(Path(__file__).resolve()),"runner_freeze":freeze_info,"source_audit":auditj,"heldout_ids":{"count":len(qids),"sha256":EXPECTED["ids"]}}

def valid_child(child:Path)->tuple[dict[str,Any],list[dict[str,float]]]:
 r=json.loads((child/"result.json").read_text())
 if r.get("status")!="PASS_EXPLORATORY_PAIRED_PROBE_V2_DISKGUARD":raise RuntimeError("child perf status")
 gate=r.get("unmeasured_gate",{})
 for m in ("baseline","static_aabb"):
  x=gate.get(m,{})
  if any(x.get(k)!=0 for k in ("invalid","duplicate","gt_set_mismatch","distance_mismatch")):raise RuntimeError("child gate "+m)
 if gate.get("id_sets_equal") is not True or r.get("snapshot_pre_post_byte_equal") is not True:raise RuntimeError("child equality")
 t=r.get("timing_gpu_ms",{})
 b=t.get("baseline",[]);a=t.get("static_aabb",[])
 if len(b)!=REPS or len(a)!=REPS or any(x<=0 for x in b+a):raise RuntimeError("child timing arrays")
 pairs=[{"rep":i,"baseline_gpu_ms":b[i],"static_aabb_gpu_ms":a[i],"ratio_static_over_baseline":a[i]/b[i]} for i in range(REPS)]
 suffixes=("nodes","empty","maxd","ids","aabb_lo","aabb_hi","aabb_dims")
 for s in suffixes:
  if sha(child/f"snapshot_before_{s}.bin")!=sha(child/f"snapshot_after_{s}.bin"):raise RuntimeError("child snapshot "+s)
 return r,pairs

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--preflight-only",action="store_true");ap.add_argument("--allow-unfrozen-audit",action="store_true");args=ap.parse_args()
 if args.allow_unfrozen_audit and not args.preflight_only:raise RuntimeError("unfrozen mode only for CPU preflight")
 pre=cpu_preflight(require_freeze=not args.allow_unfrozen_audit)
 if args.preflight_only:
  print(json.dumps(pre,sort_keys=True,indent=2));return 0
 batch_id="static_aabb_heldout_perf_evidence_v1_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+secrets.token_hex(3)
 out=RUNS/batch_id
 if out.exists():raise RuntimeError("batch collision")
 out.mkdir(mode=0o700)
 terminal=out/"terminal.json"
 atomic(terminal,{"schema":"safe-c2-static-aabb-heldout-perf-terminal-v1","status":"PREPARING","batch_id":batch_id,"started_utc":utc(),"formal_claim_eligible":False})
 collector=None
 try:
  atomic(out/"preflight.json",pre)
  shutil.copyfile(RUNNER_FREEZE,out/"runner_freeze_reference.json")
  nv=nvsmi()
  gpuinfo=physical_gpu_info(nv)
  environment={"schema":"safe-c2-static-aabb-heldout-perf-environment-v1","hostname":os.uname().nodename,"uname":list(os.uname()),"target_gpu":gpuinfo,"cuda_visible_devices":GPU_UUID,"clock_control":{"evidence_label":"DVFS_UNCONTROLLED_RECORDED","runner_attempted_clock_lock":False,"runner_attempted_application_clocks":False,"runner_attempted_power_limit_change":False,"runner_attempted_compute_mode_change":False}}
  atomic(out/"environment.json",environment)
  run_capture([nv,"-q","-x"],out/"gpu_state_before.xml")
  collector=Collector(nv,out);collector.start();collector.wait_samples(3)
  if collector.foreign:raise RuntimeError("foreign compute before launch: "+str(sorted(collector.foreign)))
  launches=[];all_pairs=[];children=[];snap_lines=[]
  env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=GPU_UUID
  for invocation in range(1,INVOCATIONS+1):
   child=out/f"invocation_{invocation:02d}";child.mkdir(mode=0o700)
   cmd=[shutil.which("nice") or "nice","-n","10",str(BINARY),"--base",str(BASE),"--queries",str(QUERY),"--groundtruth",str(GT),"--ids",str(IDS),"--outdir",str(child),"--projection-dims","128","--warmups",str(WARMUPS),"--reps",str(REPS)]
   target_start=time.monotonic_ns()
   with (child/"stdout.log").open("w") as so,(child/"stderr.log").open("w") as se:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=so,stderr=se,text=True)
    collector.set_target(proc.pid,"target")
    code=proc.wait(timeout=1800)
   target_end=time.monotonic_ns();collector.set_target(None,f"between_invocations_{invocation:02d}")
   launches.append({"invocation":invocation,"pid":proc.pid,"target_start_monotonic_ns":target_start,"target_exit_monotonic_ns":target_end,"returncode":code,"command":cmd})
   if code!=0:raise RuntimeError("child binary exit "+str(code))
   r,pairs=valid_child(child)
   for pair in pairs:
    pair["order"]="baseline_then_static_aabb" if pair["rep"]%2==0 else "static_aabb_then_baseline"
    pair["static_over_baseline"]=pair["ratio_static_over_baseline"]
   all_pairs.append({"invocation":invocation,"source_result_sha256":sha(child/"result.json"),"pairs":pairs,"median_ratio_static_over_baseline":statistics.median([x["ratio_static_over_baseline"] for x in pairs]),"raw_result":art(child/"result.json")})
   children.append({"invocation":invocation,"result":art(child/"result.json")})
   for p in sorted(child.glob("snapshot_*_*.bin")):snap_lines.append(f"{invocation:02d} {sha(p)} {p.name}\n")
  collector.set_target(None,"posttarget");time.sleep(.35);collector.stop()
  summary=collector.summary();collector=None
  run_capture([nv,"-q","-x"],out/"gpu_state_after.xml")
  atomic(out/"launch.json",{"schema":"safe-c2-static-aabb-heldout-perf-launch-v1","target_gpu":gpuinfo,"cuda_visible_devices":GPU_UUID,"binary":art(BINARY),"source":art(SOURCE),"heldout_ids":art(IDS),"base":art(BASE),"query":art(QUERY),"groundtruth":art(GT),"protocol":art(PROTOCOL),"launches":launches,"runner_attempted_clock_lock":False,"runner_attempted_application_clocks":False,"runner_attempted_power_limit_change":False,"runner_attempted_compute_mode_change":False})
  (out/"snapshot_sha256.txt").write_text("".join(snap_lines))
  atomic(out/"timing_pairs.json",{"schema":"safe-c2-static-aabb-heldout-perf-timing-pairs-v1","invocations":all_pairs,"aggregate_median_of_invocation_medians":statistics.median([x["median_ratio_static_over_baseline"] for x in all_pairs]),"range_of_invocation_medians":[min(x["median_ratio_static_over_baseline"] for x in all_pairs),max(x["median_ratio_static_over_baseline"] for x in all_pairs)]})
  atomic(out/"telemetry_summary.json",summary)
  if summary["foreign_compute_pids"]:raise RuntimeError("INVALID_FOREIGN_COMPUTE_INTERFERENCE: "+str(summary["foreign_compute_pids"]))
  for launch in launches:
   seen=summary["target_sample_times"].get(str(launch["pid"]),[])
   if not seen or min(seen)-launch["target_start_monotonic_ns"]>500_000_000 or launch["target_exit_monotonic_ns"]-max(seen)>500_000_000:
    raise RuntimeError("EXPLORATORY_TELEMETRY_INSUFFICIENT: target coverage")
  if summary["query_errors"] or not summary["numeric_field_complete"] or summary["max_inter_sample_gap_ms"]>MAX_GAP_MS or len(summary["observed_target_pids"])!=INVOCATIONS:raise RuntimeError("EXPLORATORY_TELEMETRY_INSUFFICIENT")
  aggregate={"schema":"safe-c2-static-aabb-heldout-perf-evidence-result-v1","status":"COMPLETE_CONDITIONAL_DVFS_UNCONTROLLED","scope":"paired relative timing under recorded unlocked DVFS trace; not stable absolute latency or hardware-controlled evidence","formal_claim_eligible":False,"evidence_grade":"CONDITIONAL_PAIRED_DVFS_UNCONTROLLED","children":children,"timing_pairs":json.loads((out/"timing_pairs.json").read_text()),"telemetry_summary":summary}
  atomic(out/"result.json",aggregate)
  (out/"stdout.log").write_text("\n".join(json.dumps(x,sort_keys=True) for x in launches)+"\n")
  (out/"stderr.log").write_text("")
  artifacts={str(p.relative_to(out)):art(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="terminal.json"}
  manifest={"schema":"safe-c2-static-aabb-heldout-perf-manifest-v1","status":"COMPLETE_CONDITIONAL_DVFS_UNCONTROLLED","scope":aggregate["scope"],"formal_claim_eligible":False,"protocol":art(PROTOCOL),"artifacts":artifacts,"preflight":pre}
  atomic(out/"manifest.json",manifest)
  atomic(terminal,{"schema":"safe-c2-static-aabb-heldout-perf-terminal-v1","status":"COMPLETE_CONDITIONAL_DVFS_UNCONTROLLED","batch_id":batch_id,"finished_utc":utc(),"exit_code":0,"formal_claim_eligible":False,"manifest":art(out/"manifest.json"),"result":art(out/"result.json")})
  print(json.dumps({"status":"COMPLETE_CONDITIONAL_DVFS_UNCONTROLLED","run_dir":str(out)},sort_keys=True));return 0
 except Exception as e:
  if collector is not None:
   try:collector.stop()
   except Exception:pass
  atomic(terminal,{"schema":"safe-c2-static-aabb-heldout-perf-terminal-v1","status":"FAILED","batch_id":batch_id,"finished_utc":utc(),"formal_claim_eligible":False,"error":str(e),"traceback":traceback.format_exc()})
  print(json.dumps({"status":"FAILED","run_dir":str(out),"error":str(e)},sort_keys=True),file=sys.stderr);return 2
if __name__=="__main__":
 try:raise SystemExit(main())
 except Exception as e:
  print("FAIL_STATIC_AABB_HELDOUT_PERF_RUNNER_V1: "+str(e),file=sys.stderr);raise SystemExit(2)
