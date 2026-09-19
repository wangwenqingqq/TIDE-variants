#!/usr/bin/env bash
# One formal Safe-C2 v4 calibration stage.  It creates one fresh ledger root and
# signals only its own verified setsid child process group if a failure occurs.
set -Eeuo pipefail
umask 077

ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4
PROTOCOL="$ROOT/protocols/c2_v4_formal_calibration_execution_protocol_v1.json"
PINS="$ROOT/provenance/c2_v4_formal_calibration_gpu_guard_pins_v1.json"
AUDITOR="$ROOT/tools/audit_c2_v4_formal_calibration_preflight.py"
BIN="$ROOT/bin/GTS_safe_c2_speculative_fallback_v4_sift1m"
MANIFEST="$ROOT/inputs/final_workload_v2/workload_manifest.json"
CAL_IDS="$ROOT/inputs/final_workload_v2/calibration.ids"
PARENT="$ROOT/formal_runs"
OUT="$PARENT/c2_v4_calibration_v1"
STATE="$OUT/run_state.json"
READY="$OUT/child.ready"
WRAPPER="$OUT/child_wrapper.sh"
COMMAND="$OUT/command.txt"
STDOUT="$OUT/child.stdout.log"
TELEMETRY="$OUT/gpu_telemetry.csv"
HANDOFF="$OUT/calibration_handoff.json"
ARTIFACTS="$OUT/output_artifacts.sha256"
GPU_INDEX=0
GPU_UUID=GPU-CONFIGURE-ARCHIVE-DEVICE
GPU_PCI=00000000:16:00.0
MIN_FREE_BYTES=107374182400
SUCCESS=0
LAUNCH_PID=""
CHILD_PID=""
CHILD_PGID=""
CHILD_SID=""
MONITOR_PID=""

write_state() {
  local status=$1
  local reason=$2
  python3 - "$STATE" "$status" "$reason" "$PROTOCOL" "$BIN" "$MANIFEST" "$CAL_IDS" "$GPU_UUID" "$GPU_PCI" <<'PY'
import hashlib,json,os,sys
state,status,reason,protocol,binary,manifest,ids,uuid,pci=sys.argv[1:]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
obj={
 "schema":"safe-c2-v4-formal-calibration-guard-state-v1",
 "status":status,"reason":reason,
 "scope":"formal calibration only; fixed-vector handoff is permitted only after terminal PASS; no heldout performance claim",
 "protocol":{"path":protocol,"sha256":sha(protocol)},
 "binary":{"path":binary,"sha256":sha(binary)},
 "manifest":{"path":manifest,"sha256":sha(manifest)},
 "calibration_ids":{"path":ids,"sha256":sha(ids),"count":2458},
 "target_gpu":{"physical_index":0,"uuid":uuid,"pci_bus_id":pci},
 "performance_claim_eligible":False,
}
tmp=state+".tmp"
with open(tmp,"w",encoding="utf-8") as f:
 json.dump(obj,f,sort_keys=True,indent=2);f.write("\n")
os.replace(tmp,state)
PY
}

record_gpu() {
  local phase=$1
  local now
  now=$(date -u -Is)
  nvidia-smi -i "$GPU_INDEX" --query-gpu=index,uuid,pci.bus_id,utilization.gpu,memory.used --format=csv,noheader,nounits \
    | awk -v t="$now" -v p="$phase" '{print t "," p "," $0}' >>"$TELEMETRY"
}

verify_own_child_session() {
  [[ -n "$CHILD_PID" && -f "$READY" && ! -L "$READY" ]] || return 1
  local ready_pid ready_pgid ready_sid token
  IFS=$'\t' read -r ready_pid ready_pgid ready_sid token <"$READY" || return 1
  [[ "$token" == "safe_c2_v4_formal_calibration_child_v1" ]] || return 1
  [[ "$ready_pid" == "$CHILD_PID" && "$ready_pgid" == "$CHILD_PGID" && "$ready_sid" == "$CHILD_SID" ]] || return 1
  local observed
  observed=$(ps -o pid= -o pgid= -o sid= -p "$CHILD_PID" 2>/dev/null | awk '{print $1 "\t" $2 "\t" $3}')
  [[ -n "$observed" ]] || return 1
  local observed_pid observed_pgid observed_sid
  IFS=$'\t' read -r observed_pid observed_pgid observed_sid <<<"$observed"
  [[ "$observed_pid" == "$CHILD_PID" && "$observed_pgid" == "$CHILD_PGID" && "$observed_sid" == "$CHILD_SID" && "$CHILD_PID" == "$CHILD_PGID" && "$CHILD_PID" == "$CHILD_SID" ]]
}

cleanup_own_child() {
  if verify_own_child_session; then
    kill -TERM -- "-$CHILD_PGID" 2>/dev/null || true
    sleep 2
  fi
  if [[ -n "$LAUNCH_PID" ]]; then
    wait "$LAUNCH_PID" 2>/dev/null || true
  fi
}

stop_monitor() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ "$SUCCESS" != 1 && -d "$OUT" ]]; then
    set +e
    write_state "FAILED_FORMAL_CALIBRATION" "outer_guard_exit_$rc"
    cleanup_own_child
    record_gpu "post_failure" || true
    stop_monitor
    set -e
  fi
  exit "$rc"
}
trap on_exit EXIT

for path in "$PROTOCOL" "$PINS" "$AUDITOR" "$BIN" "$MANIFEST" "$CAL_IDS"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing/symlink formal input: $path" >&2; exit 64; }
done
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo "formal calibration output exists: $OUT" >&2; exit 65; }
command -v setsid >/dev/null
setsid --help 2>&1 | grep -q -- --wait
python3 "$AUDITOR"

avail=$(df -PB1 "$ROOT" | awk 'NR==2 {print $4}')
[[ "$avail" -ge "$MIN_FREE_BYTES" ]] || { echo "insufficient free disk bytes: $avail" >&2; exit 68; }
gpu_line=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid,pci.bus_id,utilization.gpu,memory.used --format=csv,noheader,nounits)
IFS=',' read -r pre_uuid pre_pci pre_util pre_mem <<<"$gpu_line"
pre_uuid=$(echo "$pre_uuid" | tr -d ' ')
pre_pci=$(echo "$pre_pci" | tr -d ' ')
pre_util=$(echo "$pre_util" | tr -d ' ')
pre_mem=$(echo "$pre_mem" | tr -d ' ')
[[ "$pre_uuid" == "$GPU_UUID" && "$pre_pci" == "$GPU_PCI" ]] || { echo "GPU UUID/PCI mismatch" >&2; exit 69; }
[[ "$pre_util" == 0 ]] || { echo "GPU utilization not idle: $pre_util" >&2; exit 70; }
[[ "$pre_mem" -le 128 ]] || { echo "GPU memory not idle: $pre_mem" >&2; exit 71; }
apps=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
[[ -z "$(echo "$apps" | tr -d '[:space:]')" ]] || { echo "GPU has compute process: $apps" >&2; exit 72; }

mkdir -p "$PARENT"
mkdir "$OUT"
printf '%s\n' 'timestamp_utc,phase,index,uuid,pci_bus_id,utilization_gpu_percent,memory_used_mib' >"$TELEMETRY"
write_state "PREPARED_FORMAL_CALIBRATION" "CPU-only preflight and immediate GPU preflight passed; child not launched"
record_gpu "pre_launch"
cat >"$COMMAND" <<EOF
scope=formal calibration only; no validation/test IDs or canary outputs are input
protocol_sha256=$(sha256sum "$PROTOCOL" | awk '{print $1}')
pins_sha256=$(sha256sum "$PINS" | awk '{print $1}')
binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')
manifest_sha256=$(sha256sum "$MANIFEST" | awk '{print $1}')
calibration_ids_sha256=$(sha256sum "$CAL_IDS" | awk '{print $1}')
gpu_uuid=$GPU_UUID
gpu_pci=$GPU_PCI
cuda_visible_devices=0
command=$BIN --mode calibrate --base-fvecs /workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs --query-fvecs /workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/fresh_selection_pre_gt_v1/sift_learn_fresh12524.fvecs --groundtruth-ivecs /workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/fresh_selection_pre_gt_v1/exact_oracle_v1/groundtruth_fp32_top100.ivecs --query-ids $CAL_IDS --out $OUT --stage calibration --workload-manifest $MANIFEST --warmup-reps 0 --timed-reps 1
EOF
cat >"$WRAPPER" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
OUT="$OUT"
READY="$READY"
pid=\$\$
pgid=\$(ps -o pgid= -p "\$\$" | tr -d ' ')
sid=\$(ps -o sid= -p "\$\$" | tr -d ' ')
tmp="\$READY.tmp"
printf '%s\\t%s\\t%s\\t%s\\n' "\$pid" "\$pgid" "\$sid" safe_c2_v4_formal_calibration_child_v1 >"\$tmp"
mv "\$tmp" "\$READY"
exec env CUDA_VISIBLE_DEVICES=0 "$BIN" --mode calibrate \\
  --base-fvecs /workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs \\
  --query-fvecs /workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/fresh_selection_pre_gt_v1/sift_learn_fresh12524.fvecs \\
  --groundtruth-ivecs /workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/fresh_selection_pre_gt_v1/exact_oracle_v1/groundtruth_fp32_top100.ivecs \\
  --query-ids "$CAL_IDS" --out "\$OUT" --stage calibration \\
  --workload-manifest "$MANIFEST" --warmup-reps 0 --timed-reps 1
EOF
chmod 0750 "$WRAPPER"
write_state "RUNNING_FORMAL_CALIBRATION" "verified child session launch pending"
setsid --wait "$WRAPPER" >"$STDOUT" 2>&1 &
LAUNCH_PID=$!
for i in $(seq 1 100); do
  if [[ -f "$READY" ]]; then break; fi
  sleep 0.1
done
[[ -f "$READY" && ! -L "$READY" ]] || { echo "child never wrote ready record" >&2; exit 73; }
IFS=$'\t' read -r CHILD_PID CHILD_PGID CHILD_SID child_token <"$READY"
[[ "$child_token" == safe_c2_v4_formal_calibration_child_v1 ]] || { echo "ready token mismatch" >&2; exit 74; }
verify_own_child_session || { echo "child PID/PGID/SID contract failed" >&2; exit 75; }
record_gpu "child_verified"
(
  while kill -0 "$CHILD_PID" 2>/dev/null; do
    record_gpu "during_run" || true
    sleep 5
  done
) &
MONITOR_PID=$!
set +e
wait "$LAUNCH_PID"
child_rc=$?
set -e
stop_monitor
record_gpu "post_run"
[[ "$child_rc" -eq 0 ]] || { echo "formal calibration child exit code $child_rc" >&2; exit "$child_rc"; }
[[ -f "$OUT/summary.json" && ! -L "$OUT/summary.json" ]] || { echo "formal calibration summary missing" >&2; exit 76; }
python3 - "$OUT" "$MANIFEST" "$CAL_IDS" "$PROTOCOL" "$STATE" "$HANDOFF" <<'PY'
import hashlib,json,math,os,pathlib,sys
out,manifest,ids,protocol,state,handoff=map(pathlib.Path,sys.argv[1:])
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
summary=out/"summary.json"
x=json.load(open(summary))
if x.get("status")!="PASS_V4_CALIBRATION_GUARDED":raise SystemExit("unexpected summary status")
if x.get("mode")!="calibrate" or x.get("stage")!="calibration":raise SystemExit("summary mode/stage")
if x.get("workload_binding",{}).get("manifest")!=str(manifest) or not x.get("workload_binding",{}).get("sha256_verified_in_runner"):raise SystemExit("summary manifest binding")
if x.get("input_paths",{}).get("stage_query_ids")!=str(ids):raise SystemExit("summary calibration ID binding")
gate=x.get("final_guarded_per_query_no_regression_gate",{})
if not gate.get("passed") or gate.get("query_count")!=2458 or gate.get("max_violating_queries")!=0:raise SystemExit("final guarded calibration gate")
g=x.get("gamma_vector")
if not isinstance(g,list) or len(g)!=8 or any((not isinstance(v,(int,float)) or not math.isfinite(v) or v<1) for v in g):raise SystemExit("gamma vector json")
if any(float(g[i])!=1.0 for i in range(3)):raise SystemExit("frozen gamma shallow levels")
gamma=out/"final_v4_speculative_gamma_vector.txt"
if not gamma.is_file() or gamma.is_symlink():raise SystemExit("gamma file")
lines=gamma.read_text().splitlines()
if len(lines)!=8:raise SystemExit("gamma file line count")
vals=[]
for i,line in enumerate(lines):
 a=line.split()
 if len(a)!=2 or int(a[0])!=i:raise SystemExit("gamma line schema")
 vals.append(float(a[1]))
if any(abs(vals[i]-float(g[i]))>1e-5 for i in range(8)):raise SystemExit("gamma summary/file mismatch")
obj={
 "schema":"safe-c2-v4-calibration-fixed-vector-handoff-v1",
 "status":"PASS_FORMAL_CALIBRATION_FIXED_VECTOR_HANDOFF",
 "scope":"calibration-only fixed-vector handoff; no heldout/test metric or performance claim",
 "protocol":{"path":str(protocol),"sha256":sha(protocol)},
 "manifest":{"path":str(manifest),"sha256":sha(manifest)},
 "calibration_ids":{"path":str(ids),"sha256":sha(ids),"count":2458},
 "summary":{"path":str(summary),"sha256":sha(summary)},
 "gamma_vector":{"path":str(gamma),"sha256":sha(gamma),"values":vals},
 "final_guarded_calibration_gate":gate,
 "performance_claim_eligible":False
}
tmp=str(handoff)+".tmp"
with open(tmp,"w",encoding="utf-8") as f:json.dump(obj,f,sort_keys=True,indent=2);f.write("\n")
os.replace(tmp,handoff)
print("PASS_FORMAL_CALIBRATION_SUMMARY_AND_HANDOFF")
PY
write_state "COMPLETE_FORMAL_CALIBRATION_PASS" "formal calibration passed; immutable fixed-vector handoff created; no heldout metric yet"
find "$OUT" -maxdepth 1 -type f ! -name output_artifacts.sha256 -printf '%f\n' | LC_ALL=C sort | while IFS= read -r name; do
  sha256sum "$OUT/$name"
done >"$ARTIFACTS"
SUCCESS=1
printf '%s\n' "$OUT"
