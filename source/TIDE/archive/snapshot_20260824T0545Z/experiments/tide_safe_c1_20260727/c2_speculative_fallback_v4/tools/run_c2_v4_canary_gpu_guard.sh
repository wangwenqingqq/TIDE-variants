#!/usr/bin/env bash
# One isolated Safe-C2 v4 qualification-only GPU canary.
#
# This guard never touches a process unless it has first verified the exact
# ready-record PID/PGID/SID created by its own setsid child. It never targets
# GPUs other than physical GPU 0 / UUID pinned below.
set -Eeuo pipefail
umask 077

ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4
PROTOCOL="$ROOT/protocols/c2_v4_canary_gpu_execution_protocol_v1.json"
CANARY_AUDIT="$ROOT/provenance/c2_v4_canary_workload_independent_audit_v1.json"
PRELEDGER_AUDIT="$ROOT/provenance/c2_v4_preledger_gate_v2_independent_audit.json"
CANARY_RECEIPT="$ROOT/preledger_receipts/c2_v4_canary_parser_gate_v1/receipt.json"
BIN="$ROOT/bin/GTS_safe_c2_speculative_fallback_v4_sift1m"
MANIFEST="$ROOT/inputs/canary_workload_v1/workload_manifest.json"
CANARY_IDS="$ROOT/inputs/canary_workload_v1/qualification_canary.ids"
PARENT="$ROOT/canary_runs"
OUT="$PARENT/c2_v4_canary_v1"
STATE="$OUT/run_state.json"
READY="$OUT/child.ready"
WRAPPER="$OUT/child_wrapper.sh"
COMMAND="$OUT/command.txt"
STDOUT="$OUT/child.stdout.log"
TELEMETRY="$OUT/gpu_telemetry.csv"
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
  python3 - "$STATE" "$status" "$reason" "$PROTOCOL" "$BIN" "$MANIFEST" "$CANARY_IDS" "$GPU_UUID" "$GPU_PCI" <<'PY'
import hashlib,json,os,sys
state,status,reason,protocol,binary,manifest,ids,uuid,pci=sys.argv[1:]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
obj={
 "schema":"safe-c2-v4-canary-gpu-guard-state-v1",
 "status":status,"reason":reason,
 "scope":"qualification-only canary; no formal calibration/validation/test metric or tuning",
 "protocol":{"path":protocol,"sha256":sha(protocol)},
 "binary":{"path":binary,"sha256":sha(binary)},
 "manifest":{"path":manifest,"sha256":sha(manifest)},
 "canary_ids":{"path":ids,"sha256":sha(ids)},
 "target_gpu":{"physical_index":0,"uuid":uuid,"pci_bus_id":pci},
 "formal_claim_eligible":False,
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
  [[ "$token" == "safe_c2_v4_canary_child_v1" ]] || return 1
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
    write_state "FAILED_CANARY_QUALIFICATION_ONLY" "outer_guard_exit_$rc"
    cleanup_own_child
    record_gpu "post_failure" || true
    stop_monitor
    set -e
  fi
  exit "$rc"
}
trap on_exit EXIT

for path in "$PROTOCOL" "$CANARY_AUDIT" "$PRELEDGER_AUDIT" "$CANARY_RECEIPT" "$BIN" "$MANIFEST" "$CANARY_IDS"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing/symlink canary input: $path" >&2; exit 64; }
done
for path in "$OUT" "$PARENT" "$ROOT/runs" "$ROOT/.locks"; do
  [[ ! -e "$path" && ! -L "$path" ]] || { echo "refusing existing run/formal state: $path" >&2; exit 65; }
done
command -v setsid >/dev/null
setsid --help 2>&1 | grep -q -- --wait
[[ "$(sha256sum "$BIN" | awk '{print $1}')" == "7c1229eebd10cb85f9ee931f7a70c58b11dba1e4828b64812567253df152d0dc" ]] || { echo "binary hash changed" >&2; exit 66; }
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "e9e6f22a42cc15cad4154b31c134a59c73768674987e096a943ec824901f5693" ]] || { echo "canary manifest hash changed" >&2; exit 67; }
python3 - "$PROTOCOL" "$CANARY_AUDIT" "$PRELEDGER_AUDIT" "$CANARY_RECEIPT" <<'PY'
import hashlib,json,sys
protocol,audit,pre,receipt=sys.argv[1:]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
p=json.load(open(protocol));a=json.load(open(audit));q=json.load(open(pre));r=json.load(open(receipt))
if p.get("status")!="FROZEN_PRE_CANARY_GPU_EXECUTION":raise SystemExit("protocol state")
if a.get("status")!="PASS_CANARY_WORKLOAD_ISOLATION_CPU_ONLY":raise SystemExit("canary isolation audit")
if q.get("status")!="PASS_PRELEDGER_GATE_V2_STRICT_RECEIPT_VERIFIED":raise SystemExit("formal preledger audit")
if r.get("status")!="PASS_NO_CUDA_OR_DATASET_LOAD":raise SystemExit("canary preledger receipt")
if p["binary"]["sha256"]!=sha("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/bin/GTS_safe_c2_speculative_fallback_v4_sift1m"):raise SystemExit("binary binding")
if p["canary_manifest"]["sha256"]!=sha("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/canary_workload_v1/workload_manifest.json"):raise SystemExit("manifest binding")
print("PASS_CANARY_GPU_GUARD_STATIC_INPUT_BINDING")
PY

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

mkdir "$PARENT"
mkdir "$OUT"
printf '%s\n' 'timestamp_utc,phase,index,uuid,pci_bus_id,utilization_gpu_percent,memory_used_mib' >"$TELEMETRY"
write_state "PREPARED_CANARY_QUALIFICATION_ONLY" "preflight passed; child not launched"
record_gpu "pre_launch"
cat >"$COMMAND" <<EOF
scope=qualification-only canary; no formal metric/tuning
protocol_sha256=$(sha256sum "$PROTOCOL" | awk '{print $1}')
binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')
manifest_sha256=$(sha256sum "$MANIFEST" | awk '{print $1}')
canary_ids_sha256=$(sha256sum "$CANARY_IDS" | awk '{print $1}')
gpu_uuid=$GPU_UUID
gpu_pci=$GPU_PCI
cuda_visible_devices=0
command=$BIN --mode calibrate --base-fvecs /workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs --query-fvecs /workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/fresh_selection_pre_gt_v1/sift_learn_fresh12524.fvecs --groundtruth-ivecs /workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/fresh_selection_pre_gt_v1/exact_oracle_v1/groundtruth_fp32_top100.ivecs --query-ids $CANARY_IDS --out $OUT --stage calibration --workload-manifest $MANIFEST --warmup-reps 0 --timed-reps 1
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
printf '%s\\t%s\\t%s\\t%s\\n' "\$pid" "\$pgid" "\$sid" safe_c2_v4_canary_child_v1 >"\$tmp"
mv "\$tmp" "\$READY"
exec env CUDA_VISIBLE_DEVICES=0 "$BIN" --mode calibrate \\
  --base-fvecs /workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs \\
  --query-fvecs /workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/fresh_selection_pre_gt_v1/sift_learn_fresh12524.fvecs \\
  --groundtruth-ivecs /workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/fresh_selection_pre_gt_v1/exact_oracle_v1/groundtruth_fp32_top100.ivecs \\
  --query-ids "$CANARY_IDS" --out "\$OUT" --stage calibration \\
  --workload-manifest "$MANIFEST" --warmup-reps 0 --timed-reps 1
EOF
chmod 0750 "$WRAPPER"
write_state "RUNNING_CANARY_QUALIFICATION_ONLY" "verified child session launch pending"
setsid --wait "$WRAPPER" >"$STDOUT" 2>&1 &
LAUNCH_PID=$!
for i in $(seq 1 100); do
  if [[ -f "$READY" ]]; then break; fi
  sleep 0.1
done
[[ -f "$READY" && ! -L "$READY" ]] || { echo "child never wrote ready record" >&2; exit 73; }
IFS=$'\t' read -r CHILD_PID CHILD_PGID CHILD_SID child_token <"$READY"
[[ "$child_token" == safe_c2_v4_canary_child_v1 ]] || { echo "ready token mismatch" >&2; exit 74; }
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
[[ "$child_rc" -eq 0 ]] || { echo "canary child exit code $child_rc" >&2; exit "$child_rc"; }
[[ -f "$OUT/summary.json" && ! -L "$OUT/summary.json" ]] || { echo "canary summary missing" >&2; exit 76; }
python3 - "$OUT/summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if x.get("status")!="PASS_V4_CALIBRATION_GUARDED":
 raise SystemExit("unexpected canary summary status: "+str(x.get("status")))
print("PASS_CANARY_SUMMARY_STATUS")
PY
write_state "COMPLETE_CANARY_QUALIFICATION_ONLY_PASS" "canary passed; its gamma/output remain excluded from formal work"
find "$OUT" -maxdepth 1 -type f ! -name output_artifacts.sha256 -printf '%f\n' | LC_ALL=C sort | while IFS= read -r name; do
  sha256sum "$OUT/$name"
done >"$ARTIFACTS"
SUCCESS=1
printf '%s\n' "$OUT"
