#!/usr/bin/env bash
# Fixed-vector heldout validation: only own verified setsid child group may be signalled.
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4
PROTOCOL="$ROOT/protocols/c2_v4_metadatafix_validation_execution_protocol_v1.json"
PINS="$ROOT/provenance/c2_v4_metadatafix_validation_gpu_guard_pins_v1.json"
AUDITOR="$ROOT/tools/audit_c2_v4_metadatafix_validation_preflight_v1.py"
BIN="$ROOT/bin/GTS_safe_c2_speculative_fallback_v4_metadatafix_v1"
MANIFEST="$ROOT/inputs/final_workload_v2/workload_manifest.json"
IDS="$ROOT/inputs/final_workload_v2/validation.ids"
GAMMA="$ROOT/formal_runs_v2/c2_v4_calibration_v2/final_v4_speculative_gamma_vector.txt"
PARENT="$ROOT/formal_runs_v2";OUT="$PARENT/c2_v4_validation_v1"
STATE="$OUT/run_state.json";READY="$OUT/child.ready";WRAPPER="$OUT/child_wrapper.sh";COMMAND="$OUT/command.txt";STDOUT="$OUT/child.stdout.log";TELEMETRY="$OUT/gpu_telemetry.csv";ARTIFACTS="$OUT/output_artifacts.sha256"
GPU_INDEX=0;GPU_UUID=GPU-CONFIGURE-ARCHIVE-DEVICE;GPU_PCI=00000000:16:00.0;MIN_FREE_BYTES=107374182400
SUCCESS=0;LAUNCH_PID="";CHILD_PID="";CHILD_PGID="";CHILD_SID="";MONITOR_PID=""
write_state(){ local status=$1 reason=$2;python3 - "$STATE" "$status" "$reason" "$PROTOCOL" "$BIN" "$MANIFEST" "$IDS" "$GAMMA" "$GPU_UUID" "$GPU_PCI" <<'PY'
import hashlib,json,os,sys
state,status,reason,protocol,binary,manifest,ids,gamma,uuid,pci=sys.argv[1:]
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
x={"schema":"safe-c2-v4-metadatafix-validation-guard-state-v1","status":status,"reason":reason,"scope":"fixed-vector heldout validation only; sealed test remains locked pending independent postrun decision","protocol":{"path":protocol,"sha256":sha(protocol)},"binary":{"path":binary,"sha256":sha(binary)},"manifest":{"path":manifest,"sha256":sha(manifest)},"validation_ids":{"path":ids,"sha256":sha(ids),"count":2463},"gamma_vector":{"path":gamma,"sha256":sha(gamma)},"target_gpu":{"physical_index":0,"uuid":uuid,"pci_bus_id":pci},"sealed_test_unlocked":False}
t=state+".tmp"
with open(t,"w",encoding="utf-8") as f:json.dump(x,f,sort_keys=True,indent=2);f.write("\n")
os.replace(t,state)
PY
}
record_gpu(){ local phase=$1 now;now=$(date -u -Is);nvidia-smi -i "$GPU_INDEX" --query-gpu=index,uuid,pci.bus_id,utilization.gpu,memory.used --format=csv,noheader,nounits|awk -v t="$now" -v p="$phase" '{print t "," p "," $0}'>>"$TELEMETRY"; }
verify_own(){ [[ -n "$CHILD_PID" && -f "$READY" && ! -L "$READY" ]]||return 1;local a b c tok;IFS=$'\t' read -r a b c tok <"$READY"||return 1;[[ "$tok" == safe_c2_v4_metadatafix_validation_child_v1 && "$a" == "$CHILD_PID" && "$b" == "$CHILD_PGID" && "$c" == "$CHILD_SID" ]]||return 1;local r;r=$(ps -o pid= -o pgid= -o sid= -p "$CHILD_PID" 2>/dev/null|awk '{print $1 "\t" $2 "\t" $3}');local x y z;IFS=$'\t' read -r x y z <<<"$r";[[ "$x" == "$CHILD_PID" && "$y" == "$CHILD_PGID" && "$z" == "$CHILD_SID" && "$CHILD_PID" == "$CHILD_PGID" && "$CHILD_PID" == "$CHILD_SID" ]]; }
cleanup_own(){ if verify_own;then kill -TERM -- "-$CHILD_PGID" 2>/dev/null||true;sleep 2;fi;if [[ -n "$LAUNCH_PID" ]];then wait "$LAUNCH_PID" 2>/dev/null||true;fi; }
stop_monitor(){ if [[ -n "$MONITOR_PID" ]];then kill "$MONITOR_PID" 2>/dev/null||true;wait "$MONITOR_PID" 2>/dev/null||true;MONITOR_PID="";fi; }
on_exit(){ local rc=$?;trap - EXIT;if [[ "$SUCCESS" != 1 && -d "$OUT" ]];then set +e;write_state "FAILED_METADATAFIX_VALIDATION" "outer_guard_exit_$rc";cleanup_own;record_gpu post_failure||true;stop_monitor;set -e;fi;exit "$rc"; }
trap on_exit EXIT
for p in "$PROTOCOL" "$PINS" "$AUDITOR" "$BIN" "$MANIFEST" "$IDS" "$GAMMA";do [[ -f "$p" && ! -L "$p" ]]||{ echo "missing/symlink input $p";exit 64;};done
[[ -d "$PARENT" && ! -L "$PARENT" && "$(realpath "$PARENT")" == "$PARENT" ]]||{ echo "formal parent non-direct";exit 65;}
for p in "$OUT" "$ROOT/runs" "$ROOT/.locks";do [[ ! -e "$p" && ! -L "$p" ]]||{ echo "existing output/lock $p";exit 66;};done
command -v setsid>/dev/null;setsid --help 2>&1|grep -q -- --wait
python3 "$AUDITOR"
avail=$(df -PB1 "$ROOT"|awk 'NR==2{print $4}');[[ "$avail" -ge "$MIN_FREE_BYTES" ]]||{ echo disk;exit 67;}
line=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid,pci.bus_id,utilization.gpu,memory.used --format=csv,noheader,nounits);IFS=',' read -r u p util mem <<<"$line";u=$(echo "$u"|tr -d ' ');p=$(echo "$p"|tr -d ' ');util=$(echo "$util"|tr -d ' ');mem=$(echo "$mem"|tr -d ' ')
[[ "$u" == "$GPU_UUID" && "$p" == "$GPU_PCI" && "$util" == 0 && "$mem" -le 128 ]]||{ echo GPU;exit 68;}
apps=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null||true);[[ -z "$(echo "$apps"|tr -d '[:space:]')" ]]||{ echo "GPU compute $apps";exit 69;}
mkdir "$OUT"
[[ -d "$OUT" && ! -L "$OUT" && "$(realpath "$OUT")" == "$OUT" ]]||{ echo output_non_direct;exit 70;}
printf '%s\n' 'timestamp_utc,phase,index,uuid,pci_bus_id,utilization_gpu_percent,memory_used_mib' >"$TELEMETRY"
write_state "PREPARED_METADATAFIX_VALIDATION" "preflight passed; child not launched"
record_gpu pre_launch
cat >"$COMMAND" <<EOF
scope=fixed-vector heldout validation; sealed test locked; final guarded ABBA only
protocol_sha256=$(sha256sum "$PROTOCOL"|awk '{print $1}')
binary_sha256=$(sha256sum "$BIN"|awk '{print $1}')
manifest_sha256=$(sha256sum "$MANIFEST"|awk '{print $1}')
ids_sha256=$(sha256sum "$IDS"|awk '{print $1}')
gamma_sha256=$(sha256sum "$GAMMA"|awk '{print $1}')
gpu_uuid=$GPU_UUID
gpu_pci=$GPU_PCI
cuda_visible_devices=0
command=$BIN --mode evaluate --base-fvecs /workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs --query-fvecs $ROOT/inputs/fresh_selection_pre_gt_v1/sift_learn_fresh12524.fvecs --groundtruth-ivecs $ROOT/inputs/fresh_selection_pre_gt_v1/exact_oracle_v1/groundtruth_fp32_top100.ivecs --query-ids $IDS --out $OUT --stage validation --workload-manifest $MANIFEST --gamma-vector-file $GAMMA --warmup-reps 1 --timed-reps 7
EOF
cat >"$WRAPPER" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
OUT="$OUT";READY="$READY"
pid=\$\$;pgid=\$(ps -o pgid= -p "\$\$"|tr -d ' ');sid=\$(ps -o sid= -p "\$\$"|tr -d ' ')
t="\$READY.tmp";printf '%s\\t%s\\t%s\\t%s\\n' "\$pid" "\$pgid" "\$sid" safe_c2_v4_metadatafix_validation_child_v1 >"\$t";mv "\$t" "\$READY"
exec env CUDA_VISIBLE_DEVICES=0 "$BIN" --mode evaluate --base-fvecs /workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs --query-fvecs "$ROOT/inputs/fresh_selection_pre_gt_v1/sift_learn_fresh12524.fvecs" --groundtruth-ivecs "$ROOT/inputs/fresh_selection_pre_gt_v1/exact_oracle_v1/groundtruth_fp32_top100.ivecs" --query-ids "$IDS" --out "\$OUT" --stage validation --workload-manifest "$MANIFEST" --gamma-vector-file "$GAMMA" --warmup-reps 1 --timed-reps 7
EOF
chmod 0750 "$WRAPPER"
write_state "RUNNING_METADATAFIX_VALIDATION" "verified child session launch pending"
setsid --wait "$WRAPPER" >"$STDOUT" 2>&1 & LAUNCH_PID=$!
for i in $(seq 1 100);do [[ -f "$READY" ]]&&break;sleep 0.1;done
[[ -f "$READY" && ! -L "$READY" ]]||{ echo no_ready;exit 71;}
IFS=$'\t' read -r CHILD_PID CHILD_PGID CHILD_SID token <"$READY";[[ "$token" == safe_c2_v4_metadatafix_validation_child_v1 ]]||{ echo token;exit 72;};verify_own||{ echo session;exit 73;}
record_gpu child_verified
(while kill -0 "$CHILD_PID" 2>/dev/null;do record_gpu during_run||true;sleep 5;done)& MONITOR_PID=$!
set +e;wait "$LAUNCH_PID";rc=$?;set -e;stop_monitor;record_gpu post_run
[[ "$rc" -eq 0 ]]||{ echo "validation child exit $rc";exit "$rc";}
python3 - "$OUT" "$MANIFEST" "$IDS" "$GAMMA" <<'PY'
import json,math,pathlib,sys
out,manifest,ids,gamma=map(pathlib.Path,sys.argv[1:])
s=out/"summary.json"
if not s.is_file() or s.is_symlink():raise SystemExit("summary")
x=json.load(open(s))
if x.get("schema")!="safe-c2-speculative-fallback-v4-run-v1" or x.get("status")!="PASS_V4_HELDOUT_GUARDED" or x.get("mode")!="evaluate" or x.get("stage")!="validation":raise SystemExit("summary")
if x.get("scope")!="Safe-C2 v4 speculative gamma traversal with gamma-only fallback on a strict fresh manifest-bound workload; calibration selects a fixed gamma vector only and held-out evaluation is separately gated":raise SystemExit("scope")
if x.get("implementation_identity")!="v4 speculative-fallback skeleton; not v2 and not the submitted legacy C2 kernel" or x.get("v4_speculative_fallback") is not True or "v3_speculative_fallback" in x:raise SystemExit("identity")
wb=x.get("workload_binding",{})
if wb.get("manifest")!=str(manifest) or wb.get("schema")!="gts-v4-fresh-sift-learn-workload-v1" or not wb.get("sha256_verified_in_runner"):raise SystemExit("workload")
if x.get("input_paths",{}).get("stage_query_ids")!=str(ids):raise SystemExit("IDs")
tc=x.get("timing_comparison",{})
if tc.get("eligible_for_speed_comparison") is not True or tc.get("shared_warmup_pairs")!=1 or len(tc.get("timed_pair_order",[]))!=7:raise SystemExit("ABBA")
expect=["baseline_then_speculative_guarded" if i%2==0 else "speculative_guarded_then_baseline" for i in range(7)]
if tc.get("timed_pair_order")!=expect:raise SystemExit("ABBA order")
gate=x.get("final_guarded_per_query_no_regression_gate",{})
if not gate.get("passed") or gate.get("query_count")!=2463 or gate.get("max_violating_queries")!=0 or gate.get("repetitions_checked")!=7:raise SystemExit("gate")
v=x.get("gamma_vector")
if not isinstance(v,list) or len(v)!=8 or any(not isinstance(a,(int,float)) or not math.isfinite(a) or a<1 for a in v):raise SystemExit("gamma")
rows=gamma.read_text().splitlines()
vals=[float(line.split()[1]) for line in rows]
if len(vals)!=8 or any(abs(vals[i]-float(v[i]))>1e-5 for i in range(8)):raise SystemExit("fixed gamma mismatch")
print("PASS_METADATAFIX_VALIDATION_SUMMARY_AND_ABBA_GATE")
PY
write_state "COMPLETE_METADATAFIX_VALIDATION_PASS_PENDING_SEALED_DECISION" "validation passed; sealed test remains locked pending independent precommitted decision"
find "$OUT" -maxdepth 1 -type f ! -name output_artifacts.sha256 -printf '%f\n'|LC_ALL=C sort|while IFS= read -r n;do sha256sum "$OUT/$n";done >"$ARTIFACTS"
SUCCESS=1
printf '%s\n' "$OUT"
