#!/usr/bin/env bash
# C1 v8 measured-release guard.  Authorization is an external one-time file
# consumed by approval_gate_v8.py; no inherited C1_* environment or launcher
# marker is an authorization signal.
set -Eeuo pipefail
umask 077
PATH='/usr/bin:/bin'
export PATH
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH BASH_ENV ENV \
  C1_ALLOW_GPU0 C1_V7_TRUST_ROOT C1_V7_LAUNCHER C1_V7_TRUST_GUARD_SHA \
  C1_V7_TRUST_PINS_SHA C1_V7_TRUST_PIN_VERIFY_SHA C1_RUN_OUT C1_EXECUTION_APPROVED

ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_execution_release_v8_external_one_time_approval'
GUARD="$ROOT/control/run_c1_execution_guard_v8.sh"
GATE="$ROOT/control/approval_gate_v8.py"
VERIFIER="$ROOT/control/verify_release_v8.py"
PLAN="$ROOT/control/measured_execution_plan_v8.json"
PYTHON='/usr/bin/python3'
SMI='/usr/bin/nvidia-smi'
SHA256='/usr/bin/sha256sum'
FLOCK='/usr/bin/flock'
SETSID='/usr/bin/setsid'
REALPATH='/usr/bin/realpath'
HOST='CONFIGURE_ARCHIVE_HOST'
GPU_INDEX=0
GPU_UUID_EXPECTED='GPU-CONFIGURE-ARCHIVE-DEVICE'
RUNS="$ROOT/runs"
LOCK="$ROOT/.c1_v8_gpu0.lock"
OUT=''
CLAIM=''
APPROVAL_ID=''
FINAL_STATUS='FAILED'
FINAL_REASON='guard did not pass external approval'

block() { printf 'C1-V8-GUARD BLOCKED: %s\n' "$*" >&2; exit 69; }
sha() { "$SHA256" -- "$1" | /usr/bin/awk '{print $1}'; }
regular() {
  local p="$1" resolved
  [[ "$p" == /* && -f "$p" && ! -L "$p" ]] || return 1
  resolved="$("$REALPATH" -e -- "$p")" || return 1
  [[ "$resolved" == "$p" ]]
}
regular_dir() {
  local p="$1" resolved
  [[ "$p" == /* && -d "$p" && ! -L "$p" ]] || return 1
  resolved="$("$REALPATH" -e -- "$p")" || return 1
  [[ "$resolved" == "$p" ]]
}
trim() { /usr/bin/sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }

# This is called only after the external capability was hash-verified and
# atomically consumed.  It is deliberately absent from the pre-claim path.
write_run_card() {
  [[ -n "$OUT" && -d "$OUT" ]] || return 0
  FINAL_STATUS="$FINAL_STATUS" FINAL_REASON="$FINAL_REASON" CLAIM="$CLAIM" APPROVAL_ID="$APPROVAL_ID" OUT="$OUT" \
    "$PYTHON" -B -I - "$OUT/run_card_v8.json" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1])
x={
 "schema":"gtspp-c1-v8-run-card-v1",
 "updated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "status":os.environ["FINAL_STATUS"],"reason":os.environ["FINAL_REASON"],
 "approval_claim_path":os.environ["CLAIM"],"approval_id":os.environ["APPROVAL_ID"],
 "run_root":os.environ["OUT"],"mode":"MEASURED_5X4_C1_V8",
 "policy":{"profile_nsys":False,"kill_preexisting_processes":False,"c2_residual_pruning_mode":0,"c3_mutations":"forbidden"},
}
tmp=p.with_name(p.name+".tmp")
tmp.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
tmp.replace(p)
PY
}
cleanup() { write_run_card || true; }

# Strictly post-claim GPU telemetry.  It never kills/restarts any process.
gpu_preflight() {
  local tag="$1" line index uuid mem util apps filtered
  line="$("$SMI" -i "$GPU_INDEX" --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)" || return 1
  IFS=, read -r index uuid mem util <<<"$line"
  index="$(printf '%s' "$index" | trim)"; uuid="$(printf '%s' "$uuid" | trim)"
  mem="$(printf '%s' "$mem" | trim)"; util="$(printf '%s' "$util" | trim)"
  [[ "$index" == "$GPU_INDEX" && "$uuid" == "$GPU_UUID_EXPECTED" && "$mem" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || return 1
  (( mem <= 256 && util == 0 )) || return 1
  apps="$("$SMI" -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>&1)" || return 1
  filtered="$(printf '%s\n' "$apps" | /usr/bin/grep -Ev '^[[:space:]]*$|No running compute processes found' || true)"
  [[ -z "$filtered" ]] || return 1
  mkdir -p -- "$OUT/gpu_snapshots" || return 1
  TAG="$tag" INDEX="$index" UUID="$uuid" MEM="$mem" UTIL="$util" OUT="$OUT" \
    "$PYTHON" -B -I - "$OUT/gpu_snapshots/${tag}.json" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1])
p.write_text(json.dumps({"schema":"gtspp-c1-v8-gpu-snapshot-v1","tag":os.environ["TAG"],"captured_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"physical_gpu_index":int(os.environ["INDEX"]),"physical_gpu_uuid":os.environ["UUID"],"memory_used_mib":int(os.environ["MEM"]),"utilization_gpu_pct":int(os.environ["UTIL"]),"compute_apps_empty":True},sort_keys=True)+"\n")
PY
  printf '%s\n' "$uuid"
}

variant_binary() {
  case "$1" in
    E_G_c1_off_reference) printf '%s\t%s\n' "$ROOT/build_stage/variants/E_G_c1_off_reference/C1Microbench" 'd6c9beb86aa24ec731316b3db655d6bcf3e8142d8197e38bcc4998f6d54f7fa8' ;;
    P_G_workspace_only) printf '%s\t%s\n' "$ROOT/build_stage/variants/P_G_workspace_only/C1Microbench" '4f4f7b94dfe16e9bc5986139f8c8eb844d47b2a6df4f3aea0158104dde4bbf86' ;;
    E_F_fastpath_only) printf '%s\t%s\n' "$ROOT/build_stage/variants/E_F_fastpath_only/C1Microbench" '6ac44b189443bfee682038be2854f77b52cb4b8299f4db07bcb8fd301bad4e24' ;;
    P_F_full_C1) printf '%s\t%s\n' "$ROOT/build_stage/variants/P_F_full_C1/C1Microbench" '724fc97a0c59828546064457c2cd18615beff5d6f246c8d1e6eb6b6ea9734dc4' ;;
    *) return 1 ;;
  esac
}

record_variant() {
  local rep="$1" variant="$2" rc="$3" bin="$4"
  REP="$rep" VARIANT="$variant" RC="$rc" BIN="$bin" OUT="$OUT" "$PYTHON" -B -I - "$OUT/variant_receipts.jsonl" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1])
x={"schema":"gtspp-c1-v8-variant-receipt-v1","utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"replicate":int(os.environ["REP"]),"variant":os.environ["VARIANT"],"binary":os.environ["BIN"],"returncode":int(os.environ["RC"]),"mode":"MEASURED_5X4_C1_V8"}
with p.open("a",encoding="utf-8") as f: f.write(json.dumps(x,sort_keys=True)+"\n")
PY
}

# No mkdir, lock creation, nvidia-smi, or binary launch appears on the path
# above this point. Direct invocations with forged environment variables still
# need a valid, external, unconsumed exact-binding approval file.
[[ "$0" == "$GUARD" && "$#" -eq 2 && "$1" == '--approval-file' && "$2" == /* ]] || block 'external --approval-file ABSOLUTE_PATH required; inherited environment is ignored'
regular "$GUARD" && regular "$GATE" && regular "$VERIFIER" && regular "$PLAN" || block 'release control file missing/symlink'
[[ "$(/usr/bin/hostname)" == "$HOST" ]] || block 'host mismatch'
GATE_RESULT="$("$PYTHON" -B -I "$GATE" --consume-production --approval-file "$2")" || exit $?
mapfile -t GATE_FIELDS < <(printf '%s' "$GATE_RESULT" | "$PYTHON" -B -I -c 'import json,sys; x=json.load(sys.stdin); assert x.get("pass") is True and x.get("gpu_tools_invoked") is False and x.get("cuda_binary_executed") is False; print(x["claim_path"]); print(x["output_path"]); print(x["approval_id"])') || block 'approval consumer response malformed'
[[ "${#GATE_FIELDS[@]}" -eq 3 ]] || block 'approval consumer response incomplete'
CLAIM="${GATE_FIELDS[0]}"; OUT="${GATE_FIELDS[1]}"; APPROVAL_ID="${GATE_FIELDS[2]}"
[[ "$OUT" == "$RUNS/c1_v8_measured_$APPROVAL_ID" && -f "$CLAIM" && ! -L "$CLAIM" && ! -e "$OUT" && ! -L "$OUT" ]] || block 'consumed approval output/claim binding drift'
"$PYTHON" -B -I "$VERIFIER" --root "$ROOT" >/dev/null || block 'post-claim CPU-only release closure verifier failed'

# The approval is now irreversibly consumed. All subsequent state creation and
# all GPU telemetry are therefore explicitly within that single authorization.
if [[ -e "$RUNS" || -L "$RUNS" ]]; then regular_dir "$RUNS" || block 'runs root unsafe'; else mkdir -- "$RUNS" || block 'cannot create runs root'; fi
mkdir -- "$OUT" || block 'cannot create approved output root'
trap cleanup EXIT INT TERM HUP
write_run_card || block 'cannot initialize run card'
exec 9>>"$LOCK"; "$FLOCK" -n 9 || { FINAL_REASON='another C1 v8 GPU0 session holds lock'; block "$FINAL_REASON"; }

session_uuid="$(gpu_preflight outer_prelaunch)" || { FINAL_REASON='strict outer GPU0 UUID/idle preflight failed; no binary launched'; block "$FINAL_REASON"; }
[[ "$session_uuid" == "$GPU_UUID_EXPECTED" ]] || { FINAL_REASON='unexpected outer UUID'; block "$FINAL_REASON"; }

schedule=(
 '1:E_G_c1_off_reference' '1:P_G_workspace_only' '1:E_F_fastpath_only' '1:P_F_full_C1'
 '2:P_G_workspace_only' '2:E_F_fastpath_only' '2:P_F_full_C1' '2:E_G_c1_off_reference'
 '3:E_F_fastpath_only' '3:P_F_full_C1' '3:E_G_c1_off_reference' '3:P_G_workspace_only'
 '4:P_F_full_C1' '4:E_G_c1_off_reference' '4:P_G_workspace_only' '4:E_F_fastpath_only'
 '5:E_G_c1_off_reference' '5:E_F_fastpath_only' '5:P_G_workspace_only' '5:P_F_full_C1'
)
for tuple in "${schedule[@]}"; do
  rep="${tuple%%:*}"; variant="${tuple#*:}"
  IFS=$'\t' read -r binary expected_sha < <(variant_binary "$variant") || { FINAL_REASON="unknown variant $variant"; block "$FINAL_REASON"; }
  regular "$binary" && [[ "$(sha "$binary")" == "$expected_sha" ]] || { FINAL_REASON="binary hash/path drift for $variant"; block "$FINAL_REASON"; }
  uuid="$(gpu_preflight "pre_rep${rep}_${variant}")" || { FINAL_REASON="strict prelaunch GPU0 idle/UUID failed for $rep/$variant; no later binary launched"; block "$FINAL_REASON"; }
  [[ "$uuid" == "$session_uuid" ]] || { FINAL_REASON="GPU UUID continuity failed for $rep/$variant"; block "$FINAL_REASON"; }
  variant_out="$OUT/rep${rep}/${variant}"; mkdir -p -- "$variant_out" || { FINAL_REASON="cannot create variant output"; block "$FINAL_REASON"; }
  set +e
  "$SETSID" --wait /usr/bin/env -i PATH='/usr/bin:/bin' HOME='/workspace' LANG='C' \
    LD_LIBRARY_PATH='/usr/local/cuda-13.1/lib64' C1_EXECUTION_MODE='MEASURED' CUDA_DEVICE_ORDER='PCI_BUS_ID' \
    CUDA_VISIBLE_DEVICES="$session_uuid" NVIDIA_VISIBLE_DEVICES="$session_uuid" \
    "$binary" --base '/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt' \
    --trace "$ROOT/input_stage/sift1m_10k_442_first1024_type2_query_only.txt" --radius 500 \
    --out "$variant_out" --replicate "$rep" --variant "$variant" >"$variant_out/stdout.log" 2>"$variant_out/stderr.log"
  rc=$?
  set -e
  record_variant "$rep" "$variant" "$rc" "$binary" || { FINAL_REASON='cannot record variant receipt'; block "$FINAL_REASON"; }
  [[ "$rc" -eq 0 ]] || { FINAL_REASON="C1Microbench failed for $rep/$variant rc=$rc"; block "$FINAL_REASON"; }
done
FINAL_STATUS='COMPLETE'
FINAL_REASON='approved fixed 5x4 C1 measured schedule completed; post-analysis remains a separate evidence-validation step'
exit 0
