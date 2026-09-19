#!/usr/bin/env bash
# Guarded one-shot CUDA certificate canary. It does not run a traversal or benchmark.
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1
BIN="$ROOT/bin/aabb_device_canary_v1"
PREFLIGHT="$ROOT/tools/preflight_device_canary_v1.py"
BASE=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs
QUERIES=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs
RUNS="$ROOT/runs"
[[ -x "$BIN" && -x "$PREFLIGHT" && -f "$BASE" && -f "$QUERIES" ]] || exit 65
PREFLIGHT_JSON="$("$PREFLIGHT")"
# No GPU interaction occurs above this point.
GPU_QUERY="index,uuid,name,driver_version,memory.total,memory.used"
GPU_INFO="$(nvidia-smi -i 0 --query-gpu="$GPU_QUERY" --format=csv,noheader,nounits)"
APPS="$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
APPS="$(printf '%s\n' "$APPS" | sed '/^[[:space:]]*$/d;/No running processes found/d')"
[[ -z "$APPS" ]] || { echo "GPU0 is busy; refusing canary: $APPS" >&2; exit 75; }
mkdir -p "$RUNS"
RUN="$(mktemp -d "$RUNS/cuda_aabb_canary_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
chmod 0700 "$RUN"
FAIL_CONTEXT="initializing"
on_err() {
  local rc=$?
  python3 - "$RUN/terminal.json" "$rc" "$FAIL_CONTEXT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); rc=int(sys.argv[2]); ctx=sys.argv[3]
p.write_text(json.dumps({'schema':'safe-c2-static-aabb-device-canary-terminal-v1','status':'FAILED','exit_code':rc,'context':ctx,'formal_claim_eligible':False},sort_keys=True,indent=2)+'\n')
PY
  exit "$rc"
}
trap on_err ERR
printf '%s\n' "$PREFLIGHT_JSON" > "$RUN/preflight.json"
printf '%s\n' "$GPU_INFO" > "$RUN/gpu_before.csv"
printf '%s\n' "$APPS" > "$RUN/gpu_processes_before.txt"
FAIL_CONTEXT="running static AABB device certificate canary"
CUDA_VISIBLE_DEVICES=0 "$BIN" --base "$BASE" --queries "$QUERIES" --outdir "$RUN" \
  --projection-dims 64 --query-start 0 --query-count 8 >"$RUN/stdout.log" 2>"$RUN/stderr.log"
FAIL_CONTEXT="verifying canary artifacts"
python3 - "$RUN/result.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if x.get('status')!='PASS_STATIC_AABB_DEVICE_CERTIFICATE': raise SystemExit('canary result not PASS')
if x.get('gpu_executed') is not True or x.get('formal_claim_eligible') is not False: raise SystemExit('invalid result scope')
if x.get('projection',{}).get('dimensions')!=64: raise SystemExit('unexpected projection width')
if x.get('device_rounding',{}).get('violations')!=0: raise SystemExit('device rounding violation')
if x.get('host_cover',{}).get('violations')!=0: raise SystemExit('host cover violation')
PY
cmp "$RUN/aabb_lo_host.bin" "$RUN/aabb_lo_d2h.bin"
cmp "$RUN/aabb_hi_host.bin" "$RUN/aabb_hi_d2h.bin"
cmp "$RUN/aabb_dims_host.bin" "$RUN/aabb_dims_d2h.bin"
sha256sum "$RUN/aabb_lo_host.bin" "$RUN/aabb_hi_host.bin" "$RUN/aabb_dims_host.bin" \
          "$RUN/aabb_lo_d2h.bin" "$RUN/aabb_hi_d2h.bin" "$RUN/aabb_dims_d2h.bin" > "$RUN/snapshot_sha256.txt"
nvidia-smi -i 0 --query-gpu="$GPU_QUERY" --format=csv,noheader,nounits > "$RUN/gpu_after.csv"
python3 - "$RUN/manifest.json" "$RUN" "$ROOT" "$BIN" <<'PY'
import hashlib,json,pathlib,sys
out,run,root,binary=map(pathlib.Path,sys.argv[1:])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
names=['preflight.json','gpu_before.csv','gpu_processes_before.txt','stdout.log','stderr.log','result.json',
       'snapshot_sha256.txt','gpu_after.csv','aabb_lo_host.bin','aabb_hi_host.bin','aabb_dims_host.bin',
       'aabb_lo_d2h.bin','aabb_hi_d2h.bin','aabb_dims_d2h.bin']
x={'schema':'safe-c2-static-aabb-device-canary-manifest-v1','status':'COMPLETE_CANARY_ONLY',
   'scope':'one guarded static-AABB lower-bound CUDA canary; no traversal/top-K/timing/formal claim',
   'run_root':str(run),'binary':{'path':str(binary),'sha256':sha(binary)},
   'artifacts':{n:{'sha256':sha(run/n),'bytes':(run/n).stat().st_size} for n in names},
   'gpu_binary_executed':True,'formal_claim_eligible':False}
out.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
PY
trap - ERR
printf '%s\n' "$RUN"
