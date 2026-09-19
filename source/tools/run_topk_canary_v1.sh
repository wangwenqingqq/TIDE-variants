#!/usr/bin/env bash
# Guarded static-AABB top-K correctness canary. No timing/benchmark mode.
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1
BIN="$ROOT/bin/static_aabb_topk_canary_v1"
PREF="$ROOT/tools/preflight_topk_canary_v1.py"
BASE=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs
QUERIES=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs
IDS="$ROOT/inputs/standard_sift_tiefree_smoke32_v1.ids"
[[ -x "$BIN" && -x "$PREF" && -f "$BASE" && -f "$QUERIES" && -f "$IDS" ]] || exit 65
PREF_JSON="$("$PREF")"
GPU_QUERY="index,uuid,name,driver_version,memory.total,memory.used"
GPU_INFO="$(nvidia-smi -i 0 --query-gpu="$GPU_QUERY" --format=csv,noheader,nounits)"
APPS="$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
APPS="$(printf '%s\n' "$APPS"|sed '/^[[:space:]]*$/d;/No running processes found/d')"
[[ -z "$APPS" ]] || { echo "GPU0 busy; top-K canary refused: $APPS" >&2; exit 75; }
RUN="$(mktemp -d "$ROOT/runs/static_aabb_topk_canary_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
chmod 0700 "$RUN"; CONTEXT=initializing
on_err() {
 local rc=$?
 python3 - "$RUN/terminal.json" "$rc" "$CONTEXT" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'schema':'safe-c2-static-aabb-topk-canary-terminal-v1','status':'FAILED','exit_code':int(sys.argv[2]),'context':sys.argv[3],'formal_claim_eligible':False},sort_keys=True,indent=2)+'\n')
PY
 exit "$rc"
}
trap on_err ERR
printf '%s\n' "$PREF_JSON" > "$RUN/preflight.json"; printf '%s\n' "$GPU_INFO" > "$RUN/gpu_before.csv"; printf '%s\n' "$APPS" > "$RUN/gpu_processes_before.txt"
CONTEXT="running independent CPU oracle plus baseline/static-AABB top-K canary"
CUDA_VISIBLE_DEVICES=0 "$BIN" --base "$BASE" --queries "$QUERIES" --ids "$IDS" --outdir "$RUN" --projection-dims 64 >"$RUN/stdout.log" 2>"$RUN/stderr.log"
CONTEXT="verifying top-K canary artifacts"
python3 - "$RUN/result.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if x.get('status')!='PASS_STATIC_AABB_TOPK_CANARY': raise SystemExit('top-K canary failed')
if x.get('formal_claim_eligible') is not False or x.get('gpu_executed') is not True: raise SystemExit('scope violation')
for k in ('baseline','static_aabb'):
 y=x.get(k,{})
 if any(y.get(z)!=0 for z in ('invalid','duplicate','oracle_set_mismatch','distance_mismatch')): raise SystemExit(k+' did not match oracle')
if x.get('baseline_and_static_aabb_id_sets_equal') is not True: raise SystemExit('baseline/candidate mismatch')
if x.get('snapshot_pre_post_byte_equal') is not True: raise SystemExit('snapshot status false')
if x.get('host_cover',{}).get('violations')!=0 or x.get('projection',{}).get('dimensions')!=64: raise SystemExit('cover/projection failure')
PY
for n in nodes empty maxd ids aabb_lo aabb_hi aabb_dims; do cmp "$RUN/snapshot_before_$n.bin" "$RUN/snapshot_after_$n.bin"; done
sha256sum "$RUN"/snapshot_before_*.bin "$RUN"/snapshot_after_*.bin > "$RUN/snapshot_sha256.txt"
nvidia-smi -i 0 --query-gpu="$GPU_QUERY" --format=csv,noheader,nounits > "$RUN/gpu_after.csv"
python3 - "$RUN/manifest.json" "$RUN" "$BIN" <<'PY'
import hashlib,json,pathlib,sys
out,run,binary=map(pathlib.Path,sys.argv[1:])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
names=['preflight.json','gpu_before.csv','gpu_processes_before.txt','stdout.log','stderr.log','result.json','snapshot_sha256.txt','gpu_after.csv']
names += sorted(p.name for p in run.glob('snapshot_before_*.bin'))+sorted(p.name for p in run.glob('snapshot_after_*.bin'))
x={'schema':'safe-c2-static-aabb-topk-canary-manifest-v1','status':'COMPLETE_CANARY_ONLY','scope':'guarded static-AABB top-K correctness canary; no timing/performance/formal claim','run_root':str(run),'binary':{'path':str(binary),'sha256':sha(binary)},'artifacts':{n:{'sha256':sha(run/n),'bytes':(run/n).stat().st_size} for n in names},'gpu_binary_executed':True,'formal_claim_eligible':False}
out.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
PY
trap - ERR
printf '%s\n' "$RUN"
