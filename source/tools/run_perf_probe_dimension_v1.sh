#!/usr/bin/env bash
# Guarded repetition of the fixed exploratory paired probe at an explicit AABB width.
set -Eeuo pipefail
umask 077
[[ $# -eq 1 ]] || { echo "usage: $0 {16|32|128}" >&2; exit 64; }
DIM="$1";case "$DIM" in 16|32|128);;*)exit 64;;esac
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1
BIN="$ROOT/bin/static_aabb_perf_probe_v1";PREF="$ROOT/tools/preflight_perf_probe_v1.py"
BASE=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs
QUERIES=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs
GT=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs
IDS="$ROOT/inputs/standard_sift_tiefree_perf256_v1.ids"
PREF_JSON="$("$PREF")"
GPU_QUERY="index,uuid,name,driver_version,memory.total,memory.used"
GPU_INFO="$(nvidia-smi -i 0 --query-gpu="$GPU_QUERY" --format=csv,noheader,nounits)"
APPS="$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null||true)"
APPS="$(printf '%s\n' "$APPS"|sed '/^[[:space:]]*$/d;/No running processes found/d')"
[[ -z "$APPS" ]]||{ echo "GPU0 busy; probe refused: $APPS" >&2;exit 75; }
RUN="$(mktemp -d "$ROOT/runs/static_aabb_perf_probe_m""$DIM""_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")";chmod 0700 "$RUN";CTX=initializing
on_err(){ local rc=$?;python3 - "$RUN/terminal.json" "$rc" "$CTX" "$DIM" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'schema':'safe-c2-static-aabb-paired-perf-terminal-v1','status':'FAILED','exit_code':int(sys.argv[2]),'context':sys.argv[3],'projection_dims':int(sys.argv[4]),'formal_claim_eligible':False},sort_keys=True,indent=2)+'\n')
PY
exit "$rc";}
trap on_err ERR
printf '%s\n' "$PREF_JSON" >"$RUN/preflight.json";printf '%s\n' "$GPU_INFO" >"$RUN/gpu_before.csv";printf '%s\n' "$APPS" >"$RUN/gpu_processes_before.txt"
CTX="running fixed paired probe at projection width $DIM"
CUDA_VISIBLE_DEVICES=0 "$BIN" --base "$BASE" --queries "$QUERIES" --groundtruth "$GT" --ids "$IDS" --outdir "$RUN" --projection-dims "$DIM" --warmups 2 --reps 7 >"$RUN/stdout.log" 2>"$RUN/stderr.log"
CTX="verifying output"
python3 - "$RUN/result.json" "$DIM" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));m=int(sys.argv[2])
if x.get('status')!='PASS_EXPLORATORY_PAIRED_PROBE' or x.get('projection',{}).get('dimensions')!=m:raise SystemExit('result status/dim failure')
for role in ('baseline','static_aabb'):
 y=x['unmeasured_gate'][role]
 if any(y[z]!=0 for z in ('invalid','duplicate','gt_set_mismatch','distance_mismatch')):raise SystemExit(role+' gate')
t=x['timing_gpu_ms']
if len(t['baseline'])!=7 or len(t['static_aabb'])!=7 or min(t['baseline'])<=0 or min(t['static_aabb'])<=0:raise SystemExit('timing')
if not x.get('baseline_and_static_aabb_id_sets_equal',x['unmeasured_gate'].get('id_sets_equal')):raise SystemExit('id equality')
PY
for n in nodes empty maxd ids aabb_lo aabb_hi aabb_dims;do cmp "$RUN/snapshot_before_$n.bin" "$RUN/snapshot_after_$n.bin";done
sha256sum "$RUN"/snapshot_before_*.bin "$RUN"/snapshot_after_*.bin >"$RUN/snapshot_sha256.txt"
nvidia-smi -i 0 --query-gpu="$GPU_QUERY" --format=csv,noheader,nounits >"$RUN/gpu_after.csv"
python3 - "$RUN/manifest.json" "$RUN" "$BIN" "$DIM" <<'PY'
import hashlib,json,pathlib,sys
out,run,binary=map(pathlib.Path,sys.argv[1:4]);dim=int(sys.argv[4])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
names=['preflight.json','gpu_before.csv','gpu_processes_before.txt','stdout.log','stderr.log','result.json','snapshot_sha256.txt','gpu_after.csv']
names+=sorted(p.name for p in run.glob('snapshot_before_*.bin'))+sorted(p.name for p in run.glob('snapshot_after_*.bin'))
x={'schema':'safe-c2-static-aabb-paired-perf-manifest-v1','status':'COMPLETE_EXPLORATORY_PAIRED_PROBE','scope':'fixed paired exploration only; no formal performance claim','projection_dims':dim,'run_root':str(run),'binary':{'path':str(binary),'sha256':sha(binary)},'artifacts':{n:{'sha256':sha(run/n),'bytes':(run/n).stat().st_size} for n in names},'gpu_binary_executed':True,'formal_claim_eligible':False}
out.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
PY
trap - ERR
printf '%s\n' "$RUN"
