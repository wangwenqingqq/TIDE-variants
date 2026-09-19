#!/usr/bin/env bash
# Guarded nonformal paired timing probe. Static tree only, no update or formal claim.
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1
BIN="$ROOT/bin/static_aabb_perf_probe_v1"; PREF="$ROOT/tools/preflight_perf_probe_v1.py"
BASE=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs
QUERIES=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs
GT=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs
IDS="$ROOT/inputs/standard_sift_tiefree_perf256_v1.ids"
[[ -x "$BIN" && -x "$PREF" ]]||exit 65
PREF_JSON="$("$PREF")"
GPU_QUERY="index,uuid,name,driver_version,memory.total,memory.used"
GPU_INFO="$(nvidia-smi -i 0 --query-gpu="$GPU_QUERY" --format=csv,noheader,nounits)"
APPS="$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null||true)"
APPS="$(printf '%s\n' "$APPS"|sed '/^[[:space:]]*$/d;/No running processes found/d')"
[[ -z "$APPS" ]]||{ echo "GPU0 busy; refusing paired probe: $APPS" >&2;exit 75; }
RUN="$(mktemp -d "$ROOT/runs/static_aabb_perf_probe_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")";chmod 0700 "$RUN";CTX=initializing
on_err(){ local rc=$?;python3 - "$RUN/terminal.json" "$rc" "$CTX" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'schema':'safe-c2-static-aabb-paired-perf-terminal-v1','status':'FAILED','exit_code':int(sys.argv[2]),'context':sys.argv[3],'formal_claim_eligible':False},sort_keys=True,indent=2)+'\n')
PY
exit "$rc";}
trap on_err ERR
printf '%s\n' "$PREF_JSON" >"$RUN/preflight.json";printf '%s\n' "$GPU_INFO" >"$RUN/gpu_before.csv";printf '%s\n' "$APPS" >"$RUN/gpu_processes_before.txt"
CTX="running unmeasured GT gate then paired static-AABB timing probe"
CUDA_VISIBLE_DEVICES=0 "$BIN" --base "$BASE" --queries "$QUERIES" --groundtruth "$GT" --ids "$IDS" --outdir "$RUN" --projection-dims 64 --warmups 2 --reps 7 >"$RUN/stdout.log" 2>"$RUN/stderr.log"
CTX="verifying paired probe outputs"
python3 - "$RUN/result.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if x.get('status')!='PASS_EXPLORATORY_PAIRED_PROBE':raise SystemExit('probe did not pass gates')
if x.get('formal_claim_eligible') is not False or x.get('gpu_executed') is not True:raise SystemExit('scope invalid')
for role in ('baseline','static_aabb'):
 y=x['unmeasured_gate'][role]
 if any(y[z]!=0 for z in ('invalid','duplicate','gt_set_mismatch','distance_mismatch')):raise SystemExit(role+' GT gate fail')
t=x['timing_gpu_ms']
if len(t['baseline'])!=7 or len(t['static_aabb'])!=7 or min(t['baseline'])<=0 or min(t['static_aabb'])<=0:raise SystemExit('timing series invalid')
if x.get('snapshot_pre_post_byte_equal') is not True or x['host_cover']['violations']!=0:raise SystemExit('snapshot or cover failure')
PY
for n in nodes empty maxd ids aabb_lo aabb_hi aabb_dims;do cmp "$RUN/snapshot_before_$n.bin" "$RUN/snapshot_after_$n.bin";done
sha256sum "$RUN"/snapshot_before_*.bin "$RUN"/snapshot_after_*.bin >"$RUN/snapshot_sha256.txt"
nvidia-smi -i 0 --query-gpu="$GPU_QUERY" --format=csv,noheader,nounits >"$RUN/gpu_after.csv"
python3 - "$RUN/manifest.json" "$RUN" "$BIN" <<'PY'
import hashlib,json,pathlib,sys
out,run,binary=map(pathlib.Path,sys.argv[1:])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
names=['preflight.json','gpu_before.csv','gpu_processes_before.txt','stdout.log','stderr.log','result.json','snapshot_sha256.txt','gpu_after.csv']
names+=sorted(p.name for p in run.glob('snapshot_before_*.bin'))+sorted(p.name for p in run.glob('snapshot_after_*.bin'))
x={'schema':'safe-c2-static-aabb-paired-perf-manifest-v1','status':'COMPLETE_EXPLORATORY_PAIRED_PROBE','scope':'guarded paired timing after GT gate; exploratory only, no heldout/formal performance claim','run_root':str(run),'binary':{'path':str(binary),'sha256':sha(binary)},'artifacts':{n:{'sha256':sha(run/n),'bytes':(run/n).stat().st_size} for n in names},'gpu_binary_executed':True,'formal_claim_eligible':False}
out.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
PY
trap - ERR
printf '%s\n' "$RUN"
