#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1
PREF="$ROOT/tools/preflight_diskguard_v2_1.py"
MODE="$1";case "$MODE" in topk|perf);;*)exit 64;;esac
if [[ "$MODE" == topk ]];then
 BIN="$ROOT/bin/static_aabb_topk_canary_v2_1_diskguard";IDS="$ROOT/inputs/standard_sift_tiefree_smoke32_v1.ids";EXPECT=PASS_STATIC_AABB_TOPK_CANARY_V2_DISKGUARD;TAG=topk
else
 BIN="$ROOT/bin/static_aabb_perf_probe_v2_1_diskguard";IDS="$ROOT/inputs/standard_sift_tiefree_perf256_v1.ids";EXPECT=PASS_EXPLORATORY_PAIRED_PROBE_V2_DISKGUARD;TAG=perf
fi
BASE=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs
QRY=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs
GT=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs
P="$("$PREF")"
Q="index,uuid,name,driver_version,memory.total,memory.used"
G="$(nvidia-smi -i 0 --query-gpu="$Q" --format=csv,noheader,nounits)"
A="$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null||true)"
A="$(printf '%s\n' "$A"|sed '/^[[:space:]]*$/d;/No running processes found/d')"
[[ -z "$A" ]]||exit 75
RUN="$(mktemp -d "$ROOT/runs/static_aabb_v2_1_diskguard_""$TAG""_m128_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")";chmod 0700 "$RUN";C=run
trap 'rc=$?;python3 - "$RUN/terminal.json" "$rc" "$C" "$MODE" <<PY
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"schema":"safe-c2-static-aabb-diskguard-v2.1-terminal","status":"FAILED","exit_code":int(sys.argv[2]),"context":sys.argv[3],"mode":sys.argv[4],"formal_claim_eligible":False},sort_keys=True,indent=2)+"\n")
PY
exit $rc' ERR
printf '%s\n' "$P" >"$RUN/preflight.json";printf '%s\n' "$G" >"$RUN/gpu_before.csv";printf '%s\n' "$A" >"$RUN/gpu_processes_before.txt"
if [[ "$MODE" == topk ]];then
 CUDA_VISIBLE_DEVICES=0 "$BIN" --base "$BASE" --queries "$QRY" --ids "$IDS" --outdir "$RUN" --projection-dims 128 >"$RUN/stdout.log" 2>"$RUN/stderr.log"
else
 CUDA_VISIBLE_DEVICES=0 "$BIN" --base "$BASE" --queries "$QRY" --groundtruth "$GT" --ids "$IDS" --outdir "$RUN" --projection-dims 128 --warmups 2 --reps 7 >"$RUN/stdout.log" 2>"$RUN/stderr.log"
fi
C=verify
python3 - "$RUN/result.json" "$MODE" "$EXPECT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));mode=sys.argv[2];expect=sys.argv[3]
assert x['status']==expect and x['projection']['dimensions']==128 and x['snapshot_pre_post_byte_equal'] is True and x['host_cover']['violations']==0
if mode=='topk':
 for k in ('baseline','static_aabb'):assert all(x[k][z]==0 for z in ('invalid','duplicate','oracle_set_mismatch','distance_mismatch'))
 assert x['baseline_and_static_aabb_id_sets_equal'] is True
else:
 for k in ('baseline','static_aabb'):assert all(x['unmeasured_gate'][k][z]==0 for z in ('invalid','duplicate','gt_set_mismatch','distance_mismatch'))
 assert x['unmeasured_gate']['id_sets_equal'] is True
 t=x['timing_gpu_ms'];assert len(t['baseline'])==7 and len(t['static_aabb'])==7 and min(t['baseline'])>0 and min(t['static_aabb'])>0
PY
for n in nodes empty maxd ids aabb_lo aabb_hi aabb_dims;do cmp "$RUN/snapshot_before_$n.bin" "$RUN/snapshot_after_$n.bin";done
sha256sum "$RUN"/snapshot_before_*.bin "$RUN"/snapshot_after_*.bin >"$RUN/snapshot_sha256.txt"
nvidia-smi -i 0 --query-gpu="$Q" --format=csv,noheader,nounits >"$RUN/gpu_after.csv"
python3 - "$RUN/manifest.json" "$RUN" "$BIN" "$MODE" <<'PY'
import hashlib,json,pathlib,sys
o,r,b=map(pathlib.Path,sys.argv[1:4]);mode=sys.argv[4]
def h(p):
 x=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):x.update(z)
 return x.hexdigest()
ns=['preflight.json','gpu_before.csv','gpu_processes_before.txt','stdout.log','stderr.log','result.json','snapshot_sha256.txt','gpu_after.csv']+sorted(p.name for p in r.glob('snapshot_before_*.bin'))+sorted(p.name for p in r.glob('snapshot_after_*.bin'))
o.write_text(json.dumps({'schema':'safe-c2-static-aabb-diskguard-v2.1-manifest','status':'COMPLETE_EXPLORATORY_PAIRED_PROBE' if mode=='perf' else 'COMPLETE_CANARY_ONLY','scope':'SIFT-integer disk-guard v2.1; top-K/perf gate only; no formal claim','mode':mode,'binary':{'path':str(b),'sha256':h(b)},'artifacts':{n:{'sha256':h(r/n),'bytes':(r/n).stat().st_size} for n in ns},'gpu_binary_executed':True,'formal_claim_eligible':False},sort_keys=True,indent=2)+'\n')
PY
trap - ERR
echo "$RUN"
