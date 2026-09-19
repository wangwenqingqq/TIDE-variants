#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1
PREF="$ROOT/tools/preflight_diskguard_v2.py";BIN="$ROOT/bin/static_aabb_perf_probe_v2_diskguard"
BASE=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs;QRY=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs;GT=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs;IDS="$ROOT/inputs/standard_sift_tiefree_perf256_v1.ids"
P="$("$PREF")";Q="index,uuid,name,driver_version,memory.total,memory.used";G="$(nvidia-smi -i 0 --query-gpu="$Q" --format=csv,noheader,nounits)";A="$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null||true)";A="$(printf '%s\n' "$A"|sed '/^[[:space:]]*$/d;/No running processes found/d')";[[ -z "$A" ]]||exit 75
RUN="$(mktemp -d "$ROOT/runs/static_aabb_v2_diskguard_perf_m128_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")";chmod 0700 "$RUN";C=run
trap 'rc=$?;python3 - "$RUN/terminal.json" "$rc" "$C" <<PY
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"schema":"safe-c2-static-aabb-diskguard-v2-perf-terminal","status":"FAILED","exit_code":int(sys.argv[2]),"context":sys.argv[3],"formal_claim_eligible":False},sort_keys=True,indent=2)+"\n")
PY
exit $rc' ERR
printf '%s\n' "$P" >"$RUN/preflight.json";printf '%s\n' "$G" >"$RUN/gpu_before.csv";printf '%s\n' "$A" >"$RUN/gpu_processes_before.txt"
CUDA_VISIBLE_DEVICES=0 "$BIN" --base "$BASE" --queries "$QRY" --groundtruth "$GT" --ids "$IDS" --outdir "$RUN" --projection-dims 128 --warmups 2 --reps 7 >"$RUN/stdout.log" 2>"$RUN/stderr.log"
C=verify
python3 - "$RUN/result.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x['status']=='PASS_EXPLORATORY_PAIRED_PROBE_V2_DISKGUARD' and x['projection']['dimensions']==128 and x['snapshot_pre_post_byte_equal'] is True
for k in ('baseline','static_aabb'):assert all(x['unmeasured_gate'][k][z]==0 for z in ('invalid','duplicate','gt_set_mismatch','distance_mismatch'))
t=x['timing_gpu_ms'];assert len(t['baseline'])==7 and len(t['static_aabb'])==7 and min(t['baseline'])>0 and min(t['static_aabb'])>0
assert x['unmeasured_gate']['id_sets_equal'] is True and x['host_cover']['violations']==0
PY
for n in nodes empty maxd ids aabb_lo aabb_hi aabb_dims;do cmp "$RUN/snapshot_before_$n.bin" "$RUN/snapshot_after_$n.bin";done
sha256sum "$RUN"/snapshot_before_*.bin "$RUN"/snapshot_after_*.bin >"$RUN/snapshot_sha256.txt";nvidia-smi -i 0 --query-gpu="$Q" --format=csv,noheader,nounits >"$RUN/gpu_after.csv"
python3 - "$RUN/manifest.json" "$RUN" "$BIN" <<'PY'
import hashlib,json,pathlib,sys
o,r,b=map(pathlib.Path,sys.argv[1:])
def h(p):
 x=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):x.update(z)
 return x.hexdigest()
ns=['preflight.json','gpu_before.csv','gpu_processes_before.txt','stdout.log','stderr.log','result.json','snapshot_sha256.txt','gpu_after.csv']+sorted(p.name for p in r.glob('snapshot_before_*.bin'))+sorted(p.name for p in r.glob('snapshot_after_*.bin'))
o.write_text(json.dumps({'schema':'safe-c2-static-aabb-diskguard-v2-perf-manifest','status':'COMPLETE_EXPLORATORY_PAIRED_PROBE','scope':'SIFT-integer 128d disk-guard paired timing after GT gate; exploratory only','binary':{'path':str(b),'sha256':h(b)},'artifacts':{n:{'sha256':h(r/n),'bytes':(r/n).stat().st_size} for n in ns},'gpu_binary_executed':True,'formal_claim_eligible':False},sort_keys=True,indent=2)+'\n')
PY
trap - ERR
echo "$RUN"
