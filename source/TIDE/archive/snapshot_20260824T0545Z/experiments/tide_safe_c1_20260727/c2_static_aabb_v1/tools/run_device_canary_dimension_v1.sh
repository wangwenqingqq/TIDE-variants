#!/usr/bin/env bash
# Guarded certificate canary at a chosen projection width.
set -Eeuo pipefail
umask 077
[[ $# -eq 1 ]] || exit 64
DIM="$1";case "$DIM" in 128);;*)exit 64;;esac
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1
BIN="$ROOT/bin/aabb_device_canary_v1";PREF="$ROOT/tools/preflight_device_canary_v1.py"
BASE=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs
QUERIES=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs
P="$("$PREF")";Q="index,uuid,name,driver_version,memory.total,memory.used";G="$(nvidia-smi -i 0 --query-gpu="$Q" --format=csv,noheader,nounits)"
A="$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null||true)";A="$(printf '%s\n' "$A"|sed '/^[[:space:]]*$/d;/No running processes found/d')";[[ -z "$A" ]]||exit 75
RUN="$(mktemp -d "$ROOT/runs/cuda_aabb_canary_m""$DIM""_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")";chmod 0700 "$RUN";C=run
trap 'rc=$?;python3 - "$RUN/terminal.json" "$rc" "$C" "$DIM" <<PY
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"schema":"safe-c2-static-aabb-device-canary-terminal-v1","status":"FAILED","exit_code":int(sys.argv[2]),"context":sys.argv[3],"projection_dims":int(sys.argv[4]),"formal_claim_eligible":False},sort_keys=True,indent=2)+"\n")
PY
exit $rc' ERR
printf '%s\n' "$P" >"$RUN/preflight.json";printf '%s\n' "$G" >"$RUN/gpu_before.csv";printf '%s\n' "$A" >"$RUN/gpu_processes_before.txt"
CUDA_VISIBLE_DEVICES=0 "$BIN" --base "$BASE" --queries "$QUERIES" --outdir "$RUN" --projection-dims "$DIM" --query-start 0 --query-count 8 >"$RUN/stdout.log" 2>"$RUN/stderr.log"
C=verify
python3 - "$RUN/result.json" "$DIM" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));m=int(sys.argv[2])
assert x['status']=='PASS_STATIC_AABB_DEVICE_CERTIFICATE' and x['projection']['dimensions']==m
assert x['host_cover']['violations']==0 and x['device_rounding']['violations']==0 and x['snapshot']['host_to_device_to_host_byte_equal'] is True
PY
cmp "$RUN/aabb_lo_host.bin" "$RUN/aabb_lo_d2h.bin";cmp "$RUN/aabb_hi_host.bin" "$RUN/aabb_hi_d2h.bin";cmp "$RUN/aabb_dims_host.bin" "$RUN/aabb_dims_d2h.bin"
sha256sum "$RUN"/aabb_*_host.bin "$RUN"/aabb_*_d2h.bin >"$RUN/snapshot_sha256.txt";nvidia-smi -i 0 --query-gpu="$Q" --format=csv,noheader,nounits >"$RUN/gpu_after.csv"
python3 - "$RUN/manifest.json" "$RUN" "$BIN" "$DIM" <<'PY'
import hashlib,json,pathlib,sys
o,r,b=map(pathlib.Path,sys.argv[1:4]);m=int(sys.argv[4])
def h(p):
 x=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):x.update(z)
 return x.hexdigest()
ns=['preflight.json','gpu_before.csv','gpu_processes_before.txt','stdout.log','stderr.log','result.json','snapshot_sha256.txt','gpu_after.csv']+[p.name for p in r.glob('aabb_*.bin')]
o.write_text(json.dumps({'schema':'safe-c2-static-aabb-device-canary-manifest-v1','status':'COMPLETE_CANARY_ONLY','scope':'128-d static AABB device lower-bound certificate canary; no traversal/timing/formal claim','projection_dims':m,'binary':{'path':str(b),'sha256':h(b)},'artifacts':{n:{'sha256':h(r/n),'bytes':(r/n).stat().st_size} for n in ns},'gpu_binary_executed':True,'formal_claim_eligible':False},sort_keys=True,indent=2)+'\n')
PY
trap - ERR
echo "$RUN"
