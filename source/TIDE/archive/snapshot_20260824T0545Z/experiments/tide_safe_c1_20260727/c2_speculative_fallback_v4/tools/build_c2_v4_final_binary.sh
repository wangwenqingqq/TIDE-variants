#!/usr/bin/env bash
# Build only: the Safe-C2 v4 CUDA binary is never invoked by this script.
# The final workload admission has already passed CPU-only checks. This build
# merely creates a pinned executable required for a later parser-only gate.
set -Eeuo pipefail
umask 077

ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4
NVCC=/usr/local/cuda-13.1/bin/nvcc
CUDA_INCLUDE=/usr/local/cuda-13.1/include
SRC="$ROOT/src/gts_speculative_fallback_v4_sift1m.cu"
INVENTORY="$ROOT/provenance/source_only_inventory_v1.json"
MANIFEST="$ROOT/inputs/final_workload_v1/workload_manifest.json"
ADMISSION_AUDIT="$ROOT/provenance/c2_v4_oracle_admission_independent_audit_v1.json"
BUILDER="$ROOT/tools/build_c2_v4_final_binary.sh"
BIN_DIR="$ROOT/bin"
BIN="$BIN_DIR/GTS_safe_c2_speculative_fallback_v4_sift1m"
LOG="$ROOT/provenance/c2_v4_final_binary_build_v1.log"
CARD="$ROOT/provenance/c2_v4_final_binary_build_v1.json"
TMP=/tmp/GTS_safe_c2_v4_final_binary.$$

cleanup() { rm -f -- "$TMP" || true; }
trap cleanup EXIT

for path in "$NVCC" "$CUDA_INCLUDE/cuda_runtime.h" "$SRC" "$INVENTORY" "$MANIFEST" "$ADMISSION_AUDIT" "$BUILDER"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing/symlink build input: $path" >&2; exit 64; }
done
for path in "$BIN_DIR" "$BIN" "$LOG" "$CARD"; do
  [[ ! -e "$path" && ! -L "$path" ]] || { echo "refusing pre-existing build output: $path" >&2; exit 65; }
done
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "6373431c50806e7c6a993282722e58f5453a01d8a8e9d17cae27018302eaccd7" ]] || {
  echo "final workload manifest hash changed" >&2; exit 66;
}
python3 - "$ROOT" "$INVENTORY" "$ADMISSION_AUDIT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,inventory,audit=map(Path,sys.argv[1:])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
x=json.load(open(inventory))
if x.get('status')!='SOURCE_ONLY_PRE_GT_NO_GPU_NO_BINARY_NO_LEDGER':
 raise SystemExit('source inventory identity mismatch')
for e in x.get('entries',[]):
 p=Path(e['path'])
 if not p.is_file() or p.is_symlink() or sha(p)!=e['sha256'] or p.stat().st_size!=e['bytes']:
  raise SystemExit('source inventory mismatch: '+str(p))
a=json.load(open(audit))
if a.get('status')!='PASS_CPU_ONLY_ORACLE_ADMISSION_INDEPENDENT_AUDIT':
 raise SystemExit('CPU admission audit not PASS')
print('PASS_C2_V4_CORE_SOURCE_AND_CPU_ADMISSION_BINDING')
PY

{
  echo "started_utc=$(date -u -Is)"
  echo "scope=NVCC compilation only; final CUDA binary is not executed; no GPU management command"
  echo "compiler=$($NVCC --version | tail -n 1)"
  "$NVCC" -std=c++17 -O3 --generate-code=arch=compute_120,code=[compute_120,sm_120] \
    -I"$CUDA_INCLUDE" -I"$ROOT/include" "$SRC" -o "$TMP"
  mkdir "$BIN_DIR"
  mv "$TMP" "$BIN"
  chmod 0750 "$BIN"
  echo "binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')"
  echo "binary_bytes=$(stat -c %s "$BIN")"
  echo "finished_utc=$(date -u -Is)"
} >"$LOG" 2>&1

python3 - "$CARD" "$ROOT" "$SRC" "$INVENTORY" "$MANIFEST" "$ADMISSION_AUDIT" "$BUILDER" "$BIN" "$LOG" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
card,root,src,inventory,manifest,audit,builder,binary,log=map(Path,sys.argv[1:])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
obj={
 'schema':'safe-c2-v4-final-binary-build-v1',
 'status':'COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION',
 'scope':'NVCC compilation only. The resulting binary must still pass a separate final manifest parser-only receipt gate before any formal ledger or GPU stage.',
 'canonical_root':str(root),
 'source':{'path':str(src),'sha256':sha(src)},
 'source_inventory':{'path':str(inventory),'sha256':sha(inventory)},
 'workload_manifest':{'path':str(manifest),'sha256':sha(manifest)},
 'cpu_admission_independent_audit':{'path':str(audit),'sha256':sha(audit)},
 'builder':{'path':str(builder),'sha256':sha(builder)},
 'binary':{'path':str(binary),'sha256':sha(binary),'bytes':binary.stat().st_size},
 'build_log':{'path':str(log),'sha256':sha(log)},
 'compile_flags':'-std=c++17 -O3 --generate-code=arch=compute_120,code=[compute_120,sm_120] -I/usr/local/cuda-13.1/include -I<root>/include',
 'gpu_binary_executed':False,
 'nvidia_smi_called':False,
 'formal_claim_eligible':False,
}
fd=os.open(card,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n')
PY
printf '%s\n' "$CARD"
