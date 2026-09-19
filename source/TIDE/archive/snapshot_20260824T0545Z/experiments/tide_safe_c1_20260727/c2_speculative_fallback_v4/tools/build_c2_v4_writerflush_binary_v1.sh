#!/usr/bin/env bash
# Compile-only provenance writerflush successor; no CUDA binary execution.
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4
NVCC=/usr/local/cuda-13.1/bin/nvcc
SRC="$ROOT/src/gts_speculative_fallback_v4_writerflush_v1.cu"
AUDIT="$ROOT/tools/audit_c2_v4_writerflush_source_v1.py"
AUDIT_RESULT="$ROOT/provenance/c2_v4_writerflush_source_audit_v1.json"
SELF="$ROOT/tools/build_c2_v4_writerflush_binary_v1.sh"
BIN="$ROOT/bin/GTS_safe_c2_speculative_fallback_v4_writerflush_v1"
LOG="$ROOT/provenance/c2_v4_writerflush_binary_build_v1.log"
CARD="$ROOT/provenance/c2_v4_writerflush_binary_build_v1.json"
TMP=/tmp/GTS_safe_c2_v4_writerflush_binary.$$
cleanup(){ rm -f -- "$TMP" /tmp/c2_v4_writerflush_audit.$$.json || true; }
trap cleanup EXIT
for p in "$NVCC" "$SRC" "$AUDIT" "$AUDIT_RESULT" "$SELF"; do
 [[ -f "$p" && ! -L "$p" ]] || { echo "missing/symlink build input: $p" >&2; exit 65; }
done
for p in "$BIN" "$LOG" "$CARD"; do
 [[ ! -e "$p" && ! -L "$p" ]] || { echo "pre-existing build output: $p" >&2; exit 66; }
done
PYTHONDONTWRITEBYTECODE=1 python3 "$AUDIT" >/tmp/c2_v4_writerflush_audit.$$.json
[[ "$(sha256sum "$AUDIT_RESULT" | awk '{print $1}')" == "$(sha256sum /tmp/c2_v4_writerflush_audit.$$.json | awk '{print $1}')" ]] || {
 echo "source audit result changed" >&2; exit 67; }
{
 echo "started_utc=$(date -u -Is)"
 echo "scope=NVCC compilation only; writerflush successor is not executed; no GPU management command"
 echo "compiler=$($NVCC --version | tail -n 1)"
 "$NVCC" -std=c++17 -O3 --generate-code=arch=compute_120,code=[compute_120,sm_120] \
   -I/usr/local/cuda-13.1/include -I"$ROOT/include" "$SRC" -o "$TMP"
 mv "$TMP" "$BIN"
 chmod 0750 "$BIN"
 echo "binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')"
 echo "binary_bytes=$(stat -c %s "$BIN")"
 echo "finished_utc=$(date -u -Is)"
} >"$LOG" 2>&1
python3 - "$CARD" "$ROOT" "$SRC" "$AUDIT" "$AUDIT_RESULT" "$SELF" "$BIN" "$LOG" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
card,root,src,audit,audit_result,builder,binary,log=map(Path,sys.argv[1:])
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
x={"schema":"safe-c2-v4-writerflush-binary-build-v1","status":"COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION","scope":"NVCC compilation only. This successor fixes completed-JSONL FNV provenance. It is not authorized for any existing calibration/validation/sealed split.","canonical_root":str(root),"source":{"path":str(src),"sha256":sha(src)},"source_audit":{"path":str(audit),"sha256":sha(audit)},"source_audit_result":{"path":str(audit_result),"sha256":sha(audit_result)},"builder":{"path":str(builder),"sha256":sha(builder)},"binary":{"path":str(binary),"sha256":sha(binary),"bytes":binary.stat().st_size},"build_log":{"path":str(log),"sha256":sha(log)},"gpu_binary_executed":False,"nvidia_smi_called":False,"formal_claim_eligible":False}
fd=os.open(card,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
with os.fdopen(fd,"w",encoding="utf-8") as f:
 json.dump(x,f,sort_keys=True,indent=2);f.write("\n")
PY
printf '%s\n' "$CARD"
