#!/bin/bash
# One-shot controlled runner for the Safe-C3 4099 ancestor/fence gate.
set -euo pipefail
umask 077

ROOT="/workspace/experiments/tide_safe_c1_20260727/c3_safe_ancestor_fence_v1"
BIN="$ROOT/bin/GTS_safe_c3_ancestor_fence"
SRC="$ROOT/src/gts_safe_c3_ancestor_fence.cu"
PROTO="$ROOT/protocol/safe_c3_ancestor_fence_v1.json"
STATIC="$ROOT/preflight/static_audit_post_posixfix.json"
HELPER="$ROOT/tools/.nvml_snapshot.py"
VALIDATOR="$ROOT/tools/validate_safe_c3.py"
BUNDLE="/workspace/experiments/tide_safe_c1_20260727/runs/e1gi_b_quantized_gts_bundles_v1_20260727/sift128_slice__sift128_slice__seed_20260727"
AUTH="$ROOT/preflight/gpu_authorizations"
PENDING="$AUTH/pending"
CONSUMED="$AUTH/consumed"
LEASES="$ROOT/preflight/gpu_leases"
RUNS="$ROOT/runs"

fail() { echo "Safe-C3 guarded run refused: $*" >&2; exit 69; }
sha() { /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'; }
boot_ns() { /usr/bin/python3.12 -I -S -c 'import time; print(time.clock_gettime_ns(time.CLOCK_BOOTTIME))'; }
private_dir() {
  [[ -d "$1" && ! -L "$1" ]] || fail "unsafe directory $1"
  [[ "$(/usr/bin/stat -c '%u:%g:%a' "$1")" == "0:0:700" ]] || fail "non-private directory $1"
}
private_file() {
  [[ -f "$1" && ! -L "$1" ]] || fail "unsafe file $1"
  [[ "$(/usr/bin/stat -c '%u:%g:%a' "$1")" == "0:0:600" ||
     "$(/usr/bin/stat -c '%u:%g:%a' "$1")" == "0:0:700" ]] || fail "non-private file $1"
}
usage() { echo "usage: run_safe_c3_gpu1_once.sh --run-name <authorized-name>" >&2; exit 69; }
field() {
  local key="$1" value
  value="$(/usr/bin/awk -F= -v k="$key" '$1==k {if(++n==1) print substr($0,length(k)+2)} END{if(n!=1) exit 1}' "$TOKEN")" ||
    fail "token field $key"
  [[ -n "$value" && "$value" != *[[:space:]]* ]] || fail "unsafe token field $key"
  printf '%s' "$value"
}
[[ "$#" -eq 2 && "$1" == "--run-name" ]] || usage
NAME="$2"
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]] || usage
TOKEN="$PENDING/$NAME.token"

for p in "$ROOT" "$ROOT/preflight" "$RUNS" "$ROOT/bin" "$ROOT/src" "$ROOT/tools" "$AUTH" "$PENDING" "$CONSUMED" "$LEASES"; do private_dir "$p"; done
for p in "$BIN" "$SRC" "$PROTO" "$STATIC" "$HELPER" "$VALIDATOR" "$TOKEN"; do private_file "$p"; done
[[ "$(grep -c '"status":"PASS_STATIC_AUDIT"' "$STATIC")" -eq 1 ]] || fail "static audit"
[[ "$(sha "$HELPER")" == "d5d57ccd816f377209f6a4f2709f75cb030e65739423b8967c126197513891c8" ]] ||
  fail "NVML helper fingerprint drift"

[[ "$(field schema)" == "safe-c3-gpu-token-v1" &&
   "$(field run_name)" == "$NAME" &&
   "$(field gpu_ordinal)" == "1" &&
   "$(field binary_path)" == "$BIN" &&
   "$(field binary_sha256)" == "$(sha "$BIN")" &&
   "$(field source_sha256)" == "$(sha "$SRC")" &&
   "$(field protocol_sha256)" == "$(sha "$PROTO")" &&
   "$(field static_audit_sha256)" == "$(sha "$STATIC")" &&
   "$(field nvml_helper_sha256)" == "$(sha "$HELPER")" &&
   "$(field validator_sha256)" == "$(sha "$VALIDATOR")" ]] || fail "token binding"
GPU_UUID="$(field gpu_uuid)"
GPU_PCI="$(field gpu_pci_bus_id)"
EXPIRES="$(field expires_boottime_ns)"
NOW="$(boot_ns)"
[[ "$GPU_UUID" =~ ^GPU-[0-9a-fA-F-]+$ &&
   "$GPU_PCI" =~ ^[0-9a-fA-F]{8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$ &&
   "$EXPIRES" =~ ^[0-9]+$ && "$NOW" -le "$EXPIRES" ]] || fail "token expiry/identity"
[[ ! -e "$RUNS/$NAME" && ! -e "$RUNS/.staging-$NAME" && ! -e "$CONSUMED/$NAME.token" ]] ||
  fail "run name state"

snapshot_gpu() {
  local output ordinal uuid pci count
  output="$(/usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent /usr/bin/python3.12 -I -S "$HELPER" --gpu-ordinal 1)" ||
    fail "idle NVML snapshot failed"
  ordinal="$(printf '%s\n' "$output" | /usr/bin/awk -F= '$1=="gpu_ordinal"{if(++n==1) print $2} END{if(n!=1) exit 1}')" ||
    fail "NVML ordinal"
  uuid="$(printf '%s\n' "$output" | /usr/bin/awk -F= '$1=="gpu_uuid"{if(++n==1) print $2} END{if(n!=1) exit 1}')" ||
    fail "NVML UUID"
  pci="$(printf '%s\n' "$output" | /usr/bin/awk -F= '$1=="gpu_pci_bus_id"{if(++n==1) print $2} END{if(n!=1) exit 1}')" ||
    fail "NVML PCI"
  count="$(printf '%s\n' "$output" | /usr/bin/awk -F= '$1=="compute_process_count"{if(++n==1) print $2} END{if(n!=1) exit 1}')" ||
    fail "NVML process count"
  [[ "$ordinal" == "1" && "$uuid" == "$GPU_UUID" && "$pci" == "$GPU_PCI" && "$count" == "0" ]] ||
    fail "GPU no longer idle or identity changed"
}
snapshot_gpu
/bin/sleep 2
snapshot_gpu
[[ "$(boot_ns)" -le "$EXPIRES" ]] || fail "token expired during prelaunch"

LEASE="$LEASES/$GPU_UUID"
[[ ! -e "$LEASE" ]] || fail "another controlled lease already exists"
mkdir "$LEASE"
chmod 700 "$LEASE"
cleanup() { rmdir "$LEASE" 2>/dev/null || true; }
trap cleanup EXIT HUP INT TERM

STAGING="$RUNS/.staging-$NAME"
mkdir "$STAGING"
chmod 700 "$STAGING"
TOKEN_SHA="$(sha "$TOKEN")"
# Consume before launching: this name cannot be reused after a partial/failed run.
mv "$TOKEN" "$CONSUMED/$NAME.token"
{
  printf 'schema=safe-c3-prelaunch-v1\n'
  printf 'run_name=%s\n' "$NAME"
  printf 'gpu_ordinal=1\n'
  printf 'gpu_uuid=%s\n' "$GPU_UUID"
  printf 'gpu_pci_bus_id=%s\n' "$GPU_PCI"
  printf 'token_sha256=%s\n' "$TOKEN_SHA"
  printf 'binary_sha256=%s\n' "$(sha "$BIN")"
  printf 'source_sha256=%s\n' "$(sha "$SRC")"
  printf 'prelaunch_boottime_ns=%s\n' "$(boot_ns)"
} > "$STAGING/prelaunch.txt"
chmod 600 "$STAGING/prelaunch.txt"

set +e
/usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent \
  LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/lib/x86_64-linux-gnu \
  CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  "$BIN" --bundle "$BUNDLE" --out "$STAGING/engine.jsonl" \
  --summary "$STAGING/summary.json" \
  --initial-snapshot "$STAGING/initial_snapshot.json" \
  --final-snapshot "$STAGING/final_snapshot.json" \
  --candidate-id 4099 --expected-leaf 790 \
  >"$STAGING/runner.stdout.log" 2>"$STAGING/runner.stderr.log"
RUN_STATUS=$?
set -e
printf '%s\n' "$RUN_STATUS" > "$STAGING/runner.exit_code"
chmod 600 "$STAGING"/*
[[ "$RUN_STATUS" -eq 0 ]] || fail "runner failed; preserved $STAGING"

/usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent \
  /usr/bin/python3.12 -I -S "$VALIDATOR" \
  --bundle "$BUNDLE" --initial "$STAGING/initial_snapshot.json" \
  --final "$STAGING/final_snapshot.json" --engine "$STAGING/engine.jsonl" \
  --summary "$STAGING/summary.json" >"$STAGING/independent_validation.json"
chmod 600 "$STAGING/independent_validation.json"
[[ "$(grep -c '"status":"PASS"' "$STAGING/independent_validation.json")" -eq 1 ]] ||
  fail "independent validator did not PASS"

(
  cd "$STAGING"
  /usr/bin/sha256sum prelaunch.txt runner.stdout.log runner.stderr.log runner.exit_code \
    initial_snapshot.json final_snapshot.json engine.jsonl summary.json \
    independent_validation.json > sha256sums.txt
  chmod 600 sha256sums.txt
)
mv "$STAGING" "$RUNS/$NAME"
printf 'SAFE_C3_GPU_RUN_PASS run_dir=%s\n' "$RUNS/$NAME"

