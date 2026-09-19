#!/bin/bash
# Controlled one-shot GPU token issuer for the Safe-C3 4099 ancestor/fence gate.
set -euo pipefail
umask 077

ROOT="/workspace/experiments/tide_safe_c1_20260727/c3_safe_ancestor_fence_v1"
BIN="$ROOT/bin/GTS_safe_c3_ancestor_fence"
SRC="$ROOT/src/gts_safe_c3_ancestor_fence.cu"
PROTO="$ROOT/protocol/safe_c3_ancestor_fence_v1.json"
STATIC="$ROOT/preflight/static_audit_post_posixfix.json"
HELPER="$ROOT/tools/.nvml_snapshot.py"
VALIDATOR="$ROOT/tools/validate_safe_c3.py"
AUTH="$ROOT/preflight/gpu_authorizations"
PENDING="$AUTH/pending"
CONSUMED="$AUTH/consumed"
LEASES="$ROOT/preflight/gpu_leases"

fail() { echo "Safe-C3 authorization refused: $*" >&2; exit 69; }
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
usage() {
  echo "usage: issue_safe_c3_gpu_token.sh --gpu-ordinal 1 --run-name <name> --approval-id <id> --ttl-seconds <1..120>" >&2
  exit 69
}

GPU=""; NAME=""; APPROVAL=""; TTL=""
while [[ "$#" -gt 0 ]]; do
  [[ "$#" -ge 2 ]] || usage
  case "$1" in
    --gpu-ordinal) GPU="$2" ;;
    --run-name) NAME="$2" ;;
    --approval-id) APPROVAL="$2" ;;
    --ttl-seconds) TTL="$2" ;;
    *) usage ;;
  esac
  shift 2
done
[[ "$GPU" == "1" &&
   "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ &&
   "$APPROVAL" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$ &&
   "$TTL" =~ ^[0-9]+$ && "$TTL" -ge 1 && "$TTL" -le 120 ]] || usage

for p in "$ROOT" "$ROOT/preflight" "$ROOT/runs" "$ROOT/bin" "$ROOT/src" "$ROOT/tools"; do private_dir "$p"; done
for p in "$BIN" "$SRC" "$PROTO" "$STATIC" "$HELPER" "$VALIDATOR"; do private_file "$p"; done
[[ "$(grep -c '"status":"PASS_STATIC_AUDIT"' "$STATIC")" -eq 1 ]] || fail "static audit is not a unique PASS"
[[ "$(sha "$HELPER")" == "d5d57ccd816f377209f6a4f2709f75cb030e65739423b8967c126197513891c8" ]] ||
  fail "NVML helper fingerprint drift"

install -d -m 700 "$AUTH" "$PENDING" "$CONSUMED" "$LEASES"
for p in "$AUTH" "$PENDING" "$CONSUMED" "$LEASES"; do private_dir "$p"; done
TOKEN="$PENDING/$NAME.token"
[[ ! -e "$TOKEN" && ! -e "$CONSUMED/$NAME.token" && ! -e "$ROOT/runs/$NAME" &&
   ! -e "$ROOT/runs/.staging-$NAME" ]] || fail "run name has existing state"

snapshot_gpu() {
  local output ordinal uuid pci count
  output="$(/usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent /usr/bin/python3.12 -I -S "$HELPER" --gpu-ordinal "$GPU")" ||
    fail "idle NVML snapshot failed"
  ordinal="$(printf '%s\n' "$output" | /usr/bin/awk -F= '$1=="gpu_ordinal"{if(++n==1) print $2} END{if(n!=1) exit 1}')" ||
    fail "NVML ordinal parse"
  uuid="$(printf '%s\n' "$output" | /usr/bin/awk -F= '$1=="gpu_uuid"{if(++n==1) print $2} END{if(n!=1) exit 1}')" ||
    fail "NVML UUID parse"
  pci="$(printf '%s\n' "$output" | /usr/bin/awk -F= '$1=="gpu_pci_bus_id"{if(++n==1) print $2} END{if(n!=1) exit 1}')" ||
    fail "NVML PCI parse"
  count="$(printf '%s\n' "$output" | /usr/bin/awk -F= '$1=="compute_process_count"{if(++n==1) print $2} END{if(n!=1) exit 1}')" ||
    fail "NVML process count parse"
  [[ "$ordinal" == "$GPU" && "$count" == "0" &&
     "$uuid" =~ ^GPU-[0-9a-fA-F-]+$ &&
     "$pci" =~ ^[0-9a-fA-F]{8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$ ]] ||
    fail "NVML idle identity contract"
  printf '%s %s\n' "$uuid" "$pci"
}

IDENTITY=""
for sample in 1 2 3; do
  current="$(snapshot_gpu)"
  if [[ -z "$IDENTITY" ]]; then IDENTITY="$current"; else [[ "$IDENTITY" == "$current" ]] || fail "GPU identity drift"; fi
  [[ "$sample" == "3" ]] || /bin/sleep 2
done
GPU_UUID="${IDENTITY%% *}"
GPU_PCI="${IDENTITY##* }"
ISSUED="$(boot_ns)"
EXPIRES=$((ISSUED + TTL * 1000000000))
NONCE="$("/usr/bin/od" -An -N32 -tx1 /dev/urandom | /usr/bin/tr -d ' \n')"
[[ "$NONCE" =~ ^[0-9a-f]{64}$ ]] || fail "nonce generation"
TMP="$PENDING/.token-$NAME-$NONCE"
{
  printf 'schema=safe-c3-gpu-token-v1\n'
  printf 'token_nonce=%s\n' "$NONCE"
  printf 'run_name=%s\n' "$NAME"
  printf 'gpu_ordinal=%s\n' "$GPU"
  printf 'gpu_uuid=%s\n' "$GPU_UUID"
  printf 'gpu_pci_bus_id=%s\n' "$GPU_PCI"
  printf 'issued_boottime_ns=%s\n' "$ISSUED"
  printf 'expires_boottime_ns=%s\n' "$EXPIRES"
  printf 'approval_id=%s\n' "$APPROVAL"
  printf 'binary_path=%s\n' "$BIN"
  printf 'binary_sha256=%s\n' "$(sha "$BIN")"
  printf 'source_sha256=%s\n' "$(sha "$SRC")"
  printf 'protocol_sha256=%s\n' "$(sha "$PROTO")"
  printf 'static_audit_sha256=%s\n' "$(sha "$STATIC")"
  printf 'nvml_helper_sha256=%s\n' "$(sha "$HELPER")"
  printf 'validator_sha256=%s\n' "$(sha "$VALIDATOR")"
  printf 'purpose=safe_c3_4099_ancestor_fence_correctness_gate\n'
} > "$TMP"
chmod 600 "$TMP"
/bin/ln "$TMP" "$TOKEN" || { /bin/rm -f "$TMP"; fail "atomic token publication"; }
/bin/rm -f "$TMP"
private_file "$TOKEN"
printf 'SAFE_C3_GPU_TOKEN_ISSUED run_name=%s gpu_ordinal=%s expires_boottime_ns=%s\n' \
  "$NAME" "$GPU" "$EXPIRES"

