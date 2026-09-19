#!/bin/sh
# Root-private sanitizer trampoline. The operational payload below is never
# interpreted in this first shell: it is re-executed under env -i before any
# hash, filesystem, compiler, CUDA, or GPU-control operation can occur.
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent /bin/bash --noprofile --norc -s -- "$@" <<'G3_CLEAN_PAYLOAD'
#!/bin/bash
# This payload is entered only through the root-private env -i trampoline.
# A malicious root can replace root-owned files; that is outside this release's
# trusted-control-plane model. This check prevents accidental/direct shell
# environment pollution from becoming build/run provenance.
require_clean_payload_environment() {
  local entry path_count=0 home_count=0 pwd_count=0 shlvl_count=0 underscore_count=0
  while IFS= read -r -d '' entry; do
    case "$entry" in
      PATH=/usr/bin:/bin) path_count=$((path_count + 1)) ;;
      HOME=/nonexistent) home_count=$((home_count + 1)) ;;
      PWD=*) [[ "${entry#PWD=}" == "$(pwd -P)" ]] || {
        echo "unsafe trampoline PWD" >&2; exit 69; }
        pwd_count=$((pwd_count + 1)) ;;
      # Bash creates SHLVL after env -i. Bash may emit canonical 0 in a
      # nested `-s` trampoline; it is shell metadata, not caller provenance.
      SHLVL=*) [[ "${entry#SHLVL=}" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo "unsafe trampoline SHLVL" >&2; exit 69; }
        shlvl_count=$((shlvl_count + 1)) ;;
      _=*) underscore_count=$((underscore_count + 1)) ;;
      *) echo "unapproved payload environment variable: ${entry%%=*}" >&2; exit 69 ;;
    esac
  done < <(/usr/bin/env -0)
  [[ "$path_count" -eq 1 && "$home_count" -eq 1 && "$pwd_count" -eq 1 &&
     "$shlvl_count" -eq 1 && "$underscore_count" -eq 1 ]] || {
    echo "payload environment is not the exact env-i trampoline contract" >&2; exit 69; }
}
clean_tool() {
  /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent "$@"
}
require_clean_payload_environment
# Body reached only through the root-private env-i trampoline.
set -euo pipefail
umask 077
ROOT="/workspace/experiments/tide_safe_c1_20260727/safe_c1_native_matrix_g3_v8_loader_allowlist_pilot"
LAUNCHER="$ROOT/tools/run_g3_4k_single_tu.sh"
BODY="$ROOT/tools/.run_g3_4k_single_tu.body.sh"
BINARY="$ROOT/preflight/bin/g3_4k_pilot_wrapper"
ADMISSION="$ROOT/preflight/admissions/g3_4k_pilot_wrapper.admission"
RUNS="$ROOT/runs"
AUTH="$ROOT/preflight/gpu_authorizations"
PENDING="$AUTH/pending"
CONSUMED="$AUTH/consumed"
LEASES="$ROOT/preflight/gpu_leases"
TOKEN="$PENDING/g3_4k_pilot.token"
PROOF="$PENDING/g3_4k_pilot.prelaunch"
NVIDIA_SMI="/usr/bin/nvidia-smi"
fail() { echo "guarded run refused: $*" >&2; exit 69; }
usage() { echo "usage: run_g3_4k_single_tu.sh --run-name <authorized-safe-name>" >&2; exit 69; }
trim() { printf '%s' "$1" | /usr/bin/sed 's/^[[:space:]]*//;s/[[:space:]]*$//'; }
boot_ns() { /usr/bin/python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_BOOTTIME))'; }
sha() { /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'; }
private_dir() {
  local p="$1" x
  [[ -d "$p" && ! -L "$p" && "$(/usr/bin/readlink -f "$p")" == "$p" ]] || fail "unsafe directory $p"
  x="$("/usr/bin/stat" -c '%u:%g:%a' "$p")"
  [[ "$x" == "0:0:700" ]] || fail "directory is not root-private 0700 $p"
}
private_file() {
  local p="$1" x
  [[ -f "$p" && ! -L "$p" ]] || fail "unsafe file $p"
  x="$("/usr/bin/stat" -c '%u:%g:%a:%h' "$p")"
  [[ "$x" == "0:0:600:1" ]] || fail "file is not root-private single-link 0600 $p"
}
private_exec() {
  local p="$1" x
  [[ -f "$p" && ! -L "$p" ]] || fail "unsafe executable $p"
  x="$("/usr/bin/stat" -c '%u:%g:%a:%h' "$p")"
  [[ "$x" == "0:0:700:1" ]] || fail "executable is not root-private single-link 0700 $p"
}
field() {
  local key="$1" value
  value="$("/usr/bin/awk" -F= -v k="$key" '
    $1 == k { if (++n != 1) exit 70; v=substr($0,length(k)+2) }
    END { if (n == 1 && v != "") print v; else exit 69 }
  ' "$TOKEN")" || fail "token lacks unique field $key"
  [[ "$value" != *[[:space:]]* ]] || fail "token field has whitespace $key"
  printf '%s' "$value"
}
admission_field() {
  local key="$1" value
  value="$("/usr/bin/awk" -F= -v k="$key" '
    $1 == k { if (++n != 1) exit 70; v=substr($0,length(k)+2) }
    END { if (n == 1 && v != "") print v; else exit 69 }
  ' "$ADMISSION")" || fail "admission lacks unique field $key"
  printf '%s' "$value"
}
gpu_idle() {
  local got ord uuid pci extra raw
  got="$(clean_tool "$NVIDIA_SMI" --id="$GPU_ORD" --query-gpu=index,uuid,pci.bus_id --format=csv,noheader,nounits)" ||
    fail "nvidia-smi identity query failed"
  [[ "$got" != *$'\n'* ]] || fail "GPU identity query is ambiguous"
  IFS=, read -r ord uuid pci extra <<< "$got"
  ord="$(trim "$ord")"; uuid="$(trim "$uuid")"; pci="$(trim "$pci")"; extra="$(trim "$extra")"
  [[ -z "$extra" && "$ord" == "$GPU_ORD" && "$uuid" == "$GPU_UUID" && "$pci" == "$GPU_PCI" ]] ||
    fail "GPU ordinal/UUID/PCI identity drift"
  raw="$(clean_tool "$NVIDIA_SMI" --id="$GPU_ORD" --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>&1)" ||
    fail "nvidia-smi compute-process query failed"
  raw="$(trim "$raw")"
  [[ -z "$raw" || "$raw" == "No running compute processes found" ]] ||
    fail "target GPU has a compute process; no process was touched"
}
[[ "$#" -eq 2 && "$1" == "--run-name" ]] || usage
NAME="$2"
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]] || fail "unsafe run name"
RUN_DIR="$RUNS/$NAME"
for p in "$ROOT" "$RUNS" "$AUTH" "$PENDING" "$CONSUMED" "$LEASES"; do private_dir "$p"; done
for p in "$LAUNCHER" "$BINARY"; do private_exec "$p"; done
for p in "$BODY" "$ADMISSION" "$TOKEN"; do private_file "$p"; done
[[ -x "$NVIDIA_SMI" && ! -e "$PROOF" && ! -e "$RUN_DIR" && ! -e "$RUNS/.staging-$NAME" ]] ||
  fail "run prerequisite/proof/output state is unsafe"
TOKEN_SHA="$(sha "$TOKEN")"
NONCE="$(field token_nonce)"
GPU_ORD="$(field gpu_ordinal)"
GPU_UUID="$(field gpu_uuid)"
GPU_PCI="$(field gpu_pci_bus_id)"
ISSUED="$(field issued_boottime_ns)"
EXPIRES="$(field expires_boottime_ns)"
NOW="$(boot_ns)"
[[ "$NONCE" =~ ^[0-9a-f]{64}$ && "$GPU_ORD" =~ ^(0|[1-9][0-9]*)$ &&
   "$GPU_UUID" =~ ^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ &&
   "$GPU_PCI" =~ ^[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]$ &&
   "$ISSUED" =~ ^[0-9]+$ && "$EXPIRES" =~ ^[0-9]+$ && "$NOW" -ge "$ISSUED" && "$NOW" -le "$EXPIRES" ]] ||
  fail "token GPU/nonce/time fields are invalid"
[[ "$(field schema)" == "safe-c1-g3-gpu-launch-token-v1" &&
   "$(field release_root)" == "$ROOT" &&
   "$(field host_name)" == "$(/bin/hostname)" &&
   "$(field host_boot_id)" == "$(/bin/cat /proc/sys/kernel/random/boot_id)" &&
   "$(field run_name)" == "$NAME" &&
   "$(field run_dir)" == "$RUN_DIR" &&
   "$(field admission_path)" == "$ADMISSION" &&
   "$(field binary_path)" == "$BINARY" &&
   "$(field admission_sha256)" == "$(sha "$ADMISSION")" &&
   "$(field binary_sha256)" == "$(sha "$BINARY")" &&
   "$(field wrapper_sha256)" == "$(sha "$ROOT/runner/g3_4k_pilot_wrapper.cu")" &&
   "$(field run_launcher_sha256)" == "$(sha "$LAUNCHER")" &&
   "$(field run_body_sha256)" == "$(sha "$BODY")" &&
   "$(field source_closure_descriptor_sha256)" == "$(admission_field source_closure_descriptor_sha256)" &&
   "$(field pre_native_host_loader_descriptor_sha256)" == "$(admission_field pre_native_host_loader_descriptor_sha256)" &&
   "$(field idle_sample_count)" == "3" && "$(field idle_sample_interval_seconds)" == "2" ]] ||
  fail "token is not bound to current controlled launch material"
LEASE="$LEASES/$GPU_UUID"
[[ ! -e "$LEASE" ]] || fail "existing GPU lease; never remove another control-plane lease"
mkdir "$LEASE" || fail "cannot acquire G3 GPU lease"
chmod 700 "$LEASE"
printf 'token_nonce=%s\nrun_name=%s\nholder_pid=%s\n' "$NONCE" "$NAME" "$$" > "$LEASE/holder"
chmod 600 "$LEASE/holder"
cleanup() { /bin/rm -f "$LEASE/holder"; rmdir "$LEASE" 2>/dev/null || true; }
trap cleanup EXIT
trap 'cleanup; exit 69' HUP INT TERM
SAMPLES=""
for i in 1 2 3; do
  gpu_idle
  SAMPLES="$SAMPLES""sample-$i-boot-ns=$(boot_ns)-compute-pids-none\n"
  [[ "$i" -eq 3 ]] || /bin/sleep 2
done
gpu_idle
FINAL_NOW="$(boot_ns)"
[[ "$FINAL_NOW" -ge "$ISSUED" && "$FINAL_NOW" -le "$EXPIRES" ]] ||
  fail "token expired before prelaunch proof publication"
SAMPLES="$SAMPLES""final-boot-ns=$FINAL_NOW-compute-pids-none
"
IDLE_SHA="$(printf '%b' "$SAMPLES" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"
TMP="$PENDING/.prelaunch.$NONCE.$$"
( umask 077; : > "$TMP" ) || fail "cannot stage prelaunch proof"
printf '%s\n' \
  "schema=safe-c1-g3-prelaunch-proof-v1" \
  "token_sha256=$TOKEN_SHA" "token_nonce=$NONCE" \
  "host_name=$(/bin/hostname)" "host_boot_id=$(/bin/cat /proc/sys/kernel/random/boot_id)" \
  "observed_boottime_ns=$(boot_ns)" \
  "gpu_ordinal=$GPU_ORD" "gpu_uuid=$GPU_UUID" "gpu_pci_bus_id=$GPU_PCI" \
  "run_name=$NAME" "run_dir=$RUN_DIR" "prelaunch_idle_check_sha256=$IDLE_SHA" \
  "scope=issuer_and_run_body_prelaunch_sampling_not_hardware_reservation" > "$TMP"
chmod 600 "$TMP"
/bin/ln "$TMP" "$PROOF" || { /bin/rm -f "$TMP"; fail "cannot atomically publish prelaunch proof"; }
/bin/rm -f "$TMP"
private_file "$PROOF"
set +e
/usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  "$BINARY" --run-dir "$RUN_DIR" --admission "$ADMISSION"
STATUS=$?
set -e
exit "$STATUS"

G3_CLEAN_PAYLOAD
exit 69
