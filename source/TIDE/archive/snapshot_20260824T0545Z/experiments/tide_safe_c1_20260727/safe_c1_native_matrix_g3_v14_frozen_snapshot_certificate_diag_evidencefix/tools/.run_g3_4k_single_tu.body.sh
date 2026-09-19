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
ROOT="/workspace/experiments/tide_safe_c1_20260727/safe_c1_native_matrix_g3_v14_frozen_snapshot_certificate_diag_evidencefix"
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
NVML_HELPER="$ROOT/tools/.g3_nvml_snapshot.py"
NVML_HELPER_SHA="d5d57ccd816f377209f6a4f2709f75cb030e65739423b8967c126197513891c8"
NVML_SNAPSHOT_SCHEMA="safe-c1-g3-nvml-snapshot-v1"
NVML_PYTHON_REALPATH="/usr/bin/python3.12"
NVML_PYTHON_SHA="1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
NVML_LIBRARY_REALPATH="/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.590.48.01"
NVML_LIBRARY_SHA="12f3bcd4ba447599a2077297e3a4ff4288205b082c2e76346718ae85c058c4b2"
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
assert_nvml_helper_integrity() {
  private_file "$NVML_HELPER"
  [[ "$(sha "$NVML_HELPER")" == "$NVML_HELPER_SHA" ]] ||
    fail "NVML helper bytes differ from the fixed admission binding"
}
parse_nvml_snapshot() {
  local ordinal="$1" got
  local -a rows
  assert_nvml_helper_integrity
  got="$(clean_tool /usr/bin/python3 -I -S "$NVML_HELPER" --gpu-ordinal "$ordinal")" ||
    fail "NVML snapshot helper refused the requested ordinal"
  [[ "$got" != *$'\r'* ]] || fail "NVML snapshot has carriage return"
  mapfile -t rows < <(printf '%s\n' "$got")
  [[ "${#rows[@]}" -eq 10 ]] || fail "NVML snapshot field count is not canonical"
  [[ "${rows[0]}" == "schema=$NVML_SNAPSHOT_SCHEMA" &&
     "${rows[1]}" == gpu_ordinal=* && "${rows[2]}" == gpu_uuid=* &&
     "${rows[3]}" == gpu_pci_bus_id=* &&
     "${rows[4]}" == compute_process_count=* &&
     "${rows[5]}" == python_realpath=* && "${rows[6]}" == python_sha256=* &&
     "${rows[7]}" == nvml_library_realpath=* &&
     "${rows[8]}" == nvml_library_sha256=* &&
     "${rows[9]}" == nvml_driver_version=* ]] ||
    fail "NVML snapshot keys/order differ from the fixed protocol"
  SNAP_ORD="${rows[1]#gpu_ordinal=}"
  SNAP_UUID="${rows[2]#gpu_uuid=}"
  SNAP_PCI="${rows[3]#gpu_pci_bus_id=}"
  SNAP_PROCESS_COUNT="${rows[4]#compute_process_count=}"
  SNAP_PYTHON_REALPATH="${rows[5]#python_realpath=}"
  SNAP_PYTHON_SHA="${rows[6]#python_sha256=}"
  SNAP_NVML_LIBRARY_REALPATH="${rows[7]#nvml_library_realpath=}"
  SNAP_NVML_LIBRARY_SHA="${rows[8]#nvml_library_sha256=}"
  SNAP_DRIVER_VERSION="${rows[9]#nvml_driver_version=}"
  for value in "$SNAP_ORD" "$SNAP_UUID" "$SNAP_PCI" "$SNAP_PROCESS_COUNT" \
               "$SNAP_PYTHON_REALPATH" "$SNAP_PYTHON_SHA" \
               "$SNAP_NVML_LIBRARY_REALPATH" "$SNAP_NVML_LIBRARY_SHA" "$SNAP_DRIVER_VERSION"; do
    [[ -n "$value" && "$value" != *[[:space:]]* && "$value" != *=* ]] ||
      fail "NVML snapshot has an empty/unsafe value"
  done
  [[ "$SNAP_ORD" =~ ^(0|[1-9][0-9]*)$ &&
     "$SNAP_UUID" =~ ^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ &&
     "$SNAP_PCI" =~ ^[0-9a-fA-F]{8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$ &&
     "$SNAP_PROCESS_COUNT" == "0" &&
     "$SNAP_PYTHON_REALPATH" == "$NVML_PYTHON_REALPATH" &&
     "$SNAP_PYTHON_SHA" == "$NVML_PYTHON_SHA" &&
     "$SNAP_NVML_LIBRARY_REALPATH" == "$NVML_LIBRARY_REALPATH" &&
     "$SNAP_NVML_LIBRARY_SHA" == "$NVML_LIBRARY_SHA" &&
     "$SNAP_DRIVER_VERSION" =~ ^[0-9]+(\.[0-9]+){1,3}$ ]] ||
    fail "NVML snapshot identity/library/idle contract is invalid"
}
gpu_idle() {
  parse_nvml_snapshot "$GPU_ORD"
  [[ "$SNAP_ORD" == "$GPU_ORD" && "$SNAP_UUID" == "$GPU_UUID" && "$SNAP_PCI" == "$GPU_PCI" &&
     "$SNAP_PYTHON_REALPATH" == "$TOKEN_NVML_PYTHON_REALPATH" &&
     "$SNAP_PYTHON_SHA" == "$TOKEN_NVML_PYTHON_SHA" &&
     "$SNAP_NVML_LIBRARY_REALPATH" == "$TOKEN_NVML_LIBRARY_REALPATH" &&
     "$SNAP_NVML_LIBRARY_SHA" == "$TOKEN_NVML_LIBRARY_SHA" &&
     "$SNAP_DRIVER_VERSION" == "$TOKEN_NVML_DRIVER_VERSION" ]] ||
    fail "NVML identity/fingerprint differs from the authorized token"
}
[[ "$#" -eq 2 && "$1" == "--run-name" ]] || usage
NAME="$2"
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]] || fail "unsafe run name"
RUN_DIR="$RUNS/$NAME"
for p in "$ROOT" "$RUNS" "$AUTH" "$PENDING" "$CONSUMED" "$LEASES"; do private_dir "$p"; done
for p in "$LAUNCHER" "$BINARY"; do private_exec "$p"; done
for p in "$BODY" "$ADMISSION" "$TOKEN" "$NVML_HELPER"; do private_file "$p"; done
[[ ! -e "$PROOF" && ! -e "$RUN_DIR" && ! -e "$RUNS/.staging-$NAME" ]] ||
  fail "run prerequisite/proof/output state is unsafe"
assert_nvml_helper_integrity
TOKEN_SHA="$(sha "$TOKEN")"
NONCE="$(field token_nonce)"
GPU_ORD="$(field gpu_ordinal)"
GPU_UUID="$(field gpu_uuid)"
GPU_PCI="$(field gpu_pci_bus_id)"
ISSUED="$(field issued_boottime_ns)"
EXPIRES="$(field expires_boottime_ns)"
TOKEN_NVML_SCHEMA="$(field nvml_snapshot_schema)"
TOKEN_NVML_HELPER_PATH="$(field nvml_snapshot_helper_path)"
TOKEN_NVML_HELPER_SHA="$(field nvml_snapshot_helper_sha256)"
TOKEN_NVML_PYTHON_REALPATH="$(field nvml_python_realpath)"
TOKEN_NVML_PYTHON_SHA="$(field nvml_python_sha256)"
TOKEN_NVML_LIBRARY_REALPATH="$(field issuer_nvml_library_realpath)"
TOKEN_NVML_LIBRARY_SHA="$(field issuer_nvml_library_sha256)"
TOKEN_NVML_DRIVER_VERSION="$(field issuer_nvml_driver_version)"
NOW="$(boot_ns)"
[[ "$NONCE" =~ ^[0-9a-f]{64}$ && "$GPU_ORD" =~ ^(0|[1-9][0-9]*)$ &&
   "$GPU_UUID" =~ ^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ &&
   "$GPU_PCI" =~ ^[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]$ &&
   "$ISSUED" =~ ^[0-9]+$ && "$EXPIRES" =~ ^[0-9]+$ && "$NOW" -ge "$ISSUED" && "$NOW" -le "$EXPIRES" ]] ||
  fail "token GPU/nonce/time fields are invalid"
[[ "$(field schema)" == "safe-c1-g3-gpu-launch-token-v2" &&
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
   "$TOKEN_NVML_SCHEMA" == "$(admission_field nvml_snapshot_schema)" &&
   "$TOKEN_NVML_HELPER_PATH" == "$NVML_HELPER" && "$TOKEN_NVML_HELPER_SHA" == "$NVML_HELPER_SHA" &&
   "$TOKEN_NVML_PYTHON_REALPATH" == "$(admission_field nvml_python_realpath)" &&
   "$TOKEN_NVML_PYTHON_SHA" == "$(admission_field nvml_python_sha256)" &&
   "$TOKEN_NVML_LIBRARY_REALPATH" == "$(admission_field nvml_library_realpath)" &&
   "$TOKEN_NVML_LIBRARY_SHA" == "$(admission_field nvml_library_sha256)" &&
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
  SAMPLES="$SAMPLES""sample-$i-boot-ns=$(boot_ns)-schema=$TOKEN_NVML_SCHEMA-uuid=$GPU_UUID-pci=$GPU_PCI-compute-process-count=0-helper-sha=$TOKEN_NVML_HELPER_SHA-python=$SNAP_PYTHON_REALPATH-python-sha=$SNAP_PYTHON_SHA-nvml-library=$SNAP_NVML_LIBRARY_REALPATH-nvml-library-sha=$SNAP_NVML_LIBRARY_SHA-driver=$SNAP_DRIVER_VERSION\n"
  [[ "$i" -eq 3 ]] || /bin/sleep 2
done
gpu_idle
FINAL_NOW="$(boot_ns)"
[[ "$FINAL_NOW" -ge "$ISSUED" && "$FINAL_NOW" -le "$EXPIRES" ]] ||
  fail "token expired before prelaunch proof publication"
SAMPLES="$SAMPLES""final-boot-ns=$FINAL_NOW-schema=$TOKEN_NVML_SCHEMA-uuid=$GPU_UUID-pci=$GPU_PCI-compute-process-count=0-helper-sha=$TOKEN_NVML_HELPER_SHA-python=$SNAP_PYTHON_REALPATH-python-sha=$SNAP_PYTHON_SHA-nvml-library=$SNAP_NVML_LIBRARY_REALPATH-nvml-library-sha=$SNAP_NVML_LIBRARY_SHA-driver=$SNAP_DRIVER_VERSION
"
IDLE_SHA="$(printf '%b' "$SAMPLES" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"
TMP="$PENDING/.prelaunch.$NONCE.$$"
( umask 077; : > "$TMP" ) || fail "cannot stage prelaunch proof"
printf '%s\n' \
  "schema=safe-c1-g3-prelaunch-proof-v2" \
  "token_sha256=$TOKEN_SHA" "token_nonce=$NONCE" \
  "host_name=$(/bin/hostname)" "host_boot_id=$(/bin/cat /proc/sys/kernel/random/boot_id)" \
  "observed_boottime_ns=$(boot_ns)" \
  "gpu_ordinal=$GPU_ORD" "gpu_uuid=$GPU_UUID" "gpu_pci_bus_id=$GPU_PCI" \
  "run_name=$NAME" "run_dir=$RUN_DIR" "prelaunch_idle_check_sha256=$IDLE_SHA" \
  "nvml_snapshot_schema=$TOKEN_NVML_SCHEMA" "nvml_snapshot_helper_path=$TOKEN_NVML_HELPER_PATH" \
  "nvml_snapshot_helper_sha256=$TOKEN_NVML_HELPER_SHA" \
  "prelaunch_nvml_python_realpath=$SNAP_PYTHON_REALPATH" "prelaunch_nvml_python_sha256=$SNAP_PYTHON_SHA" \
  "prelaunch_nvml_library_realpath=$SNAP_NVML_LIBRARY_REALPATH" "prelaunch_nvml_library_sha256=$SNAP_NVML_LIBRARY_SHA" \
  "prelaunch_nvml_driver_version=$SNAP_DRIVER_VERSION" \
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
