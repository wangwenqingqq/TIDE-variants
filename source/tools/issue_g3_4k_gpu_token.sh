#!/bin/sh
# Root-private sanitizer trampoline. Issuing a token is permitted only after
# explicit human authorization; this script records an approval id but cannot
# substitute for that authorization.
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent /bin/bash --noprofile --norc -s -- "$@" <<'G3_CLEAN_PAYLOAD'
#!/bin/bash
# This payload is entered only through the root-private env -i trampoline.
# A malicious root can replace root-owned files; that is outside this release's
# trusted-control-plane model. This check prevents accidental/direct shell
# environment pollution from becoming authorization provenance.
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
clean_tool() { /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent "$@"; }
require_clean_payload_environment
set -euo pipefail
umask 077
ROOT="/workspace/experiments/tide_safe_c1_20260727/safe_c1_native_matrix_g3_v21_fulltrace_nativecertfix_jsonseal"
ADMISSION="$ROOT/preflight/admissions/g3_4k_pilot_wrapper.admission"
BINARY="$ROOT/preflight/bin/g3_4k_pilot_wrapper"
WRAPPER="$ROOT/runner/g3_4k_pilot_wrapper.cu"
RUN_LAUNCHER="$ROOT/tools/run_g3_4k_single_tu.sh"
RUN_BODY="$ROOT/tools/.run_g3_4k_single_tu.body.sh"
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
MANIFEST="$ROOT/inputs/g3_sift4096_branchstress_l2_d3/manifest.json"
BOOTSTRAP="$ROOT/preflight/bootstrap_oracles/g3_sift4096_branchstress_l2_d3/bootstrap_oracle.json"
fail() { echo "GPU authorization refused: $*" >&2; exit 69; }
usage() { echo "usage: issue_g3_4k_gpu_token.sh --gpu-ordinal <n> --run-name <safe-name> --approval-id <human-id> --ttl-seconds <1..120>" >&2; exit 69; }
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
  [[ "$x" == "0:0:600:1" || "$x" == "0:0:700:1" ]] || fail "file is not root-private $p"
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
  parse_nvml_snapshot "$GPU"
  [[ "$SNAP_ORD" == "$GPU" ]] || fail "NVML ordinal response drift"
  if [[ -z "$EXPECTED_GPU_UUID" ]]; then
    EXPECTED_GPU_UUID="$SNAP_UUID"; EXPECTED_GPU_PCI="$SNAP_PCI"
    EXPECTED_PYTHON_REALPATH="$SNAP_PYTHON_REALPATH"; EXPECTED_PYTHON_SHA="$SNAP_PYTHON_SHA"
    EXPECTED_NVML_LIBRARY_REALPATH="$SNAP_NVML_LIBRARY_REALPATH"; EXPECTED_NVML_LIBRARY_SHA="$SNAP_NVML_LIBRARY_SHA"
    EXPECTED_DRIVER_VERSION="$SNAP_DRIVER_VERSION"
  else
    [[ "$SNAP_UUID" == "$EXPECTED_GPU_UUID" && "$SNAP_PCI" == "$EXPECTED_GPU_PCI" &&
       "$SNAP_PYTHON_REALPATH" == "$EXPECTED_PYTHON_REALPATH" && "$SNAP_PYTHON_SHA" == "$EXPECTED_PYTHON_SHA" &&
       "$SNAP_NVML_LIBRARY_REALPATH" == "$EXPECTED_NVML_LIBRARY_REALPATH" &&
       "$SNAP_NVML_LIBRARY_SHA" == "$EXPECTED_NVML_LIBRARY_SHA" && "$SNAP_DRIVER_VERSION" == "$EXPECTED_DRIVER_VERSION" ]] ||
      fail "NVML identity/fingerprint drift across issuer samples"
  fi
  GPU_UUID="$SNAP_UUID"; GPU_PCI="$SNAP_PCI"
}
GPU=""; NAME=""; APPROVAL=""; TTL=""
GPU_UUID=""; GPU_PCI=""; EXPECTED_GPU_UUID=""; EXPECTED_GPU_PCI=""
EXPECTED_PYTHON_REALPATH=""; EXPECTED_PYTHON_SHA=""
EXPECTED_NVML_LIBRARY_REALPATH=""; EXPECTED_NVML_LIBRARY_SHA=""; EXPECTED_DRIVER_VERSION=""
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
[[ "$GPU" =~ ^(0|[1-9][0-9]*)$ && "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ &&
   "$APPROVAL" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$ && "$TTL" =~ ^[0-9]+$ &&
   "$TTL" -ge 1 && "$TTL" -le 120 ]] || usage
for p in "$ROOT" "$ROOT/preflight" "$ROOT/preflight/admissions" "$ROOT/preflight/bin" "$ROOT/runs"; do private_dir "$p"; done
for p in "$ADMISSION" "$BINARY" "$WRAPPER" "$RUN_LAUNCHER" "$RUN_BODY" "$NVML_HELPER"; do private_file "$p"; done
[[ -f "$MANIFEST" && ! -L "$MANIFEST" && -f "$BOOTSTRAP" && ! -L "$BOOTSTRAP" ]] ||
  fail "fixed build/fixture/NVML-helper prerequisite is unavailable"
assert_nvml_helper_integrity
[[ "$(admission_field schema)" == "safe-c1-g3-single-tu-admission-v4" &&
   "$(admission_field binary_path)" == "$BINARY" &&
   "$(admission_field binary_sha256)" == "$(sha "$BINARY")" &&
   "$(admission_field wrapper_sha256)" == "$(sha "$WRAPPER")" &&
   "$(admission_field run_launcher_sha256)" == "$(sha "$RUN_LAUNCHER")" &&
   "$(admission_field run_body_sha256)" == "$(sha "$RUN_BODY")" ]] ||
  fail "admission/image/script binding drifted"
[[ "$(admission_field nvml_snapshot_helper_path)" == "$NVML_HELPER" &&
   "$(admission_field nvml_snapshot_helper_sha256)" == "$NVML_HELPER_SHA" &&
   "$(admission_field nvml_snapshot_schema)" == "$NVML_SNAPSHOT_SCHEMA" &&
   "$(admission_field nvml_python_realpath)" == "$NVML_PYTHON_REALPATH" &&
   "$(admission_field nvml_python_sha256)" == "$NVML_PYTHON_SHA" &&
   "$(admission_field nvml_library_realpath)" == "$NVML_LIBRARY_REALPATH" &&
   "$(admission_field nvml_library_sha256)" == "$NVML_LIBRARY_SHA" ]] ||
  fail "admission NVML-control binding drifted"
install -d -m 700 "$AUTH" "$PENDING" "$CONSUMED" "$LEASES"
for p in "$AUTH" "$PENDING" "$CONSUMED" "$LEASES"; do private_dir "$p"; done
[[ ! -e "$TOKEN" && ! -e "$PROOF" ]] || fail "pending authorization/proof already exists"
SAMPLES=""
for i in 1 2 3; do
  gpu_idle
  SAMPLES="$SAMPLES""sample-$i-boot-ns=$(boot_ns)-schema=$NVML_SNAPSHOT_SCHEMA-uuid=$GPU_UUID-pci=$GPU_PCI-compute-process-count=0-helper-sha=$NVML_HELPER_SHA-python=$SNAP_PYTHON_REALPATH-python-sha=$SNAP_PYTHON_SHA-nvml-library=$SNAP_NVML_LIBRARY_REALPATH-nvml-library-sha=$SNAP_NVML_LIBRARY_SHA-driver=$SNAP_DRIVER_VERSION\n"
  [[ "$i" -eq 3 ]] || /bin/sleep 2
done
IDLE_SHA="$(printf '%b' "$SAMPLES" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"
ISSUED="$(boot_ns)"; EXPIRES=$((ISSUED + TTL * 1000000000))
NONCE="$("/usr/bin/od" -An -N32 -tx1 /dev/urandom | /usr/bin/tr -d ' \n')"
[[ "$NONCE" =~ ^[0-9a-f]{64}$ ]] || fail "cannot obtain token nonce"
TMP="$PENDING/.token.$NONCE.$$"
( umask 077; : > "$TMP" ) || fail "cannot stage token"
printf '%s\n' \
  "schema=safe-c1-g3-gpu-launch-token-v2" "token_nonce=$NONCE" "release_root=$ROOT" \
  "host_name=$(/bin/hostname)" "host_boot_id=$(/bin/cat /proc/sys/kernel/random/boot_id)" \
  "issued_boottime_ns=$ISSUED" "expires_boottime_ns=$EXPIRES" \
  "gpu_ordinal=$GPU" "gpu_uuid=$GPU_UUID" "gpu_pci_bus_id=$GPU_PCI" \
  "run_name=$NAME" "run_dir=$ROOT/runs/$NAME" "admission_path=$ADMISSION" \
  "admission_sha256=$(sha "$ADMISSION")" "binary_path=$BINARY" "binary_sha256=$(sha "$BINARY")" \
  "wrapper_sha256=$(sha "$WRAPPER")" \
  "source_closure_descriptor_sha256=$(admission_field source_closure_descriptor_sha256)" \
  "run_launcher_sha256=$(sha "$RUN_LAUNCHER")" "run_body_sha256=$(sha "$RUN_BODY")" \
  "pre_native_host_loader_descriptor_sha256=$(admission_field pre_native_host_loader_descriptor_sha256)" \
  "nvml_snapshot_schema=$NVML_SNAPSHOT_SCHEMA" "nvml_snapshot_helper_path=$NVML_HELPER" \
  "nvml_snapshot_helper_sha256=$NVML_HELPER_SHA" "nvml_python_realpath=$SNAP_PYTHON_REALPATH" \
  "nvml_python_sha256=$SNAP_PYTHON_SHA" "issuer_nvml_library_realpath=$SNAP_NVML_LIBRARY_REALPATH" \
  "issuer_nvml_library_sha256=$SNAP_NVML_LIBRARY_SHA" "issuer_nvml_driver_version=$SNAP_DRIVER_VERSION" \
  "fixture_manifest_sha256=$(sha "$MANIFEST")" "bootstrap_oracle_sha256=$(sha "$BOOTSTRAP")" \
  "idle_sample_count=3" "idle_sample_interval_seconds=2" "idle_check_sha256=$IDLE_SHA" \
  "issuer_approval_id=$APPROVAL" "issuer_script_sha256=$(sha "$ROOT/tools/issue_g3_4k_gpu_token.sh")" \
  "purpose=safe_c1_g3_4k_correctness_pilot" > "$TMP"
chmod 600 "$TMP"
/bin/ln "$TMP" "$TOKEN" || { /bin/rm -f "$TMP"; fail "cannot atomically issue token"; }
/bin/rm -f "$TMP"
private_file "$TOKEN"
printf 'GPU_AUTHORIZATION_ISSUED nonce=%s run_name=%s gpu_ordinal=%s gpu_uuid=%s expires_boottime_ns=%s\n' \
  "$NONCE" "$NAME" "$GPU" "$GPU_UUID" "$EXPIRES"
G3_CLEAN_PAYLOAD
exit 69
