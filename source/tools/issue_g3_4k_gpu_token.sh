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
ROOT="/workspace/experiments/tide_safe_c1_20260727/safe_c1_native_matrix_g3_v8_loader_allowlist_pilot"
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
NVIDIA_SMI="/usr/bin/nvidia-smi"
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
gpu_idle() {
  local got ord uuid pci extra raw
  got="$(clean_tool "$NVIDIA_SMI" --id="$GPU" --query-gpu=index,uuid,pci.bus_id --format=csv,noheader,nounits)" ||
    fail "nvidia-smi identity query failed"
  [[ "$got" != *$'\n'* ]] || fail "GPU identity query is ambiguous"
  IFS=, read -r ord uuid pci extra <<< "$got"
  ord="$(trim "$ord")"; uuid="$(trim "$uuid")"; pci="$(trim "$pci")"; extra="$(trim "$extra")"
  [[ -z "$extra" && "$ord" == "$GPU" &&
     "$uuid" =~ ^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ &&
     "$pci" =~ ^[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]$ ]] ||
    fail "invalid GPU identity response"
  if [[ -z "$EXPECTED_GPU_UUID" ]]; then
    EXPECTED_GPU_UUID="$uuid"
    EXPECTED_GPU_PCI="$pci"
  else
    [[ "$uuid" == "$EXPECTED_GPU_UUID" && "$pci" == "$EXPECTED_GPU_PCI" ]] ||
      fail "GPU ordinal/UUID/PCI identity drift across idle samples"
  fi
  raw="$(clean_tool "$NVIDIA_SMI" --id="$GPU" --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>&1)" ||
    fail "nvidia-smi compute-process query failed"
  raw="$(trim "$raw")"
  [[ -z "$raw" || "$raw" == "No running compute processes found" ]] ||
    fail "target GPU has a compute process; none was touched"
  GPU_UUID="$uuid"; GPU_PCI="$pci"
}
GPU=""; NAME=""; APPROVAL=""; TTL=""
GPU_UUID=""; GPU_PCI=""; EXPECTED_GPU_UUID=""; EXPECTED_GPU_PCI=""
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
for p in "$ADMISSION" "$BINARY" "$WRAPPER" "$RUN_LAUNCHER" "$RUN_BODY"; do private_file "$p"; done
[[ -f "$MANIFEST" && ! -L "$MANIFEST" && -f "$BOOTSTRAP" && ! -L "$BOOTSTRAP" && -x "$NVIDIA_SMI" ]] ||
  fail "fixed build/fixture/nvidia-smi prerequisite is unavailable"
[[ "$(admission_field schema)" == "safe-c1-g3-single-tu-admission-v3" &&
   "$(admission_field binary_path)" == "$BINARY" &&
   "$(admission_field binary_sha256)" == "$(sha "$BINARY")" &&
   "$(admission_field wrapper_sha256)" == "$(sha "$WRAPPER")" &&
   "$(admission_field run_launcher_sha256)" == "$(sha "$RUN_LAUNCHER")" &&
   "$(admission_field run_body_sha256)" == "$(sha "$RUN_BODY")" ]] ||
  fail "admission/image/script binding drifted"
install -d -m 700 "$AUTH" "$PENDING" "$CONSUMED" "$LEASES"
for p in "$AUTH" "$PENDING" "$CONSUMED" "$LEASES"; do private_dir "$p"; done
[[ ! -e "$TOKEN" && ! -e "$PROOF" ]] || fail "pending authorization/proof already exists"
SAMPLES=""
for i in 1 2 3; do
  gpu_idle
  SAMPLES="$SAMPLES""sample-$i-boot-ns=$(boot_ns)-uuid=$GPU_UUID-pci=$GPU_PCI-compute-pids-none\n"
  [[ "$i" -eq 3 ]] || /bin/sleep 2
done
IDLE_SHA="$(printf '%b' "$SAMPLES" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"
ISSUED="$(boot_ns)"; EXPIRES=$((ISSUED + TTL * 1000000000))
NONCE="$("/usr/bin/od" -An -N32 -tx1 /dev/urandom | /usr/bin/tr -d ' \n')"
[[ "$NONCE" =~ ^[0-9a-f]{64}$ ]] || fail "cannot obtain token nonce"
TMP="$PENDING/.token.$NONCE.$$"
( umask 077; : > "$TMP" ) || fail "cannot stage token"
printf '%s\n' \
  "schema=safe-c1-g3-gpu-launch-token-v1" "token_nonce=$NONCE" "release_root=$ROOT" \
  "host_name=$(/bin/hostname)" "host_boot_id=$(/bin/cat /proc/sys/kernel/random/boot_id)" \
  "issued_boottime_ns=$ISSUED" "expires_boottime_ns=$EXPIRES" \
  "gpu_ordinal=$GPU" "gpu_uuid=$GPU_UUID" "gpu_pci_bus_id=$GPU_PCI" \
  "run_name=$NAME" "run_dir=$ROOT/runs/$NAME" "admission_path=$ADMISSION" \
  "admission_sha256=$(sha "$ADMISSION")" "binary_path=$BINARY" "binary_sha256=$(sha "$BINARY")" \
  "wrapper_sha256=$(sha "$WRAPPER")" \
  "source_closure_descriptor_sha256=$(admission_field source_closure_descriptor_sha256)" \
  "run_launcher_sha256=$(sha "$RUN_LAUNCHER")" "run_body_sha256=$(sha "$RUN_BODY")" \
  "pre_native_host_loader_descriptor_sha256=$(admission_field pre_native_host_loader_descriptor_sha256)" \
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
