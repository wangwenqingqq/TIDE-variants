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
# Body reached only via the /bin/sh env -i launcher. It compiles exactly one
# sealed wrapper TU, performs link inspection, and emits no native pilot result.
set -euo pipefail
umask 077
ROOT="/workspace/experiments/tide_safe_c1_20260727/safe_c1_native_matrix_g3_v22_fulltrace_nativecertfix_jsonseal"
NVCC="/usr/local/cuda-13.1/bin/nvcc"
HOST_CXX="/usr/bin/g++"
READELF="/usr/bin/readelf"
LDD="/usr/bin/ldd"
WRAPPER="$ROOT/runner/g3_4k_pilot_wrapper.cu"
COMPILE_DIR="$ROOT/preflight/compile_wrapper_only"
COMPILE_SNAPSHOT="$COMPILE_DIR/sealed_source"
SNAPSHOT="$ROOT/preflight/sealed_source"
SNAPSHOT_WRAPPER="$SNAPSHOT/runner/g3_4k_pilot_wrapper.cu"
BUILD_LAUNCHER="$ROOT/tools/build_and_attest_g3_4k_single_tu.sh"
BUILD_BODY="$ROOT/tools/.build_and_attest_g3_4k_single_tu.body.sh"
COMPILE_LAUNCHER="$ROOT/tools/compile_g3_4k_wrapper_only.sh"
COMPILE_BODY="$ROOT/tools/.compile_g3_4k_wrapper_only.body.sh"
RUN_LAUNCHER="$ROOT/tools/run_g3_4k_single_tu.sh"
RUN_BODY="$ROOT/tools/.run_g3_4k_single_tu.body.sh"
NVML_HELPER="$ROOT/tools/.g3_nvml_snapshot.py"
NVML_HELPER_SHA="d5d57ccd816f377209f6a4f2709f75cb030e65739423b8967c126197513891c8"
NVML_SNAPSHOT_SCHEMA="safe-c1-g3-nvml-snapshot-v1"
NVML_PYTHON_REALPATH="/usr/bin/python3.12"
NVML_PYTHON_SHA="1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
NVML_LIBRARY_REALPATH="/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.590.48.01"
NVML_LIBRARY_SHA="12f3bcd4ba447599a2077297e3a4ff4288205b082c2e76346718ae85c058c4b2"
BIN_DIR="$ROOT/preflight/bin"
ADMISSION_DIR="$ROOT/preflight/admissions"
BINARY="$BIN_DIR/g3_4k_pilot_wrapper"
ADMISSION="$ADMISSION_DIR/g3_4k_pilot_wrapper.admission"

usage() { echo "usage: build_and_attest_g3_4k_single_tu.sh --arch sm_XX" >&2; exit 69; }
[[ "$#" -eq 2 && "$1" == "--arch" ]] || usage
ARCH="$2"
[[ "$ARCH" =~ ^sm_[0-9]{2,3}$ ]] || { echo "arch must be explicit sm_XX" >&2; exit 69; }
[[ -d "$ROOT" && ! -L "$ROOT" && "$(readlink -f "$ROOT")" == "$ROOT" ]] ||
  { echo "unsafe fixed v11 root" >&2; exit 69; }
[[ -x "$NVCC" && -x "$HOST_CXX" && -x "$READELF" && -x "$LDD" &&
   -f "$BUILD_LAUNCHER" && ! -L "$BUILD_LAUNCHER" &&
   -f "$BUILD_BODY" && ! -L "$BUILD_BODY" &&
   -f "$RUN_LAUNCHER" && ! -L "$RUN_LAUNCHER" &&
   -f "$RUN_BODY" && ! -L "$RUN_BODY" &&
   -f "$NVML_HELPER" && ! -L "$NVML_HELPER" ]] ||
  { echo "missing fixed tool/launcher/body" >&2; exit 69; }
[[ -d "$COMPILE_DIR" && ! -L "$COMPILE_DIR" &&
   -d "$COMPILE_SNAPSHOT" && ! -L "$COMPILE_SNAPSHOT" &&
   -f "$COMPILE_DIR/compile_only_contract.txt" &&
   -f "$COMPILE_DIR/g3_4k_pilot_wrapper.compute_80.o" &&
   -f "$COMPILE_DIR/object.sha256" &&
   -f "$COMPILE_DIR/compile_command.txt" &&
   -f "$COMPILE_DIR/compile_witness.txt" &&
   -f "$COMPILE_DIR/nvcc_compile.log" &&
   -f "$COMPILE_DIR/nvcc_version.txt" &&
   -f "$COMPILE_DIR/host_cxx_version.txt" ]] ||
  { echo "missing required compile-only closure" >&2; exit 69; }
[[ ! -e "$BIN_DIR" && ! -e "$ADMISSION_DIR" && ! -e "$SNAPSHOT" ]] ||
  { echo "build/admission/sealed snapshot already exists; do not overwrite/retry" >&2; exit 69; }

SOURCE_DESCRIPTOR_SHA="a6662d6da25ee45fead82551a136c57316c5e94fee179365bba2e4ed36939e0f"
WRAPPER_SHA="259bf27092d93e163259797d6491eddbb44a6a02e8d89405f290b4752657c7d2"
COMPILE_COMMAND="$COMPILE_DIR/compile_command.txt"
COMPILE_WITNESS="$COMPILE_DIR/compile_witness.txt"
COMPILE_OBJECT="$COMPILE_DIR/g3_4k_pilot_wrapper.compute_80.o"
COMPILE_CONTRACT="$COMPILE_DIR/compile_only_contract.txt"
COMPILE_COMMAND_SHA="$(sha256sum "$COMPILE_COMMAND" | awk '{print $1}')"
COMPILE_WITNESS_SHA="$(sha256sum "$COMPILE_WITNESS" | awk '{print $1}')"
COMPILE_OBJECT_SHA="$(sha256sum "$COMPILE_OBJECT" | awk '{print $1}')"
COMPILE_LOG_SHA="$(sha256sum "$COMPILE_DIR/nvcc_compile.log" | awk '{print $1}')"
NVCC_VERSION_SHA="$(sha256sum "$COMPILE_DIR/nvcc_version.txt" | awk '{print $1}')"
HOST_CXX_VERSION_SHA="$(sha256sum "$COMPILE_DIR/host_cxx_version.txt" | awk '{print $1}')"
COMPILE_LAUNCHER_SHA="$(sha256sum "$COMPILE_LAUNCHER" | awk '{print $1}')"
COMPILE_BODY_SHA="$(sha256sum "$COMPILE_BODY" | awk '{print $1}')"
require_unique_line() {
  local line="$1" file="$2" count
  count="$(/usr/bin/grep -Fxc "$line" "$file" || true)"
  [[ "$count" == "1" ]] || {
    echo "missing/non-unique compile evidence line: $line" >&2; exit 69; }
}
LISTED_EXTRA=""
read -r LISTED_OBJECT_SHA LISTED_OBJECT_PATH LISTED_EXTRA < "$COMPILE_DIR/object.sha256"
[[ -z "$LISTED_EXTRA" && "$LISTED_OBJECT_SHA" == "$COMPILE_OBJECT_SHA" &&
   "$LISTED_OBJECT_PATH" == "$COMPILE_OBJECT" ]] ||
  { echo "compile-only object.sha256 does not bind the actual object" >&2; exit 69; }

# The pre-link gate checks every emitted compile contract field before this
# script creates the final binary. The final runner repeats a strict parser/
# byte-hash check, but link never proceeds on a partially checked witness.
for requirement in \
  "schema=safe-c1-g3-compile-only-manifest-v6" \
  "single_translation_unit=true" \
  "link_or_runtime=false" \
  "sanitized_build_environment=env_i_private_body_trampoline_v2" \
  "compiler_path=$NVCC" \
  "host_compiler_path=$HOST_CXX" \
  "translation_unit=$COMPILE_SNAPSHOT/runner/g3_4k_pilot_wrapper.cu" \
  "source_snapshot_path=$COMPILE_SNAPSHOT" \
  "source_snapshot_descriptor_sha256=$SOURCE_DESCRIPTOR_SHA" \
  "wrapper_sha256=$WRAPPER_SHA" \
  "source_input_count=1" \
  "object_or_archive_inputs=absent" \
  "include_order=$COMPILE_SNAPSHOT/src:$COMPILE_SNAPSHOT/reference/include" \
  "compile_command_sha256=$COMPILE_COMMAND_SHA" \
  "compile_witness_sha256=$COMPILE_WITNESS_SHA" \
  "compile_object_sha256=$COMPILE_OBJECT_SHA" \
  "compile_launcher_path=$COMPILE_LAUNCHER" \
  "compile_launcher_sha256=$COMPILE_LAUNCHER_SHA" \
  "compile_body_path=$COMPILE_BODY" \
  "compile_body_sha256=$COMPILE_BODY_SHA"; do
  require_unique_line "$requirement" "$COMPILE_CONTRACT"
done
for requirement in \
  "schema=safe-c1-g3-compile-command-v1" \
  "single_translation_unit=true" \
  "link_or_runtime=false" \
  "sanitized_build_environment=env_i_private_body_trampoline_v2" \
  "compiler_path=$NVCC" \
  "host_compiler_path=$HOST_CXX" \
  "translation_unit=$COMPILE_SNAPSHOT/runner/g3_4k_pilot_wrapper.cu" \
  "source_snapshot_path=$COMPILE_SNAPSHOT" \
  "source_snapshot_descriptor_sha256=$SOURCE_DESCRIPTOR_SHA" \
  "wrapper_sha256=$WRAPPER_SHA" \
  "language_standard=c++17" \
  "gpu_arch=compute_80" \
  "gpu_code=compute_80" \
  "source_input_count=1" \
  "object_or_archive_inputs=absent" \
  "include_order=$COMPILE_SNAPSHOT/src:$COMPILE_SNAPSHOT/reference/include" \
  "compile_launcher_path=$COMPILE_LAUNCHER" \
  "compile_launcher_sha256=$COMPILE_LAUNCHER_SHA" \
  "compile_body_path=$COMPILE_BODY" \
  "compile_body_sha256=$COMPILE_BODY_SHA"; do
  require_unique_line "$requirement" "$COMPILE_COMMAND"
done
for requirement in \
  "schema=safe-c1-g3-compile-witness-v1" \
  "single_translation_unit=true" \
  "link_or_runtime=false" \
  "witness_role=sealed_source_pre_link_compile_not_link_input" \
  "sanitized_build_environment=env_i_private_body_trampoline_v2" \
  "compiler_path=$NVCC" \
  "host_compiler_path=$HOST_CXX" \
  "translation_unit=$COMPILE_SNAPSHOT/runner/g3_4k_pilot_wrapper.cu" \
  "source_snapshot_path=$COMPILE_SNAPSHOT" \
  "source_snapshot_descriptor_sha256=$SOURCE_DESCRIPTOR_SHA" \
  "wrapper_sha256=$WRAPPER_SHA" \
  "compile_command_sha256=$COMPILE_COMMAND_SHA" \
  "compile_object_sha256=$COMPILE_OBJECT_SHA" \
  "compile_log_sha256=$COMPILE_LOG_SHA" \
  "nvcc_version_sha256=$NVCC_VERSION_SHA" \
  "host_cxx_version_sha256=$HOST_CXX_VERSION_SHA" \
  "compile_launcher_path=$COMPILE_LAUNCHER" \
  "compile_launcher_sha256=$COMPILE_LAUNCHER_SHA" \
  "compile_body_path=$COMPILE_BODY" \
  "compile_body_sha256=$COMPILE_BODY_SHA"; do
  require_unique_line "$requirement" "$COMPILE_WITNESS"
done

expect_sha() {
  local expected="$1" file="$2" actual
  [[ -f "$file" && ! -L "$file" ]] || { echo "unsafe source path: $file" >&2; exit 69; }
  actual="$(sha256sum "$file" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || { echo "source SHA mismatch: $file" >&2; exit 69; }
}
snapshot_file() {
  local expected="$1" rel="$2" src="$ROOT/$2" dst="$SNAPSHOT/$2"
  expect_sha "$expected" "$src"
  install -d -m 700 "$(dirname "$dst")"
  install -m 600 "$src" "$dst"
  expect_sha "$expected" "$dst"
}
[[ "$(/usr/bin/stat -c '%u:%g:%a:%h' "$NVML_HELPER")" == "0:0:600:1" ]] ||
  { echo "NVML helper is not root-private single-link 0600" >&2; exit 69; }
expect_sha "$NVML_HELPER_SHA" "$NVML_HELPER"
source_closure_descriptor() {
  {
    printf 'safe-c1-g3-source-closure-v1\n'
    printf '%s\n' \
      'src/g3_safe_c1_native_matrix.cu:c987fbec73c7f7324a2337e08fd3342a74cdd040021119ec72fd13169a2a7dcd' \
      'src/g3_safe_search_v2.cuh:65688fedcbb05a1fff680555e3139b6e640bfdd72288b93dfccddcec90dfbf01' \
      'reference/include/tree.cuh:c1324bef173358c31a8371e1f050d4372cd1c92cd98c71af3832b2d19c769fd6' \
      'reference/include/file.cuh:b8be03246ec82cadd3228b75a6dfd91c552ccfa3124d2ca9470e28e0031ffd8a' \
      'reference/include/config.cuh:622d0977e49d80d5791364bc12463de9bce5aaca324d6a681512004c20d100cb' \
      'reference/include/mlp_constant.cuh:cf6545624c0cbc51744b978298e6d138591521c7b600ece8812b6195f735e559' \
      'reference/include/residual_pruning.cuh:745a4bd5564b8085c40a48abac625f24dcbacb8a996d29e4cba311dd936e33b2'
  } | sha256sum | awk '{print $1}'
}
needed_sonames_sha() {
  { printf 'safe-c1-g3-needed-sonames-v1\n'; for name in "$@"; do printf '%s\n' "$name"; done; } |
    sha256sum | awk '{print $1}'
}
allowed_needed() {
  case "$1" in
    libc.so.6|libm.so.6|libpthread.so.0|librt.so.1|libdl.so.2|libstdc++.so.6|libgcc_s.so.1|ld-linux-x86-64.so.2) return 0 ;;
    *) return 1 ;;
  esac
}

expect_sha "$WRAPPER_SHA" "$WRAPPER"
[[ "$(grep -Ec '^#include "\.\./src/g3_safe_c1_native_matrix\.cu"$' "$WRAPPER")" -eq 1 ]] ||
  { echo "wrapper must include Matrix exactly once" >&2; exit 69; }
if grep -nE '(^|[^[:alnum:]_])(searchIndexKnnV2|searchIndexRnnV2|indexConstru|upload_rp_constants|c_rp_mode)([^[:alnum:]_]|$)' "$WRAPPER" >/dev/null; then
  echo "wrapper contains raw GTS/RP identifier" >&2; exit 69
fi
for controlled in "$WRAPPER" "$ROOT/src/g3_safe_c1_native_matrix.cu" \
  "$ROOT/src/g3_safe_search_v2.cuh" "$ROOT/reference/include/tree.cuh" \
  "$ROOT/reference/include/file.cuh" "$ROOT/reference/include/config.cuh" \
  "$ROOT/reference/include/mlp_constant.cuh" "$ROOT/reference/include/residual_pruning.cuh"; do
  if grep -nEi 'dlopen|dlsym|dlmopen|dlvsym' "$controlled" >/dev/null; then
    echo "controlled closure has dynamic-loader symbol: $controlled" >&2; exit 69
  fi
done

install -d -m 700 "$SNAPSHOT"
snapshot_file "$WRAPPER_SHA" "runner/g3_4k_pilot_wrapper.cu"
snapshot_file "c987fbec73c7f7324a2337e08fd3342a74cdd040021119ec72fd13169a2a7dcd" "src/g3_safe_c1_native_matrix.cu"
snapshot_file "65688fedcbb05a1fff680555e3139b6e640bfdd72288b93dfccddcec90dfbf01" "src/g3_safe_search_v2.cuh"
snapshot_file "c1324bef173358c31a8371e1f050d4372cd1c92cd98c71af3832b2d19c769fd6" "reference/include/tree.cuh"
snapshot_file "b8be03246ec82cadd3228b75a6dfd91c552ccfa3124d2ca9470e28e0031ffd8a" "reference/include/file.cuh"
snapshot_file "622d0977e49d80d5791364bc12463de9bce5aaca324d6a681512004c20d100cb" "reference/include/config.cuh"
snapshot_file "cf6545624c0cbc51744b978298e6d138591521c7b600ece8812b6195f735e559" "reference/include/mlp_constant.cuh"
snapshot_file "745a4bd5564b8085c40a48abac625f24dcbacb8a996d29e4cba311dd936e33b2" "reference/include/residual_pruning.cuh"
[[ "$(source_closure_descriptor)" == "$SOURCE_DESCRIPTOR_SHA" ]] ||
  { echo "sealed source descriptor drift" >&2; exit 69; }
expect_sha "$WRAPPER_SHA" "$COMPILE_SNAPSHOT/runner/g3_4k_pilot_wrapper.cu"
expect_sha "c987fbec73c7f7324a2337e08fd3342a74cdd040021119ec72fd13169a2a7dcd" "$COMPILE_SNAPSHOT/src/g3_safe_c1_native_matrix.cu"
expect_sha "65688fedcbb05a1fff680555e3139b6e640bfdd72288b93dfccddcec90dfbf01" "$COMPILE_SNAPSHOT/src/g3_safe_search_v2.cuh"
expect_sha "c1324bef173358c31a8371e1f050d4372cd1c92cd98c71af3832b2d19c769fd6" "$COMPILE_SNAPSHOT/reference/include/tree.cuh"
expect_sha "b8be03246ec82cadd3228b75a6dfd91c552ccfa3124d2ca9470e28e0031ffd8a" "$COMPILE_SNAPSHOT/reference/include/file.cuh"
expect_sha "622d0977e49d80d5791364bc12463de9bce5aaca324d6a681512004c20d100cb" "$COMPILE_SNAPSHOT/reference/include/config.cuh"
expect_sha "cf6545624c0cbc51744b978298e6d138591521c7b600ece8812b6195f735e559" "$COMPILE_SNAPSHOT/reference/include/mlp_constant.cuh"
expect_sha "745a4bd5564b8085c40a48abac625f24dcbacb8a996d29e4cba311dd936e33b2" "$COMPILE_SNAPSHOT/reference/include/residual_pruning.cuh"

install -d -m 700 "$BIN_DIR"
install -d -m 700 "$ADMISSION_DIR"
BUILD_LAUNCHER_SHA="$(sha256sum "$BUILD_LAUNCHER" | awk '{print $1}')"
BUILD_BODY_SHA="$(sha256sum "$BUILD_BODY" | awk '{print $1}')"
RUN_LAUNCHER_SHA="$(sha256sum "$RUN_LAUNCHER" | awk '{print $1}')"
RUN_BODY_SHA="$(sha256sum "$RUN_BODY" | awk '{print $1}')"
NVCC_REALPATH="$(readlink -f "$NVCC")"
HOST_CXX_REALPATH="$(readlink -f "$HOST_CXX")"
NVCC_BIN_SHA="$(sha256sum "$NVCC_REALPATH" | awk '{print $1}')"
HOST_CXX_BIN_SHA="$(sha256sum "$HOST_CXX_REALPATH" | awk '{print $1}')"

TOOLCHAIN="$ADMISSION.toolchain.txt"
printf '%s\n' \
  "schema=safe-c1-g3-toolchain-v1" \
  "sanitized_build_environment=env_i_private_body_trampoline_v2" \
  "compiler_path=$NVCC" "compiler_realpath=$NVCC_REALPATH" "compiler_sha256=$NVCC_BIN_SHA" \
  "compiler_version_sha256=$(clean_tool "$NVCC" --version | sha256sum | awk '{print $1}')" \
  "host_compiler_path=$HOST_CXX" "host_compiler_realpath=$HOST_CXX_REALPATH" \
  "host_compiler_sha256=$HOST_CXX_BIN_SHA" \
  "host_compiler_version_sha256=$(clean_tool "$HOST_CXX" --version | sha256sum | awk '{print $1}')" \
  "build_launcher_path=$BUILD_LAUNCHER" "build_launcher_sha256=$BUILD_LAUNCHER_SHA" \
  "build_body_path=$BUILD_BODY" "build_body_sha256=$BUILD_BODY_SHA" \
  "run_launcher_path=$RUN_LAUNCHER" "run_launcher_sha256=$RUN_LAUNCHER_SHA" \
  "run_body_path=$RUN_BODY" "run_body_sha256=$RUN_BODY_SHA" \
  "nvml_snapshot_helper_path=$NVML_HELPER" "nvml_snapshot_helper_sha256=$NVML_HELPER_SHA" \
  "nvml_snapshot_schema=$NVML_SNAPSHOT_SCHEMA" "nvml_python_realpath=$NVML_PYTHON_REALPATH" \
  "nvml_python_sha256=$NVML_PYTHON_SHA" "nvml_library_realpath=$NVML_LIBRARY_REALPATH" \
  "nvml_library_sha256=$NVML_LIBRARY_SHA" "source_snapshot_path=$SNAPSHOT" "source_snapshot_descriptor_sha256=$SOURCE_DESCRIPTOR_SHA" \
  >"$TOOLCHAIN"

COMMAND="$ADMISSION.link-command.txt"
printf '%s\n' \
  "schema=safe-c1-g3-link-command-manifest-v3" \
  "compiler_path=$NVCC" "host_compiler_path=$HOST_CXX" \
  "translation_unit=$SNAPSHOT_WRAPPER" \
  "source_snapshot_path=$SNAPSHOT" "source_snapshot_descriptor_sha256=$SOURCE_DESCRIPTOR_SHA" \
  "final_binary_path=$BINARY" "language_standard=c++17" "gpu_arch=$ARCH" \
  "link_mode=executable" "translation_unit_count=1" "object_or_archive_inputs=absent" \
  "explicit_link_libraries=dl" "cudart_linkage=static" \
  "raw_legacy_gts_rp_calls_in_wrapper=absent" \
  "dynamic_loading_calls_in_controlled_closure=absent" \
  "host_elf_loader_enumeration=dl_iterate_phdr_pre_native_only" \
  "compile_only_contract_sha256=$(sha256sum "$COMPILE_DIR/compile_only_contract.txt" | awk '{print $1}')" \
  "compile_witness_sha256=$COMPILE_WITNESS_SHA" \
  "compile_only_object_sha256=$COMPILE_OBJECT_SHA" \
  "compile_only_gate_role=sealed_source_pre_link_compile_not_link_input" \
  "build_launcher_path=$BUILD_LAUNCHER" "build_launcher_sha256=$BUILD_LAUNCHER_SHA" \
  "build_body_path=$BUILD_BODY" "build_body_sha256=$BUILD_BODY_SHA" \
  "sanitized_build_environment=env_i_private_body_trampoline_v2" >"$COMMAND"
COMMAND_SHA="$(sha256sum "$COMMAND" | awk '{print $1}')"

clean_tool "$NVCC" -std=c++17 -ccbin "$HOST_CXX" -arch="$ARCH" -cudart static \
  -I"$SNAPSHOT/src" -I"$SNAPSHOT/reference/include" "$SNAPSHOT_WRAPPER" \
  -o "$BINARY" -ldl >"$BIN_DIR/nvcc_link.log" 2>&1

BINARY_SHA="$(sha256sum "$BINARY" | awk '{print $1}')"
REPORT="$ADMISSION.link-image.txt"
READELF_REPORT="$REPORT.readelf-dynamic.txt"
LDD_REPORT="$REPORT.ldd.txt"
clean_tool "$READELF" -d "$BINARY" >"$READELF_REPORT"
clean_tool "$LDD" "$BINARY" >"$LDD_REPORT"
if grep -E '\((RPATH|RUNPATH)\)' "$READELF_REPORT" >/dev/null; then
  echo "RPATH/RUNPATH is forbidden" >&2; exit 69
fi
if grep -Ei 'libgts|not found' "$READELF_REPORT" "$LDD_REPORT" >/dev/null; then
  echo "legacy GTS or unresolved dynamic dependency found" >&2; exit 69
fi
mapfile -t NEEDED < <(
  /usr/bin/awk '/\(NEEDED\)/ { x=$0; sub(/^.*Shared library: \[/, "", x); sub(/\].*$/, "", x); if (x != "") print x }' \
    "$READELF_REPORT" | /usr/bin/sort -u
)
for soname in "${NEEDED[@]}"; do
  allowed_needed "$soname" || { echo "NEEDED SONAME outside allowlist: $soname" >&2; exit 69; }
done
NEEDED_SHA="$(needed_sonames_sha "${NEEDED[@]}")"
READELF_SHA="$(sha256sum "$READELF_REPORT" | awk '{print $1}')"
LDD_SHA="$(sha256sum "$LDD_REPORT" | awk '{print $1}')"
TOOLCHAIN_SHA="$(sha256sum "$TOOLCHAIN" | awk '{print $1}')"

# CPU-only control executable path, with exactly PATH/HOME under env -i.
/usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent \
  "$BINARY" --emit-runtime-image-descriptor >"$ADMISSION.pre-native-host-loader.txt"
grep -Fx 'schema=safe-c1-g3-pre-native-host-loader-image-v2' \
  "$ADMISSION.pre-native-host-loader.txt" >/dev/null ||
  { echo "pre-native loader descriptor schema mismatch" >&2; exit 69; }
DESCRIPTOR_SHA="$(sha256sum "$ADMISSION.pre-native-host-loader.txt" | awk '{print $1}')"

printf '%s\n' \
  "schema=safe-c1-g3-link-image-report-v4" \
  "single_translation_unit=true" "raw_legacy_gts_rp_calls_in_wrapper=absent" \
  "dynamic_loading_calls_in_controlled_closure=absent" \
  "host_elf_loader_enumeration=dl_iterate_phdr_pre_native_only" \
  "legacy_gts_dso=absent" \
  "elf_rpath_or_runpath=absent" "needed_allowlist=system_cxx_static_cudart_glibc_loader_v2" \
  "needed_sonames_sha256=$NEEDED_SHA" "binary_sha256=$BINARY_SHA" \
  "wrapper_sha256=$WRAPPER_SHA" "source_closure_descriptor_sha256=$SOURCE_DESCRIPTOR_SHA" \
  "link_command_sha256=$COMMAND_SHA" "readelf_dynamic_sha256=$READELF_SHA" \
  "ldd_sha256=$LDD_SHA" "toolchain_report_sha256=$TOOLCHAIN_SHA" \
  "compile_witness_sha256=$COMPILE_WITNESS_SHA" \
  "compile_only_object_sha256=$COMPILE_OBJECT_SHA" >"$REPORT"
REPORT_SHA="$(sha256sum "$REPORT" | awk '{print $1}')"

printf '%s\n' \
  "schema=safe-c1-g3-single-tu-admission-v4" "binary_path=$BINARY" \
  "binary_sha256=$BINARY_SHA" "wrapper_sha256=$WRAPPER_SHA" \
  "source_closure_descriptor_sha256=$SOURCE_DESCRIPTOR_SHA" "single_translation_unit=true" \
  "wrapper_raw_legacy_gts_rp_calls_absent=true" "link_image_report_sha256=$REPORT_SHA" \
  "link_command_sha256=$COMMAND_SHA" \
  "compile_witness_sha256=$COMPILE_WITNESS_SHA" \
  "compile_only_object_sha256=$COMPILE_OBJECT_SHA" \
  "pre_native_host_loader_descriptor_sha256=$DESCRIPTOR_SHA" \
  "loader_snapshot_scope=pre_native_host_elf_only" \
  "sanitized_exec_environment=env_i_path_home_cuda_uuid_v2" \
  "run_launcher_path=$RUN_LAUNCHER" "run_launcher_sha256=$RUN_LAUNCHER_SHA" \
  "run_body_path=$RUN_BODY" "run_body_sha256=$RUN_BODY_SHA" \
  "nvml_snapshot_helper_path=$NVML_HELPER" "nvml_snapshot_helper_sha256=$NVML_HELPER_SHA" \
  "nvml_snapshot_schema=$NVML_SNAPSHOT_SCHEMA" "nvml_python_realpath=$NVML_PYTHON_REALPATH" \
  "nvml_python_sha256=$NVML_PYTHON_SHA" "nvml_library_realpath=$NVML_LIBRARY_REALPATH" \
  "nvml_library_sha256=$NVML_LIBRARY_SHA" >"$ADMISSION"
printf 'BUILD_AND_ATTEST_OK\n'

G3_CLEAN_PAYLOAD
# exec only returns on a launcher failure.
exit 69
