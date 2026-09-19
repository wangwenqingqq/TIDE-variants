#!/bin/bash
# Body reached only via the /bin/sh env -i launcher. It compiles exactly one
# sealed wrapper TU, performs link inspection, and emits no native pilot result.
set -euo pipefail
umask 077
ROOT="/workspace/experiments/tide_safe_c1_20260727/safe_c1_native_matrix_g3_v5_failstop_boundinput_pilot"
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
BIN_DIR="$ROOT/preflight/bin"
ADMISSION_DIR="$ROOT/preflight/admissions"
BINARY="$BIN_DIR/g3_4k_pilot_wrapper"
ADMISSION="$ADMISSION_DIR/g3_4k_pilot_wrapper.admission"

usage() { echo "usage: build_and_attest_g3_4k_single_tu.sh --arch sm_XX" >&2; exit 69; }
[[ "$#" -eq 2 && "$1" == "--arch" ]] || usage
ARCH="$2"
[[ "$ARCH" =~ ^sm_[0-9]{2,3}$ ]] || { echo "arch must be explicit sm_XX" >&2; exit 69; }
[[ -d "$ROOT" && ! -L "$ROOT" && "$(readlink -f "$ROOT")" == "$ROOT" ]] ||
  { echo "unsafe fixed v5 root" >&2; exit 69; }
[[ -x "$NVCC" && -x "$HOST_CXX" && -x "$READELF" && -x "$LDD" &&
   -f "$BUILD_LAUNCHER" && ! -L "$BUILD_LAUNCHER" &&
   -f "$BUILD_BODY" && ! -L "$BUILD_BODY" &&
   -f "$RUN_LAUNCHER" && ! -L "$RUN_LAUNCHER" &&
   -f "$RUN_BODY" && ! -L "$RUN_BODY" ]] ||
  { echo "missing fixed tool/launcher/body" >&2; exit 69; }
[[ -d "$COMPILE_DIR" && ! -L "$COMPILE_DIR" &&
   -d "$COMPILE_SNAPSHOT" && ! -L "$COMPILE_SNAPSHOT" &&
   -f "$COMPILE_DIR/compile_only_contract.txt" &&
   -f "$COMPILE_DIR/g3_4k_pilot_wrapper.compute_80.o" &&
   -f "$COMPILE_DIR/object.sha256" ]] ||
  { echo "missing required compile-only closure" >&2; exit 69; }
[[ ! -e "$BIN_DIR" && ! -e "$ADMISSION_DIR" && ! -e "$SNAPSHOT" ]] ||
  { echo "build/admission/sealed snapshot already exists; do not overwrite/retry" >&2; exit 69; }

SOURCE_DESCRIPTOR_SHA="7d66558db901f01c6151ee24559e124c0bf81ffc3362b00a0fe2b448a136a712"
WRAPPER_SHA="f87886b0cb97abfae37af7da36409574d454064d694f1b46ee170832af41d0a4"
grep -Fx "schema=safe-c1-g3-compile-only-manifest-v4" "$COMPILE_DIR/compile_only_contract.txt" >/dev/null ||
  { echo "compile-only schema mismatch" >&2; exit 69; }
grep -Fx "single_translation_unit=true" "$COMPILE_DIR/compile_only_contract.txt" >/dev/null ||
  { echo "compile-only not single-TU" >&2; exit 69; }
grep -Fx "wrapper_sha256=$WRAPPER_SHA" "$COMPILE_DIR/compile_only_contract.txt" >/dev/null ||
  { echo "compile-only wrapper seal mismatch" >&2; exit 69; }
grep -Fx "source_snapshot_path=$COMPILE_SNAPSHOT" "$COMPILE_DIR/compile_only_contract.txt" >/dev/null ||
  { echo "compile-only source snapshot path mismatch" >&2; exit 69; }
grep -Fx "source_snapshot_descriptor_sha256=$SOURCE_DESCRIPTOR_SHA" "$COMPILE_DIR/compile_only_contract.txt" >/dev/null ||
  { echo "compile-only source snapshot descriptor mismatch" >&2; exit 69; }
grep -Fx "compile_launcher_path=$COMPILE_LAUNCHER" "$COMPILE_DIR/compile_only_contract.txt" >/dev/null ||
  { echo "compile launcher path mismatch" >&2; exit 69; }
grep -Fx "compile_launcher_sha256=$(sha256sum "$COMPILE_LAUNCHER" | awk '{print $1}')" "$COMPILE_DIR/compile_only_contract.txt" >/dev/null ||
  { echo "compile launcher SHA mismatch" >&2; exit 69; }
grep -Fx "compile_body_path=$COMPILE_BODY" "$COMPILE_DIR/compile_only_contract.txt" >/dev/null ||
  { echo "compile body path mismatch" >&2; exit 69; }
grep -Fx "compile_body_sha256=$(sha256sum "$COMPILE_BODY" | awk '{print $1}')" "$COMPILE_DIR/compile_only_contract.txt" >/dev/null ||
  { echo "compile body SHA mismatch" >&2; exit 69; }

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
source_closure_descriptor() {
  {
    printf 'safe-c1-g3-source-closure-v1\n'
    printf '%s\n' \
      'src/g3_safe_c1_native_matrix.cu:12ef95e115c92ed2ebc86fc116aa441aacdbadfe850676a684c0b9119aa8b138' \
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
    libc.so.6|libm.so.6|libpthread.so.0|librt.so.1|libdl.so.2|libstdc++.so.6|libgcc_s.so.1) return 0 ;;
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
snapshot_file "12ef95e115c92ed2ebc86fc116aa441aacdbadfe850676a684c0b9119aa8b138" "src/g3_safe_c1_native_matrix.cu"
snapshot_file "65688fedcbb05a1fff680555e3139b6e640bfdd72288b93dfccddcec90dfbf01" "src/g3_safe_search_v2.cuh"
snapshot_file "c1324bef173358c31a8371e1f050d4372cd1c92cd98c71af3832b2d19c769fd6" "reference/include/tree.cuh"
snapshot_file "b8be03246ec82cadd3228b75a6dfd91c552ccfa3124d2ca9470e28e0031ffd8a" "reference/include/file.cuh"
snapshot_file "622d0977e49d80d5791364bc12463de9bce5aaca324d6a681512004c20d100cb" "reference/include/config.cuh"
snapshot_file "cf6545624c0cbc51744b978298e6d138591521c7b600ece8812b6195f735e559" "reference/include/mlp_constant.cuh"
snapshot_file "745a4bd5564b8085c40a48abac625f24dcbacb8a996d29e4cba311dd936e33b2" "reference/include/residual_pruning.cuh"
[[ "$(source_closure_descriptor)" == "$SOURCE_DESCRIPTOR_SHA" ]] ||
  { echo "sealed source descriptor drift" >&2; exit 69; }

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
  "sanitized_build_environment=env_i_sh_launcher_v1" \
  "compiler_path=$NVCC" "compiler_realpath=$NVCC_REALPATH" "compiler_sha256=$NVCC_BIN_SHA" \
  "compiler_version_sha256=$("$NVCC" --version | sha256sum | awk '{print $1}')" \
  "host_compiler_path=$HOST_CXX" "host_compiler_realpath=$HOST_CXX_REALPATH" \
  "host_compiler_sha256=$HOST_CXX_BIN_SHA" \
  "host_compiler_version_sha256=$("$HOST_CXX" --version | sha256sum | awk '{print $1}')" \
  "build_launcher_path=$BUILD_LAUNCHER" "build_launcher_sha256=$BUILD_LAUNCHER_SHA" \
  "build_body_path=$BUILD_BODY" "build_body_sha256=$BUILD_BODY_SHA" \
  "run_launcher_path=$RUN_LAUNCHER" "run_launcher_sha256=$RUN_LAUNCHER_SHA" \
  "run_body_path=$RUN_BODY" "run_body_sha256=$RUN_BODY_SHA" \
  "source_snapshot_path=$SNAPSHOT" "source_snapshot_descriptor_sha256=$SOURCE_DESCRIPTOR_SHA" \
  >"$TOOLCHAIN"

COMMAND="$ADMISSION.link-command.txt"
printf '%s\n' \
  "schema=safe-c1-g3-link-command-manifest-v2" \
  "compiler_path=$NVCC" "host_compiler_path=$HOST_CXX" \
  "translation_unit=$SNAPSHOT_WRAPPER" \
  "source_snapshot_path=$SNAPSHOT" "source_snapshot_descriptor_sha256=$SOURCE_DESCRIPTOR_SHA" \
  "final_binary_path=$BINARY" "language_standard=c++17" "gpu_arch=$ARCH" \
  "link_mode=executable" "translation_unit_count=1" "object_or_archive_inputs=absent" \
  "explicit_link_libraries=dl" "cudart_linkage=static" \
  "raw_legacy_gts_rp_calls_in_wrapper=absent" \
  "dynamic_loader_calls_in_controlled_closure=absent" \
  "build_launcher_path=$BUILD_LAUNCHER" "build_launcher_sha256=$BUILD_LAUNCHER_SHA" \
  "build_body_path=$BUILD_BODY" "build_body_sha256=$BUILD_BODY_SHA" \
  "sanitized_build_environment=env_i_sh_launcher_v1" >"$COMMAND"
COMMAND_SHA="$(sha256sum "$COMMAND" | awk '{print $1}')"

"$NVCC" -std=c++17 -ccbin "$HOST_CXX" -arch="$ARCH" -cudart static \
  -I"$SNAPSHOT/src" -I"$SNAPSHOT/reference/include" "$SNAPSHOT_WRAPPER" \
  -o "$BINARY" -ldl >"$BIN_DIR/nvcc_link.log" 2>&1

BINARY_SHA="$(sha256sum "$BINARY" | awk '{print $1}')"
REPORT="$ADMISSION.link-image.txt"
READELF_REPORT="$REPORT.readelf-dynamic.txt"
LDD_REPORT="$REPORT.ldd.txt"
"$READELF" -d "$BINARY" >"$READELF_REPORT"
"$LDD" "$BINARY" >"$LDD_REPORT"
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
  "schema=safe-c1-g3-link-image-report-v3" \
  "single_translation_unit=true" "raw_legacy_gts_rp_calls_in_wrapper=absent" \
  "dynamic_loader_calls_in_controlled_closure=absent" "legacy_gts_dso=absent" \
  "elf_rpath_or_runpath=absent" "needed_allowlist=system_cxx_static_cudart_v1" \
  "needed_sonames_sha256=$NEEDED_SHA" "binary_sha256=$BINARY_SHA" \
  "wrapper_sha256=$WRAPPER_SHA" "source_closure_descriptor_sha256=$SOURCE_DESCRIPTOR_SHA" \
  "link_command_sha256=$COMMAND_SHA" "readelf_dynamic_sha256=$READELF_SHA" \
  "ldd_sha256=$LDD_SHA" "toolchain_report_sha256=$TOOLCHAIN_SHA" >"$REPORT"
REPORT_SHA="$(sha256sum "$REPORT" | awk '{print $1}')"

printf '%s\n' \
  "schema=safe-c1-g3-single-tu-admission-v2" "binary_path=$BINARY" \
  "binary_sha256=$BINARY_SHA" "wrapper_sha256=$WRAPPER_SHA" \
  "source_closure_descriptor_sha256=$SOURCE_DESCRIPTOR_SHA" "single_translation_unit=true" \
  "wrapper_raw_legacy_gts_rp_calls_absent=true" "link_image_report_sha256=$REPORT_SHA" \
  "link_command_sha256=$COMMAND_SHA" \
  "pre_native_host_loader_descriptor_sha256=$DESCRIPTOR_SHA" \
  "loader_snapshot_scope=pre_native_host_elf_only" \
  "sanitized_exec_environment=env_i_path_home_cuda_v1" \
  "run_launcher_path=$RUN_LAUNCHER" "run_launcher_sha256=$RUN_LAUNCHER_SHA" \
  "run_body_path=$RUN_BODY" "run_body_sha256=$RUN_BODY_SHA" >"$ADMISSION"
printf 'BUILD_AND_ATTEST_OK\n'
