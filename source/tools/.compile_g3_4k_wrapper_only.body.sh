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
# Body reached only via the /bin/sh env -i launcher. It never links or runs a
# CUDA binary; it compiles a sealed source snapshot after static audit approval.
set -euo pipefail
umask 077
ROOT="/workspace/experiments/tide_safe_c1_20260727/safe_c1_native_matrix_g3_v8_loader_allowlist_pilot"
NVCC="/usr/local/cuda-13.1/bin/nvcc"
HOST_CXX="/usr/bin/g++"
OUT="$ROOT/preflight/compile_wrapper_only"
SNAPSHOT="$OUT/sealed_source"
WRAPPER="$ROOT/runner/g3_4k_pilot_wrapper.cu"
SNAPSHOT_WRAPPER="$SNAPSHOT/runner/g3_4k_pilot_wrapper.cu"
LAUNCHER="$ROOT/tools/compile_g3_4k_wrapper_only.sh"
BODY="$ROOT/tools/.compile_g3_4k_wrapper_only.body.sh"
WRAPPER_SHA="859152ef7782298c881fccb33f851417f6f0f34cac35480e1a0b2da0d05e028d"

[[ "$#" -eq 0 ]] || { echo "usage: compile_g3_4k_wrapper_only.sh" >&2; exit 69; }
[[ -d "$ROOT" && ! -L "$ROOT" && "$(readlink -f "$ROOT")" == "$ROOT" ]] ||
  { echo "unsafe fixed v6 root" >&2; exit 69; }
[[ -x "$NVCC" && -x "$HOST_CXX" && -f "$WRAPPER" && ! -L "$WRAPPER" &&
   -f "$LAUNCHER" && ! -L "$LAUNCHER" && -f "$BODY" && ! -L "$BODY" ]] ||
  { echo "missing fixed compiler/source/launcher" >&2; exit 69; }
[[ ! -e "$OUT" ]] || { echo "compile-only output exists; do not overwrite/retry" >&2; exit 69; }

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

mkdir "$OUT"
install -d -m 700 "$SNAPSHOT"
snapshot_file "$WRAPPER_SHA" "runner/g3_4k_pilot_wrapper.cu"
snapshot_file "12ef95e115c92ed2ebc86fc116aa441aacdbadfe850676a684c0b9119aa8b138" "src/g3_safe_c1_native_matrix.cu"
snapshot_file "65688fedcbb05a1fff680555e3139b6e640bfdd72288b93dfccddcec90dfbf01" "src/g3_safe_search_v2.cuh"
snapshot_file "c1324bef173358c31a8371e1f050d4372cd1c92cd98c71af3832b2d19c769fd6" "reference/include/tree.cuh"
snapshot_file "b8be03246ec82cadd3228b75a6dfd91c552ccfa3124d2ca9470e28e0031ffd8a" "reference/include/file.cuh"
snapshot_file "622d0977e49d80d5791364bc12463de9bce5aaca324d6a681512004c20d100cb" "reference/include/config.cuh"
snapshot_file "cf6545624c0cbc51744b978298e6d138591521c7b600ece8812b6195f735e559" "reference/include/mlp_constant.cuh"
snapshot_file "745a4bd5564b8085c40a48abac625f24dcbacb8a996d29e4cba311dd936e33b2" "reference/include/residual_pruning.cuh"

clean_tool "$NVCC" --version >"$OUT/nvcc_version.txt"
clean_tool "$HOST_CXX" --version >"$OUT/host_cxx_version.txt"
LAUNCHER_SHA="$(sha256sum "$LAUNCHER" | awk '{print $1}')"
BODY_SHA="$(sha256sum "$BODY" | awk '{print $1}')"
SOURCE_SNAPSHOT_SHA="$(source_closure_descriptor)"
COMPILE_COMMAND="$OUT/compile_command.txt"
printf '%s\n' \
  "schema=safe-c1-g3-compile-command-v1" \
  "single_translation_unit=true" \
  "link_or_runtime=false" \
  "sanitized_build_environment=env_i_private_body_trampoline_v2" \
  "compiler_path=$NVCC" \
  "host_compiler_path=$HOST_CXX" \
  "translation_unit=$SNAPSHOT_WRAPPER" \
  "source_snapshot_path=$SNAPSHOT" \
  "source_snapshot_descriptor_sha256=$SOURCE_SNAPSHOT_SHA" \
  "wrapper_sha256=$WRAPPER_SHA" \
  "language_standard=c++17" \
  "gpu_arch=compute_80" \
  "gpu_code=compute_80" \
  "source_input_count=1" \
  "object_or_archive_inputs=absent" \
  "include_order=$SNAPSHOT/src:$SNAPSHOT/reference/include" \
  "compile_launcher_path=$LAUNCHER" \
  "compile_launcher_sha256=$LAUNCHER_SHA" \
  "compile_body_path=$BODY" \
  "compile_body_sha256=$BODY_SHA" >"$COMPILE_COMMAND"
COMPILE_COMMAND_SHA="$(sha256sum "$COMPILE_COMMAND" | awk '{print $1}')"
clean_tool "$NVCC" -std=c++17 -ccbin "$HOST_CXX" -arch=compute_80 -code=compute_80 \
  -c -I"$SNAPSHOT/src" -I"$SNAPSHOT/reference/include" "$SNAPSHOT_WRAPPER" \
  -o "$OUT/g3_4k_pilot_wrapper.compute_80.o" >"$OUT/nvcc_compile.log" 2>&1
OBJECT_SHA="$(sha256sum "$OUT/g3_4k_pilot_wrapper.compute_80.o" | awk '{print $1}')"
printf '%s  %s\n' "$OBJECT_SHA" "$OUT/g3_4k_pilot_wrapper.compute_80.o" >"$OUT/object.sha256"
COMPILE_LOG_SHA="$(sha256sum "$OUT/nvcc_compile.log" | awk '{print $1}')"
NVCC_VERSION_SHA="$(sha256sum "$OUT/nvcc_version.txt" | awk '{print $1}')"
HOST_CXX_VERSION_SHA="$(sha256sum "$OUT/host_cxx_version.txt" | awk '{print $1}')"
COMPILE_WITNESS="$OUT/compile_witness.txt"
printf '%s\n' \
  "schema=safe-c1-g3-compile-witness-v1" \
  "single_translation_unit=true" \
  "link_or_runtime=false" \
  "witness_role=sealed_source_pre_link_compile_not_link_input" \
  "sanitized_build_environment=env_i_private_body_trampoline_v2" \
  "compiler_path=$NVCC" \
  "host_compiler_path=$HOST_CXX" \
  "translation_unit=$SNAPSHOT_WRAPPER" \
  "source_snapshot_path=$SNAPSHOT" \
  "source_snapshot_descriptor_sha256=$SOURCE_SNAPSHOT_SHA" \
  "wrapper_sha256=$WRAPPER_SHA" \
  "compile_command_sha256=$COMPILE_COMMAND_SHA" \
  "compile_object_sha256=$OBJECT_SHA" \
  "compile_log_sha256=$COMPILE_LOG_SHA" \
  "nvcc_version_sha256=$NVCC_VERSION_SHA" \
  "host_cxx_version_sha256=$HOST_CXX_VERSION_SHA" \
  "compile_launcher_path=$LAUNCHER" \
  "compile_launcher_sha256=$LAUNCHER_SHA" \
  "compile_body_path=$BODY" \
  "compile_body_sha256=$BODY_SHA" >"$COMPILE_WITNESS"
COMPILE_WITNESS_SHA="$(sha256sum "$COMPILE_WITNESS" | awk '{print $1}')"
printf '%s\n' \
  "schema=safe-c1-g3-compile-only-manifest-v6" \
  "single_translation_unit=true" \
  "link_or_runtime=false" \
  "sanitized_build_environment=env_i_private_body_trampoline_v2" \
  "compiler_path=$NVCC" \
  "host_compiler_path=$HOST_CXX" \
  "translation_unit=$SNAPSHOT_WRAPPER" \
  "source_snapshot_path=$SNAPSHOT" \
  "source_snapshot_descriptor_sha256=$SOURCE_SNAPSHOT_SHA" \
  "wrapper_sha256=$WRAPPER_SHA" \
  "source_input_count=1" \
  "object_or_archive_inputs=absent" \
  "include_order=$SNAPSHOT/src:$SNAPSHOT/reference/include" \
  "compile_command_sha256=$COMPILE_COMMAND_SHA" \
  "compile_witness_sha256=$COMPILE_WITNESS_SHA" \
  "compile_object_sha256=$OBJECT_SHA" \
  "compile_launcher_path=$LAUNCHER" \
  "compile_launcher_sha256=$LAUNCHER_SHA" \
  "compile_body_path=$BODY" \
  "compile_body_sha256=$BODY_SHA" \
  >"$OUT/compile_only_contract.txt"
printf 'COMPILE_ONLY_OK\n'

G3_CLEAN_PAYLOAD
# exec only returns on a launcher failure.
exit 69
