#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PINNED_UPSTREAM_CHECKOUT BUILD_DIRECTORY" >&2
  exit 2
fi

readonly upstream=$(cd "$1" && pwd)
readonly build=$2
readonly script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly adapter_root=$(cd "${script_dir}/.." && pwd)
readonly expected_commit=453149369708e84fb25433c4c9d1d748ecf6afaf

test "$(git -C "${upstream}" rev-parse HEAD)" = "${expected_commit}"
test -z "$(git -C "${upstream}" status --porcelain)"
test "$(sha256sum "${upstream}/fingerprintdb_cuda.h" | awk '{print $1}')" = \
  bbce788edfe3dab4096cad8fd19c13e892fc2c6406847cccba365f7208bee935
test "$(sha256sum "${upstream}/fingerprintdb_cuda.cu" | awk '{print $1}')" = \
  ece1df84a66a3dba066e870507b6f339b47b0978c06d600f78941cbeca2a8478

if [[ -e "${build}" ]]; then
  echo "refusing to replace existing build directory: ${build}" >&2
  exit 3
fi

cmake -S "${adapter_root}/tools" -B "${build}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DGPUSIMILARITY_SOURCE="${upstream}"
cmake --build "${build}" --verbose
