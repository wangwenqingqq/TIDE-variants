#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DESTINATION" >&2
  exit 2
fi

readonly destination=$1
readonly repository=https://github.com/schrodinger/gpusimilarity
readonly commit=453149369708e84fb25433c4c9d1d748ecf6afaf

if [[ -e "${destination}" ]]; then
  echo "refusing to replace existing destination: ${destination}" >&2
  exit 3
fi

git clone --no-checkout "${repository}" "${destination}"
git -C "${destination}" checkout --detach "${commit}"
test "$(git -C "${destination}" rev-parse HEAD)" = "${commit}"
test -z "$(git -C "${destination}" status --porcelain)"

cat <<EOF
repository=${repository}
commit=${commit}
license_sha256=$(sha256sum "${destination}/LICENSE" | awk '{print $1}')
core_header_sha256=$(sha256sum "${destination}/fingerprintdb_cuda.h" | awk '{print $1}')
core_cuda_sha256=$(sha256sum "${destination}/fingerprintdb_cuda.cu" | awk '{print $1}')
EOF

