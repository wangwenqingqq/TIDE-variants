#!/usr/bin/env bash
# Deliberately non-executing v3 entry point.  It cannot launch a CUDA binary.
set -Eeuo pipefail
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v3
case "${1:-}" in
  --verify-static)
    exec python3 "$ROOT/tools/verify_v3_static.py"
    ;;
  --execute)
    echo 'REFUSED: v3 GPU execution is not authorized by this guard.' >&2
    exit 64
    ;;
  *)
    echo 'usage: run_v3_non_execute_guard.sh --verify-static' >&2
    exit 64
    ;;
esac
