#!/usr/bin/env bash
# CPU-only plan viewer/verifier.  It has no CUDA-binary or nvidia-smi path.
set -Eeuo pipefail
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v3
PYTHON=/usr/bin/python3
VERIFY="$ROOT/tools/verify_c2_v3_execution_pins.py"
PLAN="$ROOT/protocols/c2_v3_execution_plan_v1.json"
PINS="$ROOT/provenance/c2_v3_execution_pins_v1.json"
case "${1:-}" in
  --verify-source-only)
    exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/root LANG=C "$PYTHON" "$VERIFY" --source-only --pins "$PINS"
    ;;
  --emit-plan)
    /usr/bin/env -i PATH=/usr/bin:/bin HOME=/root LANG=C "$PYTHON" "$VERIFY" --source-only --pins "$PINS" >/dev/null
    exec /bin/cat "$PLAN"
    ;;
  *)
    echo 'usage: run_c2_v3_execution_plan.sh --verify-source-only|--emit-plan' >&2
    exit 64
    ;;
esac
