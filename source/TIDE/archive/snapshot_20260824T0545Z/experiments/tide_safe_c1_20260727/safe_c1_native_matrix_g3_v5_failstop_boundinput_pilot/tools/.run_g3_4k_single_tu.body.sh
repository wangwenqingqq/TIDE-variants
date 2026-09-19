#!/bin/bash
# Body reached only through the /bin/sh env -i launcher. This script makes no
# hardware-idle decision; invoke it only after user authorization and preflight.
set -euo pipefail
umask 077
ROOT="/workspace/experiments/tide_safe_c1_20260727/safe_c1_native_matrix_g3_v5_failstop_boundinput_pilot"
LAUNCHER="$ROOT/tools/run_g3_4k_single_tu.sh"
BODY="$ROOT/tools/.run_g3_4k_single_tu.body.sh"
BINARY="$ROOT/preflight/bin/g3_4k_pilot_wrapper"
ADMISSION="$ROOT/preflight/admissions/g3_4k_pilot_wrapper.admission"
RUNS="$ROOT/runs"

usage() {
  echo "usage: run_g3_4k_single_tu.sh --gpu <single-ordinal> --run-name <new-safe-name>" >&2
  exit 69
}
[[ "$#" -eq 4 && "$1" == "--gpu" && "$3" == "--run-name" ]] || usage
GPU="$2"
NAME="$4"
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU must be a single explicit ordinal" >&2; exit 69; }
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]] ||
  { echo "run name is unsafe" >&2; exit 69; }
[[ -d "$ROOT" && ! -L "$ROOT" && "$(readlink -f "$ROOT")" == "$ROOT" &&
   -f "$LAUNCHER" && ! -L "$LAUNCHER" && -f "$BODY" && ! -L "$BODY" &&
   -f "$BINARY" && ! -L "$BINARY" && -f "$ADMISSION" && ! -L "$ADMISSION" &&
   -d "$RUNS" && ! -L "$RUNS" ]] ||
  { echo "controlled run prerequisites are missing or unsafe" >&2; exit 69; }
RUN_DIR="$RUNS/$NAME"
[[ ! -e "$RUN_DIR" && ! -e "$RUNS/.staging-$NAME" ]] ||
  { echo "run/staging name already exists; do not overwrite/retry" >&2; exit 69; }

# Exact runtime environment contract checked again by the wrapper:
# PATH, HOME, and one explicit CUDA_VISIBLE_DEVICES ordinal only.
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent CUDA_VISIBLE_DEVICES="$GPU" \
  "$BINARY" --run-dir "$RUN_DIR" --admission "$ADMISSION"
