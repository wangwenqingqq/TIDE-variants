#!/usr/bin/env bash
# CPU-only primary-profile plan. Only the outer guard may ever execute nsys.
set -Eeuo pipefail
umask 077
ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v6_capacity_snapshot'
SELF="$ROOT/run_c1_profile_primary_v6.sh"
[[ "$0" == "$SELF" && -f "$SELF" && ! -L "$SELF" ]] || { echo 'C1-V6-PROFILE-PLAN BLOCKED: noncanonical path' >&2; exit 64; }
[[ -z "${PYTHONOPTIMIZE:-}" ]] || { echo 'C1-V6-PROFILE-PLAN BLOCKED: PYTHONOPTIMIZE must be unset' >&2; exit 64; }
[[ "${1:-}" == --emit-primary-profile-plan && "$#" -eq 1 ]] || { echo 'Usage: run_c1_profile_primary_v6.sh --emit-primary-profile-plan' >&2; exit 64; }
printf '1\tE_G_c1_off_reference\n1\tP_F_full_C1\n'

