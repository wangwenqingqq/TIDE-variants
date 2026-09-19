#!/usr/bin/env bash
# CPU-only measured-plan emitter for the v7 outer guard.
# It intentionally cannot launch C1Microbench, nsys, CUDA, or nvidia-smi.
set -Eeuo pipefail
umask 077
ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v7_profile_manifest_repair'
SELF="$ROOT/run_c1_four_variants_v7.sh"
[[ "$0" == "$SELF" && -f "$SELF" && ! -L "$SELF" ]] || { echo 'C1-V7-PLAN BLOCKED: noncanonical engine path' >&2; exit 64; }
[[ "${PYTHONOPTIMIZE:-}" == "" ]] || { echo 'C1-V7-PLAN BLOCKED: PYTHONOPTIMIZE must be unset' >&2; exit 64; }
[[ "${1:-}" == --emit-measured-plan && "$#" -eq 1 ]] || {
  echo 'Usage: run_c1_four_variants_v7.sh --emit-measured-plan' >&2; exit 64; }
# Output is a fixed, machine-readable 5x4 Latin-order protocol. The outer
# guard owns every GPU preflight, binary invocation and post-launch snapshot.
cat <<'PLAN'
1	E_G_c1_off_reference
1	P_G_workspace_only
1	E_F_fastpath_only
1	P_F_full_C1
2	P_G_workspace_only
2	E_F_fastpath_only
2	P_F_full_C1
2	E_G_c1_off_reference
3	E_F_fastpath_only
3	P_F_full_C1
3	E_G_c1_off_reference
3	P_G_workspace_only
4	P_F_full_C1
4	E_G_c1_off_reference
4	P_G_workspace_only
4	E_F_fastpath_only
5	E_G_c1_off_reference
5	E_F_fastpath_only
5	P_G_workspace_only
5	P_F_full_C1
PLAN

