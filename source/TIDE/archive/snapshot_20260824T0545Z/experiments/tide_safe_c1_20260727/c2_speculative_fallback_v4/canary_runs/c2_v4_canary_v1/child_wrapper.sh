#!/usr/bin/env bash
set -Eeuo pipefail
OUT="/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/canary_runs/c2_v4_canary_v1"
READY="/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/canary_runs/c2_v4_canary_v1/child.ready"
pid=$$
pgid=$(ps -o pgid= -p "$$" | tr -d ' ')
sid=$(ps -o sid= -p "$$" | tr -d ' ')
tmp="$READY.tmp"
printf '%s\t%s\t%s\t%s\n' "$pid" "$pgid" "$sid" safe_c2_v4_canary_child_v1 >"$tmp"
mv "$tmp" "$READY"
exec env CUDA_VISIBLE_DEVICES=0 "/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/bin/GTS_safe_c2_speculative_fallback_v4_sift1m" --mode calibrate \
  --base-fvecs /workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs \
  --query-fvecs /workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/fresh_selection_pre_gt_v1/sift_learn_fresh12524.fvecs \
  --groundtruth-ivecs /workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/fresh_selection_pre_gt_v1/exact_oracle_v1/groundtruth_fp32_top100.ivecs \
  --query-ids "/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/canary_workload_v1/qualification_canary.ids" --out "$OUT" --stage calibration \
  --workload-manifest "/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/canary_workload_v1/workload_manifest.json" --warmup-reps 0 --timed-reps 1
