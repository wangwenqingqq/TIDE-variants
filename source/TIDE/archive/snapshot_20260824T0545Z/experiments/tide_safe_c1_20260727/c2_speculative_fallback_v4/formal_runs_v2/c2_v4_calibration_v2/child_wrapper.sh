#!/usr/bin/env bash
set -Eeuo pipefail
OUT="/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/formal_runs_v2/c2_v4_calibration_v2"
READY="/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/formal_runs_v2/c2_v4_calibration_v2/child.ready"
pid=$$
pgid=$(ps -o pgid= -p "$$"|tr -d ' ')
sid=$(ps -o sid= -p "$$"|tr -d ' ')
t="$READY.tmp"
printf '%s\t%s\t%s\t%s\n' "$pid" "$pgid" "$sid" safe_c2_v4_metadatafix_formal_calibration_child_v2 >"$t"
mv "$t" "$READY"
exec env CUDA_VISIBLE_DEVICES=0 "/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/bin/GTS_safe_c2_speculative_fallback_v4_metadatafix_v1" --mode calibrate --base-fvecs /workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs --query-fvecs "/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/fresh_selection_pre_gt_v1/sift_learn_fresh12524.fvecs" --groundtruth-ivecs "/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/fresh_selection_pre_gt_v1/exact_oracle_v1/groundtruth_fp32_top100.ivecs" --query-ids "/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/final_workload_v2/calibration.ids" --out "$OUT" --stage calibration --workload-manifest "/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/final_workload_v2/workload_manifest.json" --warmup-reps 0 --timed-reps 1
