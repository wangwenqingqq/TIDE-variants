#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an explicitly leased idle GPU, e.g. 0}"
W=/workspace/RT-TIDE
P="$W/build/rt-tide-fused-probe-optix91-sm120/rt_tide_fused_probe"
BASE="$W/data/sift1m/sift_base.fvecs"
QUERY="$W/data/sift1m/sift_query.fvecs"

test -x "$P"
test -f "$BASE"
test -f "$QUERY"

run() {
  local out="$1"
  shift
  "$P" "$@" | tee "$out"
}

# LibRTS official correctness suite.
(
  cd "$W/build/rtspatial-optix91-sm120-v2/bin"
  ./rtspatial_tests --gtest_color=no     2>&1 | tee "$W/logs/rtspatial_optix91_sm120_tests_gpu${CUDA_VISIBLE_DEVICES}.log"
)

# TIDE split: frozen Base=900k, live Delta=100k, k=10.
D="$W/results/sift1m_base900k_delta100k_q32_k10/pca2_rt_case"
run "$D/fused_batch100000_optix91_sm120_rerun.json"   "$D/points.f32" "$D/query_boxes.f32" "$BASE" "$QUERY" "$D/taus.f32"   900000 100000 32 128 100000 20
run "$D/fused_batch10000_optix91_sm120_rerun.json"   "$D/points.f32" "$D/query_boxes.f32" "$BASE" "$QUERY" "$D/taus.f32"   900000 100000 32 128 10000 20

D="$W/results/sift1m_base900k_delta100k_q1_k10/pca2_rt_case"
run "$D/fused_batch100000_optix91_sm120_rerun.json"   "$D/points.f32" "$D/query_boxes.f32" "$BASE" "$QUERY" "$D/taus.f32"   900000 100000 1 128 100000 50

# Unified searchable pool scaling: 1m vectors with the safe Base-derived radius.
D="$W/results/sift1m_pool1m_q32_k10/pca2_rt_case"
run "$D/fused_batch1000000_optix91_sm120_rerun.json"   "$D/points.f32" "$D/query_boxes.f32" "$BASE" "$QUERY" "$D/taus.f32"   0 1000000 32 128 1000000 10
run "$D/fused_batch100000_optix91_sm120_rerun.json"   "$D/points.f32" "$D/query_boxes.f32" "$BASE" "$QUERY" "$D/taus.f32"   0 1000000 32 128 100000 10

D="$W/results/sift1m_pool1m_q1_k10/pca2_rt_case"
run "$D/fused_batch1000000_optix91_sm120_rerun.json"   "$D/points.f32" "$D/query_boxes.f32" "$BASE" "$QUERY" "$D/taus.f32"   0 1000000 1 128 1000000 20

# Empirical approximate alpha=0.5 mode; this is not an exactness certificate.
D="$W/results/sift1m_base900k_delta100k_q32_k10/pca2_alpha0.5"
run "$D/fused_optix91_sm120_rerun.json"   "$D/points.f32" "$D/query_boxes.f32" "$BASE" "$QUERY" "$D/taus.f32"   900000 100000 32 128 100000 20

D="$W/results/sift1m_pool1m_q32_k10/pca2_alpha0.5"
run "$D/fused_optix91_sm120_rerun.json"   "$D/points.f32" "$D/query_boxes.f32" "$BASE" "$QUERY" "$D/taus.f32"   0 1000000 32 128 1000000 10
run "$D/fused_batch100000_optix91_sm120_rerun.json"   "$D/points.f32" "$D/query_boxes.f32" "$BASE" "$QUERY" "$D/taus.f32"   0 1000000 32 128 100000 10
