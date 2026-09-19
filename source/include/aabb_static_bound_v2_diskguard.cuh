#pragma once
// Static-only Safe-C2-AABB v2 disk-guard: raw-float32 L2 projected-box lower bound.
// No gamma, fallback, update, insertion, or online calibration API is present.
#include <cuda_runtime.h>
#include <math.h>

struct StaticProjectedAabbDeviceView {
  const float* lo;       // node-major [node_count * projection_dims]
  const float* hi;       // node-major [node_count * projection_dims]
  const int* dimensions; // original coordinate index, length projection_dims
  int projection_dims;
  int full_dimension;
};

// Every arithmetic operation that contributes to the lower bound rounds down.
// This deliberately produces a weaker bound rather than risking an upward-rounded
// false prune. disk is an L2 distance; comparison is done in squared space.
__device__ __forceinline__ float static_aabb_rd_sub_nonnegative(float a, float b) {
  const float x = __fsub_rd(a, b);
  return x > 0.0f ? x : 0.0f;
}
__device__ __forceinline__ float static_aabb_lb_sq_rd(
    const StaticProjectedAabbDeviceView view, int nid, const float* query) {
  float sum = 0.0f;
  const size_t base = static_cast<size_t>(nid) * static_cast<size_t>(view.projection_dims);
  for (int slot = 0; slot < view.projection_dims; ++slot) {
    const int d = view.dimensions[slot];
    const float q = query[d];
    const float lo = view.lo[base + static_cast<size_t>(slot)];
    const float hi = view.hi[base + static_cast<size_t>(slot)];
    float delta = 0.0f;
    if (q < lo) delta = static_aabb_rd_sub_nonnegative(lo, q);
    else if (q > hi) delta = static_aabb_rd_sub_nonnegative(q, hi);
    const float term = __fmul_rd(delta, delta);
    sum = __fadd_rd(sum, term);
  }
  return sum;
}
// For the SIFT integer-float contract audited outside this header, disk_l2 is
// sqrtf of an exactly represented squared sum. Advance one IEEE-754 float so
// disk_upper is an upper bound on the real L2 radius before the upward square.
__device__ __forceinline__ float static_aabb_next_up_nonnegative(float x) {
  if (!isfinite(x) || x < 0.0f) return x;
  const unsigned int bits = __float_as_uint(x);
  if (bits >= 0x7f7fffffU) return __uint_as_float(0x7f800000U);
  return __uint_as_float(bits + 1U);
}
__device__ __forceinline__ bool static_aabb_strict_prune(
    const StaticProjectedAabbDeviceView view, int nid, const float* query, float disk_l2) {
  if (!(disk_l2 >= 0.0f) || !isfinite(disk_l2)) return false;
  const float disk_upper = static_aabb_next_up_nonnegative(disk_l2);
  const float disk_sq_up = __fmul_ru(disk_upper, disk_upper);
  return static_aabb_lb_sq_rd(view, nid, query) > disk_sq_up;
}
