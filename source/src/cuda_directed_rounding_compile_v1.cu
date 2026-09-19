#include "aabb_static_bound_v1.cuh"

__global__ void directed_rounding_compile_probe(
    StaticProjectedAabbDeviceView view, const float* query, const int* node_ids, float* out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const int nid = node_ids[i];
    const float lb_sq = static_aabb_lb_sq_rd(view, nid, query);
    // Force both the lower-bound and strict-prune paths to be compiled.
    out[i] = static_aabb_strict_prune(view, nid, query, 1.0f) ? -lb_sq : lb_sq;
  }
}
int main() { return 0; } // compile-only artifact; it is never executed by this build.
