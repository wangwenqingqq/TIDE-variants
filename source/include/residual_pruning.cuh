// Safe-C2 corrected-after-submit Eq. (2) scale implementation.
//
// Scope: static vector-KNN empirical held-out calibration only.  gamma_l >= 1
// scales the already-certified Eq. (1) interval lower bound:
//   LB_eff = gamma_l * LB_tri.
// gamma>1 is NOT a new metric lower bound and has no universal/future-query
// zero-false-negative guarantee.  This header intentionally contains no online
// adaptation, learned residual, LUT, or deployment policy.
#pragma once

#include <cuda_runtime.h>

#define RP_MAX_LEVELS 8
#define RP_LUT_SIZE 16  // retained only for the legacy upload ABI; Safe-C2 disables LUTs.

#ifdef RP_DEFINE_CONSTANTS
#define RP_DECL
#else
#define RP_DECL extern
#endif

// alpha/beta/LUT arrays are ABI compatibility storage for the isolated legacy
// uploader. Safe-C2 uploads zeros and never reads them in rp_predict.
RP_DECL __device__ __constant__ float c_rp_alpha[RP_MAX_LEVELS];
RP_DECL __device__ __constant__ float c_rp_beta[RP_MAX_LEVELS];
RP_DECL __device__ __constant__ float c_rp_gamma[RP_MAX_LEVELS];
RP_DECL __device__ __constant__ float c_lut_breaks[RP_LUT_SIZE];
RP_DECL __device__ __constant__ float c_lut_slopes[RP_LUT_SIZE];
RP_DECL __device__ __constant__ float c_lut_intercepts[RP_LUT_SIZE];
RP_DECL __device__ __constant__ int c_lut_num_segments;
// 0: all gamma=1 baseline; 1: Safe-C2 Eq. (2) scaling. No other mode is used.
RP_DECL __device__ __constant__ int c_rp_mode;

__device__ __forceinline__ float rp_predict(float dis_lb, float /*disk_val*/, int level) {
  if (c_rp_mode == 0) return 0.0f;
  const float scale = c_rp_gamma[level];
  // dis_lb + (scale-1)*dis_lb = scale*dis_lb (Eq. 2).
  return fmaxf(0.0f, (scale - 1.0f) * dis_lb);
}

void upload_rp_constants(float* h_alpha, float* h_beta, float* h_gamma, int num_levels,
                         float* h_lut_breaks, float* h_lut_slopes, float* h_lut_intercepts,
                         int lut_size, int mode);
