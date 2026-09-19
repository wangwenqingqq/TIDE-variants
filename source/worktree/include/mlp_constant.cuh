#ifndef MLP_CONSTANT_CUH
#define MLP_CONSTANT_CUH

#include <cuda_runtime.h>

// =========================================================
// 关键点：必须使用 extern 声明，告诉编译器 "变量在别处定义"
// =========================================================
extern __device__ __constant__ float input_scale[3];
extern __device__ __constant__ float c_W1[3 * 8];
extern __device__ __constant__ float c_b1[8];
extern __device__ __constant__ float c_W2[8];
extern __device__ __constant__ float c_b2; // 注意这里是标量

// 声明上传函数 (实现放在 src/mlp_constants.cu)
void upload_mlp_constants(float* h_scale, float* h_W1, float* h_b1, float* h_W2, float* h_b2);

#endif // MLP_CONSTANT_CUH