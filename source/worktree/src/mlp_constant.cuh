#ifndef MLP_CONSTANT_CUH
#define MLP_CONSTANT_CUH

#include <cuda_runtime.h>

// =========================================================
// 宏开关魔法：
// 如果定义了 MLP_MAIN_FILE，则分配内存 (定义)
// 否则，只告诉编译器变量存在 (extern 声明)
// =========================================================
#ifdef MLP_MAIN_FILE
    #define MLP_EXTERN 
#else
    #define MLP_EXTERN extern
#endif

// 使用宏来修饰变量
MLP_EXTERN __device__ __constant__ float input_scale[3];
MLP_EXTERN __device__ __constant__ float c_W1[24]; // 3*8
MLP_EXTERN __device__ __constant__ float c_b1[8];
MLP_EXTERN __device__ __constant__ float c_W2[8];
MLP_EXTERN __device__ __constant__ float c_b2; 

// 声明上传函数 (实现放在 main.cu)
void upload_mlp_constants(float* h_scale, float* h_W1, float* h_b1, float* h_W2, float* h_b2);

#endif // MLP_CONSTANT_CUH