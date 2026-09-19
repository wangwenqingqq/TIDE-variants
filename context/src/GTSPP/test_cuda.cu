#include <cuda_runtime.h>
int main() { 
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);
  if (error == cudaSuccess) {
    printf("CUDA devices: %d\\n", deviceCount);
    return 0;
  }
  return 1;
}
