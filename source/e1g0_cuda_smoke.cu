// E1-G0 infrastructure-only CUDA runtime smoke.
// This does not invoke GTS, Safe-C1, any updater, or any benchmark workload.
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static std::string json_escape(const char* text) {
  std::string out;
  for (const unsigned char c : std::string(text)) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          const char hex[] = "0123456789abcdef";
          out += "\\u00";
          out += hex[(c >> 4) & 0x0f];
          out += hex[c & 0x0f];
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

static int fail(const char* stage, cudaError_t err) {
  std::cout << "{\"experiment_id\":\"E1-G0\",\"scope\":\"CUDA runtime infrastructure smoke only; not GTS evidence\","
            << "\"status\":\"FAIL\",\"stage\":\"" << json_escape(stage) << "\","
            << "\"cuda_error\":\"" << json_escape(cudaGetErrorString(err)) << "\","
            << "\"cuda_error_code\":" << static_cast<int>(err) << "}" << std::endl;
  return 2;
}

#define CUDA_OR_RETURN(expr, stage) \
  do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) return fail((stage), _err); \
  } while (0)

__global__ void affine_kernel(const int* in, int* out, int n) {
  const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 3 * in[i] + 1;
}

int main() {
  constexpr int kN = 4096;
  int logical_device = -1;
  int driver_version = -1;
  int runtime_version = -1;
  CUDA_OR_RETURN(cudaGetDevice(&logical_device), "cudaGetDevice");
  CUDA_OR_RETURN(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
  CUDA_OR_RETURN(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");

  cudaDeviceProp prop{};
  CUDA_OR_RETURN(cudaGetDeviceProperties(&prop, logical_device), "cudaGetDeviceProperties");

  std::vector<int> host_in(kN), host_out(kN, 0);
  for (int i = 0; i < kN; ++i) host_in[i] = i;

  int* device_in = nullptr;
  int* device_out = nullptr;
  CUDA_OR_RETURN(cudaMalloc(&device_in, kN * sizeof(int)), "cudaMalloc(device_in)");
  CUDA_OR_RETURN(cudaMalloc(&device_out, kN * sizeof(int)), "cudaMalloc(device_out)");
  CUDA_OR_RETURN(cudaMemcpy(device_in, host_in.data(), kN * sizeof(int), cudaMemcpyHostToDevice),
                 "cudaMemcpy(H2D)");

  constexpr int kThreads = 256;
  const int blocks = (kN + kThreads - 1) / kThreads;
  affine_kernel<<<blocks, kThreads>>>(device_in, device_out, kN);
  CUDA_OR_RETURN(cudaGetLastError(), "affine_kernel launch");
  CUDA_OR_RETURN(cudaDeviceSynchronize(), "affine_kernel synchronize");
  CUDA_OR_RETURN(cudaMemcpy(host_out.data(), device_out, kN * sizeof(int), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(D2H)");

  bool passed = true;
  std::int64_t checksum = 0;
  for (int i = 0; i < kN; ++i) {
    passed = passed && (host_out[i] == 3 * i + 1);
    checksum += host_out[i];
  }
  cudaFree(device_in);
  cudaFree(device_out);

  std::cout << "{\"experiment_id\":\"E1-G0\","
            << "\"scope\":\"CUDA runtime infrastructure smoke only; not GTS evidence\","
            << "\"status\":\"" << (passed ? "PASS" : "FAIL") << "\","
            << "\"logical_cuda_device\":" << logical_device << ","
            << "\"device_name\":\"" << json_escape(prop.name) << "\","
            << "\"compute_capability\":\"" << prop.major << "." << prop.minor << "\","
            << "\"sm_count\":" << prop.multiProcessorCount << ","
            << "\"driver_version\":" << driver_version << ","
            << "\"runtime_version\":" << runtime_version << ","
            << "\"elements\":" << kN << ","
            << "\"checksum\":" << checksum << "}" << std::endl;
  return passed ? 0 : 3;
}
