// GPU计时器使用示例
// 展示如何使用AdvancedGPUTimer进行精确的GPU性能测量

#include "../include/gpu_timer.cuh"
#include <cuda_runtime.h>
#include <iostream>

// 示例CUDA内核
__global__ void example_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 模拟一些计算
        for (int i = 0; i < 1000; i++) {
            data[idx] = data[idx] * 1.1f + 0.1f;
        }
    }
}

int main() {
    const int N = 1024 * 1024;
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    
    // 分配GPU内存
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    // 初始化数据
    cudaMemset(d_data, 1.0f, N * sizeof(float));
    
    // 创建高级GPU计时器
    AdvancedGPUTimer index_timer("Index Construction");
    AdvancedGPUTimer search_timer("Search Operations");
    
    std::cout << "=== GPU计时器使用示例 ===" << std::endl;
    
    // 示例1: 单次测量
    std::cout << "\n1. 单次测量示例:" << std::endl;
    index_timer.start();
    example_kernel<<<grid_size, block_size>>>(d_data, N);
    cudaDeviceSynchronize();
    index_timer.stop();
    std::cout << "内核执行时间: " << index_timer.elapsed() << " ms" << std::endl;
    
    // 示例2: 多次测量和统计
    std::cout << "\n2. 多次测量示例:" << std::endl;
    for (int i = 0; i < 5; i++) {
        search_timer.start();
        example_kernel<<<grid_size, block_size>>>(d_data, N);
        cudaDeviceSynchronize();
        search_timer.add_measurement(); // 自动停止并记录
    }
    
    // 打印统计信息
    search_timer.print_statistics();
    
    // 示例3: 使用简单计时器
    std::cout << "\n3. 简单计时器示例:" << std::endl;
    GPUTimer simple_timer;
    simple_timer.start();
    example_kernel<<<grid_size, block_size>>>(d_data, N);
    cudaDeviceSynchronize();
    simple_timer.stop();
    std::cout << "简单计时器结果: " << simple_timer.elapsed() << " ms" << std::endl;
    
    // 清理
    cudaFree(d_data);
    
    std::cout << "\n=== 示例完成 ===" << std::endl;
    return 0;
}
