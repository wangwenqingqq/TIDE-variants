#ifndef GPU_TIMER_CUH
#define GPU_TIMER_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

/**
 * 高级GPU计时器类
 * 支持多次测量、统计分析和自动报告
 */
class AdvancedGPUTimer {
private:
    cudaEvent_t start_event, stop_event;
    std::vector<float> measurements;
    std::string timer_name;
    bool is_running;
    
public:
    AdvancedGPUTimer(const std::string& name = "GPU Timer") 
        : timer_name(name), is_running(false) {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    ~AdvancedGPUTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        if (is_running) {
            std::cerr << "Warning: Timer " << timer_name << " is already running!" << std::endl;
            return;
        }
        cudaEventRecord(start_event, 0);
        is_running = true;
    }
    
    void stop() {
        if (!is_running) {
            std::cerr << "Warning: Timer " << timer_name << " is not running!" << std::endl;
            return;
        }
        cudaEventRecord(stop_event, 0);
        is_running = false;
    }
    
    float elapsed() {
        if (is_running) {
            std::cerr << "Warning: Timer " << timer_name << " is still running!" << std::endl;
            return 0.0f;
        }
        
        float milliseconds = 0;
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        measurements.push_back(milliseconds);
        return milliseconds;
    }
    
    float elapsed_seconds() {
        return elapsed() / 1000.0f;
    }
    
    void add_measurement() {
        if (is_running) {
            stop();
        }
        elapsed();
    }
    
    void reset() {
        measurements.clear();
        is_running = false;
    }
    
    void print_statistics() {
        if (measurements.empty()) {
            std::cout << timer_name << ": No measurements recorded." << std::endl;
            return;
        }
        
        float total = 0, min_time = measurements[0], max_time = measurements[0];
        for (float time : measurements) {
            total += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        
        float avg_time = total / measurements.size();
        
        std::cout << "\n=== " << timer_name << " Statistics ===" << std::endl;
        std::cout << "Measurements: " << measurements.size() << std::endl;
        std::cout << "Total time: " << total << " ms" << std::endl;
        std::cout << "Average time: " << avg_time << " ms" << std::endl;
        std::cout << "Min time: " << min_time << " ms" << std::endl;
        std::cout << "Max time: " << max_time << " ms" << std::endl;
        std::cout << "================================" << std::endl;
    }
    
    float get_total_time() {
        float total = 0;
        for (float time : measurements) {
            total += time;
        }
        return total;
    }
    
    float get_average_time() {
        if (measurements.empty()) return 0.0f;
        return get_total_time() / measurements.size();
    }
};

/**
 * 简单的GPU计时器类（保持向后兼容）
 */
class GPUTimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    GPUTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    ~GPUTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        cudaEventRecord(start_event, 0);
    }
    
    void stop() {
        cudaEventRecord(stop_event, 0);
    }
    
    float elapsed() {
        float milliseconds = 0;
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        return milliseconds;
    }
    
    float elapsed_seconds() {
        return elapsed() / 1000.0f;
    }
};

#endif // GPU_TIMER_CUH
