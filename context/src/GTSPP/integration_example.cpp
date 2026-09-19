// Example integration of optimized search into existing GTS codebase
// This shows how to integrate search_v2_optimized.cuh with minimal changes

#include <iostream>
#include <cuda_runtime.h>
#include "search_v2_optimized.cuh"  // Use optimized version
#include "mlp_tuner_optimized.cuh"   // Use optimized tuner

// Original main function structure (simplified)
int main(int argc, char** argv) {
    // ========== Step 1: Load data (unchanged) ==========
    int n = 1000000;  // number of data points
    int dim = 128;     // dimension
    int qnum = 1000;   // number of queries
    int k = 10;        // k for kNN
    int metric_type = 2; // L2 distance
    
    // Allocate and load data (your existing code)
    short* data_h;
    short* data_d;
    int* qid_list;
    // ... your data loading code ...
    
    // ========== Step 2: Build tree (unchanged) ==========
    TN* node_list;
    int* id_list;
    int* max_node_num;
    int tree_h;
    int* empty_list;
    char* data_s;
    int* size_s;
    int* data_info;
    
    // ... your tree building code ...
    
    // ========== Step 3: Train optimized MLP model ==========
    std::cout << "\n========== Training Optimized MLP Model ==========\n";
    MLPTunerOptimized tuner;
    tuner.AutoTuneAndUpload(
        node_list,      // Tree nodes
        tree_h * 1000,  // Approximate number of nodes (adjust based on your tree)
        data_h,         // Host data
        id_list,        // ID list
        dim,            // Dimension
        n,              // Number of data points
        metric_type     // Distance metric
    );
    
    // ========== Step 4: Run optimized search ==========
    std::cout << "\n========== Running Optimized Search ==========\n";
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Call optimized search function
    searchIndexKnnV2Optimized(
        data_d,         // Device data
        node_list,      // Tree nodes
        id_list,        // ID list
        max_node_num,   // Max nodes per level
        qid_list,       // Query IDs
        qnum,           // Number of queries
        k,              // k for kNN
        tree_h,         // Tree height
        data_info,      // Data info
        empty_list,     // Empty node list
        data_s,         // String data (if applicable)
        size_s          // String sizes (if applicable)
    );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float search_time_ms = 0;
    cudaEventElapsedTime(&search_time_ms, start, stop);
    
    std::cout << "\nTotal search time: " << search_time_ms << " ms\n";
    std::cout << "Average time per query: " << (search_time_ms / qnum) << " ms\n";
    
    // ========== Step 5: Retrieve results (unchanged) ==========
    // res_dis contains the k-nearest distances for each query
    // You can retrieve results using your existing code
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // ... your cleanup code ...
    
    return 0;
}

// ========== Alternative: Gradual Migration ==========
// If you want to compare both versions side-by-side:

void compare_versions(/* parameters */) {
    // Run original version
    std::cout << "\n========== Running Original Version ==========\n";
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    
    // searchIndexKnnV2(...);  // Original function
    
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float time1 = 0;
    cudaEventElapsedTime(&time1, start1, stop1);
    
    // Save results
    float* results_original = new float[qnum * k];
    cudaMemcpy(results_original, res_dis, qnum * k * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Run optimized version
    std::cout << "\n========== Running Optimized Version ==========\n";
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    
    searchIndexKnnV2Optimized(/* parameters */);
    
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float time2 = 0;
    cudaEventElapsedTime(&time2, start2, stop2);
    
    // Save results
    float* results_optimized = new float[qnum * k];
    cudaMemcpy(results_optimized, res_dis, qnum * k * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare results
    std::cout << "\n========== Comparison ==========\n";
    std::cout << "Original time:  " << time1 << " ms\n";
    std::cout << "Optimized time: " << time2 << " ms\n";
    std::cout << "Speedup:        " << (time1 / time2) << "x\n";
    
    // Compute recall
    int matches = 0;
    for (int i = 0; i < qnum; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < k; l++) {
                if (std::abs(results_original[i*k+j] - results_optimized[i*k+l]) < 1e-5) {
                    matches++;
                    break;
                }
            }
        }
    }
    float recall = (float)matches / (qnum * k);
    std::cout << "Recall@" << k << ":     " << recall << "\n";
    
    // Cleanup
    delete[] results_original;
    delete[] results_optimized;
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
}

// ========== Configuration Tips ==========
/*
To fine-tune the optimization for your specific dataset:

1. If speedup is less than expected (< 5x):
   - In search_v2_optimized.cuh, increase depth_factor:
     float depth_factor = 1.0f + depth_ratio * 2.0f;  // was 1.5f
   
   - Lower the min_scale for deeper levels:
     float min_scale = (depth < tree_height / 2) ? 0.85f : 0.6f;  // was 0.7f

2. If recall drops below 0.95:
   - In mlp_tuner_optimized.cuh, increase positive sample weight:
     float beta = (s.label > 0.5f) ? 3.5f : 1.0f;  // was 2.5f
   
   - In search_v2_optimized.cuh, increase min_scale:
     float min_scale = (depth < tree_height / 2) ? 0.9f : 0.8f;  // was 0.85f/0.7f

3. If memory usage is too high:
   - Reduce training samples in mlp_tuner_optimized.cuh:
     train_data.reserve(30000);  // was 50000
   
   - Reduce pilot queries:
     for (int i = 0; i < 128; ++i)  // was 256

4. Dataset-specific tuning:
   - Dense datasets: Use higher max_scale (4.0f)
   - Sparse datasets: Use lower min_scale (0.65f)
   - High-dimensional: Increase training epochs (100)
   - Low-dimensional: Can use fewer samples (20000)
*/
