// Fair same-process lower-bound probe for TIDE-G1.
// Compares the archived native GTS range path at r=0 (its most selective
// possible radius) with a coalesced exhaustive GPU distance scan over the same
// frozen Base and query IDs. Both timings include launch + synchronization.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "update.cuh"

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(2);
}

void cuda_ok(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        std::ostringstream out;
        out << where << ": " << cudaGetErrorString(status);
        fail(out.str());
    }
}

double mean(const std::vector<double>& values) {
    return values.empty() ? 0.0 :
        std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t pos = static_cast<size_t>(std::ceil(q * values.size())) - 1;
    return values[std::min(pos, values.size() - 1)];
}

bool load_sift_prefix(const char* path, int wanted, int expected_dim,
                      std::vector<float>& out) {
    std::ifstream input(path);
    if (!input) return false;
    std::string line;
    if (!std::getline(input, line)) return false;
    std::istringstream header(line);
    int dim = 0, rows = 0, metric = -1;
    header >> dim >> rows >> metric;
    if (dim != expected_dim || rows < wanted || metric != 2) return false;
    out.assign(static_cast<size_t>(wanted) * dim, 0.0f);
    for (int row = 0; row < wanted; ++row) {
        if (!std::getline(input, line)) return false;
        char* p = line.data();
        for (int col = 0; col < dim; ++col) {
            char* end = nullptr;
            out[static_cast<size_t>(row) * dim + col] = std::strtof(p, &end);
            if (end == p) return false;
            p = end;
        }
    }
    return true;
}

float l2_squared(const std::vector<float>& values, int a, int b, int dim) {
    const float* pa = values.data() + static_cast<size_t>(a) * dim;
    const float* pb = values.data() + static_cast<size_t>(b) * dim;
    float total = 0.0f;
    for (int j = 0; j < dim; ++j) {
        const float d = pa[j] - pb[j];
        total += d * d;
    }
    return total;
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__global__ void dense_range_kernel(const float* data, const int* query_ids,
                                   float radius2, size_t n, int nq, int dim,
                                   float* output, uint32_t* counts) {
    const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t warp = thread >> 5;
    const int lane = threadIdx.x & 31;
    const size_t total_pairs = n * static_cast<size_t>(nq);
    if (warp >= total_pairs) return;
    const int qslot = static_cast<int>(warp / n);
    const size_t point = warp - static_cast<size_t>(qslot) * n;
    const int query = query_ids[qslot];
    const float* a = data + point * dim;
    const float* b = data + static_cast<size_t>(query) * dim;
    float sum = 0.0f;
    for (int j = lane; j < dim; j += 32) {
        const float d = a[j] - b[j];
        sum += d * d;
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        output[warp] = sum;
        if (counts != nullptr && sum <= radius2) atomicAdd(counts + qslot, 1U);
    }
}

struct Timing {
    double device_mean_ms = 0.0;
    double device_p50_ms = 0.0;
    double device_p95_ms = 0.0;
    double wall_mean_ms = 0.0;
    double wall_p50_ms = 0.0;
    double wall_p95_ms = 0.0;
};

template <typename Launch>
Timing benchmark(Launch launch, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        launch();
        cuda_ok(cudaDeviceSynchronize(), "warmup sync");
    }
    cudaEvent_t start = nullptr, stop = nullptr;
    cuda_ok(cudaEventCreate(&start), "create start event");
    cuda_ok(cudaEventCreate(&stop), "create stop event");
    std::vector<double> device, wall;
    for (int i = 0; i < repeats; ++i) {
        const auto begin = Clock::now();
        cuda_ok(cudaEventRecord(start), "record start");
        launch();
        cuda_ok(cudaEventRecord(stop), "record stop");
        cuda_ok(cudaEventSynchronize(stop), "event sync");
        const auto end = Clock::now();
        float ms = 0.0f;
        cuda_ok(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
        device.push_back(ms);
        wall.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    cuda_ok(cudaEventDestroy(start), "destroy start");
    cuda_ok(cudaEventDestroy(stop), "destroy stop");
    return Timing{mean(device), percentile(device, 0.50), percentile(device, 0.95),
                  mean(wall), percentile(wall, 0.50), percentile(wall, 0.95)};
}

void print_timing(std::ostream& out, const Timing& value) {
    out << "{\"device_mean_ms\":" << value.device_mean_ms
        << ",\"device_p50_ms\":" << value.device_p50_ms
        << ",\"device_p95_ms\":" << value.device_p95_ms
        << ",\"wall_mean_ms\":" << value.wall_mean_ms
        << ",\"wall_p50_ms\":" << value.wall_p50_ms
        << ",\"wall_p95_ms\":" << value.wall_p95_ms << "}";
}

struct CaseResult {
    int qnum = 0;
    Timing tree;
    Timing scan;
    int tree_oracle_failures = 0;
    int scan_oracle_failures = 0;
    double scan_over_tree = 0.0;
};

CaseResult run_case(int nq, int prefix, int dim, int repeats, float radius,
                    short* data_d, int* data_info, TN* node_list, int* id_list,
                    int* max_node_num, int tree_h, int*& empty_list,
                    const std::vector<float>& host_values,
                    int* all_query_ids_d, int*& qresult_count,
                    int*& qresult_prefix, int*& result_id, float*& result_dis,
                    float* dense_output_d, uint32_t* dense_counts_d) {
    for (int q = 0; q < nq; ++q) {
        all_query_ids_d[q] = (q * prefix) / nq;
    }
    cuda_ok(cudaDeviceSynchronize(), "query ID publication");

    auto tree_launch = [&] {
        searchIndexRnnUpdate(data_d, node_list, id_list, max_node_num,
                             all_query_ids_d, nq, radius, tree_h, data_info,
                             empty_list, qresult_count, qresult_prefix,
                             result_id, result_dis, nullptr, nullptr);
    };
    const Timing tree_time = benchmark(tree_launch, 5, repeats);
    tree_launch();
    cuda_ok(cudaDeviceSynchronize(), "tree correctness sync");

    int tree_failures = 0;
    for (int q = 0; q < nq; ++q) {
        std::vector<int> expected;
        for (int id = 0; id < prefix; ++id) {
            if (l2_squared(host_values, id, all_query_ids_d[q], dim) <= radius * radius) {
                expected.push_back(id);
            }
        }
        std::vector<int> observed;
        const int count = qresult_count[q];
        const int begin = qresult_prefix[q];
        observed.assign(result_id + begin, result_id + begin + count);
        std::sort(expected.begin(), expected.end());
        std::sort(observed.begin(), observed.end());
        observed.erase(std::unique(observed.begin(), observed.end()), observed.end());
        tree_failures += expected != observed;
    }

    const size_t total_pairs = static_cast<size_t>(prefix) * nq;
    const int threads = 256;
    const int blocks = static_cast<int>((total_pairs * 32 + threads - 1) / threads);
    auto scan_launch = [&] {
        dense_range_kernel<<<blocks, threads>>>(data_d, all_query_ids_d, radius * radius,
                                                prefix, nq, dim, dense_output_d, nullptr);
    };
    const Timing scan_time = benchmark(scan_launch, 5, repeats);
    cuda_ok(cudaMemset(dense_counts_d, 0, nq * sizeof(uint32_t)), "clear scan counts");
    dense_range_kernel<<<blocks, threads>>>(data_d, all_query_ids_d, radius * radius,
                                            prefix, nq, dim, dense_output_d, dense_counts_d);
    cuda_ok(cudaDeviceSynchronize(), "scan correctness sync");
    std::vector<uint32_t> observed_counts(nq);
    cuda_ok(cudaMemcpy(observed_counts.data(), dense_counts_d, nq * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost), "copy scan counts");
    int scan_failures = 0;
    for (int q = 0; q < nq; ++q) {
        uint32_t expected = 0;
        for (int id = 0; id < prefix; ++id) {
            expected += l2_squared(host_values, id, all_query_ids_d[q], dim) <= radius * radius;
        }
        scan_failures += expected != observed_counts[q];
    }
    return CaseResult{nq, tree_time, scan_time, tree_failures, scan_failures,
                      scan_time.wall_mean_ms / tree_time.wall_mean_ms};
}

void print_case(std::ostream& out, const CaseResult& value) {
    out << "{\"queries\":" << value.qnum
        << ",\"tree_oracle_failures\":" << value.tree_oracle_failures
        << ",\"scan_oracle_failures\":" << value.scan_oracle_failures
        << ",\"native_tree\":";
    print_timing(out, value.tree);
    out << ",\"dense_scan\":";
    print_timing(out, value.scan);
    out << ",\"scan_over_tree_wall_ratio\":" << value.scan_over_tree << "}";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: native_gts_r0_probe SIFT_TXT PREFIX RADIUS REPEATS SUMMARY_JSON\n";
        return 2;
    }
    const char* path = argv[1];
    const int prefix = std::atoi(argv[2]);
    const float radius = std::strtof(argv[3], nullptr);
    const int repeats = std::atoi(argv[4]);
    const char* summary_path = argv[5];
    constexpr int dim = 128;
    constexpr int max_q = 32;
    if (prefix <= MAX_SIZE || radius < 0.0f || repeats <= 0) fail("invalid sizes");
    cuda_ok(cudaSetDevice(0), "select visible GPU");
    cudaDeviceProp property{};
    cuda_ok(cudaGetDeviceProperties(&property, 0), "GPU properties");

    std::vector<float> values;
    if (!load_sift_prefix(path, prefix, dim, values)) fail("load SIFT prefix");
    short* data_d = nullptr;
    int* data_info = nullptr;
    int* id_list = nullptr;
    TN* node_list = nullptr;
    int* max_node_num = nullptr;
    int* empty_list = nullptr;
    int tree_h = 0;
    cuda_ok(cudaMallocManaged(&data_d, static_cast<size_t>(prefix) * dim * sizeof(short)),
            "allocate data");
    cuda_ok(cudaMallocManaged(&data_info, 3 * sizeof(int)), "allocate info");
    cuda_ok(cudaMemcpy(data_d, values.data(), static_cast<size_t>(prefix) * dim * sizeof(short),
                       cudaMemcpyHostToDevice), "copy data");
    data_info[0] = dim;
    data_info[1] = prefix;
    data_info[2] = 2;
    indexConstru(data_d, nullptr, nullptr, data_info, id_list, node_list,
                 max_node_num, tree_h, empty_list);
    cuda_ok(cudaDeviceSynchronize(), "index build sync");

    cuda_ok(cudaMallocManaged(&is_delete, prefix * sizeof(int)), "allocate delete bitmap");
    cuda_ok(cudaMemset(is_delete, 0, prefix * sizeof(int)), "clear delete bitmap");
    int* query_ids_d = nullptr;
    int* qresult_count = nullptr;
    int* qresult_prefix = nullptr;
    int* result_id = nullptr;
    float* result_dis = nullptr;
    cuda_ok(cudaMallocManaged(&query_ids_d, max_q * sizeof(int)), "allocate query IDs");
    cuda_ok(cudaMallocManaged(&qresult_count, max_q * sizeof(int)), "allocate result counts");
    cuda_ok(cudaMallocManaged(&qresult_prefix, max_q * sizeof(int)), "allocate result prefixes");
    float* dense_output_d = nullptr;
    uint32_t* dense_counts_d = nullptr;
    cuda_ok(cudaMalloc(&dense_output_d, static_cast<size_t>(prefix) * max_q * sizeof(float)),
            "allocate dense output");
    cuda_ok(cudaMalloc(&dense_counts_d, max_q * sizeof(uint32_t)), "allocate dense counts");

    std::cout << "NATIVE_GTS_RANGE_BEGIN gpu=" << property.name << " prefix=" << prefix
              << " radius=" << radius << " tree_h=" << tree_h << "\n";
    const CaseResult q1 = run_case(1, prefix, dim, repeats, radius, data_d, data_info,
                                   node_list, id_list, max_node_num, tree_h,
                                   empty_list, values, query_ids_d, qresult_count,
                                   qresult_prefix, result_id, result_dis,
                                   dense_output_d, dense_counts_d);
    const CaseResult q32 = run_case(32, prefix, dim, repeats, radius, data_d, data_info,
                                    node_list, id_list, max_node_num, tree_h,
                                    empty_list, values, query_ids_d, qresult_count,
                                    qresult_prefix, result_id, result_dis,
                                    dense_output_d, dense_counts_d);
    const int oracle_failures = q1.tree_oracle_failures + q1.scan_oracle_failures +
                                q32.tree_oracle_failures + q32.scan_oracle_failures;
    const bool pass = oracle_failures == 0 && q1.scan_over_tree >= 1.25 &&
                      q32.scan_over_tree >= 1.25;

    std::ofstream summary(summary_path);
    if (!summary) fail("create summary");
    summary << std::setprecision(10)
            << "{\"schema\":\"tide-g1-native-range-v2\","
            << "\"host\":\"CONFIGURE_ARCHIVE_HOST\",\"gpu_name\":\"" << property.name << "\","
            << "\"dataset\":\"SIFT1M prefix\",\"dimension\":" << dim
            << ",\"prefix\":" << prefix << ",\"radius\":" << radius << ",\"tree_height\":" << tree_h
            << ",\"q1\":";
    print_case(summary, q1);
    summary << ",\"q32\":";
    print_case(summary, q32);
    summary << ",\"gate1\":{\"required_tree_speedup\":1.25,\"oracle_failures\":"
            << oracle_failures << ",\"status\":\"" << (pass ? "pass" : "fail") << "\"}}\n";
    summary.close();

    std::cout << "Q1 tree_wall_ms=" << q1.tree.wall_mean_ms
              << " scan_wall_ms=" << q1.scan.wall_mean_ms
              << " scan_over_tree=" << q1.scan_over_tree << "\n";
    std::cout << "Q32 tree_wall_ms=" << q32.tree.wall_mean_ms
              << " scan_wall_ms=" << q32.scan.wall_mean_ms
              << " scan_over_tree=" << q32.scan_over_tree << "\n";
    std::cout << "ORACLE failures=" << oracle_failures << "\n";
    std::cout << "TIDE_G1_RESULT=" << (pass ? "PASS" : "FAIL") << "\n";
    return oracle_failures == 0 ? 0 : 3;
}
