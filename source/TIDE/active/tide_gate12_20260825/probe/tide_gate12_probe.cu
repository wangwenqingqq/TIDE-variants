// Development falsification probe for TIDE-G1/TIDE-G2.
//
// The GTS tree and all interval metadata are frozen after construction.
// Genuine arrivals are either attached to a leaf-keyed certified sidecar or
// placed in a global exact fallback.  The probe compares that path with an
// Exact-Delta scan under an independent full-active-set oracle.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// GTS's config defines `short` as float for vector data. Include all standard
// headers first so that macro cannot affect them.
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

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

double mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t pos = static_cast<size_t>(
        std::ceil(q * static_cast<double>(values.size()))) - 1;
    return values[std::min(pos, values.size() - 1)];
}

bool load_sift_prefix(const char* path, int wanted, int expected_dim,
                      std::vector<float>& out) {
    std::ifstream input(path);
    if (!input) {
        std::cerr << "cannot open SIFT input: " << path << "\n";
        return false;
    }
    std::string line;
    if (!std::getline(input, line)) return false;
    std::istringstream header(line);
    int dim = 0, rows = 0, metric = -1;
    header >> dim >> rows >> metric;
    if (dim != expected_dim || metric != 2 || rows < wanted) {
        std::cerr << "unexpected SIFT header dim=" << dim << " rows=" << rows
                  << " metric=" << metric << " wanted=" << wanted << "\n";
        return false;
    }
    out.assign(static_cast<size_t>(wanted) * dim, 0.0f);
    for (int row = 0; row < wanted; ++row) {
        if (!std::getline(input, line)) {
            std::cerr << "truncated SIFT input at row " << row << "\n";
            return false;
        }
        char* p = line.data();
        for (int col = 0; col < dim; ++col) {
            char* end = nullptr;
            const float value = std::strtof(p, &end);
            if (end == p) {
                std::cerr << "parse failure row=" << row << " col=" << col << "\n";
                return false;
            }
            out[static_cast<size_t>(row) * dim + col] = value;
            p = end;
        }
    }
    return true;
}

inline float l2_squared(const std::vector<float>& values, int a, int b, int dim) {
    const float* pa = values.data() + static_cast<size_t>(a) * dim;
    const float* pb = values.data() + static_cast<size_t>(b) * dim;
    float total = 0.0f;
    for (int j = 0; j < dim; ++j) {
        const float d = pa[j] - pb[j];
        total += d * d;
    }
    return total;
}

inline float l2(const std::vector<float>& values, int a, int b, int dim) {
    return std::sqrt(l2_squared(values, a, b, dim));
}

struct RouteResult {
    int leaf = -1;
    bool saw_containment = false;
};

struct TraversalStats {
    int visited_leaves = 0;
    int base_candidates = 0;
};

struct FrozenIndex {
    int dim = 0;
    int max_nodes = 0;
    int id_capacity = 0;
    const std::vector<float>* values = nullptr;
    std::vector<TN> nodes;
    std::vector<int> empty;
    std::vector<float> max_dis;
    std::vector<int> ids;
    std::vector<std::vector<int>> sidecars;
    std::vector<int> fallback;
    std::vector<int> certified;

    float distance(int a, int b) const { return l2(*values, a, b, dim); }

    RouteResult find_leaf_rec(int parent, int incoming) const {
        RouteResult aggregate;
        for (int c = 0; c < TREE_ORDER; ++c) {
            const int child = parent * TREE_ORDER + c + 1;
            if (child >= max_nodes || empty[child] != 0) continue;
            const TN& node = nodes[child];
            const float d = distance(incoming, node.pid);
            // Boundary equality is accepted exactly as in the frozen search
            // predicate; no interval is widened or mutated.
            if (d < node.min_dis || d > max_dis[child]) continue;
            aggregate.saw_containment = true;
            if (node.is_leaf == 1) {
                aggregate.leaf = child;
                return aggregate;
            }
            RouteResult below = find_leaf_rec(child, incoming);
            if (below.leaf >= 0) return below;
            aggregate.saw_containment = aggregate.saw_containment || below.saw_containment;
        }
        return aggregate;
    }

    RouteResult find_leaf(int incoming) const {
        if (max_nodes <= 0 || empty[0] != 0) return RouteResult{};
        if (nodes[0].is_leaf == 1) return RouteResult{0, true};
        return find_leaf_rec(0, incoming);
    }

    void assign_arrival(int incoming) {
        const RouteResult route = find_leaf(incoming);
        if (route.leaf >= 0) {
            sidecars[route.leaf].push_back(incoming);
            certified.push_back(incoming);
        } else {
            fallback.push_back(incoming);
        }
    }

    void collect_rec(int parent, int query, float radius,
                     std::vector<int>& base_candidates,
                     std::vector<int>& visited_leaves,
                     TraversalStats& stats) const {
        if (nodes[parent].is_leaf == 1) {
            visited_leaves.push_back(parent);
            ++stats.visited_leaves;
            const TN& leaf = nodes[parent];
            for (int i = 0; i < leaf.size; ++i) {
                const int id = ids[leaf.lid + i];
                if (id < 0) fail("negative ID in a used frozen leaf slot");
                base_candidates.push_back(id);
                ++stats.base_candidates;
            }
            return;
        }
        for (int c = 0; c < TREE_ORDER; ++c) {
            const int child = parent * TREE_ORDER + c + 1;
            if (child >= max_nodes || empty[child] != 0) continue;
            const TN& node = nodes[child];
            const float d = distance(query, node.pid);
            float lower_bound = 0.0f;
            if (d < node.min_dis) lower_bound = node.min_dis - d;
            else if (d > max_dis[child]) lower_bound = d - max_dis[child];
            if (lower_bound > radius) continue;
            collect_rec(child, query, radius, base_candidates, visited_leaves, stats);
        }
    }

    void collect(int query, float radius, std::vector<int>& base_candidates,
                 std::vector<int>& visited_leaves, TraversalStats& stats) const {
        collect_rec(0, query, radius, base_candidates, visited_leaves, stats);
    }
};

float exact_kth_radius(const std::vector<float>& values, int total, int query,
                       int dim, int k) {
    std::vector<float> distances(total);
    for (int id = 0; id < total; ++id) {
        distances[id] = l2(values, id, query, dim);
    }
    std::nth_element(distances.begin(), distances.begin() + (k - 1), distances.end());
    // The archived host and the probe's warp reduction accumulate float L2 in
    // different orders. Keep the exact kth-neighbor target but move the range
    // boundary outward by a tiny declared guard so a rounding-only boundary
    // difference cannot masquerade as a false negative.
    return std::nextafter(distances[k - 1] * 1.00001f,
                          std::numeric_limits<float>::infinity());
}

std::vector<int> filter_candidates(const std::vector<float>& values,
                                   const std::vector<int>& candidates,
                                   int query, float radius, int dim) {
    std::vector<int> result;
    for (int id : candidates) {
        if (l2(values, id, query, dim) <= radius) result.push_back(id);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<int> brute_range(const std::vector<float>& values, int total,
                             int query, float radius, int dim) {
    std::vector<int> ids(total);
    std::iota(ids.begin(), ids.end(), 0);
    return filter_candidates(values, ids, query, radius, dim);
}

struct PairRef {
    uint32_t query_slot;
    uint32_t point_id;
};

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__global__ void dense_distance_kernel(const float* data, const int* query_ids,
                                      const float* radius2, size_t n, int nq,
                                      int dim, float* output, uint32_t* counts) {
    const size_t global_thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t warp = global_thread >> 5;
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
        if (counts != nullptr && sum <= radius2[qslot]) atomicAdd(counts + qslot, 1U);
    }
}

__global__ void pair_distance_kernel(const float* data, const int* query_ids,
                                     const float* radius2, const PairRef* pairs,
                                     size_t pair_count, int dim, float* output,
                                     uint32_t* counts) {
    const size_t global_thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t warp = global_thread >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= pair_count) return;
    const PairRef pair = pairs[warp];
    const int query = query_ids[pair.query_slot];
    const float* a = data + static_cast<size_t>(pair.point_id) * dim;
    const float* b = data + static_cast<size_t>(query) * dim;
    float sum = 0.0f;
    for (int j = lane; j < dim; j += 32) {
        const float d = a[j] - b[j];
        sum += d * d;
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        output[warp] = sum;
        if (counts != nullptr && sum <= radius2[pair.query_slot]) {
            atomicAdd(counts + pair.query_slot, 1U);
        }
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
    for (int i = 0; i < warmup; ++i) launch();
    cuda_ok(cudaDeviceSynchronize(), "benchmark warmup sync");
    cudaEvent_t start = nullptr, stop = nullptr;
    cuda_ok(cudaEventCreate(&start), "create start event");
    cuda_ok(cudaEventCreate(&stop), "create stop event");
    std::vector<double> device_ms, wall_ms;
    device_ms.reserve(repeats);
    wall_ms.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        const auto wall_begin = Clock::now();
        cuda_ok(cudaEventRecord(start), "record start");
        launch();
        cuda_ok(cudaEventRecord(stop), "record stop");
        cuda_ok(cudaEventSynchronize(stop), "sync stop");
        const auto wall_end = Clock::now();
        float milliseconds = 0.0f;
        cuda_ok(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed event");
        device_ms.push_back(milliseconds);
        wall_ms.push_back(std::chrono::duration<double, std::milli>(wall_end - wall_begin).count());
    }
    cuda_ok(cudaEventDestroy(start), "destroy start event");
    cuda_ok(cudaEventDestroy(stop), "destroy stop event");
    return Timing{mean(device_ms), percentile(device_ms, 0.50), percentile(device_ms, 0.95),
                  mean(wall_ms), percentile(wall_ms, 0.50), percentile(wall_ms, 0.95)};
}

struct GpuProbeResult {
    size_t pairs = 0;
    Timing timing;
    int count_mismatches = 0;
};

GpuProbeResult benchmark_dense(const float* data_d, const int* query_ids_d,
                               const float* radius2_d, size_t n, int nq, int dim,
                               int warmup, int repeats,
                               const std::vector<uint32_t>& expected_counts) {
    const size_t total_pairs = n * static_cast<size_t>(nq);
    float* output_d = nullptr;
    uint32_t* counts_d = nullptr;
    cuda_ok(cudaMalloc(&output_d, total_pairs * sizeof(float)), "allocate dense output");
    cuda_ok(cudaMalloc(&counts_d, nq * sizeof(uint32_t)), "allocate dense counts");
    const int threads = 256;
    const int blocks = static_cast<int>((total_pairs * 32 + threads - 1) / threads);
    auto launch = [&] {
        dense_distance_kernel<<<blocks, threads>>>(data_d, query_ids_d, radius2_d,
                                                   n, nq, dim, output_d, nullptr);
    };
    Timing timing = benchmark(launch, warmup, repeats);
    cuda_ok(cudaMemset(counts_d, 0, nq * sizeof(uint32_t)), "clear dense counts");
    dense_distance_kernel<<<blocks, threads>>>(data_d, query_ids_d, radius2_d,
                                               n, nq, dim, output_d, counts_d);
    cuda_ok(cudaDeviceSynchronize(), "dense correctness sync");
    std::vector<uint32_t> observed(nq);
    cuda_ok(cudaMemcpy(observed.data(), counts_d, nq * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost), "copy dense counts");
    int mismatches = 0;
    for (int i = 0; i < nq; ++i) mismatches += observed[i] != expected_counts[i];
    cuda_ok(cudaFree(output_d), "free dense output");
    cuda_ok(cudaFree(counts_d), "free dense counts");
    return GpuProbeResult{total_pairs, timing, mismatches};
}

GpuProbeResult benchmark_pairs(const float* data_d, const int* query_ids_d,
                               const float* radius2_d,
                               const std::vector<PairRef>& pairs, int nq, int dim,
                               int warmup, int repeats,
                               const std::vector<uint32_t>& expected_counts) {
    PairRef* pairs_d = nullptr;
    float* output_d = nullptr;
    uint32_t* counts_d = nullptr;
    cuda_ok(cudaMalloc(&pairs_d, pairs.size() * sizeof(PairRef)), "allocate pairs");
    cuda_ok(cudaMalloc(&output_d, pairs.size() * sizeof(float)), "allocate pair output");
    cuda_ok(cudaMalloc(&counts_d, nq * sizeof(uint32_t)), "allocate pair counts");
    cuda_ok(cudaMemcpy(pairs_d, pairs.data(), pairs.size() * sizeof(PairRef),
                       cudaMemcpyHostToDevice), "copy pairs");
    const int threads = 256;
    const int blocks = static_cast<int>((pairs.size() * 32 + threads - 1) / threads);
    auto launch = [&] {
        pair_distance_kernel<<<blocks, threads>>>(data_d, query_ids_d, radius2_d,
                                                  pairs_d, pairs.size(), dim,
                                                  output_d, nullptr);
    };
    Timing timing = benchmark(launch, warmup, repeats);
    cuda_ok(cudaMemset(counts_d, 0, nq * sizeof(uint32_t)), "clear pair counts");
    pair_distance_kernel<<<blocks, threads>>>(data_d, query_ids_d, radius2_d,
                                              pairs_d, pairs.size(), dim,
                                              output_d, counts_d);
    cuda_ok(cudaDeviceSynchronize(), "pair correctness sync");
    std::vector<uint32_t> observed(nq);
    cuda_ok(cudaMemcpy(observed.data(), counts_d, nq * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost), "copy pair counts");
    int mismatches = 0;
    for (int i = 0; i < nq; ++i) mismatches += observed[i] != expected_counts[i];
    cuda_ok(cudaFree(pairs_d), "free pairs");
    cuda_ok(cudaFree(output_d), "free pair output");
    cuda_ok(cudaFree(counts_d), "free pair counts");
    return GpuProbeResult{pairs.size(), timing, mismatches};
}

void print_timing_json(std::ostream& out, const Timing& value) {
    out << "{\"device_mean_ms\":" << value.device_mean_ms
        << ",\"device_p50_ms\":" << value.device_p50_ms
        << ",\"device_p95_ms\":" << value.device_p95_ms
        << ",\"wall_mean_ms\":" << value.wall_mean_ms
        << ",\"wall_p50_ms\":" << value.wall_p50_ms
        << ",\"wall_p95_ms\":" << value.wall_p95_ms << "}";
}

void print_probe_json(std::ostream& out, const GpuProbeResult& value) {
    out << "{\"pairs\":" << value.pairs << ",\"count_mismatches\":"
        << value.count_mismatches << ",\"timing\":";
    print_timing_json(out, value.timing);
    out << "}";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: tide_gate12_probe SIFT_TXT PREFIX ARRIVALS QNUM K REPEATS SUMMARY_JSON\n";
        return 2;
    }
    const char* data_path = argv[1];
    const int prefix = std::atoi(argv[2]);
    const int arrivals = std::atoi(argv[3]);
    const int qnum = std::atoi(argv[4]);
    const int k = std::atoi(argv[5]);
    const int repeats = std::atoi(argv[6]);
    const char* summary_path = argv[7];
    constexpr int dim = 128;
    const int total = prefix + arrivals;
    if (prefix <= MAX_SIZE || arrivals <= 0 || qnum <= 0 || qnum > arrivals ||
        k <= 0 || k > total || repeats <= 0) {
        fail("invalid command-line sizes");
    }

    cuda_ok(cudaSetDevice(0), "select visible GPU");
    cudaDeviceProp property{};
    cuda_ok(cudaGetDeviceProperties(&property, 0), "read GPU properties");
    std::cout << "TIDE_GATE12_BEGIN gpu=" << property.name << " prefix=" << prefix
              << " arrivals=" << arrivals << " qnum=" << qnum << " k=" << k << "\n";

    std::vector<float> values;
    const auto load_begin = Clock::now();
    if (!load_sift_prefix(data_path, total, dim, values)) return 2;
    const double load_us = elapsed_us(load_begin, Clock::now());

    short* data_d = nullptr;
    int* data_info = nullptr;
    int* id_list = nullptr;
    TN* node_list = nullptr;
    int* max_node_num = nullptr;
    int* empty_list = nullptr;
    int tree_h = 0;
    cuda_ok(cudaMallocManaged(&data_d, static_cast<size_t>(total) * dim * sizeof(short)),
            "allocate vector store");
    cuda_ok(cudaMallocManaged(&data_info, 3 * sizeof(int)), "allocate data info");
    cuda_ok(cudaMemcpy(data_d, values.data(), static_cast<size_t>(total) * dim * sizeof(short),
                       cudaMemcpyHostToDevice), "upload vector store");
    data_info[0] = dim;
    data_info[1] = prefix;
    data_info[2] = 2;
    cuda_ok(cudaDeviceSynchronize(), "vector upload sync");

    const auto build_begin = Clock::now();
    indexConstru(data_d, nullptr, nullptr, data_info, id_list, node_list,
                 max_node_num, tree_h, empty_list);
    cuda_ok(cudaDeviceSynchronize(), "index build sync");
    const double build_us = elapsed_us(build_begin, Clock::now());

    FrozenIndex index;
    index.dim = dim;
    index.max_nodes = max_node_num[0];
    index.values = &values;
    index.nodes.resize(index.max_nodes);
    index.empty.resize(index.max_nodes);
    index.max_dis.resize(index.max_nodes);
    index.sidecars.resize(index.max_nodes);
    cuda_ok(cudaMemcpy(index.nodes.data(), node_list,
                       index.max_nodes * sizeof(TN), cudaMemcpyDeviceToHost),
            "copy frozen nodes");
    cuda_ok(cudaMemcpy(index.empty.data(), empty_list,
                       index.max_nodes * sizeof(int), cudaMemcpyDeviceToHost),
            "copy empty flags");
    cuda_ok(cudaMemcpy(index.max_dis.data(), max_dis_d,
                       index.max_nodes * sizeof(float), cudaMemcpyDeviceToHost),
            "copy frozen max intervals");
    int leaves = 0;
    int id_capacity = 0;
    for (int node = 0; node < index.max_nodes; ++node) {
        if (index.empty[node] == 0 && index.nodes[node].is_leaf == 1) {
            ++leaves;
            id_capacity = std::max(id_capacity,
                                   index.nodes[node].lid + index.nodes[node].size);
        }
    }
    if (leaves == 0 || id_capacity <= 0) fail("could not mirror frozen leaves");
    index.id_capacity = id_capacity;
    index.ids.resize(id_capacity);
    cuda_ok(cudaMemcpy(index.ids.data(), id_list, id_capacity * sizeof(int),
                       cudaMemcpyDeviceToHost), "copy frozen IDs");

    std::vector<double> assignment_us;
    assignment_us.reserve(arrivals);
    for (int id = prefix; id < total; ++id) {
        const auto begin = Clock::now();
        index.assign_arrival(id);
        assignment_us.push_back(elapsed_us(begin, Clock::now()));
    }
    std::cout << "ASSIGN certified=" << index.certified.size()
              << " fallback=" << index.fallback.size() << "\n";

    std::vector<int> query_ids(qnum);
    for (int q = 0; q < qnum; ++q) {
        query_ids[q] = prefix + (q * arrivals) / qnum;
    }
    std::vector<float> radii(qnum), radius2(qnum);
    std::vector<uint32_t> full_counts(qnum), delta_counts(qnum);
    std::vector<double> exact_cpu_us, sidecar_cpu_us;
    std::vector<double> base_candidate_counts, side_delta_counts, exact_total_counts,
                        side_total_counts, visited_leaf_counts;
    std::vector<PairRef> exact_pairs, side_pairs, exact_delta_pairs, side_delta_pairs;
    int exact_oracle_failures = 0;
    int sidecar_oracle_failures = 0;

    for (int q = 0; q < qnum; ++q) {
        const int query = query_ids[q];
        radii[q] = exact_kth_radius(values, total, query, dim, k);
        radius2[q] = radii[q] * radii[q];
        const std::vector<int> oracle = brute_range(values, total, query, radii[q], dim);
        full_counts[q] = static_cast<uint32_t>(oracle.size());

        std::vector<int> base_candidates;
        std::vector<int> visited_leaves;
        TraversalStats stats;
        index.collect(query, radii[q], base_candidates, visited_leaves, stats);

        std::vector<int> exact_candidates = base_candidates;
        exact_candidates.reserve(base_candidates.size() + arrivals);
        for (int id = prefix; id < total; ++id) exact_candidates.push_back(id);

        std::vector<int> selected_arrivals = index.fallback;
        for (int leaf : visited_leaves) {
            selected_arrivals.insert(selected_arrivals.end(),
                                     index.sidecars[leaf].begin(), index.sidecars[leaf].end());
        }
        std::vector<int> side_candidates = base_candidates;
        side_candidates.insert(side_candidates.end(), selected_arrivals.begin(),
                               selected_arrivals.end());

        const auto exact_begin = Clock::now();
        const std::vector<int> exact_result =
            filter_candidates(values, exact_candidates, query, radii[q], dim);
        exact_cpu_us.push_back(elapsed_us(exact_begin, Clock::now()));
        const auto side_begin = Clock::now();
        const std::vector<int> side_result =
            filter_candidates(values, side_candidates, query, radii[q], dim);
        sidecar_cpu_us.push_back(elapsed_us(side_begin, Clock::now()));
        exact_oracle_failures += exact_result != oracle;
        sidecar_oracle_failures += side_result != oracle;

        std::vector<int> exact_delta_ids;
        exact_delta_ids.reserve(arrivals);
        std::vector<int> side_delta_ids = selected_arrivals;
        for (int id = prefix; id < total; ++id) exact_delta_ids.push_back(id);
        const std::vector<int> delta_result =
            filter_candidates(values, exact_delta_ids, query, radii[q], dim);
        const std::vector<int> side_delta_result =
            filter_candidates(values, side_delta_ids, query, radii[q], dim);
        if (delta_result != side_delta_result) ++sidecar_oracle_failures;
        delta_counts[q] = static_cast<uint32_t>(delta_result.size());

        base_candidate_counts.push_back(stats.base_candidates);
        side_delta_counts.push_back(selected_arrivals.size());
        exact_total_counts.push_back(exact_candidates.size());
        side_total_counts.push_back(side_candidates.size());
        visited_leaf_counts.push_back(stats.visited_leaves);

        for (int id : exact_candidates) exact_pairs.push_back(PairRef{static_cast<uint32_t>(q), static_cast<uint32_t>(id)});
        for (int id : side_candidates) side_pairs.push_back(PairRef{static_cast<uint32_t>(q), static_cast<uint32_t>(id)});
        for (int id : exact_delta_ids) exact_delta_pairs.push_back(PairRef{static_cast<uint32_t>(q), static_cast<uint32_t>(id)});
        for (int id : side_delta_ids) side_delta_pairs.push_back(PairRef{static_cast<uint32_t>(q), static_cast<uint32_t>(id)});
    }

    int* query_ids_d = nullptr;
    float* radius2_d = nullptr;
    cuda_ok(cudaMalloc(&query_ids_d, qnum * sizeof(int)), "allocate query IDs");
    cuda_ok(cudaMalloc(&radius2_d, qnum * sizeof(float)), "allocate radii");
    cuda_ok(cudaMemcpy(query_ids_d, query_ids.data(), qnum * sizeof(int),
                       cudaMemcpyHostToDevice), "copy query IDs");
    cuda_ok(cudaMemcpy(radius2_d, radius2.data(), qnum * sizeof(float),
                       cudaMemcpyHostToDevice), "copy radii");
    constexpr int warmup = 5;
    const GpuProbeResult dense_gpu = benchmark_dense(
        data_d, query_ids_d, radius2_d, total, qnum, dim, warmup, repeats, full_counts);
    const GpuProbeResult exact_gpu = benchmark_pairs(
        data_d, query_ids_d, radius2_d, exact_pairs, qnum, dim, warmup, repeats, full_counts);
    const GpuProbeResult side_gpu = benchmark_pairs(
        data_d, query_ids_d, radius2_d, side_pairs, qnum, dim, warmup, repeats, full_counts);
    const GpuProbeResult exact_delta_gpu = benchmark_pairs(
        data_d, query_ids_d, radius2_d, exact_delta_pairs, qnum, dim, warmup, repeats, delta_counts);
    const GpuProbeResult side_delta_gpu = benchmark_pairs(
        data_d, query_ids_d, radius2_d, side_delta_pairs, qnum, dim, warmup, repeats, delta_counts);

    const double exact_total_mean = mean(exact_total_counts);
    const double side_total_mean = mean(side_total_counts);
    const double side_delta_mean = mean(side_delta_counts);
    const double total_work_reduction = 1.0 - side_total_mean / exact_total_mean;
    const double delta_work_reduction = 1.0 - side_delta_mean / arrivals;
    const double gpu_stage_speedup = exact_gpu.timing.wall_mean_ms /
                                     side_gpu.timing.wall_mean_ms;
    const int gpu_count_mismatches = dense_gpu.count_mismatches + exact_gpu.count_mismatches +
                                     side_gpu.count_mismatches + exact_delta_gpu.count_mismatches +
                                     side_delta_gpu.count_mismatches;
    const bool g2_pass = exact_oracle_failures == 0 && sidecar_oracle_failures == 0 &&
                         gpu_count_mismatches == 0 && total_work_reduction >= 0.25 &&
                         gpu_stage_speedup >= 1.25;

    std::ofstream summary(summary_path);
    if (!summary) fail("cannot create summary JSON");
    summary << std::setprecision(10);
    summary << "{\n"
            << "  \"schema\":\"tide-gate12-development-v1\",\n"
            << "  \"host\":\"CONFIGURE_ARCHIVE_HOST\",\n"
            << "  \"gpu_name\":\"" << property.name << "\",\n"
            << "  \"dataset\":\"SIFT1M deterministic prefix/tail\",\n"
            << "  \"dimension\":" << dim << ",\n"
            << "  \"prefix\":" << prefix << ",\n"
            << "  \"arrivals\":" << arrivals << ",\n"
            << "  \"queries\":" << qnum << ",\n"
            << "  \"k_final_radius\":" << k << ",\n"
            << "  \"tree_height\":" << tree_h << ",\n"
            << "  \"leaves\":" << leaves << ",\n"
            << "  \"load_us\":" << load_us << ",\n"
            << "  \"build_us\":" << build_us << ",\n"
            << "  \"certified\":" << index.certified.size() << ",\n"
            << "  \"fallback\":" << index.fallback.size() << ",\n"
            << "  \"certified_rate\":" << static_cast<double>(index.certified.size()) / arrivals << ",\n"
            << "  \"assignment_us_mean_p50_p95\":[" << mean(assignment_us) << ","
            << percentile(assignment_us, 0.50) << "," << percentile(assignment_us, 0.95) << "],\n"
            << "  \"radius_mean_min_max\":[" << mean(std::vector<double>(radii.begin(), radii.end())) << ","
            << *std::min_element(radii.begin(), radii.end()) << ","
            << *std::max_element(radii.begin(), radii.end()) << "],\n"
            << "  \"oracle\":{\"exact_delta_failures\":" << exact_oracle_failures
            << ",\"sidecar_failures\":" << sidecar_oracle_failures
            << ",\"gpu_count_mismatches\":" << gpu_count_mismatches << "},\n"
            << "  \"work\":{\"base_candidates_mean\":" << mean(base_candidate_counts)
            << ",\"visited_leaves_mean\":" << mean(visited_leaf_counts)
            << ",\"exact_delta_arrival_candidates_mean\":" << arrivals
            << ",\"sidecar_arrival_candidates_mean\":" << side_delta_mean
            << ",\"exact_total_candidates_mean\":" << exact_total_mean
            << ",\"sidecar_total_candidates_mean\":" << side_total_mean
            << ",\"delta_work_reduction\":" << delta_work_reduction
            << ",\"total_work_reduction\":" << total_work_reduction << "},\n"
            << "  \"cpu_filter_us\":{\"exact_mean_p50_p95\":[" << mean(exact_cpu_us) << ","
            << percentile(exact_cpu_us, 0.50) << "," << percentile(exact_cpu_us, 0.95)
            << "],\"sidecar_mean_p50_p95\":[" << mean(sidecar_cpu_us) << ","
            << percentile(sidecar_cpu_us, 0.50) << "," << percentile(sidecar_cpu_us, 0.95) << "]},\n"
            << "  \"gpu\":{\n    \"dense_full_active\":";
    print_probe_json(summary, dense_gpu);
    summary << ",\n    \"tree_plus_exact_delta_optimistic\":";
    print_probe_json(summary, exact_gpu);
    summary << ",\n    \"tree_plus_sidecar_optimistic\":";
    print_probe_json(summary, side_gpu);
    summary << ",\n    \"exact_delta_only\":";
    print_probe_json(summary, exact_delta_gpu);
    summary << ",\n    \"sidecar_delta_only\":";
    print_probe_json(summary, side_delta_gpu);
    summary << "\n  },\n"
            << "  \"gate2\":{\"required_total_work_reduction\":0.25,"
            << "\"required_gpu_stage_speedup\":1.25,"
            << "\"observed_gpu_stage_speedup\":" << gpu_stage_speedup
            << ",\"status\":\"" << (g2_pass ? "pass" : "fail") << "\"}\n"
            << "}\n";
    summary.close();

    std::cout << "ORACLE exact_failures=" << exact_oracle_failures
              << " sidecar_failures=" << sidecar_oracle_failures
              << " gpu_count_mismatches=" << gpu_count_mismatches << "\n";
    std::cout << "WORK base_mean=" << mean(base_candidate_counts)
              << " side_delta_mean=" << side_delta_mean
              << " exact_total_mean=" << exact_total_mean
              << " side_total_mean=" << side_total_mean
              << " total_reduction=" << total_work_reduction << "\n";
    std::cout << "GPU_STAGE exact_wall_ms=" << exact_gpu.timing.wall_mean_ms
              << " side_wall_ms=" << side_gpu.timing.wall_mean_ms
              << " speedup=" << gpu_stage_speedup << "\n";
    std::cout << "TIDE_G2_RESULT=" << (g2_pass ? "PASS" : "FAIL") << "\n";
    return (exact_oracle_failures == 0 && sidecar_oracle_failures == 0 &&
            gpu_count_mismatches == 0) ? 0 : 3;
}
