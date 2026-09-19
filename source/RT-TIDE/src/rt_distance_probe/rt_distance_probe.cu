#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct Pair { uint32_t geom; uint32_t query; };

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <typename T>
std::vector<T> read_raw(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  size_t bytes = static_cast<size_t>(in.tellg());
  if (bytes % sizeof(T)) throw std::runtime_error("bad raw file size");
  in.seekg(0);
  std::vector<T> out(bytes / sizeof(T));
  in.read(reinterpret_cast<char*>(out.data()), bytes);
  if (!in) throw std::runtime_error("short raw read");
  return out;
}

std::vector<float> read_fvecs_slice(const std::string& path, size_t begin,
                                    size_t count, int expected_dim) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  const std::streamoff record_bytes = static_cast<std::streamoff>((expected_dim + 1) * sizeof(float));
  in.seekg(static_cast<std::streamoff>(begin) * record_bytes);
  std::vector<float> out(count * expected_dim);
  for (size_t i = 0; i < count; ++i) {
    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in || dim != expected_dim) throw std::runtime_error("bad fvec dimension/EOF");
    in.read(reinterpret_cast<char*>(out.data() + i * expected_dim), expected_dim * sizeof(float));
    if (!in) throw std::runtime_error("short fvec read");
  }
  return out;
}

__device__ float warp_sum(float v) {
  for (int offset = 16; offset; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
  return v;
}

__global__ void full_distance_kernel(const float* data, const float* queries,
                                     const float* tau2, float* out,
                                     uint32_t* counts, size_t n, int nq, int dim) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const size_t idx = static_cast<size_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const size_t total = n * static_cast<size_t>(nq);
  if (idx >= total) return;
  const size_t gi = idx % n;
  const int qi = static_cast<int>(idx / n);
  float acc = 0.0f;
  for (int d = lane; d < dim; d += 32) {
    float x = data[gi * dim + d] - queries[static_cast<size_t>(qi) * dim + d];
    acc = fmaf(x, x, acc);
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    out[idx] = acc;
    if (acc <= tau2[qi]) atomicAdd(counts + qi, 1u);
  }
}

__global__ void candidate_distance_kernel(const float* data, const float* queries,
                                          const Pair* pairs, const float* tau2,
                                          float* out, uint32_t* counts,
                                          size_t npairs, int dim) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const size_t idx = static_cast<size_t>(blockIdx.x) * warps_per_block + warp_in_block;
  if (idx >= npairs) return;
  const Pair p = pairs[idx];
  float acc = 0.0f;
  for (int d = lane; d < dim; d += 32) {
    float x = data[static_cast<size_t>(p.geom) * dim + d] -
              queries[static_cast<size_t>(p.query) * dim + d];
    acc = fmaf(x, x, acc);
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    out[idx] = acc;
    if (acc <= tau2[p.query]) atomicAdd(counts + p.query, 1u);
  }
}

template <typename Launch>
std::vector<float> benchmark(Launch launch, int warmup, int repeats) {
  cudaEvent_t a, b;
  check(cudaEventCreate(&a), "event create a");
  check(cudaEventCreate(&b), "event create b");
  for (int i = 0; i < warmup; ++i) launch();
  check(cudaDeviceSynchronize(), "warmup sync");
  std::vector<float> ms;
  for (int i = 0; i < repeats; ++i) {
    check(cudaEventRecord(a), "event record a");
    launch();
    check(cudaEventRecord(b), "event record b");
    check(cudaEventSynchronize(b), "event sync");
    float t = 0;
    check(cudaEventElapsedTime(&t, a, b), "event elapsed");
    ms.push_back(t);
  }
  check(cudaEventDestroy(a), "event destroy a");
  check(cudaEventDestroy(b), "event destroy b");
  return ms;
}

void stats(std::vector<float> v, double& mean, float& minv, float& p95) {
  std::sort(v.begin(), v.end());
  mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  minv = v.front();
  size_t i = static_cast<size_t>(std::ceil(0.95 * v.size())) - 1;
  p95 = v[i];
}

int main(int argc, char** argv) {
  try {
    if (argc != 10) {
      std::cerr << "usage: rt_distance_probe BASE_FVECS QUERY_FVECS PAIRS TAUS "
                   "DELTA_BEGIN N_POINTS N_QUERIES DIM REPEATS\n";
      return 2;
    }
    const std::string base_path = argv[1], query_path = argv[2];
    const std::string pairs_path = argv[3], tau_path = argv[4];
    const size_t delta_begin = std::stoull(argv[5]);
    const size_t n = std::stoull(argv[6]);
    const int nq = std::stoi(argv[7]);
    const int dim = std::stoi(argv[8]);
    const int repeats = std::stoi(argv[9]);
    auto data = read_fvecs_slice(base_path, delta_begin, n, dim);
    auto queries = read_fvecs_slice(query_path, 0, nq, dim);
    auto pairs = read_raw<Pair>(pairs_path);
    auto tau_raw = read_raw<float>(tau_path);
    if (tau_raw.size() != static_cast<size_t>(nq) * 2) throw std::runtime_error("bad tau file");
    std::vector<float> tau2(nq);
    for (int i = 0; i < nq; ++i) tau2[i] = tau_raw[2 * i] * tau_raw[2 * i];
    for (const auto& p : pairs) if (p.geom >= n || p.query >= static_cast<uint32_t>(nq)) throw std::runtime_error("bad pair id");

    float *d_data, *d_queries, *d_tau2, *d_full_out, *d_cand_out;
    Pair* d_pairs;
    uint32_t *d_full_counts, *d_cand_counts;
    check(cudaMalloc(&d_data, data.size() * sizeof(float)), "malloc data");
    check(cudaMalloc(&d_queries, queries.size() * sizeof(float)), "malloc queries");
    check(cudaMalloc(&d_tau2, tau2.size() * sizeof(float)), "malloc tau2");
    check(cudaMalloc(&d_pairs, pairs.size() * sizeof(Pair)), "malloc pairs");
    check(cudaMalloc(&d_full_out, n * static_cast<size_t>(nq) * sizeof(float)), "malloc full out");
    check(cudaMalloc(&d_cand_out, pairs.size() * sizeof(float)), "malloc cand out");
    check(cudaMalloc(&d_full_counts, nq * sizeof(uint32_t)), "malloc full counts");
    check(cudaMalloc(&d_cand_counts, nq * sizeof(uint32_t)), "malloc cand counts");
    check(cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice), "copy data");
    check(cudaMemcpy(d_queries, queries.data(), queries.size() * sizeof(float), cudaMemcpyHostToDevice), "copy queries");
    check(cudaMemcpy(d_tau2, tau2.data(), tau2.size() * sizeof(float), cudaMemcpyHostToDevice), "copy tau2");
    check(cudaMemcpy(d_pairs, pairs.data(), pairs.size() * sizeof(Pair), cudaMemcpyHostToDevice), "copy pairs");

    const int threads = 256;
    const int warps = threads / 32;
    const size_t full_total = n * static_cast<size_t>(nq);
    const int full_blocks = static_cast<int>((full_total + warps - 1) / warps);
    const int cand_blocks = static_cast<int>((pairs.size() + warps - 1) / warps);
    auto full_launch = [&] {
      full_distance_kernel<<<full_blocks, threads>>>(d_data, d_queries, d_tau2,
          d_full_out, d_full_counts, n, nq, dim);
    };
    auto cand_launch = [&] {
      candidate_distance_kernel<<<cand_blocks, threads>>>(d_data, d_queries, d_pairs,
          d_tau2, d_cand_out, d_cand_counts, pairs.size(), dim);
    };
    auto full_ms = benchmark(full_launch, 5, repeats);
    auto cand_ms = benchmark(cand_launch, 5, repeats);
    check(cudaMemset(d_full_counts, 0, nq * sizeof(uint32_t)), "clear full counts");
    check(cudaMemset(d_cand_counts, 0, nq * sizeof(uint32_t)), "clear cand counts");
    full_launch();
    cand_launch();
    check(cudaDeviceSynchronize(), "final sync");
    std::vector<uint32_t> full_counts(nq), cand_counts(nq);
    check(cudaMemcpy(full_counts.data(), d_full_counts, nq * sizeof(uint32_t), cudaMemcpyDeviceToHost), "copy full counts");
    check(cudaMemcpy(cand_counts.data(), d_cand_counts, nq * sizeof(uint32_t), cudaMemcpyDeviceToHost), "copy cand counts");
    bool equal = full_counts == cand_counts;
    uint64_t full_true = std::accumulate(full_counts.begin(), full_counts.end(), uint64_t{0});
    uint64_t cand_true = std::accumulate(cand_counts.begin(), cand_counts.end(), uint64_t{0});
    double fm, cm; float fmin, fp95, cmin, cp95;
    stats(full_ms, fm, fmin, fp95);
    stats(cand_ms, cm, cmin, cp95);
    std::cout << std::setprecision(10)
      << "{\"n_points\":" << n << ",\"n_queries\":" << nq
      << ",\"dimension\":" << dim << ",\"candidate_pairs\":" << pairs.size()
      << ",\"candidate_ratio\":" << static_cast<double>(pairs.size()) / full_total
      << ",\"full_distance_mean_ms\":" << fm
      << ",\"full_distance_min_ms\":" << fmin
      << ",\"full_distance_p95_ms\":" << fp95
      << ",\"candidate_distance_mean_ms\":" << cm
      << ",\"candidate_distance_min_ms\":" << cmin
      << ",\"candidate_distance_p95_ms\":" << cp95
      << ",\"threshold_counts_equal\":" << (equal ? "true" : "false")
      << ",\"full_true_within_tau\":" << full_true
      << ",\"candidate_true_within_tau\":" << cand_true << "}\n";
    return equal ? 0 : 3;
  } catch (const std::exception& e) {
    std::cerr << "rt_distance_probe error: " << e.what() << "\n";
    return 1;
  }
}
