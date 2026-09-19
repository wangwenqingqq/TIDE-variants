#include <cuda_runtime.h>
#include <optix_function_table_definition.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>

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

#include "rtspatial/spatial_index.cuh"
#include "rtspatial/utils/queue.h"
#include "rtspatial/utils/shared_value.h"
#include "rtspatial/utils/stream.h"

namespace {
using point_t = rtspatial::Point<float, 2>;
using envelope_t = rtspatial::Envelope<point_t>;
using pair_t = thrust::pair<uint32_t, uint32_t>;
using device_queue_t = rtspatial::dev::Queue<pair_t>;

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

std::vector<float> read_raw_f32(const std::string& path, size_t expected) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  size_t bytes = static_cast<size_t>(in.tellg());
  if (bytes != expected * sizeof(float)) throw std::runtime_error("bad raw f32 size: " + path);
  in.seekg(0);
  std::vector<float> out(expected);
  in.read(reinterpret_cast<char*>(out.data()), bytes);
  if (!in) throw std::runtime_error("short raw read: " + path);
  return out;
}

std::vector<float> read_fvecs_slice(const std::string& path, size_t begin,
                                    size_t count, int expected_dim) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  const std::streamoff rec = static_cast<std::streamoff>((expected_dim + 1) * sizeof(float));
  in.seekg(static_cast<std::streamoff>(begin) * rec);
  std::vector<float> out(count * static_cast<size_t>(expected_dim));
  for (size_t i = 0; i < count; ++i) {
    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in || dim != expected_dim) throw std::runtime_error("bad fvec dimension/EOF: " + path);
    in.read(reinterpret_cast<char*>(out.data() + i * expected_dim),
            expected_dim * sizeof(float));
    if (!in) throw std::runtime_error("short fvec read: " + path);
  }
  return out;
}

__device__ float warp_sum(float v) {
  for (int off = 16; off; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
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
    const float x = data[gi * dim + d] - queries[static_cast<size_t>(qi) * dim + d];
    acc = fmaf(x, x, acc);
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    out[idx] = acc;
    if (acc <= tau2[qi]) atomicAdd(counts + qi, 1u);
  }
}

__global__ void queue_distance_kernel(const float* data, const float* queries,
                                      device_queue_t queue, const float* tau2,
                                      float* out, uint32_t* counts, size_t n, int nq, int dim) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const size_t first = static_cast<size_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const size_t stride = static_cast<size_t>(gridDim.x) * warps_per_block;
  const size_t npairs = min(static_cast<size_t>(queue.size()), n * static_cast<size_t>(nq));
  for (size_t idx = first; idx < npairs; idx += stride) {
    const pair_t p = queue[static_cast<uint32_t>(idx)];
    if (p.first >= n || p.second >= static_cast<uint32_t>(nq)) continue;
    float acc = 0.0f;
    for (int d = lane; d < dim; d += 32) {
      const float x = data[static_cast<size_t>(p.first) * dim + d] -
                      queries[static_cast<size_t>(p.second) * dim + d];
      acc = fmaf(x, x, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
      out[idx] = acc;
      if (acc <= tau2[p.second]) atomicAdd(counts + p.second, 1u);
    }
  }
}

struct Stats { double mean; float minv; float p95; };

Stats stats(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  Stats s{};
  s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  s.minv = v.front();
  s.p95 = v[static_cast<size_t>(std::ceil(0.95 * v.size())) - 1];
  return s;
}

template <class Launch>
std::vector<float> measure(cudaStream_t stream, int warmup, int repeats, Launch launch) {
  for (int i = 0; i < warmup; ++i) launch();
  check(cudaStreamSynchronize(stream), "warmup sync");
  cudaEvent_t a{}, b{};
  check(cudaEventCreate(&a), "event a");
  check(cudaEventCreate(&b), "event b");
  std::vector<float> times;
  for (int i = 0; i < repeats; ++i) {
    check(cudaEventRecord(a, stream), "record a");
    launch();
    check(cudaEventRecord(b, stream), "record b");
    check(cudaEventSynchronize(b), "sync b");
    float ms = 0;
    check(cudaEventElapsedTime(&ms, a, b), "elapsed");
    times.push_back(ms);
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return times;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 12) {
      std::cerr << "usage: rt_tide_fused_probe PCA_POINTS QUERY_BOXES BASE_FVECS "
                   "QUERY_FVECS TAUS DATA_BEGIN N_POINTS N_QUERIES DIM BATCH_SIZE REPEATS\n";
      return 2;
    }
    const std::string pca_path = argv[1], boxes_path = argv[2];
    const std::string base_path = argv[3], query_path = argv[4], tau_path = argv[5];
    const size_t data_begin = std::stoull(argv[6]);
    const size_t n = std::stoull(argv[7]);
    const int nq = std::stoi(argv[8]);
    const int dim = std::stoi(argv[9]);
    const size_t batch_size = std::stoull(argv[10]);
    const int repeats = std::stoi(argv[11]);
    if (!n || nq < 1 || dim < 1 || !batch_size || repeats < 1) throw std::runtime_error("bad size");

    auto pca = read_raw_f32(pca_path, n * 2);
    auto boxes = read_raw_f32(boxes_path, static_cast<size_t>(nq) * 4);
    auto tau_raw = read_raw_f32(tau_path, static_cast<size_t>(nq) * 2);
    std::vector<float> tau2(nq);
    for (int i = 0; i < nq; ++i) tau2[i] = tau_raw[2 * i] * tau_raw[2 * i];
    auto data = read_fvecs_slice(base_path, data_begin, n, dim);
    auto queries = read_fvecs_slice(query_path, 0, nq, dim);

    constexpr float eps = 1e-4f;
    std::vector<envelope_t> hp;
    hp.reserve(n);
    for (size_t i = 0; i < n; ++i)
      hp.emplace_back(point_t(pca[2*i]-eps, pca[2*i+1]-eps),
                      point_t(pca[2*i]+eps, pca[2*i+1]+eps));
    std::vector<envelope_t> hq;
    hq.reserve(nq);
    for (int i = 0; i < nq; ++i)
      hq.emplace_back(point_t(boxes[4*i], boxes[4*i+1]),
                      point_t(boxes[4*i+2], boxes[4*i+3]));

    thrust::device_vector<envelope_t> dp = hp, dq = hq;
    float *ddata{}, *dqueries{}, *dtau{}, *dfull{}, *dcand{};
    uint32_t *dfull_counts{}, *dcand_counts{};
    const size_t total = n * static_cast<size_t>(nq);
    check(cudaMalloc(&ddata, data.size()*sizeof(float)), "malloc data");
    check(cudaMalloc(&dqueries, queries.size()*sizeof(float)), "malloc queries");
    check(cudaMalloc(&dtau, tau2.size()*sizeof(float)), "malloc tau");
    check(cudaMalloc(&dfull, total*sizeof(float)), "malloc full out");
    check(cudaMalloc(&dcand, total*sizeof(float)), "malloc cand out");
    check(cudaMalloc(&dfull_counts, nq*sizeof(uint32_t)), "malloc full counts");
    check(cudaMalloc(&dcand_counts, nq*sizeof(uint32_t)), "malloc cand counts");
    check(cudaMemcpy(ddata, data.data(), data.size()*sizeof(float), cudaMemcpyHostToDevice), "copy data");
    check(cudaMemcpy(dqueries, queries.data(), queries.size()*sizeof(float), cudaMemcpyHostToDevice), "copy queries");
    check(cudaMemcpy(dtau, tau2.data(), tau2.size()*sizeof(float), cudaMemcpyHostToDevice), "copy tau");

    rtspatial::SpatialIndex<float,2> index;
    rtspatial::Config cfg;
    cfg.ptx_root = PTX_ROOT;
    cfg.max_geometries = n;
    cfg.max_queries = nq;
    cfg.preallocate = true;
    cfg.prefer_fast_build_geom = false;
    cfg.prefer_fast_build_query = false;
    rtspatial::Stream stream;
    index.Init(cfg);
    for (size_t begin=0; begin<n; begin+=batch_size) {
      const size_t count = std::min(batch_size, n-begin);
      index.Insert(rtspatial::ArrayView<envelope_t>(
          thrust::raw_pointer_cast(dp.data())+begin, count), stream.cuda_stream());
    }
    stream.Sync();

    rtspatial::Queue<pair_t> result;
    result.Init(static_cast<uint32_t>(total));
    rtspatial::SharedValue<device_queue_t> d_result;
    const auto qdev = result.DeviceObject();
    d_result.set(stream.cuda_stream(), qdev);
    stream.Sync();

    const int threads = 256;
    const int warps = threads/32;
    const int full_blocks = static_cast<int>((total + warps - 1) / warps);
    const int queue_blocks = 4096;

    auto full_launch = [&] {
      check(cudaMemsetAsync(dfull_counts, 0, nq*sizeof(uint32_t), stream.cuda_stream()), "clear full");
      full_distance_kernel<<<full_blocks,threads,0,stream.cuda_stream()>>>(
          ddata,dqueries,dtau,dfull,dfull_counts,n,nq,dim);
    };
    auto fused_launch = [&] {
      result.Clear(stream.cuda_stream());
      check(cudaMemsetAsync(dcand_counts, 0, nq*sizeof(uint32_t), stream.cuda_stream()), "clear cand");
      index.Query(rtspatial::Predicate::kIntersects,
                  rtspatial::ArrayView<envelope_t>(dq), d_result.data(),
                  stream.cuda_stream());
      // LibRTS reuses its temporary query-GAS buffer on return. An explicit
      // barrier also supplies the candidate-count dependency before reranking.
      stream.Sync();
      queue_distance_kernel<<<queue_blocks,threads,0,stream.cuda_stream()>>>(
          ddata,dqueries,qdev,dtau,dcand,dcand_counts,n,nq,dim);
    };

    auto full_times = measure(stream.cuda_stream(), 3, repeats, full_launch);
    auto fused_times = measure(stream.cuda_stream(), 3, repeats, fused_launch);
    full_launch();
    fused_launch();
    stream.Sync();

    std::vector<uint32_t> hc_full(nq), hc_cand(nq);
    check(cudaMemcpy(hc_full.data(),dfull_counts,nq*sizeof(uint32_t),cudaMemcpyDeviceToHost),"counts full");
    check(cudaMemcpy(hc_cand.data(),dcand_counts,nq*sizeof(uint32_t),cudaMemcpyDeviceToHost),"counts cand");
    const size_t candidates = result.size(stream.cuda_stream());
    const bool counts_equal = hc_full == hc_cand;
    const uint64_t full_true = std::accumulate(hc_full.begin(),hc_full.end(),uint64_t{0});
    const uint64_t cand_true = std::accumulate(hc_cand.begin(),hc_cand.end(),uint64_t{0});
    const auto fs=stats(full_times), rs=stats(fused_times);
    std::cout << std::setprecision(10)
      << "{\"n_points\":"<<n<<",\"n_queries\":"<<nq<<",\"dimension\":"<<dim
      << ",\"batch_size\":"<<batch_size<<",\"repeats\":"<<repeats
      << ",\"candidate_pairs\":"<<candidates
      << ",\"candidate_ratio\":"<<static_cast<double>(candidates)/total
      << ",\"full_scan_mean_ms\":"<<fs.mean<<",\"full_scan_min_ms\":"<<fs.minv
      << ",\"full_scan_p95_ms\":"<<fs.p95
      << ",\"rt_plus_rerank_mean_ms\":"<<rs.mean<<",\"rt_plus_rerank_min_ms\":"<<rs.minv
      << ",\"rt_plus_rerank_p95_ms\":"<<rs.p95
      << ",\"speedup_vs_full\":"<<fs.mean/rs.mean
      << ",\"threshold_counts_equal\":"<<(counts_equal?"true":"false")
      << ",\"full_true_within_tau\":"<<full_true
      << ",\"candidate_true_within_tau\":"<<cand_true<<"}\n";
    return counts_equal ? 0 : 3;
  } catch (const std::exception& e) {
    std::cerr << "rt_tide_fused_probe error: " << e.what() << "\n";
    return 1;
  }
}
