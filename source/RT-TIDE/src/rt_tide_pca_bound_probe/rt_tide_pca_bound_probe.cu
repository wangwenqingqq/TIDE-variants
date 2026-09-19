#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_function_table_definition.h>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include "rtspatial/spatial_index.cuh"
#include "rtspatial/utils/queue.h"
#include "rtspatial/utils/shared_value.h"
#include "rtspatial/utils/stream.h"

namespace {

struct PairU32 {
  uint32_t geom;
  uint32_t query;
};

struct CandRec {
  float d;
  uint32_t id;
};

struct Stats {
  double mean = 0.0;
  float min = 0.0f;
  float p95 = 0.0f;
  std::vector<float> values;
};

struct ModeConfig {
  std::string name;
  bool use_rt = false;
  bool use_query_projection = false;
  int dim = 0;
};

ModeConfig parse_mode(const std::string& s) {
  if (s == "direct") return {"direct", false, false, 0};
  if (s == "pca16") return {"pca16", false, true, 16};
  if (s == "pca32") return {"pca32", false, true, 32};
  if (s == "rt2d_pca16") return {"rt2d_pca16", true, true, 16};
  if (s == "rt2d_pca32") return {"rt2d_pca32", true, true, 32};
  throw std::runtime_error("invalid mode: " + s);
}

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

std::vector<float> read_raw_f32(const std::string& path, size_t expected) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  const size_t bytes = static_cast<size_t>(in.tellg());
  if (bytes != expected * sizeof(float)) {
    throw std::runtime_error("bad raw f32 size for " + path + ", expect=" + std::to_string(expected * sizeof(float)) + " got=" + std::to_string(bytes));
  }
  in.seekg(0);
  std::vector<float> out(expected);
  in.read(reinterpret_cast<char*>(out.data()), bytes);
  if (!in) throw std::runtime_error("short read " + path);
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
    if (!in || dim != expected_dim) {
      throw std::runtime_error("bad fvec dim/EOF in " + path);
    }
    in.read(reinterpret_cast<char*>(out.data() + i * expected_dim), expected_dim * sizeof(float));
    if (!in) throw std::runtime_error("short fvec read " + path);
  }
  return out;
}

void summarize(std::vector<float>& times, Stats& out) {
  if (times.empty()) {
    out.mean = 0.0;
    out.min = 0.0f;
    out.p95 = 0.0f;
    out.values.clear();
    return;
  }
  std::sort(times.begin(), times.end());
  out.values = times;
  out.mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  out.min = times.front();
  const size_t p95_idx = std::max<size_t>(1, static_cast<size_t>(std::ceil(0.95 * times.size()))) - 1;
  out.p95 = times[p95_idx];
}

template <typename Launch>
std::vector<float> benchmark(cudaStream_t stream, int warmup, int repeats, Launch launch) {
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  check(cudaStreamSynchronize(stream), "benchmark warmup sync");
  cudaEvent_t t0{}, t1{};
  check(cudaEventCreate(&t0), "event create");
  check(cudaEventCreate(&t1), "event create");
  std::vector<float> times;
  for (int i = 0; i < repeats; ++i) {
    check(cudaEventRecord(t0, stream), "event start");
    launch();
    check(cudaEventRecord(t1, stream), "event end");
    check(cudaEventSynchronize(t1), "event sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, t0, t1), "elapsed");
    times.push_back(ms);
  }
  check(cudaEventDestroy(t0), "ev destroy");
  check(cudaEventDestroy(t1), "ev destroy");
  return times;
}

__global__ void build_pair_table(PairU32* out, size_t total_pairs, uint32_t n_points, uint32_t n_queries) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_pairs) return;
  const uint32_t qid = static_cast<uint32_t>(idx / n_points);
  const uint32_t pid = static_cast<uint32_t>(idx - static_cast<size_t>(qid) * n_points);
  out[idx] = PairU32{pid, qid};
}

__global__ void project_query_kernel(const float* raw_queries,
                                    const float* mean,
                                    const float* basis,
                                    uint32_t n_queries,
                                    uint32_t dim,
                                    float* out_queries) {
  const uint32_t qid = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (qid >= n_queries) return;
  const float* q = raw_queries + static_cast<size_t>(qid) * 128;
  for (uint32_t d = 0; d < dim; ++d) {
    double acc = 0.0;
    for (uint32_t j = 0; j < 128; ++j) {
      acc += static_cast<double>(q[j] - mean[j]) * static_cast<double>(basis[j * dim + d]);
    }
    out_queries[static_cast<size_t>(qid) * dim + d] = static_cast<float>(acc);
  }
}

__global__ void lb_all_true_kernel(size_t total_pairs, uint8_t* flags) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total_pairs) flags[idx] = 1u;
}

__global__ void lb_pca_all_pairs_kernel(const float* point_pca,
                                       const float* query_pca,
                                       uint32_t n_points,
                                       uint32_t n_queries,
                                       uint32_t dim,
                                       const float* tau2,
                                       uint8_t* flags) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<size_t>(n_points) * n_queries) return;
  const uint32_t qid = static_cast<uint32_t>(idx / n_points);
  const uint32_t pid = static_cast<uint32_t>(idx - static_cast<size_t>(qid) * n_points);
  const float* p = point_pca + static_cast<size_t>(pid) * dim;
  const float* q = query_pca + static_cast<size_t>(qid) * dim;
  float acc = 0.0f;
  for (uint32_t d = 0; d < dim; ++d) {
    const float v = p[d] - q[d];
    acc = fmaf(v, v, acc);
  }
  flags[idx] = (acc <= tau2[qid]) ? 1u : 0u;
}

__global__ void lb_pca_from_pairs_kernel(const float* point_pca,
                                        const float* query_pca,
                                        const PairU32* input_pairs,
                                        size_t n_pairs,
                                        uint32_t dim,
                                        const float* tau2,
                                        uint8_t* flags) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_pairs) return;
  const PairU32 p = input_pairs[idx];
  const float* pv = point_pca + static_cast<size_t>(p.geom) * dim;
  const float* qv = query_pca + static_cast<size_t>(p.query) * dim;
  float acc = 0.0f;
  for (uint32_t d = 0; d < dim; ++d) {
    const float v = pv[d] - qv[d];
    acc = fmaf(v, v, acc);
  }
  flags[idx] = (acc <= tau2[p.query]) ? 1u : 0u;
}

__global__ void rerank_128_kernel(const float* data,
                                 const float* queries,
                                 const PairU32* pairs,
                                 size_t n_pairs,
                                 float* distances,
                                 uint8_t* tau_hits,
                                 const float* tau2) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_pairs) return;
  const PairU32 p = pairs[idx];
  const float* x = data + static_cast<size_t>(p.geom) * 128ULL;
  const float* q = queries + static_cast<size_t>(p.query) * 128ULL;
  float acc = 0.0f;
  for (int d = 0; d < 128; ++d) {
    const float v = x[d] - q[d];
    acc = fmaf(v, v, acc);
  }
  distances[idx] = acc;
  tau_hits[idx] = (acc <= tau2[p.query]) ? 1u : 0u;
}

__global__ void full_distance_kernel(const float* data,
                                    const float* queries,
                                    uint32_t n_points,
                                    uint32_t n_queries,
                                    const float* tau2,
                                    float* distances,
                                    uint32_t* tau_counts) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(n_points) * n_queries;
  if (idx >= total) return;
  const uint32_t qid = static_cast<uint32_t>(idx / n_points);
  const uint32_t pid = static_cast<uint32_t>(idx - static_cast<size_t>(qid) * n_points);
  const float* p = data + static_cast<size_t>(pid) * 128ULL;
  const float* q = queries + static_cast<size_t>(qid) * 128ULL;
  float acc = 0.0f;
  for (int d = 0; d < 128; ++d) {
    const float v = p[d] - q[d];
    acc = fmaf(v, v, acc);
  }
  distances[idx] = acc;
  if (acc <= tau2[qid]) {
    atomicAdd(tau_counts + qid, 1u);
  }
}

std::vector<std::vector<CandRec>> topk_from_pairs(const std::vector<PairU32>& pairs,
                                                const std::vector<float>& dists,
                                                const std::vector<float>& tau2,
                                                int n_queries,
                                                int k) {
  struct Better {
    bool operator()(const CandRec& a, const CandRec& b) const {
      if (a.d != b.d) return a.d > b.d;
      return a.id > b.id;
    }
  };
  std::vector<std::priority_queue<CandRec, std::vector<CandRec>, Better>> heaps(n_queries);
  for (size_t i = 0; i < pairs.size(); ++i) {
    const int qi = static_cast<int>(pairs[i].query);
    const float d = dists[i];
    if (qi < 0 || qi >= n_queries) continue;
    if (d > tau2[qi]) continue;
    CandRec c{d, pairs[i].geom};
    auto& h = heaps[qi];
    if (static_cast<int>(h.size()) < k) {
      h.push(c);
    } else if (Better{}(c, h.top())) {
      h.pop();
      h.push(c);
    }
  }
  std::vector<std::vector<CandRec>> out(n_queries);
  for (int qi = 0; qi < n_queries; ++qi) {
    auto& h = heaps[qi];
    out[qi].reserve(h.size());
    while (!h.empty()) {
      out[qi].push_back(h.top());
      h.pop();
    }
    std::reverse(out[qi].begin(), out[qi].end());
  }
  return out;
}

bool topk_equal(const std::vector<std::vector<CandRec>>& a,
                const std::vector<std::vector<CandRec>>& b,
                int k) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].size() != b[i].size()) return false;
    if (static_cast<int>(a[i].size()) > k) return false;
    const int n = static_cast<int>(a[i].size());
    for (int j = 0; j < n; ++j) {
      if (a[i][j].d != b[i][j].d || a[i][j].id != b[i][j].id) return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 9 || argc > 13) {
      std::cerr << "usage: rt_tide_pca_bound_probe MODE BASE_FVECS QUERY_FVECS TAUS CASE_DIR DATA_BEGIN N_POINTS N_QUERIES REPEATS [WARMUP=5] [GAS_BATCH=1000000] [TOPK=10]\n";
      return 2;
    }

    const std::string mode_str = argv[1];
    const std::string base_path = argv[2];
    const std::string query_path = argv[3];
    const std::string tau_path = argv[4];
    const std::string case_dir = argv[5];
    const size_t data_begin = std::stoull(argv[6]);
    const size_t n_points = std::stoull(argv[7]);
    const int n_queries = std::stoi(argv[8]);
    const int repeats = argc >= 10 ? std::stoi(argv[9]) : 5;
    const int warmup = argc >= 11 ? std::stoi(argv[10]) : 5;
    const size_t gas_batch = argc >= 12 ? std::stoull(argv[11]) : 1000000ULL;
    const int top_k = argc >= 13 ? std::stoi(argv[12]) : 10;

    if (n_queries < 1) throw std::runtime_error("n_queries must be positive");
    if (repeats < 1) throw std::runtime_error("repeats must be positive");
    if (top_k < 1) throw std::runtime_error("top_k must be positive");
    if (n_points < 1) throw std::runtime_error("n_points must be positive");

    const auto mode = parse_mode(mode_str);

    auto tau_raw = read_raw_f32(tau_path, static_cast<size_t>(n_queries) * 2);
    std::vector<float> tau2(n_queries);
    for (int i = 0; i < n_queries; ++i) {
      tau2[i] = tau_raw[2 * i + 1] * tau_raw[2 * i + 1];
    }

    const auto data = read_fvecs_slice(base_path, data_begin, n_points, 128);
    const auto queries = read_fvecs_slice(query_path, 0, static_cast<size_t>(n_queries), 128);
    const size_t total_pairs = n_points * static_cast<size_t>(n_queries);

    float *d_data = nullptr;
    float *d_queries = nullptr;
    float *d_tau2 = nullptr;
    check(cudaMalloc(&d_data, data.size() * sizeof(float)), "malloc data");
    check(cudaMalloc(&d_queries, queries.size() * sizeof(float)), "malloc queries");
    check(cudaMalloc(&d_tau2, tau2.size() * sizeof(float)), "malloc tau2");
    check(cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice), "copy data");
    check(cudaMemcpy(d_queries, queries.data(), queries.size() * sizeof(float), cudaMemcpyHostToDevice), "copy queries");
    check(cudaMemcpy(d_tau2, tau2.data(), tau2.size() * sizeof(float), cudaMemcpyHostToDevice), "copy tau2");

    float *d_pca_queries = nullptr;
    float *d_pca_points = nullptr;
    float *d_pca_basis = nullptr;
    float *d_pca_mean = nullptr;
    if (mode.use_query_projection) {
      const std::string points_path = case_dir + "/pca" + std::to_string(mode.dim) + "_points.f32";
      const std::string basis_path = case_dir + "/basis" + std::to_string(mode.dim) + ".f32";
      const std::string mean_path = case_dir + "/mean" + std::to_string(mode.dim) + ".f32";
      auto pca_points = read_raw_f32(points_path, n_points * static_cast<size_t>(mode.dim));
      auto pca_basis = read_raw_f32(basis_path, 128ULL * static_cast<size_t>(mode.dim));
      auto pca_mean = read_raw_f32(mean_path, 128ULL);
      check(cudaMalloc(&d_pca_points, pca_points.size() * sizeof(float)), "malloc pca points");
      check(cudaMalloc(&d_pca_basis, pca_basis.size() * sizeof(float)), "malloc pca basis");
      check(cudaMalloc(&d_pca_mean, pca_mean.size() * sizeof(float)), "malloc pca mean");
      check(cudaMalloc(&d_pca_queries, static_cast<size_t>(n_queries) * mode.dim * sizeof(float)), "malloc pca queries");
      check(cudaMemcpy(d_pca_points, pca_points.data(), pca_points.size() * sizeof(float), cudaMemcpyHostToDevice), "copy pca points");
      check(cudaMemcpy(d_pca_basis, pca_basis.data(), pca_basis.size() * sizeof(float), cudaMemcpyHostToDevice), "copy pca basis");
      check(cudaMemcpy(d_pca_mean, pca_mean.data(), pca_mean.size() * sizeof(float), cudaMemcpyHostToDevice), "copy pca mean");
    }

    PairU32* d_all_pairs = nullptr;
    if (!mode.use_rt) {
      check(cudaMalloc(&d_all_pairs, total_pairs * sizeof(PairU32)), "malloc all pairs");
      const int threads = 256;
      const int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
      build_pair_table<<<blocks, threads>>>(d_all_pairs, total_pairs, static_cast<uint32_t>(n_points), static_cast<uint32_t>(n_queries));
      check(cudaGetLastError(), "build pair table");
      check(cudaDeviceSynchronize(), "sync pair table");
    }

    rtspatial::Stream stream;
    PairU32* d_input_pairs = nullptr;
    size_t input_pairs = total_pairs;

    if (mode.use_rt) {
      auto rt_points = read_raw_f32(case_dir + "/pca2_points.f32", n_points * 2ULL);
      auto rt_boxes = read_raw_f32(case_dir + "/pca2_query_boxes.f32", static_cast<size_t>(n_queries) * 4ULL);
      using point_t = rtspatial::Point<float, 2>;
      using envelope_t = rtspatial::Envelope<point_t>;
      constexpr float eps = 1e-4f;
      std::vector<envelope_t> h_points;
      h_points.reserve(n_points);
      for (size_t i = 0; i < n_points; ++i) {
        const float x = rt_points[2 * i + 0];
        const float y = rt_points[2 * i + 1];
        h_points.emplace_back(point_t{x - eps, y - eps}, point_t{x + eps, y + eps});
      }
      std::vector<envelope_t> h_queries;
      h_queries.reserve(n_queries);
      for (int i = 0; i < n_queries; ++i) {
        h_queries.emplace_back(point_t(rt_boxes[4 * i + 0], rt_boxes[4 * i + 1]),
                              point_t(rt_boxes[4 * i + 2], rt_boxes[4 * i + 3]));
      }
      thrust::device_vector<envelope_t> d_points(h_points);
      thrust::device_vector<envelope_t> d_query_boxes(h_queries);

      rtspatial::SpatialIndex<float, 2> index;
      rtspatial::Config cfg;
      cfg.ptx_root = PTX_ROOT;
      cfg.max_geometries = static_cast<uint32_t>(n_points);
      cfg.max_queries = static_cast<uint32_t>(n_queries);
      cfg.preallocate = true;
      cfg.prefer_fast_build_geom = false;
      cfg.prefer_fast_build_query = false;
      index.Init(cfg);
      for (size_t begin = 0; begin < n_points; begin += gas_batch) {
        const size_t c = std::min(gas_batch, n_points - begin);
        index.Insert(rtspatial::ArrayView<envelope_t>(thrust::raw_pointer_cast(d_points.data()) + begin, c), stream.cuda_stream());
      }
      stream.Sync();

      rtspatial::Queue<PairU32> result;
      result.Init(static_cast<uint32_t>(total_pairs));
      rtspatial::SharedValue<rtspatial::Queue<PairU32>::device_t> d_result;
      auto qdev = result.DeviceObject();
      d_result.set(stream.cuda_stream(), qdev);
      stream.Sync();
      result.Clear(stream.cuda_stream());
      index.Query(rtspatial::Predicate::kIntersects,
                  rtspatial::ArrayView<envelope_t>(thrust::raw_pointer_cast(d_query_boxes.data()), n_queries),
                  d_result.data(), stream.cuda_stream());
      stream.Sync();

      input_pairs = result.size(stream.cuda_stream());
      check(cudaMalloc(&d_input_pairs, input_pairs * sizeof(PairU32)), "malloc rt pairs");
      check(cudaMemcpyAsync(d_input_pairs, result.data(), input_pairs * sizeof(PairU32),
                           cudaMemcpyDeviceToDevice, stream.cuda_stream()), "copy rt pairs");
      stream.Sync();
    } else {
      d_input_pairs = d_all_pairs;
      input_pairs = total_pairs;
    }

    uint8_t* d_flags = nullptr;
    check(cudaMalloc(&d_flags, std::max<size_t>(1, input_pairs) * sizeof(uint8_t)), "malloc flags");

    void* d_select_tmp = nullptr;
    PairU32* d_select_tmp_pairs = nullptr;
    size_t select_tmp = 0;
    uint32_t* d_selected_count = nullptr;
    check(cudaMalloc(&d_selected_count, sizeof(uint32_t)), "malloc selected count");
    {
      size_t tmp = 0;
      cub::DeviceSelect::Flagged(d_select_tmp, tmp,
                                d_input_pairs,
                                d_flags,
                                d_select_tmp_pairs,
                                d_selected_count,
                                input_pairs,
                                stream.cuda_stream());
      select_tmp = tmp;
      if (select_tmp) {
        check(cudaMalloc(&d_select_tmp, select_tmp), "malloc select tmp");
      }
    }

    PairU32* d_selected_pairs = nullptr;
    check(cudaMalloc(&d_selected_pairs, std::max<size_t>(1, input_pairs) * sizeof(PairU32)), "malloc selected pairs");

    float* d_rerank_dist = nullptr;
    uint8_t* d_rerank_hits = nullptr;
    check(cudaMalloc(&d_rerank_dist, std::max<size_t>(1, input_pairs) * sizeof(float)), "malloc rerank dist");
    check(cudaMalloc(&d_rerank_hits, std::max<size_t>(1, input_pairs) * sizeof(uint8_t)), "malloc rerank hits");

    void* d_reduce_tmp = nullptr;
    size_t reduce_tmp = 0;
    uint32_t* d_hit_count = nullptr;
    check(cudaMalloc(&d_hit_count, sizeof(uint32_t)), "malloc hit count");
    cub::DeviceReduce::Sum(d_reduce_tmp, reduce_tmp, d_rerank_hits, d_hit_count, 1, stream.cuda_stream());
    if (reduce_tmp) {
      check(cudaMalloc(&d_reduce_tmp, reduce_tmp), "malloc reduce tmp");
    }

    float* d_full_dist = nullptr;
    uint32_t* d_full_counts = nullptr;
    check(cudaMalloc(&d_full_dist, total_pairs * sizeof(float)), "malloc full dist");
    check(cudaMalloc(&d_full_counts, n_queries * sizeof(uint32_t)), "malloc full counts");

    const int threads = 256;
    const int full_blocks = static_cast<int>((total_pairs + threads - 1) / threads);
    const int proj_blocks = static_cast<int>((std::max<size_t>(1, n_queries) + threads - 1) / threads);

    auto query_projection = [&] {
      if (!mode.use_query_projection) return;
      const int blocks = proj_blocks;
      project_query_kernel<<<blocks, threads, 0, stream.cuda_stream()>>>(
          d_queries, d_pca_mean, d_pca_basis,
          static_cast<uint32_t>(n_queries),
          static_cast<uint32_t>(mode.dim),
          d_pca_queries);
      check(cudaGetLastError(), "query projection");
    };

    auto lb_filter = [&] {
      const int blocks = static_cast<int>((std::max<size_t>(1, input_pairs) + threads - 1) / threads);
      if (!mode.use_query_projection) {
        lb_all_true_kernel<<<blocks, threads, 0, stream.cuda_stream()>>>(
            mode.use_rt ? input_pairs : total_pairs, d_flags);
      } else if (mode.use_rt) {
        lb_pca_from_pairs_kernel<<<blocks, threads, 0, stream.cuda_stream()>>>(
            d_pca_points,
            d_pca_queries,
            d_input_pairs,
            input_pairs,
            static_cast<uint32_t>(mode.dim),
            d_tau2,
            d_flags);
      } else {
        lb_pca_all_pairs_kernel<<<blocks, threads, 0, stream.cuda_stream()>>>(
            d_pca_points,
            d_pca_queries,
            static_cast<uint32_t>(n_points),
            static_cast<uint32_t>(n_queries),
            static_cast<uint32_t>(mode.dim),
            d_tau2,
            d_flags);
      }
      check(cudaGetLastError(), "lower-bound filter");
    };

    auto compaction = [&] {
      check(cudaMemsetAsync(d_selected_count, 0, sizeof(uint32_t), stream.cuda_stream()), "clear selected count");
      cub::DeviceSelect::Flagged(d_select_tmp, select_tmp,
                                d_input_pairs,
                                d_flags,
                                d_selected_pairs,
                                d_selected_count,
                                input_pairs,
                                stream.cuda_stream());
      check(cudaGetLastError(), "DeviceSelect");
    };

    auto rerank = [&] {
      uint32_t m = 0;
      check(cudaMemcpyAsync(&m, d_selected_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream()), "copy selected count");
      check(cudaStreamSynchronize(stream.cuda_stream()), "sync selected count");
      const size_t msize = std::max<uint64_t>(1, m);
      const int blocks = static_cast<int>((msize + threads - 1) / threads);
      rerank_128_kernel<<<blocks, threads, 0, stream.cuda_stream()>>>(
          d_data,
          d_queries,
          d_selected_pairs,
          m,
          d_rerank_dist,
          d_rerank_hits,
          d_tau2);
      check(cudaGetLastError(), "rerank kernel");
      cub::DeviceReduce::Sum(d_reduce_tmp, reduce_tmp, d_rerank_hits, d_hit_count, m, stream.cuda_stream());
      check(cudaGetLastError(), "reduce hit count");
    };

    auto full_scan = [&] {
      check(cudaMemsetAsync(d_full_counts, 0, n_queries * sizeof(uint32_t), stream.cuda_stream()), "clear full counts");
      full_distance_kernel<<<full_blocks, threads, 0, stream.cuda_stream()>>>(
          d_data,
          d_queries,
          static_cast<uint32_t>(n_points),
          static_cast<uint32_t>(n_queries),
          d_tau2,
          d_full_dist,
          d_full_counts);
      check(cudaGetLastError(), "full distance");
    };

    Stats s_full;
    {
      auto t = benchmark(stream.cuda_stream(), std::max(3, warmup), std::max(1, repeats), full_scan);
      summarize(t, s_full);
    }

    Stats s_proj, s_lb, s_compact, s_rerank;
    {
      auto t = benchmark(stream.cuda_stream(), warmup, repeats, query_projection);
      summarize(t, s_proj);
    }
    {
      auto t = benchmark(stream.cuda_stream(), warmup, repeats, lb_filter);
      summarize(t, s_lb);
    }
    {
      auto t = benchmark(stream.cuda_stream(), warmup, repeats, compaction);
      summarize(t, s_compact);
    }
    {
      auto t = benchmark(stream.cuda_stream(), warmup, repeats, rerank);
      summarize(t, s_rerank);
    }

    // Validation run: one full execution path
    if (mode.use_query_projection) {
      query_projection();
    }
    stream.Sync();
    lb_filter();
    stream.Sync();
    compaction();
    stream.Sync();
    rerank();
    stream.Sync();

    uint32_t candidate_pairs = 0;
    uint32_t candidate_hits = 0;
    check(cudaMemcpyAsync(&candidate_pairs, d_selected_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream()), "copy candidate pairs");
    check(cudaMemcpyAsync(&candidate_hits, d_hit_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream()), "copy candidate hits");
    stream.Sync();

    full_scan();
    stream.Sync();

    std::vector<uint32_t> full_counts(n_queries);
    check(cudaMemcpyAsync(full_counts.data(), d_full_counts, n_queries * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream()), "copy full counts");
    stream.Sync();

    uint64_t full_true = 0;
    for (int qi = 0; qi < n_queries; ++qi) full_true += full_counts[qi];

    std::vector<float> h_full_dist(total_pairs);
    check(cudaMemcpyAsync(h_full_dist.data(), d_full_dist, total_pairs * sizeof(float), cudaMemcpyDeviceToHost, stream.cuda_stream()), "copy full dist");
    stream.Sync();

    std::vector<PairU32> h_selected_pairs(candidate_pairs);
    std::vector<float> h_selected_dist(candidate_pairs);
    std::vector<uint8_t> h_selected_hits(candidate_pairs);
    if (candidate_pairs) {
      check(cudaMemcpyAsync(h_selected_pairs.data(), d_selected_pairs, candidate_pairs * sizeof(PairU32), cudaMemcpyDeviceToHost, stream.cuda_stream()), "copy selected pairs");
      check(cudaMemcpyAsync(h_selected_dist.data(), d_rerank_dist, candidate_pairs * sizeof(float), cudaMemcpyDeviceToHost, stream.cuda_stream()), "copy selected dist");
      check(cudaMemcpyAsync(h_selected_hits.data(), d_rerank_hits, candidate_pairs * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream.cuda_stream()), "copy selected hits");
      stream.Sync();
    }

    std::vector<PairU32> full_pairs(total_pairs);
    for (size_t i = 0; i < total_pairs; ++i) {
      const uint32_t pid = static_cast<uint32_t>(i % n_points);
      const uint32_t qi = static_cast<uint32_t>(i / n_points);
      full_pairs[i] = PairU32{pid, qi};
    }
    auto full_topk = topk_from_pairs(full_pairs, h_full_dist, tau2, n_queries, top_k);

    std::vector<PairU32> cand_pairs(candidate_pairs);
    for (size_t i = 0; i < candidate_pairs; ++i) cand_pairs[i] = h_selected_pairs[i];
    auto cand_topk = topk_from_pairs(cand_pairs, h_selected_dist, tau2, n_queries, top_k);

    bool counts_equal = true;
    std::vector<uint32_t> candidate_counts(n_queries, 0);
    for (uint32_t i = 0; i < candidate_pairs; ++i) {
      if (h_selected_hits[i] && h_selected_pairs[i].query < static_cast<uint32_t>(n_queries)) {
        ++candidate_counts[h_selected_pairs[i].query];
      }
    }
    for (int qi = 0; qi < n_queries; ++qi) {
      if (candidate_counts[qi] != full_counts[qi]) {
        counts_equal = false;
      }
    }

    const bool topk_match = topk_equal(full_topk, cand_topk, top_k);

    uint64_t false_neg = 0;
    for (int qi = 0; qi < n_queries; ++qi) {
      if (full_counts[qi] > candidate_counts[qi]) false_neg += (full_counts[qi] - candidate_counts[qi]);
    }

    const double total_ms = s_proj.mean + s_lb.mean + s_compact.mean + s_rerank.mean;

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "{\"mode\":\"" << mode.name << "\",";
    std::cout << "\"n_points\":" << n_points << ",";
    std::cout << "\"n_queries\":" << n_queries << ",";
    std::cout << "\"repeats\":" << repeats << ",";
    std::cout << "\"warmup\":" << warmup << ",";
    std::cout << "\"gas_batch\":" << gas_batch << ",";
    std::cout << "\"top_k\":" << top_k << ",";
    std::cout << "\"full_scan_mean_ms\":" << s_full.mean << ",";
    std::cout << "\"full_scan_min_ms\":" << s_full.min << ",";
    std::cout << "\"full_scan_p95_ms\":" << s_full.p95 << ",";
    std::cout << "\"query_projection_ms\":" << s_proj.mean << ",";
    std::cout << "\"query_projection_min_ms\":" << s_proj.min << ",";
    std::cout << "\"query_projection_p95_ms\":" << s_proj.p95 << ",";
    std::cout << "\"lower_bound_ms\":" << s_lb.mean << ",";
    std::cout << "\"lower_bound_min_ms\":" << s_lb.min << ",";
    std::cout << "\"lower_bound_p95_ms\":" << s_lb.p95 << ",";
    std::cout << "\"candidate_compaction_ms\":" << s_compact.mean << ",";
    std::cout << "\"candidate_compaction_min_ms\":" << s_compact.min << ",";
    std::cout << "\"candidate_compaction_p95_ms\":" << s_compact.p95 << ",";
    std::cout << "\"rerank_ms\":" << s_rerank.mean << ",";
    std::cout << "\"rerank_min_ms\":" << s_rerank.min << ",";
    std::cout << "\"rerank_p95_ms\":" << s_rerank.p95 << ",";
    std::cout << "\"topk_merge_ms\":0.0,\"topk_merge_min_ms\":0.0,\"topk_merge_p95_ms\":0.0,";
    std::cout << "\"total_ms\":" << total_ms << ",";
    std::cout << "\"candidate_pairs\":" << candidate_pairs << ",";
    std::cout << "\"candidate_ratio\":" << (static_cast<double>(candidate_pairs) / static_cast<double>(total_pairs)) << ",";
    std::cout << "\"candidate_true_within_tau\":" << candidate_hits << ",";
    std::cout << "\"full_true_within_tau\":" << full_true << ",";
    std::cout << "\"false_negatives\":" << false_neg << ",";
    std::cout << "\"threshold_counts_equal\":" << (counts_equal ? "true" : "false") << ",";
    std::cout << "\"topk_match\":" << (topk_match ? "true" : "false") << ",";

    std::cout << "\"topk_ids\":" << "[";
    for (int qi = 0; qi < n_queries; ++qi) {
      if (qi) std::cout << ',';
      std::cout << '[';
      const auto& arr = cand_topk[qi];
      for (int i = 0; i < static_cast<int>(arr.size()); ++i) {
        if (i) std::cout << ',';
        std::cout << arr[i].id;
      }
      for (int i = static_cast<int>(arr.size()); i < top_k; ++i) {
        if (i) std::cout << ',';
        std::cout << -1;
      }
      std::cout << ']';
    }
    std::cout << "],\"topk_dists\":" << "[";
    for (int qi = 0; qi < n_queries; ++qi) {
      if (qi) std::cout << ',';
      std::cout << '[';
      const auto& arr = cand_topk[qi];
      for (int i = 0; i < static_cast<int>(arr.size()); ++i) {
        if (i) std::cout << ',';
        std::cout << std::scientific << std::setprecision(8) << arr[i].d;
      }
      for (int i = static_cast<int>(arr.size()); i < top_k; ++i) {
        if (i) std::cout << ',';
        std::cout << -1.0;
      }
      std::cout << ']';
    }
    std::cout << "],\"status\":\"ok\"}\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rt_tide_pca_bound_probe error: " << e.what() << "\n";
    return 1;
  }
}
