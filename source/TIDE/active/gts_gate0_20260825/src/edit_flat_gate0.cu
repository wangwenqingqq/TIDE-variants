#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                     \
    if (_e != cudaSuccess) {                                                     \
      std::ostringstream _oss;                                                   \
      _oss << #call << " failed at " << __FILE__ << ":" << __LINE__ << ": "   \
           << cudaGetErrorString(_e);                                            \
      throw std::runtime_error(_oss.str());                                      \
    }                                                                            \
  } while (0)

struct Options {
  std::string data_path;
  std::string dataset;
  std::string out_path;
  int max_len_template = 40;
  int max_pivots = 64;
  int calib_queries = 32;
  int measure_queries = 32;
  int warmups = 3;
  int repeats = 9;
  uint64_t seed = 20260825ULL;
};

static Options parse_args(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char *name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (a == "--data") o.data_path = need("--data");
    else if (a == "--dataset") o.dataset = need("--dataset");
    else if (a == "--out") o.out_path = need("--out");
    else if (a == "--max-len") o.max_len_template = std::stoi(need("--max-len"));
    else if (a == "--max-pivots") o.max_pivots = std::stoi(need("--max-pivots"));
    else if (a == "--calib-queries") o.calib_queries = std::stoi(need("--calib-queries"));
    else if (a == "--measure-queries") o.measure_queries = std::stoi(need("--measure-queries"));
    else if (a == "--warmups") o.warmups = std::stoi(need("--warmups"));
    else if (a == "--repeats") o.repeats = std::stoi(need("--repeats"));
    else if (a == "--seed") o.seed = std::stoull(need("--seed"));
    else throw std::runtime_error("unknown argument: " + a);
  }
  if (o.data_path.empty() || o.dataset.empty() || o.out_path.empty()) {
    throw std::runtime_error("required: --data PATH --dataset NAME --out PATH");
  }
  if (o.max_pivots < 8 || o.max_pivots > 64) throw std::runtime_error("max pivots must be in [8,64]");
  if (o.measure_queries < 32) throw std::runtime_error("measure queries must be >= 32");
  return o;
}

struct Dataset {
  int declared_max_len = 0;
  size_t n = 0;
  int metric = -1;
  int row_width = 0;
  std::vector<char> bytes;
  std::vector<uint8_t> lens;
};

static Dataset load_string_dataset(const Options &o) {
  std::ifstream in(o.data_path);
  if (!in) throw std::runtime_error("cannot open " + o.data_path);
  Dataset d;
  size_t declared_n = 0;
  in >> d.declared_max_len >> declared_n >> d.metric;
  std::string line;
  std::getline(in, line);
  if (d.metric != 6) throw std::runtime_error("expected metric type 6 (edit distance)");
  d.row_width = o.max_len_template;
  d.bytes.reserve(declared_n * static_cast<size_t>(d.row_width));
  d.lens.reserve(declared_n);
  size_t line_no = 1;
  while (std::getline(in, line)) {
    ++line_no;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.size() > static_cast<size_t>(d.row_width)) {
      std::ostringstream oss;
      oss << "line " << line_no << " length " << line.size()
          << " exceeds template row width " << d.row_width;
      throw std::runtime_error(oss.str());
    }
    size_t old = d.bytes.size();
    d.bytes.resize(old + d.row_width, 0);
    std::copy(line.begin(), line.end(), d.bytes.begin() + old);
    d.lens.push_back(static_cast<uint8_t>(line.size()));
  }
  d.n = d.lens.size();
  if (d.n != declared_n) {
    std::ostringstream oss;
    oss << "declared n=" << declared_n << " but loaded n=" << d.n;
    throw std::runtime_error(oss.str());
  }
  return d;
}

static uint64_t splitmix64(uint64_t &x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static std::vector<int> make_ids(size_t n, int count, uint64_t seed,
                                 std::unordered_set<int> &used) {
  std::vector<int> out;
  out.reserve(count);
  uint64_t state = seed;
  while (static_cast<int>(out.size()) < count) {
    int id = static_cast<int>(splitmix64(state) % n);
    if (used.insert(id).second) out.push_back(id);
  }
  return out;
}

template <int MAXL>
__device__ __forceinline__ uint8_t edit_distance_device(const char *a, int n,
                                                         const char *b, int m) {
  // Keep the inner dimension short. Both dimensions remain bounded by MAXL.
  if (m > n) {
    const char *tmp_p = a;
    a = b;
    b = tmp_p;
    int tmp_n = n;
    n = m;
    m = tmp_n;
  }
  uint16_t row0[MAXL + 1];
  uint16_t row1[MAXL + 1];
  uint16_t *prev = row0;
  uint16_t *curr = row1;
  for (int j = 0; j <= m; ++j) prev[j] = static_cast<uint16_t>(j);
  for (int i = 1; i <= n; ++i) {
    curr[0] = static_cast<uint16_t>(i);
    const char ca = a[i - 1];
    for (int j = 1; j <= m; ++j) {
      uint16_t del = static_cast<uint16_t>(prev[j] + 1);
      uint16_t ins = static_cast<uint16_t>(curr[j - 1] + 1);
      uint16_t sub = static_cast<uint16_t>(prev[j - 1] + (ca == b[j - 1] ? 0 : 1));
      uint16_t best = del < ins ? del : ins;
      curr[j] = best < sub ? best : sub;
    }
    uint16_t *tmp = prev;
    prev = curr;
    curr = tmp;
  }
  return static_cast<uint8_t>(prev[m]);
}

template <int MAXL>
__global__ void signature_kernel(const char *data, const uint8_t *lens, size_t n,
                                 const int *pivot_ids, int pcount,
                                 uint8_t *signature) {
  size_t obj = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int p = blockIdx.y;
  if (obj >= n || p >= pcount) return;
  int pivot = pivot_ids[p];
  signature[static_cast<size_t>(p) * n + obj] = edit_distance_device<MAXL>(
      data + obj * MAXL, lens[obj], data + static_cast<size_t>(pivot) * MAXL,
      lens[pivot]);
}

template <int MAXL>
__global__ void query_pivot_kernel(const char *data, const uint8_t *lens,
                                   const int *query_ids, int qcount,
                                   const int *pivot_ids, int pcount,
                                   uint8_t *query_pivot) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = qcount * pcount;
  if (idx >= total) return;
  int q = idx / pcount;
  int p = idx - q * pcount;
  int qid = query_ids[q];
  int pid = pivot_ids[p];
  query_pivot[idx] = edit_distance_device<MAXL>(
      data + static_cast<size_t>(qid) * MAXL, lens[qid],
      data + static_cast<size_t>(pid) * MAXL, lens[pid]);
}

template <int MAXL>
__global__ void distance_matrix_kernel(const char *data, const uint8_t *lens,
                                       size_t n, const int *query_ids,
                                       int qcount, uint8_t *out) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = n * static_cast<size_t>(qcount);
  if (idx >= total) return;
  int q = static_cast<int>(idx / n);
  size_t obj = idx - static_cast<size_t>(q) * n;
  int qid = query_ids[q];
  out[idx] = edit_distance_device<MAXL>(
      data + obj * MAXL, lens[obj], data + static_cast<size_t>(qid) * MAXL,
      lens[qid]);
}

template <int MAXL>
__global__ void scan_count_kernel(const char *data, const uint8_t *lens,
                                  size_t n, const int *query_ids, int qcount,
                                  int radius, unsigned int *counts) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = n * static_cast<size_t>(qcount);
  if (idx >= total) return;
  int q = static_cast<int>(idx / n);
  size_t obj = idx - static_cast<size_t>(q) * n;
  int qid = query_ids[q];
  uint8_t d = edit_distance_device<MAXL>(
      data + obj * MAXL, lens[obj], data + static_cast<size_t>(qid) * MAXL,
      lens[qid]);
  if (d <= radius) atomicAdd(counts + q, 1U);
}

__device__ __forceinline__ bool bound_pass(const uint8_t *signature,
                                           const uint8_t *query_pivot,
                                           size_t n, size_t obj, int q, int pcount,
                                           int radius) {
  for (int p = 0; p < pcount; ++p) {
    int a = static_cast<int>(signature[static_cast<size_t>(p) * n + obj]);
    int b = static_cast<int>(query_pivot[q * pcount + p]);
    int diff = a > b ? a - b : b - a;
    if (diff > radius) return false;
  }
  return true;
}

template <int MAXL>
__global__ void filter_verify_kernel(const char *data, const uint8_t *lens,
                                     size_t n, const int *query_ids, int qcount,
                                     const uint8_t *signature,
                                     const uint8_t *query_pivot, int pcount,
                                     int radius, unsigned int *counts) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = n * static_cast<size_t>(qcount);
  if (idx >= total) return;
  int q = static_cast<int>(idx / n);
  size_t obj = idx - static_cast<size_t>(q) * n;
  if (!bound_pass(signature, query_pivot, n, obj, q, pcount, radius)) return;
  int qid = query_ids[q];
  uint8_t d = edit_distance_device<MAXL>(
      data + obj * MAXL, lens[obj], data + static_cast<size_t>(qid) * MAXL,
      lens[qid]);
  if (d <= radius) atomicAdd(counts + q, 1U);
}

__global__ void candidate_flag_kernel(const uint8_t *signature,
                                      const uint8_t *query_pivot, size_t n,
                                      int qcount, int pcount, int radius,
                                      uint8_t *flags) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = n * static_cast<size_t>(qcount);
  if (idx >= total) return;
  int q = static_cast<int>(idx / n);
  size_t obj = idx - static_cast<size_t>(q) * n;
  flags[idx] = bound_pass(signature, query_pivot, n, obj, q, pcount, radius) ? 1 : 0;
}

static int edit_distance_cpu(const char *a, int n, const char *b, int m) {
  if (m > n) {
    std::swap(a, b);
    std::swap(n, m);
  }
  std::vector<int> prev(m + 1), curr(m + 1);
  std::iota(prev.begin(), prev.end(), 0);
  for (int i = 1; i <= n; ++i) {
    curr[0] = i;
    for (int j = 1; j <= m; ++j) {
      curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1,
                          prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
    }
    prev.swap(curr);
  }
  return prev[m];
}

static double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t idx = static_cast<size_t>(std::ceil(p * v.size())) - 1;
  if (idx >= v.size()) idx = v.size() - 1;
  return v[idx];
}

struct Timing {
  std::vector<double> gpu_ms;
  std::vector<double> wall_ms;
  std::vector<unsigned int> counts;
};

struct ResultRow {
  std::string method;
  int pivots = 0;
  int batch = 0;
  int radius = 0;
  Timing timing;
  uint64_t candidate_calls = 0;
  double candidate_ratio = 1.0;
  int count_mismatches = 0;
};

template <typename Launch>
static Timing time_launch(Launch launch, int qcount, int warmups, int repeats,
                          unsigned int *d_counts) {
  for (int i = 0; i < warmups; ++i) {
    CUDA_CHECK(cudaMemset(d_counts, 0, qcount * sizeof(unsigned int)));
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  Timing t;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < repeats; ++i) {
    auto wall_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemsetAsync(d_counts, 0, qcount * sizeof(unsigned int)));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    auto wall_stop = std::chrono::steady_clock::now();
    float gpu = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu, start, stop));
    t.gpu_ms.push_back(gpu);
    t.wall_ms.push_back(std::chrono::duration<double, std::milli>(wall_stop - wall_start).count());
  }
  t.counts.resize(qcount);
  CUDA_CHECK(cudaMemcpy(t.counts.data(), d_counts, qcount * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return t;
}

static void json_string(std::ostream &os, const std::string &s) {
  os << '"';
  for (char c : s) {
    if (c == '"' || c == '\\') os << '\\' << c;
    else if (c == '\n') os << "\\n";
    else os << c;
  }
  os << '"';
}

template <typename T>
static void json_array(std::ostream &os, const std::vector<T> &v) {
  os << '[';
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) os << ',';
    os << v[i];
  }
  os << ']';
}

template <int MAXL>
static int run_probe(const Options &o, const Dataset &d) {
  const int threads = 128;
  std::unordered_set<int> used;
  std::vector<int> pivots = make_ids(d.n, o.max_pivots, o.seed ^ 0x11111111ULL, used);
  std::vector<int> calib_ids = make_ids(d.n, o.calib_queries, o.seed ^ 0x22222222ULL, used);
  std::vector<int> measure_ids = make_ids(d.n, o.measure_queries, o.seed ^ 0x33333333ULL, used);

  char *d_data = nullptr;
  uint8_t *d_lens = nullptr;
  int *d_pivots = nullptr;
  int *d_calib_ids = nullptr;
  int *d_measure_ids = nullptr;
  uint8_t *d_signature = nullptr;
  uint8_t *d_dist_matrix = nullptr;
  uint8_t *d_query_pivot = nullptr;
  uint8_t *d_flags = nullptr;
  unsigned int *d_counts = nullptr;

  const size_t data_bytes = d.bytes.size();
  const size_t sig_bytes = d.n * static_cast<size_t>(o.max_pivots);
  const size_t matrix_bytes = d.n * static_cast<size_t>(std::max(o.calib_queries, o.measure_queries));
  CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
  CUDA_CHECK(cudaMalloc(&d_lens, d.lens.size()));
  CUDA_CHECK(cudaMalloc(&d_pivots, pivots.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_calib_ids, calib_ids.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_measure_ids, measure_ids.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_signature, sig_bytes));
  CUDA_CHECK(cudaMalloc(&d_dist_matrix, matrix_bytes));
  CUDA_CHECK(cudaMalloc(&d_query_pivot,
                        static_cast<size_t>(o.measure_queries) * o.max_pivots));
  CUDA_CHECK(cudaMalloc(&d_flags,
                        static_cast<size_t>(o.measure_queries) * d.n));
  CUDA_CHECK(cudaMalloc(&d_counts, o.measure_queries * sizeof(unsigned int)));

  CUDA_CHECK(cudaMemcpy(d_data, d.bytes.data(), data_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lens, d.lens.data(), d.lens.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pivots, pivots.data(), pivots.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_calib_ids, calib_ids.data(), calib_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_measure_ids, measure_ids.data(), measure_ids.size() * sizeof(int), cudaMemcpyHostToDevice));

  dim3 sig_grid(static_cast<unsigned int>((d.n + threads - 1) / threads),
                static_cast<unsigned int>(o.max_pivots));
  cudaEvent_t bs, be;
  CUDA_CHECK(cudaEventCreate(&bs));
  CUDA_CHECK(cudaEventCreate(&be));
  CUDA_CHECK(cudaEventRecord(bs));
  signature_kernel<MAXL><<<sig_grid, threads>>>(d_data, d_lens, d.n, d_pivots,
                                                o.max_pivots, d_signature);
  CUDA_CHECK(cudaEventRecord(be));
  CUDA_CHECK(cudaEventSynchronize(be));
  CUDA_CHECK(cudaGetLastError());
  float signature_build_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&signature_build_ms, bs, be));
  CUDA_CHECK(cudaEventDestroy(bs));
  CUDA_CHECK(cudaEventDestroy(be));

  // Copy the signature once for checksum and independent CPU spot checks.
  std::vector<uint8_t> h_signature(sig_bytes);
  CUDA_CHECK(cudaMemcpy(h_signature.data(), d_signature, sig_bytes,
                        cudaMemcpyDeviceToHost));
  uint64_t signature_checksum = 1469598103934665603ULL;
  for (uint8_t v : h_signature) {
    signature_checksum ^= v;
    signature_checksum *= 1099511628211ULL;
  }
  int signature_sample_mismatches = 0;
  uint64_t sample_state = o.seed ^ 0x44444444ULL;
  for (int i = 0; i < 512; ++i) {
    int p = static_cast<int>(splitmix64(sample_state) % o.max_pivots);
    size_t obj = splitmix64(sample_state) % d.n;
    int pid = pivots[p];
    int expected = edit_distance_cpu(d.bytes.data() + obj * MAXL, d.lens[obj],
                                     d.bytes.data() + static_cast<size_t>(pid) * MAXL,
                                     d.lens[pid]);
    int got = h_signature[static_cast<size_t>(p) * d.n + obj];
    if (expected != got) ++signature_sample_mismatches;
  }

  // Untimed calibration scan: derive a common central k=10 radius.
  size_t calib_total = d.n * static_cast<size_t>(o.calib_queries);
  distance_matrix_kernel<MAXL><<<static_cast<unsigned int>((calib_total + threads - 1) / threads), threads>>>(
      d_data, d_lens, d.n, d_calib_ids, o.calib_queries, d_dist_matrix);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  std::vector<uint8_t> h_calib(calib_total);
  CUDA_CHECK(cudaMemcpy(h_calib.data(), d_dist_matrix, calib_total,
                        cudaMemcpyDeviceToHost));
  std::vector<int> calib_k10;
  for (int q = 0; q < o.calib_queries; ++q) {
    std::vector<uint8_t> row(h_calib.begin() + static_cast<size_t>(q) * d.n,
                             h_calib.begin() + static_cast<size_t>(q + 1) * d.n);
    std::nth_element(row.begin(), row.begin() + 9, row.end());
    calib_k10.push_back(row[9]);
  }
  std::vector<int> sorted_k10 = calib_k10;
  std::sort(sorted_k10.begin(), sorted_k10.end());
  int center_radius = sorted_k10[sorted_k10.size() / 2];
  std::vector<int> radii;
  radii.push_back(std::max(0, center_radius - 1));
  radii.push_back(center_radius);
  radii.push_back(center_radius + 1);
  std::sort(radii.begin(), radii.end());
  radii.erase(std::unique(radii.begin(), radii.end()), radii.end());

  // Independent measured-query distance matrix is the result-count oracle.
  size_t measure_total = d.n * static_cast<size_t>(o.measure_queries);
  distance_matrix_kernel<MAXL><<<static_cast<unsigned int>((measure_total + threads - 1) / threads), threads>>>(
      d_data, d_lens, d.n, d_measure_ids, o.measure_queries, d_dist_matrix);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  std::vector<uint8_t> h_measure(measure_total);
  CUDA_CHECK(cudaMemcpy(h_measure.data(), d_dist_matrix, measure_total,
                        cudaMemcpyDeviceToHost));

  std::vector<ResultRow> rows;
  int total_count_mismatches = 0;
  const int batches[2] = {1, 32};

  auto oracle_counts = [&](int qcount, int radius) {
    std::vector<unsigned int> out(qcount, 0);
    for (int q = 0; q < qcount; ++q) {
      const uint8_t *row = h_measure.data() + static_cast<size_t>(q) * d.n;
      for (size_t i = 0; i < d.n; ++i) out[q] += (row[i] <= radius);
    }
    return out;
  };

  for (int radius : radii) {
    for (int batch : batches) {
      std::vector<unsigned int> oracle = oracle_counts(batch, radius);
      size_t total = d.n * static_cast<size_t>(batch);
      auto launch_scan = [&]() {
        scan_count_kernel<MAXL><<<static_cast<unsigned int>((total + threads - 1) / threads), threads>>>(
            d_data, d_lens, d.n, d_measure_ids, batch, radius, d_counts);
      };
      ResultRow rr;
      rr.method = "exact_scan";
      rr.batch = batch;
      rr.radius = radius;
      rr.candidate_calls = total;
      rr.candidate_ratio = 1.0;
      rr.timing = time_launch(launch_scan, batch, o.warmups, o.repeats, d_counts);
      for (int q = 0; q < batch; ++q) {
        if (rr.timing.counts[q] != oracle[q]) ++rr.count_mismatches;
      }
      total_count_mismatches += rr.count_mismatches;
      rows.push_back(std::move(rr));
    }
  }

  const int p_options[] = {8, 16, 32, 64};
  for (int pcount : p_options) {
    if (pcount > o.max_pivots) continue;
    for (int radius : radii) {
      for (int batch : batches) {
        std::vector<unsigned int> oracle = oracle_counts(batch, radius);
        int qp_total = batch * pcount;
        size_t total = d.n * static_cast<size_t>(batch);
        auto launch_filter = [&]() {
          query_pivot_kernel<MAXL><<<static_cast<unsigned int>((qp_total + threads - 1) / threads), threads>>>(
              d_data, d_lens, d_measure_ids, batch, d_pivots, pcount,
              d_query_pivot);
          filter_verify_kernel<MAXL><<<static_cast<unsigned int>((total + threads - 1) / threads), threads>>>(
              d_data, d_lens, d.n, d_measure_ids, batch, d_signature,
              d_query_pivot, pcount, radius, d_counts);
        };
        ResultRow rr;
        rr.method = "flat_bound_verify";
        rr.pivots = pcount;
        rr.batch = batch;
        rr.radius = radius;
        rr.timing = time_launch(launch_filter, batch, o.warmups, o.repeats,
                                d_counts);
        for (int q = 0; q < batch; ++q) {
          if (rr.timing.counts[q] != oracle[q]) ++rr.count_mismatches;
        }
        total_count_mismatches += rr.count_mismatches;

        // Instrument candidate count outside the timed path.
        query_pivot_kernel<MAXL><<<static_cast<unsigned int>((qp_total + threads - 1) / threads), threads>>>(
            d_data, d_lens, d_measure_ids, batch, d_pivots, pcount,
            d_query_pivot);
        candidate_flag_kernel<<<static_cast<unsigned int>((total + threads - 1) / threads), threads>>>(
            d_signature, d_query_pivot, d.n, batch, pcount, radius, d_flags);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        std::vector<uint8_t> flags(total);
        CUDA_CHECK(cudaMemcpy(flags.data(), d_flags, total, cudaMemcpyDeviceToHost));
        rr.candidate_calls = std::accumulate(flags.begin(), flags.end(), uint64_t{0});
        rr.candidate_ratio = static_cast<double>(rr.candidate_calls) / total;
        rows.push_back(std::move(rr));
      }
    }
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  size_t free_mem = 0, total_mem = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

  std::ofstream out(o.out_path);
  if (!out) throw std::runtime_error("cannot create " + o.out_path);
  out << std::setprecision(10);
  out << "{\n";
  out << "  \"schema\": \"gts-gate0-edit-flat-v1\",\n";
  out << "  \"dataset\": "; json_string(out, o.dataset); out << ",\n";
  out << "  \"data_path\": "; json_string(out, o.data_path); out << ",\n";
  out << "  \"n\": " << d.n << ",\n";
  out << "  \"declared_max_len\": " << d.declared_max_len << ",\n";
  out << "  \"compiled_max_len\": " << MAXL << ",\n";
  out << "  \"gpu_name\": "; json_string(out, prop.name); out << ",\n";
  out << "  \"seed\": " << o.seed << ",\n";
  out << "  \"warmups\": " << o.warmups << ",\n";
  out << "  \"repeats\": " << o.repeats << ",\n";
  out << "  \"pivots\": "; json_array(out, pivots); out << ",\n";
  out << "  \"calibration_query_ids\": "; json_array(out, calib_ids); out << ",\n";
  out << "  \"measure_query_ids\": "; json_array(out, measure_ids); out << ",\n";
  out << "  \"calibration_k10_radii\": "; json_array(out, calib_k10); out << ",\n";
  out << "  \"center_radius\": " << center_radius << ",\n";
  out << "  \"radii\": "; json_array(out, radii); out << ",\n";
  out << "  \"signature_build_ms\": " << signature_build_ms << ",\n";
  out << "  \"signature_bytes\": " << sig_bytes << ",\n";
  out << "  \"signature_checksum_fnv1a64\": " << signature_checksum << ",\n";
  out << "  \"signature_sample_mismatches\": " << signature_sample_mismatches << ",\n";
  out << "  \"total_result_count_mismatches\": " << total_count_mismatches << ",\n";
  out << "  \"device_total_bytes\": " << total_mem << ",\n";
  out << "  \"device_free_bytes_after_allocations\": " << free_mem << ",\n";
  out << "  \"results\": [\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto &rr = rows[i];
    out << "    {\"method\": "; json_string(out, rr.method);
    out << ", \"pivots\": " << rr.pivots
        << ", \"batch\": " << rr.batch
        << ", \"radius\": " << rr.radius
        << ", \"gpu_ms\": "; json_array(out, rr.timing.gpu_ms);
    out << ", \"wall_ms\": "; json_array(out, rr.timing.wall_ms);
    out << ", \"gpu_median_ms\": " << percentile(rr.timing.gpu_ms, 0.5)
        << ", \"gpu_p95_ms\": " << percentile(rr.timing.gpu_ms, 0.95)
        << ", \"wall_median_ms\": " << percentile(rr.timing.wall_ms, 0.5)
        << ", \"wall_p95_ms\": " << percentile(rr.timing.wall_ms, 0.95)
        << ", \"result_counts\": "; json_array(out, rr.timing.counts);
    out << ", \"candidate_exact_calls\": " << rr.candidate_calls
        << ", \"candidate_ratio\": " << rr.candidate_ratio
        << ", \"result_count_mismatches\": " << rr.count_mismatches << "}";
    if (i + 1 != rows.size()) out << ',';
    out << '\n';
  }
  out << "  ]\n";
  out << "}\n";
  out.close();

  std::cerr << "dataset=" << o.dataset << " n=" << d.n
            << " center_radius=" << center_radius
            << " signature_build_ms=" << signature_build_ms
            << " signature_sample_mismatches=" << signature_sample_mismatches
            << " result_count_mismatches=" << total_count_mismatches << "\n";

  cudaFree(d_counts);
  cudaFree(d_flags);
  cudaFree(d_query_pivot);
  cudaFree(d_dist_matrix);
  cudaFree(d_signature);
  cudaFree(d_measure_ids);
  cudaFree(d_calib_ids);
  cudaFree(d_pivots);
  cudaFree(d_lens);
  cudaFree(d_data);
  return (signature_sample_mismatches == 0 && total_count_mismatches == 0) ? 0 : 2;
}

int main(int argc, char **argv) {
  try {
    Options o = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(0));
    Dataset d = load_string_dataset(o);
    if (o.max_len_template <= 40) return run_probe<40>(o, d);
    if (o.max_len_template <= 112) return run_probe<112>(o, d);
    throw std::runtime_error("supported --max-len values are <=40 or <=112");
  } catch (const std::exception &e) {
    std::cerr << "fatal: " << e.what() << "\n";
    return 1;
  }
}

