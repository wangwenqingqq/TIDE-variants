#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                     \
    if (_e != cudaSuccess) {                                                     \
      std::ostringstream _oss;                                                   \
      _oss << #call << " failed at " << __FILE__ << ":" << __LINE__ << ": "     \
           << cudaGetErrorString(_e);                                            \
      throw std::runtime_error(_oss.str());                                      \
    }                                                                            \
  } while (0)

struct Options {
  std::string base_path;
  std::string query_path;
  std::string out_dir;
  int pivots = 64;
  uint64_t seed = 20260827ULL;
};

static Options parse_args(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char *name) -> std::string {
      if (++i >= argc) throw std::runtime_error(std::string("missing ") + name);
      return argv[i];
    };
    if (a == "--base") o.base_path = need("--base");
    else if (a == "--queries") o.query_path = need("--queries");
    else if (a == "--out-dir") o.out_dir = need("--out-dir");
    else if (a == "--pivots") o.pivots = std::stoi(need("--pivots"));
    else if (a == "--seed") o.seed = std::stoull(need("--seed"));
    else throw std::runtime_error("unknown argument: " + a);
  }
  if (o.base_path.empty() || o.query_path.empty() || o.out_dir.empty()) {
    throw std::runtime_error("required: --base --queries --out-dir");
  }
  if (o.pivots != 64) throw std::runtime_error("formal contract requires 64 pivots");
  return o;
}

struct Strings {
  int declared_max_len = 0;
  int metric = -1;
  size_t n = 0;
  std::vector<char> bytes;
  std::vector<uint8_t> lens;
};

template <int MAXL>
static Strings load_strings(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  Strings out;
  size_t declared_n = 0;
  in >> out.declared_max_len >> declared_n >> out.metric;
  if (out.metric != 6) throw std::runtime_error("expected metric 6 in " + path);
  std::string line;
  std::getline(in, line);
  out.bytes.reserve(declared_n * static_cast<size_t>(MAXL));
  out.lens.reserve(declared_n);
  size_t line_no = 1;
  while (std::getline(in, line)) {
    ++line_no;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.size() > MAXL) {
      throw std::runtime_error("line too long in " + path + " at " +
                               std::to_string(line_no));
    }
    size_t old = out.bytes.size();
    out.bytes.resize(old + MAXL, 0);
    std::copy(line.begin(), line.end(), out.bytes.begin() + old);
    out.lens.push_back(static_cast<uint8_t>(line.size()));
  }
  out.n = out.lens.size();
  if (out.n != declared_n) {
    throw std::runtime_error("declared/loaded row mismatch in " + path);
  }
  return out;
}

static uint64_t splitmix64(uint64_t &x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static std::vector<int> make_pivots(size_t n, int count, uint64_t seed) {
  std::vector<int> out;
  std::unordered_set<int> used;
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
  if (m > n) {
    const char *tmp = a;
    a = b;
    b = tmp;
    int ti = n;
    n = m;
    m = ti;
  }
  uint16_t row0[MAXL + 1];
  uint16_t row1[MAXL + 1];
  uint16_t *prev = row0;
  uint16_t *curr = row1;
  for (int j = 0; j <= m; ++j) prev[j] = static_cast<uint16_t>(j);
  for (int i = 1; i <= n; ++i) {
    curr[0] = static_cast<uint16_t>(i);
    char ca = a[i - 1];
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
__global__ void signature_kernel(const char *base, const uint8_t *base_lens,
                                 size_t nbase, const int *pivot_ids, int npivot,
                                 uint8_t *signature_pmajor) {
  size_t obj = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int p = blockIdx.y;
  if (obj >= nbase || p >= npivot) return;
  int pid = pivot_ids[p];
  signature_pmajor[static_cast<size_t>(p) * nbase + obj] =
      edit_distance_device<MAXL>(
          base + obj * MAXL, base_lens[obj],
          base + static_cast<size_t>(pid) * MAXL, base_lens[pid]);
}

template <int MAXL>
__global__ void query_pivot_kernel(
    const char *base, const uint8_t *base_lens, const char *queries,
    const uint8_t *query_lens, size_t nquery, const int *pivot_ids, int npivot,
    uint8_t *query_pivot_qmajor) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = nquery * static_cast<size_t>(npivot);
  if (idx >= total) return;
  size_t q = idx / npivot;
  int p = static_cast<int>(idx - q * npivot);
  int pid = pivot_ids[p];
  query_pivot_qmajor[idx] = edit_distance_device<MAXL>(
      queries + q * MAXL, query_lens[q],
      base + static_cast<size_t>(pid) * MAXL, base_lens[pid]);
}

template <int MAXL>
__global__ void exact_matrix_kernel(
    const char *base, const uint8_t *base_lens, size_t nbase,
    const char *queries, const uint8_t *query_lens, size_t nquery,
    uint8_t *exact_qmajor) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = nbase * nquery;
  if (idx >= total) return;
  size_t q = idx / nbase;
  size_t obj = idx - q * nbase;
  exact_qmajor[idx] = edit_distance_device<MAXL>(
      queries + q * MAXL, query_lens[q], base + obj * MAXL, base_lens[obj]);
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

template <typename T>
static void write_raw(const std::filesystem::path &path,
                      const std::vector<T> &data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot create " + path.string());
  out.write(reinterpret_cast<const char *>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(T)));
  if (!out) throw std::runtime_error("write failed: " + path.string());
}

template <int MAXL>
static int run(const Options &o, const Strings &base, const Strings &queries) {
  namespace fs = std::filesystem;
  fs::path out_dir(o.out_dir);
  fs::create_directories(out_dir);
  fs::path meta_path = out_dir / "export_metadata.json";
  if (fs::exists(meta_path)) throw std::runtime_error("refusing to overwrite " + meta_path.string());

  std::vector<int> pivots = make_pivots(base.n, o.pivots, o.seed);
  const size_t sig_count = base.n * static_cast<size_t>(o.pivots);
  const size_t qp_count = queries.n * static_cast<size_t>(o.pivots);
  const size_t exact_count = base.n * queries.n;

  char *d_base = nullptr, *d_query = nullptr;
  uint8_t *d_base_lens = nullptr, *d_query_lens = nullptr;
  int *d_pivots = nullptr;
  uint8_t *d_sig = nullptr, *d_qp = nullptr, *d_exact = nullptr;
  CUDA_CHECK(cudaMalloc(&d_base, base.bytes.size()));
  CUDA_CHECK(cudaMalloc(&d_query, queries.bytes.size()));
  CUDA_CHECK(cudaMalloc(&d_base_lens, base.lens.size()));
  CUDA_CHECK(cudaMalloc(&d_query_lens, queries.lens.size()));
  CUDA_CHECK(cudaMalloc(&d_pivots, pivots.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_sig, sig_count));
  CUDA_CHECK(cudaMalloc(&d_qp, qp_count));
  CUDA_CHECK(cudaMalloc(&d_exact, exact_count));
  CUDA_CHECK(cudaMemcpy(d_base, base.bytes.data(), base.bytes.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_query, queries.bytes.data(), queries.bytes.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_base_lens, base.lens.data(), base.lens.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_query_lens, queries.lens.data(), queries.lens.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pivots, pivots.data(), pivots.size() * sizeof(int), cudaMemcpyHostToDevice));

  auto timed = [&](auto launch) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a));
    launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return ms;
  };

  const int threads = 128;
  dim3 sig_grid(static_cast<unsigned int>((base.n + threads - 1) / threads),
                static_cast<unsigned int>(o.pivots));
  float sig_ms = timed([&] {
    signature_kernel<MAXL><<<sig_grid, threads>>>(
        d_base, d_base_lens, base.n, d_pivots, o.pivots, d_sig);
  });
  float qp_ms = timed([&] {
    size_t total = qp_count;
    query_pivot_kernel<MAXL><<<static_cast<unsigned int>((total + threads - 1) / threads), threads>>>(
        d_base, d_base_lens, d_query, d_query_lens, queries.n, d_pivots,
        o.pivots, d_qp);
  });
  float exact_ms = timed([&] {
    size_t total = exact_count;
    exact_matrix_kernel<MAXL><<<static_cast<unsigned int>((total + threads - 1) / threads), threads>>>(
        d_base, d_base_lens, base.n, d_query, d_query_lens, queries.n,
        d_exact);
  });

  std::vector<uint8_t> sig(sig_count), qp(qp_count), exact(exact_count);
  CUDA_CHECK(cudaMemcpy(sig.data(), d_sig, sig.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(qp.data(), d_qp, qp.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(exact.data(), d_exact, exact.size(), cudaMemcpyDeviceToHost));

  int sig_mismatch = 0, qp_mismatch = 0, exact_mismatch = 0;
  uint64_t state = o.seed ^ 0x5a5a5a5aULL;
  for (int i = 0; i < 512; ++i) {
    int p = static_cast<int>(splitmix64(state) % o.pivots);
    size_t obj = splitmix64(state) % base.n;
    size_t q = splitmix64(state) % queries.n;
    int pid = pivots[p];
    int s_expected = edit_distance_cpu(
        base.bytes.data() + obj * MAXL, base.lens[obj],
        base.bytes.data() + static_cast<size_t>(pid) * MAXL, base.lens[pid]);
    if (sig[static_cast<size_t>(p) * base.n + obj] != s_expected) ++sig_mismatch;
    int q_expected = edit_distance_cpu(
        queries.bytes.data() + q * MAXL, queries.lens[q],
        base.bytes.data() + static_cast<size_t>(pid) * MAXL, base.lens[pid]);
    if (qp[q * o.pivots + p] != q_expected) ++qp_mismatch;
    int e_expected = edit_distance_cpu(
        queries.bytes.data() + q * MAXL, queries.lens[q],
        base.bytes.data() + obj * MAXL, base.lens[obj]);
    if (exact[q * base.n + obj] != e_expected) ++exact_mismatch;
  }

  write_raw(out_dir / "signature.pmajor.u8", sig);
  write_raw(out_dir / "query_pivot.qmajor.u8", qp);
  write_raw(out_dir / "exact.qmajor.u8", exact);
  write_raw(out_dir / "pivots.i32", pivots);

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::ofstream meta(meta_path);
  meta << std::setprecision(10)
       << "{\n"
       << "  \"schema\": \"certigraph-edit-export-v1\",\n"
       << "  \"base_path\": \"" << o.base_path << "\",\n"
       << "  \"query_path\": \"" << o.query_path << "\",\n"
       << "  \"nbase\": " << base.n << ",\n"
       << "  \"nquery\": " << queries.n << ",\n"
       << "  \"npivot\": " << o.pivots << ",\n"
       << "  \"max_len\": " << MAXL << ",\n"
       << "  \"seed\": " << o.seed << ",\n"
       << "  \"gpu_name\": \"" << prop.name << "\",\n"
       << "  \"signature_build_cuda_ms_diagnostic\": " << sig_ms << ",\n"
       << "  \"query_pivot_cuda_ms_diagnostic\": " << qp_ms << ",\n"
       << "  \"exact_matrix_cuda_ms_diagnostic\": " << exact_ms << ",\n"
       << "  \"signature_spot_mismatches\": " << sig_mismatch << ",\n"
       << "  \"query_pivot_spot_mismatches\": " << qp_mismatch << ",\n"
       << "  \"exact_spot_mismatches\": " << exact_mismatch << "\n"
       << "}\n";
  meta.close();

  cudaFree(d_exact);
  cudaFree(d_qp);
  cudaFree(d_sig);
  cudaFree(d_pivots);
  cudaFree(d_query_lens);
  cudaFree(d_base_lens);
  cudaFree(d_query);
  cudaFree(d_base);
  return (sig_mismatch || qp_mismatch || exact_mismatch) ? 2 : 0;
}

int main(int argc, char **argv) {
  try {
    Options o = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(0));
    Strings base = load_strings<40>(o.base_path);
    Strings queries = load_strings<40>(o.query_path);
    return run<40>(o, base, queries);
  } catch (const std::exception &e) {
    std::cerr << "fatal: " << e.what() << "\n";
    return 1;
  }
}

