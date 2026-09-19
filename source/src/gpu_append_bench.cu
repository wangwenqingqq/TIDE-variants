#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int K = 10;
constexpr int BLOCK = 256;

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t error__ = (expr);                                                \
    if (error__ != cudaSuccess) {                                                \
      std::ostringstream os__;                                                   \
      os__ << #expr << " failed: " << cudaGetErrorString(error__);               \
      throw std::runtime_error(os__.str());                                      \
    }                                                                           \
  } while (0)

struct alignas(16) Candidate {
  std::int64_t id;
  std::uint32_t row;
  std::uint16_t num;
  std::uint16_t den;
};
static_assert(sizeof(Candidate) == 16);

__host__ __device__ inline Candidate sentinel() {
  return Candidate{INT64_MAX, UINT32_MAX, 0, 1};
}

__host__ __device__ inline bool better(const Candidate &a, const Candidate &b) {
  const std::uint32_t lhs = static_cast<std::uint32_t>(a.num) * b.den;
  const std::uint32_t rhs = static_cast<std::uint32_t>(b.num) * a.den;
  if (lhs != rhs) return lhs > rhs;
  if (a.id != b.id) return a.id < b.id;
  return a.row < b.row;
}

struct Top10 {
  Candidate v0, v1, v2, v3, v4, v5, v6, v7, v8, v9;
  __host__ __device__ __forceinline__ void init() {
    v0 = v1 = v2 = v3 = v4 = v5 = v6 = v7 = v8 = v9 = sentinel();
  }
  __host__ __device__ __forceinline__ void insert(const Candidate c) {
    if (!better(c, v9)) return;
    if (!better(c, v8)) { v9 = c; return; }
    v9 = v8;
    if (!better(c, v7)) { v8 = c; return; }
    v8 = v7;
    if (!better(c, v6)) { v7 = c; return; }
    v7 = v6;
    if (!better(c, v5)) { v6 = c; return; }
    v6 = v5;
    if (!better(c, v4)) { v5 = c; return; }
    v5 = v4;
    if (!better(c, v3)) { v4 = c; return; }
    v4 = v3;
    if (!better(c, v2)) { v3 = c; return; }
    v3 = v2;
    if (!better(c, v1)) { v2 = c; return; }
    v2 = v1;
    if (!better(c, v0)) { v1 = c; return; }
    v1 = v0;
    v0 = c;
  }
  __host__ __device__ __forceinline__ void store(Candidate *out) const {
    out[0] = v0; out[1] = v1; out[2] = v2; out[3] = v3; out[4] = v4;
    out[5] = v5; out[6] = v6; out[7] = v7; out[8] = v8; out[9] = v9;
  }
};

__global__ void exact_topk_kernel(const std::uint64_t *__restrict__ targets,
                                  const std::int64_t *__restrict__ target_ids,
                                  const std::uint16_t *__restrict__ target_pc,
                                  std::uint64_t shard_rows,
                                  std::uint64_t global_row_offset,
                                  const std::uint64_t *__restrict__ queries,
                                  const std::int64_t *__restrict__ query_ids,
                                  const std::uint16_t *__restrict__ query_pc,
                                  int query_count,
                                  Candidate *__restrict__ output) {
  const int q = blockIdx.x;
  if (q >= query_count) return;
  const std::uint64_t q0 = queries[static_cast<std::uint64_t>(q) * 4 + 0];
  const std::uint64_t q1 = queries[static_cast<std::uint64_t>(q) * 4 + 1];
  const std::uint64_t q2 = queries[static_cast<std::uint64_t>(q) * 4 + 2];
  const std::uint64_t q3 = queries[static_cast<std::uint64_t>(q) * 4 + 3];
  const std::uint16_t qcount = query_pc[q];
  const std::int64_t qid = query_ids[q];
  Top10 top;
  top.init();
  for (std::uint64_t row = threadIdx.x; row < shard_rows; row += blockDim.x) {
    const std::int64_t id = target_ids[row];
    if (id == qid) continue;
    const std::uint64_t off = row * 4;
    const unsigned inter =
        __popcll(q0 & targets[off + 0]) + __popcll(q1 & targets[off + 1]) +
        __popcll(q2 & targets[off + 2]) + __popcll(q3 & targets[off + 3]);
    unsigned uni = static_cast<unsigned>(qcount) + target_pc[row] - inter;
    if (uni == 0) uni = 1;
    top.insert(Candidate{id, static_cast<std::uint32_t>(global_row_offset + row),
                         static_cast<std::uint16_t>(inter),
                         static_cast<std::uint16_t>(uni)});
  }
  extern __shared__ Candidate shared[];
  top.store(shared + threadIdx.x * K);
  __syncthreads();
  if (threadIdx.x == 0) {
    Top10 merged;
    merged.init();
    for (int t = 0; t < BLOCK; ++t) {
#pragma unroll
      for (int i = 0; i < K; ++i) merged.insert(shared[t * K + i]);
    }
    merged.store(output + q * K);
  }
}

class MappedFile {
 public:
  explicit MappedFile(const std::string &path) : path_(path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("open failed: " + path);
    struct stat st {};
    if (fstat(fd_, &st) != 0) throw std::runtime_error("fstat failed: " + path);
    bytes_ = static_cast<std::size_t>(st.st_size);
    data_ = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
  }
  ~MappedFile() {
    if (data_ != MAP_FAILED) munmap(data_, bytes_);
    if (fd_ >= 0) close(fd_);
  }
  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;
  std::size_t bytes() const { return bytes_; }
  template <class T> const T *as() const {
    if (bytes_ % sizeof(T)) throw std::runtime_error("bad element size: " + path_);
    return static_cast<const T *>(data_);
  }
 private:
  std::string path_;
  int fd_ = -1;
  std::size_t bytes_ = 0;
  void *data_ = MAP_FAILED;
};

std::vector<int> parse_ints(const std::string &s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoi(item));
  if (out.empty()) throw std::runtime_error("empty integer list");
  return out;
}

struct Options {
  std::string root, mode, output;
  std::vector<int> gpus;
  std::vector<int> batches{1, 8, 64, 512};
  int process_index = 0;
  int warmup = 10;
  int iterations = 50;
  int stress_cycles = 1000;
  std::uint64_t limit_rows = 0;
};

Options parse_args(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("missing value for " + a);
      return argv[i];
    };
    if (a == "--root") o.root = need();
    else if (a == "--mode") o.mode = need();
    else if (a == "--output") o.output = need();
    else if (a == "--gpus") o.gpus = parse_ints(need());
    else if (a == "--batches") o.batches = parse_ints(need());
    else if (a == "--process-index") o.process_index = std::stoi(need());
    else if (a == "--warmup") o.warmup = std::stoi(need());
    else if (a == "--iterations") o.iterations = std::stoi(need());
    else if (a == "--stress-cycles") o.stress_cycles = std::stoi(need());
    else if (a == "--limit-rows") o.limit_rows = std::stoull(need());
    else throw std::runtime_error("unknown argument: " + a);
  }
  if (o.root.empty() || o.mode.empty() || o.output.empty() || o.gpus.empty())
    throw std::runtime_error("required: --root --mode --output --gpus");
  const std::vector<std::string> modes{"correctness", "bench", "update-full",
                                       "update-append", "stress"};
  if (std::find(modes.begin(), modes.end(), o.mode) == modes.end())
    throw std::runtime_error("invalid mode");
  return o;
}

struct HostSoA {
  std::vector<std::uint64_t> fp;
  std::vector<std::int64_t> ids;
  std::vector<std::uint16_t> pc;
};

HostSoA load_delta(const MappedFile &file, bool validate) {
  if (file.bytes() % (6 * sizeof(std::uint64_t)))
    throw std::runtime_error("bad delta byte size");
  const auto *rows = file.as<std::uint64_t>();
  const std::size_t n = file.bytes() / (6 * sizeof(std::uint64_t));
  HostSoA out;
  out.fp.resize(n * 4); out.ids.resize(n); out.pc.resize(n);
  std::uint16_t prev_pc = 0;
  for (std::size_t i = 0; i < n; ++i) {
    out.ids[i] = static_cast<std::int64_t>(rows[i * 6]);
    unsigned pc = 0;
    for (int j = 0; j < 4; ++j) {
      out.fp[i * 4 + j] = rows[i * 6 + 1 + j];
      pc += __builtin_popcountll(rows[i * 6 + 1 + j]);
    }
    out.pc[i] = static_cast<std::uint16_t>(rows[i * 6 + 5]);
    if (validate && (pc != out.pc[i] || (i && out.pc[i] < prev_pc)))
      throw std::runtime_error("delta validation failed");
    prev_pc = out.pc[i];
  }
  return out;
}

struct QueryStore {
  std::vector<std::uint64_t> fp;
  std::vector<std::int64_t> ids;
  std::vector<std::uint16_t> pc;
  explicit QueryStore(const MappedFile &file) {
    if (file.bytes() % (6 * sizeof(std::uint64_t)))
      throw std::runtime_error("bad query byte size");
    const auto *rows = file.as<std::uint64_t>();
    const std::size_t n = file.bytes() / (6 * sizeof(std::uint64_t));
    fp.resize(n * 4); ids.resize(n); pc.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      ids[i] = static_cast<std::int64_t>(rows[i * 6]);
      for (int j = 0; j < 4; ++j) fp[i * 4 + j] = rows[i * 6 + 1 + j];
      pc[i] = static_cast<std::uint16_t>(rows[i * 6 + 5]);
    }
  }
  std::size_t size() const { return ids.size(); }
};

struct DeviceContext {
  int gpu = -1;
  std::uint64_t global_offset = 0;
  std::uint64_t base_rows = 0, delta_rows = 0, visible_rows = 0, capacity = 0;
  std::uint64_t *d_fp = nullptr, *d_qfp = nullptr;
  std::int64_t *d_ids = nullptr, *d_qids = nullptr;
  std::uint16_t *d_pc = nullptr, *d_qpc = nullptr;
  Candidate *d_out = nullptr;
  std::uint64_t *h_qfp = nullptr;
  std::int64_t *h_qids = nullptr;
  std::uint16_t *h_qpc = nullptr;
  Candidate *h_out = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
};

void cleanup(DeviceContext &c) {
  if (c.gpu < 0) return;
  cudaSetDevice(c.gpu);
  if (c.d_fp) cudaFree(c.d_fp);
  if (c.d_ids) cudaFree(c.d_ids);
  if (c.d_pc) cudaFree(c.d_pc);
  if (c.d_qfp) cudaFree(c.d_qfp);
  if (c.d_qids) cudaFree(c.d_qids);
  if (c.d_qpc) cudaFree(c.d_qpc);
  if (c.d_out) cudaFree(c.d_out);
  if (c.h_qfp) cudaFreeHost(c.h_qfp);
  if (c.h_qids) cudaFreeHost(c.h_qids);
  if (c.h_qpc) cudaFreeHost(c.h_qpc);
  if (c.h_out) cudaFreeHost(c.h_out);
  if (c.start) cudaEventDestroy(c.start);
  if (c.stop) cudaEventDestroy(c.stop);
  if (c.stream) cudaStreamDestroy(c.stream);
}

void init_context(DeviceContext &c, int gpu, std::uint64_t capacity,
                  std::uint64_t global_offset, int max_batch) {
  c.gpu = gpu; c.capacity = capacity; c.global_offset = global_offset;
  CUDA_CHECK(cudaSetDevice(gpu));
  CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&c.start)); CUDA_CHECK(cudaEventCreate(&c.stop));
  CUDA_CHECK(cudaMalloc(&c.d_fp, capacity * 4 * sizeof(std::uint64_t)));
  CUDA_CHECK(cudaMalloc(&c.d_ids, capacity * sizeof(std::int64_t)));
  CUDA_CHECK(cudaMalloc(&c.d_pc, capacity * sizeof(std::uint16_t)));
  CUDA_CHECK(cudaMalloc(&c.d_qfp, static_cast<std::size_t>(max_batch) * 4 * sizeof(std::uint64_t)));
  CUDA_CHECK(cudaMalloc(&c.d_qids, static_cast<std::size_t>(max_batch) * sizeof(std::int64_t)));
  CUDA_CHECK(cudaMalloc(&c.d_qpc, static_cast<std::size_t>(max_batch) * sizeof(std::uint16_t)));
  CUDA_CHECK(cudaMalloc(&c.d_out, static_cast<std::size_t>(max_batch) * K * sizeof(Candidate)));
  CUDA_CHECK(cudaMallocHost(&c.h_qfp, static_cast<std::size_t>(max_batch) * 4 * sizeof(std::uint64_t)));
  CUDA_CHECK(cudaMallocHost(&c.h_qids, static_cast<std::size_t>(max_batch) * sizeof(std::int64_t)));
  CUDA_CHECK(cudaMallocHost(&c.h_qpc, static_cast<std::size_t>(max_batch) * sizeof(std::uint16_t)));
  CUDA_CHECK(cudaMallocHost(&c.h_out, static_cast<std::size_t>(max_batch) * K * sizeof(Candidate)));
}

std::vector<DeviceContext> load_base(const Options &o, const MappedFile &fp_file,
                                     const MappedFile &id_file, const MappedFile &pc_file,
                                     std::uint64_t delta_total, int max_batch) {
  std::uint64_t rows = id_file.bytes() / sizeof(std::int64_t);
  if (o.limit_rows && rows > o.limit_rows) rows = o.limit_rows;
  if (fp_file.bytes() < rows * 4 * sizeof(std::uint64_t) ||
      pc_file.bytes() < rows * sizeof(std::uint16_t))
    throw std::runtime_error("base file size mismatch");
  const auto *fp = fp_file.as<std::uint64_t>();
  const auto *ids = id_file.as<std::int64_t>();
  const auto *pc = pc_file.as<std::uint16_t>();
  std::vector<DeviceContext> ctx(o.gpus.size());
  for (std::size_t gi = 0; gi < o.gpus.size(); ++gi) {
    const std::uint64_t begin = rows * gi / o.gpus.size();
    const std::uint64_t end = rows * (gi + 1) / o.gpus.size();
    const std::uint64_t dbegin = delta_total * gi / o.gpus.size();
    const std::uint64_t dend = delta_total * (gi + 1) / o.gpus.size();
    auto &c = ctx[gi];
    init_context(c, o.gpus[gi], (end - begin) + (dend - dbegin), begin, max_batch);
    c.base_rows = end - begin; c.delta_rows = dend - dbegin; c.visible_rows = c.base_rows;
    CUDA_CHECK(cudaMemcpyAsync(c.d_fp, fp + begin * 4, c.base_rows * 4 * sizeof(std::uint64_t), cudaMemcpyHostToDevice, c.stream));
    CUDA_CHECK(cudaMemcpyAsync(c.d_ids, ids + begin, c.base_rows * sizeof(std::int64_t), cudaMemcpyHostToDevice, c.stream));
    CUDA_CHECK(cudaMemcpyAsync(c.d_pc, pc + begin, c.base_rows * sizeof(std::uint16_t), cudaMemcpyHostToDevice, c.stream));
  }
  for (auto &c : ctx) { CUDA_CHECK(cudaSetDevice(c.gpu)); CUDA_CHECK(cudaStreamSynchronize(c.stream)); }
  return ctx;
}

void append_delta(std::vector<DeviceContext> &ctx, const HostSoA &delta) {
  const std::uint64_t total = delta.ids.size();
  for (std::size_t gi = 0; gi < ctx.size(); ++gi) {
    auto &c = ctx[gi];
    const std::uint64_t begin = total * gi / ctx.size();
    const std::uint64_t end = total * (gi + 1) / ctx.size();
    if (end - begin != c.delta_rows) throw std::runtime_error("delta shard mismatch");
    CUDA_CHECK(cudaSetDevice(c.gpu));
    CUDA_CHECK(cudaMemcpyAsync(c.d_fp + c.base_rows * 4, delta.fp.data() + begin * 4,
                               c.delta_rows * 4 * sizeof(std::uint64_t), cudaMemcpyHostToDevice, c.stream));
    CUDA_CHECK(cudaMemcpyAsync(c.d_ids + c.base_rows, delta.ids.data() + begin,
                               c.delta_rows * sizeof(std::int64_t), cudaMemcpyHostToDevice, c.stream));
    CUDA_CHECK(cudaMemcpyAsync(c.d_pc + c.base_rows, delta.pc.data() + begin,
                               c.delta_rows * sizeof(std::uint16_t), cudaMemcpyHostToDevice, c.stream));
  }
  for (auto &c : ctx) { CUDA_CHECK(cudaSetDevice(c.gpu)); CUDA_CHECK(cudaStreamSynchronize(c.stream)); }
  for (auto &c : ctx) c.visible_rows = c.base_rows + c.delta_rows;
}

struct BatchResult {
  double host_ms = 0;
  float max_kernel_ms = 0;
  std::vector<Candidate> merged;
};

BatchResult run_batch(std::vector<DeviceContext> &ctx, const QueryStore &qs,
                      const std::vector<int> &qidx) {
  const int batch = static_cast<int>(qidx.size());
  std::vector<std::uint64_t> qfp(static_cast<std::size_t>(batch) * 4);
  std::vector<std::int64_t> qids(batch);
  std::vector<std::uint16_t> qpc(batch);
  for (int i = 0; i < batch; ++i) {
    if (qidx[i] < 0 || static_cast<std::size_t>(qidx[i]) >= qs.size())
      throw std::runtime_error("query index out of range");
    qids[i] = qs.ids[qidx[i]]; qpc[i] = qs.pc[qidx[i]];
    for (int j = 0; j < 4; ++j) qfp[static_cast<std::size_t>(i) * 4 + j] = qs.fp[static_cast<std::size_t>(qidx[i]) * 4 + j];
  }
  for (auto &c : ctx) {
    std::memcpy(c.h_qfp, qfp.data(), qfp.size() * sizeof(std::uint64_t));
    std::memcpy(c.h_qids, qids.data(), qids.size() * sizeof(std::int64_t));
    std::memcpy(c.h_qpc, qpc.data(), qpc.size() * sizeof(std::uint16_t));
  }
  const auto begin = std::chrono::steady_clock::now();
  const std::size_t smem = BLOCK * K * sizeof(Candidate);
  for (auto &c : ctx) {
    CUDA_CHECK(cudaSetDevice(c.gpu));
    CUDA_CHECK(cudaMemcpyAsync(c.d_qfp, c.h_qfp, qfp.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice, c.stream));
    CUDA_CHECK(cudaMemcpyAsync(c.d_qids, c.h_qids, qids.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice, c.stream));
    CUDA_CHECK(cudaMemcpyAsync(c.d_qpc, c.h_qpc, qpc.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice, c.stream));
    CUDA_CHECK(cudaEventRecord(c.start, c.stream));
    exact_topk_kernel<<<batch, BLOCK, smem, c.stream>>>(
        c.d_fp, c.d_ids, c.d_pc, c.visible_rows, c.global_offset,
        c.d_qfp, c.d_qids, c.d_qpc, batch, c.d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(c.stop, c.stream));
    CUDA_CHECK(cudaMemcpyAsync(c.h_out, c.d_out,
                               static_cast<std::size_t>(batch) * K * sizeof(Candidate),
                               cudaMemcpyDeviceToHost, c.stream));
  }
  float max_kernel = 0;
  for (auto &c : ctx) {
    CUDA_CHECK(cudaSetDevice(c.gpu)); CUDA_CHECK(cudaStreamSynchronize(c.stream));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, c.start, c.stop));
    max_kernel = std::max(max_kernel, ms);
  }
  std::vector<Candidate> merged(static_cast<std::size_t>(batch) * K);
  for (int q = 0; q < batch; ++q) {
    Top10 top; top.init();
    for (const auto &c : ctx)
      for (int k = 0; k < K; ++k) top.insert(c.h_out[static_cast<std::size_t>(q) * K + k]);
    top.store(merged.data() + static_cast<std::size_t>(q) * K);
  }
  const double host_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
  return BatchResult{host_ms, max_kernel, std::move(merged)};
}

std::uint64_t hash_candidates(const std::vector<Candidate> &v) {
  std::uint64_t h = 1469598103934665603ull;
  const auto *p = reinterpret_cast<const unsigned char *>(v.data());
  for (std::size_t i = 0; i < v.size() * sizeof(Candidate); ++i) {
    h ^= p[i]; h *= 1099511628211ull;
  }
  return h;
}

std::vector<int> correctness_indices() {
  std::vector<int> q;
  for (int i = 0; i < 64; ++i) q.push_back(i);
  for (int i = 512; i < 768; ++i) q.push_back(i);
  return q;
}

std::vector<int> rotating_indices(int batch, int iter) {
  std::vector<int> q(batch);
  const int start = (iter * 17) % 256;
  for (int i = 0; i < batch; ++i) q[i] = 512 + ((start + i) % 256);
  return q;
}

void write_correctness(const Options &o, std::vector<DeviceContext> &ctx,
                       const QueryStore &qs) {
  const auto qidx = correctness_indices();
  const BatchResult r = run_batch(ctx, qs, qidx);
  std::ofstream f(o.output);
  f << "query_index,query_id,rank,result_id,row,num,den,similarity\n" << std::setprecision(12);
  for (std::size_t q = 0; q < qidx.size(); ++q) {
    for (int k = 0; k < K; ++k) {
      const auto &c = r.merged[q * K + k];
      f << qidx[q] << ',' << qs.ids[qidx[q]] << ',' << k << ',' << c.id << ','
        << c.row << ',' << c.num << ',' << c.den << ','
        << static_cast<double>(c.num) / c.den << '\n';
    }
  }
  std::cout << "queries=" << qidx.size() << "\nhost_ms=" << r.host_ms
            << "\nmax_kernel_ms=" << r.max_kernel_ms << "\nhash="
            << hash_candidates(r.merged) << "\n";
}

void write_bench(const Options &o, std::vector<DeviceContext> &ctx,
                 const QueryStore &qs) {
  std::ofstream f(o.output);
  f << "process_index,batch,iteration,host_ms,max_kernel_ms,qps,result_hash\n";
  f << std::setprecision(12);
  for (int batch : o.batches) {
    for (int i = -o.warmup; i < o.iterations; ++i) {
      const int logical = i < 0 ? i + o.warmup : i + o.warmup;
      const BatchResult r = run_batch(ctx, qs, rotating_indices(batch, logical));
      if (i >= 0)
        f << o.process_index << ',' << batch << ',' << i << ',' << r.host_ms << ','
          << r.max_kernel_ms << ',' << (1000.0 * batch / r.host_ms) << ','
          << hash_candidates(r.merged) << '\n';
    }
  }
}

void write_stress(const Options &o, std::vector<DeviceContext> &ctx,
                  const HostSoA &delta, const QueryStore &qs) {
  std::vector<std::uint64_t> expected(256, 0);
  std::size_t mismatches = 0;
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < o.stress_cycles; ++i) {
    for (auto &c : ctx) c.visible_rows = c.base_rows;
    append_delta(ctx, delta);
    const int slot = i % 256;
    BatchResult r = run_batch(ctx, qs, std::vector<int>{512 + slot});
    const std::uint64_t h = hash_candidates(r.merged);
    if (expected[slot] == 0) expected[slot] = h;
    else if (expected[slot] != h) ++mismatches;
  }
  const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  std::ofstream f(o.output);
  f << "{\n  \"cycles\": " << o.stress_cycles << ",\n  \"hash_mismatches\": "
    << mismatches << ",\n  \"elapsed_seconds\": " << std::setprecision(12) << elapsed << "\n}\n";
  if (mismatches) throw std::runtime_error("stress hash mismatch");
}

void write_update(const Options &o, bool full) {
  const std::string p = o.root + "/data/gate0_prepared/";
  const int max_batch = 1;
  double elapsed_ms = 0;
  std::uint64_t rows = 0, bytes = 0;
  std::vector<DeviceContext> ctx;
  if (full) {
    MappedFile fp(p + "union_fp_u64x4.bin"), ids(p + "union_id_i64.bin"), pc(p + "union_popcnt_u16.bin");
    rows = ids.bytes() / sizeof(std::int64_t);
    bytes = fp.bytes() + ids.bytes() + pc.bytes();
    const auto begin = std::chrono::steady_clock::now();
    ctx = load_base(o, fp, ids, pc, 0, max_batch);
    elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
  } else {
    MappedFile bfp(p + "base_fp_u64x4.bin"), bids(p + "base_id_i64.bin"), bpc(p + "base_popcnt_u16.bin");
    MappedFile dfile(o.root + "/data/stage_a/delta_u64x6.bin");
    const std::uint64_t delta_rows = dfile.bytes() / (6 * sizeof(std::uint64_t));
    ctx = load_base(o, bfp, bids, bpc, delta_rows, max_batch);
    const auto begin = std::chrono::steady_clock::now();
    HostSoA delta = load_delta(dfile, true);
    append_delta(ctx, delta);
    elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    rows = delta_rows; bytes = dfile.bytes();
  }
  std::ofstream f(o.output);
  f << "{\n  \"variant\": \"" << (full ? "FULL_UNION" : "DELTA_APPEND") << "\",\n"
    << "  \"process_index\": " << o.process_index << ",\n"
    << "  \"elapsed_ms\": " << std::setprecision(12) << elapsed_ms << ",\n"
    << "  \"published_rows\": " << rows << ",\n"
    << "  \"artifact_bytes\": " << bytes << "\n}\n";
  for (auto &c : ctx) cleanup(c);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options o = parse_args(argc, argv);
    if (o.mode == "update-full") { write_update(o, true); return 0; }
    if (o.mode == "update-append") { write_update(o, false); return 0; }

    const std::string p = o.root + "/data/gate0_prepared/";
    MappedFile bfp(p + "base_fp_u64x4.bin"), bids(p + "base_id_i64.bin"), bpc(p + "base_popcnt_u16.bin");
    MappedFile dfile(o.root + "/data/stage_a/delta_u64x6.bin");
    MappedFile qfile(o.root + "/data/stage_a/queries_u64x6.bin");
    const std::uint64_t delta_rows = dfile.bytes() / (6 * sizeof(std::uint64_t));
    const int max_batch = o.mode == "correctness" ? 320 :
        (o.mode == "stress" ? 1 : *std::max_element(o.batches.begin(), o.batches.end()));
    auto ctx = load_base(o, bfp, bids, bpc, delta_rows, max_batch);
    HostSoA delta = load_delta(dfile, true);
    append_delta(ctx, delta);
    QueryStore qs(qfile);
    if (o.mode == "correctness") write_correctness(o, ctx, qs);
    else if (o.mode == "bench") write_bench(o, ctx, qs);
    else if (o.mode == "stress") write_stress(o, ctx, delta, qs);
    for (auto &c : ctx) cleanup(c);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
