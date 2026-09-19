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
    const Candidate c{id, static_cast<std::uint32_t>(global_row_offset + row),
                      static_cast<std::uint16_t>(inter),
                      static_cast<std::uint16_t>(uni)};
    top.insert(c);
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
  const void *data() const { return data_; }
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

std::vector<int> parse_gpus(const std::string &s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoi(item));
  if (out.empty()) throw std::runtime_error("empty GPU list");
  return out;
}

struct Options {
  std::string root;
  std::string dataset = "union";
  std::vector<int> gpus;
  int query_offset = 0;
  int query_count = 0;
  std::string output;
};

Options parse_args(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("missing value for " + a);
      return argv[i];
    };
    if (a == "--root") o.root = need();
    else if (a == "--dataset") o.dataset = need();
    else if (a == "--gpus") o.gpus = parse_gpus(need());
    else if (a == "--query-offset") o.query_offset = std::stoi(need());
    else if (a == "--query-count") o.query_count = std::stoi(need());
    else if (a == "--output") o.output = need();
    else throw std::runtime_error("unknown argument: " + a);
  }
  if (o.root.empty() || o.gpus.empty() || o.query_count <= 0 || o.output.empty())
    throw std::runtime_error("required: --root --gpus --query-count --output");
  if (o.dataset != "base" && o.dataset != "union")
    throw std::runtime_error("dataset must be base or union");
  return o;
}

struct DeviceContext {
  int gpu = -1;
  std::uint64_t begin = 0, rows = 0;
  std::uint64_t *d_fp = nullptr;
  std::int64_t *d_ids = nullptr;
  std::uint16_t *d_pc = nullptr;
  std::uint64_t *d_qfp = nullptr;
  std::int64_t *d_qids = nullptr;
  std::uint16_t *d_qpc = nullptr;
  Candidate *d_out = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  float kernel_ms = 0;
  std::vector<Candidate> output;
};

void cleanup(DeviceContext &c) {
  cudaSetDevice(c.gpu);
  if (c.d_fp) cudaFree(c.d_fp);
  if (c.d_ids) cudaFree(c.d_ids);
  if (c.d_pc) cudaFree(c.d_pc);
  if (c.d_qfp) cudaFree(c.d_qfp);
  if (c.d_qids) cudaFree(c.d_qids);
  if (c.d_qpc) cudaFree(c.d_qpc);
  if (c.d_out) cudaFree(c.d_out);
  if (c.start) cudaEventDestroy(c.start);
  if (c.stop) cudaEventDestroy(c.stop);
  if (c.stream) cudaStreamDestroy(c.stream);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options o = parse_args(argc, argv);
    const std::string p = o.root + "/data/prepared/";
    MappedFile target_fp(p + o.dataset + "_fp_u64x4.bin");
    MappedFile target_ids(p + o.dataset + "_id_i64.bin");
    MappedFile target_pc(p + o.dataset + "_popcnt_u16.bin");
    MappedFile query_fp(p + "queries_fp_u64x4.bin");
    MappedFile query_ids(p + "queries_id_i64.bin");
    MappedFile query_pc(p + "queries_popcnt_u16.bin");

    const std::uint64_t rows = target_ids.bytes() / sizeof(std::int64_t);
    if (target_fp.bytes() != rows * 4 * sizeof(std::uint64_t) ||
        target_pc.bytes() != rows * sizeof(std::uint16_t))
      throw std::runtime_error("target file-size mismatch");
    const std::uint64_t all_queries = query_ids.bytes() / sizeof(std::int64_t);
    if (query_fp.bytes() != all_queries * 4 * sizeof(std::uint64_t) ||
        query_pc.bytes() != all_queries * sizeof(std::uint16_t))
      throw std::runtime_error("query file-size mismatch");
    if (o.query_offset < 0 || o.query_count < 1 ||
        static_cast<std::uint64_t>(o.query_offset + o.query_count) > all_queries)
      throw std::runtime_error("query range out of bounds");

    const auto *h_fp = target_fp.as<std::uint64_t>();
    const auto *h_ids = target_ids.as<std::int64_t>();
    const auto *h_pc = target_pc.as<std::uint16_t>();
    const auto *h_qfp = query_fp.as<std::uint64_t>() +
                        static_cast<std::uint64_t>(o.query_offset) * 4;
    const auto *h_qids = query_ids.as<std::int64_t>() + o.query_offset;
    const auto *h_qpc = query_pc.as<std::uint16_t>() + o.query_offset;

    std::vector<DeviceContext> ctx(o.gpus.size());
    const auto load_begin = std::chrono::steady_clock::now();
    for (std::size_t gi = 0; gi < o.gpus.size(); ++gi) {
      auto &c = ctx[gi];
      c.gpu = o.gpus[gi];
      c.begin = rows * gi / o.gpus.size();
      const std::uint64_t end = rows * (gi + 1) / o.gpus.size();
      c.rows = end - c.begin;
      c.output.resize(static_cast<std::size_t>(o.query_count) * K);
      CUDA_CHECK(cudaSetDevice(c.gpu));
      CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
      CUDA_CHECK(cudaEventCreate(&c.start));
      CUDA_CHECK(cudaEventCreate(&c.stop));
      CUDA_CHECK(cudaMalloc(&c.d_fp, c.rows * 4 * sizeof(std::uint64_t)));
      CUDA_CHECK(cudaMalloc(&c.d_ids, c.rows * sizeof(std::int64_t)));
      CUDA_CHECK(cudaMalloc(&c.d_pc, c.rows * sizeof(std::uint16_t)));
      CUDA_CHECK(cudaMalloc(&c.d_qfp, static_cast<std::size_t>(o.query_count) * 4 *
                                         sizeof(std::uint64_t)));
      CUDA_CHECK(cudaMalloc(&c.d_qids, static_cast<std::size_t>(o.query_count) *
                                          sizeof(std::int64_t)));
      CUDA_CHECK(cudaMalloc(&c.d_qpc, static_cast<std::size_t>(o.query_count) *
                                         sizeof(std::uint16_t)));
      CUDA_CHECK(cudaMalloc(&c.d_out, static_cast<std::size_t>(o.query_count) * K *
                                         sizeof(Candidate)));
      CUDA_CHECK(cudaMemcpyAsync(c.d_fp, h_fp + c.begin * 4,
                                 c.rows * 4 * sizeof(std::uint64_t),
                                 cudaMemcpyHostToDevice, c.stream));
      CUDA_CHECK(cudaMemcpyAsync(c.d_ids, h_ids + c.begin,
                                 c.rows * sizeof(std::int64_t),
                                 cudaMemcpyHostToDevice, c.stream));
      CUDA_CHECK(cudaMemcpyAsync(c.d_pc, h_pc + c.begin,
                                 c.rows * sizeof(std::uint16_t),
                                 cudaMemcpyHostToDevice, c.stream));
      CUDA_CHECK(cudaMemcpyAsync(c.d_qfp, h_qfp,
                                 static_cast<std::size_t>(o.query_count) * 4 *
                                     sizeof(std::uint64_t),
                                 cudaMemcpyHostToDevice, c.stream));
      CUDA_CHECK(cudaMemcpyAsync(c.d_qids, h_qids,
                                 static_cast<std::size_t>(o.query_count) *
                                     sizeof(std::int64_t),
                                 cudaMemcpyHostToDevice, c.stream));
      CUDA_CHECK(cudaMemcpyAsync(c.d_qpc, h_qpc,
                                 static_cast<std::size_t>(o.query_count) *
                                     sizeof(std::uint16_t),
                                 cudaMemcpyHostToDevice, c.stream));
    }
    for (auto &c : ctx) {
      CUDA_CHECK(cudaSetDevice(c.gpu));
      CUDA_CHECK(cudaStreamSynchronize(c.stream));
    }
    const double load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - load_begin).count();

    const std::size_t shared_bytes = BLOCK * K * sizeof(Candidate);
    const auto launch_begin = std::chrono::steady_clock::now();
    for (auto &c : ctx) {
      CUDA_CHECK(cudaSetDevice(c.gpu));
      CUDA_CHECK(cudaEventRecord(c.start, c.stream));
      exact_topk_kernel<<<o.query_count, BLOCK, shared_bytes, c.stream>>>(
          c.d_fp, c.d_ids, c.d_pc, c.rows, c.begin, c.d_qfp, c.d_qids,
          c.d_qpc, o.query_count, c.d_out);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(c.stop, c.stream));
    }
    for (auto &c : ctx) {
      CUDA_CHECK(cudaSetDevice(c.gpu));
      CUDA_CHECK(cudaEventSynchronize(c.stop));
      CUDA_CHECK(cudaEventElapsedTime(&c.kernel_ms, c.start, c.stop));
    }
    const double launch_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - launch_begin).count();

    for (auto &c : ctx) {
      CUDA_CHECK(cudaSetDevice(c.gpu));
      CUDA_CHECK(cudaMemcpy(c.output.data(), c.d_out,
                            c.output.size() * sizeof(Candidate),
                            cudaMemcpyDeviceToHost));
    }

    std::vector<Candidate> merged(static_cast<std::size_t>(o.query_count) * K);
    for (int q = 0; q < o.query_count; ++q) {
      Top10 top;
      top.init();
      for (const auto &c : ctx)
        for (int k = 0; k < K; ++k) top.insert(c.output[q * K + k]);
      top.store(merged.data() + static_cast<std::size_t>(q) * K);
    }

    std::ofstream csv(o.output);
    if (!csv) throw std::runtime_error("cannot open output: " + o.output);
    csv << "query_index,query_id,rank,result_id,row,num,den,similarity\n";
    csv << std::setprecision(12);
    for (int q = 0; q < o.query_count; ++q) {
      for (int k = 0; k < K; ++k) {
        const Candidate &c = merged[q * K + k];
        csv << (o.query_offset + q) << ',' << h_qids[q] << ',' << k << ','
            << c.id << ',' << c.row << ',' << c.num << ',' << c.den << ','
            << (static_cast<double>(c.num) / c.den) << '\n';
      }
    }
    csv.close();

    std::cout << "dataset=" << o.dataset << "\nrows=" << rows
              << "\nquery_offset=" << o.query_offset
              << "\nquery_count=" << o.query_count << "\ngpus=";
    for (std::size_t i = 0; i < o.gpus.size(); ++i)
      std::cout << (i ? "," : "") << o.gpus[i];
    std::cout << "\nload_seconds=" << load_seconds
              << "\nlaunch_wall_seconds=" << launch_seconds << '\n';
    for (const auto &c : ctx)
      std::cout << "gpu" << c.gpu << "_rows=" << c.rows << "\ngpu" << c.gpu
                << "_kernel_ms=" << c.kernel_ms << '\n';
    std::cout << "output=" << o.output << '\n';

    for (auto &c : ctx) cleanup(c);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
