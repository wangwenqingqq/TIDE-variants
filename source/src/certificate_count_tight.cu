#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr int BLOCK_THREADS = 256;
constexpr std::uint64_t CERT_BYTES = 48;

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t error__ = (expr);                                                \
    if (error__ != cudaSuccess) {                                                \
      std::ostringstream os__;                                                   \
      os__ << #expr << " failed: " << cudaGetErrorString(error__);               \
      throw std::runtime_error(os__.str());                                      \
    }                                                                           \
  } while (0)

struct CountResult {
  std::uint64_t surviving_blocks;
  std::uint64_t surviving_items;
};

__device__ inline std::uint64_t warp_sum(std::uint64_t v) {
  for (int delta = 16; delta > 0; delta >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, delta);
  return v;
}

__global__ void certificate_count_kernel(
    const std::uint64_t *__restrict__ block_or,
    const std::uint16_t *__restrict__ block_minpc,
    const std::uint16_t *__restrict__ block_maxpc,
    const std::uint32_t *__restrict__ block_len, std::uint64_t blocks,
    const std::uint64_t *__restrict__ queries,
    const std::uint16_t *__restrict__ query_pc,
    const std::uint16_t *__restrict__ kth_num,
    const std::uint16_t *__restrict__ kth_den, int query_count,
    CountResult *__restrict__ output) {
  const int q = blockIdx.x;
  if (q >= query_count) return;
  const std::uint64_t q0 = queries[static_cast<std::uint64_t>(q) * 4 + 0];
  const std::uint64_t q1 = queries[static_cast<std::uint64_t>(q) * 4 + 1];
  const std::uint64_t q2 = queries[static_cast<std::uint64_t>(q) * 4 + 2];
  const std::uint64_t q3 = queries[static_cast<std::uint64_t>(q) * 4 + 3];
  const unsigned qcount = query_pc[q];
  const unsigned num = kth_num[q];
  const unsigned den = kth_den[q];
  std::uint64_t surviving_blocks = 0;
  std::uint64_t surviving_items = 0;
  for (std::uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
    const std::uint64_t off = b * 4;
    const unsigned cmax =
        __popcll(q0 & block_or[off + 0]) + __popcll(q1 & block_or[off + 1]) +
        __popcll(q2 & block_or[off + 2]) + __popcll(q3 & block_or[off + 3]);
    const unsigned m = min(qcount, cmax);
    const unsigned b_star = min(max(m, static_cast<unsigned>(block_minpc[b])),
                                static_cast<unsigned>(block_maxpc[b]));
    const unsigned c_star = min(m, b_star);
    unsigned ub_den = qcount + b_star - c_star;
    if (ub_den == 0) ub_den = 1;
    const bool survives = c_star * den >= num * ub_den;
    if (survives) {
      ++surviving_blocks;
      surviving_items += block_len[b];
    }
  }
  surviving_blocks = warp_sum(surviving_blocks);
  surviving_items = warp_sum(surviving_items);
  __shared__ std::uint64_t warp_blocks[8];
  __shared__ std::uint64_t warp_items[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_blocks[warp] = surviving_blocks;
    warp_items[warp] = surviving_items;
  }
  __syncthreads();
  if (warp == 0) {
    std::uint64_t vb = lane < 8 ? warp_blocks[lane] : 0;
    std::uint64_t vi = lane < 8 ? warp_items[lane] : 0;
    vb = warp_sum(vb);
    vi = warp_sum(vi);
    if (lane == 0) output[q] = CountResult{vb, vi};
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
  template <class T> const T *as() const {
    if (bytes_ % sizeof(T)) throw std::runtime_error("bad size: " + path_);
    return static_cast<const T *>(data_);
  }
  std::size_t bytes() const { return bytes_; }

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
  return out;
}

struct Options {
  std::string root, output;
  std::vector<int> gpus;
  int block_size = 0, query_offset = 0, query_count = 0;
};

Options parse_args(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&]() {
      if (++i >= argc) throw std::runtime_error("missing value for " + a);
      return std::string(argv[i]);
    };
    if (a == "--root") o.root = need();
    else if (a == "--output") o.output = need();
    else if (a == "--gpus") o.gpus = parse_gpus(need());
    else if (a == "--block-size") o.block_size = std::stoi(need());
    else if (a == "--query-offset") o.query_offset = std::stoi(need());
    else if (a == "--query-count") o.query_count = std::stoi(need());
    else throw std::runtime_error("unknown argument: " + a);
  }
  if (o.root.empty() || o.output.empty() || o.gpus.empty() ||
      o.block_size < 1 || o.query_count < 1)
    throw std::runtime_error("missing required arguments");
  return o;
}

struct DeviceContext {
  int gpu = -1, q_begin = 0, q_count = 0;
  std::uint64_t *d_or = nullptr, *d_qfp = nullptr;
  std::uint16_t *d_minpc = nullptr, *d_maxpc = nullptr, *d_qpc = nullptr, *d_num = nullptr,
                *d_den = nullptr;
  std::uint32_t *d_len = nullptr;
  CountResult *d_out = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  float kernel_ms = 0;
  std::vector<CountResult> out;
};

void cleanup(DeviceContext &c) {
  cudaSetDevice(c.gpu);
  if (c.d_or) cudaFree(c.d_or);
  if (c.d_minpc) cudaFree(c.d_minpc);
  if (c.d_maxpc) cudaFree(c.d_maxpc);
  if (c.d_len) cudaFree(c.d_len);
  if (c.d_qfp) cudaFree(c.d_qfp);
  if (c.d_qpc) cudaFree(c.d_qpc);
  if (c.d_num) cudaFree(c.d_num);
  if (c.d_den) cudaFree(c.d_den);
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
    const std::string c = p + "certificates/block" + std::to_string(o.block_size);
    MappedFile or_file(c + "_or_u64x4.bin");
    MappedFile min_file(c + "_minpc_u16.bin");
    MappedFile max_file(c + "_maxpc_u16.bin");
    MappedFile len_file(c + "_len_u32.bin");
    MappedFile qfp_file(p + "queries_fp_u64x4.bin");
    MappedFile qpc_file(p + "queries_popcnt_u16.bin");
    MappedFile num_file(p + "certificates/kth_num_u16.bin");
    MappedFile den_file(p + "certificates/kth_den_u16.bin");
    const std::uint64_t blocks = min_file.bytes() / sizeof(std::uint16_t);
    const std::uint64_t all_queries = qpc_file.bytes() / sizeof(std::uint16_t);
    if (or_file.bytes() != blocks * 4 * sizeof(std::uint64_t) ||
        len_file.bytes() != blocks * sizeof(std::uint32_t) ||
        max_file.bytes() != blocks * sizeof(std::uint16_t) ||
        qfp_file.bytes() != all_queries * 4 * sizeof(std::uint64_t) ||
        num_file.bytes() != all_queries * sizeof(std::uint16_t) ||
        den_file.bytes() != all_queries * sizeof(std::uint16_t) ||
        o.query_offset < 0 ||
        static_cast<std::uint64_t>(o.query_offset + o.query_count) > all_queries)
      throw std::runtime_error("file-size or query-range mismatch");
    const auto *h_or = or_file.as<std::uint64_t>();
    const auto *h_min = min_file.as<std::uint16_t>();
    const auto *h_max = max_file.as<std::uint16_t>();
    const auto *h_len = len_file.as<std::uint32_t>();
    const auto *h_qfp = qfp_file.as<std::uint64_t>() +
                        static_cast<std::uint64_t>(o.query_offset) * 4;
    const auto *h_qpc = qpc_file.as<std::uint16_t>() + o.query_offset;
    const auto *h_num = num_file.as<std::uint16_t>() + o.query_offset;
    const auto *h_den = den_file.as<std::uint16_t>() + o.query_offset;

    std::vector<DeviceContext> ctx(o.gpus.size());
    const auto load_begin = std::chrono::steady_clock::now();
    for (std::size_t gi = 0; gi < ctx.size(); ++gi) {
      auto &x = ctx[gi];
      x.gpu = o.gpus[gi];
      x.q_begin = o.query_count * gi / ctx.size();
      const int end = o.query_count * (gi + 1) / ctx.size();
      x.q_count = end - x.q_begin;
      x.out.resize(x.q_count);
      CUDA_CHECK(cudaSetDevice(x.gpu));
      CUDA_CHECK(cudaStreamCreateWithFlags(&x.stream, cudaStreamNonBlocking));
      CUDA_CHECK(cudaEventCreate(&x.start));
      CUDA_CHECK(cudaEventCreate(&x.stop));
      CUDA_CHECK(cudaMalloc(&x.d_or, blocks * 4 * sizeof(std::uint64_t)));
      CUDA_CHECK(cudaMalloc(&x.d_minpc, blocks * sizeof(std::uint16_t)));
      CUDA_CHECK(cudaMalloc(&x.d_maxpc, blocks * sizeof(std::uint16_t)));
      CUDA_CHECK(cudaMalloc(&x.d_len, blocks * sizeof(std::uint32_t)));
      CUDA_CHECK(cudaMalloc(&x.d_qfp, static_cast<std::size_t>(x.q_count) * 4 * sizeof(std::uint64_t)));
      CUDA_CHECK(cudaMalloc(&x.d_qpc, static_cast<std::size_t>(x.q_count) * sizeof(std::uint16_t)));
      CUDA_CHECK(cudaMalloc(&x.d_num, static_cast<std::size_t>(x.q_count) * sizeof(std::uint16_t)));
      CUDA_CHECK(cudaMalloc(&x.d_den, static_cast<std::size_t>(x.q_count) * sizeof(std::uint16_t)));
      CUDA_CHECK(cudaMalloc(&x.d_out, static_cast<std::size_t>(x.q_count) * sizeof(CountResult)));
      CUDA_CHECK(cudaMemcpyAsync(x.d_or, h_or, blocks * 4 * sizeof(std::uint64_t), cudaMemcpyHostToDevice, x.stream));
      CUDA_CHECK(cudaMemcpyAsync(x.d_minpc, h_min, blocks * sizeof(std::uint16_t), cudaMemcpyHostToDevice, x.stream));
      CUDA_CHECK(cudaMemcpyAsync(x.d_maxpc, h_max, blocks * sizeof(std::uint16_t), cudaMemcpyHostToDevice, x.stream));
      CUDA_CHECK(cudaMemcpyAsync(x.d_len, h_len, blocks * sizeof(std::uint32_t), cudaMemcpyHostToDevice, x.stream));
      CUDA_CHECK(cudaMemcpyAsync(x.d_qfp, h_qfp + static_cast<std::uint64_t>(x.q_begin) * 4,
                                 static_cast<std::size_t>(x.q_count) * 4 * sizeof(std::uint64_t), cudaMemcpyHostToDevice, x.stream));
      CUDA_CHECK(cudaMemcpyAsync(x.d_qpc, h_qpc + x.q_begin,
                                 static_cast<std::size_t>(x.q_count) * sizeof(std::uint16_t), cudaMemcpyHostToDevice, x.stream));
      CUDA_CHECK(cudaMemcpyAsync(x.d_num, h_num + x.q_begin,
                                 static_cast<std::size_t>(x.q_count) * sizeof(std::uint16_t), cudaMemcpyHostToDevice, x.stream));
      CUDA_CHECK(cudaMemcpyAsync(x.d_den, h_den + x.q_begin,
                                 static_cast<std::size_t>(x.q_count) * sizeof(std::uint16_t), cudaMemcpyHostToDevice, x.stream));
    }
    for (auto &x : ctx) { CUDA_CHECK(cudaSetDevice(x.gpu)); CUDA_CHECK(cudaStreamSynchronize(x.stream)); }
    const double load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - load_begin).count();

    const auto kernel_begin = std::chrono::steady_clock::now();
    for (auto &x : ctx) {
      CUDA_CHECK(cudaSetDevice(x.gpu));
      CUDA_CHECK(cudaEventRecord(x.start, x.stream));
      certificate_count_kernel<<<x.q_count, BLOCK_THREADS, 0, x.stream>>>(
          x.d_or, x.d_minpc, x.d_maxpc, x.d_len, blocks, x.d_qfp, x.d_qpc, x.d_num,
          x.d_den, x.q_count, x.d_out);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(x.stop, x.stream));
    }
    for (auto &x : ctx) {
      CUDA_CHECK(cudaSetDevice(x.gpu));
      CUDA_CHECK(cudaEventSynchronize(x.stop));
      CUDA_CHECK(cudaEventElapsedTime(&x.kernel_ms, x.start, x.stop));
      CUDA_CHECK(cudaMemcpy(x.out.data(), x.d_out, x.out.size() * sizeof(CountResult), cudaMemcpyDeviceToHost));
    }
    const double kernel_wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - kernel_begin).count();

    std::vector<CountResult> out(o.query_count);
    for (const auto &x : ctx)
      std::copy(x.out.begin(), x.out.end(), out.begin() + x.q_begin);
    std::ofstream csv(o.output);
    csv << "query_index,block_size,total_blocks,surviving_blocks,surviving_items,certificate_bytes,fingerprint_bytes,charged_bytes\n";
    for (int q = 0; q < o.query_count; ++q) {
      const auto cert_bytes = blocks * CERT_BYTES;
      const auto fp_bytes = out[q].surviving_items * 32;
      csv << (o.query_offset + q) << ',' << o.block_size << ',' << blocks << ','
          << out[q].surviving_blocks << ',' << out[q].surviving_items << ','
          << cert_bytes << ',' << fp_bytes << ',' << cert_bytes + fp_bytes << '\n';
    }
    std::cout << "block_size=" << o.block_size << "\nblocks=" << blocks
              << "\nquery_offset=" << o.query_offset << "\nquery_count=" << o.query_count
              << "\nload_seconds=" << load_seconds << "\nkernel_wall_seconds=" << kernel_wall << '\n';
    for (const auto &x : ctx)
      std::cout << "gpu" << x.gpu << "_queries=" << x.q_count << "\ngpu" << x.gpu
                << "_kernel_ms=" << x.kernel_ms << '\n';
    std::cout << "output=" << o.output << '\n';
    for (auto &x : ctx) cleanup(x);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
