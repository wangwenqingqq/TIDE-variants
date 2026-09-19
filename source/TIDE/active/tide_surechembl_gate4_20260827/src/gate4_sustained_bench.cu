#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int BLOCK = 256;
constexpr std::uint32_t OUTPUT_CAPACITY = 65536;

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t error__ = (expr);                                                \
    if (error__ != cudaSuccess) {                                                \
      std::ostringstream os__;                                                   \
      os__ << #expr << " failed: " << cudaGetErrorString(error__);              \
      throw std::runtime_error(os__.str());                                      \
    }                                                                            \
  } while (0)

struct DeviceHit {
  std::uint32_t id;
  std::uint16_t num;
  std::uint16_t den;
};
static_assert(sizeof(DeviceHit) == 8);

struct ReleasedHit {
  std::uint32_t id;
  float coeff;
};
static_assert(sizeof(ReleasedHit) == 8);

struct HostHit {
  std::uint32_t id;
  std::uint16_t num;
  std::uint16_t den;
  float coeff;
};

struct alignas(16) DeviceQuery {
  std::uint64_t fp[4];
  std::uint16_t popcount;
  std::uint16_t threshold_num;
  std::uint16_t threshold_den;
  std::uint16_t reserved;
};

struct alignas(16) EpochDescriptor {
  std::uint64_t epoch;
  std::uint32_t delta_visible;
  std::uint32_t reserved;
};

__global__ void exact_segment_scan_kernel(
    const std::uint64_t *__restrict__ base_fp,
    const std::uint64_t *__restrict__ base_ids,
    const std::uint16_t *__restrict__ base_pc, std::uint64_t base_rows,
    const std::uint64_t *__restrict__ delta_fp,
    const std::uint64_t *__restrict__ delta_ids,
    const std::uint16_t *__restrict__ delta_pc, std::uint64_t delta_rows,
    const EpochDescriptor *__restrict__ epoch,
    const DeviceQuery *__restrict__ query, DeviceHit *__restrict__ hits,
    std::uint32_t capacity, std::uint32_t *__restrict__ total_hits,
    std::uint32_t *__restrict__ overflow) {
  const bool delta_visible = epoch && epoch->delta_visible != 0;
  const std::uint64_t visible_delta = delta_visible ? delta_rows : 0;
  const std::uint64_t total = base_rows + visible_delta;
  const DeviceQuery q = *query;
  for (std::uint64_t logical =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       logical < total;
       logical += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const bool from_delta = logical >= base_rows;
    const std::uint64_t row = from_delta ? logical - base_rows : logical;
    const std::uint64_t *fp = from_delta ? delta_fp : base_fp;
    const std::uint64_t *ids = from_delta ? delta_ids : base_ids;
    const std::uint16_t *pcs = from_delta ? delta_pc : base_pc;
    const std::uint64_t offset = row * 4;
    const unsigned intersection =
        __popcll(q.fp[0] & fp[offset + 0]) +
        __popcll(q.fp[1] & fp[offset + 1]) +
        __popcll(q.fp[2] & fp[offset + 2]) +
        __popcll(q.fp[3] & fp[offset + 3]);
    unsigned union_count =
        static_cast<unsigned>(q.popcount) + pcs[row] - intersection;
    if (union_count == 0) union_count = 1;
    if (intersection * static_cast<unsigned>(q.threshold_den) <
        union_count * static_cast<unsigned>(q.threshold_num))
      continue;
    const std::uint32_t slot = atomicAdd(total_hits, 1u);
    if (slot < capacity) {
      hits[slot] = DeviceHit{static_cast<std::uint32_t>(ids[row]),
                             static_cast<std::uint16_t>(intersection),
                             static_cast<std::uint16_t>(union_count)};
    } else {
      atomicExch(overflow, 1u);
    }
  }
}

// Source-faithful FPSim2 0.7.4 taniRAW comparator. The only changes are the
// function name and passing the already bounded slice as the database pointer.
__global__ void released_taniRAW(
    const unsigned long long int *query,
    const unsigned long long int *qcount,
    const unsigned long long int *db,
    const unsigned long long int *popcnts, const float *threshold, float *out) {
  __shared__ int common[4];
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  common[threadIdx.x] = __popcll(query[threadIdx.x] & db[tid]);
  __syncthreads();
  if (0 == threadIdx.x) {
    int comm_sum = 0;
    for (int i = 0; i < 4; i++) comm_sum += common[i];
    float coeff = 0.0;
    coeff = *qcount + popcnts[blockIdx.x] - comm_sum;
    if (coeff != 0.0) coeff = comm_sum / coeff;
    out[blockIdx.x] = coeff >= *threshold ? coeff : 0.0;
  }
}

__global__ void collect_released_kernel(
    const float *__restrict__ similarities,
    const std::uint64_t *__restrict__ ids, std::uint64_t rows,
    ReleasedHit *__restrict__ hits, std::uint32_t capacity,
    std::uint32_t *__restrict__ total_hits,
    std::uint32_t *__restrict__ overflow) {
  for (std::uint64_t row =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < rows;
       row += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const float coeff = similarities[row];
    if (!(coeff > 0.0f)) continue;
    const std::uint32_t slot = atomicAdd(total_hits, 1u);
    if (slot < capacity) {
      hits[slot] = ReleasedHit{static_cast<std::uint32_t>(ids[row]), coeff};
    } else {
      atomicExch(overflow, 1u);
    }
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

struct HostDBView {
  const std::uint64_t *fp = nullptr;
  const std::uint64_t *ids = nullptr;
  const std::uint16_t *pc = nullptr;
  std::uint64_t rows = 0;
  std::array<std::uint64_t, 258> cumulative{};
};

HostDBView make_host_view(const MappedFile &fp, const MappedFile &ids,
                          const MappedFile &pc, std::uint64_t limit_rows = 0) {
  HostDBView view;
  view.rows = ids.bytes() / sizeof(std::uint64_t);
  if (limit_rows && limit_rows < view.rows) view.rows = limit_rows;
  if (fp.bytes() < view.rows * 4 * sizeof(std::uint64_t) ||
      pc.bytes() < view.rows * sizeof(std::uint16_t))
    throw std::runtime_error("database component size mismatch");
  view.fp = fp.as<std::uint64_t>();
  view.ids = ids.as<std::uint64_t>();
  view.pc = pc.as<std::uint16_t>();
  std::array<std::uint64_t, 257> counts{};
  std::uint16_t previous = 0;
  for (std::uint64_t row = 0; row < view.rows; ++row) {
    const auto value = view.pc[row];
    if (value > 256 || (row && value < previous))
      throw std::runtime_error("database is not population-count sorted");
    counts[value]++;
    previous = value;
  }
  for (int popcount = 0; popcount <= 256; ++popcount)
    view.cumulative[popcount + 1] =
        view.cumulative[popcount] + counts[popcount];
  return view;
}

struct OwnedHostDB {
  std::vector<std::uint64_t> fp;
  std::vector<std::uint64_t> ids;
  std::vector<std::uint16_t> pc;
  HostDBView view() const {
    HostDBView out;
    out.fp = fp.data();
    out.ids = ids.data();
    out.pc = pc.data();
    out.rows = ids.size();
    std::array<std::uint64_t, 257> counts{};
    for (auto value : pc) counts[value]++;
    for (int i = 0; i <= 256; ++i)
      out.cumulative[i + 1] = out.cumulative[i] + counts[i];
    return out;
  }
};

OwnedHostDB load_delta(const MappedFile &file) {
  if (file.bytes() % (6 * sizeof(std::uint64_t)))
    throw std::runtime_error("bad delta byte size");
  const auto *rows = file.as<std::uint64_t>();
  const std::size_t count = file.bytes() / (6 * sizeof(std::uint64_t));
  OwnedHostDB out;
  out.fp.resize(count * 4);
  out.ids.resize(count);
  out.pc.resize(count);
  std::uint16_t previous = 0;
  for (std::size_t row = 0; row < count; ++row) {
    out.ids[row] = rows[row * 6];
    unsigned actual_pc = 0;
    for (int word = 0; word < 4; ++word) {
      out.fp[row * 4 + word] = rows[row * 6 + 1 + word];
      actual_pc += __builtin_popcountll(out.fp[row * 4 + word]);
    }
    out.pc[row] = static_cast<std::uint16_t>(rows[row * 6 + 5]);
    if (actual_pc != out.pc[row] || (row && out.pc[row] < previous))
      throw std::runtime_error("delta validation failed");
    previous = out.pc[row];
  }
  return out;
}

struct Query {
  std::uint64_t id;
  std::uint64_t fp[4];
  std::uint16_t pc;
};

class QueryStore {
 public:
  explicit QueryStore(const MappedFile &file) {
    if (file.bytes() % (6 * sizeof(std::uint64_t)))
      throw std::runtime_error("bad query byte size");
    const auto *rows = file.as<std::uint64_t>();
    const std::size_t count = file.bytes() / (6 * sizeof(std::uint64_t));
    queries_.resize(count);
    for (std::size_t row = 0; row < count; ++row) {
      Query &q = queries_[row];
      q.id = rows[row * 6];
      unsigned actual_pc = 0;
      for (int word = 0; word < 4; ++word) {
        q.fp[word] = rows[row * 6 + 1 + word];
        actual_pc += __builtin_popcountll(q.fp[word]);
      }
      q.pc = static_cast<std::uint16_t>(rows[row * 6 + 5]);
      if (actual_pc != q.pc) throw std::runtime_error("query popcount mismatch");
    }
  }
  const Query &operator[](std::size_t index) const { return queries_.at(index); }
  std::size_t size() const { return queries_.size(); }

 private:
  std::vector<Query> queries_;
};

std::pair<int, int> popcount_interval(std::uint16_t query_pc, int num, int den) {
  const int lower = (num * static_cast<int>(query_pc) + den - 1) / den;
  const int upper = den * static_cast<int>(query_pc) / num;
  return {std::max(0, lower), std::min(256, upper)};
}

struct DeviceDB {
  std::uint64_t *fp = nullptr;
  std::uint64_t *ids = nullptr;
  std::uint16_t *pc = nullptr;
  std::uint64_t *pc64 = nullptr;
  std::uint64_t rows = 0;
  int gpu = -1;

  void allocate(int target_gpu, std::uint64_t count, bool need_pc64) {
    gpu = target_gpu;
    rows = count;
    CUDA_CHECK(cudaSetDevice(gpu));
    CUDA_CHECK(cudaMalloc(&fp, rows * 4 * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&ids, rows * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&pc, rows * sizeof(std::uint16_t)));
    if (need_pc64) CUDA_CHECK(cudaMalloc(&pc64, rows * sizeof(std::uint64_t)));
  }

  void upload(const HostDBView &host, bool need_pc64) {
    allocate(gpu < 0 ? 0 : gpu, host.rows, need_pc64);
  }

  void copy_from(const HostDBView &host, bool fill_pc64) {
    if (rows != host.rows) throw std::runtime_error("device database row mismatch");
    CUDA_CHECK(cudaSetDevice(gpu));
    CUDA_CHECK(cudaMemcpy(fp, host.fp, rows * 4 * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ids, host.ids, rows * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pc, host.pc, rows * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    if (fill_pc64) {
      std::vector<std::uint64_t> widened(rows);
      for (std::uint64_t row = 0; row < rows; ++row) widened[row] = host.pc[row];
      CUDA_CHECK(cudaMemcpy(pc64, widened.data(), rows * sizeof(std::uint64_t),
                            cudaMemcpyHostToDevice));
    }
  }

  void init(int target_gpu, const HostDBView &host, bool need_pc64) {
    allocate(target_gpu, host.rows, need_pc64);
    copy_from(host, need_pc64);
  }

  void release() {
    if (gpu < 0) return;
    cudaSetDevice(gpu);
    if (fp) cudaFree(fp);
    if (ids) cudaFree(ids);
    if (pc) cudaFree(pc);
    if (pc64) cudaFree(pc64);
    fp = ids = pc64 = nullptr;
    pc = nullptr;
    rows = 0;
    gpu = -1;
  }
  ~DeviceDB() { release(); }
  DeviceDB() = default;
  DeviceDB(const DeviceDB &) = delete;
  DeviceDB &operator=(const DeviceDB &) = delete;
};

struct RunResult {
  double service_ms = 0;
  float kernel_ms = 0;
  float device_ms = 0;
  std::uint32_t observed_hits = 0;
  bool overflow = false;
  std::vector<HostHit> hits;
};

std::uint32_t float_bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void sort_hits(std::vector<HostHit> &hits) {
  std::sort(hits.begin(), hits.end(), [](const HostHit &a, const HostHit &b) {
    if (a.coeff != b.coeff) return a.coeff > b.coeff;
    return a.id < b.id;
  });
}

std::uint64_t hash_hits(const std::vector<HostHit> &hits) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const auto &hit : hits) {
    const std::uint32_t fields[2] = {hit.id, float_bits(hit.coeff)};
    const auto *bytes = reinterpret_cast<const unsigned char *>(fields);
    for (std::size_t i = 0; i < sizeof(fields); ++i) {
      hash ^= bytes[i];
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

class CustomRuntime {
 public:
  CustomRuntime(int gpu, std::uint32_t capacity = OUTPUT_CAPACITY)
      : gpu_(gpu), capacity_(capacity) {
    CUDA_CHECK(cudaSetDevice(gpu_));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&device_start_));
    CUDA_CHECK(cudaEventCreate(&kernel_start_));
    CUDA_CHECK(cudaEventCreate(&kernel_stop_));
    CUDA_CHECK(cudaEventCreate(&device_stop_));
    CUDA_CHECK(cudaMalloc(&d_query_, sizeof(DeviceQuery)));
    CUDA_CHECK(cudaMalloc(&d_epoch_, sizeof(EpochDescriptor)));
    CUDA_CHECK(cudaMalloc(&d_hits_, capacity_ * sizeof(DeviceHit)));
    CUDA_CHECK(cudaMalloc(&d_count_, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_overflow_, sizeof(std::uint32_t)));
  }
  ~CustomRuntime() {
    cudaSetDevice(gpu_);
    if (d_query_) cudaFree(d_query_);
    if (d_epoch_) cudaFree(d_epoch_);
    if (d_hits_) cudaFree(d_hits_);
    if (d_count_) cudaFree(d_count_);
    if (d_overflow_) cudaFree(d_overflow_);
    if (device_start_) cudaEventDestroy(device_start_);
    if (kernel_start_) cudaEventDestroy(kernel_start_);
    if (kernel_stop_) cudaEventDestroy(kernel_stop_);
    if (device_stop_) cudaEventDestroy(device_stop_);
    if (stream_) cudaStreamDestroy(stream_);
  }
  CustomRuntime(const CustomRuntime &) = delete;
  CustomRuntime &operator=(const CustomRuntime &) = delete;

  void set_epoch(std::uint64_t epoch, bool delta_visible, bool synchronize = true) {
    EpochDescriptor descriptor{epoch, delta_visible ? 1u : 0u, 0u};
    CUDA_CHECK(cudaMemcpyAsync(d_epoch_, &descriptor, sizeof(descriptor),
                               cudaMemcpyHostToDevice, stream_));
    if (synchronize) CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  RunResult run(const DeviceDB &base, std::uint64_t base_begin,
                std::uint64_t base_end, const DeviceDB *delta,
                std::uint64_t delta_begin, std::uint64_t delta_end,
                const Query &query, int threshold_num, int threshold_den) {
    CUDA_CHECK(cudaSetDevice(gpu_));
    DeviceQuery device_query{{query.fp[0], query.fp[1], query.fp[2], query.fp[3]},
                             query.pc,
                             static_cast<std::uint16_t>(threshold_num),
                             static_cast<std::uint16_t>(threshold_den), 0};
    const auto host_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(device_start_, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_query_, &device_query, sizeof(device_query),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemsetAsync(d_count_, 0, sizeof(std::uint32_t), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_overflow_, 0, sizeof(std::uint32_t), stream_));
    CUDA_CHECK(cudaEventRecord(kernel_start_, stream_));
    const std::uint64_t base_rows = base_end - base_begin;
    const std::uint64_t delta_rows = delta ? delta_end - delta_begin : 0;
    const std::uint64_t total_rows = base_rows + delta_rows;
    const int blocks = static_cast<int>(std::min<std::uint64_t>(
        (total_rows + BLOCK - 1) / BLOCK, 65535));
    exact_segment_scan_kernel<<<std::max(1, blocks), BLOCK, 0, stream_>>>(
        base.fp + base_begin * 4, base.ids + base_begin, base.pc + base_begin,
        base_rows, delta ? delta->fp + delta_begin * 4 : nullptr,
        delta ? delta->ids + delta_begin : nullptr,
        delta ? delta->pc + delta_begin : nullptr, delta_rows, d_epoch_, d_query_,
        d_hits_, capacity_, d_count_, d_overflow_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernel_stop_, stream_));
    std::uint32_t count = 0, overflow = 0;
    CUDA_CHECK(cudaMemcpyAsync(&count, d_count_, sizeof(count),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(&overflow, d_overflow_, sizeof(overflow),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const std::uint32_t copied = std::min(count, capacity_);
    std::vector<DeviceHit> raw(copied);
    if (copied)
      CUDA_CHECK(cudaMemcpyAsync(raw.data(), d_hits_, copied * sizeof(DeviceHit),
                                 cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaEventRecord(device_stop_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    RunResult result;
    result.observed_hits = count;
    result.overflow = overflow != 0 || count > capacity_;
    result.hits.reserve(copied);
    for (const auto &hit : raw) {
      result.hits.push_back(HostHit{hit.id, hit.num, hit.den,
                                    static_cast<float>(hit.num) / hit.den});
    }
    sort_hits(result.hits);
    result.service_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - host_start)
                            .count();
    CUDA_CHECK(cudaEventElapsedTime(&result.kernel_ms, kernel_start_, kernel_stop_));
    CUDA_CHECK(cudaEventElapsedTime(&result.device_ms, device_start_, device_stop_));
    return result;
  }

  cudaStream_t stream() const { return stream_; }
  EpochDescriptor *device_epoch() const { return d_epoch_; }

 private:
  int gpu_;
  std::uint32_t capacity_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t device_start_ = nullptr, kernel_start_ = nullptr,
              kernel_stop_ = nullptr, device_stop_ = nullptr;
  DeviceQuery *d_query_ = nullptr;
  EpochDescriptor *d_epoch_ = nullptr;
  DeviceHit *d_hits_ = nullptr;
  std::uint32_t *d_count_ = nullptr, *d_overflow_ = nullptr;
};

class ReleasedRuntime {
 public:
  ReleasedRuntime(int gpu, std::uint64_t maximum_rows,
                  std::uint32_t capacity = OUTPUT_CAPACITY)
      : gpu_(gpu), capacity_(capacity) {
    CUDA_CHECK(cudaSetDevice(gpu_));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&device_start_));
    CUDA_CHECK(cudaEventCreate(&kernel_start_));
    CUDA_CHECK(cudaEventCreate(&kernel_stop_));
    CUDA_CHECK(cudaEventCreate(&materialize_stop_));
    CUDA_CHECK(cudaEventCreate(&device_stop_));
    CUDA_CHECK(cudaMalloc(&d_query_, 4 * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_qcount_, sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_threshold_, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_similarities_, maximum_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hits_, capacity_ * sizeof(ReleasedHit)));
    CUDA_CHECK(cudaMalloc(&d_count_, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_overflow_, sizeof(std::uint32_t)));
  }
  ~ReleasedRuntime() {
    cudaSetDevice(gpu_);
    if (d_query_) cudaFree(d_query_);
    if (d_qcount_) cudaFree(d_qcount_);
    if (d_threshold_) cudaFree(d_threshold_);
    if (d_similarities_) cudaFree(d_similarities_);
    if (d_hits_) cudaFree(d_hits_);
    if (d_count_) cudaFree(d_count_);
    if (d_overflow_) cudaFree(d_overflow_);
    if (device_start_) cudaEventDestroy(device_start_);
    if (kernel_start_) cudaEventDestroy(kernel_start_);
    if (kernel_stop_) cudaEventDestroy(kernel_stop_);
    if (materialize_stop_) cudaEventDestroy(materialize_stop_);
    if (device_stop_) cudaEventDestroy(device_stop_);
    if (stream_) cudaStreamDestroy(stream_);
  }
  ReleasedRuntime(const ReleasedRuntime &) = delete;
  ReleasedRuntime &operator=(const ReleasedRuntime &) = delete;

  RunResult run(const DeviceDB &database, std::uint64_t begin, std::uint64_t end,
                const Query &query, int threshold_num, int threshold_den) {
    if (!database.pc64) throw std::runtime_error("released comparator needs uint64 popcounts");
    CUDA_CHECK(cudaSetDevice(gpu_));
    const std::uint64_t rows = end - begin;
    const std::uint64_t qcount = query.pc;
    const float threshold = static_cast<float>(threshold_num) / threshold_den;
    const auto host_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(device_start_, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_query_, query.fp, 4 * sizeof(std::uint64_t),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_qcount_, &qcount, sizeof(qcount),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_threshold_, &threshold, sizeof(threshold),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemsetAsync(d_count_, 0, sizeof(std::uint32_t), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_overflow_, 0, sizeof(std::uint32_t), stream_));
    CUDA_CHECK(cudaEventRecord(kernel_start_, stream_));
    if (rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("released grid exceeds CUDA x dimension");
    released_taniRAW<<<static_cast<unsigned>(rows), 4, 0, stream_>>>(
        reinterpret_cast<const unsigned long long *>(d_query_),
        reinterpret_cast<const unsigned long long *>(d_qcount_),
        reinterpret_cast<const unsigned long long *>(database.fp + begin * 4),
        reinterpret_cast<const unsigned long long *>(database.pc64 + begin),
        d_threshold_, d_similarities_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernel_stop_, stream_));
    const int blocks = static_cast<int>(std::min<std::uint64_t>(
        (rows + BLOCK - 1) / BLOCK, 65535));
    collect_released_kernel<<<std::max(1, blocks), BLOCK, 0, stream_>>>(
        d_similarities_, database.ids + begin, rows, d_hits_, capacity_,
        d_count_, d_overflow_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(materialize_stop_, stream_));
    std::uint32_t count = 0, overflow = 0;
    CUDA_CHECK(cudaMemcpyAsync(&count, d_count_, sizeof(count),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(&overflow, d_overflow_, sizeof(overflow),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const std::uint32_t copied = std::min(count, capacity_);
    std::vector<ReleasedHit> raw(copied);
    if (copied)
      CUDA_CHECK(cudaMemcpyAsync(raw.data(), d_hits_, copied * sizeof(ReleasedHit),
                                 cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaEventRecord(device_stop_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    RunResult result;
    result.observed_hits = count;
    result.overflow = overflow != 0 || count > capacity_;
    result.hits.reserve(copied);
    for (const auto &hit : raw)
      result.hits.push_back(HostHit{hit.id, 0, 0, hit.coeff});
    sort_hits(result.hits);
    result.service_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - host_start)
                            .count();
    CUDA_CHECK(cudaEventElapsedTime(&result.kernel_ms, kernel_start_, kernel_stop_));
    CUDA_CHECK(cudaEventElapsedTime(&result.device_ms, device_start_, device_stop_));
    return result;
  }

 private:
  int gpu_;
  std::uint32_t capacity_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t device_start_ = nullptr, kernel_start_ = nullptr,
              kernel_stop_ = nullptr, materialize_stop_ = nullptr,
              device_stop_ = nullptr;
  std::uint64_t *d_query_ = nullptr, *d_qcount_ = nullptr;
  float *d_threshold_ = nullptr, *d_similarities_ = nullptr;
  ReleasedHit *d_hits_ = nullptr;
  std::uint32_t *d_count_ = nullptr, *d_overflow_ = nullptr;
};

struct Options {
  std::string root, mode, variant, output, threshold_order = "70,80";
  int gpu = 2;
  int process_index = 0;
  int warmup = 8;
  int cycles = 1000;
  std::uint64_t limit_rows = 0;
};

Options parse_args(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto need = [&]() -> std::string {
      if (++index >= argc) throw std::runtime_error("missing value for " + argument);
      return argv[index];
    };
    if (argument == "--root") options.root = need();
    else if (argument == "--mode") options.mode = need();
    else if (argument == "--variant") options.variant = need();
    else if (argument == "--output") options.output = need();
    else if (argument == "--gpu") options.gpu = std::stoi(need());
    else if (argument == "--process-index") options.process_index = std::stoi(need());
    else if (argument == "--warmup") options.warmup = std::stoi(need());
    else if (argument == "--cycles") options.cycles = std::stoi(need());
    else if (argument == "--threshold-order") options.threshold_order = need();
    else if (argument == "--limit-rows") options.limit_rows = std::stoull(need());
    else throw std::runtime_error("unknown argument: " + argument);
  }
  if (options.mode.empty() || options.output.empty())
    throw std::runtime_error("required: --mode --output");
  if (options.mode != "synthetic" && options.root.empty())
    throw std::runtime_error("--root is required outside synthetic mode");
  if (options.variant.empty()) options.variant = "all";
  return options;
}

std::vector<std::pair<int, int>> threshold_order(const std::string &text) {
  if (text == "70,80") return {{7, 10}, {4, 5}};
  if (text == "80,70") return {{4, 5}, {7, 10}};
  throw std::runtime_error("threshold order must be 70,80 or 80,70");
}

struct RealInputs {
  MappedFile base_fp, base_ids, base_pc, union_fp, union_ids, union_pc, delta_file,
      query_file;
  HostDBView base, fresh;
  OwnedHostDB delta_owned;
  HostDBView delta;
  QueryStore queries;
  explicit RealInputs(const Options &options)
      : base_fp(options.root + "/data/gate0_prepared/base_fp_u64x4.bin"),
        base_ids(options.root + "/data/gate0_prepared/base_id_i64.bin"),
        base_pc(options.root + "/data/gate0_prepared/base_popcnt_u16.bin"),
        union_fp(options.root + "/data/gate0_prepared/union_fp_u64x4.bin"),
        union_ids(options.root + "/data/gate0_prepared/union_id_i64.bin"),
        union_pc(options.root + "/data/gate0_prepared/union_popcnt_u16.bin"),
        delta_file(options.root + "/data/stage_a/delta_u64x6.bin"),
        query_file(options.root + "/data/stage_a/queries_u64x6.bin"),
        base(make_host_view(base_fp, base_ids, base_pc, options.limit_rows)),
        fresh(make_host_view(union_fp, union_ids, union_pc, options.limit_rows)),
        delta_owned(load_delta(delta_file)),
        delta(delta_owned.view()),
        queries(query_file) {}
};

std::pair<std::uint64_t, std::uint64_t> bounds(const HostDBView &db,
                                                const Query &query, int num,
                                                int den) {
  const auto [lower, upper] = popcount_interval(query.pc, num, den);
  return {db.cumulative[lower], db.cumulative[upper + 1]};
}

void require_no_overflow(const RunResult &result, const std::string &where) {
  if (result.overflow)
    throw std::runtime_error(where + " output overflow: observed " +
                             std::to_string(result.observed_hits));
}

RunResult run_variant(const std::string &variant, const Query &query, int num,
                      int den, const HostDBView &base_host,
                      const HostDBView &delta_host,
                      const HostDBView &union_host, const DeviceDB &base,
                      const DeviceDB &delta, const DeviceDB &unified,
                      CustomRuntime &custom, ReleasedRuntime &released) {
  if (variant == "released") {
    const auto [begin, end] = bounds(union_host, query, num, den);
    return released.run(unified, begin, end, query, num, den);
  }
  if (variant == "flat") {
    const auto [begin, end] = bounds(union_host, query, num, den);
    custom.set_epoch(0, false);
    return custom.run(unified, begin, end, nullptr, 0, 0, query, num, den);
  }
  if (variant == "segmented") {
    const auto [base_begin, base_end] = bounds(base_host, query, num, den);
    const auto [delta_begin, delta_end] = bounds(delta_host, query, num, den);
    custom.set_epoch(1, true);
    return custom.run(base, base_begin, base_end, &delta, delta_begin, delta_end,
                      query, num, den);
  }
  throw std::runtime_error("unknown variant: " + variant);
}

void write_correctness(const Options &options, RealInputs &inputs) {
  DeviceDB base, delta, unified;
  base.init(options.gpu, inputs.base, false);
  delta.init(options.gpu, inputs.delta, false);
  unified.init(options.gpu, inputs.fresh, true);
  CustomRuntime custom(options.gpu);
  ReleasedRuntime released(options.gpu, inputs.fresh.rows);
  std::vector<int> query_indices;
  for (int i = 0; i < 64; ++i) query_indices.push_back(i);
  for (int i = 512; i < 768; ++i) query_indices.push_back(i);
  std::ofstream output(options.output);
  output << "variant,query_index,threshold_num,threshold_den,rank,mol_id,coeff_bits,num,den,service_ms,kernel_ms,device_ms\n";
  output << std::setprecision(12);
  std::size_t requests = 0, rows = 0;
  for (const std::string variant : {"released", "flat", "segmented"}) {
    for (int query_index : query_indices) {
      for (const auto [num, den] : std::vector<std::pair<int, int>>{{7, 10}, {4, 5}}) {
        RunResult result = run_variant(
            variant, inputs.queries[query_index], num, den, inputs.base,
            inputs.delta, inputs.fresh, base, delta, unified, custom, released);
        require_no_overflow(result, "correctness " + variant);
        for (std::size_t rank = 0; rank < result.hits.size(); ++rank) {
          const auto &hit = result.hits[rank];
          output << variant << ',' << query_index << ',' << num << ',' << den
                 << ',' << rank << ',' << hit.id << ',' << float_bits(hit.coeff)
                 << ',' << hit.num << ',' << hit.den << ',' << result.service_ms
                 << ',' << result.kernel_ms << ',' << result.device_ms << '\n';
        }
        ++requests;
        rows += result.hits.size();
      }
    }
  }
  std::cout << "requests=" << requests << "\nrows=" << rows << "\n";
}

void write_bench(const Options &options, RealInputs &inputs) {
  if (options.variant != "released" && options.variant != "flat" &&
      options.variant != "segmented")
    throw std::runtime_error("bench requires one variant");
  DeviceDB base, delta, unified;
  const bool released_variant = options.variant == "released";
  if (options.variant == "segmented") {
    base.init(options.gpu, inputs.base, false);
    delta.init(options.gpu, inputs.delta, false);
  } else {
    unified.init(options.gpu, inputs.fresh, released_variant);
  }
  // These runtimes allocate only output state for the inactive comparator;
  // ReleasedRuntime's similarity buffer is needed only for released runs.
  CustomRuntime custom(options.gpu);
  ReleasedRuntime released(options.gpu,
                           released_variant ? inputs.fresh.rows : 1);
  const auto order = threshold_order(options.threshold_order);
  for (int warmup = 0; warmup < options.warmup; ++warmup) {
    const auto [num, den] = order[warmup % order.size()];
    RunResult result = run_variant(
        options.variant, inputs.queries[warmup % 8], num, den, inputs.base,
        inputs.delta, inputs.fresh, base, delta, unified, custom, released);
    require_no_overflow(result, "warmup");
  }
  std::ofstream output(options.output);
  output << "process_index,variant,threshold_num,threshold_den,query_index,service_ms,kernel_ms,device_ms,hit_count,result_hash\n";
  output << std::setprecision(12);
  for (const auto [num, den] : order) {
    for (int query_index = 512; query_index < 768; ++query_index) {
      RunResult result = run_variant(
          options.variant, inputs.queries[query_index], num, den, inputs.base,
          inputs.delta, inputs.fresh, base, delta, unified, custom, released);
      require_no_overflow(result, "bench");
      output << options.process_index << ',' << options.variant << ',' << num
             << ',' << den << ',' << query_index << ',' << result.service_ms
             << ',' << result.kernel_ms << ',' << result.device_ms << ','
             << result.hits.size() << ',' << hash_hits(result.hits) << '\n';
    }
  }
}

double publish_delta(CustomRuntime &runtime, DeviceDB &delta,
                     const HostDBView &host_delta, std::uint64_t epoch) {
  const auto start = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaMemcpyAsync(delta.fp, host_delta.fp,
                             host_delta.rows * 4 * sizeof(std::uint64_t),
                             cudaMemcpyHostToDevice, runtime.stream()));
  CUDA_CHECK(cudaMemcpyAsync(delta.ids, host_delta.ids,
                             host_delta.rows * sizeof(std::uint64_t),
                             cudaMemcpyHostToDevice, runtime.stream()));
  CUDA_CHECK(cudaMemcpyAsync(delta.pc, host_delta.pc,
                             host_delta.rows * sizeof(std::uint16_t),
                             cudaMemcpyHostToDevice, runtime.stream()));
  EpochDescriptor descriptor{epoch, 1u, 0u};
  CUDA_CHECK(cudaMemcpyAsync(runtime.device_epoch(), &descriptor,
                             sizeof(descriptor), cudaMemcpyHostToDevice,
                             runtime.stream()));
  CUDA_CHECK(cudaStreamSynchronize(runtime.stream()));
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

void write_publication(const Options &options, RealInputs &inputs) {
  DeviceDB delta;
  delta.allocate(options.gpu, inputs.delta.rows, false);
  CustomRuntime runtime(options.gpu);
  std::ofstream output(options.output);
  output << "iteration,epoch,publication_ms\n" << std::setprecision(12);
  for (int iteration = 0; iteration < options.cycles; ++iteration) {
    runtime.set_epoch(iteration * 2, false);
    const double elapsed = publish_delta(runtime, delta, inputs.delta,
                                         iteration * 2 + 1);
    output << iteration << ',' << (iteration * 2 + 1) << ',' << elapsed << '\n';
  }
}

void write_stress(const Options &options, RealInputs &inputs) {
  DeviceDB base, delta;
  base.init(options.gpu, inputs.base, false);
  delta.allocate(options.gpu, inputs.delta.rows, false);
  CustomRuntime custom(options.gpu);
  std::array<std::uint64_t, 512> expected_pre{};
  std::array<std::uint64_t, 512> expected_post{};
  std::size_t mismatches = 0, overflow = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int cycle = 0; cycle < options.cycles; ++cycle) {
    const int query_index = 512 + (cycle % 256);
    const int threshold_slot = (cycle / 256) % 2;
    const int num = threshold_slot ? 4 : 7;
    const int den = threshold_slot ? 5 : 10;
    const int slot = threshold_slot * 256 + query_index - 512;
    const Query &query = inputs.queries[query_index];
    const auto [base_begin, base_end] = bounds(inputs.base, query, num, den);
    const auto [delta_begin, delta_end] = bounds(inputs.delta, query, num, den);
    custom.set_epoch(cycle * 2, false);
    RunResult pre = custom.run(base, base_begin, base_end, &delta, delta_begin,
                               delta_end, query, num, den);
    publish_delta(custom, delta, inputs.delta, cycle * 2 + 1);
    RunResult post = custom.run(base, base_begin, base_end, &delta, delta_begin,
                                delta_end, query, num, den);
    overflow += pre.overflow || post.overflow;
    const std::uint64_t pre_hash = hash_hits(pre.hits);
    const std::uint64_t post_hash = hash_hits(post.hits);
    if (!expected_pre[slot]) expected_pre[slot] = pre_hash;
    else if (expected_pre[slot] != pre_hash) ++mismatches;
    if (!expected_post[slot]) expected_post[slot] = post_hash;
    else if (expected_post[slot] != post_hash) ++mismatches;
  }
  const double elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - started)
                             .count();
  std::ofstream output(options.output);
  output << "{\n  \"cycles\": " << options.cycles
         << ",\n  \"hash_mismatches\": " << mismatches
         << ",\n  \"overflow_events\": " << overflow
         << ",\n  \"elapsed_s\": " << std::setprecision(12) << elapsed
         << "\n}\n";
  if (mismatches || overflow) throw std::runtime_error("stress gate failed");
}

void copy_database_slice(DeviceDB &destination, std::uint64_t destination_begin,
                         const DeviceDB &source, std::uint64_t source_begin,
                         std::uint64_t rows, cudaStream_t stream) {
  if (!rows) return;
  CUDA_CHECK(cudaMemcpyAsync(destination.fp + destination_begin * 4,
                             source.fp + source_begin * 4,
                             rows * 4 * sizeof(std::uint64_t),
                             cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(destination.ids + destination_begin,
                             source.ids + source_begin,
                             rows * sizeof(std::uint64_t),
                             cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(destination.pc + destination_begin,
                             source.pc + source_begin,
                             rows * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToDevice, stream));
}

void initialize_untouched_bins(DeviceDB &compacted, const DeviceDB &base,
                               const HostDBView &base_host,
                               const HostDBView &delta_host,
                               const HostDBView &union_host,
                               cudaStream_t stream) {
  for (int popcount = 0; popcount <= 256; ++popcount) {
    const std::uint64_t delta_rows =
        delta_host.cumulative[popcount + 1] - delta_host.cumulative[popcount];
    if (delta_rows) continue;
    const std::uint64_t base_begin = base_host.cumulative[popcount];
    const std::uint64_t base_rows =
        base_host.cumulative[popcount + 1] - base_begin;
    copy_database_slice(compacted, union_host.cumulative[popcount], base,
                        base_begin, base_rows, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

double compact_touched_bins(DeviceDB &compacted, const DeviceDB &base,
                            const DeviceDB &delta,
                            const HostDBView &base_host,
                            const HostDBView &delta_host,
                            const HostDBView &union_host,
                            cudaStream_t stream) {
  const auto started = std::chrono::steady_clock::now();
  for (int popcount = 0; popcount <= 256; ++popcount) {
    const std::uint64_t delta_begin = delta_host.cumulative[popcount];
    const std::uint64_t delta_rows =
        delta_host.cumulative[popcount + 1] - delta_begin;
    if (!delta_rows) continue;
    const std::uint64_t base_begin = base_host.cumulative[popcount];
    const std::uint64_t base_rows =
        base_host.cumulative[popcount + 1] - base_begin;
    const std::uint64_t target_begin = union_host.cumulative[popcount];
    copy_database_slice(compacted, target_begin, base, base_begin, base_rows,
                        stream);
    copy_database_slice(compacted, target_begin + base_rows, delta, delta_begin,
                        delta_rows, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - started)
      .count();
}

void write_sustained(const Options &options, RealInputs &inputs) {
  constexpr int query_count = 1024;
  DeviceDB base, delta, compacted;
  base.init(options.gpu, inputs.base, false);
  delta.allocate(options.gpu, inputs.delta.rows, false);
  compacted.allocate(options.gpu, inputs.fresh.rows, false);
  CustomRuntime custom(options.gpu);
  initialize_untouched_bins(compacted, base, inputs.base, inputs.delta,
                            inputs.fresh, custom.stream());
  publish_delta(custom, delta, inputs.delta, 1);

  std::vector<std::uint64_t> expected_hashes(query_count);
  std::vector<double> query_only_ms;
  query_only_ms.reserve(query_count);
  const auto query_only_started = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < query_count; ++iteration) {
    const int query_index = 512 + iteration % 256;
    const int num = iteration % 2 ? 4 : 7;
    const int den = iteration % 2 ? 5 : 10;
    const Query &query = inputs.queries[query_index];
    const auto [base_begin, base_end] = bounds(inputs.base, query, num, den);
    const auto [delta_begin, delta_end] = bounds(inputs.delta, query, num, den);
    custom.set_epoch(1, true);
    RunResult result = custom.run(base, base_begin, base_end, &delta,
                                  delta_begin, delta_end, query, num, den);
    require_no_overflow(result, "sustained query-only");
    query_only_ms.push_back(result.service_ms);
    expected_hashes[iteration] = hash_hits(result.hits);
  }
  const double query_only_wall_s = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() -
                                       query_only_started)
                                       .count();

  std::vector<double> publication_ms, compaction_ms, mixed_query_ms;
  std::size_t mismatches = 0;
  bool compacted_active = false;
  std::uint64_t epoch = 2;
  const auto mixed_started = std::chrono::steady_clock::now();
  publication_ms.push_back(publish_delta(custom, delta, inputs.delta, epoch++));
  for (int iteration = 0; iteration < query_count; ++iteration) {
    if (iteration > 0 && iteration % 64 == 0) {
      publication_ms.push_back(
          publish_delta(custom, delta, inputs.delta, epoch++));
      compacted_active = false;
    }
    if (iteration > 0 && iteration % 256 == 0) {
      compaction_ms.push_back(compact_touched_bins(
          compacted, base, delta, inputs.base, inputs.delta, inputs.fresh,
          custom.stream()));
      compacted_active = true;
      custom.set_epoch(epoch++, false);
    }
    const int query_index = 512 + iteration % 256;
    const int num = iteration % 2 ? 4 : 7;
    const int den = iteration % 2 ? 5 : 10;
    const Query &query = inputs.queries[query_index];
    RunResult result;
    if (compacted_active) {
      const auto [begin, end] = bounds(inputs.fresh, query, num, den);
      custom.set_epoch(epoch, false);
      result = custom.run(compacted, begin, end, nullptr, 0, 0, query, num, den);
    } else {
      const auto [base_begin, base_end] = bounds(inputs.base, query, num, den);
      const auto [delta_begin, delta_end] = bounds(inputs.delta, query, num, den);
      custom.set_epoch(epoch, true);
      result = custom.run(base, base_begin, base_end, &delta, delta_begin,
                          delta_end, query, num, den);
    }
    require_no_overflow(result, "sustained mixed");
    mixed_query_ms.push_back(result.service_ms);
    if (hash_hits(result.hits) != expected_hashes[iteration]) ++mismatches;
  }
  compaction_ms.push_back(compact_touched_bins(
      compacted, base, delta, inputs.base, inputs.delta, inputs.fresh,
      custom.stream()));
  const double mixed_wall_s = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - mixed_started)
                                  .count();
  const double query_only_qps = query_count / query_only_wall_s;
  const double mixed_qps = query_count / mixed_wall_s;
  const double ratio = mixed_qps / query_only_qps;
  const double max_compaction =
      *std::max_element(compaction_ms.begin(), compaction_ms.end());

  std::ofstream samples(options.output + ".samples.csv");
  samples << "iteration,query_only_ms,mixed_query_ms,result_hash\n"
          << std::setprecision(12);
  for (int iteration = 0; iteration < query_count; ++iteration)
    samples << iteration << ',' << query_only_ms[iteration] << ','
            << mixed_query_ms[iteration] << ',' << expected_hashes[iteration]
            << '\n';
  std::ofstream output(options.output);
  output << "{\n"
         << "  \"queries\": " << query_count << ",\n"
         << "  \"publication_count\": " << publication_ms.size() << ",\n"
         << "  \"compaction_count\": " << compaction_ms.size() << ",\n"
         << "  \"hash_mismatches\": " << mismatches << ",\n"
         << "  \"query_only_wall_s\": " << std::setprecision(12)
         << query_only_wall_s << ",\n"
         << "  \"mixed_wall_s\": " << mixed_wall_s << ",\n"
         << "  \"query_only_qps\": " << query_only_qps << ",\n"
         << "  \"mixed_qps\": " << mixed_qps << ",\n"
         << "  \"mixed_over_query_only\": " << ratio << ",\n"
         << "  \"publication_max_ms\": "
         << *std::max_element(publication_ms.begin(), publication_ms.end())
         << ",\n"
         << "  \"compaction_max_ms\": " << max_compaction << ",\n"
         << "  \"G4_B_SUSTAINED\": "
         << ((mismatches == 0 && ratio >= 0.90 && std::isfinite(max_compaction))
                 ? "true"
                 : "false")
         << "\n}\n";
  if (mismatches) throw std::runtime_error("sustained hash mismatch");
}

std::vector<HostHit> cpu_oracle(const HostDBView &base,
                                const HostDBView *delta, const Query &query,
                                int num, int den) {
  std::vector<HostHit> hits;
  auto scan = [&](const HostDBView &db) {
    const auto [begin, end] = bounds(db, query, num, den);
    for (std::uint64_t row = begin; row < end; ++row) {
      unsigned intersection = 0;
      for (int word = 0; word < 4; ++word)
        intersection += __builtin_popcountll(query.fp[word] & db.fp[row * 4 + word]);
      unsigned union_count = query.pc + db.pc[row] - intersection;
      if (!union_count) union_count = 1;
      if (intersection * den < union_count * num) continue;
      hits.push_back(HostHit{static_cast<std::uint32_t>(db.ids[row]),
                             static_cast<std::uint16_t>(intersection),
                             static_cast<std::uint16_t>(union_count),
                             static_cast<float>(intersection) / union_count});
    }
  };
  scan(base);
  if (delta) scan(*delta);
  sort_hits(hits);
  return hits;
}

void set_bit(std::array<std::uint64_t, 4> &fp, int bit) {
  fp[bit / 64] |= 1ull << (bit % 64);
}

void write_synthetic(const Options &options) {
  constexpr int base_rows = 8192;
  constexpr int delta_rows = 1024;
  std::vector<std::array<std::uint64_t, 4>> raw(base_rows + delta_rows);
  std::mt19937_64 rng(20260827);
  std::array<std::uint64_t, 4> repeated{};
  for (int bit = 0; bit < 64; ++bit) set_bit(repeated, bit);
  for (int row = 0; row < base_rows + delta_rows; ++row) {
    if (row < 512) raw[row] = repeated;
    else
      for (int word = 0; word < 4; ++word) raw[row][word] = rng();
  }
  struct Row { std::array<std::uint64_t, 4> fp; std::uint64_t id; std::uint16_t pc; };
  std::vector<Row> rows;
  rows.reserve(raw.size());
  for (std::size_t row = 0; row < raw.size(); ++row) {
    unsigned pc = 0;
    for (auto word : raw[row]) pc += __builtin_popcountll(word);
    rows.push_back(Row{raw[row], static_cast<std::uint64_t>(row + 1),
                       static_cast<std::uint16_t>(pc)});
  }
  std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) {
    return std::tie(a.pc, a.id) < std::tie(b.pc, b.id);
  });
  OwnedHostDB base_owned, delta_owned, union_owned;
  auto append = [](OwnedHostDB &db, const Row &row) {
    db.ids.push_back(row.id); db.pc.push_back(row.pc);
    db.fp.insert(db.fp.end(), row.fp.begin(), row.fp.end());
  };
  for (const auto &row : rows) {
    append(union_owned, row);
    if (row.id <= base_rows) append(base_owned, row);
    else append(delta_owned, row);
  }
  const HostDBView base = base_owned.view(), delta = delta_owned.view(),
                   unified = union_owned.view();
  DeviceDB d_base, d_delta, d_union;
  d_base.init(options.gpu, base, false);
  d_delta.init(options.gpu, delta, false);
  d_union.init(options.gpu, unified, true);
  CustomRuntime custom(options.gpu);
  ReleasedRuntime released(options.gpu, unified.rows);
  std::vector<Query> queries;
  Query repeated_query{};
  repeated_query.id = 1;
  std::copy(repeated.begin(), repeated.end(), repeated_query.fp);
  repeated_query.pc = 64;
  queries.push_back(repeated_query);
  for (int i = 0; i < 7; ++i) {
    Query query{};
    query.id = 100000 + i;
    for (int word = 0; word < 4; ++word) query.fp[word] = rng();
    unsigned pc = 0;
    for (auto word : query.fp) pc += __builtin_popcountll(word);
    query.pc = pc;
    queries.push_back(query);
  }
  std::size_t mismatches = 0;
  for (const Query &query : queries) {
    for (const auto [num, den] : std::vector<std::pair<int, int>>{{7, 10}, {4, 5}}) {
      const auto oracle = cpu_oracle(base, &delta, query, num, den);
      const auto [bb, be] = bounds(base, query, num, den);
      const auto [db, de] = bounds(delta, query, num, den);
      const auto [ub, ue] = bounds(unified, query, num, den);
      custom.set_epoch(1, true);
      RunResult segmented = custom.run(d_base, bb, be, &d_delta, db, de, query, num, den);
      custom.set_epoch(0, false);
      RunResult flat = custom.run(d_union, ub, ue, nullptr, 0, 0, query, num, den);
      RunResult released_result = released.run(d_union, ub, ue, query, num, den);
      if (hash_hits(oracle) != hash_hits(segmented.hits) ||
          hash_hits(oracle) != hash_hits(flat.hits) ||
          hash_hits(oracle) != hash_hits(released_result.hits) ||
          segmented.overflow || flat.overflow || released_result.overflow)
        ++mismatches;
    }
  }
  CustomRuntime overflow_custom(options.gpu, 128);
  ReleasedRuntime overflow_released(options.gpu, unified.rows, 128);
  const auto [bb, be] = bounds(base, repeated_query, 7, 10);
  const auto [db, de] = bounds(delta, repeated_query, 7, 10);
  const auto [ub, ue] = bounds(unified, repeated_query, 7, 10);
  overflow_custom.set_epoch(1, true);
  RunResult overflow_segmented = overflow_custom.run(
      d_base, bb, be, &d_delta, db, de, repeated_query, 7, 10);
  RunResult overflow_released_result = overflow_released.run(
      d_union, ub, ue, repeated_query, 7, 10);
  const bool overflow_detected = overflow_segmented.overflow &&
                                 overflow_released_result.overflow &&
                                 overflow_segmented.observed_hits >= 512 &&
                                 overflow_released_result.observed_hits >= 512;
  std::ofstream output(options.output);
  output << "{\n  \"queries\": " << queries.size()
         << ",\n  \"thresholds\": 2,\n  \"mismatches\": " << mismatches
         << ",\n  \"overflow_detected\": "
         << (overflow_detected ? "true" : "false")
         << ",\n  \"custom_overflow_count\": "
         << overflow_segmented.observed_hits
         << ",\n  \"released_overflow_count\": "
         << overflow_released_result.observed_hits << "\n}\n";
  if (mismatches || !overflow_detected)
    throw std::runtime_error("synthetic correctness gate failed");
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(options.gpu));
    if (options.mode == "synthetic") {
      write_synthetic(options);
      return 0;
    }
    RealInputs inputs(options);
    if (options.mode == "correctness") write_correctness(options, inputs);
    else if (options.mode == "bench") write_bench(options, inputs);
    else if (options.mode == "publication") write_publication(options, inputs);
    else if (options.mode == "stress") write_stress(options, inputs);
    else if (options.mode == "sustained") write_sustained(options, inputs);
    else throw std::runtime_error("unknown mode: " + options.mode);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
