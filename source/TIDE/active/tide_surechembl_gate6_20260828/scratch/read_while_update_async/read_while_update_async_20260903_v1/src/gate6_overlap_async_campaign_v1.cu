#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int BLOCK = 256;
constexpr int MAX_RUNS = 65;
constexpr std::uint32_t OUTPUT_CAPACITY = 65536;

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    const cudaError_t error__ = (expr);                                          \
    if (error__ != cudaSuccess) {                                                \
      std::ostringstream stream__;                                               \
      stream__ << #expr << " failed: " << cudaGetErrorString(error__);          \
      throw std::runtime_error(stream__.str());                                  \
    }                                                                            \
  } while (0)

struct DeviceHit {
  std::uint64_t id;
  std::uint16_t intersection;
  std::uint16_t union_count;
  std::uint32_t reserved;
};
static_assert(sizeof(DeviceHit) == 16);

struct HostHit {
  std::uint64_t id;
  std::uint16_t intersection;
  std::uint16_t union_count;
};

template <int WORDS>
struct alignas(16) DeviceQuery {
  std::uint64_t fp[WORDS];
  std::uint16_t popcount;
  std::uint16_t threshold_num;
  std::uint16_t threshold_den;
  std::uint16_t reserved;
};

struct DeviceSlice {
  const std::uint64_t *fp;
  const std::uint64_t *ids;
  const std::uint16_t *pc;
  std::uint64_t rows;
  std::uint64_t first_block;
};

template <int WORDS>
__device__ __forceinline__ unsigned intersection_count(
    const DeviceQuery<WORDS> *query, const std::uint64_t *fingerprint) {
  unsigned total = 0;
  if constexpr (WORDS == 4) {
#pragma unroll
    for (int word = 0; word < WORDS; ++word)
      total += __popcll(query->fp[word] & fingerprint[word]);
  } else {
#pragma unroll 1
    for (int word = 0; word < WORDS; ++word)
      total += __popcll(query->fp[word] & fingerprint[word]);
  }
  return total;
}

template <int WORDS>
__global__ void exact_runs_kernel(
    const DeviceSlice *__restrict__ slices, int slice_count,
    const DeviceQuery<WORDS> *__restrict__ query,
    DeviceHit *__restrict__ hits, std::uint32_t capacity,
    std::uint32_t *__restrict__ total_hits,
    std::uint32_t *__restrict__ overflow) {
  const std::uint64_t global_block = blockIdx.x;
  int low = 0;
  int high = slice_count;
  while (low + 1 < high) {
    const int middle = low + (high - low) / 2;
    if (slices[middle].first_block <= global_block)
      low = middle;
    else
      high = middle;
  }
  const DeviceSlice slice = slices[low];
  const std::uint64_t local_block = global_block - slice.first_block;
  const std::uint64_t row = local_block * blockDim.x + threadIdx.x;
  if (row >= slice.rows) return;
  const unsigned intersection =
      intersection_count<WORDS>(query, slice.fp + row * WORDS);
  unsigned union_count = static_cast<unsigned>(query->popcount) +
                         static_cast<unsigned>(slice.pc[row]) - intersection;
  if (union_count == 0) union_count = 1;
  if (intersection * static_cast<unsigned>(query->threshold_den) <
      union_count * static_cast<unsigned>(query->threshold_num))
    return;
  const std::uint32_t slot = atomicAdd(total_hits, 1u);
  if (slot < capacity) {
    hits[slot] = DeviceHit{slice.ids[row],
                           static_cast<std::uint16_t>(intersection),
                           static_cast<std::uint16_t>(union_count), 0};
  } else {
    atomicExch(overflow, 1u);
  }
}

class MappedFile {
 public:
  explicit MappedFile(const std::string &path) : path_(path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("open failed: " + path);
    struct stat state {};
    if (fstat(fd_, &state) != 0)
      throw std::runtime_error("fstat failed: " + path);
    bytes_ = static_cast<std::size_t>(state.st_size);
    if (bytes_ == 0) return;
    data_ = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED)
      throw std::runtime_error("mmap failed: " + path);
  }
  ~MappedFile() {
    if (data_ != MAP_FAILED) munmap(data_, bytes_);
    if (fd_ >= 0) close(fd_);
  }
  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;
  std::size_t bytes() const { return bytes_; }
  template <class T>
  const T *as() const {
    if (bytes_ % sizeof(T))
      throw std::runtime_error("bad element size: " + path_);
    return static_cast<const T *>(data_);
  }

 private:
  std::string path_;
  int fd_ = -1;
  std::size_t bytes_ = 0;
  void *data_ = MAP_FAILED;
};

template <int WORDS>
struct HostDBView {
  const std::uint64_t *fp = nullptr;
  const std::uint64_t *ids = nullptr;
  const std::uint16_t *pc = nullptr;
  std::uint64_t rows = 0;
  std::array<std::uint64_t, WORDS * 64 + 2> cumulative{};
};

template <int WORDS>
HostDBView<WORDS> make_host_view(const MappedFile &fp, const MappedFile &ids,
                                 const MappedFile &pc) {
  HostDBView<WORDS> view;
  view.rows = ids.bytes() / sizeof(std::uint64_t);
  if (fp.bytes() != view.rows * WORDS * sizeof(std::uint64_t) ||
      pc.bytes() != view.rows * sizeof(std::uint16_t))
    throw std::runtime_error("database component size mismatch");
  view.fp = fp.as<std::uint64_t>();
  view.ids = ids.as<std::uint64_t>();
  view.pc = pc.as<std::uint16_t>();
  std::array<std::uint64_t, WORDS * 64 + 1> counts{};
  std::uint16_t previous = 0;
  for (std::uint64_t row = 0; row < view.rows; ++row) {
    const auto value = view.pc[row];
    if (value > WORDS * 64 || (row && value < previous))
      throw std::runtime_error("database is not population-count sorted");
    counts[value]++;
    previous = value;
  }
  for (int value = 0; value <= WORDS * 64; ++value)
    view.cumulative[value + 1] = view.cumulative[value] + counts[value];
  return view;
}

template <int WORDS>
struct OwnedHostDB {
  std::vector<std::uint64_t> fp;
  std::vector<std::uint64_t> ids;
  std::vector<std::uint16_t> pc;
  HostDBView<WORDS> view() const {
    HostDBView<WORDS> result;
    result.fp = fp.data();
    result.ids = ids.data();
    result.pc = pc.data();
    result.rows = ids.size();
    std::array<std::uint64_t, WORDS * 64 + 1> counts{};
    std::uint16_t previous = 0;
    for (std::size_t row = 0; row < pc.size(); ++row) {
      if (pc[row] > WORDS * 64 || (row && pc[row] < previous))
        throw std::runtime_error("owned run is not population-count sorted");
      counts[pc[row]]++;
      previous = pc[row];
    }
    for (int value = 0; value <= WORDS * 64; ++value)
      result.cumulative[value + 1] =
          result.cumulative[value] + counts[value];
    return result;
  }
};

std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

template <int WORDS>
OwnedHostDB<WORDS> load_interleaved(const MappedFile &file) {
  constexpr int ROW_WORDS = WORDS + 2;
  if (file.bytes() % (ROW_WORDS * sizeof(std::uint64_t)))
    throw std::runtime_error("invalid interleaved row file size");
  const std::size_t rows = file.bytes() / (ROW_WORDS * sizeof(std::uint64_t));
  const auto *input = file.as<std::uint64_t>();
  OwnedHostDB<WORDS> output;
  output.fp.resize(rows * WORDS);
  output.ids.resize(rows);
  output.pc.resize(rows);
  std::uint16_t previous = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    output.ids[row] = input[row * ROW_WORDS];
    unsigned actual = 0;
    for (int word = 0; word < WORDS; ++word) {
      const auto value = input[row * ROW_WORDS + 1 + word];
      output.fp[row * WORDS + word] = value;
      actual += __builtin_popcountll(value);
    }
    output.pc[row] = static_cast<std::uint16_t>(input[row * ROW_WORDS + WORDS + 1]);
    if (actual != output.pc[row] || (row && output.pc[row] < previous))
      throw std::runtime_error("interleaved row validation failed");
    previous = output.pc[row];
  }
  return output;
}

template <int WORDS>
std::vector<OwnedHostDB<WORDS>> split_runs(const HostDBView<WORDS> &delta,
                                           int run_count) {
  struct RowRef {
    std::uint64_t source;
    std::uint64_t id;
    std::uint16_t pc;
  };
  std::vector<std::vector<RowRef>> buckets(run_count);
  for (std::uint64_t row = 0; row < delta.rows; ++row)
    buckets[splitmix64(delta.ids[row]) % run_count].push_back(
        RowRef{row, delta.ids[row], delta.pc[row]});
  std::vector<OwnedHostDB<WORDS>> output(run_count);
  for (int run = 0; run < run_count; ++run) {
    auto &rows = buckets[run];
    std::sort(rows.begin(), rows.end(), [](const RowRef &left, const RowRef &right) {
      return std::tie(left.pc, left.id) < std::tie(right.pc, right.id);
    });
    auto &target = output[run];
    target.fp.reserve(rows.size() * WORDS);
    target.ids.reserve(rows.size());
    target.pc.reserve(rows.size());
    for (const auto &row : rows) {
      target.ids.push_back(row.id);
      target.pc.push_back(row.pc);
      target.fp.insert(target.fp.end(), delta.fp + row.source * WORDS,
                       delta.fp + (row.source + 1) * WORDS);
    }
  }
  return output;
}

template <int WORDS>
struct Query {
  std::uint64_t id = 0;
  std::array<std::uint64_t, WORDS> fp{};
  std::uint16_t pc = 0;
};

template <int WORDS>
class QueryStore {
 public:
  explicit QueryStore(const MappedFile &file) {
    constexpr int ROW_WORDS = WORDS + 2;
    if (file.bytes() % (ROW_WORDS * sizeof(std::uint64_t)))
      throw std::runtime_error("invalid query file size");
    const auto *input = file.as<std::uint64_t>();
    const std::size_t rows = file.bytes() / (ROW_WORDS * sizeof(std::uint64_t));
    queries_.resize(rows);
    for (std::size_t row = 0; row < rows; ++row) {
      auto &query = queries_[row];
      query.id = input[row * ROW_WORDS];
      unsigned actual = 0;
      for (int word = 0; word < WORDS; ++word) {
        query.fp[word] = input[row * ROW_WORDS + 1 + word];
        actual += __builtin_popcountll(query.fp[word]);
      }
      query.pc = static_cast<std::uint16_t>(input[row * ROW_WORDS + WORDS + 1]);
      if (actual != query.pc) throw std::runtime_error("query popcount mismatch");
    }
  }
  const Query<WORDS> &operator[](std::size_t index) const {
    return queries_.at(index);
  }
  std::size_t size() const { return queries_.size(); }

 private:
  std::vector<Query<WORDS>> queries_;
};

template <int WORDS>
std::pair<int, int> popcount_interval(std::uint16_t query_pc, int num,
                                      int den) {
  const int lower = (num * static_cast<int>(query_pc) + den - 1) / den;
  const int upper = den * static_cast<int>(query_pc) / num;
  return {std::max(0, lower), std::min(WORDS * 64, upper)};
}

template <int WORDS>
std::pair<std::uint64_t, std::uint64_t> bounds(
    const HostDBView<WORDS> &database, const Query<WORDS> &query, int num,
    int den) {
  const auto interval = popcount_interval<WORDS>(query.pc, num, den);
  return {database.cumulative[interval.first],
          database.cumulative[interval.second + 1]};
}

struct CudaStreamOwner {
  explicit CudaStreamOwner(int target_gpu) : gpu(target_gpu) {
    CUDA_CHECK(cudaSetDevice(gpu));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  ~CudaStreamOwner() {
    if (!stream) return;
    cudaSetDevice(gpu);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }
  CudaStreamOwner(const CudaStreamOwner &) = delete;
  CudaStreamOwner &operator=(const CudaStreamOwner &) = delete;
  cudaStream_t get() const { return stream; }

 private:
  int gpu = -1;
  cudaStream_t stream = nullptr;
};

template <int WORDS>
struct DeviceRun {
  std::uint64_t *fp = nullptr;
  std::uint64_t *ids = nullptr;
  std::uint16_t *pc = nullptr;
  std::uint64_t rows = 0;
  int gpu = -1;
  cudaStream_t allocation_stream = nullptr;
  bool async_owned = false;
  DeviceRun(int target_gpu, const HostDBView<WORDS> &host)
      : rows(host.rows), gpu(target_gpu) {
    CUDA_CHECK(cudaSetDevice(gpu));
    if (!rows) return;
    CUDA_CHECK(cudaMalloc(&fp, rows * WORDS * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&ids, rows * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&pc, rows * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMemcpy(fp, host.fp, rows * WORDS * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ids, host.ids, rows * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pc, host.pc, rows * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
  }
  DeviceRun(int target_gpu, const HostDBView<WORDS> &host,
            cudaStream_t upload_stream)
      : rows(host.rows),
        gpu(target_gpu),
        allocation_stream(upload_stream),
        async_owned(true) {
    CUDA_CHECK(cudaSetDevice(gpu));
    if (!rows) return;
    try {
      CUDA_CHECK(cudaMallocAsync(&fp, rows * WORDS * sizeof(std::uint64_t),
                                 allocation_stream));
      CUDA_CHECK(cudaMallocAsync(&ids, rows * sizeof(std::uint64_t),
                                 allocation_stream));
      CUDA_CHECK(cudaMallocAsync(&pc, rows * sizeof(std::uint16_t),
                                 allocation_stream));
      CUDA_CHECK(cudaMemcpyAsync(fp, host.fp,
                                 rows * WORDS * sizeof(std::uint64_t),
                                 cudaMemcpyHostToDevice, allocation_stream));
      CUDA_CHECK(cudaMemcpyAsync(ids, host.ids,
                                 rows * sizeof(std::uint64_t),
                                 cudaMemcpyHostToDevice, allocation_stream));
      CUDA_CHECK(cudaMemcpyAsync(pc, host.pc, rows * sizeof(std::uint16_t),
                                 cudaMemcpyHostToDevice, allocation_stream));
      CUDA_CHECK(cudaStreamSynchronize(allocation_stream));
    } catch (...) {
      if (fp) cudaFreeAsync(fp, allocation_stream);
      if (ids) cudaFreeAsync(ids, allocation_stream);
      if (pc) cudaFreeAsync(pc, allocation_stream);
      cudaStreamSynchronize(allocation_stream);
      fp = nullptr;
      ids = nullptr;
      pc = nullptr;
      throw;
    }
  }
  ~DeviceRun() {
    if (gpu < 0) return;
    cudaSetDevice(gpu);
    if (async_owned) {
      if (fp) cudaFreeAsync(fp, allocation_stream);
      if (ids) cudaFreeAsync(ids, allocation_stream);
      if (pc) cudaFreeAsync(pc, allocation_stream);
    } else {
      if (fp) cudaFree(fp);
      if (ids) cudaFree(ids);
      if (pc) cudaFree(pc);
    }
  }
  DeviceRun(const DeviceRun &) = delete;
  DeviceRun &operator=(const DeviceRun &) = delete;
};

template <int WORDS>
struct SnapshotEntry {
  std::shared_ptr<DeviceRun<WORDS>> device;
  HostDBView<WORDS> host;
};

template <int WORDS>
struct Snapshot {
  std::uint64_t epoch = 0;
  std::vector<SnapshotEntry<WORDS>> runs;
  std::vector<std::shared_ptr<OwnedHostDB<WORDS>>> host_owners;
};

template <int WORDS>
std::vector<DeviceSlice> make_slices(const Snapshot<WORDS> &snapshot,
                                     const Query<WORDS> &query, int num,
                                     int den, bool bounded,
                                     std::uint64_t *candidate_rows) {
  std::vector<DeviceSlice> slices;
  std::uint64_t first_block = 0;
  std::uint64_t candidates = 0;
  for (const auto &run : snapshot.runs) {
    const auto range = bounded
                           ? bounds<WORDS>(run.host, query, num, den)
                           : std::pair<std::uint64_t, std::uint64_t>{0,
                                                                    run.host.rows};
    const std::uint64_t rows = range.second - range.first;
    if (!rows) continue;
    slices.push_back(DeviceSlice{run.device->fp + range.first * WORDS,
                                 run.device->ids + range.first,
                                 run.device->pc + range.first, rows,
                                 first_block});
    first_block += (rows + BLOCK - 1) / BLOCK;
    candidates += rows;
  }
  *candidate_rows = candidates;
  return slices;
}

void sort_hits(std::vector<HostHit> &hits) {
  std::sort(hits.begin(), hits.end(), [](const HostHit &left, const HostHit &right) {
    if (left.id != right.id) return left.id < right.id;
    if (left.intersection != right.intersection)
      return left.intersection < right.intersection;
    return left.union_count < right.union_count;
  });
}

std::uint64_t hash_hits(const std::vector<HostHit> &hits) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const auto &hit : hits) {
    const std::uint64_t values[2] = {
        hit.id, static_cast<std::uint64_t>(hit.intersection) |
                    (static_cast<std::uint64_t>(hit.union_count) << 16)};
    const auto *bytes = reinterpret_cast<const unsigned char *>(values);
    for (std::size_t index = 0; index < sizeof(values); ++index) {
      hash ^= bytes[index];
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

std::uint64_t hash_hit_ids(const std::vector<HostHit> &hits) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const auto &hit : hits) {
    const auto *bytes = reinterpret_cast<const unsigned char *>(&hit.id);
    for (std::size_t index = 0; index < sizeof(hit.id); ++index) {
      hash ^= bytes[index];
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

struct RunResult {
  double service_ms = 0;
  float kernel_ms = 0;
  std::uint64_t candidate_rows = 0;
  std::uint32_t observed_hits = 0;
  bool overflow = false;
  std::vector<HostHit> hits;
  std::uint64_t result_hash = 0;
};

template <int WORDS>
class QueryRuntime {
 public:
  explicit QueryRuntime(int gpu) : gpu_(gpu) {
    CUDA_CHECK(cudaSetDevice(gpu_));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&kernel_start_));
    CUDA_CHECK(cudaEventCreate(&kernel_stop_));
    CUDA_CHECK(cudaMalloc(&d_query_, sizeof(DeviceQuery<WORDS>)));
    CUDA_CHECK(cudaMalloc(&d_slices_, MAX_RUNS * sizeof(DeviceSlice)));
    CUDA_CHECK(cudaMalloc(&d_hits_, OUTPUT_CAPACITY * sizeof(DeviceHit)));
    CUDA_CHECK(cudaMalloc(&d_count_, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_overflow_, sizeof(std::uint32_t)));
  }
  ~QueryRuntime() {
    cudaSetDevice(gpu_);
    cudaStreamSynchronize(stream_);
    if (d_query_) cudaFree(d_query_);
    if (d_slices_) cudaFree(d_slices_);
    if (d_hits_) cudaFree(d_hits_);
    if (d_count_) cudaFree(d_count_);
    if (d_overflow_) cudaFree(d_overflow_);
    if (kernel_start_) cudaEventDestroy(kernel_start_);
    if (kernel_stop_) cudaEventDestroy(kernel_stop_);
    if (stream_) cudaStreamDestroy(stream_);
  }
  RunResult run(const Snapshot<WORDS> &snapshot, const Query<WORDS> &query,
                int num, int den, bool bounded) {
    CUDA_CHECK(cudaSetDevice(gpu_));
    const auto begin = std::chrono::steady_clock::now();
    RunResult result;
    auto slices = make_slices<WORDS>(snapshot, query, num, den, bounded,
                                     &result.candidate_rows);
    if (slices.empty()) {
      result.result_hash = hash_hits(result.hits);
      result.service_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - begin)
              .count();
      return result;
    }
    if (slices.size() > MAX_RUNS) throw std::runtime_error("too many slices");
    const auto &last = slices.back();
    const std::uint64_t blocks =
        last.first_block + (last.rows + BLOCK - 1) / BLOCK;
    if (blocks > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("grid too large");
    DeviceQuery<WORDS> device_query{};
    std::copy(query.fp.begin(), query.fp.end(), device_query.fp);
    device_query.popcount = query.pc;
    device_query.threshold_num = static_cast<std::uint16_t>(num);
    device_query.threshold_den = static_cast<std::uint16_t>(den);
    std::uint32_t count = 0;
    std::uint32_t overflow = 0;
    CUDA_CHECK(cudaMemcpyAsync(d_query_, &device_query, sizeof(device_query),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_slices_, slices.data(),
                               slices.size() * sizeof(DeviceSlice),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemsetAsync(d_count_, 0, sizeof(std::uint32_t), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_overflow_, 0, sizeof(std::uint32_t), stream_));
    CUDA_CHECK(cudaEventRecord(kernel_start_, stream_));
    exact_runs_kernel<WORDS><<<static_cast<unsigned>(blocks), BLOCK, 0, stream_>>>(
        d_slices_, static_cast<int>(slices.size()), d_query_, d_hits_,
        OUTPUT_CAPACITY, d_count_, d_overflow_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernel_stop_, stream_));
    CUDA_CHECK(cudaMemcpyAsync(&count, d_count_, sizeof(count),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(&overflow, d_overflow_, sizeof(overflow),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    result.observed_hits = count;
    result.overflow = overflow != 0 || count > OUTPUT_CAPACITY;
    if (!result.overflow && count) {
      std::vector<DeviceHit> device_hits(count);
      CUDA_CHECK(cudaMemcpyAsync(device_hits.data(), d_hits_,
                                 count * sizeof(DeviceHit),
                                 cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      result.hits.reserve(count);
      for (const auto &hit : device_hits)
        result.hits.push_back(
            HostHit{hit.id, hit.intersection, hit.union_count});
      sort_hits(result.hits);
    }
    result.result_hash = hash_hits(result.hits);
    result.service_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin)
            .count();
    CUDA_CHECK(cudaEventElapsedTime(&result.kernel_ms, kernel_start_,
                                    kernel_stop_));
    return result;
  }

 private:
  int gpu_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t kernel_start_ = nullptr;
  cudaEvent_t kernel_stop_ = nullptr;
  DeviceQuery<WORDS> *d_query_ = nullptr;
  DeviceSlice *d_slices_ = nullptr;
  DeviceHit *d_hits_ = nullptr;
  std::uint32_t *d_count_ = nullptr;
  std::uint32_t *d_overflow_ = nullptr;
};

template <int WORDS>
std::vector<HostHit> cpu_scan(const HostDBView<WORDS> &database,
                              const Query<WORDS> &query, int num, int den) {
  std::vector<HostHit> hits;
  const auto range = bounds<WORDS>(database, query, num, den);
  for (std::uint64_t row = range.first; row < range.second; ++row) {
    unsigned intersection = 0;
    for (int word = 0; word < WORDS; ++word)
      intersection += __builtin_popcountll(
          query.fp[word] & database.fp[row * WORDS + word]);
    unsigned union_count = query.pc + database.pc[row] - intersection;
    if (union_count == 0) union_count = 1;
    if (intersection * den < union_count * num) continue;
    hits.push_back(HostHit{database.ids[row],
                           static_cast<std::uint16_t>(intersection),
                           static_cast<std::uint16_t>(union_count)});
  }
  sort_hits(hits);
  return hits;
}

struct Options {
  std::string root;
  std::string mode = "correctness";
  std::string output;
  std::string order = "compacted,logical64,unbounded64";
  std::string threshold_order = "70,80";
  int gpu = 2;
  int words = 4;
  int query_limit = 512;
  int sustained_requests = 4096;
  int update_warmups = 16;
  int update_requests = 512;
  std::string oracle;
  int reader_requests = 4096;
  int formal_overlap_gates = 1;
  int update_after_completions = 0;
};

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    if (index + 1 >= argc) throw std::runtime_error("missing option value: " + key);
    const std::string value = argv[++index];
    if (key == "--root") options.root = value;
    else if (key == "--mode") options.mode = value;
    else if (key == "--output") options.output = value;
    else if (key == "--order") options.order = value;
    else if (key == "--threshold-order") options.threshold_order = value;
    else if (key == "--gpu") options.gpu = std::stoi(value);
    else if (key == "--words") options.words = std::stoi(value);
    else if (key == "--query-limit") options.query_limit = std::stoi(value);
    else if (key == "--sustained-requests")
      options.sustained_requests = std::stoi(value);
    else if (key == "--update-warmups")
      options.update_warmups = std::stoi(value);
    else if (key == "--update-requests")
      options.update_requests = std::stoi(value);
    else if (key == "--oracle") options.oracle = value;
    else if (key == "--reader-requests")
      options.reader_requests = std::stoi(value);
    else if (key == "--formal-overlap-gates")
      options.formal_overlap_gates = std::stoi(value);
    else if (key == "--update-after-completions")
      options.update_after_completions = std::stoi(value);
    else throw std::runtime_error("unknown option: " + key);
  }
  if (options.root.empty() || options.output.empty())
    throw std::runtime_error("--root and --output are required");
  if (options.words != 4 && options.words != 32)
    throw std::runtime_error("--words must be 4 or 32");
  if (options.mode != "correctness" && options.mode != "benchmark" &&
      options.mode != "sustained" && options.mode != "update" &&
      options.mode != "overlap" && options.mode != "control")
    throw std::runtime_error(
        "--mode must be correctness, benchmark, sustained, update, overlap, or "
        "control");
  if (options.sustained_requests <= 0)
    throw std::runtime_error("--sustained-requests must be positive");
  if (options.update_warmups < 0 || options.update_requests <= 0)
    throw std::runtime_error("invalid update warmup/request count");
  if ((options.mode == "overlap" || options.mode == "control") &&
      options.oracle.empty())
    throw std::runtime_error("--oracle is required for overlap/control mode");
  if (options.reader_requests <= 0)
    throw std::runtime_error("--reader-requests must be positive");
  if ((options.formal_overlap_gates != 0 &&
       options.formal_overlap_gates != 1) ||
      options.update_after_completions < 0 ||
      options.update_after_completions >= options.reader_requests)
    throw std::runtime_error("invalid overlap gate/completion options");
  return options;
}

std::vector<std::string> split_text(const std::string &text) {
  std::vector<std::string> result;
  std::size_t start = 0;
  while (start <= text.size()) {
    const auto comma = text.find(',', start);
    result.push_back(text.substr(start, comma - start));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return result;
}

std::vector<std::pair<int, int>> thresholds(const std::string &order) {
  if (order == "70,80") return {{7, 10}, {4, 5}};
  if (order == "80,70") return {{4, 5}, {7, 10}};
  throw std::runtime_error("invalid threshold order");
}

template <int WORDS>
struct Inputs {
  MappedFile base_fp;
  MappedFile base_ids;
  MappedFile base_pc;
  MappedFile union_fp;
  MappedFile union_ids;
  MappedFile union_pc;
  MappedFile delta_file;
  MappedFile query_file;
  HostDBView<WORDS> base;
  HostDBView<WORDS> fresh;
  OwnedHostDB<WORDS> delta_owned;
  HostDBView<WORDS> delta;
  QueryStore<WORDS> queries;
  explicit Inputs(const std::string &root)
      : base_fp(root + "/data/gate0_prepared/base_fp_u64x" +
                std::to_string(WORDS) + ".bin"),
        base_ids(root + "/data/gate0_prepared/base_id_i64.bin"),
        base_pc(root + "/data/gate0_prepared/base_popcnt_u16.bin"),
        union_fp(root + "/data/gate0_prepared/union_fp_u64x" +
                 std::to_string(WORDS) + ".bin"),
        union_ids(root + "/data/gate0_prepared/union_id_i64.bin"),
        union_pc(root + "/data/gate0_prepared/union_popcnt_u16.bin"),
        delta_file(root + "/data/stage_a/delta_u64x" +
                   std::to_string(WORDS + 2) + ".bin"),
        query_file(root + "/data/stage_a/queries_u64x" +
                   std::to_string(WORDS + 2) + ".bin"),
        base(make_host_view<WORDS>(base_fp, base_ids, base_pc)),
        fresh(make_host_view<WORDS>(union_fp, union_ids, union_pc)),
        delta_owned(load_interleaved<WORDS>(delta_file)),
        delta(delta_owned.view()),
        queries(query_file) {}
};

template <int WORDS>
struct LogicalIndex {
  std::vector<OwnedHostDB<WORDS>> owned;
  std::vector<HostDBView<WORDS>> host;
  std::vector<std::shared_ptr<DeviceRun<WORDS>>> device;
  Snapshot<WORDS> snapshot;
};

template <int WORDS>
LogicalIndex<WORDS> build_logical(
    int gpu, const HostDBView<WORDS> &base,
    const std::shared_ptr<DeviceRun<WORDS>> &base_device,
    const HostDBView<WORDS> &delta, int layout) {
  LogicalIndex<WORDS> result;
  result.owned = split_runs<WORDS>(delta, layout);
  result.snapshot.runs.push_back({base_device, base});
  for (int run = 0; run < layout; ++run) {
    result.host.push_back(result.owned[run].view());
    result.device.push_back(
        std::make_shared<DeviceRun<WORDS>>(gpu, result.host.back()));
    result.snapshot.runs.push_back({result.device.back(), result.host.back()});
  }
  return result;
}

template <int WORDS>
bool equal_result(const RunResult &result, const std::vector<HostHit> &oracle) {
  return !result.overflow && result.hits.size() == oracle.size() &&
         result.result_hash == hash_hits(oracle);
}

template <int WORDS>
void run_correctness(const Options &options, Inputs<WORDS> &inputs) {
  if (inputs.base.rows + inputs.delta.rows != inputs.fresh.rows)
    throw std::runtime_error("insert-only row balance failed");
  auto base_device =
      std::make_shared<DeviceRun<WORDS>>(options.gpu, inputs.base);
  auto union_device =
      std::make_shared<DeviceRun<WORDS>>(options.gpu, inputs.fresh);
  Snapshot<WORDS> compacted;
  compacted.runs.push_back({union_device, inputs.fresh});
  const std::array<int, 4> layouts{{1, 4, 16, 64}};
  std::vector<LogicalIndex<WORDS>> logical;
  for (const int layout : layouts)
    logical.push_back(build_logical<WORDS>(options.gpu, inputs.base, base_device,
                                           inputs.delta, layout));
  QueryRuntime<WORDS> runtime(options.gpu);
  const int query_count =
      std::min<int>(options.query_limit, static_cast<int>(inputs.queries.size()));
  std::ofstream samples(options.output + ".samples.csv");
  samples << "query,threshold_num,threshold_den,variant,candidates,hits,hash,"
             "oracle_hits,oracle_hash,overflow,match\n";
  std::uint64_t checks = 0;
  std::uint64_t mismatches = 0;
  std::uint64_t candidate_mismatches = 0;
  for (int query_index = 0; query_index < query_count; ++query_index) {
    const auto &query = inputs.queries[query_index];
    for (const auto &[num, den] :
         std::array<std::pair<int, int>, 2>{{{7, 10}, {4, 5}}}) {
      const auto oracle = cpu_scan<WORDS>(inputs.fresh, query, num, den);
      const auto oracle_hash = hash_hits(oracle);
      const auto keeper = runtime.run(compacted, query, num, den, true);
      bool match = equal_result<WORDS>(keeper, oracle);
      checks++;
      mismatches += !match;
      samples << query_index << ',' << num << ',' << den << ",compacted,"
              << keeper.candidate_rows << ',' << keeper.hits.size() << ','
              << keeper.result_hash << ',' << oracle.size() << ',' << oracle_hash
              << ',' << keeper.overflow << ',' << match << '\n';
      for (std::size_t slot = 0; slot < logical.size(); ++slot) {
        const auto bounded =
            runtime.run(logical[slot].snapshot, query, num, den, true);
        match = equal_result<WORDS>(bounded, oracle);
        checks++;
        mismatches += !match;
        candidate_mismatches += bounded.candidate_rows != keeper.candidate_rows;
        samples << query_index << ',' << num << ',' << den << ",logical"
                << layouts[slot] << ',' << bounded.candidate_rows << ','
                << bounded.hits.size() << ',' << bounded.result_hash << ','
                << oracle.size() << ',' << oracle_hash << ',' << bounded.overflow
                << ',' << match << '\n';
        const auto unbounded =
            runtime.run(logical[slot].snapshot, query, num, den, false);
        match = equal_result<WORDS>(unbounded, oracle);
        checks++;
        mismatches += !match;
        samples << query_index << ',' << num << ',' << den << ",unbounded"
                << layouts[slot] << ',' << unbounded.candidate_rows << ','
                << unbounded.hits.size() << ',' << unbounded.result_hash << ','
                << oracle.size() << ',' << oracle_hash << ',' << unbounded.overflow
                << ',' << match << '\n';
      }
    }
  }
  const bool pass = mismatches == 0 && candidate_mismatches == 0 &&
                    checks == static_cast<std::uint64_t>(query_count) * 2 * 9;
  std::ofstream output(options.output);
  output << "{\n"
         << "  \"experiment_id\": \"tide_20260828_gate6_exact_query\",\n"
         << "  \"word_count\": " << WORDS << ",\n"
         << "  \"query_count\": " << query_count << ",\n"
         << "  \"checks\": " << checks << ",\n"
         << "  \"result_mismatches\": " << mismatches << ",\n"
         << "  \"candidate_count_mismatches\": " << candidate_mismatches
         << ",\n"
         << "  \"G6_EXACT_COMPONENT\": " << (pass ? "true" : "false")
         << "\n}\n";
  if (!pass) throw std::runtime_error("Gate-6 exact query component failed");
}

template <int WORDS>
void run_benchmark(
    const Options &options, Inputs<WORDS> &inputs,
    const std::chrono::steady_clock::time_point setup_begin) {
  std::size_t free_before = 0;
  std::size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_before, &total_bytes));
  auto base_device =
      std::make_shared<DeviceRun<WORDS>>(options.gpu, inputs.base);
  auto union_device =
      std::make_shared<DeviceRun<WORDS>>(options.gpu, inputs.fresh);
  Snapshot<WORDS> compacted;
  compacted.runs.push_back({union_device, inputs.fresh});
  auto logical = build_logical<WORDS>(options.gpu, inputs.base, base_device,
                                      inputs.delta, 64);
  QueryRuntime<WORDS> runtime(options.gpu);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::size_t free_after = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_after, &total_bytes));
  const double setup_s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - setup_begin)
                             .count();
  std::ofstream setup(options.output + ".setup.json");
  setup << "{\n"
        << "  \"experiment_id\": "
           "\"tide_20260829_gate6_external_tide_setup\",\n"
        << "  \"setup_scope\": \"input mapping through resident device-run "
           "and runtime construction plus synchronization\",\n"
        << "  \"setup_s\": " << std::setprecision(17) << setup_s << ",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"cuda_free_before_bytes\": " << free_before << ",\n"
        << "  \"cuda_free_after_bytes\": " << free_after << ",\n"
        << "  \"cuda_used_delta_bytes\": "
        << (free_before >= free_after ? free_before - free_after : 0) << ",\n"
        << "  \"cuda_total_bytes\": " << total_bytes << "\n"
        << "}\n";
  const int query_count =
      std::min<int>(options.query_limit, static_cast<int>(inputs.queries.size()));
  std::ofstream samples(options.output);
  samples << "variant,threshold_num,threshold_den,query,service_ms,kernel_ms,"
             "candidate_rows,hits,result_hash,overflow\n";
  for (const auto &[num, den] : thresholds(options.threshold_order)) {
    for (const auto &variant : split_text(options.order)) {
      const bool bounded = variant != "unbounded64";
      const Snapshot<WORDS> *snapshot =
          variant == "compacted" ? &compacted : &logical.snapshot;
      if (variant != "compacted" && variant != "logical64" &&
          variant != "unbounded64")
        throw std::runtime_error("invalid benchmark variant: " + variant);
      for (int warmup = 0; warmup < 16; ++warmup)
        runtime.run(*snapshot, inputs.queries[warmup % query_count], num, den,
                    bounded);
      for (int query_index = 0; query_index < query_count; ++query_index) {
        const auto result = runtime.run(*snapshot, inputs.queries[query_index],
                                        num, den, bounded);
        if (result.overflow) throw std::runtime_error("benchmark output overflow");
        samples << variant << ',' << num << ',' << den << ',' << query_index
                << ',' << std::setprecision(12) << result.service_ms << ','
                << result.kernel_ms << ',' << result.candidate_rows << ','
                << result.hits.size() << ',' << result.result_hash << ','
                << result.overflow << '\n';
      }
    }
  }
}

struct SustainedSample {
  int reader = -1;
  int cycle = -1;
  int query = -1;
  int numerator = 0;
  int denominator = 0;
  double service_ms = 0;
  float kernel_ms = 0;
  std::uint64_t candidate_rows = 0;
  std::size_t hits = 0;
  std::uint64_t result_hash = 0;
  bool overflow = false;
  bool completed = false;
};

void barrier_wait(pthread_barrier_t *barrier) {
  const int rc = pthread_barrier_wait(barrier);
  if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
    throw std::runtime_error("pthread_barrier_wait failed");
}

template <int WORDS>
void run_sustained(
    const Options &options, Inputs<WORDS> &inputs,
    const std::chrono::steady_clock::time_point setup_begin) {
  constexpr int READERS = 4;
  std::size_t free_before = 0;
  std::size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_before, &total_bytes));
  auto base_device =
      std::make_shared<DeviceRun<WORDS>>(options.gpu, inputs.base);
  auto union_device =
      std::make_shared<DeviceRun<WORDS>>(options.gpu, inputs.fresh);
  Snapshot<WORDS> compacted;
  compacted.runs.push_back({union_device, inputs.fresh});
  auto logical = build_logical<WORDS>(options.gpu, inputs.base, base_device,
                                      inputs.delta, 64);
  std::array<std::unique_ptr<QueryRuntime<WORDS>>, READERS> runtimes;
  for (auto &runtime : runtimes)
    runtime = std::make_unique<QueryRuntime<WORDS>>(options.gpu);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::size_t free_after = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_after, &total_bytes));
  const double setup_s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - setup_begin)
                             .count();
  std::ofstream setup(options.output + ".setup.json");
  setup << "{\n"
        << "  \"experiment_id\": "
           "\"tide_20260829_gate6_sustained_setup\",\n"
        << "  \"setup_scope\": \"input mapping, shared resident layouts, "
           "four private runtimes, and synchronization\",\n"
        << "  \"setup_s\": " << std::setprecision(17) << setup_s << ",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"reader_streams\": " << READERS << ",\n"
        << "  \"cuda_free_before_bytes\": " << free_before << ",\n"
        << "  \"cuda_free_after_bytes\": " << free_after << ",\n"
        << "  \"cuda_used_delta_bytes\": "
        << (free_before >= free_after ? free_before - free_after : 0) << ",\n"
        << "  \"cuda_total_bytes\": " << total_bytes << "\n"
        << "}\n";

  const int query_count =
      std::min<int>(options.query_limit, static_cast<int>(inputs.queries.size()));
  if (query_count <= 0) throw std::runtime_error("empty query set");
  const int unique_requests = query_count * 2;
  if (options.sustained_requests % unique_requests != 0)
    throw std::runtime_error(
        "--sustained-requests must be a multiple of query_count * 2");
  const auto threshold_order = thresholds(options.threshold_order);
  const auto variants = split_text(options.order);
  if (variants.size() != 1 ||
      (variants.front() != "compacted" && variants.front() != "logical64" &&
       variants.front() != "unbounded64"))
    throw std::runtime_error(
        "sustained mode requires exactly one of compacted, logical64, or "
        "unbounded64");
  const std::string variant = variants.front();
  const Snapshot<WORDS> *snapshot =
      variant == "compacted" ? &compacted : &logical.snapshot;
  const bool bounded = variant != "unbounded64";
  std::vector<SustainedSample> samples(options.sustained_requests);
  std::array<std::exception_ptr, READERS> errors{};
  pthread_barrier_t ready_barrier{};
  pthread_barrier_t start_barrier{};
  if (pthread_barrier_init(&ready_barrier, nullptr, READERS + 1) != 0 ||
      pthread_barrier_init(&start_barrier, nullptr, READERS + 1) != 0)
    throw std::runtime_error("pthread_barrier_init failed");

  std::array<std::thread, READERS> workers;
  for (int reader = 0; reader < READERS; ++reader) {
    workers[reader] = std::thread([&, reader] {
      try {
        CUDA_CHECK(cudaSetDevice(options.gpu));
        for (const auto &[num, den] : threshold_order) {
          for (int warmup = 0; warmup < 16; ++warmup) {
            const int query = (warmup * READERS + reader) % query_count;
            const auto result = runtimes[reader]->run(
                *snapshot, inputs.queries[query], num, den, bounded);
            if (result.overflow)
              throw std::runtime_error("sustained warmup overflow");
          }
        }
      } catch (...) {
        errors[reader] = std::current_exception();
      }
      try {
        barrier_wait(&ready_barrier);
        barrier_wait(&start_barrier);
        if (errors[reader]) return;
        for (int task = reader; task < options.sustained_requests;
             task += READERS) {
          const int within = task % unique_requests;
          const int threshold_slot = within / query_count;
          const int query = within % query_count;
          const auto [num, den] = threshold_order[threshold_slot];
          const auto result = runtimes[reader]->run(
              *snapshot, inputs.queries[query], num, den, bounded);
          if (result.overflow)
            throw std::runtime_error("sustained retained request overflow");
          samples[task] = SustainedSample{
              reader,
              task / unique_requests,
              query,
              num,
              den,
              result.service_ms,
              result.kernel_ms,
              result.candidate_rows,
              result.hits.size(),
              result.result_hash,
              result.overflow,
              true};
        }
      } catch (...) {
        errors[reader] = std::current_exception();
      }
    });
  }
  barrier_wait(&ready_barrier);
  const auto retained_begin = std::chrono::steady_clock::now();
  barrier_wait(&start_barrier);
  for (auto &worker : workers) worker.join();
  const double retained_wall_s = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() -
                                     retained_begin)
                                     .count();
  pthread_barrier_destroy(&ready_barrier);
  pthread_barrier_destroy(&start_barrier);
  for (const auto &error : errors)
    if (error) std::rethrow_exception(error);

  std::uint64_t missing = 0;
  std::uint64_t overflow = 0;
  std::ofstream output(options.output);
  output << "variant,task,reader,cycle,threshold_num,threshold_den,query,"
            "service_ms,kernel_ms,candidate_rows,hits,result_hash,overflow,"
            "completed\n";
  for (int task = 0; task < options.sustained_requests; ++task) {
    const auto &sample = samples[task];
    missing += !sample.completed;
    overflow += sample.overflow;
    output << variant << ',' << task << ',' << sample.reader << ','
           << sample.cycle << ',' << sample.numerator << ','
           << sample.denominator << ',' << sample.query << ','
           << std::setprecision(12) << sample.service_ms << ','
           << sample.kernel_ms << ',' << sample.candidate_rows << ','
           << sample.hits << ',' << sample.result_hash << ','
           << sample.overflow << ',' << sample.completed << '\n';
  }
  const double throughput_qps = options.sustained_requests / retained_wall_s;
  std::ofstream summary(options.output + ".summary.json");
  summary << "{\n"
          << "  \"experiment_id\": "
             "\"tide_20260829_gate6_sustained_reader\",\n"
          << "  \"variant\": \"" << variant << "\",\n"
          << "  \"reader_streams\": " << READERS << ",\n"
          << "  \"query_count\": " << query_count << ",\n"
          << "  \"retained_requests\": " << options.sustained_requests
          << ",\n"
          << "  \"cycles\": "
          << options.sustained_requests / unique_requests << ",\n"
          << "  \"retained_wall_s\": " << std::setprecision(17)
          << retained_wall_s << ",\n"
          << "  \"throughput_qps\": " << throughput_qps << ",\n"
          << "  \"missing_samples\": " << missing << ",\n"
          << "  \"overflow_requests\": " << overflow << ",\n"
          << "  \"pass\": "
          << (missing == 0 && overflow == 0 ? "true" : "false") << "\n"
          << "}\n";
  if (missing || overflow)
    throw std::runtime_error("sustained completeness failed");
}

template <int WORDS>
struct UpdateInputs {
  MappedFile base_fp;
  MappedFile base_ids;
  MappedFile base_pc;
  MappedFile union_fp;
  MappedFile union_ids;
  MappedFile union_pc;
  MappedFile query_file;
  HostDBView<WORDS> base;
  HostDBView<WORDS> fresh;
  QueryStore<WORDS> queries;
  explicit UpdateInputs(const std::string &root)
      : base_fp(root + "/data/gate0_prepared/base_fp_u64x" +
                std::to_string(WORDS) + ".bin"),
        base_ids(root + "/data/gate0_prepared/base_id_i64.bin"),
        base_pc(root + "/data/gate0_prepared/base_popcnt_u16.bin"),
        union_fp(root + "/data/gate0_prepared/union_fp_u64x" +
                 std::to_string(WORDS) + ".bin"),
        union_ids(root + "/data/gate0_prepared/union_id_i64.bin"),
        union_pc(root + "/data/gate0_prepared/union_popcnt_u16.bin"),
        query_file(root + "/data/stage_a/queries_u64x" +
                   std::to_string(WORDS + 2) + ".bin"),
        base(make_host_view<WORDS>(base_fp, base_ids, base_pc)),
        fresh(make_host_view<WORDS>(union_fp, union_ids, union_pc)),
        queries(query_file) {}
};

bool contains_id(const std::vector<HostHit> &hits, std::uint64_t id) {
  return std::any_of(hits.begin(), hits.end(),
                     [id](const HostHit &hit) { return hit.id == id; });
}

bool has_duplicate_id(const std::vector<HostHit> &hits) {
  return std::adjacent_find(
             hits.begin(), hits.end(),
             [](const HostHit &left, const HostHit &right) {
               return left.id == right.id;
             }) != hits.end();
}

struct OracleExpected {
  bool present = false;
  std::uint64_t candidate_rows = 0;
  std::size_t hits = 0;
  std::uint64_t id_hash = 0;
  bool contains_query_id = false;
};

class EpochOracle {
 public:
  EpochOracle(const std::string &path, int query_count)
      : query_count_(query_count) {
    if (query_count_ <= 0) throw std::runtime_error("invalid oracle query count");
    for (auto &records : records_) records.resize(query_count_);
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open epoch oracle: " + path);
    std::string line;
    if (!std::getline(input, line) ||
        line != "epoch,query,threshold_num,threshold_den,candidate_rows,hits,"
                "id_hash,contains_query_id,duplicate_id,diagnostic_cpu_seconds")
      throw std::runtime_error("unexpected epoch oracle header");
    std::size_t loaded = 0;
    while (std::getline(input, line)) {
      if (line.empty()) continue;
      const auto fields = split_text(line);
      if (fields.size() != 10)
        throw std::runtime_error("malformed epoch oracle row");
      const int epoch = std::stoi(fields[0]);
      const int query = std::stoi(fields[1]);
      const int num = std::stoi(fields[2]);
      const int den = std::stoi(fields[3]);
      if (epoch < 0 || epoch > 1 || query < 0 || query >= query_count_)
        throw std::runtime_error("epoch oracle key out of range");
      const int slot = record_slot(epoch, num, den);
      auto &record = records_[slot][query];
      if (record.present) throw std::runtime_error("duplicate epoch oracle row");
      if (std::stoi(fields[8]) != 0)
        throw std::runtime_error("duplicate ID in epoch oracle row");
      record = OracleExpected{true,
                              std::stoull(fields[4]),
                              static_cast<std::size_t>(std::stoull(fields[5])),
                              std::stoull(fields[6]),
                              std::stoi(fields[7]) != 0};
      if ((epoch == 0 && record.contains_query_id) ||
          (epoch == 1 && !record.contains_query_id))
        throw std::runtime_error("epoch oracle query-ID membership mismatch");
      loaded++;
    }
    const std::size_t expected = static_cast<std::size_t>(query_count_) * 4;
    if (loaded != expected)
      throw std::runtime_error("incomplete epoch oracle: expected " +
                               std::to_string(expected) + ", got " +
                               std::to_string(loaded));
    for (const auto &records : records_)
      for (const auto &record : records)
        if (!record.present) throw std::runtime_error("epoch oracle hole");
  }

  const OracleExpected &at(int epoch, int query, int num, int den) const {
    if (query < 0 || query >= query_count_)
      throw std::runtime_error("oracle query out of range");
    return records_[record_slot(epoch, num, den)][query];
  }

 private:
  static int record_slot(int epoch, int num, int den) {
    if (epoch < 0 || epoch > 1) throw std::runtime_error("invalid epoch");
    int threshold = -1;
    if (num == 7 && den == 10) threshold = 0;
    if (num == 4 && den == 5) threshold = 1;
    if (threshold < 0) throw std::runtime_error("invalid oracle threshold");
    return epoch * 2 + threshold;
  }

  int query_count_ = 0;
  std::array<std::vector<OracleExpected>, 4> records_;
};

template <class Clock>
double elapsed_ms(const typename Clock::time_point &begin,
                  const typename Clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool matches_oracle(const RunResult &result, const OracleExpected &expected,
                    const Query<4> &query) {
  return !result.overflow && !has_duplicate_id(result.hits) &&
         result.candidate_rows == expected.candidate_rows &&
         result.hits.size() == expected.hits &&
         hash_hit_ids(result.hits) == expected.id_hash &&
         contains_id(result.hits, query.id) == expected.contains_query_id;
}

struct PublicationResult {
  std::shared_ptr<Snapshot<4>> snapshot;
  double artifact_map_ms = 0;
  double host_prepare_ms = 0;
  double device_run_ms = 0;
  double snapshot_publish_ms = 0;
  double update_start_us = 0;
  double publish_us = 0;
  std::size_t free_during_update = 0;
};

template <class Clock>
double relative_us(const typename Clock::time_point &origin,
                   const typename Clock::time_point &point) {
  return std::chrono::duration<double, std::micro>(point - origin).count();
}

PublicationResult publish_epoch_one(
    const Options &options, const HostDBView<4> &base,
    const std::shared_ptr<DeviceRun<4>> &base_device,
    std::shared_ptr<Snapshot<4>> &current_snapshot,
    cudaStream_t upload_stream,
    const std::chrono::steady_clock::time_point origin) {
  using Clock = std::chrono::steady_clock;
  CUDA_CHECK(cudaSetDevice(options.gpu));
  const std::string delta_path = options.root + "/data/stage_a/delta_u64x6.bin";
  const auto t0 = Clock::now();
  auto delta_file = std::make_unique<MappedFile>(delta_path);
  const auto t1 = Clock::now();
  auto delta_owned =
      std::make_shared<OwnedHostDB<4>>(load_interleaved<4>(*delta_file));
  const auto delta_view = delta_owned->view();
  const auto t2 = Clock::now();
  auto delta_device =
      std::make_shared<DeviceRun<4>>(options.gpu, delta_view, upload_stream);
  const auto t3 = Clock::now();
  std::size_t free_during_update = 0;
  std::size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_during_update, &total_bytes));
  auto published = std::make_shared<Snapshot<4>>();
  published->epoch = 1;
  published->runs.reserve(2);
  published->runs.push_back({base_device, base});
  published->runs.push_back({delta_device, delta_view});
  published->host_owners.push_back(delta_owned);
  std::atomic_store_explicit(&current_snapshot, published,
                             std::memory_order_release);
  const auto t4 = Clock::now();
  PublicationResult result;
  result.snapshot = std::move(published);
  result.artifact_map_ms = elapsed_ms<Clock>(t0, t1);
  result.host_prepare_ms = elapsed_ms<Clock>(t1, t2);
  result.device_run_ms = elapsed_ms<Clock>(t2, t3);
  result.snapshot_publish_ms = elapsed_ms<Clock>(t3, t4);
  result.update_start_us = relative_us<Clock>(origin, t0);
  result.publish_us = relative_us<Clock>(origin, t4);
  result.free_during_update = free_during_update;
  return result;
}

struct OverlapSample {
  int reader = -1;
  int task = -1;
  int cycle = -1;
  int query = -1;
  int numerator = 0;
  int denominator = 0;
  int epoch = -1;
  double acquire_us = 0;
  double complete_us = 0;
  double service_ms = 0;
  float kernel_ms = 0;
  std::uint64_t candidate_rows = 0;
  std::size_t hits = 0;
  std::uint64_t id_hash = 0;
  bool overflow = false;
  bool duplicate_id = false;
  bool contains_query_id = false;
  std::uint64_t expected_candidates = 0;
  std::size_t expected_hits = 0;
  std::uint64_t expected_id_hash = 0;
  bool match = false;
  bool completed = false;
};

void write_publication_json(const std::string &path,
                            const PublicationResult &publication,
                            double first_visible_ms) {
  std::ofstream output(path);
  output << "{\n"
         << "  \"artifact_map_ms\": " << std::setprecision(17)
         << publication.artifact_map_ms << ",\n"
         << "  \"host_prepare_ms\": " << publication.host_prepare_ms << ",\n"
         << "  \"device_run_ms\": " << publication.device_run_ms << ",\n"
         << "  \"snapshot_publish_ms\": "
         << publication.snapshot_publish_ms << ",\n"
         << "  \"update_start_us\": " << publication.update_start_us << ",\n"
         << "  \"publish_us\": " << publication.publish_us << ",\n"
         << "  \"update_to_publish_ms\": "
         << (publication.publish_us - publication.update_start_us) / 1000.0
         << ",\n"
         << "  \"update_to_first_visible_ms\": " << first_visible_ms << ",\n"
         << "  \"cuda_free_during_update_bytes\": "
         << publication.free_during_update << "\n"
         << "}\n";
}

void run_overlap(const Options &options) {
  using Clock = std::chrono::steady_clock;
  constexpr int READERS = 4;
  const bool do_update = options.mode == "overlap";
  const auto setup_begin = Clock::now();
  UpdateInputs<4> inputs(options.root);
  const int query_count =
      std::min<int>(options.query_limit, static_cast<int>(inputs.queries.size()));
  if (query_count <= 0)
    throw std::runtime_error("overlap campaign has no queries");
  if (options.formal_overlap_gates && query_count != 512)
    throw std::runtime_error("formal overlap campaign requires 512 queries");
  const int unique_requests = query_count * 2;
  if (options.reader_requests % unique_requests != 0)
    throw std::runtime_error(
        "--reader-requests must be a multiple of query_count * 2");
  EpochOracle oracle(options.oracle, query_count);
  CudaStreamOwner upload_stream(options.gpu);
  auto base_device =
      std::make_shared<DeviceRun<4>>(options.gpu, inputs.base);
  auto base_snapshot = std::make_shared<Snapshot<4>>();
  base_snapshot->epoch = 0;
  base_snapshot->runs.push_back({base_device, inputs.base});
  std::shared_ptr<Snapshot<4>> current_snapshot = base_snapshot;
  std::array<std::unique_ptr<QueryRuntime<4>>, READERS> runtimes;
  for (auto &runtime : runtimes)
    runtime = std::make_unique<QueryRuntime<4>>(options.gpu);
  CUDA_CHECK(cudaDeviceSynchronize());

  const auto &probe_query = inputs.queries[0];
  auto retained_old = std::atomic_load_explicit(
      &current_snapshot, std::memory_order_acquire);
  const auto probe_origin = Clock::now();
  auto probe_publication = publish_epoch_one(
      options, inputs.base, base_device, current_snapshot, upload_stream.get(),
      probe_origin);
  const auto old_probe =
      runtimes[0]->run(*retained_old, probe_query, 7, 10, true);
  auto acquired_new = std::atomic_load_explicit(
      &current_snapshot, std::memory_order_acquire);
  const auto new_probe =
      runtimes[0]->run(*acquired_new, probe_query, 7, 10, true);
  const bool old_probe_match = matches_oracle(
      old_probe, oracle.at(0, 0, 7, 10), probe_query);
  const bool new_probe_match = matches_oracle(
      new_probe, oracle.at(1, 0, 7, 10), probe_query);
  std::ofstream probe_output(options.output + ".probe.json");
  probe_output << "{\n"
               << "  \"old_epoch\": " << retained_old->epoch << ",\n"
               << "  \"new_epoch\": " << acquired_new->epoch << ",\n"
               << "  \"old_match\": "
               << (old_probe_match ? "true" : "false") << ",\n"
               << "  \"new_match\": "
               << (new_probe_match ? "true" : "false") << ",\n"
               << "  \"pass\": "
               << (old_probe_match && new_probe_match ? "true" : "false")
               << "\n}\n";
  probe_output.close();
  if (!old_probe_match || !new_probe_match)
    throw std::runtime_error("retained ownership probe failed");
  std::atomic_store_explicit(&current_snapshot, base_snapshot,
                             std::memory_order_release);
  retained_old.reset();
  acquired_new.reset();
  probe_publication.snapshot.reset();
  CUDA_CHECK(cudaStreamSynchronize(upload_stream.get()));

  std::vector<OverlapSample> samples(options.reader_requests);
  std::array<std::exception_ptr, READERS> reader_errors{};
  std::exception_ptr update_error;
  PublicationResult publication;
  std::atomic<int> completed_requests{0};
  std::atomic<bool> reader_failed{false};
  pthread_barrier_t start_barrier{};
  const unsigned participants = READERS + 1 + (do_update ? 1 : 0);
  if (pthread_barrier_init(&start_barrier, nullptr, participants) != 0)
    throw std::runtime_error("pthread_barrier_init failed");
  Clock::time_point retained_begin;

  std::array<std::thread, READERS> readers;
  for (int reader = 0; reader < READERS; ++reader) {
    readers[reader] = std::thread([&, reader] {
      try {
        CUDA_CHECK(cudaSetDevice(options.gpu));
        for (const auto &[num, den] :
             std::array<std::pair<int, int>, 2>{{{7, 10}, {4, 5}}}) {
          for (int warmup = 0; warmup < 16; ++warmup) {
            const int query = (warmup * READERS + reader) % query_count;
            const auto result = runtimes[reader]->run(
                *base_snapshot, inputs.queries[query], num, den, true);
            if (!matches_oracle(result, oracle.at(0, query, num, den),
                                inputs.queries[query]))
              throw std::runtime_error("base warmup oracle mismatch");
          }
        }
      } catch (...) {
        reader_errors[reader] = std::current_exception();
        reader_failed.store(true, std::memory_order_release);
      }
      try {
        barrier_wait(&start_barrier);
        if (reader_errors[reader]) return;
        for (int task = reader; task < options.reader_requests; task += READERS) {
          const int within = task % unique_requests;
          const int threshold_slot = within / query_count;
          const int query = within % query_count;
          const auto [num, den] =
              threshold_slot == 0 ? std::pair<int, int>{7, 10}
                                  : std::pair<int, int>{4, 5};
          const auto acquire_time = Clock::now();
          auto observed = std::atomic_load_explicit(
              &current_snapshot, std::memory_order_acquire);
          const int epoch = static_cast<int>(observed->epoch);
          const auto result = runtimes[reader]->run(
              *observed, inputs.queries[query], num, den, true);
          const auto complete_time = Clock::now();
          const auto &expected = oracle.at(epoch, query, num, den);
          const auto id_hash = hash_hit_ids(result.hits);
          const bool duplicate_id = has_duplicate_id(result.hits);
          const bool query_membership =
              contains_id(result.hits, inputs.queries[query].id);
          const bool match = !result.overflow && !duplicate_id &&
                             result.candidate_rows == expected.candidate_rows &&
                             result.hits.size() == expected.hits &&
                             id_hash == expected.id_hash &&
                             query_membership == expected.contains_query_id;
          samples[task] = OverlapSample{
              reader,
              task,
              task / unique_requests,
              query,
              num,
              den,
              epoch,
              relative_us<Clock>(retained_begin, acquire_time),
              relative_us<Clock>(retained_begin, complete_time),
              result.service_ms,
              result.kernel_ms,
              result.candidate_rows,
              result.hits.size(),
              id_hash,
              result.overflow,
              duplicate_id,
              query_membership,
              expected.candidate_rows,
              expected.hits,
              expected.id_hash,
              match,
              true};
          completed_requests.fetch_add(1, std::memory_order_release);
          if (!match) throw std::runtime_error("retained epoch oracle mismatch");
        }
      } catch (...) {
        reader_errors[reader] = std::current_exception();
        reader_failed.store(true, std::memory_order_release);
      }
    });
  }

  std::thread updater;
  if (do_update) {
    updater = std::thread([&] {
      try {
        CUDA_CHECK(cudaSetDevice(options.gpu));
        barrier_wait(&start_barrier);
        while (completed_requests.load(std::memory_order_acquire) <
                   options.update_after_completions &&
               !reader_failed.load(std::memory_order_acquire))
          std::this_thread::yield();
        if (reader_failed.load(std::memory_order_acquire))
          throw std::runtime_error("reader failed before scheduled publication");
        publication = publish_epoch_one(
            options, inputs.base, base_device, current_snapshot,
            upload_stream.get(), retained_begin);
      } catch (...) {
        update_error = std::current_exception();
      }
    });
  }

  retained_begin = Clock::now();
  barrier_wait(&start_barrier);
  for (auto &reader : readers) reader.join();
  const auto readers_end = Clock::now();
  if (updater.joinable()) updater.join();
  pthread_barrier_destroy(&start_barrier);
  for (const auto &error : reader_errors)
    if (error) std::rethrow_exception(error);
  if (update_error) std::rethrow_exception(update_error);

  std::uint64_t missing = 0;
  std::uint64_t mismatches = 0;
  std::uint64_t overflows = 0;
  std::array<std::uint64_t, 2> epoch_counts{};
  std::array<std::array<std::uint64_t, 2>, READERS> reader_epoch_counts{};
  double first_epoch_one_complete_us = std::numeric_limits<double>::infinity();
  std::ofstream output(options.output);
  output << "variant,task,reader,cycle,threshold_num,threshold_den,query,epoch,"
            "acquire_us,complete_us,service_ms,kernel_ms,candidate_rows,hits,"
            "id_hash,overflow,duplicate_id,contains_query_id,expected_candidates,"
            "expected_hits,expected_id_hash,match,completed\n";
  for (const auto &sample : samples) {
    missing += !sample.completed;
    mismatches += sample.completed && !sample.match;
    overflows += sample.overflow;
    if (sample.completed && sample.epoch >= 0 && sample.epoch <= 1) {
      epoch_counts[sample.epoch]++;
      reader_epoch_counts[sample.reader][sample.epoch]++;
      if (sample.epoch == 1)
        first_epoch_one_complete_us =
            std::min(first_epoch_one_complete_us, sample.complete_us);
    }
    output << options.mode << ',' << sample.task << ',' << sample.reader << ','
           << sample.cycle << ',' << sample.numerator << ','
           << sample.denominator << ',' << sample.query << ',' << sample.epoch
           << ',' << std::setprecision(17) << sample.acquire_us << ','
           << sample.complete_us << ',' << sample.service_ms << ','
           << sample.kernel_ms << ',' << sample.candidate_rows << ','
           << sample.hits << ',' << sample.id_hash << ',' << sample.overflow
           << ',' << sample.duplicate_id << ',' << sample.contains_query_id
           << ',' << sample.expected_candidates << ',' << sample.expected_hits
           << ',' << sample.expected_id_hash << ',' << sample.match << ','
           << sample.completed << '\n';
  }
  output.close();

  const double reader_wall_s =
      std::chrono::duration<double>(readers_end - retained_begin).count();
  const double throughput_qps = options.reader_requests / reader_wall_s;
  const double readers_end_us = relative_us<Clock>(retained_begin, readers_end);
  const bool publication_before_reader_end =
      !do_update || publication.publish_us < readers_end_us;
  bool every_reader_both_epochs = true;
  for (int reader = 0; reader < READERS; ++reader)
    every_reader_both_epochs &= reader_epoch_counts[reader][0] > 0 &&
                                reader_epoch_counts[reader][1] > 0;
  const bool epoch_count_gate = !do_update ||
      (options.formal_overlap_gates
           ? (epoch_counts[0] >= 64 && epoch_counts[1] >= 512 &&
              every_reader_both_epochs)
           : (epoch_counts[0] > 0 && epoch_counts[1] > 0));
  const bool control_epoch_gate =
      do_update || (epoch_counts[0] ==
                        static_cast<std::uint64_t>(options.reader_requests) &&
                    epoch_counts[1] == 0);
  const double first_visible_ms =
      do_update && std::isfinite(first_epoch_one_complete_us)
          ? (first_epoch_one_complete_us - publication.update_start_us) / 1000.0
          : -1.0;
  const bool pass = missing == 0 && mismatches == 0 && overflows == 0 &&
                    publication_before_reader_end && epoch_count_gate &&
                    control_epoch_gate && (!do_update || first_visible_ms >= 0);

  std::ofstream summary(options.output + ".summary.json");
  summary << "{\n"
          << "  \"experiment_id\": "
             "\"tide_20260903_read_while_update_async_upload_epoch_consistency\",\n"
          << "  \"variant\": \"" << options.mode << "\",\n"
          << "  \"reader_streams\": " << READERS << ",\n"
          << "  \"query_count\": " << query_count << ",\n"
          << "  \"retained_requests\": " << options.reader_requests << ",\n"
          << "  \"formal_overlap_gates\": "
          << (options.formal_overlap_gates ? "true" : "false") << ",\n"
          << "  \"update_after_completions\": "
          << options.update_after_completions << ",\n"
          << "  \"cycles\": " << options.reader_requests / unique_requests
          << ",\n"
          << "  \"reader_wall_s\": " << std::setprecision(17) << reader_wall_s
          << ",\n"
          << "  \"throughput_qps\": " << throughput_qps << ",\n"
          << "  \"epoch0_requests\": " << epoch_counts[0] << ",\n"
          << "  \"epoch1_requests\": " << epoch_counts[1] << ",\n"
          << "  \"reader_epoch_counts\": [[" << reader_epoch_counts[0][0]
          << ',' << reader_epoch_counts[0][1] << "],["
          << reader_epoch_counts[1][0] << ',' << reader_epoch_counts[1][1]
          << "],[" << reader_epoch_counts[2][0] << ','
          << reader_epoch_counts[2][1] << "],[" << reader_epoch_counts[3][0]
          << ',' << reader_epoch_counts[3][1] << "]],\n"
          << "  \"missing_samples\": " << missing << ",\n"
          << "  \"oracle_mismatches\": " << mismatches << ",\n"
          << "  \"overflow_requests\": " << overflows << ",\n"
          << "  \"every_reader_both_epochs\": "
          << (every_reader_both_epochs ? "true" : "false") << ",\n"
          << "  \"publication_before_reader_end\": "
          << (publication_before_reader_end ? "true" : "false") << ",\n"
          << "  \"update_to_first_visible_ms\": " << first_visible_ms
          << ",\n"
          << "  \"pass\": " << (pass ? "true" : "false") << "\n}\n";
  summary.close();
  if (do_update)
    write_publication_json(options.output + ".update.json", publication,
                           first_visible_ms);

  std::atomic_store_explicit(&current_snapshot, base_snapshot,
                             std::memory_order_release);
  publication.snapshot.reset();
  CUDA_CHECK(cudaStreamSynchronize(upload_stream.get()));

  std::ofstream setup(options.output + ".setup.json");
  setup << "{\n"
        << "  \"scope\": \"input/oracle mapping, persistent base, four "
           "private runtimes, and ownership probe; excluded from retained "
           "reader/update denominators\",\n"
        << "  \"setup_s\": " << std::setprecision(17)
        << std::chrono::duration<double>(retained_begin - setup_begin).count()
        << ",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"base_rows\": " << inputs.base.rows << ",\n"
        << "  \"union_rows\": " << inputs.fresh.rows << "\n"
        << "}\n";
  if (!pass) throw std::runtime_error("overlap/control correctness gate failed");
}

struct UpdateTrial {
  double artifact_map_ms = 0;
  double host_prepare_ms = 0;
  double device_run_ms = 0;
  double snapshot_publish_ms = 0;
  double first_query_visible_ms = 0;
  double update_to_query_visible_ms = 0;
  RunResult query;
  bool contains_added_id = false;
  bool candidate_match = false;
  bool result_match = false;
};

template <int WORDS>
UpdateTrial run_update_trial(
    const Options &options, const HostDBView<WORDS> &base,
    const std::shared_ptr<DeviceRun<WORDS>> &base_device,
    const std::shared_ptr<Snapshot<WORDS>> &base_snapshot,
    std::shared_ptr<Snapshot<WORDS>> &current_snapshot,
    QueryRuntime<WORDS> &runtime, const Query<WORDS> &query, int num, int den,
    std::size_t expected_hits, std::uint64_t expected_hash,
    std::uint64_t expected_candidates, std::size_t *free_during_update) {
  using Clock = std::chrono::steady_clock;
  const std::string delta_path =
      options.root + "/data/stage_a/delta_u64x" +
      std::to_string(WORDS + 2) + ".bin";
  const auto t0 = Clock::now();
  auto delta_file = std::make_unique<MappedFile>(delta_path);
  const auto t1 = Clock::now();
  auto delta_owned = load_interleaved<WORDS>(*delta_file);
  const auto delta_view = delta_owned.view();
  const auto t2 = Clock::now();
  auto delta_device =
      std::make_shared<DeviceRun<WORDS>>(options.gpu, delta_view);
  const auto t3 = Clock::now();
  if (free_during_update) {
    std::size_t total = 0;
    CUDA_CHECK(cudaMemGetInfo(free_during_update, &total));
  }
  auto published = std::make_shared<Snapshot<WORDS>>();
  published->runs.reserve(2);
  published->runs.push_back({base_device, base});
  published->runs.push_back({delta_device, delta_view});
  std::atomic_store_explicit(&current_snapshot, published,
                             std::memory_order_release);
  const auto t4 = Clock::now();
  auto observed =
      std::atomic_load_explicit(&current_snapshot, std::memory_order_acquire);
  auto result = runtime.run(*observed, query, num, den, true);
  const auto t5 = Clock::now();

  UpdateTrial trial;
  trial.artifact_map_ms = elapsed_ms<Clock>(t0, t1);
  trial.host_prepare_ms = elapsed_ms<Clock>(t1, t2);
  trial.device_run_ms = elapsed_ms<Clock>(t2, t3);
  trial.snapshot_publish_ms = elapsed_ms<Clock>(t3, t4);
  trial.first_query_visible_ms = elapsed_ms<Clock>(t4, t5);
  trial.update_to_query_visible_ms = elapsed_ms<Clock>(t0, t5);
  trial.contains_added_id = contains_id(result.hits, query.id);
  trial.candidate_match = result.candidate_rows == expected_candidates;
  trial.result_match = !result.overflow &&
                       result.hits.size() == expected_hits &&
                       result.result_hash == expected_hash &&
                       trial.contains_added_id && trial.candidate_match;
  trial.query = std::move(result);

  std::atomic_store_explicit(&current_snapshot, base_snapshot,
                             std::memory_order_release);
  observed.reset();
  published.reset();
  delta_device.reset();
  return trial;
}

template <int WORDS>
void run_update(const Options &options) {
  if constexpr (WORDS != 4)
    throw std::runtime_error("Gate-6 update campaign is frozen to WORDS4");
  using Clock = std::chrono::steady_clock;
  const auto setup_begin = Clock::now();
  UpdateInputs<WORDS> inputs(options.root);
  if (inputs.queries.size() == 0)
    throw std::runtime_error("update campaign has no query");
  const std::string delta_path =
      options.root + "/data/stage_a/delta_u64x" +
      std::to_string(WORDS + 2) + ".bin";
  const auto delta_bytes = std::filesystem::file_size(delta_path);
  if (delta_bytes % ((WORDS + 2) * sizeof(std::uint64_t)))
    throw std::runtime_error("invalid update artifact size");
  const std::uint64_t delta_rows =
      delta_bytes / ((WORDS + 2) * sizeof(std::uint64_t));
  if (inputs.base.rows + delta_rows != inputs.fresh.rows)
    throw std::runtime_error("update insert-only row balance failed");

  std::size_t free_initial = 0;
  std::size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_initial, &total_bytes));
  auto base_device =
      std::make_shared<DeviceRun<WORDS>>(options.gpu, inputs.base);
  auto base_snapshot = std::make_shared<Snapshot<WORDS>>();
  base_snapshot->runs.push_back({base_device, inputs.base});
  std::shared_ptr<Snapshot<WORDS>> current_snapshot = base_snapshot;
  QueryRuntime<WORDS> runtime(options.gpu);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::size_t free_persistent = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_persistent, &total_bytes));
  const auto setup_end = Clock::now();

  const auto &query = inputs.queries[0];
  const auto base_result =
      runtime.run(*base_snapshot, query, 7, 10, true);
  if (base_result.overflow || contains_id(base_result.hits, query.id))
    throw std::runtime_error("added query ID is already visible in base");

  const auto oracle70 = cpu_scan<WORDS>(inputs.fresh, query, 7, 10);
  const auto oracle80 = cpu_scan<WORDS>(inputs.fresh, query, 4, 5);
  if (!contains_id(oracle70, query.id) || !contains_id(oracle80, query.id))
    throw std::runtime_error("added query ID absent from union oracle");
  const auto expected70_hash = hash_hits(oracle70);
  const auto expected80_hash = hash_hits(oracle80);
  const auto range70 = bounds<WORDS>(inputs.fresh, query, 7, 10);
  const auto range80 = bounds<WORDS>(inputs.fresh, query, 4, 5);
  const auto expected70_candidates = range70.second - range70.first;
  const auto expected80_candidates = range80.second - range80.first;

  std::size_t free_during_update = free_persistent;
  for (int warmup = 0; warmup < options.update_warmups; ++warmup) {
    auto trial = run_update_trial<WORDS>(
        options, inputs.base, base_device, base_snapshot, current_snapshot,
        runtime, query, 7, 10, oracle70.size(), expected70_hash,
        expected70_candidates, warmup == 0 ? &free_during_update : nullptr);
    if (!trial.result_match)
      throw std::runtime_error("update warmup visibility mismatch");
  }

  std::ofstream samples(options.output);
  samples << "sample,artifact_map_ms,host_prepare_ms,device_run_ms,"
             "snapshot_publish_ms,first_query_visible_ms,"
             "update_to_query_visible_ms,kernel_ms,candidate_rows,hits,"
             "result_hash,overflow,contains_added_id,candidate_match,match\n";
  std::uint64_t mismatches = 0;
  for (int sample = 0; sample < options.update_requests; ++sample) {
    const auto trial = run_update_trial<WORDS>(
        options, inputs.base, base_device, base_snapshot, current_snapshot,
        runtime, query, 7, 10, oracle70.size(), expected70_hash,
        expected70_candidates, nullptr);
    mismatches += !trial.result_match;
    samples << sample << ',' << std::setprecision(12)
            << trial.artifact_map_ms << ',' << trial.host_prepare_ms << ','
            << trial.device_run_ms << ',' << trial.snapshot_publish_ms << ','
            << trial.first_query_visible_ms << ','
            << trial.update_to_query_visible_ms << ',' << trial.query.kernel_ms
            << ',' << trial.query.candidate_rows << ','
            << trial.query.hits.size() << ',' << trial.query.result_hash << ','
            << trial.query.overflow << ',' << trial.contains_added_id << ','
            << trial.candidate_match << ',' << trial.result_match << '\n';
    if (!trial.result_match)
      throw std::runtime_error("retained update visibility mismatch");
  }
  samples.flush();

  const auto post80 = run_update_trial<WORDS>(
      options, inputs.base, base_device, base_snapshot, current_snapshot,
      runtime, query, 4, 5, oracle80.size(), expected80_hash,
      expected80_candidates, nullptr);
  if (!post80.result_match)
    throw std::runtime_error("post-campaign 4/5 visibility mismatch");

  std::ofstream setup(options.output + ".setup.json");
  setup << "{\n"
        << "  \"experiment_id\": \"tide_20260829_gate6_update_setup\",\n"
        << "  \"scope\": \"persistent base and query runtime; setup excluded "
           "from update-to-query-visible denominator\",\n"
        << "  \"setup_s\": " << std::setprecision(17)
        << elapsed_ms<Clock>(setup_begin, setup_end) / 1000.0 << ",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"base_rows\": " << inputs.base.rows << ",\n"
        << "  \"delta_rows\": " << delta_rows << ",\n"
        << "  \"union_rows\": " << inputs.fresh.rows << ",\n"
        << "  \"delta_artifact_bytes\": " << delta_bytes << ",\n"
        << "  \"cuda_free_initial_bytes\": " << free_initial << ",\n"
        << "  \"cuda_free_persistent_bytes\": " << free_persistent << ",\n"
        << "  \"cuda_free_during_update_bytes\": " << free_during_update << ",\n"
        << "  \"persistent_gpu_used_delta_bytes\": "
        << (free_initial >= free_persistent ? free_initial - free_persistent : 0)
        << ",\n"
        << "  \"update_gpu_used_delta_bytes\": "
        << (free_persistent >= free_during_update
                ? free_persistent - free_during_update
                : 0)
        << ",\n"
        << "  \"cuda_total_bytes\": " << total_bytes << "\n"
        << "}\n";

  std::ofstream validation(options.output + ".validation.json");
  validation << "{\n"
             << "  \"experiment_id\": "
                "\"tide_20260829_gate6_update_visibility\",\n"
             << "  \"warmups\": " << options.update_warmups << ",\n"
             << "  \"retained_requests\": " << options.update_requests
             << ",\n"
             << "  \"base_query_hits\": " << base_result.hits.size() << ",\n"
             << "  \"base_contains_added_id\": false,\n"
             << "  \"oracle_7_10_hits\": " << oracle70.size() << ",\n"
             << "  \"oracle_7_10_hash\": " << expected70_hash << ",\n"
             << "  \"oracle_7_10_candidates\": "
             << expected70_candidates << ",\n"
             << "  \"oracle_4_5_hits\": " << oracle80.size() << ",\n"
             << "  \"oracle_4_5_hash\": " << expected80_hash << ",\n"
             << "  \"oracle_4_5_candidates\": "
             << expected80_candidates << ",\n"
             << "  \"post_4_5_match\": "
             << (post80.result_match ? "true" : "false") << ",\n"
             << "  \"retained_mismatches\": " << mismatches << ",\n"
             << "  \"pass\": "
             << (mismatches == 0 && post80.result_match ? "true" : "false")
             << "\n"
             << "}\n";
}

template <int WORDS>
void dispatch(const Options &options) {
  if (options.mode == "update") {
    run_update<WORDS>(options);
    return;
  }
  if (options.mode == "overlap" || options.mode == "control") {
    if constexpr (WORDS != 4)
      throw std::runtime_error("overlap campaign is frozen to WORDS4");
    else
      run_overlap(options);
    return;
  }
  const auto setup_begin = std::chrono::steady_clock::now();
  Inputs<WORDS> inputs(options.root);
  if (options.mode == "correctness")
    run_correctness<WORDS>(options, inputs);
  else if (options.mode == "benchmark")
    run_benchmark<WORDS>(options, inputs, setup_begin);
  else
    run_sustained<WORDS>(options, inputs, setup_begin);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    CUDA_CHECK(cudaSetDevice(options.gpu));
    if (options.words == 4)
      dispatch<4>(options);
    else
      dispatch<32>(options);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 2;
  }
}
