#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
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
#include <tuple>
#include <utility>
#include <vector>

#include <fcntl.h>
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

template <int WORDS>
struct DeviceRun {
  std::uint64_t *fp = nullptr;
  std::uint64_t *ids = nullptr;
  std::uint16_t *pc = nullptr;
  std::uint64_t rows = 0;
  int gpu = -1;
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
  ~DeviceRun() {
    if (gpu < 0) return;
    cudaSetDevice(gpu);
    if (fp) cudaFree(fp);
    if (ids) cudaFree(ids);
    if (pc) cudaFree(pc);
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
  std::vector<SnapshotEntry<WORDS>> runs;
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

struct RunResult {
  double service_ms = 0;
  float kernel_ms = 0;
  std::uint64_t candidate_rows = 0;
  std::uint64_t launched_blocks = 0;
  std::uint32_t active_slices = 0;
  std::uint32_t launches = 0;
  std::uint32_t completion_syncs = 0;
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
                int num, int den, bool bounded, bool queued = false) {
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
    result.active_slices = static_cast<std::uint32_t>(slices.size());
    result.launched_blocks = blocks;
    result.launches = queued ? result.active_slices : 1;
    std::uint64_t block_sum = 0;
    for (auto &slice : slices) {
      block_sum += (slice.rows + BLOCK - 1) / BLOCK;
      if (queued) slice.first_block = 0;
    }
    if (block_sum != blocks) throw std::runtime_error("block ledger mismatch");
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
    if (queued) {
      for (std::size_t i = 0; i < slices.size(); ++i) {
        const auto local_blocks = (slices[i].rows + BLOCK - 1) / BLOCK;
        exact_runs_kernel<WORDS><<<static_cast<unsigned>(local_blocks), BLOCK, 0, stream_>>>(
            d_slices_ + i, 1, d_query_, d_hits_, OUTPUT_CAPACITY, d_count_, d_overflow_);
        CUDA_CHECK(cudaGetLastError());
      }
    } else {
      exact_runs_kernel<WORDS><<<static_cast<unsigned>(blocks), BLOCK, 0, stream_>>>(
          d_slices_, static_cast<int>(slices.size()), d_query_, d_hits_,
          OUTPUT_CAPACITY, d_count_, d_overflow_);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(kernel_stop_, stream_));
    CUDA_CHECK(cudaMemcpyAsync(&count, d_count_, sizeof(count),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(&overflow, d_overflow_, sizeof(overflow),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    result.completion_syncs = 1;
    result.observed_hits = count;
    result.overflow = overflow != 0 || count > OUTPUT_CAPACITY;
    if (!result.overflow && count) {
      std::vector<DeviceHit> device_hits(count);
      CUDA_CHECK(cudaMemcpyAsync(device_hits.data(), d_hits_,
                                 count * sizeof(DeviceHit),
                                 cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      result.completion_syncs = 2;
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

} // namespace: frozen donor runtime
