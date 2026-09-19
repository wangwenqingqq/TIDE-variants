#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr int BLOCK = 256;
constexpr std::uint32_t OUTPUT_CAPACITY = 65536;
constexpr int MAX_RUNS = 65;

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
  std::uint64_t id;
  std::uint16_t num;
  std::uint16_t den;
  std::uint32_t reserved;
};
static_assert(sizeof(DeviceHit) == 16);

struct HostHit {
  std::uint64_t id;
  std::uint16_t num;
  std::uint16_t den;
};

struct alignas(16) DeviceQuery {
  std::uint64_t fp[4];
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

struct MergeSlice {
  const std::uint64_t *fp;
  const std::uint64_t *ids;
  const std::uint16_t *pc;
  std::uint64_t rows;
};

__global__ void merge_delta_bin_kernel(
    const MergeSlice *__restrict__ slices, int slice_count,
    std::uint64_t *__restrict__ output_fp,
    std::uint64_t *__restrict__ output_ids,
    std::uint16_t *__restrict__ output_pc, std::uint64_t output_begin,
    std::uint16_t popcount, std::uint64_t output_rows) {
  if (blockIdx.x || threadIdx.x) return;
  std::uint64_t cursor[64];
#pragma unroll
  for (int index = 0; index < 64; ++index) cursor[index] = 0;
  for (std::uint64_t output = 0; output < output_rows; ++output) {
    int selected = -1;
    std::uint64_t selected_id = ~std::uint64_t{0};
    for (int slice = 0; slice < slice_count; ++slice) {
      if (cursor[slice] >= slices[slice].rows) continue;
      const std::uint64_t id = slices[slice].ids[cursor[slice]];
      if (selected < 0 || id < selected_id) {
        selected = slice;
        selected_id = id;
      }
    }
    if (selected < 0) return;
    const std::uint64_t source = cursor[selected]++;
    const std::uint64_t target = output_begin + output;
    output_ids[target] = slices[selected].ids[source];
    output_pc[target] = popcount;
    for (int word = 0; word < 4; ++word)
      output_fp[target * 4 + word] = slices[selected].fp[source * 4 + word];
  }
}

__device__ unsigned differing_bytes_u64(std::uint64_t left,
                                        std::uint64_t right) {
  std::uint64_t value = left ^ right;
  unsigned count = 0;
#pragma unroll
  for (int byte = 0; byte < 8; ++byte)
    count += ((value >> (byte * 8)) & 0xffu) != 0;
  return count;
}

__global__ void compare_database_kernel(
    const std::uint64_t *__restrict__ left_fp,
    const std::uint64_t *__restrict__ left_ids,
    const std::uint16_t *__restrict__ left_pc,
    const std::uint64_t *__restrict__ right_fp,
    const std::uint64_t *__restrict__ right_ids,
    const std::uint16_t *__restrict__ right_pc, std::uint64_t rows,
    unsigned long long *__restrict__ row_mismatches,
    unsigned long long *__restrict__ byte_mismatches) {
  const std::uint64_t row =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  unsigned bytes = differing_bytes_u64(left_ids[row], right_ids[row]);
  for (int word = 0; word < 4; ++word)
    bytes += differing_bytes_u64(left_fp[row * 4 + word],
                                 right_fp[row * 4 + word]);
  const std::uint16_t pc_xor = left_pc[row] ^ right_pc[row];
  bytes += (pc_xor & 0xffu) != 0;
  bytes += (pc_xor >> 8) != 0;
  if (bytes) {
    atomicAdd(row_mismatches, 1ull);
    atomicAdd(byte_mismatches, static_cast<unsigned long long>(bytes));
  }
}

__global__ void exact_logical_runs_kernel(
    const DeviceSlice *__restrict__ slices, int slice_count,
    const DeviceQuery *__restrict__ query, DeviceHit *__restrict__ hits,
    std::uint32_t capacity, std::uint32_t *__restrict__ total_hits,
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
  const DeviceQuery q = *query;
  const std::uint64_t offset = row * 4;
  const unsigned intersection =
      __popcll(q.fp[0] & slice.fp[offset + 0]) +
      __popcll(q.fp[1] & slice.fp[offset + 1]) +
      __popcll(q.fp[2] & slice.fp[offset + 2]) +
      __popcll(q.fp[3] & slice.fp[offset + 3]);
  unsigned union_count =
      static_cast<unsigned>(q.popcount) + slice.pc[row] - intersection;
  if (union_count == 0) union_count = 1;
  if (intersection * static_cast<unsigned>(q.threshold_den) <
      union_count * static_cast<unsigned>(q.threshold_num))
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

struct HostDBView {
  const std::uint64_t *fp = nullptr;
  const std::uint64_t *ids = nullptr;
  const std::uint16_t *pc = nullptr;
  std::uint64_t rows = 0;
  std::array<std::uint64_t, 258> cumulative{};
};

HostDBView make_host_view(const MappedFile &fp, const MappedFile &ids,
                          const MappedFile &pc,
                          std::uint64_t limit_rows = 0) {
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
  for (int value = 0; value <= 256; ++value)
    view.cumulative[value + 1] =
        view.cumulative[value] + counts[value];
  return view;
}

struct OwnedHostDB {
  std::vector<std::uint64_t> fp;
  std::vector<std::uint64_t> ids;
  std::vector<std::uint16_t> pc;
  HostDBView view() const {
    HostDBView output;
    output.fp = fp.data();
    output.ids = ids.data();
    output.pc = pc.data();
    output.rows = ids.size();
    std::array<std::uint64_t, 257> counts{};
    std::uint16_t previous = 0;
    for (std::size_t row = 0; row < pc.size(); ++row) {
      if (pc[row] > 256 || (row && pc[row] < previous))
        throw std::runtime_error("owned run is not population-count sorted");
      counts[pc[row]]++;
      previous = pc[row];
    }
    for (int value = 0; value <= 256; ++value)
      output.cumulative[value + 1] =
          output.cumulative[value] + counts[value];
    return output;
  }
};

OwnedHostDB load_delta(const MappedFile &file) {
  if (file.bytes() % (6 * sizeof(std::uint64_t)))
    throw std::runtime_error("bad delta byte size");
  const auto *input = file.as<std::uint64_t>();
  const std::size_t rows = file.bytes() / (6 * sizeof(std::uint64_t));
  OwnedHostDB output;
  output.fp.resize(rows * 4);
  output.ids.resize(rows);
  output.pc.resize(rows);
  std::uint16_t previous = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    output.ids[row] = input[row * 6];
    unsigned actual = 0;
    for (int word = 0; word < 4; ++word) {
      output.fp[row * 4 + word] = input[row * 6 + 1 + word];
      actual += __builtin_popcountll(output.fp[row * 4 + word]);
    }
    output.pc[row] = static_cast<std::uint16_t>(input[row * 6 + 5]);
    if (actual != output.pc[row] || (row && output.pc[row] < previous))
      throw std::runtime_error("delta validation failed");
    previous = output.pc[row];
  }
  return output;
}

std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

std::vector<OwnedHostDB> split_runs(const HostDBView &delta, int run_count) {
  if (run_count < 1 || run_count > 64)
    throw std::runtime_error("run count must be in [1,64]");
  struct Row {
    std::array<std::uint64_t, 4> fp;
    std::uint64_t id;
    std::uint16_t pc;
  };
  std::vector<std::vector<Row>> buckets(run_count);
  for (std::uint64_t row = 0; row < delta.rows; ++row) {
    Row value{{delta.fp[row * 4], delta.fp[row * 4 + 1],
               delta.fp[row * 4 + 2], delta.fp[row * 4 + 3]},
              delta.ids[row], delta.pc[row]};
    buckets[splitmix64(value.id) % run_count].push_back(value);
  }
  std::vector<OwnedHostDB> output(run_count);
  for (int run = 0; run < run_count; ++run) {
    auto &rows = buckets[run];
    std::sort(rows.begin(), rows.end(), [](const Row &left, const Row &right) {
      return std::tie(left.pc, left.id) < std::tie(right.pc, right.id);
    });
    auto &target = output[run];
    target.fp.reserve(rows.size() * 4);
    target.ids.reserve(rows.size());
    target.pc.reserve(rows.size());
    for (const Row &row : rows) {
      target.ids.push_back(row.id);
      target.pc.push_back(row.pc);
      target.fp.insert(target.fp.end(), row.fp.begin(), row.fp.end());
    }
  }
  return output;
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
    const auto *input = file.as<std::uint64_t>();
    const std::size_t rows = file.bytes() / (6 * sizeof(std::uint64_t));
    queries_.resize(rows);
    for (std::size_t row = 0; row < rows; ++row) {
      Query &query = queries_[row];
      query.id = input[row * 6];
      unsigned actual = 0;
      for (int word = 0; word < 4; ++word) {
        query.fp[word] = input[row * 6 + 1 + word];
        actual += __builtin_popcountll(query.fp[word]);
      }
      query.pc = static_cast<std::uint16_t>(input[row * 6 + 5]);
      if (actual != query.pc)
        throw std::runtime_error("query popcount mismatch");
    }
  }
  const Query &operator[](std::size_t index) const { return queries_.at(index); }
  std::size_t size() const { return queries_.size(); }

 private:
  std::vector<Query> queries_;
};

std::pair<int, int> popcount_interval(std::uint16_t query_pc, int num,
                                      int den) {
  const int lower = (num * static_cast<int>(query_pc) + den - 1) / den;
  const int upper = den * static_cast<int>(query_pc) / num;
  return {std::max(0, lower), std::min(256, upper)};
}

std::pair<std::uint64_t, std::uint64_t> bounds(const HostDBView &database,
                                                const Query &query, int num,
                                                int den) {
  const auto interval = popcount_interval(query.pc, num, den);
  return {database.cumulative[interval.first],
          database.cumulative[interval.second + 1]};
}

struct DeviceRun {
  std::uint64_t *fp = nullptr;
  std::uint64_t *ids = nullptr;
  std::uint16_t *pc = nullptr;
  std::uint64_t rows = 0;
  int gpu = -1;

  explicit DeviceRun(int target_gpu, std::uint64_t count)
      : rows(count), gpu(target_gpu) {
    CUDA_CHECK(cudaSetDevice(gpu));
    if (rows) {
      CUDA_CHECK(cudaMalloc(&fp, rows * 4 * sizeof(std::uint64_t)));
      CUDA_CHECK(cudaMalloc(&ids, rows * sizeof(std::uint64_t)));
      CUDA_CHECK(cudaMalloc(&pc, rows * sizeof(std::uint16_t)));
    }
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

  void upload(const HostDBView &host, cudaStream_t stream = nullptr,
              bool synchronize = true) {
    if (host.rows != rows)
      throw std::runtime_error("device run row mismatch");
    CUDA_CHECK(cudaSetDevice(gpu));
    if (rows) {
      CUDA_CHECK(cudaMemcpyAsync(fp, host.fp, rows * 4 * sizeof(std::uint64_t),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(ids, host.ids, rows * sizeof(std::uint64_t),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(pc, host.pc, rows * sizeof(std::uint16_t),
                                 cudaMemcpyHostToDevice, stream));
    }
    if (synchronize) CUDA_CHECK(cudaStreamSynchronize(stream));
  }
};

struct SnapshotEntry {
  std::shared_ptr<DeviceRun> device;
  HostDBView host;
};

struct Snapshot {
  std::uint64_t epoch = 0;
  std::vector<SnapshotEntry> runs;
};

std::vector<DeviceSlice> make_slices(const Snapshot &snapshot,
                                     const Query &query, int num, int den,
                                     std::uint64_t *candidate_rows = nullptr) {
  std::vector<DeviceSlice> slices;
  slices.reserve(snapshot.runs.size());
  std::uint64_t first_block = 0;
  std::uint64_t candidates = 0;
  for (const auto &run : snapshot.runs) {
    const auto range = bounds(run.host, query, num, den);
    const std::uint64_t rows = range.second - range.first;
    if (!rows) continue;
    slices.push_back(DeviceSlice{run.device->fp + range.first * 4,
                                 run.device->ids + range.first,
                                 run.device->pc + range.first, rows,
                                 first_block});
    first_block += (rows + BLOCK - 1) / BLOCK;
    candidates += rows;
  }
  if (candidate_rows) *candidate_rows = candidates;
  return slices;
}

void sort_hits(std::vector<HostHit> &hits) {
  std::sort(hits.begin(), hits.end(), [](const HostHit &left,
                                         const HostHit &right) {
    const std::uint32_t left_num = left.num;
    const std::uint32_t right_num = right.num;
    const std::uint64_t lhs = static_cast<std::uint64_t>(left_num) * right.den;
    const std::uint64_t rhs = static_cast<std::uint64_t>(right_num) * left.den;
    if (lhs != rhs) return lhs > rhs;
    return left.id < right.id;
  });
}

std::uint64_t hash_hits(const std::vector<HostHit> &hits) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const HostHit &hit : hits) {
    const std::uint64_t fields[2] = {
        hit.id, static_cast<std::uint64_t>(hit.num) |
                    (static_cast<std::uint64_t>(hit.den) << 16)};
    const auto *bytes = reinterpret_cast<const unsigned char *>(fields);
    for (std::size_t index = 0; index < sizeof(fields); ++index) {
      hash ^= bytes[index];
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

struct RunResult {
  double service_ms = 0;
  float kernel_ms = 0;
  float device_ms = 0;
  std::uint64_t candidate_rows = 0;
  std::uint32_t observed_hits = 0;
  bool overflow = false;
  std::vector<HostHit> hits;
};

class QueryRuntime {
 public:
  explicit QueryRuntime(int gpu, std::uint32_t capacity = OUTPUT_CAPACITY)
      : gpu_(gpu), capacity_(capacity) {
    CUDA_CHECK(cudaSetDevice(gpu_));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&device_start_));
    CUDA_CHECK(cudaEventCreate(&kernel_start_));
    CUDA_CHECK(cudaEventCreate(&kernel_stop_));
    CUDA_CHECK(cudaEventCreate(&device_stop_));
    CUDA_CHECK(cudaMalloc(&d_query_, sizeof(DeviceQuery)));
    CUDA_CHECK(cudaMalloc(&d_slices_, MAX_RUNS * sizeof(DeviceSlice)));
    CUDA_CHECK(cudaMalloc(&d_hits_, capacity_ * sizeof(DeviceHit)));
    CUDA_CHECK(cudaMalloc(&d_count_, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_overflow_, sizeof(std::uint32_t)));
  }
  ~QueryRuntime() {
    cudaSetDevice(gpu_);
    if (stream_) cudaStreamSynchronize(stream_);
    if (d_query_) cudaFree(d_query_);
    if (d_slices_) cudaFree(d_slices_);
    if (d_hits_) cudaFree(d_hits_);
    if (d_count_) cudaFree(d_count_);
    if (d_overflow_) cudaFree(d_overflow_);
    if (device_start_) cudaEventDestroy(device_start_);
    if (kernel_start_) cudaEventDestroy(kernel_start_);
    if (kernel_stop_) cudaEventDestroy(kernel_stop_);
    if (device_stop_) cudaEventDestroy(device_stop_);
    if (stream_) cudaStreamDestroy(stream_);
  }
  QueryRuntime(const QueryRuntime &) = delete;
  QueryRuntime &operator=(const QueryRuntime &) = delete;

  RunResult run(const Snapshot &snapshot, const Query &query, int num,
                int den) {
    CUDA_CHECK(cudaSetDevice(gpu_));
    std::uint64_t candidate_rows = 0;
    const auto slices = make_slices(snapshot, query, num, den, &candidate_rows);
    if (slices.empty()) return RunResult{};
    if (slices.size() > MAX_RUNS)
      throw std::runtime_error("too many device slices");
    const auto &last = slices.back();
    const std::uint64_t blocks =
        last.first_block + (last.rows + BLOCK - 1) / BLOCK;
    if (blocks > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("grid exceeds CUDA x dimension");
    DeviceQuery device_query{{query.fp[0], query.fp[1], query.fp[2], query.fp[3]},
                             query.pc,
                             static_cast<std::uint16_t>(num),
                             static_cast<std::uint16_t>(den), 0};
    const auto host_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(device_start_, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_query_, &device_query, sizeof(device_query),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_slices_, slices.data(),
                               slices.size() * sizeof(DeviceSlice),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemsetAsync(d_count_, 0, sizeof(std::uint32_t), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_overflow_, 0, sizeof(std::uint32_t), stream_));
    CUDA_CHECK(cudaEventRecord(kernel_start_, stream_));
    exact_logical_runs_kernel<<<static_cast<unsigned>(blocks), BLOCK, 0, stream_>>>(
        d_slices_, static_cast<int>(slices.size()), d_query_, d_hits_, capacity_,
        d_count_, d_overflow_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernel_stop_, stream_));
    std::uint32_t count = 0;
    std::uint32_t overflow = 0;
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
    result.candidate_rows = candidate_rows;
    result.observed_hits = count;
    result.overflow = overflow != 0 || count > capacity_;
    result.hits.reserve(copied);
    for (const auto &hit : raw)
      result.hits.push_back(HostHit{hit.id, hit.num, hit.den});
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
  cudaEvent_t device_start_ = nullptr;
  cudaEvent_t kernel_start_ = nullptr;
  cudaEvent_t kernel_stop_ = nullptr;
  cudaEvent_t device_stop_ = nullptr;
  DeviceQuery *d_query_ = nullptr;
  DeviceSlice *d_slices_ = nullptr;
  DeviceHit *d_hits_ = nullptr;
  std::uint32_t *d_count_ = nullptr;
  std::uint32_t *d_overflow_ = nullptr;
};

struct Options {
  std::string root;
  std::string mode;
  std::string variant = "all";
  std::string output;
  std::string threshold_order = "70,80";
  std::string durable_script;
  std::string store_root;
  std::string crash_point = "none";
  std::string recovered_delta;
  std::string order = "control-first";
  std::uint64_t expected_hash = 0;
  int expected_epoch = 0;
  int query_index = 512;
  int threshold_num = 7;
  int threshold_den = 10;
  int gpu = 2;
  int runs = 64;
  int process_index = 0;
  int warmup = 8;
  int cycles = 1000;
};

Options parse_args(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto need = [&]() -> std::string {
      if (++index >= argc)
        throw std::runtime_error("missing value for " + argument);
      return argv[index];
    };
    if (argument == "--root")
      options.root = need();
    else if (argument == "--mode")
      options.mode = need();
    else if (argument == "--variant")
      options.variant = need();
    else if (argument == "--output")
      options.output = need();
    else if (argument == "--gpu")
      options.gpu = std::stoi(need());
    else if (argument == "--runs")
      options.runs = std::stoi(need());
    else if (argument == "--process-index")
      options.process_index = std::stoi(need());
    else if (argument == "--warmup")
      options.warmup = std::stoi(need());
    else if (argument == "--cycles")
      options.cycles = std::stoi(need());
    else if (argument == "--threshold-order")
      options.threshold_order = need();
    else if (argument == "--durable-script")
      options.durable_script = need();
    else if (argument == "--store-root")
      options.store_root = need();
    else if (argument == "--crash-point")
      options.crash_point = need();
    else if (argument == "--recovered-delta")
      options.recovered_delta = need();
    else if (argument == "--order")
      options.order = need();
    else if (argument == "--expected-hash")
      options.expected_hash = std::stoull(need());
    else if (argument == "--expected-epoch")
      options.expected_epoch = std::stoi(need());
    else if (argument == "--query-index")
      options.query_index = std::stoi(need());
    else if (argument == "--threshold-num")
      options.threshold_num = std::stoi(need());
    else if (argument == "--threshold-den")
      options.threshold_den = std::stoi(need());
    else
      throw std::runtime_error("unknown argument: " + argument);
  }
  if (options.mode.empty() || options.output.empty())
    throw std::runtime_error("required: --mode --output");
  if (options.mode != "synthetic" &&
      options.mode != "synthetic-concurrent" && options.root.empty())
    throw std::runtime_error("--root is required outside synthetic mode");
  if (options.runs < 1 || options.runs > 64)
    throw std::runtime_error("--runs must be in [1,64]");
  return options;
}

std::vector<std::pair<int, int>> threshold_order(const std::string &text) {
  if (text == "70,80") return {{7, 10}, {4, 5}};
  if (text == "80,70") return {{4, 5}, {7, 10}};
  throw std::runtime_error("threshold order must be 70,80 or 80,70");
}

struct RealInputs {
  MappedFile base_fp;
  MappedFile base_ids;
  MappedFile base_pc;
  MappedFile union_fp;
  MappedFile union_ids;
  MappedFile union_pc;
  MappedFile delta_file;
  MappedFile query_file;
  HostDBView base;
  HostDBView fresh;
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
        base(make_host_view(base_fp, base_ids, base_pc)),
        fresh(make_host_view(union_fp, union_ids, union_pc)),
        delta_owned(load_delta(delta_file)),
        delta(delta_owned.view()),
        queries(query_file) {}
};

std::shared_ptr<DeviceRun> upload_run(int gpu, const HostDBView &host) {
  auto device = std::make_shared<DeviceRun>(gpu, host.rows);
  device->upload(host);
  return device;
}

Snapshot single_snapshot(std::uint64_t epoch,
                         const std::shared_ptr<DeviceRun> &device,
                         const HostDBView &host) {
  Snapshot snapshot;
  snapshot.epoch = epoch;
  snapshot.runs.push_back(SnapshotEntry{device, host});
  return snapshot;
}

struct FragmentedIndex {
  std::vector<OwnedHostDB> owned;
  std::vector<HostDBView> host;
  std::vector<std::shared_ptr<DeviceRun>> device;
  Snapshot snapshot;
};

FragmentedIndex build_fragmented(int gpu, const HostDBView &base_host,
                                 const std::shared_ptr<DeviceRun> &base_device,
                                 const HostDBView &delta, int run_count,
                                 bool upload_now = true) {
  FragmentedIndex output;
  output.owned = split_runs(delta, run_count);
  output.host.reserve(run_count);
  output.device.reserve(run_count);
  output.snapshot.epoch = run_count;
  output.snapshot.runs.push_back(SnapshotEntry{base_device, base_host});
  for (int run = 0; run < run_count; ++run) {
    output.host.push_back(output.owned[run].view());
    auto device = std::make_shared<DeviceRun>(gpu, output.host.back().rows);
    if (upload_now) device->upload(output.host.back());
    output.device.push_back(device);
    if (upload_now)
      output.snapshot.runs.push_back(
          SnapshotEntry{output.device.back(), output.host.back()});
  }
  return output;
}

void require_valid(const RunResult &result, const std::string &where) {
  if (result.overflow)
    throw std::runtime_error(where + " output overflow: " +
                             std::to_string(result.observed_hits));
}

std::vector<HostHit> cpu_scan(const HostDBView &database, const Query &query,
                              int num, int den) {
  std::vector<HostHit> hits;
  const auto range = bounds(database, query, num, den);
  for (std::uint64_t row = range.first; row < range.second; ++row) {
    unsigned intersection = 0;
    for (int word = 0; word < 4; ++word)
      intersection +=
          __builtin_popcountll(query.fp[word] & database.fp[row * 4 + word]);
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

void append_hits(std::vector<HostHit> &target, const std::vector<HostHit> &extra) {
  target.insert(target.end(), extra.begin(), extra.end());
  sort_hits(target);
}

double percentile(std::vector<double> values, double quantile) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const double position = (values.size() - 1) * quantile;
  const auto low = static_cast<std::size_t>(std::floor(position));
  const auto high = static_cast<std::size_t>(std::ceil(position));
  if (low == high) return values[low];
  return values[low] * (high - position) +
         values[high] * (position - low);
}

void write_correctness(const Options &options, RealInputs &inputs) {
  auto base_device = upload_run(options.gpu, inputs.base);
  auto compacted_device = upload_run(options.gpu, inputs.fresh);
  const Snapshot compacted = single_snapshot(64, compacted_device, inputs.fresh);
  const std::array<int, 4> counts{{1, 4, 16, 64}};
  std::vector<FragmentedIndex> variants;
  variants.reserve(counts.size());
  for (int count : counts)
    variants.push_back(build_fragmented(options.gpu, inputs.base, base_device,
                                        inputs.delta, count));
  QueryRuntime runtime(options.gpu);
  std::vector<int> query_indices;
  for (int index = 0; index < 64; ++index) query_indices.push_back(index);
  for (int index = 512; index < 768; ++index) query_indices.push_back(index);
  std::size_t requests = 0;
  std::size_t mismatches = 0;
  std::size_t candidate_mismatches = 0;
  std::size_t cpu_requests = 0;
  std::size_t cpu_mismatches = 0;
  std::ofstream samples(options.output + ".samples.csv");
  samples << "query_index,threshold_num,threshold_den,variant,hit_count,"
             "candidate_rows,result_hash,reference_hash\n";
  for (int query_index : query_indices) {
    const Query &query = inputs.queries[query_index];
    for (const auto threshold : std::array<std::pair<int, int>, 2>{
             std::pair<int, int>{7, 10}, {4, 5}}) {
      const int num = threshold.first;
      const int den = threshold.second;
      const RunResult reference = runtime.run(compacted, query, num, den);
      require_valid(reference, "correctness compacted");
      const std::uint64_t reference_hash = hash_hits(reference.hits);
      samples << query_index << ',' << num << ',' << den << ",compacted,"
              << reference.hits.size() << ',' << reference.candidate_rows << ','
              << reference_hash << ',' << reference_hash << '\n';
      for (std::size_t slot = 0; slot < variants.size(); ++slot) {
        const RunResult result =
            runtime.run(variants[slot].snapshot, query, num, den);
        require_valid(result, "correctness logical runs");
        const auto result_hash = hash_hits(result.hits);
        if (result_hash != reference_hash ||
            result.hits.size() != reference.hits.size())
          ++mismatches;
        if (result.candidate_rows != reference.candidate_rows)
          ++candidate_mismatches;
        samples << query_index << ',' << num << ',' << den << ",runs"
                << counts[slot] << ',' << result.hits.size() << ','
                << result.candidate_rows << ',' << result_hash << ','
                << reference_hash << '\n';
      }
      if (query_index < 4) {
        const auto oracle = cpu_scan(inputs.fresh, query, num, den);
        ++cpu_requests;
        if (hash_hits(oracle) != reference_hash ||
            oracle.size() != reference.hits.size())
          ++cpu_mismatches;
      }
      ++requests;
    }
  }
  std::ofstream output(options.output);
  output << "{\n"
         << "  \"experiment_id\": \"tide_20260827_gate5_exactness\",\n"
         << "  \"requests\": " << requests << ",\n"
         << "  \"variants_per_request\": 5,\n"
         << "  \"logical_mismatches\": " << mismatches << ",\n"
         << "  \"candidate_count_mismatches\": " << candidate_mismatches
         << ",\n"
         << "  \"full_cpu_oracle_requests\": " << cpu_requests << ",\n"
         << "  \"full_cpu_oracle_mismatches\": " << cpu_mismatches << ",\n"
         << "  \"G5_EXACT\": "
         << ((mismatches == 0 && candidate_mismatches == 0 &&
              cpu_mismatches == 0)
                 ? "true"
                 : "false")
         << "\n}\n";
  if (mismatches || candidate_mismatches || cpu_mismatches)
    throw std::runtime_error("correctness gate failed");
}

void write_bench(const Options &options, RealInputs &inputs) {
  if (options.variant != "compacted" && options.variant != "runs")
    throw std::runtime_error("bench variant must be compacted or runs");
  std::shared_ptr<DeviceRun> base_device;
  std::shared_ptr<DeviceRun> compacted_device;
  std::unique_ptr<FragmentedIndex> fragmented;
  Snapshot snapshot;
  if (options.variant == "compacted") {
    compacted_device = upload_run(options.gpu, inputs.fresh);
    snapshot = single_snapshot(options.runs, compacted_device, inputs.fresh);
  } else {
    base_device = upload_run(options.gpu, inputs.base);
    fragmented = std::make_unique<FragmentedIndex>(build_fragmented(
        options.gpu, inputs.base, base_device, inputs.delta, options.runs));
    snapshot = fragmented->snapshot;
  }
  QueryRuntime runtime(options.gpu);
  const auto order = threshold_order(options.threshold_order);
  for (int warmup = 0; warmup < options.warmup; ++warmup) {
    const auto threshold = order[warmup % order.size()];
    const RunResult result = runtime.run(
        snapshot, inputs.queries[warmup % 8], threshold.first, threshold.second);
    require_valid(result, "bench warmup");
  }
  std::ofstream output(options.output);
  output << "process_index,variant,run_count,threshold_num,threshold_den,"
            "query_index,service_ms,kernel_ms,device_ms,candidate_rows,"
            "hit_count,result_hash\n"
         << std::setprecision(12);
  for (const auto threshold : order) {
    for (int query_index = 512; query_index < 768; ++query_index) {
      const RunResult result = runtime.run(snapshot, inputs.queries[query_index],
                                           threshold.first, threshold.second);
      require_valid(result, "bench");
      output << options.process_index << ',' << options.variant << ','
             << options.runs << ',' << threshold.first << ','
             << threshold.second << ',' << query_index << ','
             << result.service_ms << ',' << result.kernel_ms << ','
             << result.device_ms << ',' << result.candidate_rows << ','
             << result.hits.size() << ',' << hash_hits(result.hits) << '\n';
    }
  }
}

struct QuerySpec {
  int query_index;
  int num;
  int den;
};

std::vector<QuerySpec> concurrency_specs() {
  std::vector<QuerySpec> specs;
  for (int query_index = 512; query_index < 544; ++query_index) {
    specs.push_back(QuerySpec{query_index, 7, 10});
    specs.push_back(QuerySpec{query_index, 4, 5});
  }
  return specs;
}

std::vector<std::vector<std::uint64_t>> build_prefix_expectations(
    int gpu, const Snapshot &base_snapshot,
    const std::vector<HostDBView> &delta_runs, const QueryStore &queries,
    const std::vector<QuerySpec> &specs) {
  std::vector<std::vector<std::uint64_t>> expected(
      delta_runs.size() + 1, std::vector<std::uint64_t>(specs.size()));
  QueryRuntime runtime(gpu);
  for (std::size_t slot = 0; slot < specs.size(); ++slot) {
    const auto &spec = specs[slot];
    const Query &query = queries[spec.query_index];
    RunResult base = runtime.run(base_snapshot, query, spec.num, spec.den);
    require_valid(base, "prefix expectation base");
    std::vector<HostHit> accumulated = base.hits;
    expected[0][slot] = hash_hits(accumulated);
    for (std::size_t epoch = 1; epoch <= delta_runs.size(); ++epoch) {
      append_hits(accumulated,
                  cpu_scan(delta_runs[epoch - 1], query, spec.num, spec.den));
      expected[epoch][slot] = hash_hits(accumulated);
    }
  }
  return expected;
}

struct ConcurrentSample {
  int reader = 0;
  int request = 0;
  int slot = 0;
  std::uint64_t epoch = 0;
  double service_ms = 0;
  std::uint64_t observed_hash = 0;
  std::uint64_t expected_hash = 0;
  bool overflow = false;
};

std::string shell_quote(const std::string &value) {
  std::string output = "'";
  for (char character : value) {
    if (character == '\'')
      output += "'\\''";
    else
      output += character;
  }
  output += "'";
  return output;
}

void write_persistent_visible(const Options &options, RealInputs &inputs) {
  if (options.durable_script.empty() || options.store_root.empty())
    throw std::runtime_error(
        "persistent mode requires --durable-script and --store-root");
  namespace fs = std::filesystem;
  fs::create_directories(options.store_root);
  auto base_device = upload_run(options.gpu, inputs.base);
  auto delta_device = std::make_shared<DeviceRun>(options.gpu, inputs.delta.rows);
  auto expected_index = build_fragmented(options.gpu, inputs.base, base_device,
                                         inputs.delta, 1, true);
  QueryRuntime runtime(options.gpu);
  const Query &query = inputs.queries[512];
  const RunResult expected_result =
      runtime.run(expected_index.snapshot, query, 7, 10);
  require_valid(expected_result, "persistent expected");
  const auto expected_hash = hash_hits(expected_result.hits);
  auto initial = std::make_shared<Snapshot>();
  initial->epoch = 0;
  initial->runs.push_back(SnapshotEntry{base_device, inputs.base});
  std::shared_ptr<Snapshot> active = initial;
  cudaStream_t writer_stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&writer_stream, cudaStreamNonBlocking));
  std::vector<double> visible_ms;
  std::vector<double> recovery_ms;
  std::size_t mismatches = 0;
  std::ofstream samples(options.output + ".samples.csv");
  samples << "repetition,persistent_to_query_visible_ms,recovery_ms,result_hash,"
             "expected_hash\n"
          << std::setprecision(12);
  const std::string delta_path =
      options.root + "/data/stage_a/delta_u64x6.bin";
  for (int repetition = 0; repetition < options.cycles; ++repetition) {
    const fs::path store =
        fs::path(options.store_root) / ("rep-" + std::to_string(repetition));
    const fs::path commit_output = store.parent_path() /
                                   ("rep-" + std::to_string(repetition) +
                                    "-commit.json");
    const fs::path commit_error = store.parent_path() /
                                  ("rep-" + std::to_string(repetition) +
                                   "-commit.stderr");
    std::atomic_store_explicit(&active, initial, std::memory_order_release);
    const auto begin = std::chrono::steady_clock::now();
    const std::string command =
        "python3 " + shell_quote(options.durable_script) + " commit --store " +
        shell_quote(store.string()) + " --input " + shell_quote(delta_path) +
        " --epoch 1 > " + shell_quote(commit_output.string()) + " 2> " +
        shell_quote(commit_error.string());
    const int status = std::system(command.c_str());
    if (status != 0)
      throw std::runtime_error("durable commit subprocess failed: " +
                               std::to_string(status));
    delta_device->upload(inputs.delta, writer_stream, true);
    auto next = std::make_shared<Snapshot>(*initial);
    next->epoch = 1;
    next->runs.push_back(SnapshotEntry{delta_device, inputs.delta});
    std::atomic_store_explicit(&active, next, std::memory_order_release);
    const auto captured =
        std::atomic_load_explicit(&active, std::memory_order_acquire);
    const RunResult observed = runtime.run(*captured, query, 7, 10);
    const auto visible = std::chrono::steady_clock::now();
    const auto observed_hash = hash_hits(observed.hits);
    if (observed.overflow || observed_hash != expected_hash) ++mismatches;
    const fs::path recovery_output = store.parent_path() /
                                     ("rep-" + std::to_string(repetition) +
                                      "-recover.json");
    const fs::path recovery_error = store.parent_path() /
                                    ("rep-" + std::to_string(repetition) +
                                     "-recover.stderr");
    const std::string recovery_command =
        "python3 " + shell_quote(options.durable_script) + " recover --store " +
        shell_quote(store.string()) + " > " +
        shell_quote(recovery_output.string()) + " 2> " +
        shell_quote(recovery_error.string());
    const int recovery_status = std::system(recovery_command.c_str());
    const auto recovered = std::chrono::steady_clock::now();
    if (recovery_status != 0)
      throw std::runtime_error("durable recovery subprocess failed");
    const double visible_value = std::chrono::duration<double, std::milli>(
                                     visible - begin)
                                     .count();
    const double recovery_value = std::chrono::duration<double, std::milli>(
                                      recovered - visible)
                                      .count();
    visible_ms.push_back(visible_value);
    recovery_ms.push_back(recovery_value);
    samples << repetition << ',' << visible_value << ',' << recovery_value << ','
            << observed_hash << ',' << expected_hash << '\n';
  }
  CUDA_CHECK(cudaStreamDestroy(writer_stream));
  const double visible_p95 = percentile(visible_ms, 0.95);
  std::ofstream output(options.output);
  output << "{\n"
         << "  \"experiment_id\": \"tide_20260827_gate5_persistent_to_query_visible\",\n"
         << "  \"scope\": \"durable run+manifest subprocess, resident GPU upload, atomic snapshot, exact verification query\",\n"
         << "  \"repetitions\": " << options.cycles << ",\n"
         << "  \"hash_mismatches\": " << mismatches << ",\n"
         << "  \"visible_median_ms\": " << std::setprecision(12)
         << percentile(visible_ms, 0.50) << ",\n"
         << "  \"visible_p95_ms\": " << visible_p95 << ",\n"
         << "  \"visible_max_ms\": "
         << *std::max_element(visible_ms.begin(), visible_ms.end()) << ",\n"
         << "  \"recovery_p95_ms\": " << percentile(recovery_ms, 0.95)
         << ",\n"
         << "  \"gate_ms\": 1905.55,\n"
         << "  \"G5_PERSIST_VISIBLE\": "
         << ((mismatches == 0 && visible_p95 <= 1905.55) ? "true" : "false")
         << "\n}\n";
  if (mismatches || visible_p95 > 1905.55)
    throw std::runtime_error("persistent-to-visible gate failed");
}

void write_concurrent(const Options &options, RealInputs &inputs) {
  constexpr int reader_count = 4;
  constexpr int requests_per_reader = 1024;
  constexpr int total_requests = reader_count * requests_per_reader;
  constexpr int run_count = 64;
  CUDA_CHECK(cudaSetDevice(options.gpu));
  auto base_device = upload_run(options.gpu, inputs.base);
  auto fragmented = build_fragmented(options.gpu, inputs.base, base_device,
                                     inputs.delta, run_count, false);
  auto initial = std::make_shared<Snapshot>();
  initial->epoch = 0;
  initial->runs.push_back(SnapshotEntry{base_device, inputs.base});
  std::shared_ptr<Snapshot> active = initial;
  const auto specs = concurrency_specs();
  const auto expected = build_prefix_expectations(
      options.gpu, *initial, fragmented.host, inputs.queries, specs);

  std::atomic<bool> start{false};
  std::atomic<int> completed{0};
  std::atomic<int> ready{0};
  std::vector<ConcurrentSample> samples(total_requests);
  std::array<std::uint64_t, run_count + 1> epoch_counts{};
  std::vector<std::thread> readers;
  readers.reserve(reader_count);
  for (int reader = 0; reader < reader_count; ++reader) {
    readers.emplace_back([&, reader] {
      CUDA_CHECK(cudaSetDevice(options.gpu));
      QueryRuntime runtime(options.gpu);
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int request = 0; request < requests_per_reader; ++request) {
        const int slot = (reader * requests_per_reader + request) % specs.size();
        const auto snapshot =
            std::atomic_load_explicit(&active, std::memory_order_acquire);
        const auto &spec = specs[slot];
        RunResult result = runtime.run(*snapshot, inputs.queries[spec.query_index],
                                       spec.num, spec.den);
        const int sample_index = reader * requests_per_reader + request;
        samples[sample_index] = ConcurrentSample{
            reader, request, slot, snapshot->epoch, result.service_ms,
            hash_hits(result.hits), expected.at(snapshot->epoch).at(slot),
            result.overflow};
        completed.fetch_add(1, std::memory_order_release);
        if ((splitmix64(sample_index + 20260827) & 7) == 0)
          std::this_thread::sleep_for(std::chrono::microseconds(5));
      }
    });
  }

  std::vector<double> publication_ms;
  publication_ms.reserve(run_count);
  std::thread writer([&] {
    CUDA_CHECK(cudaSetDevice(options.gpu));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    while (ready.load(std::memory_order_acquire) != reader_count)
      std::this_thread::yield();
    start.store(true, std::memory_order_release);
    auto current = initial;
    for (int epoch = 1; epoch <= run_count; ++epoch) {
      const int target = epoch * 63;
      while (completed.load(std::memory_order_acquire) < target)
        std::this_thread::yield();
      const auto begin = std::chrono::steady_clock::now();
      fragmented.device[epoch - 1]->upload(fragmented.host[epoch - 1], stream,
                                            true);
      auto next = std::make_shared<Snapshot>(*current);
      next->epoch = epoch;
      next->runs.push_back(SnapshotEntry{fragmented.device[epoch - 1],
                                         fragmented.host[epoch - 1]});
      std::atomic_store_explicit(&active, next, std::memory_order_release);
      current = next;
      publication_ms.push_back(std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - begin)
                                   .count());
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  });
  for (auto &reader : readers) reader.join();
  writer.join();

  std::size_t mismatches = 0;
  std::size_t overflow = 0;
  std::vector<double> query_ms;
  query_ms.reserve(samples.size());
  for (const auto &sample : samples) {
    if (sample.observed_hash != sample.expected_hash) ++mismatches;
    if (sample.overflow) ++overflow;
    ++epoch_counts.at(sample.epoch);
    query_ms.push_back(sample.service_ms);
  }
  std::ofstream raw(options.output + ".samples.csv");
  raw << "reader,request,slot,epoch,service_ms,observed_hash,expected_hash,"
         "overflow\n"
      << std::setprecision(12);
  for (const auto &sample : samples)
    raw << sample.reader << ',' << sample.request << ',' << sample.slot << ','
        << sample.epoch << ',' << sample.service_ms << ','
        << sample.observed_hash << ',' << sample.expected_hash << ','
        << (sample.overflow ? 1 : 0) << '\n';
  std::ofstream output(options.output);
  output << "{\n"
         << "  \"experiment_id\": \"tide_20260827_gate5_concurrent_snapshot\",\n"
         << "  \"reader_streams\": " << reader_count << ",\n"
         << "  \"queries\": " << samples.size() << ",\n"
         << "  \"publications\": " << publication_ms.size() << ",\n"
         << "  \"hash_mismatches\": " << mismatches << ",\n"
         << "  \"overflow_events\": " << overflow << ",\n"
         << "  \"query_p99_ms\": " << std::setprecision(12)
         << percentile(query_ms, 0.99) << ",\n"
         << "  \"publication_p95_ms\": "
         << percentile(publication_ms, 0.95) << ",\n"
         << "  \"publication_max_ms\": "
         << *std::max_element(publication_ms.begin(), publication_ms.end())
         << ",\n"
         << "  \"observed_epoch_count\": "
         << std::count_if(epoch_counts.begin(), epoch_counts.end(),
                          [](std::uint64_t count) { return count != 0; })
         << ",\n"
         << "  \"G5_SNAPSHOT\": "
         << ((mismatches == 0 && overflow == 0 &&
              publication_ms.size() == run_count)
                 ? "true"
                 : "false")
         << "\n}\n";
  if (mismatches || overflow)
    throw std::runtime_error("concurrent snapshot gate failed");
}

void write_stress(const Options &options, RealInputs &inputs) {
  constexpr int reader_count = 4;
  constexpr int requests_per_reader = 2500;
  constexpr int total_requests = reader_count * requests_per_reader;
  auto base_device = upload_run(options.gpu, inputs.base);
  auto fragmented = build_fragmented(options.gpu, inputs.base, base_device,
                                     inputs.delta, 64, true);
  auto base = std::make_shared<Snapshot>();
  base->epoch = 0;
  base->runs.push_back(SnapshotEntry{base_device, inputs.base});
  auto full = std::make_shared<Snapshot>(fragmented.snapshot);
  full->epoch = 1;
  const auto specs = concurrency_specs();
  std::vector<std::array<std::uint64_t, 2>> expected(specs.size());
  QueryRuntime oracle(options.gpu);
  for (std::size_t slot = 0; slot < specs.size(); ++slot) {
    const auto &spec = specs[slot];
    expected[slot][0] = hash_hits(
        oracle.run(*base, inputs.queries[spec.query_index], spec.num, spec.den)
            .hits);
    expected[slot][1] = hash_hits(
        oracle.run(*full, inputs.queries[spec.query_index], spec.num, spec.den)
            .hits);
  }
  std::shared_ptr<Snapshot> active = base;
  std::atomic<bool> start{false};
  std::atomic<int> ready{0};
  std::atomic<int> completed{0};
  std::atomic<std::uint64_t> mismatches{0};
  std::atomic<std::uint64_t> overflow{0};
  std::vector<std::thread> readers;
  for (int reader = 0; reader < reader_count; ++reader) {
    readers.emplace_back([&, reader] {
      CUDA_CHECK(cudaSetDevice(options.gpu));
      QueryRuntime runtime(options.gpu);
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int request = 0; request < requests_per_reader; ++request) {
        const int slot = (reader * requests_per_reader + request) % specs.size();
        auto snapshot =
            std::atomic_load_explicit(&active, std::memory_order_acquire);
        const auto &spec = specs[slot];
        RunResult result = runtime.run(*snapshot, inputs.queries[spec.query_index],
                                       spec.num, spec.den);
        const int state = snapshot->runs.size() == 1 ? 0 : 1;
        if (hash_hits(result.hits) != expected[slot][state])
          mismatches.fetch_add(1, std::memory_order_relaxed);
        if (result.overflow)
          overflow.fetch_add(1, std::memory_order_relaxed);
        completed.fetch_add(1, std::memory_order_release);
      }
    });
  }
  std::thread writer([&] {
    while (ready.load(std::memory_order_acquire) != reader_count)
      std::this_thread::yield();
    start.store(true, std::memory_order_release);
    for (int cycle = 1; cycle <= options.cycles; ++cycle) {
      const int target = cycle * 9;
      while (completed.load(std::memory_order_acquire) < target)
        std::this_thread::yield();
      auto source = cycle & 1 ? full : base;
      auto next = std::make_shared<Snapshot>(*source);
      next->epoch = cycle;
      std::atomic_store_explicit(&active, next, std::memory_order_release);
    }
  });
  const auto begin = std::chrono::steady_clock::now();
  for (auto &reader : readers) reader.join();
  writer.join();
  const double elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
  std::ofstream output(options.output);
  output << "{\n"
         << "  \"snapshot_captures\": " << total_requests << ",\n"
         << "  \"publication_reset_cycles\": " << options.cycles << ",\n"
         << "  \"hash_mismatches\": " << mismatches.load() << ",\n"
         << "  \"overflow_events\": " << overflow.load() << ",\n"
         << "  \"elapsed_s\": " << std::setprecision(12) << elapsed << ",\n"
         << "  \"G5_STRESS\": "
         << ((mismatches.load() == 0 && overflow.load() == 0 &&
              options.cycles >= 1000)
                 ? "true"
                 : "false")
         << "\n}\n";
  if (mismatches.load() || overflow.load())
    throw std::runtime_error("snapshot stress failed");
}

void copy_slice(DeviceRun &destination, std::uint64_t destination_begin,
                const DeviceRun &source, std::uint64_t source_begin,
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

void initialize_untouched_bins(
    DeviceRun &compacted, const DeviceRun &base, const HostDBView &base_host,
    const std::vector<HostDBView> &delta_host, const HostDBView &union_host,
    cudaStream_t stream) {
  for (int popcount = 0; popcount <= 256; ++popcount) {
    std::uint64_t delta_rows = 0;
    for (const auto &run : delta_host)
      delta_rows += run.cumulative[popcount + 1] - run.cumulative[popcount];
    if (delta_rows) continue;
    const std::uint64_t base_begin = base_host.cumulative[popcount];
    const std::uint64_t base_rows =
        base_host.cumulative[popcount + 1] - base_begin;
    copy_slice(compacted, union_host.cumulative[popcount], base, base_begin,
               base_rows, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

double compact_touched_bins(
    DeviceRun &compacted, const DeviceRun &base,
    const std::vector<std::shared_ptr<DeviceRun>> &delta_device,
    const HostDBView &base_host, const std::vector<HostDBView> &delta_host,
    const HostDBView &union_host, cudaStream_t stream) {
  const auto begin = std::chrono::steady_clock::now();
  for (int popcount = 0; popcount <= 256; ++popcount) {
    std::uint64_t delta_rows = 0;
    for (const auto &run : delta_host)
      delta_rows += run.cumulative[popcount + 1] - run.cumulative[popcount];
    if (!delta_rows) continue;
    std::uint64_t target = union_host.cumulative[popcount];
    const std::uint64_t base_begin = base_host.cumulative[popcount];
    const std::uint64_t base_rows =
        base_host.cumulative[popcount + 1] - base_begin;
    copy_slice(compacted, target, base, base_begin, base_rows, stream);
    target += base_rows;
    for (std::size_t run = 0; run < delta_host.size(); ++run) {
      const std::uint64_t source = delta_host[run].cumulative[popcount];
      const std::uint64_t rows =
          delta_host[run].cumulative[popcount + 1] - source;
      copy_slice(compacted, target, *delta_device[run], source, rows, stream);
      target += rows;
    }
    if (target != union_host.cumulative[popcount + 1])
      throw std::runtime_error("compaction bin cardinality mismatch");
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - begin)
      .count();
}

struct PhaseResult {
  double wall_s = 0;
  std::vector<double> query_ms;
  std::size_t mismatches = 0;
  std::size_t overflow = 0;
};

PhaseResult run_fixed_snapshot_phase(
    int gpu, const std::shared_ptr<Snapshot> &snapshot, const QueryStore &queries,
    const std::vector<QuerySpec> &specs,
    const std::vector<std::uint64_t> &expected, int total_requests) {
  constexpr int readers = 4;
  std::atomic<bool> start{false};
  std::atomic<int> ready{0};
  std::vector<double> latency(total_requests);
  std::vector<unsigned char> mismatch(total_requests);
  std::vector<unsigned char> overflow(total_requests);
  std::vector<std::thread> threads;
  const auto begin = std::chrono::steady_clock::now();
  for (int reader = 0; reader < readers; ++reader) {
    threads.emplace_back([&, reader] {
      CUDA_CHECK(cudaSetDevice(gpu));
      QueryRuntime runtime(gpu);
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int request = reader; request < total_requests; request += readers) {
        const int slot = request % specs.size();
        const auto &spec = specs[slot];
        const RunResult result = runtime.run(
            *snapshot, queries[spec.query_index], spec.num, spec.den);
        latency[request] = result.service_ms;
        mismatch[request] = hash_hits(result.hits) != expected[slot];
        overflow[request] = result.overflow;
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != readers)
    std::this_thread::yield();
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) thread.join();
  PhaseResult result;
  result.wall_s = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - begin)
                      .count();
  result.query_ms = std::move(latency);
  result.mismatches = std::accumulate(mismatch.begin(), mismatch.end(), 0ull);
  result.overflow = std::accumulate(overflow.begin(), overflow.end(), 0ull);
  return result;
}

void write_mixed(const Options &options, RealInputs &inputs) {
  constexpr int total_requests = 4096;
  constexpr int reader_count = 4;
  auto base_device = upload_run(options.gpu, inputs.base);
  auto fragmented = build_fragmented(options.gpu, inputs.base, base_device,
                                     inputs.delta, 64, true);
  auto old_snapshot = std::make_shared<Snapshot>(fragmented.snapshot);
  old_snapshot->epoch = 64;
  auto compacted_device =
      std::make_shared<DeviceRun>(options.gpu, inputs.fresh.rows);
  cudaStream_t writer_stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&writer_stream, cudaStreamNonBlocking));
  initialize_untouched_bins(*compacted_device, *base_device, inputs.base,
                            fragmented.host, inputs.fresh, writer_stream);
  const auto specs = concurrency_specs();
  QueryRuntime oracle(options.gpu);
  std::vector<std::uint64_t> expected(specs.size());
  for (std::size_t slot = 0; slot < specs.size(); ++slot) {
    const auto &spec = specs[slot];
    const RunResult result = oracle.run(*old_snapshot,
                                        inputs.queries[spec.query_index],
                                        spec.num, spec.den);
    require_valid(result, "mixed expected");
    expected[slot] = hash_hits(result.hits);
  }
  const PhaseResult query_only = run_fixed_snapshot_phase(
      options.gpu, old_snapshot, inputs.queries, specs, expected, total_requests);

  std::shared_ptr<Snapshot> active = old_snapshot;
  std::atomic<bool> start{false};
  std::atomic<int> ready{0};
  std::atomic<int> completed{0};
  std::vector<double> mixed_latency(total_requests);
  std::vector<unsigned char> mismatch(total_requests);
  std::vector<unsigned char> overflow(total_requests);
  std::vector<std::thread> readers;
  for (int reader = 0; reader < reader_count; ++reader) {
    readers.emplace_back([&, reader] {
      CUDA_CHECK(cudaSetDevice(options.gpu));
      QueryRuntime runtime(options.gpu);
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int request = reader; request < total_requests;
           request += reader_count) {
        const int slot = request % specs.size();
        const auto snapshot =
            std::atomic_load_explicit(&active, std::memory_order_acquire);
        const auto &spec = specs[slot];
        const RunResult result = runtime.run(
            *snapshot, inputs.queries[spec.query_index], spec.num, spec.den);
        mixed_latency[request] = result.service_ms;
        mismatch[request] = hash_hits(result.hits) != expected[slot];
        overflow[request] = result.overflow;
        completed.fetch_add(1, std::memory_order_release);
      }
    });
  }
  double compaction_ms = 0;
  std::thread writer([&] {
    CUDA_CHECK(cudaSetDevice(options.gpu));
    while (ready.load(std::memory_order_acquire) != reader_count)
      std::this_thread::yield();
    start.store(true, std::memory_order_release);
    while (completed.load(std::memory_order_acquire) < 1024)
      std::this_thread::yield();
    compaction_ms = compact_touched_bins(
        *compacted_device, *base_device, fragmented.device, inputs.base,
        fragmented.host, inputs.fresh, writer_stream);
    auto next = std::make_shared<Snapshot>(
        single_snapshot(65, compacted_device, inputs.fresh));
    std::atomic_store_explicit(&active, next, std::memory_order_release);
  });
  const auto mixed_begin = std::chrono::steady_clock::now();
  for (auto &reader : readers) reader.join();
  writer.join();
  const double mixed_wall_s = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - mixed_begin)
                                  .count();
  CUDA_CHECK(cudaStreamDestroy(writer_stream));
  const std::size_t mixed_mismatches =
      std::accumulate(mismatch.begin(), mismatch.end(), 0ull);
  const std::size_t mixed_overflow =
      std::accumulate(overflow.begin(), overflow.end(), 0ull);
  const double query_only_qps = total_requests / query_only.wall_s;
  const double mixed_qps = total_requests / mixed_wall_s;
  const double throughput_ratio = mixed_qps / query_only_qps;
  const double query_only_p99 = percentile(query_only.query_ms, 0.99);
  const double mixed_p99 = percentile(mixed_latency, 0.99);
  const double p99_ratio = mixed_p99 / query_only_p99;
  std::ofstream samples(options.output + ".samples.csv");
  samples << "request,query_only_ms,mixed_ms\n" << std::setprecision(12);
  for (int request = 0; request < total_requests; ++request)
    samples << request << ',' << query_only.query_ms[request] << ','
            << mixed_latency[request] << '\n';
  const bool pass = query_only.mismatches == 0 && query_only.overflow == 0 &&
                    mixed_mismatches == 0 && mixed_overflow == 0 &&
                    throughput_ratio >= 0.90 && p99_ratio <= 1.25;
  std::ofstream output(options.output);
  output << "{\n"
         << "  \"queries_per_phase\": " << total_requests << ",\n"
         << "  \"reader_streams\": " << reader_count << ",\n"
         << "  \"query_only_mismatches\": " << query_only.mismatches << ",\n"
         << "  \"mixed_mismatches\": " << mixed_mismatches << ",\n"
         << "  \"query_only_overflow\": " << query_only.overflow << ",\n"
         << "  \"mixed_overflow\": " << mixed_overflow << ",\n"
         << "  \"query_only_qps\": " << std::setprecision(12)
         << query_only_qps << ",\n"
         << "  \"mixed_qps\": " << mixed_qps << ",\n"
         << "  \"mixed_over_query_only\": " << throughput_ratio << ",\n"
         << "  \"query_only_p99_ms\": " << query_only_p99 << ",\n"
         << "  \"mixed_p99_ms\": " << mixed_p99 << ",\n"
         << "  \"mixed_over_query_only_p99\": " << p99_ratio << ",\n"
         << "  \"compaction_ms\": " << compaction_ms << ",\n"
         << "  \"G5_CONCURRENT_QUERY\": " << (pass ? "true" : "false")
         << ",\n"
         << "  \"G5_COMPACTION\": " << (pass ? "true" : "false")
         << "\n}\n";
  if (!pass) throw std::runtime_error("mixed service gate failed");
}

void append_owned(OwnedHostDB &database, std::uint64_t id,
                  const std::array<std::uint64_t, 4> &fp,
                  std::uint16_t pc) {
  database.ids.push_back(id);
  database.pc.push_back(pc);
  database.fp.insert(database.fp.end(), fp.begin(), fp.end());
}

void write_synthetic(const Options &options) {
  constexpr int base_rows = 8192;
  constexpr int delta_rows = 1024;
  struct Row {
    std::uint64_t id;
    std::array<std::uint64_t, 4> fp;
    std::uint16_t pc;
  };
  std::mt19937_64 generator(20260827);
  std::array<std::uint64_t, 4> repeated{};
  repeated[0] = std::numeric_limits<std::uint64_t>::max();
  std::vector<Row> base_values;
  std::vector<Row> delta_values;
  for (int index = 0; index < base_rows + delta_rows; ++index) {
    Row row;
    row.id = index + 1;
    if (index < 512)
      row.fp = repeated;
    else
      for (auto &word : row.fp) word = generator();
    unsigned pc = 0;
    for (auto word : row.fp) pc += __builtin_popcountll(word);
    row.pc = static_cast<std::uint16_t>(pc);
    (index < base_rows ? base_values : delta_values).push_back(row);
  }
  auto order = [](const Row &left, const Row &right) {
    return std::tie(left.pc, left.id) < std::tie(right.pc, right.id);
  };
  std::sort(base_values.begin(), base_values.end(), order);
  std::sort(delta_values.begin(), delta_values.end(), order);
  std::vector<Row> union_values = base_values;
  union_values.insert(union_values.end(), delta_values.begin(), delta_values.end());
  std::sort(union_values.begin(), union_values.end(), order);
  OwnedHostDB base_owned;
  OwnedHostDB delta_owned;
  OwnedHostDB union_owned;
  for (const auto &row : base_values)
    append_owned(base_owned, row.id, row.fp, row.pc);
  for (const auto &row : delta_values)
    append_owned(delta_owned, row.id, row.fp, row.pc);
  for (const auto &row : union_values)
    append_owned(union_owned, row.id, row.fp, row.pc);
  const HostDBView base_host = base_owned.view();
  const HostDBView delta_host = delta_owned.view();
  const HostDBView union_host = union_owned.view();
  auto base_device = upload_run(options.gpu, base_host);
  auto union_device = upload_run(options.gpu, union_host);
  const Snapshot compacted = single_snapshot(64, union_device, union_host);
  auto fragmented = build_fragmented(options.gpu, base_host, base_device,
                                     delta_host, 64, true);
  QueryRuntime runtime(options.gpu);
  std::vector<Query> queries;
  Query repeated_query{};
  repeated_query.id = 1;
  std::copy(repeated.begin(), repeated.end(), repeated_query.fp);
  repeated_query.pc = 64;
  queries.push_back(repeated_query);
  for (int index = 0; index < 7; ++index) {
    Query query{};
    query.id = 100000 + index;
    unsigned pc = 0;
    for (auto &word : query.fp) {
      word = generator();
      pc += __builtin_popcountll(word);
    }
    query.pc = static_cast<std::uint16_t>(pc);
    queries.push_back(query);
  }
  std::size_t mismatches = 0;
  std::size_t candidate_mismatches = 0;
  for (const Query &query : queries) {
    for (const auto threshold : std::array<std::pair<int, int>, 2>{
             std::pair<int, int>{7, 10}, {4, 5}}) {
      const auto oracle = cpu_scan(union_host, query, threshold.first,
                                   threshold.second);
      const auto flat = runtime.run(compacted, query, threshold.first,
                                    threshold.second);
      const auto runs = runtime.run(fragmented.snapshot, query, threshold.first,
                                    threshold.second);
      if (hash_hits(oracle) != hash_hits(flat.hits) ||
          hash_hits(oracle) != hash_hits(runs.hits) || flat.overflow ||
          runs.overflow)
        ++mismatches;
      if (flat.candidate_rows != runs.candidate_rows)
        ++candidate_mismatches;
    }
  }
  QueryRuntime limited(options.gpu, 128);
  const auto overflow = limited.run(fragmented.snapshot, repeated_query, 7, 10);
  const bool overflow_detected = overflow.overflow && overflow.observed_hits >= 512;
  std::ofstream output(options.output);
  output << "{\n"
         << "  \"queries\": " << queries.size() << ",\n"
         << "  \"thresholds\": 2,\n"
         << "  \"runs\": 64,\n"
         << "  \"mismatches\": " << mismatches << ",\n"
         << "  \"candidate_count_mismatches\": " << candidate_mismatches
         << ",\n"
         << "  \"overflow_detected\": "
         << (overflow_detected ? "true" : "false") << ",\n"
         << "  \"overflow_count\": " << overflow.observed_hits << ",\n"
         << "  \"G5_SYNTHETIC\": "
         << ((mismatches == 0 && candidate_mismatches == 0 &&
              overflow_detected)
                 ? "true"
                 : "false")
         << "\n}\n";
  if (mismatches || candidate_mismatches || !overflow_detected)
    throw std::runtime_error("synthetic gate failed");
}

void write_synthetic_concurrent(const Options &options) {
  constexpr int base_rows = 4096;
  constexpr int delta_rows = 512;
  constexpr int reader_count = 4;
  constexpr int requests_per_reader = 64;
  struct Row {
    std::uint64_t id;
    std::array<std::uint64_t, 4> fp;
    std::uint16_t pc;
  };
  std::mt19937_64 generator(2026082705);
  std::vector<Row> base_values;
  std::vector<Row> delta_values;
  for (int index = 0; index < base_rows + delta_rows; ++index) {
    Row row;
    row.id = index + 1;
    unsigned pc = 0;
    for (auto &word : row.fp) {
      word = generator();
      pc += __builtin_popcountll(word);
    }
    row.pc = static_cast<std::uint16_t>(pc);
    (index < base_rows ? base_values : delta_values).push_back(row);
  }
  auto order = [](const Row &left, const Row &right) {
    return std::tie(left.pc, left.id) < std::tie(right.pc, right.id);
  };
  std::sort(base_values.begin(), base_values.end(), order);
  std::sort(delta_values.begin(), delta_values.end(), order);
  OwnedHostDB base_owned;
  OwnedHostDB delta_owned;
  for (const auto &row : base_values)
    append_owned(base_owned, row.id, row.fp, row.pc);
  for (const auto &row : delta_values)
    append_owned(delta_owned, row.id, row.fp, row.pc);
  const HostDBView base_host = base_owned.view();
  const HostDBView delta_host = delta_owned.view();
  auto base_device = upload_run(options.gpu, base_host);
  auto fragmented = build_fragmented(options.gpu, base_host, base_device,
                                     delta_host, 64, false);
  auto initial = std::make_shared<Snapshot>();
  initial->epoch = 0;
  initial->runs.push_back(SnapshotEntry{base_device, base_host});
  std::vector<Query> queries(16);
  for (std::size_t index = 0; index < queries.size(); ++index) {
    queries[index].id = 100000 + index;
    unsigned pc = 0;
    for (auto &word : queries[index].fp) {
      word = generator();
      pc += __builtin_popcountll(word);
    }
    queries[index].pc = static_cast<std::uint16_t>(pc);
  }
  std::vector<std::vector<std::uint64_t>> expected(
      65, std::vector<std::uint64_t>(queries.size()));
  QueryRuntime oracle(options.gpu);
  for (std::size_t slot = 0; slot < queries.size(); ++slot) {
    const int num = slot & 1 ? 4 : 7;
    const int den = slot & 1 ? 5 : 10;
    auto hits = oracle.run(*initial, queries[slot], num, den).hits;
    expected[0][slot] = hash_hits(hits);
    for (int epoch = 1; epoch <= 64; ++epoch) {
      append_hits(hits, cpu_scan(fragmented.host[epoch - 1], queries[slot],
                                 num, den));
      expected[epoch][slot] = hash_hits(hits);
    }
  }
  std::shared_ptr<Snapshot> active = initial;
  std::atomic<bool> start{false};
  std::atomic<int> ready{0};
  std::atomic<int> completed{0};
  std::atomic<std::uint64_t> mismatches{0};
  std::atomic<std::uint64_t> overflow{0};
  std::vector<std::thread> readers;
  for (int reader = 0; reader < reader_count; ++reader) {
    readers.emplace_back([&, reader] {
      CUDA_CHECK(cudaSetDevice(options.gpu));
      QueryRuntime runtime(options.gpu);
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int request = 0; request < requests_per_reader; ++request) {
        const int slot = (reader * requests_per_reader + request) % queries.size();
        const int num = slot & 1 ? 4 : 7;
        const int den = slot & 1 ? 5 : 10;
        const auto snapshot =
            std::atomic_load_explicit(&active, std::memory_order_acquire);
        const RunResult result = runtime.run(*snapshot, queries[slot], num, den);
        if (hash_hits(result.hits) != expected[snapshot->epoch][slot])
          mismatches.fetch_add(1, std::memory_order_relaxed);
        if (result.overflow)
          overflow.fetch_add(1, std::memory_order_relaxed);
        completed.fetch_add(1, std::memory_order_release);
      }
    });
  }
  std::thread writer([&] {
    CUDA_CHECK(cudaSetDevice(options.gpu));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    while (ready.load(std::memory_order_acquire) != reader_count)
      std::this_thread::yield();
    start.store(true, std::memory_order_release);
    auto current = initial;
    for (int epoch = 1; epoch <= 64; ++epoch) {
      while (completed.load(std::memory_order_acquire) < epoch * 3)
        std::this_thread::yield();
      fragmented.device[epoch - 1]->upload(fragmented.host[epoch - 1], stream,
                                            true);
      auto next = std::make_shared<Snapshot>(*current);
      next->epoch = epoch;
      next->runs.push_back(SnapshotEntry{fragmented.device[epoch - 1],
                                         fragmented.host[epoch - 1]});
      std::atomic_store_explicit(&active, next, std::memory_order_release);
      current = next;
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  });
  for (auto &reader : readers) reader.join();
  writer.join();
  std::ofstream output(options.output);
  output << "{\n"
         << "  \"reader_streams\": " << reader_count << ",\n"
         << "  \"queries\": " << reader_count * requests_per_reader << ",\n"
         << "  \"publications\": 64,\n"
         << "  \"hash_mismatches\": " << mismatches.load() << ",\n"
         << "  \"overflow_events\": " << overflow.load() << ",\n"
         << "  \"G5_SYNTHETIC_CONCURRENT\": "
         << ((mismatches.load() == 0 && overflow.load() == 0) ? "true"
                                                               : "false")
         << "\n}\n";
  if (mismatches.load() || overflow.load())
    throw std::runtime_error("synthetic concurrent gate failed");
}

class CanonicalScratch {
 public:
  explicit CanonicalScratch(int gpu) : gpu_(gpu) {
    CUDA_CHECK(cudaSetDevice(gpu_));
    CUDA_CHECK(cudaMalloc(&merge_, 64 * sizeof(MergeSlice)));
  }
  ~CanonicalScratch() {
    cudaSetDevice(gpu_);
    if (merge_) cudaFree(merge_);
  }
  CanonicalScratch(const CanonicalScratch &) = delete;
  CanonicalScratch &operator=(const CanonicalScratch &) = delete;
  MergeSlice *merge() const { return merge_; }

 private:
  int gpu_;
  MergeSlice *merge_ = nullptr;
};

double canonical_compact(
    DeviceRun &compacted, const DeviceRun &base,
    const std::vector<std::shared_ptr<DeviceRun>> &delta_device,
    const HostDBView &base_host, const std::vector<HostDBView> &delta_host,
    const HostDBView &union_host, CanonicalScratch &scratch,
    cudaStream_t stream) {
  if (delta_device.size() != delta_host.size() || delta_device.size() > 64)
    throw std::runtime_error("invalid canonical compaction run set");
  const auto begin = std::chrono::steady_clock::now();
  std::vector<MergeSlice> slices;
  slices.reserve(delta_host.size());
  for (int popcount = 0; popcount <= 256; ++popcount) {
    const std::uint64_t target_begin = union_host.cumulative[popcount];
    const std::uint64_t base_begin = base_host.cumulative[popcount];
    const std::uint64_t base_rows =
        base_host.cumulative[popcount + 1] - base_begin;
    copy_slice(compacted, target_begin, base, base_begin, base_rows, stream);
    slices.clear();
    std::uint64_t delta_rows = 0;
    for (std::size_t run = 0; run < delta_host.size(); ++run) {
      const std::uint64_t source = delta_host[run].cumulative[popcount];
      const std::uint64_t rows =
          delta_host[run].cumulative[popcount + 1] - source;
      if (!rows) continue;
      slices.push_back(MergeSlice{delta_device[run]->fp + source * 4,
                                  delta_device[run]->ids + source,
                                  delta_device[run]->pc + source, rows});
      delta_rows += rows;
    }
    if (base_rows + delta_rows !=
        union_host.cumulative[popcount + 1] - target_begin)
      throw std::runtime_error("canonical compaction bin cardinality mismatch");
    if (!delta_rows) continue;
    CUDA_CHECK(cudaMemcpyAsync(scratch.merge(), slices.data(),
                               slices.size() * sizeof(MergeSlice),
                               cudaMemcpyHostToDevice, stream));
    merge_delta_bin_kernel<<<1, 1, 0, stream>>>(
        scratch.merge(), static_cast<int>(slices.size()), compacted.fp,
        compacted.ids, compacted.pc, target_begin + base_rows,
        static_cast<std::uint16_t>(popcount), delta_rows);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - begin)
      .count();
}

struct CompareResult {
  std::uint64_t row_mismatches = 0;
  std::uint64_t byte_mismatches = 0;
};

CompareResult compare_databases(int gpu, const DeviceRun &left,
                                const DeviceRun &right) {
  if (left.rows != right.rows)
    throw std::runtime_error("database comparison row mismatch");
  CUDA_CHECK(cudaSetDevice(gpu));
  unsigned long long *d_rows = nullptr;
  unsigned long long *d_bytes = nullptr;
  CUDA_CHECK(cudaMalloc(&d_rows, sizeof(*d_rows)));
  CUDA_CHECK(cudaMalloc(&d_bytes, sizeof(*d_bytes)));
  CUDA_CHECK(cudaMemset(d_rows, 0, sizeof(*d_rows)));
  CUDA_CHECK(cudaMemset(d_bytes, 0, sizeof(*d_bytes)));
  const unsigned blocks = static_cast<unsigned>((left.rows + BLOCK - 1) / BLOCK);
  compare_database_kernel<<<blocks, BLOCK>>>(
      left.fp, left.ids, left.pc, right.fp, right.ids, right.pc, left.rows,
      d_rows, d_bytes);
  CUDA_CHECK(cudaGetLastError());
  CompareResult result;
  CUDA_CHECK(cudaMemcpy(&result.row_mismatches, d_rows, sizeof(*d_rows),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&result.byte_mismatches, d_bytes, sizeof(*d_bytes),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_rows));
  CUDA_CHECK(cudaFree(d_bytes));
  return result;
}

OwnedHostDB prefix_from_delta(const HostDBView &delta, int epoch) {
  struct Row {
    std::array<std::uint64_t, 4> fp;
    std::uint64_t id;
    std::uint16_t pc;
  };
  std::vector<Row> rows;
  rows.reserve(delta.rows * epoch / 64 + 64);
  for (std::uint64_t row = 0; row < delta.rows; ++row) {
    if (static_cast<int>(splitmix64(delta.ids[row]) % 64) >= epoch) continue;
    rows.push_back(Row{{delta.fp[row * 4], delta.fp[row * 4 + 1],
                        delta.fp[row * 4 + 2], delta.fp[row * 4 + 3]},
                       delta.ids[row], delta.pc[row]});
  }
  std::sort(rows.begin(), rows.end(), [](const Row &left, const Row &right) {
    return std::tie(left.pc, left.id) < std::tie(right.pc, right.id);
  });
  OwnedHostDB output;
  output.fp.reserve(rows.size() * 4);
  output.ids.reserve(rows.size());
  output.pc.reserve(rows.size());
  for (const auto &row : rows) {
    output.ids.push_back(row.id);
    output.pc.push_back(row.pc);
    output.fp.insert(output.fp.end(), row.fp.begin(), row.fp.end());
  }
  return output;
}

struct OracleState {
  std::uint64_t hash = 0;
  std::uint64_t hits = 0;
};

std::vector<QuerySpec> closure_specs() {
  std::vector<QuerySpec> specs;
  for (int query : std::array<int, 8>{{0, 1, 2, 3, 512, 513, 640, 767}}) {
    specs.push_back(QuerySpec{query, 7, 10});
    specs.push_back(QuerySpec{query, 4, 5});
  }
  return specs;
}

std::vector<std::vector<OracleState>> build_incremental_cpu_oracle(
    const RealInputs &inputs, const std::vector<OwnedHostDB> &arrival,
    const std::vector<QuerySpec> &specs) {
  std::vector<std::vector<OracleState>> output(
      65, std::vector<OracleState>(specs.size()));
  for (std::size_t slot = 0; slot < specs.size(); ++slot) {
    const auto &spec = specs[slot];
    std::vector<HostHit> hits = cpu_scan(
        inputs.base, inputs.queries[spec.query_index], spec.num, spec.den);
    sort_hits(hits);
    output[0][slot] = OracleState{hash_hits(hits), hits.size()};
    for (int epoch = 1; epoch <= 64; ++epoch) {
      const auto delta_hits = cpu_scan(arrival[epoch - 1].view(),
                                       inputs.queries[spec.query_index], spec.num,
                                       spec.den);
      hits.insert(hits.end(), delta_hits.begin(), delta_hits.end());
      sort_hits(hits);
      output[epoch][slot] = OracleState{hash_hits(hits), hits.size()};
    }
  }
  return output;
}

void write_exact_closure(const Options &options, RealInputs &inputs) {
  const auto specs = closure_specs();
  auto arrival = split_runs(inputs.delta, 64);
  const auto oracle = build_incremental_cpu_oracle(inputs, arrival, specs);
  auto base_device = upload_run(options.gpu, inputs.base);
  const Snapshot base_snapshot = single_snapshot(0, base_device, inputs.base);
  QueryRuntime runtime(options.gpu);
  std::ofstream samples(options.output + ".samples.csv");
  samples << "epoch,layout,query_index,threshold_num,threshold_den,"
             "gpu_hits,cpu_hits,gpu_hash,cpu_hash,overflow,match\n";
  std::uint64_t checks = 0;
  std::uint64_t mismatches = 0;
  std::uint64_t overflows = 0;
  for (int epoch = 0; epoch <= 64; ++epoch) {
    OwnedHostDB prefix = prefix_from_delta(inputs.delta, epoch);
    for (int layout : std::array<int, 4>{{1, 4, 16, 64}}) {
      std::unique_ptr<FragmentedIndex> fragmented;
      const Snapshot *snapshot = &base_snapshot;
      if (epoch) {
        fragmented = std::make_unique<FragmentedIndex>(build_fragmented(
            options.gpu, inputs.base, base_device, prefix.view(), layout, true));
        fragmented->snapshot.epoch = epoch;
        snapshot = &fragmented->snapshot;
      }
      for (std::size_t slot = 0; slot < specs.size(); ++slot) {
        const auto &spec = specs[slot];
        const RunResult result = runtime.run(*snapshot,
                                             inputs.queries[spec.query_index],
                                             spec.num, spec.den);
        const std::uint64_t hash = hash_hits(result.hits);
        const bool match = !result.overflow &&
                           hash == oracle[epoch][slot].hash &&
                           result.hits.size() == oracle[epoch][slot].hits;
        ++checks;
        mismatches += !match;
        overflows += result.overflow;
        samples << epoch << ',' << layout << ',' << spec.query_index << ','
                << spec.num << ',' << spec.den << ',' << result.hits.size()
                << ',' << oracle[epoch][slot].hits << ',' << hash << ','
                << oracle[epoch][slot].hash << ',' << (result.overflow ? 1 : 0)
                << ',' << (match ? 1 : 0) << '\n';
      }
    }
  }

  auto fragmented = build_fragmented(options.gpu, inputs.base, base_device,
                                     inputs.delta, 64, true);
  auto compacted_device =
      std::make_shared<DeviceRun>(options.gpu, inputs.fresh.rows);
  CanonicalScratch scratch(options.gpu);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const double compaction_ms = canonical_compact(
      *compacted_device, *base_device, fragmented.device, inputs.base,
      fragmented.host, inputs.fresh, scratch, stream);
  CUDA_CHECK(cudaStreamDestroy(stream));
  auto fresh_device = upload_run(options.gpu, inputs.fresh);
  const CompareResult comparison =
      compare_databases(options.gpu, *compacted_device, *fresh_device);
  const Snapshot compacted =
      single_snapshot(65, compacted_device, inputs.fresh);
  std::uint64_t post_checks = 0;
  std::uint64_t post_mismatches = 0;
  for (std::size_t slot = 0; slot < specs.size(); ++slot) {
    const auto &spec = specs[slot];
    const RunResult result = runtime.run(compacted,
                                         inputs.queries[spec.query_index],
                                         spec.num, spec.den);
    const std::uint64_t hash = hash_hits(result.hits);
    const bool match = !result.overflow && hash == oracle[64][slot].hash &&
                       result.hits.size() == oracle[64][slot].hits;
    ++post_checks;
    post_mismatches += !match;
    overflows += result.overflow;
    samples << 65 << ",compacted," << spec.query_index << ',' << spec.num << ','
            << spec.den << ',' << result.hits.size() << ','
            << oracle[64][slot].hits << ',' << hash << ','
            << oracle[64][slot].hash << ',' << (result.overflow ? 1 : 0) << ','
            << (match ? 1 : 0) << '\n';
  }
  std::size_t recovery_slot = 0;
  for (; recovery_slot < specs.size(); ++recovery_slot)
    if (specs[recovery_slot].query_index == 512 &&
        specs[recovery_slot].num == 7)
      break;
  const bool pass = mismatches == 0 && post_mismatches == 0 && overflows == 0 &&
                    comparison.row_mismatches == 0 &&
                    comparison.byte_mismatches == 0 && checks == 4160 &&
                    post_checks == 16;
  std::ofstream output(options.output);
  output << "{\n"
         << "  \"experiment_id\": \"tide_20260828_gate5_blocker_exact\",\n"
         << "  \"oracle_queries\": 8,\n"
         << "  \"thresholds\": 2,\n"
         << "  \"epochs\": 65,\n"
         << "  \"layouts\": 4,\n"
         << "  \"state_checks\": " << checks << ",\n"
         << "  \"state_mismatches\": " << mismatches << ",\n"
         << "  \"post_compaction_checks\": " << post_checks << ",\n"
         << "  \"post_compaction_mismatches\": " << post_mismatches << ",\n"
         << "  \"overflow_events\": " << overflows << ",\n"
         << "  \"compaction_ms\": " << std::setprecision(12)
         << compaction_ms << ",\n"
         << "  \"compaction_row_mismatches\": "
         << comparison.row_mismatches << ",\n"
         << "  \"compaction_byte_mismatches\": "
         << comparison.byte_mismatches << ",\n"
         << "  \"recovery_query_index\": 512,\n"
         << "  \"recovery_threshold_num\": 7,\n"
         << "  \"recovery_threshold_den\": 10,\n"
         << "  \"recovery_base_hash\": " << oracle[0][recovery_slot].hash
         << ",\n"
         << "  \"recovery_union_hash\": " << oracle[64][recovery_slot].hash
         << ",\n"
         << "  \"B5_EXACT_CLOSURE\": " << (pass ? "true" : "false")
         << "\n}\n";
  if (!pass) throw std::runtime_error("blocker exactness closure failed");
}

struct MixedClosureResult {
  double wall_s = 0;
  double compaction_ms = 0;
  std::vector<double> query_ms;
  std::size_t mismatches = 0;
  std::size_t overflow = 0;
  std::uint64_t captures_during_compaction = 0;
  double max_capture_us = 0;
};

MixedClosureResult run_canonical_mixed(
    int gpu, const std::shared_ptr<Snapshot> &old_snapshot,
    const QueryStore &queries, const std::vector<QuerySpec> &specs,
    const std::vector<std::uint64_t> &expected, DeviceRun &compacted,
    const DeviceRun &base,
    const std::vector<std::shared_ptr<DeviceRun>> &delta_device,
    const HostDBView &base_host, const std::vector<HostDBView> &delta_host,
    const HostDBView &union_host, CanonicalScratch &scratch) {
  constexpr int total_requests = 4096;
  constexpr int readers = 4;
  std::shared_ptr<Snapshot> active = old_snapshot;
  std::atomic<bool> start{false};
  std::atomic<bool> compacting{false};
  std::atomic<int> ready{0};
  std::atomic<int> completed{0};
  std::atomic<std::uint64_t> capture_count{0};
  std::atomic<std::uint64_t> max_capture_ns{0};
  std::vector<double> latency(total_requests);
  std::vector<unsigned char> mismatch(total_requests);
  std::vector<unsigned char> overflow(total_requests);
  std::vector<std::thread> threads;
  for (int reader = 0; reader < readers; ++reader) {
    threads.emplace_back([&, reader] {
      CUDA_CHECK(cudaSetDevice(gpu));
      QueryRuntime runtime(gpu);
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int request = reader; request < total_requests; request += readers) {
        const auto capture_begin = std::chrono::steady_clock::now();
        const auto snapshot =
            std::atomic_load_explicit(&active, std::memory_order_acquire);
        const auto capture_end = std::chrono::steady_clock::now();
        if (compacting.load(std::memory_order_acquire)) {
          capture_count.fetch_add(1, std::memory_order_relaxed);
          const auto ns = static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  capture_end - capture_begin)
                  .count());
          std::uint64_t previous = max_capture_ns.load();
          while (previous < ns &&
                 !max_capture_ns.compare_exchange_weak(previous, ns)) {
          }
        }
        const int slot = request % specs.size();
        const auto &spec = specs[slot];
        const RunResult result = runtime.run(
            *snapshot, queries[spec.query_index], spec.num, spec.den);
        latency[request] = result.service_ms;
        mismatch[request] = hash_hits(result.hits) != expected[slot];
        overflow[request] = result.overflow;
        completed.fetch_add(1, std::memory_order_release);
      }
    });
  }
  double compaction_ms = 0;
  std::thread writer([&] {
    CUDA_CHECK(cudaSetDevice(gpu));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    while (ready.load(std::memory_order_acquire) != readers)
      std::this_thread::yield();
    start.store(true, std::memory_order_release);
    while (completed.load(std::memory_order_acquire) < 1024)
      std::this_thread::yield();
    compacting.store(true, std::memory_order_release);
    compaction_ms = canonical_compact(compacted, base, delta_device, base_host,
                                      delta_host, union_host, scratch, stream);
    compacting.store(false, std::memory_order_release);
    auto next = std::make_shared<Snapshot>(
        single_snapshot(65, old_snapshot->runs.front().device, base_host));
    next = std::make_shared<Snapshot>(
        single_snapshot(65, std::shared_ptr<DeviceRun>(&compacted,
                                                       [](DeviceRun *) {}),
                        union_host));
    std::atomic_store_explicit(&active, next, std::memory_order_release);
    CUDA_CHECK(cudaStreamDestroy(stream));
  });
  const auto begin = std::chrono::steady_clock::now();
  for (auto &thread : threads) thread.join();
  writer.join();
  MixedClosureResult result;
  result.wall_s = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - begin)
                      .count();
  result.compaction_ms = compaction_ms;
  result.query_ms = std::move(latency);
  result.mismatches = std::accumulate(mismatch.begin(), mismatch.end(), 0ull);
  result.overflow = std::accumulate(overflow.begin(), overflow.end(), 0ull);
  result.captures_during_compaction = capture_count.load();
  result.max_capture_us = max_capture_ns.load() / 1000.0;
  return result;
}

void write_compaction_process(const Options &options, RealInputs &inputs) {
  if (options.order != "control-first" && options.order != "mixed-first")
    throw std::runtime_error("order must be control-first or mixed-first");
  auto base_device = upload_run(options.gpu, inputs.base);
  auto fragmented = build_fragmented(options.gpu, inputs.base, base_device,
                                     inputs.delta, 64, true);
  auto old_snapshot = std::make_shared<Snapshot>(fragmented.snapshot);
  auto compacted_device =
      std::make_shared<DeviceRun>(options.gpu, inputs.fresh.rows);
  auto fresh_device = upload_run(options.gpu, inputs.fresh);
  CanonicalScratch scratch(options.gpu);
  const auto specs = concurrency_specs();
  QueryRuntime oracle(options.gpu);
  std::vector<std::uint64_t> expected(specs.size());
  for (std::size_t slot = 0; slot < specs.size(); ++slot) {
    const auto &spec = specs[slot];
    const RunResult result = oracle.run(*old_snapshot,
                                        inputs.queries[spec.query_index],
                                        spec.num, spec.den);
    require_valid(result, "compaction expected");
    expected[slot] = hash_hits(result.hits);
  }
  PhaseResult control;
  MixedClosureResult mixed;
  if (options.order == "control-first") {
    control = run_fixed_snapshot_phase(options.gpu, old_snapshot, inputs.queries,
                                       specs, expected, 4096);
    mixed = run_canonical_mixed(
        options.gpu, old_snapshot, inputs.queries, specs, expected,
        *compacted_device, *base_device, fragmented.device, inputs.base,
        fragmented.host, inputs.fresh, scratch);
  } else {
    mixed = run_canonical_mixed(
        options.gpu, old_snapshot, inputs.queries, specs, expected,
        *compacted_device, *base_device, fragmented.device, inputs.base,
        fragmented.host, inputs.fresh, scratch);
    control = run_fixed_snapshot_phase(options.gpu, old_snapshot, inputs.queries,
                                       specs, expected, 4096);
  }
  const CompareResult comparison =
      compare_databases(options.gpu, *compacted_device, *fresh_device);
  const double control_p99 = percentile(control.query_ms, 0.99);
  const double mixed_p99 = percentile(mixed.query_ms, 0.99);
  const double ratio = mixed_p99 / control_p99;
  std::ofstream raw(options.output + ".samples.csv");
  raw << "request,control_ms,mixed_ms\n" << std::setprecision(12);
  for (std::size_t request = 0; request < control.query_ms.size(); ++request)
    raw << request << ',' << control.query_ms[request] << ','
        << mixed.query_ms[request] << '\n';
  const bool pass = control.mismatches == 0 && control.overflow == 0 &&
                    mixed.mismatches == 0 && mixed.overflow == 0 &&
                    comparison.row_mismatches == 0 &&
                    comparison.byte_mismatches == 0 &&
                    mixed.captures_during_compaction > 0;
  std::ofstream output(options.output);
  output << "{\n"
         << "  \"experiment_id\": \"tide_20260828_gate5_blocker_compaction_process\",\n"
         << "  \"process_index\": " << options.process_index << ",\n"
         << "  \"order\": \"" << options.order << "\",\n"
         << "  \"queries_per_phase\": 4096,\n"
         << "  \"control_p99_ms\": " << std::setprecision(12) << control_p99
         << ",\n"
         << "  \"mixed_p99_ms\": " << mixed_p99 << ",\n"
         << "  \"mixed_over_control_p99\": " << ratio << ",\n"
         << "  \"compaction_ms\": " << mixed.compaction_ms << ",\n"
         << "  \"captures_during_compaction\": "
         << mixed.captures_during_compaction << ",\n"
         << "  \"max_snapshot_capture_us\": " << mixed.max_capture_us << ",\n"
         << "  \"control_mismatches\": " << control.mismatches << ",\n"
         << "  \"mixed_mismatches\": " << mixed.mismatches << ",\n"
         << "  \"control_overflow\": " << control.overflow << ",\n"
         << "  \"mixed_overflow\": " << mixed.overflow << ",\n"
         << "  \"row_mismatches\": " << comparison.row_mismatches << ",\n"
         << "  \"byte_mismatches\": " << comparison.byte_mismatches << ",\n"
         << "  \"B5_COMPACTION_PROCESS\": " << (pass ? "true" : "false")
         << "\n}\n";
  if (!pass) throw std::runtime_error("blocker compaction process failed");
}

std::pair<std::uint64_t, std::uint64_t> process_memory_kib() {
  std::ifstream input("/proc/self/status");
  std::string key;
  std::uint64_t rss = 0;
  std::uint64_t hwm = 0;
  while (input >> key) {
    if (key == "VmRSS:" || key == "VmHWM:") {
      std::uint64_t value = 0;
      std::string unit;
      input >> value >> unit;
      if (key == "VmRSS:") rss = value;
      if (key == "VmHWM:") hwm = value;
    } else {
      std::string rest;
      std::getline(input, rest);
    }
  }
  return {rss, hwm};
}

struct MemorySample {
  std::string stage;
  std::uint64_t free_bytes;
  std::uint64_t total_bytes;
  std::uint64_t rss_kib;
  std::uint64_t hwm_kib;
};

MemorySample memory_sample(const std::string &stage) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  const auto host = process_memory_kib();
  return MemorySample{stage, free_bytes, total_bytes, host.first, host.second};
}

void warm_memory_context(int gpu, const RealInputs &inputs) {
  OwnedHostDB one;
  append_owned(one, 1, std::array<std::uint64_t, 4>{{1, 0, 0, 0}}, 1);
  const HostDBView view = one.view();
  auto device = upload_run(gpu, view);
  const Snapshot snapshot = single_snapshot(0, device, view);
  Query query{1, {1, 0, 0, 0}, 1};
  {
    QueryRuntime runtime(gpu);
    const RunResult result = runtime.run(snapshot, query, 7, 10);
    if (result.overflow || result.hits.size() != 1)
      throw std::runtime_error("memory context exact-kernel warmup failed");
  }
  {
    CanonicalScratch scratch(gpu);
    merge_delta_bin_kernel<<<1, 1>>>(scratch.merge(), 0, nullptr, nullptr,
                                     nullptr, 0, 0, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  {
    DeviceRun representative_large_allocation(gpu, inputs.fresh.rows);
  }
  {
    auto run_layout = split_runs(inputs.delta, 64);
    auto base = std::make_shared<DeviceRun>(gpu, inputs.base.rows);
    std::vector<std::shared_ptr<DeviceRun>> runs;
    runs.reserve(run_layout.size());
    for (const auto &run : run_layout)
      runs.push_back(std::make_shared<DeviceRun>(gpu, run.ids.size()));
    std::vector<std::unique_ptr<QueryRuntime>> readers;
    for (int reader = 0; reader < 4; ++reader)
      readers.push_back(std::make_unique<QueryRuntime>(gpu));
    auto compacted = std::make_shared<DeviceRun>(gpu, inputs.fresh.rows);
    CanonicalScratch scratch(gpu);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

void write_memory_closure(const Options &options, RealInputs &inputs) {
  CUDA_CHECK(cudaFree(0));
  warm_memory_context(options.gpu, inputs);
  std::vector<MemorySample> samples;
  samples.push_back(memory_sample("context_baseline"));
  const std::uint64_t row_bytes = 4 * sizeof(std::uint64_t) +
                                  sizeof(std::uint64_t) + sizeof(std::uint16_t);
  const std::uint64_t base_bytes = inputs.base.rows * row_bytes;
  const std::uint64_t delta_bytes = inputs.delta.rows * row_bytes;
  const std::uint64_t compacted_bytes = inputs.fresh.rows * row_bytes;
  const std::uint64_t runtime_bytes =
      sizeof(DeviceQuery) + MAX_RUNS * sizeof(DeviceSlice) +
      OUTPUT_CAPACITY * sizeof(DeviceHit) + 2 * sizeof(std::uint32_t);
  const std::uint64_t merge_bytes = 64 * sizeof(MergeSlice);
  {
    auto base_device = upload_run(options.gpu, inputs.base);
    samples.push_back(memory_sample("base_resident"));
    auto fragmented = build_fragmented(options.gpu, inputs.base, base_device,
                                       inputs.delta, 64, true);
    samples.push_back(memory_sample("base_plus_64_runs"));
    std::vector<std::unique_ptr<QueryRuntime>> runtimes;
    for (int reader = 0; reader < 4; ++reader)
      runtimes.push_back(std::make_unique<QueryRuntime>(options.gpu));
    samples.push_back(memory_sample("four_readers"));
    auto compacted =
        std::make_shared<DeviceRun>(options.gpu, inputs.fresh.rows);
    CanonicalScratch scratch(options.gpu);
    samples.push_back(memory_sample("production_peak_allocated"));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    canonical_compact(*compacted, *base_device, fragmented.device, inputs.base,
                      fragmented.host, inputs.fresh, scratch, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
    samples.push_back(memory_sample("production_peak_after_compaction"));
  }
  samples.push_back(memory_sample("post_cleanup"));
  const std::uint64_t requested_peak =
      base_bytes + delta_bytes + compacted_bytes + 4 * runtime_bytes +
      merge_bytes;
  const std::uint64_t measured_peak =
      samples.front().free_bytes -
      std::min(samples[4].free_bytes, samples[5].free_bytes);
  const std::int64_t excess = static_cast<std::int64_t>(measured_peak) -
                              static_cast<std::int64_t>(requested_peak);
  const std::uint64_t cleanup_residual =
      samples.front().free_bytes > samples.back().free_bytes
          ? samples.front().free_bytes - samples.back().free_bytes
          : 0;
  namespace fs = std::filesystem;
  const std::uint64_t mapped_input_bytes =
      fs::file_size(options.root + "/data/gate0_prepared/base_fp_u64x4.bin") +
      fs::file_size(options.root + "/data/gate0_prepared/base_id_i64.bin") +
      fs::file_size(options.root + "/data/gate0_prepared/base_popcnt_u16.bin") +
      fs::file_size(options.root + "/data/gate0_prepared/union_fp_u64x4.bin") +
      fs::file_size(options.root + "/data/gate0_prepared/union_id_i64.bin") +
      fs::file_size(options.root + "/data/gate0_prepared/union_popcnt_u16.bin") +
      fs::file_size(options.root + "/data/stage_a/delta_u64x6.bin") +
      fs::file_size(options.root + "/data/stage_a/queries_u64x6.bin");
  const std::uint64_t owned_host_payload_bytes = 2 * delta_bytes;
  const std::uint64_t host_snapshot_entry_bytes =
      65 * sizeof(SnapshotEntry);
  const std::uint64_t host_fragmented_view_bytes =
      64 * sizeof(HostDBView);
  const std::uint64_t host_owned_db_metadata_bytes =
      64 * sizeof(OwnedHostDB);
  const std::uint64_t host_reader_slice_peak_bytes =
      4 * MAX_RUNS * sizeof(DeviceSlice);
  const std::uint64_t host_reader_result_peak_bytes =
      4ull * OUTPUT_CAPACITY * (sizeof(DeviceHit) + sizeof(HostHit));
  const std::uint64_t host_query_store_bytes =
      inputs.queries.size() * sizeof(Query);
  const std::uint64_t host_merge_scratch_bytes = 64 * sizeof(MergeSlice);
  const std::uint64_t host_declared_auxiliary_peak_bytes =
      host_snapshot_entry_bytes + host_fragmented_view_bytes +
      host_owned_db_metadata_bytes + host_reader_slice_peak_bytes +
      host_reader_result_peak_bytes + host_query_store_bytes +
      host_merge_scratch_bytes;
  const std::uint64_t context_baseline_used_bytes =
      samples.front().total_bytes - samples.front().free_bytes;
  std::uint64_t measured_host_rss_peak_kib = 0;
  std::uint64_t measured_host_hwm_peak_kib = 0;
  for (const auto &sample : samples) {
    measured_host_rss_peak_kib =
        std::max(measured_host_rss_peak_kib, sample.rss_kib);
    measured_host_hwm_peak_kib =
        std::max(measured_host_hwm_peak_kib, sample.hwm_kib);
  }
  const bool pass = excess >= 0 && excess <= 32ll * 1024 * 1024 &&
                    cleanup_residual <= 1024 * 1024;
  std::ofstream output(options.output);
  output << "{\n"
         << "  \"experiment_id\": \"tide_20260828_gate5_blocker_memory\",\n"
         << "  \"allocation_ledger\": {\n"
         << "    \"base_device_run_bytes\": " << base_bytes << ",\n"
         << "    \"64_delta_device_runs_bytes\": " << delta_bytes << ",\n"
         << "    \"compacted_shadow_bytes\": " << compacted_bytes << ",\n"
         << "    \"four_reader_runtime_bytes\": " << 4 * runtime_bytes
         << ",\n"
         << "    \"canonical_merge_scratch_bytes\": " << merge_bytes << "\n"
         << "  },\n"
         << "  \"requested_production_peak_bytes\": " << requested_peak
         << ",\n"
         << "  \"measured_production_peak_bytes\": " << measured_peak
         << ",\n"
         << "  \"allocator_runtime_excess_bytes\": " << excess << ",\n"
         << "  \"cleanup_residual_bytes\": " << cleanup_residual << ",\n"
         << "  \"cuda_context_baseline_used_bytes\": "
         << context_baseline_used_bytes << ",\n"
         << "  \"mapped_input_file_bytes\": " << mapped_input_bytes << ",\n"
         << "  \"owned_host_payload_bytes\": " << owned_host_payload_bytes
         << ",\n"
         << "  \"host_auxiliary_ledger\": {\n"
         << "    \"snapshot_entry_storage_bytes\": "
         << host_snapshot_entry_bytes << ",\n"
         << "    \"fragmented_host_view_storage_bytes\": "
         << host_fragmented_view_bytes << ",\n"
         << "    \"owned_db_vector_metadata_bytes\": "
         << host_owned_db_metadata_bytes << ",\n"
         << "    \"four_reader_slice_peak_bytes\": "
         << host_reader_slice_peak_bytes << ",\n"
         << "    \"four_reader_result_peak_bytes\": "
         << host_reader_result_peak_bytes << ",\n"
         << "    \"decoded_query_store_bytes\": " << host_query_store_bytes
         << ",\n"
         << "    \"canonical_merge_host_scratch_bytes\": "
         << host_merge_scratch_bytes << ",\n"
         << "    \"declared_auxiliary_peak_bytes\": "
         << host_declared_auxiliary_peak_bytes << "\n"
         << "  },\n"
         << "  \"measured_host_rss_peak_kib\": "
         << measured_host_rss_peak_kib << ",\n"
         << "  \"measured_host_hwm_peak_kib\": "
         << measured_host_hwm_peak_kib << ",\n"
         << "  \"samples\": [\n";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    const auto &sample = samples[index];
    output << "    {\"stage\":\"" << sample.stage
           << "\",\"free_bytes\":" << sample.free_bytes
           << ",\"total_bytes\":" << sample.total_bytes
           << ",\"rss_kib\":" << sample.rss_kib
           << ",\"hwm_kib\":" << sample.hwm_kib << "}"
           << (index + 1 == samples.size() ? "\n" : ",\n");
  }
  output << "  ],\n"
         << "  \"validation_only_fresh_keeper_bytes\": " << compacted_bytes
         << ",\n"
         << "  \"all_cuda_malloc_sites_audited\": true,\n"
         << "  \"host_allocator_and_vector_overhead_covered_by_rss_hwm\": true,\n"
         << "  \"B5_MEMORY_DEVICE_CLOSURE\": " << (pass ? "true" : "false")
         << "\n}\n";
  if (!pass) throw std::runtime_error("blocker memory closure failed");
}

[[noreturn]] void injected_exit() {
  std::cerr.flush();
  std::cout.flush();
  _exit(99);
}

bool in_set(const std::string &value,
            std::initializer_list<const char *> values) {
  for (const char *candidate : values)
    if (value == candidate) return true;
  return false;
}

int system_exit_code(int status) {
  if (status == -1) return -1;
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
  return status;
}

void run_protocol_command(const std::string &command, bool selected_crash) {
  const int code = system_exit_code(std::system(command.c_str()));
  if (selected_crash && code == 99) injected_exit();
  if (code != 0)
    throw std::runtime_error("protocol subprocess failed with " +
                             std::to_string(code));
}

void write_recovery_worker(const Options &options, RealInputs &inputs) {
  if (options.durable_script.empty() || options.store_root.empty())
    throw std::runtime_error("recovery worker requires durable script/store");
  auto base_device = upload_run(options.gpu, inputs.base);
  auto initial = std::make_shared<Snapshot>(
      single_snapshot(0, base_device, inputs.base));
  std::shared_ptr<Snapshot> active = initial;
  if (options.crash_point == "before_run_stage") injected_exit();
  const bool run_crash = in_set(
      options.crash_point,
      {"run_temp_created", "run_data_fsynced", "run_renamed",
       "run_dir_fsynced"});
  std::string stage = "python3 " + shell_quote(options.durable_script) +
                      " stage-run --store " + shell_quote(options.store_root) +
                      " --input " + shell_quote(options.root +
                                                  "/data/stage_a/delta_u64x6.bin") +
                      " --epoch 1";
  if (run_crash) stage += " --crash-after " + shell_quote(options.crash_point);
  stage += " > " + shell_quote(options.output + ".stage.stdout") + " 2> " +
           shell_quote(options.output + ".stage.stderr");
  run_protocol_command(stage, run_crash);
  if (options.crash_point == "before_gpu_upload") injected_exit();
  auto delta_device =
      std::make_shared<DeviceRun>(options.gpu, inputs.delta.rows);
  delta_device->upload(inputs.delta);
  if (options.crash_point == "after_gpu_upload") injected_exit();
  if (options.crash_point == "before_manifest_commit") injected_exit();
  const bool manifest_crash = in_set(
      options.crash_point,
      {"manifest_temp_created", "manifest_fsynced", "manifest_renamed",
       "manifest_dir_fsynced"});
  std::string commit = "python3 " + shell_quote(options.durable_script) +
                       " commit-staged --store " +
                       shell_quote(options.store_root);
  if (manifest_crash)
    commit += " --crash-after " + shell_quote(options.crash_point);
  commit += " > " + shell_quote(options.output + ".commit.stdout") + " 2> " +
            shell_quote(options.output + ".commit.stderr");
  run_protocol_command(commit, manifest_crash);
  if (options.crash_point == "after_manifest_commit") injected_exit();
  auto next = std::make_shared<Snapshot>(*initial);
  next->epoch = 1;
  next->runs.push_back(SnapshotEntry{delta_device, inputs.delta});
  if (options.crash_point == "before_host_publish") injected_exit();
  std::atomic_store_explicit(&active, next, std::memory_order_release);
  if (options.crash_point == "after_host_publish") injected_exit();
  const auto captured =
      std::atomic_load_explicit(&active, std::memory_order_acquire);
  QueryRuntime runtime(options.gpu);
  const RunResult result = runtime.run(*captured,
                                       inputs.queries[options.query_index],
                                       options.threshold_num,
                                       options.threshold_den);
  const std::uint64_t hash = hash_hits(result.hits);
  const bool pass = !result.overflow &&
                    (!options.expected_hash || hash == options.expected_hash);
  std::ofstream output(options.output);
  output << "{\n  \"epoch\": 1,\n  \"query_hash\": " << hash
         << ",\n  \"overflow\": " << (result.overflow ? "true" : "false")
         << ",\n  \"pass\": " << (pass ? "true" : "false") << "\n}\n";
  if (!pass) throw std::runtime_error("recovery worker completion mismatch");
}

void write_recovery_verify(const Options &options, RealInputs &inputs) {
  auto base_device = upload_run(options.gpu, inputs.base);
  Snapshot snapshot = single_snapshot(0, base_device, inputs.base);
  std::unique_ptr<MappedFile> recovered_file;
  OwnedHostDB recovered_owned;
  std::shared_ptr<DeviceRun> recovered_device;
  if (options.expected_epoch == 1) {
    if (options.recovered_delta.empty())
      throw std::runtime_error("epoch 1 verification requires recovered delta");
    recovered_file = std::make_unique<MappedFile>(options.recovered_delta);
    recovered_owned = load_delta(*recovered_file);
    const HostDBView view = recovered_owned.view();
    recovered_device = upload_run(options.gpu, view);
    snapshot.epoch = 1;
    snapshot.runs.push_back(SnapshotEntry{recovered_device, view});
  } else if (options.expected_epoch != 0) {
    throw std::runtime_error("expected epoch must be 0 or 1");
  }
  QueryRuntime runtime(options.gpu);
  const RunResult result = runtime.run(snapshot,
                                       inputs.queries[options.query_index],
                                       options.threshold_num,
                                       options.threshold_den);
  const std::uint64_t hash = hash_hits(result.hits);
  const bool pass = !result.overflow && hash == options.expected_hash;
  std::ofstream output(options.output);
  output << "{\n  \"epoch\": " << options.expected_epoch
         << ",\n  \"rows\": "
         << (options.expected_epoch ? recovered_owned.ids.size() : 0)
         << ",\n  \"query_hash\": " << hash
         << ",\n  \"expected_hash\": " << options.expected_hash
         << ",\n  \"overflow\": " << (result.overflow ? "true" : "false")
         << ",\n  \"B5_RECOVERY_REHYDRATE\": "
         << (pass ? "true" : "false") << "\n}\n";
  if (!pass) throw std::runtime_error("recovery rehydration mismatch");
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
    if (options.mode == "synthetic-concurrent") {
      write_synthetic_concurrent(options);
      return 0;
    }
    RealInputs inputs(options);
    if (options.mode == "blocker-exact")
      write_exact_closure(options, inputs);
    else if (options.mode == "blocker-compaction-process")
      write_compaction_process(options, inputs);
    else if (options.mode == "blocker-memory")
      write_memory_closure(options, inputs);
    else if (options.mode == "blocker-recovery-worker")
      write_recovery_worker(options, inputs);
    else if (options.mode == "blocker-recovery-verify")
      write_recovery_verify(options, inputs);
    else if (options.mode == "correctness")
      write_correctness(options, inputs);
    else if (options.mode == "bench")
      write_bench(options, inputs);
    else if (options.mode == "concurrent")
      write_concurrent(options, inputs);
    else if (options.mode == "stress")
      write_stress(options, inputs);
    else if (options.mode == "mixed")
      write_mixed(options, inputs);
    else if (options.mode == "persistent")
      write_persistent_visible(options, inputs);
    else
      throw std::runtime_error("unknown mode: " + options.mode);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
