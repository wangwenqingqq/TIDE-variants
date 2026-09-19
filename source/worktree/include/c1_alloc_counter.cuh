// Isolated C1 source-level allocator counter skeleton.
// Not integrated or compiled yet.  A future copied worktree must explicitly
// replace its C1-relevant cudaMalloc/cudaMallocManaged/cudaFree calls with the
// C1_* macros below; archive source files must remain unchanged.
#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <fstream>
#include <mutex>
#include <string>

namespace gtspp_c1_counter {

enum class Phase : std::uint8_t { kSetup, kCold, kProvision, kWarmup, kSteady, kTeardown, kCount };
enum class Kind : std::uint8_t { kMalloc, kMallocManaged, kFree };

struct Bucket {
  std::uint64_t malloc_calls = 0;
  std::uint64_t managed_calls = 0;
  std::uint64_t free_calls = 0;
  std::uint64_t requested_malloc_bytes = 0;
  std::uint64_t requested_managed_bytes = 0;
  std::uint64_t failed_calls = 0;
  double api_host_ms = 0.0;
};

struct State {
  std::array<Bucket, static_cast<std::size_t>(Phase::kCount)> buckets{};
  Phase phase = Phase::kSetup;
  std::mutex mutex;
};

inline State& state() {
  static State value;
  return value;
}

inline const char* phase_name(Phase phase) {
  switch (phase) {
    case Phase::kSetup: return "setup";
    case Phase::kCold: return "cold";
    case Phase::kProvision: return "provision";
    case Phase::kWarmup: return "warmup";
    case Phase::kSteady: return "steady_state";
    case Phase::kTeardown: return "teardown";
    default: return "unknown";
  }
}

inline void set_phase(Phase phase) {
  std::lock_guard<std::mutex> lock(state().mutex);
  state().phase = phase;
}

inline void reset_all() {
  std::lock_guard<std::mutex> lock(state().mutex);
  state().buckets = {};
}

inline void record(Kind kind, std::size_t bytes, cudaError_t status, double elapsed_ms) {
  std::lock_guard<std::mutex> lock(state().mutex);
  Bucket& bucket = state().buckets[static_cast<std::size_t>(state().phase)];
  if (kind == Kind::kMalloc) {
    ++bucket.malloc_calls;
    bucket.requested_malloc_bytes += bytes;
  } else if (kind == Kind::kMallocManaged) {
    ++bucket.managed_calls;
    bucket.requested_managed_bytes += bytes;
  } else {
    ++bucket.free_calls;
  }
  if (status != cudaSuccess) ++bucket.failed_calls;
  bucket.api_host_ms += elapsed_ms;
}

inline cudaError_t malloc(void** ptr, std::size_t bytes) {
  const auto start = std::chrono::steady_clock::now();
  const cudaError_t status = cudaMalloc(ptr, bytes);
  const auto stop = std::chrono::steady_clock::now();
  record(Kind::kMalloc, bytes, status,
         std::chrono::duration<double, std::milli>(stop - start).count());
  return status;
}

inline cudaError_t malloc_managed(void** ptr, std::size_t bytes, unsigned int flags = cudaMemAttachGlobal) {
  const auto start = std::chrono::steady_clock::now();
  const cudaError_t status = cudaMallocManaged(ptr, bytes, flags);
  const auto stop = std::chrono::steady_clock::now();
  record(Kind::kMallocManaged, bytes, status,
         std::chrono::duration<double, std::milli>(stop - start).count());
  return status;
}

inline cudaError_t free(void* ptr) {
  const auto start = std::chrono::steady_clock::now();
  const cudaError_t status = cudaFree(ptr);
  const auto stop = std::chrono::steady_clock::now();
  record(Kind::kFree, 0, status,
         std::chrono::duration<double, std::milli>(stop - start).count());
  return status;
}

struct Snapshot {
  std::uint64_t malloc_calls = 0;
  std::uint64_t managed_calls = 0;
  std::uint64_t free_calls = 0;
  std::uint64_t requested_malloc_bytes = 0;
  std::uint64_t requested_managed_bytes = 0;
  std::uint64_t failed_calls = 0;
  double api_host_ms = 0.0;
};

inline Snapshot snapshot(Phase phase) {
  std::lock_guard<std::mutex> lock(state().mutex);
  const Bucket& b = state().buckets[static_cast<std::size_t>(phase)];
  return Snapshot{b.malloc_calls, b.managed_calls, b.free_calls,
                  b.requested_malloc_bytes, b.requested_managed_bytes,
                  b.failed_calls, b.api_host_ms};
}

inline Snapshot subtract(const Snapshot& after, const Snapshot& before) {
  return Snapshot{after.malloc_calls - before.malloc_calls,
                  after.managed_calls - before.managed_calls,
                  after.free_calls - before.free_calls,
                  after.requested_malloc_bytes - before.requested_malloc_bytes,
                  after.requested_managed_bytes - before.requested_managed_bytes,
                  after.failed_calls - before.failed_calls,
                  after.api_host_ms - before.api_host_ms};
}

inline void write_json(const std::string& path) {
  std::lock_guard<std::mutex> lock(state().mutex);
  std::ofstream out(path);
  out << "{\n  \"schema\": \"gtspp-c1-source-allocator-counter-v1\",\n  \"phases\": {\n";
  for (std::size_t i = 0; i < state().buckets.size(); ++i) {
    const Bucket& b = state().buckets[i];
    out << "    \"" << phase_name(static_cast<Phase>(i)) << "\": {"
        << "\"cudaMalloc_calls\":" << b.malloc_calls
        << ",\"cudaMallocManaged_calls\":" << b.managed_calls
        << ",\"cudaFree_calls\":" << b.free_calls
        << ",\"cudaMalloc_requested_bytes\":" << b.requested_malloc_bytes
        << ",\"cudaMallocManaged_requested_bytes\":" << b.requested_managed_bytes
        << ",\"failed_calls\":" << b.failed_calls
        << ",\"api_host_ms\":" << b.api_host_ms << "}";
    out << (i + 1 == state().buckets.size() ? "\n" : ",\n");
  }
  out << "  }\n}\n";
}

}  // namespace gtspp_c1_counter

#define C1_CUDA_MALLOC(ptr, bytes) ::gtspp_c1_counter::malloc(reinterpret_cast<void**>(ptr), (bytes))
#define C1_CUDA_MALLOC_MANAGED(ptr, bytes) ::gtspp_c1_counter::malloc_managed(reinterpret_cast<void**>(ptr), (bytes))
#define C1_CUDA_FREE(ptr) ::gtspp_c1_counter::free((ptr))
