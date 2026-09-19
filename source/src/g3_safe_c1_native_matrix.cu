// Safe-C1 G3 native correctness-matrix implementation.
//
// This translation unit is deliberately a library, not a trace executable.  It
// is intended for compile-only closure checks first.  No build/run result is
// evidence of correctness or performance until a later, permitted native gate
// binds each fixture to an actual frozen GTS tree and an independent oracle.
//
// Production candidate flow is intentionally narrow:
//   KNN: GTS base traversal receipt -> sidecars of exactly received leaves ->
//        complete global delta.
//   Range: exact scan of the current immutable base -> complete global delta.
// The range branch is a correctness-only fallback after the branch-aligned
// predicate mirror failed its bootstrap oracle.  It is intentionally not a
// GTS range receipt, native archive-equivalence, or performance claim.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define RP_DEFINE_CONSTANTS
// Include tree first: its archive include order must load Thrust before config.cuh
// defines the legacy `short` alias.  The receipt header then reuses that guard.
#include "tree.cuh"
#include "g3_safe_search_v2.cuh"

// mlp_constant.cuh provides archive extern declarations.  G3 uses only the
// mode-0 residual path; no learned-pruning or MLP constant upload is called.

namespace safe_c1_g3 {

using StableId = int;
using LocalRow = int;
using DistanceSq = std::uint64_t;

constexpr float kStrictEpsilon = 1.0e-5F;
constexpr int kRangeRadiusSafetyUlps = 128;
constexpr int kResidualPruningMode = 0;
constexpr bool G3_NATIVE_TOPK_RECEIPT_REUSED_FROM_G1 = true;
// The legacy range entrypoint is ID-query only.  G3 therefore exposes a
// separately named, branch-aligned vector predicate mirror; it is not a claim
// that searchIndexRnnV2 itself ran for vector queries.
constexpr bool G3_NATIVE_RANGE_RECEIPT_IMPLEMENTED = false;
constexpr bool G3_RANGE_BRANCH_ALIGNED_VECTOR_PREDICATE_MIRROR_IMPLEMENTED = true;
// The range answer path intentionally bypasses the unproven branch-aligned
// predicate mirror.  It scans every local row of the current immutable base
// exactly, then merge_query_partitions() scans the complete global delta.
constexpr bool G3_RANGE_SAFE_FULL_IMMUTABLE_BASE_FALLBACK_IMPLEMENTED = true;
// Public rebuilds accept no arbitrary live-ID set after initialization: only
// build_initial_base() receives the initial base and later rebuilds snapshot
// state_.live_ids() internally.
constexpr bool G3_REBUILD_FROM_CURRENT_LIVE_ONLY_IMPLEMENTED = true;
// Range support in this source is deliberately limited to this claim scope.
constexpr const char* G3_RANGE_CLAIM_SCOPE =
    "exact full immutable-base plus global-delta range fallback; branch predicate mirror diagnostic only; "
    "not native archive equivalence or a range-performance claim";
// Direct sidecars remain fail-closed until a future runner has bound this exact
// source closure to device-mode/readback, KNN disk behavior, receipts and an
// independent oracle.  This static/compile-only phase never opens the gate.
constexpr bool G3_DIRECT_SIDECAR_RUNTIME_GUARD_OPEN = false;
constexpr bool G3_DIRECT_CERTIFICATE_KNN_VISIBILITY_ONLY = true;
constexpr bool G3_KNN_DISK_MONOTONIC_ASSUMPTION_RUNTIME_UNVERIFIED = true;
// A full frozen-tree/base snapshot is captured and validated at each build.
// Per-operation checks below deliberately use only immutable owner/generation
// identities: they are not a device-byte mutation proof and are valid only
// under the single-image runner admission contract.
constexpr bool G3_LIGHTWEIGHT_GENERATION_IDENTITY_GUARD_IMPLEMENTED = true;
constexpr bool G3_FULL_FROZEN_BASE_SNAPSHOT_ON_EACH_OPERATION = false;
constexpr bool G3_RUNNER_SINGLE_IMAGE_ADMISSION_REQUIRED = true;
// The copied archive query code retains a builder-global start_idx.  Its
// pnum_level reduction is valid for this controlled fixture only when the
// builder produced exactly this height/final heap-level offset; height > 4 is
// fail-closed rather than silently treated as a general dynamic-GTS result.
constexpr int G3_CONTROLLED_ARCHIVE_TREE_HEIGHT = 4;
constexpr int G3_CONTROLLED_ARCHIVE_FINAL_START_INDEX = 111;
constexpr int G3_CONTROLLED_ARCHIVE_MAX_SIZE = 20;
// This TU intentionally has no main and is never a launchable benchmark by
// itself.  A later trace runner must produce its own build/run attestation.
constexpr bool G3_NATIVE_ENGINE_EXECUTABLE = false;
constexpr bool G3_NATIVE_ENGINE_EXECUTED = false;

[[noreturn]] inline void fail(const std::string& message) {
  throw std::runtime_error("Safe-C1 G3 invariant failure: " + message);
}

// GTS keeps process-global state (including max_dis_d and the copied-header
// safe_c1_last_visited_leaf_pairs receipt vector).  A matrix therefore owns an
// exclusive same-process lease for its entire lifetime.  It is intentionally
// acquired before all other NativeSafeC1Matrix members, so any constructor
// failure unwinds it automatically and a second live engine fails closed.
class SameProcessExclusiveGtsLease final {
 public:
  SameProcessExclusiveGtsLease() : lock_(mutex(), std::try_to_lock) {
    if (!lock_.owns_lock()) {
      fail("same-process native GTS lease is already held");
    }
  }
  ~SameProcessExclusiveGtsLease() noexcept = default;
  SameProcessExclusiveGtsLease(const SameProcessExclusiveGtsLease&) = delete;
  SameProcessExclusiveGtsLease& operator=(const SameProcessExclusiveGtsLease&) = delete;
  SameProcessExclusiveGtsLease(SameProcessExclusiveGtsLease&&) = delete;
  SameProcessExclusiveGtsLease& operator=(SameProcessExclusiveGtsLease&&) = delete;

 private:
  static std::mutex& mutex() {
    static std::mutex value;
    return value;
  }
  std::unique_lock<std::mutex> lock_;
};

inline void cuda_check(cudaError_t status, const char* expression,
                       const char* file, int line) {
  if (status != cudaSuccess) {
    std::ostringstream out;
    out << expression << " failed at " << file << ':' << line << ": "
        << cudaGetErrorString(status);
    fail(out.str());
  }
}

#define G3_CUDA(call) ::safe_c1_g3::cuda_check((call), #call, __FILE__, __LINE__)

// Minimal in-tree SHA-256 implementation.  The output encodings are part of
// the fixture contract; do not replace them with std::hash or pointer hashes.
class Sha256 {
 public:
  Sha256() { reset(); }

  void reset() {
    state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
              0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    block_.fill(0);
    block_size_ = 0;
    bit_length_ = 0;
  }

  void update(const void* address, std::size_t bytes) {
    const auto* input = static_cast<const std::uint8_t*>(address);
    for (std::size_t index = 0; index < bytes; ++index) {
      block_[block_size_++] = input[index];
      if (block_size_ == block_.size()) {
        transform(block_.data());
        bit_length_ += 512;
        block_size_ = 0;
      }
    }
  }

  void update(const std::string& text) { update(text.data(), text.size()); }

  [[nodiscard]] std::string final_hex() {
    std::array<std::uint8_t, 64> tail = block_;
    std::size_t index = block_size_;
    tail[index++] = 0x80U;
    if (index > 56U) {
      while (index < 64U) tail[index++] = 0;
      transform(tail.data());
      tail.fill(0);
      index = 0;
    }
    while (index < 56U) tail[index++] = 0;
    const std::uint64_t total_bits = bit_length_ + static_cast<std::uint64_t>(block_size_) * 8ULL;
    for (int shift = 7; shift >= 0; --shift) {
      tail[56 + (7 - shift)] = static_cast<std::uint8_t>((total_bits >> (shift * 8)) & 0xffU);
    }
    transform(tail.data());
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::uint32_t value : state_) out << std::setw(8) << value;
    return out.str();
  }

 private:
  static constexpr std::array<std::uint32_t, 64> kConstants = {
      0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
      0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
      0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
      0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
      0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
      0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
      0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
      0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};

  static constexpr std::uint32_t rotr(std::uint32_t value, std::uint32_t bits) {
    return (value >> bits) | (value << (32U - bits));
  }
  static constexpr std::uint32_t choose(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return (x & y) ^ (~x & z);
  }
  static constexpr std::uint32_t majority(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
  }
  static constexpr std::uint32_t big0(std::uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
  static constexpr std::uint32_t big1(std::uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
  static constexpr std::uint32_t small0(std::uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
  static constexpr std::uint32_t small1(std::uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

  void transform(const std::uint8_t* input) {
    std::array<std::uint32_t, 64> words{};
    for (int index = 0; index < 16; ++index) {
      words[index] = (static_cast<std::uint32_t>(input[index * 4]) << 24) |
                     (static_cast<std::uint32_t>(input[index * 4 + 1]) << 16) |
                     (static_cast<std::uint32_t>(input[index * 4 + 2]) << 8) |
                     static_cast<std::uint32_t>(input[index * 4 + 3]);
    }
    for (int index = 16; index < 64; ++index) {
      words[index] = small1(words[index - 2]) + words[index - 7] +
                     small0(words[index - 15]) + words[index - 16];
    }
    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int index = 0; index < 64; ++index) {
      const std::uint32_t t1 = h + big1(e) + choose(e, f, g) + kConstants[index] + words[index];
      const std::uint32_t t2 = big0(a) + majority(a, b, c);
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
    state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_ = 0;
  std::uint64_t bit_length_ = 0;
};

inline std::string sha256_text(const std::string& text) {
  Sha256 hash;
  hash.update(text);
  return hash.final_hex();
}

template <typename T>
inline std::string sha256_raw_vector(const std::vector<T>& values,
                                     const std::string& domain) {
  Sha256 hash;
  hash.update(domain);
  if (!values.empty()) hash.update(values.data(), values.size() * sizeof(T));
  return hash.final_hex();
}

inline std::string stable_set_sha256(const std::vector<StableId>& ids) {
  std::vector<StableId> canonical = ids;
  std::sort(canonical.begin(), canonical.end());
  if (std::adjacent_find(canonical.begin(), canonical.end()) != canonical.end()) {
    fail("stable_set_sha256 received duplicate stable ID");
  }
  std::ostringstream bytes;
  for (StableId id : canonical) bytes << id << '\n';
  return sha256_text(bytes.str());
}

inline std::string local_to_stable_sha256(const std::vector<StableId>& local_to_stable) {
  std::ostringstream bytes;
  bytes << "safe-c1-g3-local-to-stable-v1\n";
  for (std::size_t local = 0; local < local_to_stable.size(); ++local) {
    bytes << local << ':' << local_to_stable[local] << '\n';
  }
  return sha256_text(bytes.str());
}

struct StableDistance {
  StableId stable_id = -1;
  DistanceSq distance_sq = 0;
  [[nodiscard]] bool operator==(const StableDistance& other) const {
    return stable_id == other.stable_id && distance_sq == other.distance_sq;
  }
};

inline bool distance_then_stable(const StableDistance& left,
                                 const StableDistance& right) {
  return left.distance_sq != right.distance_sq
             ? left.distance_sq < right.distance_sq
             : left.stable_id < right.stable_id;
}

inline void sort_and_require_unique(std::vector<StableDistance>* rows,
                                    const char* label) {
  std::sort(rows->begin(), rows->end(), distance_then_stable);
  std::set<StableId> seen;
  for (const StableDistance& row : *rows) {
    if (row.stable_id < 0) fail(std::string(label) + " contains invalid stable ID");
    if (!seen.insert(row.stable_id).second) {
      fail(std::string(label) + " contains duplicate stable ID");
    }
  }
}

// Unlike sort_and_require_unique(), this verifier refuses a caller-owned
// result sequence that is not already in the canonical order.  The independent
// CPU oracle must provide an unambiguous, sorted stable-ID sequence.
inline void require_canonical_stable_distances(const std::vector<StableDistance>& rows,
                                               const char* label) {
  std::set<StableId> seen;
  for (std::size_t index = 0; index < rows.size(); ++index) {
    const StableDistance& row = rows[index];
    if (row.stable_id < 0) fail(std::string(label) + " contains invalid stable ID");
    if (!seen.insert(row.stable_id).second) {
      fail(std::string(label) + " contains duplicate stable ID");
    }
    if (index != 0 && !distance_then_stable(rows[index - 1], row)) {
      fail(std::string(label) + " is not in canonical distance/stable-ID order");
    }
  }
}

inline std::string stable_distance_vector_sha256(const std::vector<StableDistance>& rows,
                                                  const char* domain) {
  require_canonical_stable_distances(rows, domain);
  std::ostringstream bytes;
  bytes << domain << '\n';
  for (const StableDistance& row : rows) {
    bytes << row.stable_id << ':' << static_cast<unsigned long long>(row.distance_sq) << '\n';
  }
  return sha256_text(bytes.str());
}

class ImmutablePool {
 public:
  // `values` is physical immutable-pool row major int16 data.  Stable IDs are
  // deliberately translated through stable_to_pool_row: current fixtures are
  // identity but the production contract must not assume it.
  void initialize(int dimension, std::vector<std::int16_t> values,
                  std::vector<int> stable_to_pool_row) {
    if (dimension <= 0 || values.empty() || values.size() % static_cast<std::size_t>(dimension) != 0) {
      fail("invalid immutable pool shape");
    }
    const int rows = static_cast<int>(values.size() / static_cast<std::size_t>(dimension));
    if (stable_to_pool_row.empty()) fail("stable_to_pool_row is empty");
    std::vector<int> seen(rows, 0);
    for (int row : stable_to_pool_row) {
      if (row < 0 || row >= rows) fail("stable_to_pool_row points outside immutable pool");
      if (++seen[row] > 1) fail("stable_to_pool_row is not injective");
    }
    dimension_ = dimension;
    values_ = std::move(values);
    stable_to_pool_row_ = std::move(stable_to_pool_row);
    std::vector<float> device_values(values_.size());
    for (std::size_t index = 0; index < values_.size(); ++index) {
      device_values[index] = static_cast<float>(values_[index]);
    }
    G3_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_values_), device_values.size() * sizeof(float)));
    G3_CUDA(cudaMemcpy(device_values_, device_values.data(),
                       device_values.size() * sizeof(float), cudaMemcpyHostToDevice));
    G3_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_stable_to_pool_row_),
                       stable_to_pool_row_.size() * sizeof(int)));
    G3_CUDA(cudaMemcpy(device_stable_to_pool_row_, stable_to_pool_row_.data(),
                       stable_to_pool_row_.size() * sizeof(int), cudaMemcpyHostToDevice));
  }

  void release() noexcept {
    if (device_values_) cudaFree(device_values_);
    if (device_stable_to_pool_row_) cudaFree(device_stable_to_pool_row_);
    device_values_ = nullptr;
    device_stable_to_pool_row_ = nullptr;
    values_.clear();
    stable_to_pool_row_.clear();
    dimension_ = 0;
  }

  ~ImmutablePool() { release(); }
  ImmutablePool() = default;
  ImmutablePool(const ImmutablePool&) = delete;
  ImmutablePool& operator=(const ImmutablePool&) = delete;

  [[nodiscard]] int dimension() const { return dimension_; }
  [[nodiscard]] int stable_capacity() const { return static_cast<int>(stable_to_pool_row_.size()); }
  [[nodiscard]] int pool_rows() const { return dimension_ == 0 ? 0 : static_cast<int>(values_.size() / dimension_); }
  [[nodiscard]] const float* device_values() const { return device_values_; }
  [[nodiscard]] const int* device_stable_to_pool_row() const { return device_stable_to_pool_row_; }
  [[nodiscard]] const std::int16_t* vector_for_stable(StableId stable_id) const {
    if (stable_id < 0 || stable_id >= stable_capacity()) fail("stable ID outside immutable mapping");
    return values_.data() + static_cast<std::size_t>(stable_to_pool_row_[stable_id]) * dimension_;
  }
  [[nodiscard]] std::string stable_mapping_sha256() const {
    return sha256_raw_vector(stable_to_pool_row_, "safe-c1-g3-stable-to-pool-row-v1\n");
  }

 private:
  int dimension_ = 0;
  std::vector<std::int16_t> values_;
  std::vector<int> stable_to_pool_row_;
  float* device_values_ = nullptr;
  int* device_stable_to_pool_row_ = nullptr;
};

struct BaseTreeRuntime {
  int* data_info = nullptr;
  // config.cuh aliases archive `short` to float.  Keep the explicit type here
  // so the compact GTS rows cannot accidentally be filled with raw i16 bytes.
  float* data_d = nullptr;
  char* data_s = nullptr;
  int* size_s = nullptr;
  int* id_list = nullptr;
  TN* node_list = nullptr;
  int* max_node_num = nullptr;
  int* empty_list = nullptr;
  // max_dis_d is file-global in tree.cuh.  It is captured into this owner as a
  // snapshot generation marker; all rebuilds run under the single-engine
  // barrier and release the old generation before indexConstru can replace it.
  float* owned_max_dis_d = nullptr;
  int tree_height = 0;
  int base_count = 0;
  // tree.cuh may repack leaves into a padded id_list; TN.lid is a physical
  // allocation offset, not a compact [0,base_count) row offset.
  int id_list_capacity = 0;
  std::vector<StableId> local_to_stable;
  std::vector<LocalRow> stable_to_local;

  [[nodiscard]] bool ready() const {
    return data_info != nullptr && data_d != nullptr && id_list != nullptr &&
           node_list != nullptr && max_node_num != nullptr && empty_list != nullptr &&
           owned_max_dis_d != nullptr && base_count > 0 && id_list_capacity > 0;
  }
};

enum class GtsKnnPruningBranch {
  kAllIncludeOnly,
  kPredicateNonLastSibling,
  kPredicateLastSibling,
};

inline const char* gts_knn_pruning_branch_name(GtsKnnPruningBranch branch) {
  switch (branch) {
    case GtsKnnPruningBranch::kAllIncludeOnly: return "all_include";
    case GtsKnnPruningBranch::kPredicateNonLastSibling: return "predicate_nonlast_sibling";
    case GtsKnnPruningBranch::kPredicateLastSibling: return "predicate_last_sibling";
  }
  return "unknown";
}

struct CertificateLevel {
  int parent_leaf_or_internal = -1;
  int child = -1;
  int next_sibling = -1;
  LocalRow pivot_local_row = -1;
  StableId pivot_stable_id = -1;
  float lower = 0.0F;
  float upper = std::numeric_limits<float>::infinity();
  float pivot_distance = 0.0F;
  bool last_child = false;
  // The production KNN header has an optional all-include branch
  // (`labelCNode`).  This certificate never relies on that branch: it proves
  // the stricter predicate branch at every ancestor, so all-include can only
  // enlarge a future receipt.
  bool all_include_branch_not_relied_on = true;
  GtsKnnPruningBranch predicate_branch = GtsKnnPruningBranch::kAllIncludeOnly;
};

struct StrictCertificate {
  bool ok = false;
  int sidecar_leaf_id = -1;
  std::string failure_reason;
  int failure_level = -1;
  int matching_children = 0;
  bool all_include_branch_not_relied_on = true;
  bool every_predicate_branch_mirrored = false;
  // This is deliberately a KNN predicate-visibility certificate only.  It
  // does not establish a direct-sidecar range correctness claim.
  bool knn_visibility_only = true;
  bool range_correctness_not_claimed = true;
  std::vector<CertificateLevel> levels;
};


// Diagnostic-only raw predicate facts.  These records never select a leaf,
// mutate a tier, or stand in for a native traversal membership receipt.
struct CertificateSiblingDiagnostic {
  int child = -1;
  int next_sibling = -1;
  int child_pivot_local_row = -1;
  StableId child_pivot_stable_id = -1;
  float lower = std::numeric_limits<float>::quiet_NaN();
  float upper = std::numeric_limits<float>::quiet_NaN();
  float pivot_distance = std::numeric_limits<float>::quiet_NaN();
  bool child_empty = true;
  bool child_pivot_matches_parent = false;
  bool child_min_dis_finite = false;
  bool last_child = false;
  bool next_sibling_available = false;
  bool next_sibling_min_dis_finite = false;
  bool upper_is_unbounded_last_child = false;
  bool strict_native_predicate_match = false;
};

struct CertificateDiagnosticLevel {
  int level = -1;
  int parent = -1;
  int parent_pivot_local_row = -1;
  StableId parent_pivot_stable_id = -1;
  float pivot_distance = std::numeric_limits<float>::quiet_NaN();
  bool parent_valid = false;
  int diagnostic_strict_match_count = 0;
  std::vector<CertificateSiblingDiagnostic> siblings;
};

struct FrozenCertificateDiagnostic {
  StrictCertificate certificate;
  std::vector<CertificateDiagnosticLevel> levels;
};

struct FrozenTreeSnapshot {
  int tree_height = 0;
  int fanout = 0;
  int node_count = 0;
  int logical_leaf_row_count = 0;
  int base_dimension = 0;
  int metric_code = -1;
  std::vector<TN> nodes;
  std::vector<int> empty;
  std::vector<float> max_distance;
  // These are the immutable base ownership facts.  They deliberately exclude
  // SafeC1State::active_ (base + sidecars + delta); mutable rows are not native
  // GTS tree bytes until an explicit rebuild creates a new generation.
  std::vector<StableId> immutable_base_stable_ids;
  std::string immutable_base_stable_ids_sha256;
  std::string immutable_base_payload_sha256;
  std::string local_to_stable_sha256_value;
  std::string logical_leaf_local_rows_sha256;
  std::string logical_leaf_stable_ids_sha256;
  std::string native_tree_bytes_sha256;
  std::string tree_payload_sha256;
  // Pointer/shape identities captured after the complete generation snapshot.
  // They allow a no-D2H, no-vector-payload-scan consistency check inside the
  // isolated Matrix owner; they do not prove that an arbitrary external DSO
  // has not modified device bytes.
  std::uintptr_t data_info_identity = 0;
  std::uintptr_t data_d_identity = 0;
  std::uintptr_t id_list_identity = 0;
  std::uintptr_t node_list_identity = 0;
  std::uintptr_t max_node_num_identity = 0;
  std::uintptr_t empty_list_identity = 0;
  std::uintptr_t max_dis_d_identity = 0;
  std::uintptr_t local_to_stable_identity = 0;
  std::uintptr_t stable_to_local_identity = 0;
  int runtime_base_count = 0;
  int runtime_id_list_capacity = 0;
  std::size_t runtime_local_to_stable_size = 0;
  std::size_t runtime_stable_to_local_size = 0;
  // The copied archive traversal reads this file-global builder offset during
  // pnum_level reduction.  It is host-resident, so include it in the cheap
  // generation identity guard rather than treating fixed-height admission as a
  // one-time check only.
  int archived_query_start_idx = -1;
  int archived_query_tree_order = -1;
  int archived_query_dis_code = -1;
  int archived_query_infi_dis = -1;
  int archived_query_max_size = -1;

  [[nodiscard]] bool initialized() const { return node_count > 0 && !nodes.empty(); }

  void validate_pruning_topology() const {
    if (fanout != TREE_ORDER || fanout <= 1) {
      fail("frozen GTS fanout does not match the native KNN predicate");
    }
    for (int parent = 0; parent < node_count; ++parent) {
      if (empty[parent] != 0 || nodes[parent].is_leaf == 1) continue;
      const int first_child = parent * fanout + 1;
      if (first_child < 0 || first_child + fanout > node_count) {
        fail("frozen GTS parent lacks a full contiguous sibling family");
      }
      int shared_pivot = -1;
      float previous_min = -std::numeric_limits<float>::infinity();
      for (int slot = 0; slot < fanout; ++slot) {
        const int child = first_child + slot;
        if (empty[child] != 0) {
          fail("frozen GTS sibling family has an empty child: fail-close to delta");
        }
        const TN& node = nodes[child];
        if (node.pid < 0 || !std::isfinite(node.min_dis)) {
          fail("frozen GTS child pivot/min distance is not a finite native predicate input");
        }
        if (shared_pivot < 0) shared_pivot = node.pid;
        if (node.pid != shared_pivot) {
          fail("frozen GTS sibling family does not share the native pivot");
        }
        if (slot > 0 && node.min_dis + kStrictEpsilon < previous_min) {
          fail("frozen GTS sibling min_dis bridge is nonmonotone");
        }
        previous_min = node.min_dis;
        if (!std::isfinite(max_distance[child]) ||
            max_distance[child] + kStrictEpsilon < node.min_dis) {
          fail("frozen GTS max-distance diagnostic is inconsistent with min_dis");
        }
        // This is exactly the native branch test: a last child has
        // child % TREE_ORDER == 0 and reads no right sibling.  Every other
        // child reads node_list[child + 1].min_dis, regardless of any
        // next-sibling emptiness predicate (there is none in nodeProcessKnn).
        const bool native_last = (child % fanout) == 0;
        if (native_last != (slot == fanout - 1)) {
          fail("frozen GTS heap slot does not match native last-child predicate");
        }
        if (!native_last) {
          const int next = child + 1;
          if (next >= node_count || empty[next] != 0 ||
              !std::isfinite(nodes[next].min_dis)) {
            fail("frozen GTS native next-sibling predicate input is unavailable");
          }
          if (nodes[next].min_dis + kStrictEpsilon < node.min_dis) {
            fail("frozen GTS native next-sibling min_dis bridge is invalid");
          }
          // Diagnostic only: the certificate below uses next.min_dis, never
          // max_dis_d.  Requiring this ordered-tree property prevents an
          // unverified bridge from silently becoming a direct placement.
          if (max_distance[child] > nodes[next].min_dis + kStrictEpsilon) {
            fail("frozen GTS max-distance crosses native next-sibling bridge");
          }
        }
      }
    }
  }

  void capture(const BaseTreeRuntime& runtime) {
    if (!runtime.ready()) fail("cannot capture incomplete GTS runtime");
    const int count = runtime.max_node_num[0];
    if (count <= 0) fail("GTS reports nonpositive node capacity");
    if (runtime.owned_max_dis_d != max_dis_d) {
      fail("tree global max_dis_d ownership changed outside destructive rebuild barrier");
    }
    std::array<int, 3> data_info_values{};
    G3_CUDA(cudaMemcpy(data_info_values.data(), runtime.data_info,
                       data_info_values.size() * sizeof(int), cudaMemcpyDeviceToHost));
    if (data_info_values[0] <= 0 || data_info_values[1] != runtime.base_count ||
        data_info_values[2] != 2) {
      fail("immutable native base payload has invalid dimension/count/L2 contract");
    }
    tree_height = runtime.tree_height;
    fanout = TREE_ORDER;
    node_count = count;
    base_dimension = data_info_values[0];
    metric_code = data_info_values[2];
    data_info_identity = reinterpret_cast<std::uintptr_t>(runtime.data_info);
    data_d_identity = reinterpret_cast<std::uintptr_t>(runtime.data_d);
    id_list_identity = reinterpret_cast<std::uintptr_t>(runtime.id_list);
    node_list_identity = reinterpret_cast<std::uintptr_t>(runtime.node_list);
    max_node_num_identity = reinterpret_cast<std::uintptr_t>(runtime.max_node_num);
    empty_list_identity = reinterpret_cast<std::uintptr_t>(runtime.empty_list);
    max_dis_d_identity = reinterpret_cast<std::uintptr_t>(runtime.owned_max_dis_d);
    local_to_stable_identity = reinterpret_cast<std::uintptr_t>(runtime.local_to_stable.data());
    stable_to_local_identity = reinterpret_cast<std::uintptr_t>(runtime.stable_to_local.data());
    runtime_base_count = runtime.base_count;
    runtime_id_list_capacity = runtime.id_list_capacity;
    runtime_local_to_stable_size = runtime.local_to_stable.size();
    runtime_stable_to_local_size = runtime.stable_to_local.size();
    archived_query_start_idx = start_idx;
    archived_query_tree_order = TREE_ORDER;
    archived_query_dis_code = DIS_CODE;
    archived_query_infi_dis = INFI_DIS;
    archived_query_max_size = MAX_SIZE;
    nodes.assign(static_cast<std::size_t>(count), TN{});
    empty.assign(static_cast<std::size_t>(count), 1);
    max_distance.assign(static_cast<std::size_t>(count), 0.0F);
    G3_CUDA(cudaMemcpy(nodes.data(), runtime.node_list, nodes.size() * sizeof(TN), cudaMemcpyDeviceToHost));
    G3_CUDA(cudaMemcpy(empty.data(), runtime.empty_list, empty.size() * sizeof(int), cudaMemcpyDeviceToHost));
    G3_CUDA(cudaMemcpy(max_distance.data(), runtime.owned_max_dis_d,
                       max_distance.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> immutable_base_values(
        static_cast<std::size_t>(runtime.base_count) * static_cast<std::size_t>(base_dimension));
    if (!immutable_base_values.empty()) {
      G3_CUDA(cudaMemcpy(immutable_base_values.data(), runtime.data_d,
                         immutable_base_values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    if (static_cast<int>(runtime.local_to_stable.size()) != runtime.base_count) {
      fail("local_to_stable count differs from immutable base count");
    }
    immutable_base_stable_ids = runtime.local_to_stable;
    std::sort(immutable_base_stable_ids.begin(), immutable_base_stable_ids.end());
    if (std::adjacent_find(immutable_base_stable_ids.begin(), immutable_base_stable_ids.end()) !=
        immutable_base_stable_ids.end()) {
      fail("immutable compact mapping has duplicate stable IDs");
    }

    std::vector<int> raw_seen(static_cast<std::size_t>(runtime.base_count), 0);
    std::vector<StableId> stable_seen;
    stable_seen.reserve(static_cast<std::size_t>(runtime.base_count));
    std::ostringstream local_leaf_layout;
    std::ostringstream stable_leaf_layout;
    local_leaf_layout << "safe-c1-g3-logical-leaf-local-layout-v2\n";
    stable_leaf_layout << "safe-c1-g3-logical-leaf-stable-layout-v2\n";
    logical_leaf_row_count = 0;
    for (int node_id = 0; node_id < count; ++node_id) {
      if (empty[node_id] != 0 || nodes[node_id].is_leaf != 1) continue;
      const TN& node = nodes[node_id];
      if (node.lid < 0 || node.size < 0 ||
          static_cast<std::uint64_t>(node.lid) + static_cast<std::uint64_t>(node.size) >
              static_cast<std::uint64_t>(runtime.id_list_capacity)) {
        fail("immutable GTS leaf lid/size exceeds physical padded id_list capacity");
      }
      std::vector<LocalRow> local_rows(static_cast<std::size_t>(node.size));
      if (!local_rows.empty()) {
        G3_CUDA(cudaMemcpy(local_rows.data(), runtime.id_list + node.lid,
                           local_rows.size() * sizeof(int), cudaMemcpyDeviceToHost));
      }
      local_leaf_layout << "leaf=" << node_id << ";lid=" << node.lid << ";size=" << node.size << ';';
      stable_leaf_layout << "leaf=" << node_id << ";lid=" << node.lid << ";size=" << node.size << ';';
      for (LocalRow local : local_rows) {
        if (local < 0 || local >= runtime.base_count) {
          fail("immutable GTS leaf contains local row outside compact range");
        }
        if (++raw_seen[local] != 1) fail("immutable GTS leaf raw local rows are not a permutation");
        const StableId stable = runtime.local_to_stable[local];
        if (stable < 0 || stable >= static_cast<int>(runtime.stable_to_local.size()) ||
            runtime.stable_to_local[stable] != local) {
          fail("local_to_stable/stable_to_local immutable bijection violation");
        }
        stable_seen.push_back(stable);
        local_leaf_layout << local << ',';
        stable_leaf_layout << local << ':' << stable << ',';
      }
      local_leaf_layout << '\n';
      stable_leaf_layout << '\n';
      logical_leaf_row_count += node.size;
    }
    if (logical_leaf_row_count != runtime.base_count) {
      fail("immutable GTS leaves do not contain exactly base_count rows");
    }
    for (int count_seen : raw_seen) {
      if (count_seen != 1) fail("immutable GTS raw row coverage is not exactly [0,N)");
    }
    std::sort(stable_seen.begin(), stable_seen.end());
    if (stable_seen != immutable_base_stable_ids) {
      fail("immutable GTS leaf stable IDs do not equal all-and-only immutable base IDs");
    }
    validate_pruning_topology();

    immutable_base_stable_ids_sha256 = stable_set_sha256(immutable_base_stable_ids);
    local_to_stable_sha256_value = local_to_stable_sha256(runtime.local_to_stable);
    logical_leaf_local_rows_sha256 = sha256_text(local_leaf_layout.str());
    logical_leaf_stable_ids_sha256 = sha256_text(stable_leaf_layout.str());
    Sha256 base_payload;
    base_payload.update("safe-c1-g3-immutable-native-base-payload-v2\n");
    base_payload.update(data_info_values.data(), data_info_values.size() * sizeof(int));
    if (!immutable_base_values.empty()) {
      base_payload.update(immutable_base_values.data(), immutable_base_values.size() * sizeof(float));
    }
    if (!runtime.local_to_stable.empty()) {
      base_payload.update(runtime.local_to_stable.data(),
                          runtime.local_to_stable.size() * sizeof(StableId));
    }
    immutable_base_payload_sha256 = base_payload.final_hex();

    Sha256 native_tree;
    native_tree.update("safe-c1-g3-immutable-native-tree-bytes-v2\n");
    native_tree.update(nodes.data(), nodes.size() * sizeof(TN));
    native_tree.update(empty.data(), empty.size() * sizeof(int));
    native_tree.update(max_distance.data(), max_distance.size() * sizeof(float));
    native_tree.update(local_leaf_layout.str());
    native_tree_bytes_sha256 = native_tree.final_hex();

    Sha256 payload;
    payload.update("safe-c1-g3-tree-payload-v2\n");
    payload.update(native_tree_bytes_sha256);
    payload.update(immutable_base_payload_sha256);
    payload.update(immutable_base_stable_ids_sha256);
    payload.update(local_to_stable_sha256_value);
    payload.update(logical_leaf_local_rows_sha256);
    payload.update(logical_leaf_stable_ids_sha256);
    tree_payload_sha256 = payload.final_hex();
  }

  // Explicit full device/base-byte verification for offline diagnostic use.
  // It copies the complete frozen payload and must never be called from a
  // production candidate/query path or this controlled 4K pilot.
  void assert_full_snapshot_unchanged_debug_only(const BaseTreeRuntime& runtime) const {
    if (!initialized()) fail("cannot verify an uninitialized frozen GTS snapshot");
    FrozenTreeSnapshot current;
    current.capture(runtime);
    if (current.tree_payload_sha256 != tree_payload_sha256 ||
        current.native_tree_bytes_sha256 != native_tree_bytes_sha256 ||
        current.immutable_base_payload_sha256 != immutable_base_payload_sha256 ||
        current.logical_leaf_local_rows_sha256 != logical_leaf_local_rows_sha256 ||
        current.logical_leaf_stable_ids_sha256 != logical_leaf_stable_ids_sha256 ||
        current.local_to_stable_sha256_value != local_to_stable_sha256_value ||
        current.immutable_base_stable_ids_sha256 != immutable_base_stable_ids_sha256) {
      fail("immutable native GTS tree/base payload changed outside explicit destructive rebuild");
    }
  }

  // This constant-time identity check deliberately performs no cudaMemcpy and
  // does not iterate frozen vector payloads or leaf rows.  It detects owner
  // replacement/rebuild drift within this Matrix implementation.  Its device
  // immutability boundary is the runner admission contract: one wrapper TU,
  // one linked GTS image, no raw legacy GTS/RP/upload caller or second DSO.
  void assert_lightweight_generation_identity(const BaseTreeRuntime& runtime) const {
    if (!initialized() || !runtime.ready() ||
        runtime.tree_height != tree_height || runtime.base_count != runtime_base_count ||
        runtime.id_list_capacity != runtime_id_list_capacity ||
        runtime.local_to_stable.size() != runtime_local_to_stable_size ||
        runtime.stable_to_local.size() != runtime_stable_to_local_size ||
        start_idx != archived_query_start_idx ||
        TREE_ORDER != archived_query_tree_order || TREE_ORDER != fanout ||
        DIS_CODE != archived_query_dis_code || INFI_DIS != archived_query_infi_dis ||
        MAX_SIZE != archived_query_max_size || MAX_SIZE != G3_CONTROLLED_ARCHIVE_MAX_SIZE ||
        reinterpret_cast<std::uintptr_t>(runtime.data_info) != data_info_identity ||
        reinterpret_cast<std::uintptr_t>(runtime.data_d) != data_d_identity ||
        reinterpret_cast<std::uintptr_t>(runtime.id_list) != id_list_identity ||
        reinterpret_cast<std::uintptr_t>(runtime.node_list) != node_list_identity ||
        reinterpret_cast<std::uintptr_t>(runtime.max_node_num) != max_node_num_identity ||
        reinterpret_cast<std::uintptr_t>(runtime.empty_list) != empty_list_identity ||
        reinterpret_cast<std::uintptr_t>(runtime.owned_max_dis_d) != max_dis_d_identity ||
        reinterpret_cast<std::uintptr_t>(runtime.local_to_stable.data()) != local_to_stable_identity ||
        reinterpret_cast<std::uintptr_t>(runtime.stable_to_local.data()) != stable_to_local_identity ||
        runtime.owned_max_dis_d != max_dis_d) {
      fail("lightweight frozen generation owner/identity drift");
    }
  }

  // This is a host replay over the captured frozen tree snapshot.  It is a
  // diagnostic candidate-placement predicate only, not an actual native vector
  // query traversal receipt.  Direct insertion remains runtime-disabled until
  // a future design cross-checks emitted traversal membership for each object.
  [[nodiscard]] StrictCertificate certify_leaf(const ImmutablePool& pool,
                                                const BaseTreeRuntime& runtime,
                                                StableId inserted) const {
    StrictCertificate output;
    output.all_include_branch_not_relied_on = true;
    if (!initialized() || tree_height <= 0 || fanout <= 1 ||
        kResidualPruningMode != 0) {
      output.failure_reason = "uninitialized_or_nonzero_residual_pruning";
      return output;
    }
    int current = 0;
    if (empty[current] != 0) {
      output.failure_reason = "empty_root";
      return output;
    }
    if (nodes[current].is_leaf == 1) {
      if (tree_height != 1) {
        output.failure_reason = "root_leaf_is_not_final_native_payload_depth";
        return output;
      }
      output.ok = true;
      output.every_predicate_branch_mirrored = true;
      output.sidecar_leaf_id = current;
      return output;
    }
    for (int level = 0; level < tree_height - 1; ++level) {
      if (current < 0 || current >= node_count || empty[current] != 0 ||
          nodes[current].is_leaf == 1) {
        output.failure_reason = "invalid_nonleaf_certificate_parent";
        output.failure_level = level;
        return output;
      }
      const int first_child = current * fanout + 1;
      if (first_child < 0 || first_child + fanout > node_count) {
        output.failure_reason = "incomplete_native_sibling_family";
        output.failure_level = level;
        return output;
      }
      const LocalRow pivot_local = nodes[first_child].pid;
      if (pivot_local < 0 || pivot_local >= runtime.base_count) {
        output.failure_reason = "invalid_pivot_local_row";
        output.failure_level = level;
        return output;
      }
      const StableId pivot_stable = runtime.local_to_stable[pivot_local];
      float sum = 0.0F;
      const std::int16_t* point = pool.vector_for_stable(inserted);
      const std::int16_t* pivot = pool.vector_for_stable(pivot_stable);
      for (int dim = 0; dim < pool.dimension(); ++dim) {
        const float delta = static_cast<float>(point[dim]) - static_cast<float>(pivot[dim]);
        sum += delta * delta;
      }
      const float distance = std::sqrt(sum);
      if (!std::isfinite(distance)) {
        output.failure_reason = "nonfinite_inserted_to_native_pivot_distance";
        output.failure_level = level;
        return output;
      }
      std::vector<CertificateLevel> matches;
      for (int slot = 0; slot < fanout; ++slot) {
        const int child = first_child + slot;
        if (empty[child] != 0 || nodes[child].pid != pivot_local ||
            !std::isfinite(nodes[child].min_dis)) {
          output.failure_reason = "native_sibling_topology_not_certifiable";
          output.failure_level = level;
          return output;
        }
        // This is deliberately the exact branch spelling from nodeProcessKnn:
        // non-last iff child % TREE_ORDER != 0; last children have no right
        // fence.  max_dis_d is never an acceptance bound.
        const bool last_child = (child % fanout) == 0;
        if (last_child != (slot == fanout - 1)) {
          output.failure_reason = "native_last_child_branch_mismatch";
          output.failure_level = level;
          return output;
        }
        int next_sibling = -1;
        float upper = std::numeric_limits<float>::infinity();
        if (!last_child) {
          next_sibling = child + 1;
          if (next_sibling >= node_count || empty[next_sibling] != 0 ||
              !std::isfinite(nodes[next_sibling].min_dis)) {
            output.failure_reason = "native_next_sibling_branch_unavailable";
            output.failure_level = level;
            return output;
          }
          upper = nodes[next_sibling].min_dis;
          if (upper + kStrictEpsilon < nodes[child].min_dis) {
            output.failure_reason = "native_next_sibling_bridge_nonmonotone";
            output.failure_level = level;
            return output;
          }
        }
        const bool matches_native_predicate_interval =
            distance > nodes[child].min_dis + kStrictEpsilon &&
            (last_child || distance < upper - kStrictEpsilon);
        if (matches_native_predicate_interval) {
          matches.push_back(CertificateLevel{
              current, child, next_sibling, pivot_local, pivot_stable,
              nodes[child].min_dis, upper, distance, last_child, true,
              last_child ? GtsKnnPruningBranch::kPredicateLastSibling
                         : GtsKnnPruningBranch::kPredicateNonLastSibling});
        }
      }
      output.matching_children = static_cast<int>(matches.size());
      if (matches.size() != 1U) {
        output.failure_reason = matches.empty() ? "native_sibling_gap_or_boundary"
                                                : "native_sibling_overlap_or_ambiguous";
        output.failure_level = level;
        return output;
      }
      const CertificateLevel chosen = matches.front();
      output.levels.push_back(chosen);
      current = chosen.child;
      if (nodes[current].is_leaf == 1) {
        if (level + 1 != tree_height - 1) {
          output.failure_reason = "leaf_is_not_at_final_native_knn_payload_depth";
          output.failure_level = level;
          return output;
        }
        output.ok = true;
        output.every_predicate_branch_mirrored = true;
        output.sidecar_leaf_id = current;
        return output;
      }
    }
    output.failure_reason = "depth_exhausted_without_final_native_leaf";
    output.failure_level = tree_height - 1;
    return output;
  }

  // This observer deliberately obtains the outcome from certify_leaf() itself;
  // the sibling records below are explanatory raw inputs only.  Thus a
  // diagnostic cannot become a second placement predicate or a direct-admission
  // path.  It traverses only the existing certifier's selected prefix and stops
  // at the failure level when there is no unique selected child.
  [[nodiscard]] FrozenCertificateDiagnostic inspect_certificate(
      const ImmutablePool& pool, const BaseTreeRuntime& runtime, StableId inserted) const {
    FrozenCertificateDiagnostic output;
    output.certificate = certify_leaf(pool, runtime, inserted);
    if (!initialized() || tree_height <= 0 || fanout <= 1 ||
        kResidualPruningMode != 0 || inserted < 0 ||
        inserted >= pool.stable_capacity()) {
      return output;
    }

    int current = 0;
    for (int level_index = 0; level_index < tree_height - 1; ++level_index) {
      CertificateDiagnosticLevel level;
      level.level = level_index;
      level.parent = current;
      if (current < 0 || current >= node_count || empty[current] != 0 ||
          nodes[current].is_leaf == 1) {
        output.levels.push_back(std::move(level));
        break;
      }
      const int first_child = current * fanout + 1;
      if (first_child < 0 || first_child + fanout > node_count) {
        output.levels.push_back(std::move(level));
        break;
      }
      const LocalRow pivot_local = nodes[first_child].pid;
      if (pivot_local < 0 || pivot_local >= runtime.base_count) {
        output.levels.push_back(std::move(level));
        break;
      }
      const StableId pivot_stable = runtime.local_to_stable[pivot_local];
      level.parent_valid = true;
      level.parent_pivot_local_row = pivot_local;
      level.parent_pivot_stable_id = pivot_stable;
      const std::int16_t* point = pool.vector_for_stable(inserted);
      const std::int16_t* pivot = pool.vector_for_stable(pivot_stable);
      float sum = 0.0F;
      for (int dim = 0; dim < pool.dimension(); ++dim) {
        const float delta = static_cast<float>(point[dim]) - static_cast<float>(pivot[dim]);
        sum += delta * delta;
      }
      const float distance = std::sqrt(sum);
      level.pivot_distance = distance;

      for (int slot = 0; slot < fanout; ++slot) {
        const int child = first_child + slot;
        CertificateSiblingDiagnostic sibling;
        sibling.child = child;
        sibling.last_child = (child % fanout) == 0;
        sibling.upper_is_unbounded_last_child = sibling.last_child;
        if (child >= 0 && child < node_count) {
          sibling.child_empty = empty[child] != 0;
          if (!sibling.child_empty) {
            sibling.child_pivot_local_row = nodes[child].pid;
            sibling.child_pivot_matches_parent = nodes[child].pid == pivot_local;
            sibling.child_min_dis_finite = std::isfinite(nodes[child].min_dis);
            sibling.lower = nodes[child].min_dis;
          }
        }
        sibling.pivot_distance = distance;
        if (!sibling.last_child) {
          sibling.next_sibling = child + 1;
          if (sibling.next_sibling >= 0 && sibling.next_sibling < node_count) {
            sibling.next_sibling_available = empty[sibling.next_sibling] == 0;
            if (sibling.next_sibling_available) {
              sibling.next_sibling_min_dis_finite =
                  std::isfinite(nodes[sibling.next_sibling].min_dis);
              sibling.upper = nodes[sibling.next_sibling].min_dis;
            }
          }
        }
        if (sibling.child_pivot_local_row >= 0 &&
            sibling.child_pivot_local_row < runtime.base_count) {
          sibling.child_pivot_stable_id =
              runtime.local_to_stable[sibling.child_pivot_local_row];
        }
        const bool upper_ok = sibling.last_child ||
            (sibling.next_sibling_available && sibling.next_sibling_min_dis_finite);
        if (std::isfinite(distance) && !sibling.child_empty &&
            sibling.child_pivot_matches_parent && sibling.child_min_dis_finite && upper_ok) {
          sibling.strict_native_predicate_match =
              distance > sibling.lower + kStrictEpsilon &&
              (sibling.last_child || distance < sibling.upper - kStrictEpsilon);
          if (sibling.strict_native_predicate_match) {
            ++level.diagnostic_strict_match_count;
          }
        }
        level.siblings.push_back(sibling);
      }
      output.levels.push_back(std::move(level));
      if (static_cast<std::size_t>(level_index) >= output.certificate.levels.size()) {
        break;
      }
      const CertificateLevel& chosen = output.certificate.levels[static_cast<std::size_t>(level_index)];
      if (chosen.parent_leaf_or_internal != current || chosen.child < 0 ||
          chosen.child >= node_count || nodes[chosen.child].is_leaf == 1) {
        break;
      }
      current = chosen.child;
    }
    return output;
  }

};

enum class PlacementKind { kBase, kDirect, kDelta, kDeleted, kUnknown };

inline const char* placement_name(PlacementKind kind) {
  switch (kind) {
    case PlacementKind::kBase: return "base";
    case PlacementKind::kDirect: return "direct";
    case PlacementKind::kDelta: return "delta";
    case PlacementKind::kDeleted: return "deleted";
    default: return "unknown";
  }
}

struct Placement {
  PlacementKind kind = PlacementKind::kUnknown;
  int sidecar_leaf_id = -1;
  StrictCertificate certificate;
  std::string fallback_reason;
  int sidecar_size_before = -1;
  // Direct placements bind to the frozen generation that certified their
  // sibling-min path; a later rebuild clears them rather than reusing it.
  std::uint64_t frozen_tree_version = 0;
  std::string frozen_tree_payload_sha256;
};

class SafeC1State {
 public:
  SafeC1State() = default;
  SafeC1State(int stable_capacity, int leaf_capacity) { reset(stable_capacity, leaf_capacity); }

  void reset(int stable_capacity, int leaf_capacity) {
    if (stable_capacity <= 0 || leaf_capacity <= 0) fail("invalid Safe-C1 state capacity");
    active_.assign(static_cast<std::size_t>(stable_capacity), 0);
    placement_.assign(static_cast<std::size_t>(stable_capacity), Placement{});
    sidecars_.clear();
    delta_.clear();
    leaf_capacity_ = leaf_capacity;
  }

  void initialize_base(const std::vector<StableId>& base_ids) {
    for (StableId stable : base_ids) {
      check_stable(stable);
      if (active_[stable] != 0) fail("duplicate immutable base stable ID when initializing state");
      active_[stable] = 1;
      placement_[stable] = Placement{PlacementKind::kBase, -1, StrictCertificate{}, "", -1};
    }
  }

  // This method mutates only a caller-owned draft.  NativeSafeC1Matrix commits
  // that draft with swap_noexcept only after frozen-tree and partition checks
  // pass, so a failed postcheck cannot leave a partial public state.
  [[nodiscard]] Placement insert(StableId stable, const FrozenTreeSnapshot& frozen,
                                 const BaseTreeRuntime& runtime, const ImmutablePool& pool,
                                 std::uint64_t tree_version,
                                 bool direct_visibility_runtime_gate_open) {
    check_stable(stable);
    if (active_[stable] != 0) fail("inserted stable ID is already live");
    const StrictCertificate certificate = frozen.certify_leaf(pool, runtime, stable);
    Placement next;
    next.certificate = certificate;
    next.frozen_tree_version = tree_version;
    next.frozen_tree_payload_sha256 = frozen.tree_payload_sha256;
    if (certificate.ok && direct_visibility_runtime_gate_open) {
      auto& values = sidecars_[certificate.sidecar_leaf_id];
      next.sidecar_size_before = static_cast<int>(values.size());
      if (next.sidecar_size_before < leaf_capacity_) {
        values.push_back(stable);
        next.kind = PlacementKind::kDirect;
        next.sidecar_leaf_id = certificate.sidecar_leaf_id;
        active_[stable] = 1;
        placement_[stable] = next;
        return next;
      }
      next.kind = PlacementKind::kDelta;
      next.fallback_reason = "capacity";
    } else {
      next.kind = PlacementKind::kDelta;
      if (!certificate.ok) {
        next.fallback_reason = certificate.failure_reason.empty()
                                   ? "native_predicate_certificate"
                                   : certificate.failure_reason;
      } else {
        // Safe default for this static phase: no direct placement becomes live
        // until a future runtime guard has independently bound predicate/
        // receipt/oracle evidence for this exact source closure.
        next.fallback_reason = "direct_visibility_runtime_guard_closed";
      }
    }
    delta_.push_back(stable);
    active_[stable] = 1;
    placement_[stable] = next;
    return next;
  }

  [[nodiscard]] Placement erase_mutable(StableId stable) {
    check_stable(stable);
    if (active_[stable] == 0) fail("delete targets inactive stable ID");
    Placement prior = placement_[stable];
    if (prior.kind == PlacementKind::kBase) {
      fail("immutable base deletion requires an explicit fresh destructive rebuild plan");
    }
    if (prior.kind == PlacementKind::kDirect) {
      auto sidecar = sidecars_.find(prior.sidecar_leaf_id);
      if (sidecar == sidecars_.end()) fail("direct placement has no sidecar");
      auto& values = sidecar->second;
      const auto erase_at = std::find(values.begin(), values.end(), stable);
      if (erase_at == values.end()) fail("direct stable ID missing from recorded sidecar");
      values.erase(erase_at);
      if (values.empty()) sidecars_.erase(sidecar);
    } else if (prior.kind == PlacementKind::kDelta) {
      const auto erase_at = std::find(delta_.begin(), delta_.end(), stable);
      if (erase_at == delta_.end()) fail("delta stable ID missing from global delta");
      delta_.erase(erase_at);
    } else {
      fail("active stable ID has invalid nonmutable placement");
    }
    active_[stable] = 0;
    placement_[stable] = Placement{PlacementKind::kDeleted, -1, StrictCertificate{}, "", -1};
    return prior;
  }

  // Called only on a prevalidated replacement state before it is published as
  // the new destructive generation.  It intentionally does not mutate the
  // old dynamic state in place.
  void after_successful_rebuild(const std::vector<StableId>& immutable_base_ids) {
    std::fill(active_.begin(), active_.end(), 0);
    std::fill(placement_.begin(), placement_.end(), Placement{});
    sidecars_.clear();
    delta_.clear();
    initialize_base(immutable_base_ids);
  }

  // Exact control-plane invariant: active == immutable-base disjoint-union
  // sidecars disjoint-union delta.  It is never an answer path and never scans
  // vector data; query execution still scans only receipt leaves + global delta.
  void assert_partition(const FrozenTreeSnapshot& frozen,
                        std::uint64_t current_tree_version) const {
    if (static_cast<int>(active_.size()) != static_cast<int>(placement_.size())) {
      fail("Safe-C1 active/placement capacity mismatch");
    }
    std::vector<std::uint8_t> owner(active_.size(), 0);
    for (StableId stable : frozen.immutable_base_stable_ids) {
      check_stable(stable);
      if (owner[stable] != 0) fail("duplicate stable ID in immutable base partition");
      owner[stable] = 1;
      if (active_[stable] == 0 || placement_[stable].kind != PlacementKind::kBase) {
        fail("immutable base stable ID is not active/base in Safe-C1 partition");
      }
    }
    for (const auto& sidecar : sidecars_) {
      const int leaf = sidecar.first;
      const std::vector<StableId>& values = sidecar.second;
      if (leaf < 0 || leaf >= frozen.node_count || frozen.empty[leaf] != 0 ||
          frozen.nodes[leaf].is_leaf != 1 || values.empty() ||
          static_cast<int>(values.size()) > leaf_capacity_) {
        fail("sidecar partition has invalid frozen leaf/capacity");
      }
      for (StableId stable : values) {
        check_stable(stable);
        if (owner[stable] != 0) fail("base/sidecar/delta partitions are not pairwise disjoint");
        owner[stable] = 2;
        const Placement& placement = placement_[stable];
        if (active_[stable] == 0 || placement.kind != PlacementKind::kDirect ||
            placement.sidecar_leaf_id != leaf || !placement.certificate.ok ||
            placement.certificate.sidecar_leaf_id != leaf ||
            !placement.certificate.every_predicate_branch_mirrored ||
            placement.frozen_tree_version != current_tree_version ||
            placement.frozen_tree_payload_sha256 != frozen.tree_payload_sha256) {
          fail("direct sidecar placement does not bind to current frozen certificate");
        }
      }
    }
    for (StableId stable : delta_) {
      check_stable(stable);
      if (owner[stable] != 0) fail("base/sidecar/delta partitions are not pairwise disjoint");
      owner[stable] = 3;
      if (active_[stable] == 0 || placement_[stable].kind != PlacementKind::kDelta) {
        fail("delta placement does not match active partition");
      }
    }
    for (StableId stable = 0; stable < static_cast<StableId>(active_.size()); ++stable) {
      if (active_[stable] != 0) {
        if (owner[stable] == 0) fail("active stable ID is absent from base/sidecar/delta partition");
      } else if (owner[stable] != 0) {
        fail("inactive stable ID appears in base/sidecar/delta partition");
      }
    }
  }

  void swap_noexcept(SafeC1State& other) noexcept {
    using std::swap;
    swap(active_, other.active_);
    swap(placement_, other.placement_);
    swap(sidecars_, other.sidecars_);
    swap(delta_, other.delta_);
    swap(leaf_capacity_, other.leaf_capacity_);
  }

  [[nodiscard]] std::vector<StableId> sidecar_candidates_for(
      const std::vector<int>& visited_leaf_ids) const {
    std::set<int> visited(visited_leaf_ids.begin(), visited_leaf_ids.end());
    std::vector<StableId> output;
    for (int leaf : visited) {
      const auto it = sidecars_.find(leaf);
      if (it == sidecars_.end()) continue;
      output.insert(output.end(), it->second.begin(), it->second.end());
    }
    std::sort(output.begin(), output.end());
    if (std::adjacent_find(output.begin(), output.end()) != output.end()) {
      fail("a live direct stable ID appears in more than one receipt sidecar");
    }
    return output;
  }

  [[nodiscard]] std::vector<std::pair<int, StableId>> sidecar_leaf_pairs_for(
      const std::vector<int>& visited_leaf_ids) const {
    std::set<int> visited(visited_leaf_ids.begin(), visited_leaf_ids.end());
    std::vector<std::pair<int, StableId>> output;
    for (int leaf : visited) {
      const auto it = sidecars_.find(leaf);
      if (it == sidecars_.end()) continue;
      for (StableId stable : it->second) output.emplace_back(leaf, stable);
    }
    std::sort(output.begin(), output.end());
    return output;
  }

  [[nodiscard]] std::vector<StableId> delta_ids() const {
    std::vector<StableId> output = delta_;
    std::sort(output.begin(), output.end());
    if (std::adjacent_find(output.begin(), output.end()) != output.end()) {
      fail("global delta contains duplicate stable ID");
    }
    return output;
  }

  [[nodiscard]] std::vector<StableId> live_ids() const {
    std::vector<StableId> output;
    for (StableId stable = 0; stable < static_cast<StableId>(active_.size()); ++stable) {
      if (active_[stable] != 0) output.push_back(stable);
    }
    return output;
  }

  [[nodiscard]] const Placement& placement(StableId stable) const {
    check_stable(stable);
    return placement_[stable];
  }
  [[nodiscard]] bool is_active(StableId stable) const { check_stable(stable); return active_[stable] != 0; }
  [[nodiscard]] int sidecar_live() const {
    int total = 0;
    for (const auto& row : sidecars_) total += static_cast<int>(row.second.size());
    return total;
  }
  [[nodiscard]] int delta_live() const { return static_cast<int>(delta_.size()); }

 private:
  void check_stable(StableId stable) const {
    if (stable < 0 || stable >= static_cast<StableId>(active_.size())) {
      fail("stable ID outside configured immutable mapping");
    }
  }

  std::vector<std::uint8_t> active_;
  std::vector<Placement> placement_;
  std::map<int, std::vector<StableId>> sidecars_;
  std::vector<StableId> delta_;
  int leaf_capacity_ = 0;
};

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { allocate(count); }
  ~DeviceBuffer() { reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept { if (this != &other) { reset(); swap(other); } return *this; }

  void allocate(std::size_t count) {
    reset();
    if (count == 0) return;
    G3_CUDA(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    count_ = count;
  }
  void reset() noexcept {
    if (data_) cudaFree(data_);
    data_ = nullptr;
    count_ = 0;
  }
  void copy_from_host(const std::vector<T>& input) {
    allocate(input.size());
    if (!input.empty()) G3_CUDA(cudaMemcpy(data_, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  [[nodiscard]] std::vector<T> copy_to_host() const {
    std::vector<T> output(count_);
    if (!output.empty()) G3_CUDA(cudaMemcpy(output.data(), data_, output.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return output;
  }
  [[nodiscard]] T* data() { return data_; }
  [[nodiscard]] const T* data() const { return data_; }
  [[nodiscard]] std::size_t size() const { return count_; }

 private:
  void swap(DeviceBuffer& other) noexcept { std::swap(data_, other.data_); std::swap(count_, other.count_); }
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

struct RangeLeafPair { int query_id; int leaf_id; };
// Audit identity for one physical native id_list slot materialized only after
// a traversal receipt selected its leaf.  `id_list_slot` is intentionally
// physical/padded; it is not a compact local-row index.
struct ReceiptBaseRow {
  int leaf_id = -1;
  int id_list_slot = -1;
  LocalRow local_row = -1;
  StableId stable_id = -1;
};

// One exact physical span selected by the traversal receipt.  Exporting this
// together with every ReceiptBaseRow lets a future CPU validator establish
// that rows cover [lid,lid+size) without a MAX_SIZE truncation or an N-sized
// id_list assumption.
struct ReceiptLeafSpan {
  int leaf_id = -1;
  int id_list_lid = -1;
  int size = -1;
};

__device__ __forceinline__ unsigned long long g3_exact_l2sq(const float* left,
                                                              const float* right,
                                                              int dimension) {
  unsigned long long sum = 0ULL;
  for (int dim = 0; dim < dimension; ++dim) {
    const long long a = llrintf(left[dim]);
    const long long b = llrintf(right[dim]);
    const long long delta = a - b;
    sum += static_cast<unsigned long long>(delta * delta);
  }
  return sum;
}

__global__ void g3_set_range_root(unsigned char* flags, int qnum, int node_count) {
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query < qnum) flags[query * node_count] = 1U;
}

__global__ void g3_mark_range_level(const TN* nodes, const int* empty_list,
                                    const float* base_data, const float* query_data,
                                    int dimension, int fanout, int node_count,
                                    int level_start, int level_count, int qnum,
                                    float conservative_radius,
                                    unsigned char* flags) {
  const int work = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = qnum * level_count;
  if (work >= total) return;
  const int query = work / level_count;
  const int node_id = level_start + (work % level_count);
  if (node_id <= 0 || node_id >= node_count || empty_list[node_id] != 0) return;
  const int parent = (node_id - 1) / fanout;
  if (parent < 0 || parent >= node_count || flags[query * node_count + parent] == 0U) return;
  const TN child = nodes[node_id];
  if (child.pid < 0) return;
  const float* pivot = base_data + static_cast<std::size_t>(child.pid) * dimension;
  const float* q = query_data + static_cast<std::size_t>(query) * dimension;
  float squared = 0.0F;
  for (int dim = 0; dim < dimension; ++dim) {
    const float delta = pivot[dim] - q[dim];
    squared += delta * delta;
  }
  const float distance = sqrtf(squared);
  float lower_bound = fmaxf(child.min_dis - distance, 0.0F);
  // Exact branch mirror of nodeProcessRnn/nodeProcessKnn in the pinned GTS
  // source: non-last is `nid % TREE_ORDER != 0`, and that branch reads the
  // next sibling's min_dis without an empty-list exception.  Snapshot capture
  // rejects a topology for which that read would be unavailable; do not weaken
  // the branch here, because weakening would no longer be the same predicate.
  const bool native_last_child = (node_id % fanout) == 0;
  if (!native_last_child) {
    const int next = node_id + 1;
    if (next >= node_count) return;  // snapshot validation fail-closes first.
    lower_bound = fmaxf(lower_bound, distance - nodes[next].min_dis);
  }
  if (lower_bound <= conservative_radius) {
    flags[query * node_count + node_id] = 1U;
  }
}

__global__ void g3_scan_local_candidates(const float* base_data, const int* local_rows,
                                         int count, const float* query, int dimension,
                                         unsigned long long* distances) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int local = local_rows[index];
  distances[index] = g3_exact_l2sq(base_data + static_cast<std::size_t>(local) * dimension,
                                    query, dimension);
}

__global__ void g3_scan_stable_candidates(const float* pool_data,
                                          const int* stable_to_pool_row,
                                          const int* stable_ids, int count,
                                          const float* query, int dimension,
                                          unsigned long long* distances) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int stable = stable_ids[index];
  const int pool_row = stable_to_pool_row[stable];
  distances[index] = g3_exact_l2sq(pool_data + static_cast<std::size_t>(pool_row) * dimension,
                                    query, dimension);
}

struct TraversalReceipt {
  std::vector<StableDistance> base_results;
  std::vector<int> visited_leaf_ids;
  std::vector<ReceiptBaseRow> base_receipt_rows;
  int raw_receipt_leaf_pair_count = 0;
  int unique_receipt_leaf_count = 0;
  // KNN res_ids are diagnostics only.  They must be attributable to an
  // already materialized receipt row and never define the candidate set.
  std::vector<LocalRow> native_final_res_ids;
  bool native_final_res_ids_cross_checked = false;
  std::uint64_t tree_version = 0;
  std::string tree_payload_sha256;
  std::string native_tree_bytes_sha256;
  std::string immutable_base_stable_ids_sha256;
  std::string immutable_base_payload_sha256;
  std::string local_to_stable_sha256_value;
  bool receipt_before_leaf_materialization = false;
  bool range_predicate_mirror_branch_aligned = false;
  // A typed, receipt-hashed statement that range used every current immutable
  // base row rather than a native/branch-mirror leaf receipt.  This fallback is
  // correctness-only and must never be re-labelled as GTS range traversal.
  bool exact_full_immutable_base_range_fallback = false;
  int full_immutable_base_candidate_count = 0;
  int id_list_capacity = 0;
  std::vector<ReceiptLeafSpan> receipt_leaf_spans;
  std::string base_path;
};

inline int level_start_for(int depth, int fanout) {
  int start = 0;
  int width = 1;
  for (int level = 0; level < depth; ++level) {
    start += width;
    if (width > std::numeric_limits<int>::max() / fanout) fail("tree level offset overflow");
    width *= fanout;
  }
  return start;
}

inline float conservative_range_radius(DistanceSq radius_sq) {
  const double exact = std::sqrt(static_cast<double>(radius_sq));
  float output = static_cast<float>(exact);
  for (int step = 0; step < kRangeRadiusSafetyUlps; ++step) {
    output = std::nextafter(output, std::numeric_limits<float>::infinity());
  }
  return output;
}

inline std::vector<int> normalized_local_rows(const std::vector<int>& rows, int upper) {
  std::vector<int> output = rows;
  for (int local : output) {
    if (local < 0 || local >= upper) fail("GTS traversal returned local row outside compact map");
  }
  std::sort(output.begin(), output.end());
  output.erase(std::unique(output.begin(), output.end()), output.end());
  return output;
}

inline std::vector<StableDistance> scan_local_rows_exact(const BaseTreeRuntime& runtime,
                                                          const float* query_device,
                                                          const std::vector<int>& input_rows) {
  const std::vector<int> rows = normalized_local_rows(input_rows, runtime.base_count);
  if (rows.empty()) return {};
  DeviceBuffer<int> rows_device;
  rows_device.copy_from_host(rows);
  DeviceBuffer<unsigned long long> distances_device(rows.size());
  constexpr int threads = 256;
  const int blocks = static_cast<int>((rows.size() + threads - 1U) / threads);
  g3_scan_local_candidates<<<blocks, threads>>>(runtime.data_d, rows_device.data(),
                                                 static_cast<int>(rows.size()), query_device,
                                                 runtime.data_info[0], distances_device.data());
  G3_CUDA(cudaGetLastError());
  G3_CUDA(cudaDeviceSynchronize());
  const std::vector<unsigned long long> distances = distances_device.copy_to_host();
  std::vector<StableDistance> output;
  output.reserve(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    output.push_back(StableDistance{runtime.local_to_stable[rows[index]], distances[index]});
  }
  sort_and_require_unique(&output, "local GTS candidate export");
  return output;
}

inline std::vector<StableDistance> scan_stable_rows_exact(const ImmutablePool& pool,
                                                           const float* query_device,
                                                           const std::vector<StableId>& input_ids) {
  std::vector<StableId> ids = input_ids;
  for (StableId stable : ids) {
    if (stable < 0 || stable >= pool.stable_capacity()) fail("candidate stable ID outside immutable mapping");
  }
  std::sort(ids.begin(), ids.end());
  if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
    fail("stable candidate scan received duplicate ID");
  }
  if (ids.empty()) return {};
  DeviceBuffer<int> ids_device;
  ids_device.copy_from_host(ids);
  DeviceBuffer<unsigned long long> distances_device(ids.size());
  constexpr int threads = 256;
  const int blocks = static_cast<int>((ids.size() + threads - 1U) / threads);
  g3_scan_stable_candidates<<<blocks, threads>>>(pool.device_values(), pool.device_stable_to_pool_row(),
                                                  ids_device.data(), static_cast<int>(ids.size()),
                                                  query_device, pool.dimension(), distances_device.data());
  G3_CUDA(cudaGetLastError());
  G3_CUDA(cudaDeviceSynchronize());
  const std::vector<unsigned long long> distances = distances_device.copy_to_host();
  std::vector<StableDistance> output;
  output.reserve(ids.size());
  for (std::size_t index = 0; index < ids.size(); ++index) {
    output.push_back(StableDistance{ids[index], distances[index]});
  }
  sort_and_require_unique(&output, "stable candidate export");
  return output;
}


// Materialize all physical rows of receipt-selected leaves after the receipt
// exists.  This is deliberately not derived from res_ids and not capped by
// MAX_SIZE: tree.cuh can repack padded id_list slots and can leave a leaf with
// a true size greater than the legacy leaf kernel's fixed bound.
inline std::vector<ReceiptLeafSpan> receipt_leaf_spans(
    const BaseTreeRuntime& runtime, const FrozenTreeSnapshot& snapshot,
    const std::vector<int>& receipt_leaf_ids) {
  std::vector<int> leaves = receipt_leaf_ids;
  std::sort(leaves.begin(), leaves.end());
  leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
  std::vector<ReceiptLeafSpan> output;
  output.reserve(leaves.size());
  for (int leaf_id : leaves) {
    if (leaf_id < 0 || leaf_id >= snapshot.node_count || snapshot.empty[leaf_id] != 0 ||
        snapshot.nodes[leaf_id].is_leaf != 1) {
      fail("receipt contains a non-leaf or absent native GTS node");
    }
    const TN& leaf = snapshot.nodes[leaf_id];
    if (leaf.lid < 0 || leaf.size < 0 ||
        static_cast<std::uint64_t>(leaf.lid) + static_cast<std::uint64_t>(leaf.size) >
            static_cast<std::uint64_t>(runtime.id_list_capacity)) {
      fail("receipt leaf exceeds the physical padded id_list allocation");
    }
    output.push_back(ReceiptLeafSpan{leaf_id, leaf.lid, leaf.size});
  }
  return output;
}

inline std::vector<ReceiptBaseRow> materialize_receipt_leaf_rows(
    const BaseTreeRuntime& runtime, const FrozenTreeSnapshot& snapshot,
    const std::vector<int>& receipt_leaf_ids) {
  const std::vector<ReceiptLeafSpan> spans =
      receipt_leaf_spans(runtime, snapshot, receipt_leaf_ids);
  std::set<LocalRow> seen_local_rows;
  std::vector<ReceiptBaseRow> output;
  for (const ReceiptLeafSpan& span : spans) {
    std::vector<LocalRow> local_rows(static_cast<std::size_t>(span.size));
    if (!local_rows.empty()) {
      // This D2H copy is intentionally after receipt capture.  It reads the
      // exact native physical payload [lid,lid+size), not a compact-row guess.
      G3_CUDA(cudaMemcpy(local_rows.data(), runtime.id_list + span.id_list_lid,
                         local_rows.size() * sizeof(LocalRow), cudaMemcpyDeviceToHost));
    }
    for (int offset = 0; offset < span.size; ++offset) {
      const LocalRow local = local_rows[static_cast<std::size_t>(offset)];
      if (local < 0 || local >= runtime.base_count) {
        fail("receipt leaf native id_list payload has an invalid compact local row");
      }
      if (!seen_local_rows.insert(local).second) {
        fail("receipt-selected leaves share a native local row; cannot silently deduplicate");
      }
      const StableId stable = runtime.local_to_stable[local];
      if (stable < 0 || stable >= static_cast<int>(runtime.stable_to_local.size()) ||
          runtime.stable_to_local[stable] != local) {
        fail("receipt leaf local-to-stable mapping is not an immutable bijection");
      }
      output.push_back(ReceiptBaseRow{span.leaf_id, span.id_list_lid + offset, local, stable});
    }
  }
  return output;
}

inline TraversalReceipt run_gts_base_topk_with_receipt(
    BaseTreeRuntime& runtime, const float* query_device, int qnum, int k,
    std::uint64_t tree_version, const FrozenTreeSnapshot& snapshot) {
  if (!runtime.ready() || query_device == nullptr || qnum != 1 || k <= 0) {
    fail("G3 top-k adapter requires one valid external vector query and positive k");
  }
  int* local_result_ids = nullptr;
  G3_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&local_result_ids),
                            static_cast<std::size_t>(qnum) * k * sizeof(int)));
  for (int index = 0; index < qnum * k; ++index) local_result_ids[index] = -1;
  try {
    // c_rp_mode is configured to zero by NativeSafeC1Matrix.  The copied GTS
    // traversal itself produces this receipt before it scans GTS base leaves.
    update_disk = false;
    searchIndexKnnV2(runtime.data_d, runtime.node_list, runtime.id_list,
                      runtime.max_node_num, const_cast<float*>(query_device), local_result_ids,
                      qnum, k, runtime.tree_height, runtime.data_info,
                      runtime.empty_list, runtime.data_s, runtime.size_s);
    G3_CUDA(cudaDeviceSynchronize());
    G3_CUDA(cudaGetLastError());

    TraversalReceipt output;
    output.raw_receipt_leaf_pair_count =
        static_cast<int>(safe_c1_last_visited_leaf_pairs.size());
    for (const SafeC1VisitedLeafPair& pair : safe_c1_last_visited_leaf_pairs) {
      if (pair.query_id != 0 || pair.leaf_id < 0 || pair.leaf_id >= snapshot.node_count ||
          snapshot.empty[pair.leaf_id] != 0 || snapshot.nodes[pair.leaf_id].is_leaf != 1) {
        fail("real GTS top-k receipt emitted invalid leaf/query pair");
      }
      output.visited_leaf_ids.push_back(pair.leaf_id);
    }
    std::sort(output.visited_leaf_ids.begin(), output.visited_leaf_ids.end());
    output.visited_leaf_ids.erase(
        std::unique(output.visited_leaf_ids.begin(), output.visited_leaf_ids.end()),
        output.visited_leaf_ids.end());
    output.unique_receipt_leaf_count = static_cast<int>(output.visited_leaf_ids.size());

    // Receipt boundary: the actual GTS traversal has selected the leaf pairs;
    // only now may G3 materialize every physical id_list row of those leaves.
    output.receipt_before_leaf_materialization = true;
    output.id_list_capacity = runtime.id_list_capacity;
    output.receipt_leaf_spans =
        receipt_leaf_spans(runtime, snapshot, output.visited_leaf_ids);
    output.base_receipt_rows =
        materialize_receipt_leaf_rows(runtime, snapshot, output.visited_leaf_ids);
    std::vector<LocalRow> receipt_locals;
    receipt_locals.reserve(output.base_receipt_rows.size());
    std::set<LocalRow> receipt_local_set;
    for (const ReceiptBaseRow& row : output.base_receipt_rows) {
      receipt_locals.push_back(row.local_row);
      if (!receipt_local_set.insert(row.local_row).second) {
        fail("top-k receipt materialization did not preserve one owner per native local row");
      }
    }
    // All base candidates come from the full receipt leaf payload.  Native
    // res_ids are deliberately not an input to this exact scan.
    output.base_results = scan_local_rows_exact(runtime, query_device, receipt_locals);

    // res_ids are retained only as a diagnostic cross-check.  A non-sentinel
    // result not attributable to a receipt leaf is a fail-close mismatch.
    for (int index = 0; index < qnum * k; ++index) {
      const LocalRow local = local_result_ids[index];
      output.native_final_res_ids.push_back(local);
      if (local == -1) continue;
      if (local < 0 || local >= runtime.base_count) {
        fail("native top-k res_ids contains an invalid non-sentinel local row");
      }
      if (receipt_local_set.find(local) == receipt_local_set.end()) {
        fail("native top-k res_ids is not attributable to a receipt leaf payload");
      }
    }
    output.native_final_res_ids_cross_checked = true;
    output.tree_version = tree_version;
    output.tree_payload_sha256 = snapshot.tree_payload_sha256;
    output.native_tree_bytes_sha256 = snapshot.native_tree_bytes_sha256;
    output.immutable_base_stable_ids_sha256 = snapshot.immutable_base_stable_ids_sha256;
    output.immutable_base_payload_sha256 = snapshot.immutable_base_payload_sha256;
    output.local_to_stable_sha256_value = snapshot.local_to_stable_sha256_value;
    output.base_path = "real_gts_vector_topk_receipt_all_native_leaf_rows";
    if (res_dis) { G3_CUDA(cudaFree(res_dis)); res_dis = nullptr; }
    G3_CUDA(cudaFree(local_result_ids));
    return output;
  } catch (...) {
    if (res_dis) { cudaFree(res_dis); res_dis = nullptr; }
    cudaFree(local_result_ids);
    throw;
  }
}

inline TraversalReceipt run_gts_base_range_with_receipt(
    BaseTreeRuntime& runtime, const float* query_device, int qnum, DistanceSq radius_sq,
    std::uint64_t tree_version, const FrozenTreeSnapshot& snapshot) {
  if (!runtime.ready() || query_device == nullptr || qnum != 1 || !snapshot.initialized()) {
    fail("G3 vector range adapter requires one valid frozen GTS runtime");
  }
  const int node_count = snapshot.node_count;
  if (node_count <= 0 || snapshot.fanout <= 1) fail("invalid frozen GTS node topology for range");
  DeviceBuffer<unsigned char> flags(static_cast<std::size_t>(qnum) * node_count);
  G3_CUDA(cudaMemset(flags.data(), 0, flags.size() * sizeof(unsigned char)));
  constexpr int threads = 256;
  g3_set_range_root<<<(qnum + threads - 1) / threads>>>(flags.data(), qnum, node_count);
  G3_CUDA(cudaGetLastError());
  for (int depth = 1; depth < runtime.tree_height; ++depth) {
    const int start = level_start_for(depth, snapshot.fanout);
    if (start >= node_count) break;
    const int next = level_start_for(depth + 1, snapshot.fanout);
    const int count = std::min(node_count, next) - start;
    if (count <= 0) continue;
    const int work = qnum * count;
    g3_mark_range_level<<<(work + threads - 1) / threads>>>(
        runtime.node_list, runtime.empty_list, runtime.data_d, query_device,
        runtime.data_info[0], snapshot.fanout, node_count, start, count, qnum,
        conservative_range_radius(radius_sq), flags.data());
    G3_CUDA(cudaGetLastError());
  }
  G3_CUDA(cudaDeviceSynchronize());
  const std::vector<unsigned char> host_flags = flags.copy_to_host();
  std::vector<RangeLeafPair> selected;
  for (int query = 0; query < qnum; ++query) {
    for (int node = 0; node < node_count; ++node) {
      if (host_flags[static_cast<std::size_t>(query) * node_count + node] == 0U ||
          snapshot.empty[node] != 0 || snapshot.nodes[node].is_leaf != 1) {
        continue;
      }
      selected.push_back(RangeLeafPair{query, node});
    }
  }
  std::sort(selected.begin(), selected.end(), [](const RangeLeafPair& left, const RangeLeafPair& right) {
    return left.query_id != right.query_id ? left.query_id < right.query_id : left.leaf_id < right.leaf_id;
  });
  selected.erase(std::unique(selected.begin(), selected.end(), [](const RangeLeafPair& left, const RangeLeafPair& right) {
    return left.query_id == right.query_id && left.leaf_id == right.leaf_id;
  }), selected.end());

  TraversalReceipt output;
  // This is a branch-aligned vector predicate mirror, not an invocation of
  // legacy searchIndexRnnV2 (which accepts only ID queries).  The selected
  // leaves are the receipt boundary; no leaf payload is read before this flag.
  output.raw_receipt_leaf_pair_count = static_cast<int>(selected.size());
  for (const RangeLeafPair& pair : selected) {
    if (pair.query_id == 0) output.visited_leaf_ids.push_back(pair.leaf_id);
  }
  std::sort(output.visited_leaf_ids.begin(), output.visited_leaf_ids.end());
  output.visited_leaf_ids.erase(
      std::unique(output.visited_leaf_ids.begin(), output.visited_leaf_ids.end()),
      output.visited_leaf_ids.end());
  output.unique_receipt_leaf_count = static_cast<int>(output.visited_leaf_ids.size());
  output.receipt_before_leaf_materialization = true;
  output.range_predicate_mirror_branch_aligned = true;
  output.id_list_capacity = runtime.id_list_capacity;
  output.receipt_leaf_spans =
      receipt_leaf_spans(runtime, snapshot, output.visited_leaf_ids);
  output.base_receipt_rows =
      materialize_receipt_leaf_rows(runtime, snapshot, output.visited_leaf_ids);
  std::vector<LocalRow> receipt_locals;
  receipt_locals.reserve(output.base_receipt_rows.size());
  for (const ReceiptBaseRow& row : output.base_receipt_rows) receipt_locals.push_back(row.local_row);
  output.base_results = scan_local_rows_exact(runtime, query_device, receipt_locals);
  output.base_results.erase(
      std::remove_if(output.base_results.begin(), output.base_results.end(),
                     [radius_sq](const StableDistance& row) { return row.distance_sq > radius_sq; }),
      output.base_results.end());
  output.tree_version = tree_version;
  output.tree_payload_sha256 = snapshot.tree_payload_sha256;
  output.native_tree_bytes_sha256 = snapshot.native_tree_bytes_sha256;
  output.immutable_base_stable_ids_sha256 = snapshot.immutable_base_stable_ids_sha256;
  output.immutable_base_payload_sha256 = snapshot.immutable_base_payload_sha256;
  output.local_to_stable_sha256_value = snapshot.local_to_stable_sha256_value;
  output.base_path = "gts_vector_range_branch_aligned_predicate_mirror_receipt";
  return output;
}

// Correctness-only range fallback.  The v10 branch-aligned vector predicate
// mirror failed its bootstrap range oracle; do not try to repair that failure
// by guessing at radius/ULP/ancestor predicates.  Instead, scan all local rows
// of the current immutable generation exactly.  This intentionally produces no
// GTS leaf receipt and must not be used for range traversal/performance claims.
inline TraversalReceipt run_exact_full_immutable_base_range_fallback(
    BaseTreeRuntime& runtime, const float* query_device, int qnum, DistanceSq radius_sq,
    std::uint64_t tree_version, const FrozenTreeSnapshot& snapshot) {
  if (!G3_RANGE_SAFE_FULL_IMMUTABLE_BASE_FALLBACK_IMPLEMENTED || !runtime.ready() ||
      query_device == nullptr || qnum != 1 || !snapshot.initialized()) {
    fail("G3 range-safe fallback requires one valid frozen GTS runtime");
  }
  if (runtime.base_count != snapshot.runtime_base_count ||
      runtime.base_count != static_cast<int>(snapshot.immutable_base_stable_ids.size()) ||
      static_cast<int>(runtime.local_to_stable.size()) != runtime.base_count ||
      runtime.stable_to_local.size() != snapshot.runtime_stable_to_local_size ||
      runtime.id_list_capacity != snapshot.runtime_id_list_capacity ||
      local_to_stable_sha256(runtime.local_to_stable) != snapshot.local_to_stable_sha256_value ||
      stable_set_sha256(runtime.local_to_stable) != snapshot.immutable_base_stable_ids_sha256) {
    fail("range-safe fallback immutable-base mapping does not match frozen generation");
  }
  for (LocalRow local = 0; local < runtime.base_count; ++local) {
    const StableId stable = runtime.local_to_stable[static_cast<std::size_t>(local)];
    if (stable < 0 || stable >= static_cast<int>(runtime.stable_to_local.size()) ||
        runtime.stable_to_local[static_cast<std::size_t>(stable)] != local) {
      fail("range-safe fallback immutable-base mapping is not a current bijection");
    }
  }

  std::vector<LocalRow> all_locals;
  all_locals.reserve(static_cast<std::size_t>(runtime.base_count));
  for (LocalRow local = 0; local < runtime.base_count; ++local) all_locals.push_back(local);

  TraversalReceipt output;
  output.exact_full_immutable_base_range_fallback = true;
  output.full_immutable_base_candidate_count = runtime.base_count;
  output.id_list_capacity = runtime.id_list_capacity;
  output.base_results = scan_local_rows_exact(runtime, query_device, all_locals);
  output.base_results.erase(
      std::remove_if(output.base_results.begin(), output.base_results.end(),
                     [radius_sq](const StableDistance& row) { return row.distance_sq > radius_sq; }),
      output.base_results.end());
  output.tree_version = tree_version;
  output.tree_payload_sha256 = snapshot.tree_payload_sha256;
  output.native_tree_bytes_sha256 = snapshot.native_tree_bytes_sha256;
  output.immutable_base_stable_ids_sha256 = snapshot.immutable_base_stable_ids_sha256;
  output.immutable_base_payload_sha256 = snapshot.immutable_base_payload_sha256;
  output.local_to_stable_sha256_value = snapshot.local_to_stable_sha256_value;
  output.base_path = "exact_full_immutable_base_range_fallback_no_gts_receipt";
  return output;
}

struct RebuildReceipt {
  std::uint64_t tree_version = 0;
  // These topology facts are copied from the frozen candidate snapshot before
  // publish.  They report what indexConstru actually built; a runner must not
  // substitute source-mirror height/shape claims for these receipt values.
  int tree_height = 0;
  // max_node_num[0] is heap capacity (including empty slots), not a count of
  // populated nodes.  Export both terms explicitly to prevent a runner/paper
  // from overstating allocated capacity as actual tree occupancy.
  int node_capacity = 0;
  int nonempty_node_count = 0;
  int fanout = 0;
  // Kept for trace compatibility: after a successful rebuild this equals the
  // new immutable-base set, never the mutable pre-rebuild active assertion.
  std::string live_ids_sha256;
  std::string immutable_base_stable_ids_sha256;
  std::string immutable_base_payload_sha256;
  std::string native_tree_bytes_sha256;
  std::string local_to_stable_sha256_value;
  std::string logical_leaf_stable_ids_sha256;
  std::string tree_payload_sha256;
  int base_count = 0;
  int sidecar_live = -1;
  int delta_live = -1;
  bool compact_mapping_bijection_ok = false;
  bool raw_leaf_rows_cover_compact_range = false;
  // The legacy max_dis_d global prevents a concurrent old/candidate tree.
  // Rebuild is intentionally destructive; candidate failure latches FailedStop.
  bool destructive_fail_stop_contract = true;
  bool old_generation_retired_before_candidate_build = true;
  bool dynamic_tiers_replaced_after_generation_publish = false;
  bool post_rebuild_knn_oracle_required = true;
  bool post_rebuild_range_oracle_required = true;
};

struct QueryExport {
  std::string kind;
  int query_id = -1;
  int query_dimension = 0;
  // KNN carries its requested k.  Range has requested_k == 0 and carries
  // radius_sq instead; both are included in the post-rebuild binding.
  int requested_k = 0;
  DistanceSq radius_sq = 0;
  std::uint64_t tree_version = 0;
  std::uint64_t state_epoch = 0;
  std::string tree_payload_sha256;
  std::string native_tree_bytes_sha256;
  std::string immutable_base_stable_ids_sha256;
  std::string immutable_base_payload_sha256;
  std::string local_to_stable_sha256_value;
  std::string query_vector_sha256;
  std::string active_stable_ids_sha256;
  std::string base_path;
  bool receipt_before_leaf_materialization = false;
  bool range_receipt_before_leaf_materialization = false;
  bool range_predicate_mirror_branch_aligned = false;
  // Typed fallback declaration; unlike base_path text it is included in the
  // receipt hash and must be true only for the range-safe full-base scan.
  bool exact_full_immutable_base_range_fallback = false;
  int full_immutable_base_candidate_count = 0;
  int id_list_capacity = 0;
  bool native_final_res_ids_cross_checked = false;
  // Range sidecars are independently exact-filtered by d^2 <= r^2.  This
  // marker is not a claim that the KNN direct certificate proves range.
  bool direct_sidecars_exact_range_filtered = false;
  // This describes candidate/result materialization.  KNN has no full-base
  // scan; the explicitly marked range fallback does.  The separate
  // control-plane state partition audit may iterate stable-ID metadata.
  bool production_full_live_scan = false;
  bool full_frozen_base_snapshot_integrity_audit_on_operation = false;
  // The SafeC1 partition checker intentionally audits stable-ID metadata on
  // each controlled operation; it never copies/scans the frozen vector payload.
  bool control_plane_partition_metadata_audit_on_operation = true;
  bool runner_single_image_admission_required = true;
  bool post_rebuild_oracle_required = false;
  // This marks only a private verification ticket issuance.  It does not
  // attest oracle provenance, native correctness, or a successful comparison.
  bool post_rebuild_verification_ticket_issued = false;
  std::uint64_t post_rebuild_issuance_nonce = 0;
  std::string post_rebuild_receipt_sha256;
  std::string post_rebuild_result_sha256;
  std::string post_rebuild_binding_sha256;
  int raw_receipt_leaf_pair_count = 0;
  int unique_receipt_leaf_count = 0;
  std::vector<int> gts_visited_leaf_ids;
  std::vector<ReceiptLeafSpan> receipt_leaf_spans;
  std::vector<ReceiptBaseRow> base_receipt_rows;
  std::vector<LocalRow> native_final_res_ids;
  std::vector<std::pair<int, StableId>> sidecar_source_leaf_pairs;
  std::vector<StableId> sidecar_candidate_ids;
  std::vector<StableId> delta_candidate_ids;
  std::vector<StableDistance> base_results;
  std::vector<StableDistance> sidecar_results;
  std::vector<StableDistance> delta_results;
  std::vector<StableDistance> results;
};

struct QueryInputSnapshot {
  int dimension = 0;
  std::vector<std::int16_t> values;
  std::string sha256;
};

inline std::string query_input_snapshot_sha256(const std::vector<std::int16_t>& values,
                                               int dimension) {
  if (dimension <= 0 || static_cast<int>(values.size()) != dimension) {
    fail("query snapshot has invalid immutable vector shape");
  }
  std::ostringstream bytes;
  bytes << "safe-c1-g3-post-rebuild-query-vector-v1\n" << dimension << '\n';
  for (std::int16_t value : values) bytes << static_cast<int>(value) << '\n';
  return sha256_text(bytes.str());
}

inline std::string query_receipt_sha256(const QueryExport& output) {
  std::ostringstream bytes;
  bytes << "safe-c1-g3-post-rebuild-receipt-v2\n"
        << output.kind << '\n' << output.base_path << '\n'
        << output.receipt_before_leaf_materialization << '\n'
        << output.range_receipt_before_leaf_materialization << '\n'
        << output.range_predicate_mirror_branch_aligned << '\n'
        << output.exact_full_immutable_base_range_fallback << '\n'
        << output.full_immutable_base_candidate_count << '\n'
        << output.id_list_capacity << '\n'
        << output.production_full_live_scan << '\n'
        << output.full_frozen_base_snapshot_integrity_audit_on_operation << '\n'
        << output.control_plane_partition_metadata_audit_on_operation << '\n'
        << output.runner_single_image_admission_required << '\n';
  for (int leaf : output.gts_visited_leaf_ids) bytes << "leaf:" << leaf << '\n';
  for (const ReceiptLeafSpan& span : output.receipt_leaf_spans) {
    bytes << "span:" << span.leaf_id << ':' << span.id_list_lid << ':' << span.size << '\n';
  }
  for (const ReceiptBaseRow& row : output.base_receipt_rows) {
    bytes << "row:" << row.leaf_id << ':' << row.id_list_slot << ':'
          << row.local_row << ':' << row.stable_id << '\n';
  }
  for (LocalRow row : output.native_final_res_ids) bytes << "native:" << row << '\n';
  for (const auto& pair : output.sidecar_source_leaf_pairs) {
    bytes << "sidecar-pair:" << pair.first << ':' << pair.second << '\n';
  }
  for (StableId stable : output.sidecar_candidate_ids) bytes << "sidecar:" << stable << '\n';
  for (StableId stable : output.delta_candidate_ids) bytes << "delta:" << stable << '\n';
  return sha256_text(bytes.str());
}

inline std::string post_rebuild_binding_sha256(const QueryExport& output,
                                               const QueryInputSnapshot& snapshot,
                                               const std::vector<StableId>& active_ids,
                                               std::uint64_t state_epoch,
                                               std::uint64_t issuance_nonce,
                                               const std::string& receipt_sha256,
                                               const std::string& result_sha256) {
  const std::string active_sha256 = stable_set_sha256(active_ids);
  if (output.query_dimension != snapshot.dimension ||
      output.query_vector_sha256 != snapshot.sha256 ||
      output.active_stable_ids_sha256 != active_sha256 ||
      output.state_epoch != state_epoch) {
    fail("post-rebuild binding input does not match the immutable query/state snapshot");
  }
  std::ostringstream bytes;
  bytes << "safe-c1-g3-post-rebuild-binding-v2\n"
        << "kind=" << output.kind << '\n'
        << "query_id=" << output.query_id << '\n'
        << "dimension=" << output.query_dimension << '\n'
        << "query_vector=" << output.query_vector_sha256 << '\n'
        << "requested_k=" << output.requested_k << '\n'
        << "radius_sq=" << static_cast<unsigned long long>(output.radius_sq) << '\n'
        << "tree_version=" << output.tree_version << '\n'
        << "tree_payload=" << output.tree_payload_sha256 << '\n'
        << "state_epoch=" << state_epoch << '\n'
        << "active_ids=" << active_sha256 << '\n'
        << "issuance_nonce=" << issuance_nonce << '\n'
        << "receipt=" << receipt_sha256 << '\n'
        << "results=" << result_sha256 << '\n';
  return sha256_text(bytes.str());
}

enum class EngineMode {
  kUninitialized,
  kReady,
  kAwaitingPostRebuildOracle,
  kRebuildDestructive,
  kFailedStop,
  kDestroyed,
};

inline const char* engine_mode_name(EngineMode mode) {
  switch (mode) {
    case EngineMode::kUninitialized: return "uninitialized";
    case EngineMode::kReady: return "ready";
    case EngineMode::kAwaitingPostRebuildOracle: return "awaiting_post_rebuild_oracle";
    case EngineMode::kRebuildDestructive: return "rebuild_destructive";
    case EngineMode::kFailedStop: return "failed_stop";
    case EngineMode::kDestroyed: return "destroyed";
  }
  return "unknown";
}

class NativeSafeC1Matrix;
class PostRebuildOracleGate;

// This secret has no public constructor.  A ticket carries pointer identity to
// one gate-issued secret; public QueryExport diagnostic fields are never an
// authorization channel.
class PostRebuildTicketSecret final {
 private:
  PostRebuildTicketSecret(std::uint64_t owner_nonce, std::uint64_t serial,
                          std::string kind, std::string binding_sha256)
      : owner_nonce_(owner_nonce), serial_(serial), kind_(std::move(kind)),
        binding_sha256_(std::move(binding_sha256)) {}

  std::uint64_t owner_nonce_ = 0;
  std::uint64_t serial_ = 0;
  std::string kind_;
  std::string binding_sha256_;
  friend class PostRebuildOracleGate;
};

class PostRebuildVerificationTicket final {
 public:
  PostRebuildVerificationTicket(const PostRebuildVerificationTicket&) = delete;
  PostRebuildVerificationTicket& operator=(const PostRebuildVerificationTicket&) = delete;
  PostRebuildVerificationTicket(PostRebuildVerificationTicket&&) noexcept = default;
  PostRebuildVerificationTicket& operator=(PostRebuildVerificationTicket&&) noexcept = default;
  ~PostRebuildVerificationTicket() = default;

 private:
  explicit PostRebuildVerificationTicket(std::shared_ptr<const PostRebuildTicketSecret> secret)
      : secret_(std::move(secret)) {}
  void consume() noexcept { secret_.reset(); }

  std::shared_ptr<const PostRebuildTicketSecret> secret_;
  friend class NativeSafeC1Matrix;
  friend class PostRebuildOracleGate;
};

// A post-rebuild query is move-only precisely because its verification ticket
// is move-only.  A caller may inspect/export export_data, but the gate compares
// against its own private copy, not a caller-supplied QueryExport.
struct IssuedQuery final {
  QueryExport export_data;
  std::optional<PostRebuildVerificationTicket> post_rebuild_ticket;

  IssuedQuery(QueryExport export_value,
              std::optional<PostRebuildVerificationTicket> ticket_value)
      : export_data(std::move(export_value)),
        post_rebuild_ticket(std::move(ticket_value)) {}
  IssuedQuery(const IssuedQuery&) = delete;
  IssuedQuery& operator=(const IssuedQuery&) = delete;
  IssuedQuery(IssuedQuery&&) noexcept = default;
  IssuedQuery& operator=(IssuedQuery&&) noexcept = default;
};

class PostRebuildOracleGate {
 public:
  void reset(std::uint64_t tree_version, std::uint64_t state_epoch,
             std::uint64_t owner_nonce) {
    if (tree_version == 0 || owner_nonce == 0) {
      fail("post-rebuild gate requires nonzero generation and owner nonce");
    }
    tree_version_ = tree_version;
    state_epoch_ = state_epoch;
    owner_nonce_ = owner_nonce;
    next_serial_ = 0;
    knn_ = IssuedRecord{};
    range_ = IssuedRecord{};
    knn_.needs = true;
    range_.needs = true;
  }

  [[nodiscard]] bool query_may_issue(const std::string& kind) const {
    const IssuedRecord& record = record_for(kind);
    return record.needs && !record.issued && !record.consumed;
  }

  [[nodiscard]] bool requires(const std::string& kind) const {
    return record_for(kind).needs;
  }

  // Called only after native traversal, exact sidecar/delta merge, canonical
  // result shaping, and frozen/partition postchecks succeeded.  It makes a
  // private record before returning the opaque ticket.
  PostRebuildVerificationTicket issue_successful_query(
      QueryExport* actual, const QueryInputSnapshot& snapshot,
      const std::vector<StableId>& active_ids, std::uint64_t state_epoch) {
    if (!actual) fail("post-rebuild gate cannot issue a null query export");
    IssuedRecord& current = record_for(actual->kind);
    if (!current.needs || current.issued || current.consumed) {
      fail("post-rebuild query kind already issued or verified");
    }
    if (actual->tree_version != tree_version_ || state_epoch != state_epoch_) {
      fail("post-rebuild issuance crosses a frozen generation/state epoch");
    }
    if (snapshot.dimension <= 0 || snapshot.values.empty() ||
        snapshot.sha256 != query_input_snapshot_sha256(snapshot.values, snapshot.dimension)) {
      fail("post-rebuild issuance received a noncanonical query snapshot");
    }
    if (actual->kind == "knn") {
      if (actual->requested_k <= 0 || actual->radius_sq != 0) {
        fail("post-rebuild KNN binding has invalid k/radius");
      }
    } else if (actual->kind == "range") {
      if (actual->requested_k != 0) fail("post-rebuild range binding carries a KNN k");
    } else {
      fail("unknown query kind in post-rebuild issuance");
    }
    if (actual->tree_payload_sha256.empty() || actual->query_id < 0) {
      fail("post-rebuild issuance lacks immutable tree/query identity");
    }
    require_canonical_stable_distances(actual->results,
                                      "internally issued post-rebuild result");
    std::set<StableId> active_set(active_ids.begin(), active_ids.end());
    if (active_set.size() != active_ids.size()) {
      fail("post-rebuild active set is not canonical/unique");
    }
    for (const StableDistance& row : actual->results) {
      if (active_set.find(row.stable_id) == active_set.end()) {
        fail("internally issued post-rebuild result contains inactive stable ID");
      }
      if (actual->kind == "range" && row.distance_sq > actual->radius_sq) {
        fail("internally issued range result exceeds requested radius");
      }
    }

    // Stage all fallible copies/hashes before changing the issued record.
    QueryExport staged = *actual;
    staged.query_dimension = snapshot.dimension;
    staged.query_vector_sha256 = snapshot.sha256;
    staged.active_stable_ids_sha256 = stable_set_sha256(active_ids);
    staged.state_epoch = state_epoch;
    staged.post_rebuild_issuance_nonce = ++next_serial_;
    staged.post_rebuild_receipt_sha256 = query_receipt_sha256(staged);
    staged.post_rebuild_result_sha256 =
        stable_distance_vector_sha256(staged.results, "safe-c1-g3-post-rebuild-result-v1");
    staged.post_rebuild_binding_sha256 = post_rebuild_binding_sha256(
        staged, snapshot, active_ids, state_epoch, staged.post_rebuild_issuance_nonce,
        staged.post_rebuild_receipt_sha256, staged.post_rebuild_result_sha256);
    staged.post_rebuild_verification_ticket_issued = true;
    const std::shared_ptr<const PostRebuildTicketSecret> secret(
        new PostRebuildTicketSecret(owner_nonce_, staged.post_rebuild_issuance_nonce,
                                    staged.kind, staged.post_rebuild_binding_sha256));
    IssuedRecord next;
    next.needs = true;
    next.issued = true;
    next.kind = staged.kind;
    next.tree_version = staged.tree_version;
    next.tree_payload_sha256 = staged.tree_payload_sha256;
    next.query_id = staged.query_id;
    next.query_dimension = staged.query_dimension;
    next.requested_k = staged.requested_k;
    next.radius_sq = staged.radius_sq;
    next.query_vector_sha256 = staged.query_vector_sha256;
    next.active_stable_ids = active_ids;
    next.active_stable_ids_sha256 = staged.active_stable_ids_sha256;
    next.state_epoch = staged.state_epoch;
    next.issuance_nonce = staged.post_rebuild_issuance_nonce;
    next.receipt_sha256 = staged.post_rebuild_receipt_sha256;
    next.result_sha256 = staged.post_rebuild_result_sha256;
    next.binding_sha256 = staged.post_rebuild_binding_sha256;
    next.engine_results = staged.results;
    next.ticket_secret = secret;

    // QueryExport is only a diagnostic export.  The record assigned afterward
    // retains a distinct private snapshot of its result/binding fields.
    *actual = std::move(staged);
    current = std::move(next);
    return PostRebuildVerificationTicket(secret);
  }

  // The external runner supplies only an independently computed expectation.
  // The actual result, receipt, vector binding, and ticket authority are all
  // private gate state.  A successful call consumes the exact ticket once.
  void verify_issued_query(PostRebuildVerificationTicket&& ticket,
                           const std::vector<StableDistance>& independent_expected) {
    if (!ticket.secret_) fail("post-rebuild verification ticket is empty or already consumed");
    const PostRebuildTicketSecret& secret = *ticket.secret_;
    if (secret.owner_nonce_ != owner_nonce_) {
      fail("post-rebuild verification ticket belongs to another matrix/generation");
    }
    IssuedRecord& current = record_for(secret.kind_);
    if (!current.needs || !current.issued || current.consumed ||
        current.ticket_secret.get() != ticket.secret_.get() ||
        current.issuance_nonce != secret.serial_ || current.kind != secret.kind_ ||
        current.binding_sha256 != secret.binding_sha256_) {
      fail("post-rebuild verification ticket instance/serial/kind/binding mismatch");
    }
    validate_independent_expected(current, independent_expected);
    if (current.engine_results != independent_expected) {
      fail("internally issued post-rebuild stable-ID result differs from independent oracle");
    }
    current.consumed = true;
    current.needs = false;
    ticket.consume();
  }

  [[nodiscard]] bool complete() const { return !knn_.needs && !range_.needs; }
  [[nodiscard]] bool needs_knn() const { return knn_.needs; }
  [[nodiscard]] bool needs_range() const { return range_.needs; }

 private:
  struct IssuedRecord {
    bool needs = false;
    bool issued = false;
    bool consumed = false;
    std::string kind;
    std::uint64_t tree_version = 0;
    std::string tree_payload_sha256;
    int query_id = -1;
    int query_dimension = 0;
    int requested_k = 0;
    DistanceSq radius_sq = 0;
    std::string query_vector_sha256;
    std::vector<StableId> active_stable_ids;
    std::string active_stable_ids_sha256;
    std::uint64_t state_epoch = 0;
    std::uint64_t issuance_nonce = 0;
    std::string receipt_sha256;
    std::string result_sha256;
    std::string binding_sha256;
    std::vector<StableDistance> engine_results;
    std::shared_ptr<const PostRebuildTicketSecret> ticket_secret;
  };

  [[nodiscard]] IssuedRecord& record_for(const std::string& kind) {
    if (kind == "knn") return knn_;
    if (kind == "range") return range_;
    fail("unknown query kind in post-rebuild gate");
  }
  [[nodiscard]] const IssuedRecord& record_for(const std::string& kind) const {
    if (kind == "knn") return knn_;
    if (kind == "range") return range_;
    fail("unknown query kind in post-rebuild gate");
  }

  static void validate_independent_expected(
      const IssuedRecord& record,
      const std::vector<StableDistance>& independent_expected) {
    require_canonical_stable_distances(independent_expected,
                                      "independent post-rebuild oracle result");
    if (record.tree_version == 0 || record.tree_payload_sha256.empty() ||
        record.query_vector_sha256.empty() || record.active_stable_ids_sha256.empty() ||
        record.receipt_sha256.empty() || record.result_sha256.empty() ||
        record.binding_sha256.empty()) {
      fail("issued post-rebuild record lacks a mandatory immutable binding");
    }
    if (record.active_stable_ids_sha256 != stable_set_sha256(record.active_stable_ids)) {
      fail("issued post-rebuild record active-set digest is inconsistent");
    }
    std::set<StableId> active(record.active_stable_ids.begin(), record.active_stable_ids.end());
    if (active.size() != record.active_stable_ids.size()) {
      fail("issued post-rebuild record active IDs are not unique");
    }
    for (const StableDistance& row : independent_expected) {
      if (active.find(row.stable_id) == active.end()) {
        fail("independent post-rebuild oracle returned an inactive stable ID");
      }
      if (record.kind == "range" && row.distance_sq > record.radius_sq) {
        fail("independent range oracle returned a row outside requested radius");
      }
    }
    if (record.kind == "knn") {
      if (record.requested_k <= 0 || record.query_dimension <= 0) {
        fail("issued KNN post-rebuild record has invalid query contract");
      }
      const std::size_t expected_count = std::min(
          static_cast<std::size_t>(record.requested_k), record.active_stable_ids.size());
      if (independent_expected.size() != expected_count) {
        fail("independent KNN oracle cardinality is not min(k, live)");
      }
    } else if (record.kind == "range") {
      if (record.requested_k != 0 || record.query_dimension <= 0) {
        fail("issued range post-rebuild record has invalid query contract");
      }
    } else {
      fail("issued post-rebuild record has unknown query kind");
    }
  }

  std::uint64_t tree_version_ = 0;
  std::uint64_t state_epoch_ = 0;
  std::uint64_t owner_nonce_ = 0;
  std::uint64_t next_serial_ = 0;
  IssuedRecord knn_;
  IssuedRecord range_;
};

inline std::vector<StableDistance> merge_partition_results(
    const std::vector<StableDistance>& base,
    const std::vector<StableDistance>& sidecar,
    const std::vector<StableDistance>& delta) {
  std::vector<StableDistance> merged;
  merged.reserve(base.size() + sidecar.size() + delta.size());
  merged.insert(merged.end(), base.begin(), base.end());
  merged.insert(merged.end(), sidecar.begin(), sidecar.end());
  merged.insert(merged.end(), delta.begin(), delta.end());
  sort_and_require_unique(&merged, "production receipt candidate partitions");
  return merged;
}

class NativeSafeC1Matrix {
 public:
  NativeSafeC1Matrix(int sidecar_leaf_capacity, int requested_k)
      : sidecar_leaf_capacity_(sidecar_leaf_capacity), requested_k_(requested_k) {
    if (sidecar_leaf_capacity_ <= 0 || requested_k_ <= 0) {
      fail("invalid native Safe-C1 matrix configuration");
    }
  }

  ~NativeSafeC1Matrix() {
    std::lock_guard<std::mutex> api_guard(api_mutex_);
    release_runtime(&runtime_);
    mode_ = EngineMode::kDestroyed;
  }
  NativeSafeC1Matrix(const NativeSafeC1Matrix&) = delete;
  NativeSafeC1Matrix& operator=(const NativeSafeC1Matrix&) = delete;

  void initialize_immutable_pool(int dimension, std::vector<std::int16_t> values,
                                 std::vector<int> stable_to_pool_row) {
    std::lock_guard<std::mutex> api_guard(api_mutex_);
    if (mode_ != EngineMode::kUninitialized || runtime_.ready()) {
      fail("cannot replace immutable pool after engine initialization or fail-stop");
    }
    pool_.initialize(dimension, std::move(values), std::move(stable_to_pool_row));
    state_.reset(pool_.stable_capacity(), sidecar_leaf_capacity_);
    try {
      configure_no_residual_pruning();
    } catch (...) {
      enter_failed_stop_noexcept();
      throw;
    }
  }

  // This is the sole public initialization-time base-list input.  Later
  // rebuilds always obtain their live IDs from state_ internally.
  RebuildReceipt build_initial_base(const std::vector<StableId>& base_ids) {
    std::lock_guard<std::mutex> api_guard(api_mutex_);
    if (pool_.dimension() == 0) fail("initialize immutable pool before building base");
    if (mode_ != EngineMode::kUninitialized || runtime_.ready()) {
      fail("initial base requires an uninitialized engine");
    }
    require_device_residual_mode_zero_or_fail_stop();
    return rebuild_from_validated_ids_private(base_ids);
  }

  // Diagnostic-only observer for the first immutable generation.  This is not
  // an admission API: direct remains compile-time disabled and the return value
  // cannot mutate state or stand in for native traversal membership.
  [[nodiscard]] FrozenCertificateDiagnostic inspect_initial_frozen_certificate(
      StableId stable) {
    std::lock_guard<std::mutex> api_guard(api_mutex_);
    if (G3_DIRECT_SIDECAR_RUNTIME_GUARD_OPEN || mode_ != EngineMode::kAwaitingPostRebuildOracle ||
        tree_version_ != 1 || state_epoch_ != 1 || !runtime_.ready() ||
        !frozen_.initialized() || state_.sidecar_live() != 0 || state_.delta_live() != 0) {
      fail("initial frozen certificate diagnostic requires untouched guard-closed generation one");
    }
    try {
      assert_current_generation_and_partition_or_fail_stop();
      FrozenCertificateDiagnostic diagnostic =
          frozen_.inspect_certificate(pool_, runtime_, stable);
      // Recheck exactly the same generation/partition after the host-only
      // observer; any drift is fail-stop rather than a published diagnostic.
      assert_current_generation_and_partition_or_fail_stop();
      if (mode_ != EngineMode::kAwaitingPostRebuildOracle || tree_version_ != 1 ||
          state_epoch_ != 1 || state_.sidecar_live() != 0 || state_.delta_live() != 0) {
        enter_failed_stop_noexcept();
        fail("initial frozen certificate diagnostic observed mutable-state drift");
      }
      return diagnostic;
    } catch (...) {
      enter_failed_stop_noexcept();
      throw;
    }
  }

  Placement insert(StableId stable) {
    std::lock_guard<std::mutex> api_guard(api_mutex_);
    require_ready_for_mutation();
    assert_current_generation_and_partition_or_fail_stop();
    SafeC1State draft = state_;
    try {
      Placement placed = draft.insert(stable, frozen_, runtime_, pool_, tree_version_,
                                      G3_DIRECT_SIDECAR_RUNTIME_GUARD_OPEN);
      assert_draft_generation_and_partition_or_fail_stop(draft);
      state_.swap_noexcept(draft);
      ++state_epoch_;
      return placed;
    } catch (...) {
      // Public state_ was never written; this is an all-or-nothing mutation.
      throw;
    }
  }

  Placement erase_mutable(StableId stable) {
    std::lock_guard<std::mutex> api_guard(api_mutex_);
    require_ready_for_mutation();
    assert_current_generation_and_partition_or_fail_stop();
    SafeC1State draft = state_;
    try {
      Placement prior = draft.erase_mutable(stable);
      assert_draft_generation_and_partition_or_fail_stop(draft);
      state_.swap_noexcept(draft);
      ++state_epoch_;
      return prior;
    } catch (...) {
      // Public state_ was never written; this is an all-or-nothing mutation.
      throw;
    }
  }

  // Only the current partition state may request a ready-state public rebuild.
  // No public raw live-ID vector entry point exists; the private helper below
  // is reachable only from build_initial_base() and this state snapshot.
  RebuildReceipt rebuild_from_current_live() {
    std::lock_guard<std::mutex> api_guard(api_mutex_);
    require_ready_for_mutation();
    return rebuild_from_validated_ids_private(state_.live_ids());
  }

  IssuedQuery query_knn(int query_id, const std::int16_t* query_values, int k = -1) {
    std::lock_guard<std::mutex> api_guard(api_mutex_);
    if (k < 0) k = requested_k_;
    if (k <= 0) fail("KNN requires positive k");
    require_ready_for_query("knn");
    assert_current_generation_and_partition_or_fail_stop();
    const bool oracle_required = mode_ == EngineMode::kAwaitingPostRebuildOracle &&
                                 post_rebuild_gate_.query_may_issue("knn");
    const std::vector<StableId> active_snapshot = state_.live_ids();
    const QueryInputSnapshot input_snapshot = capture_query_input(query_values);
    try {
      DeviceBuffer<float> query_device;
      upload_query_snapshot(input_snapshot, &query_device);
      TraversalReceipt base = run_gts_base_topk_with_receipt(runtime_, query_device.data(), 1, k,
                                                              tree_version_, frozen_);
      QueryExport output = merge_query_partitions("knn", query_id, 0, base,
                                                   query_device.data(), false);
      output.query_dimension = input_snapshot.dimension;
      output.requested_k = k;
      output.query_vector_sha256 = input_snapshot.sha256;
      output.active_stable_ids_sha256 = stable_set_sha256(active_snapshot);
      output.state_epoch = state_epoch_;
      output.post_rebuild_oracle_required = oracle_required;
      if (static_cast<int>(output.results.size()) > k) {
        output.results.resize(static_cast<std::size_t>(k));
      }
      require_canonical_stable_distances(output.results, "KNN query result");
      assert_current_generation_and_partition_or_fail_stop();
      if (oracle_required) {
        PostRebuildVerificationTicket ticket = post_rebuild_gate_.issue_successful_query(
            &output, input_snapshot, active_snapshot, state_epoch_);
        return IssuedQuery{std::move(output),
                           std::optional<PostRebuildVerificationTicket>(std::move(ticket))};
      }
      return IssuedQuery{std::move(output), std::nullopt};
    } catch (...) {
      // Any traversal/receipt/binding failure after a native query attempt is
      // terminal.  No partial receipt can be reused or verified.
      enter_failed_stop_noexcept();
      throw;
    }
  }

  IssuedQuery query_range(int query_id, const std::int16_t* query_values, DistanceSq radius_sq) {
    std::lock_guard<std::mutex> api_guard(api_mutex_);
    require_ready_for_query("range");
    assert_current_generation_and_partition_or_fail_stop();
    const bool oracle_required = mode_ == EngineMode::kAwaitingPostRebuildOracle &&
                                 post_rebuild_gate_.query_may_issue("range");
    const std::vector<StableId> active_snapshot = state_.live_ids();
    const QueryInputSnapshot input_snapshot = capture_query_input(query_values);
    try {
      // The fallback has no leaf-routing basis for direct sidecars.  Refuse
      // rather than silently omitting them if a future edit opens that path.
      if (G3_DIRECT_SIDECAR_RUNTIME_GUARD_OPEN || state_.sidecar_live() != 0) {
        fail("range-safe fallback requires a closed direct-sidecar guard and no live sidecars");
      }
      DeviceBuffer<float> query_device;
      upload_query_snapshot(input_snapshot, &query_device);
      TraversalReceipt base = run_exact_full_immutable_base_range_fallback(
          runtime_, query_device.data(), 1, radius_sq, tree_version_, frozen_);
      QueryExport output = merge_query_partitions("range", query_id, radius_sq, base,
                                                   query_device.data(), true);
      output.query_dimension = input_snapshot.dimension;
      output.requested_k = 0;
      output.query_vector_sha256 = input_snapshot.sha256;
      output.active_stable_ids_sha256 = stable_set_sha256(active_snapshot);
      output.state_epoch = state_epoch_;
      output.post_rebuild_oracle_required = oracle_required;
      require_canonical_stable_distances(output.results, "range query result");
      assert_current_generation_and_partition_or_fail_stop();
      if (oracle_required) {
        PostRebuildVerificationTicket ticket = post_rebuild_gate_.issue_successful_query(
            &output, input_snapshot, active_snapshot, state_epoch_);
        return IssuedQuery{std::move(output),
                           std::optional<PostRebuildVerificationTicket>(std::move(ticket))};
      }
      return IssuedQuery{std::move(output), std::nullopt};
    } catch (...) {
      // A range fallback failure is fail-stop.  This path is not represented as
      // archive-native equivalence, a native range receipt, or a performance proof.
      enter_failed_stop_noexcept();
      throw;
    }
  }

  // Required runner action.  The runner supplies its independent CPU oracle
  // result, but cannot supply/alter the actual query export: the opaque,
  // move-only ticket identifies a private issued record and is consumed once.
  void verify_first_post_rebuild_query(
      PostRebuildVerificationTicket&& ticket,
      const std::vector<StableDistance>& independent_expected) {
    std::lock_guard<std::mutex> api_guard(api_mutex_);
    try {
      require_device_residual_mode_zero_or_fail_stop();
      if (mode_ != EngineMode::kAwaitingPostRebuildOracle) {
        fail("post-rebuild oracle verification is not pending");
      }
      assert_current_generation_and_partition_or_fail_stop();
      post_rebuild_gate_.verify_issued_query(std::move(ticket), independent_expected);
      if (post_rebuild_gate_.complete()) mode_ = EngineMode::kReady;
    } catch (...) {
      enter_failed_stop_noexcept();
      throw;
    }
  }

  [[nodiscard]] std::uint64_t tree_version() const {
    std::lock_guard<std::mutex> api_guard(api_mutex_);
    return tree_version_;
  }
  [[nodiscard]] EngineMode mode() const {
    std::lock_guard<std::mutex> api_guard(api_mutex_);
    return mode_;
  }
  [[nodiscard]] bool post_rebuild_oracle_pending() const {
    std::lock_guard<std::mutex> api_guard(api_mutex_);
    return mode_ == EngineMode::kAwaitingPostRebuildOracle && !post_rebuild_gate_.complete();
  }

 private:
  RebuildReceipt rebuild_from_validated_ids_private(const std::vector<StableId>& requested_live_ids) {
    if (pool_.dimension() == 0) fail("immutable pool is not initialized");
    if (mode_ == EngineMode::kFailedStop || mode_ == EngineMode::kDestroyed ||
        mode_ == EngineMode::kRebuildDestructive) {
      fail("destructive rebuild fail-stop/state barrier rejects this operation");
    }
    if (mode_ == EngineMode::kAwaitingPostRebuildOracle) {
      fail("post-rebuild KNN and range independent oracle checks must complete before another rebuild");
    }
    if (mode_ != EngineMode::kUninitialized && mode_ != EngineMode::kReady) {
      fail("rebuild entered from an invalid engine state");
    }
    if (mode_ == EngineMode::kReady) {
      assert_current_generation_and_partition_or_fail_stop();
    }

    std::vector<StableId> live = requested_live_ids;
    std::sort(live.begin(), live.end());
    if (live.empty()) fail("fresh destructive rebuild refuses an empty GTS base");
    if (std::adjacent_find(live.begin(), live.end()) != live.end()) {
      fail("fresh destructive rebuild has duplicate live stable ID");
    }
    for (StableId stable : live) {
      if (stable < 0 || stable >= pool_.stable_capacity()) {
        fail("fresh destructive rebuild stable ID outside immutable map");
      }
    }

    // Host-only replacement state is fully prepared before the destructive
    // barrier.  No mutation of current state_ has occurred yet.
    SafeC1State replacement(pool_.stable_capacity(), sidecar_leaf_capacity_);
    replacement.initialize_base(live);

    // tree.cuh owns max_dis_d globally, so an old and candidate tree cannot
    // coexist.  This is intentionally a single-engine destructive fail-stop:
    // once old bytes are retired, any candidate failure latches FailedStop.
    mode_ = EngineMode::kRebuildDestructive;
    BaseTreeRuntime candidate;
    try {
      release_runtime(&runtime_);
      frozen_ = FrozenTreeSnapshot{};
      build_compact_runtime(live, &candidate);
      FrozenTreeSnapshot candidate_snapshot;
      candidate_snapshot.capture(candidate);
      if (candidate_snapshot.logical_leaf_row_count != static_cast<int>(live.size()) ||
          candidate_snapshot.immutable_base_stable_ids_sha256 != stable_set_sha256(live)) {
        fail("candidate immutable base/leaf payload does not equal requested live IDs");
      }
      const std::uint64_t next_version = tree_version_ + 1;
      const std::uint64_t candidate_state_epoch = state_epoch_ + 1;
      replacement.assert_partition(candidate_snapshot, next_version);
      PostRebuildOracleGate candidate_gate;
      candidate_gate.reset(next_version, candidate_state_epoch,
                           ++next_post_rebuild_gate_owner_nonce_);
      RebuildReceipt prepared;
      prepared.tree_version = next_version;
      prepared.tree_height = candidate_snapshot.tree_height;
      prepared.node_capacity = candidate_snapshot.node_count;
      prepared.nonempty_node_count = static_cast<int>(
          std::count(candidate_snapshot.empty.begin(), candidate_snapshot.empty.end(), 0));
      prepared.fanout = candidate_snapshot.fanout;
      if (prepared.tree_height <= 0 || prepared.node_capacity <= 0 ||
          prepared.nonempty_node_count <= 0 ||
          prepared.nonempty_node_count > prepared.node_capacity ||
          prepared.fanout != TREE_ORDER) {
        fail("candidate rebuild receipt has invalid actual GTS topology");
      }
      prepared.live_ids_sha256 = candidate_snapshot.immutable_base_stable_ids_sha256;
      prepared.immutable_base_stable_ids_sha256 = candidate_snapshot.immutable_base_stable_ids_sha256;
      prepared.immutable_base_payload_sha256 = candidate_snapshot.immutable_base_payload_sha256;
      prepared.native_tree_bytes_sha256 = candidate_snapshot.native_tree_bytes_sha256;
      prepared.local_to_stable_sha256_value = candidate_snapshot.local_to_stable_sha256_value;
      prepared.logical_leaf_stable_ids_sha256 = candidate_snapshot.logical_leaf_stable_ids_sha256;
      prepared.tree_payload_sha256 = candidate_snapshot.tree_payload_sha256;
      prepared.base_count = candidate.base_count;
      prepared.sidecar_live = replacement.sidecar_live();
      prepared.delta_live = replacement.delta_live();
      prepared.compact_mapping_bijection_ok = true;
      prepared.raw_leaf_rows_cover_compact_range = true;
      prepared.destructive_fail_stop_contract = true;
      prepared.old_generation_retired_before_candidate_build = true;
      prepared.dynamic_tiers_replaced_after_generation_publish =
          prepared.sidecar_live == 0 && prepared.delta_live == 0;
      if (!prepared.dynamic_tiers_replaced_after_generation_publish) {
        fail("replacement state retained dynamic tiers before generation publish");
      }

      // All fallible validation/allocation/string construction is above.  An
      // unexpected publish fault is caught below and latches FailedStop; the
      // retired generation is never revived.
      commit_generation_noexcept(&candidate, &candidate_snapshot, &replacement,
                                 next_version, &candidate_gate);
      return prepared;
    } catch (...) {
      release_runtime(&candidate);
      enter_failed_stop_noexcept();
      throw;
    }
  }

  static void take_runtime_noexcept(BaseTreeRuntime* destination, BaseTreeRuntime* source) noexcept {
    destination->data_info = source->data_info;
    destination->data_d = source->data_d;
    destination->data_s = source->data_s;
    destination->size_s = source->size_s;
    destination->id_list = source->id_list;
    destination->node_list = source->node_list;
    destination->max_node_num = source->max_node_num;
    destination->empty_list = source->empty_list;
    destination->owned_max_dis_d = source->owned_max_dis_d;
    destination->tree_height = source->tree_height;
    destination->base_count = source->base_count;
    destination->id_list_capacity = source->id_list_capacity;
    destination->local_to_stable.swap(source->local_to_stable);
    destination->stable_to_local.swap(source->stable_to_local);
    source->data_info = nullptr;
    source->data_d = nullptr;
    source->data_s = nullptr;
    source->size_s = nullptr;
    source->id_list = nullptr;
    source->node_list = nullptr;
    source->max_node_num = nullptr;
    source->empty_list = nullptr;
    source->owned_max_dis_d = nullptr;
    source->tree_height = 0;
    source->base_count = 0;
    source->id_list_capacity = 0;
  }

  void commit_generation_noexcept(BaseTreeRuntime* candidate,
                                  FrozenTreeSnapshot* candidate_snapshot,
                                  SafeC1State* replacement,
                                  std::uint64_t next_version,
                                  PostRebuildOracleGate* candidate_gate) noexcept {
    // candidate_gate is unissued at publish time; move it so no copying of a
    // prior private issued record can allocate inside this no-throw boundary.
    if (!candidate_gate) std::terminate();
    take_runtime_noexcept(&runtime_, candidate);
    using std::swap;
    swap(frozen_, *candidate_snapshot);
    state_.swap_noexcept(*replacement);
    tree_version_ = next_version;
    post_rebuild_gate_ = std::move(*candidate_gate);
    ++state_epoch_;
    mode_ = EngineMode::kAwaitingPostRebuildOracle;
  }

  static void release_runtime(BaseTreeRuntime* runtime) noexcept {
    if (!runtime) return;
    if (runtime->id_list) cudaFree(runtime->id_list);
    if (runtime->node_list) cudaFree(runtime->node_list);
    if (runtime->max_node_num) cudaFree(runtime->max_node_num);
    if (runtime->empty_list) cudaFree(runtime->empty_list);
    if (runtime->data_d) cudaFree(runtime->data_d);
    if (runtime->data_info) cudaFree(runtime->data_info);
    // Single-engine ownership: if indexConstru failed before ownership was
    // copied into candidate->owned_max_dis_d, max_dis_d still belongs to this
    // destructive generation and must be released rather than leaked.
    float* global_to_free = runtime->owned_max_dis_d ? runtime->owned_max_dis_d : max_dis_d;
    if (global_to_free && max_dis_d == global_to_free) cudaFree(max_dis_d);
    runtime->data_info = nullptr;
    runtime->data_d = nullptr;
    runtime->data_s = nullptr;
    runtime->size_s = nullptr;
    runtime->id_list = nullptr;
    runtime->node_list = nullptr;
    runtime->max_node_num = nullptr;
    runtime->empty_list = nullptr;
    runtime->owned_max_dis_d = nullptr;
    runtime->tree_height = 0;
    runtime->base_count = 0;
    runtime->id_list_capacity = 0;
    runtime->local_to_stable.clear();
    runtime->stable_to_local.clear();
    max_dis_d = nullptr;
    // The lease also owns this copied-header process-global receipt vector;
    // clear it at every generation retirement/fail-stop so no stale leaf
    // receipt can outlive its frozen runtime.
    safe_c1_last_visited_leaf_pairs.clear();
    split_list = nullptr;
    pid_list = nullptr;
    dis_list = nullptr;
    split_num = nullptr;
  }

  void enter_failed_stop_noexcept() noexcept {
    release_runtime(&runtime_);
    frozen_ = FrozenTreeSnapshot{};
    mode_ = EngineMode::kFailedStop;
  }

  // Identity and control-plane checks are correctness boundaries, not merely
  // diagnostics.  A detected drift latches FailedStop before the exception
  // escapes, so a later caller cannot resume the prior Ready state.
  void assert_current_generation_and_partition_or_fail_stop() {
    try {
      frozen_.assert_lightweight_generation_identity(runtime_);
      require_device_residual_mode_zero_or_fail_stop();
      state_.assert_partition(frozen_, tree_version_);
    } catch (...) {
      enter_failed_stop_noexcept();
      throw;
    }
  }

  void assert_draft_generation_and_partition_or_fail_stop(const SafeC1State& draft) {
    try {
      frozen_.assert_lightweight_generation_identity(runtime_);
      require_device_residual_mode_zero_or_fail_stop();
      draft.assert_partition(frozen_, tree_version_);
    } catch (...) {
      enter_failed_stop_noexcept();
      throw;
    }
  }

  void build_compact_runtime(const std::vector<StableId>& live, BaseTreeRuntime* candidate) {
    candidate->base_count = static_cast<int>(live.size());
    candidate->local_to_stable = live;  // deterministic compact physical-row order
    candidate->stable_to_local.assign(static_cast<std::size_t>(pool_.stable_capacity()), -1);
    for (LocalRow local = 0; local < candidate->base_count; ++local) {
      const StableId stable = candidate->local_to_stable[local];
      if (candidate->stable_to_local[stable] != -1) fail("nonbijective compact mapping before GTS build");
      candidate->stable_to_local[stable] = local;
    }
    G3_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&candidate->data_info), 3 * sizeof(int)));
    candidate->data_info[0] = pool_.dimension();
    candidate->data_info[1] = candidate->base_count;
    candidate->data_info[2] = 2;  // L2; G3 never uses C2 residual calibration here.
    G3_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&candidate->data_d),
                              static_cast<std::size_t>(candidate->base_count) * pool_.dimension() * sizeof(float)));
    for (LocalRow local = 0; local < candidate->base_count; ++local) {
      const std::int16_t* source = pool_.vector_for_stable(candidate->local_to_stable[local]);
      float* target = candidate->data_d + static_cast<std::size_t>(local) * pool_.dimension();
      for (int dim = 0; dim < pool_.dimension(); ++dim) target[dim] = static_cast<float>(source[dim]);
    }
    indexConstru(candidate->data_d, nullptr, nullptr, candidate->data_info,
                 candidate->id_list, candidate->node_list, candidate->max_node_num,
                 candidate->tree_height, candidate->empty_list);
    G3_CUDA(cudaDeviceSynchronize());
    G3_CUDA(cudaGetLastError());
    if (candidate->tree_height != G3_CONTROLLED_ARCHIVE_TREE_HEIGHT ||
        start_idx != G3_CONTROLLED_ARCHIVE_FINAL_START_INDEX ||
        MAX_SIZE != G3_CONTROLLED_ARCHIVE_MAX_SIZE) {
      fail("controlled G3 archive traversal rejects a tree height/start_idx/MAX_SIZE outside the fixed 4K contract");
    }
    // The pinned tree.cuh repacks id_list as exactly
    // base_count + leaf_count * LEAF_PAD_SLOTS physical int entries.  CUDA 13
    // no longer exposes the old runtime address-range query, so derive the
    // allocation contract from the same isolated builder source and verify all
    // leaf [lid,lid+size) spans against it in FrozenTreeSnapshot::capture.
    const int native_node_count = candidate->max_node_num[0];
    if (native_node_count <= 0) fail("native tree reports invalid node capacity for padded id_list");
    std::vector<TN> host_nodes(static_cast<std::size_t>(native_node_count));
    std::vector<int> host_empty(static_cast<std::size_t>(native_node_count));
    G3_CUDA(cudaMemcpy(host_nodes.data(), candidate->node_list,
                       host_nodes.size() * sizeof(TN), cudaMemcpyDeviceToHost));
    G3_CUDA(cudaMemcpy(host_empty.data(), candidate->empty_list,
                       host_empty.size() * sizeof(int), cudaMemcpyDeviceToHost));
    int native_leaf_count = 0;
    for (int node = 0; node < native_node_count; ++node) {
      if (host_empty[node] == 0 && host_nodes[node].is_leaf == 1) ++native_leaf_count;
    }
    const std::uint64_t padded_capacity =
        static_cast<std::uint64_t>(candidate->base_count) +
        static_cast<std::uint64_t>(native_leaf_count) * static_cast<std::uint64_t>(LEAF_PAD_SLOTS);
    if (native_leaf_count <= 0 || padded_capacity > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
      fail("cannot establish physical padded id_list capacity from pinned builder contract");
    }
    candidate->id_list_capacity = static_cast<int>(padded_capacity);
    candidate->owned_max_dis_d = max_dis_d;
    if (!candidate->ready()) fail("indexConstru did not produce a complete compact GTS runtime");
    // The constructor frees these temporaries itself.  Ensure subsequent release
    // cannot double-free dangling archive aliases.
    split_list = nullptr;
    pid_list = nullptr;
    dis_list = nullptr;
    split_num = nullptr;
  }

  void configure_no_residual_pruning() {
    float alpha[RP_MAX_LEVELS]{};
    float beta[RP_MAX_LEVELS]{};
    float gamma[RP_MAX_LEVELS];
    std::fill(std::begin(gamma), std::end(gamma), 1.0F);
    upload_rp_constants(alpha, beta, gamma, RP_MAX_LEVELS,
                        nullptr, nullptr, nullptr, 0, kResidualPruningMode);
    int device_mode = -1;
    G3_CUDA(cudaMemcpyFromSymbol(&device_mode, c_rp_mode, sizeof(int), 0,
                                  cudaMemcpyDeviceToHost));
    if (device_mode != kResidualPruningMode) {
      fail("device residual pruning mode readback is not zero");
    }
    residual_pruning_mode_zero_attested_ = true;
  }

  QueryInputSnapshot capture_query_input(const std::int16_t* query_values) const {
    if (!query_values) fail("null external vector query");
    if (pool_.dimension() <= 0) fail("cannot snapshot a query before immutable pool initialization");
    QueryInputSnapshot snapshot;
    snapshot.dimension = pool_.dimension();
    snapshot.values.assign(query_values, query_values + snapshot.dimension);
    snapshot.sha256 = query_input_snapshot_sha256(snapshot.values, snapshot.dimension);
    return snapshot;
  }

  // Upload strictly from the immutable host snapshot.  The caller's pointer is
  // never read after capture_query_input(), so an external buffer mutation
  // cannot alter the ticket-bound native query.
  void upload_query_snapshot(const QueryInputSnapshot& snapshot,
                             DeviceBuffer<float>* target) const {
    if (!target || snapshot.dimension != pool_.dimension() ||
        snapshot.sha256 != query_input_snapshot_sha256(snapshot.values, snapshot.dimension)) {
      fail("invalid immutable external query snapshot");
    }
    std::vector<float> converted(static_cast<std::size_t>(snapshot.dimension));
    for (int dim = 0; dim < snapshot.dimension; ++dim) {
      converted[static_cast<std::size_t>(dim)] = static_cast<float>(snapshot.values[dim]);
    }
    target->copy_from_host(converted);
  }

  QueryExport merge_query_partitions(const std::string& kind, int query_id, DistanceSq radius_sq,
                                     const TraversalReceipt& base, const float* query_device,
                                     bool is_range) const {
    QueryExport output;
    output.kind = kind;
    output.query_id = query_id;
    output.radius_sq = radius_sq;
    output.tree_version = base.tree_version;
    output.tree_payload_sha256 = base.tree_payload_sha256;
    output.native_tree_bytes_sha256 = base.native_tree_bytes_sha256;
    output.immutable_base_stable_ids_sha256 = base.immutable_base_stable_ids_sha256;
    output.immutable_base_payload_sha256 = base.immutable_base_payload_sha256;
    output.local_to_stable_sha256_value = base.local_to_stable_sha256_value;
    output.base_path = base.base_path;
    output.receipt_before_leaf_materialization = base.receipt_before_leaf_materialization;
    output.range_receipt_before_leaf_materialization = is_range && base.receipt_before_leaf_materialization;
    output.range_predicate_mirror_branch_aligned = base.range_predicate_mirror_branch_aligned;
    output.exact_full_immutable_base_range_fallback =
        base.exact_full_immutable_base_range_fallback;
    output.full_immutable_base_candidate_count = base.full_immutable_base_candidate_count;
    output.id_list_capacity = base.id_list_capacity;
    output.native_final_res_ids_cross_checked = base.native_final_res_ids_cross_checked;
    if (base.exact_full_immutable_base_range_fallback && !is_range) {
      fail("full immutable-base range fallback was attached to a non-range query");
    }
    const bool range_full_base_fallback =
        is_range && base.exact_full_immutable_base_range_fallback;
    if (range_full_base_fallback &&
        (G3_DIRECT_SIDECAR_RUNTIME_GUARD_OPEN || state_.sidecar_live() != 0)) {
      fail("range-safe fallback cannot silently omit live direct sidecars");
    }
    output.direct_sidecars_exact_range_filtered = is_range && !range_full_base_fallback;
    output.production_full_live_scan = range_full_base_fallback;
    output.full_frozen_base_snapshot_integrity_audit_on_operation = false;
    output.control_plane_partition_metadata_audit_on_operation = true;
    output.runner_single_image_admission_required = true;
    output.raw_receipt_leaf_pair_count = base.raw_receipt_leaf_pair_count;
    output.unique_receipt_leaf_count = base.unique_receipt_leaf_count;
    output.gts_visited_leaf_ids = base.visited_leaf_ids;
    output.receipt_leaf_spans = base.receipt_leaf_spans;
    output.base_receipt_rows = base.base_receipt_rows;
    output.native_final_res_ids = base.native_final_res_ids;
    if (!range_full_base_fallback) {
      output.sidecar_source_leaf_pairs = state_.sidecar_leaf_pairs_for(base.visited_leaf_ids);
      output.sidecar_candidate_ids = state_.sidecar_candidates_for(base.visited_leaf_ids);
    }
    output.delta_candidate_ids = state_.delta_ids();
    output.base_results = base.base_results;
    output.sidecar_results = scan_stable_rows_exact(pool_, query_device, output.sidecar_candidate_ids);
    output.delta_results = scan_stable_rows_exact(pool_, query_device, output.delta_candidate_ids);
    if (is_range) {
      const auto keep = [radius_sq](const StableDistance& row) { return row.distance_sq <= radius_sq; };
      output.sidecar_results.erase(std::remove_if(output.sidecar_results.begin(), output.sidecar_results.end(),
                                                   [&keep](const StableDistance& row) { return !keep(row); }),
                                   output.sidecar_results.end());
      output.delta_results.erase(std::remove_if(output.delta_results.begin(), output.delta_results.end(),
                                                 [&keep](const StableDistance& row) { return !keep(row); }),
                                 output.delta_results.end());
    }
    output.results = merge_partition_results(output.base_results, output.sidecar_results, output.delta_results);
    return output;
  }

  // A prior initialization-time attestation is insufficient because c_rp_mode
  // is a process-global CUDA symbol and another upload can drift it later.
  // Every public query/mutation/rebuild/verification path calls this helper
  // before touching native state; a readback error or nonzero mode latches
  // FailedStop rather than allowing a receipt to be treated as complete.
  void require_device_residual_mode_zero_or_fail_stop() {
    if (!residual_pruning_mode_zero_attested_) {
      enter_failed_stop_noexcept();
      fail("device residual pruning mode has not been initialized/read back as zero");
    }
    int device_mode = -1;
    try {
      G3_CUDA(cudaMemcpyFromSymbol(&device_mode, c_rp_mode, sizeof(int), 0,
                                    cudaMemcpyDeviceToHost));
    } catch (...) {
      enter_failed_stop_noexcept();
      throw;
    }
    if (device_mode != kResidualPruningMode) {
      enter_failed_stop_noexcept();
      fail("device residual pruning mode drifted from zero after initialization");
    }
  }

  void require_ready_for_mutation() {
    require_device_residual_mode_zero_or_fail_stop();
    if (mode_ != EngineMode::kReady || !runtime_.ready() || !frozen_.initialized()) {
      fail("mutation requires Ready engine after post-rebuild independent oracle completion");
    }
    if (!G3_DIRECT_SIDECAR_RUNTIME_GUARD_OPEN && state_.sidecar_live() != 0) {
      enter_failed_stop_noexcept();
      fail("closed direct guard observed a live sidecar");
    }
  }

  void require_ready_for_query(const std::string& kind) {
    require_device_residual_mode_zero_or_fail_stop();
    if ((mode_ != EngineMode::kReady && mode_ != EngineMode::kAwaitingPostRebuildOracle) ||
        !runtime_.ready() || !frozen_.initialized()) {
      fail("query requires a ready frozen base; failed-stop/rebuild modes are forbidden");
    }
    if (!G3_DIRECT_SIDECAR_RUNTIME_GUARD_OPEN && state_.sidecar_live() != 0) {
      enter_failed_stop_noexcept();
      fail("closed direct guard observed a live sidecar");
    }
    if (mode_ == EngineMode::kAwaitingPostRebuildOracle &&
        !post_rebuild_gate_.query_may_issue(kind)) {
      fail("this post-rebuild query kind is already issued, verified, or not pending");
    }
  }

  // Declaration order is part of the safety contract: process_lease_ is
  // acquired before any engine state and released after the destructor body.
  SameProcessExclusiveGtsLease process_lease_;
  mutable std::mutex api_mutex_;
  int sidecar_leaf_capacity_ = 0;
  int requested_k_ = 0;
  ImmutablePool pool_;
  SafeC1State state_;
  BaseTreeRuntime runtime_;
  FrozenTreeSnapshot frozen_;
  std::uint64_t tree_version_ = 0;
  std::uint64_t state_epoch_ = 0;
  std::uint64_t next_post_rebuild_gate_owner_nonce_ = 0;
  bool residual_pruning_mode_zero_attested_ = false;
  EngineMode mode_ = EngineMode::kUninitialized;
  PostRebuildOracleGate post_rebuild_gate_;
};

// JSONL serializers are intentionally data-only.  They may serialize ticket
// issuance diagnostics but never assert independent-oracle provenance or a
// successful oracle comparison; those remain runner responsibilities.
inline void json_int_array(std::ostringstream* out, const std::vector<int>& values) {
  *out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) *out << ',';
    *out << values[index];
  }
  *out << ']';
}

inline void json_stable_array(std::ostringstream* out, const std::vector<StableId>& values) {
  *out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) *out << ',';
    *out << values[index];
  }
  *out << ']';
}

inline void json_stable_distance_array(std::ostringstream* out,
                                       const std::vector<StableDistance>& values) {
  *out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) *out << ',';
    *out << '[' << values[index].stable_id << ','
         << static_cast<unsigned long long>(values[index].distance_sq) << ']';
  }
  *out << ']';
}

inline void json_sidecar_leaf_pairs(std::ostringstream* out,
                                    const std::vector<std::pair<int, StableId>>& values) {
  *out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) *out << ',';
    *out << '[' << values[index].first << ',' << values[index].second << ']';
  }
  *out << ']';
}

inline void json_receipt_leaf_spans(std::ostringstream* out,
                                    const std::vector<ReceiptLeafSpan>& values) {
  *out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) *out << ',';
    const ReceiptLeafSpan& span = values[index];
    *out << '[' << span.leaf_id << ',' << span.id_list_lid << ',' << span.size << ']';
  }
  *out << ']';
}

inline void json_receipt_base_rows(std::ostringstream* out,
                                   const std::vector<ReceiptBaseRow>& values) {
  *out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) *out << ',';
    const ReceiptBaseRow& row = values[index];
    *out << '[' << row.leaf_id << ',' << row.id_list_slot << ','
         << row.local_row << ',' << row.stable_id << ']';
  }
  *out << ']';
}

inline std::string serialize_query_jsonl_record(int op_index, const QueryExport& row) {
  std::ostringstream out;
  out << "{\"record\":\"query\",\"op_index\":" << op_index
      << ",\"kind\":\"" << row.kind << "\",\"query_id\":" << row.query_id
      << ",\"query_dimension\":" << row.query_dimension
      << ",\"requested_k\":" << row.requested_k
      << ",\"radius_sq\":";
  if (row.kind == "range") out << static_cast<unsigned long long>(row.radius_sq); else out << "null";
  out << ",\"tree_version\":" << row.tree_version
      << ",\"state_epoch\":" << row.state_epoch
      << ",\"tree_payload_sha256\":\"" << row.tree_payload_sha256 << "\""
      << ",\"native_tree_bytes_sha256\":\"" << row.native_tree_bytes_sha256 << "\""
      << ",\"immutable_base_stable_ids_sha256\":\"" << row.immutable_base_stable_ids_sha256 << "\""
      << ",\"immutable_base_payload_sha256\":\"" << row.immutable_base_payload_sha256 << "\""
      << ",\"local_to_stable_sha256\":\"" << row.local_to_stable_sha256_value << "\""
      << ",\"query_vector_sha256\":\"" << row.query_vector_sha256 << "\""
      << ",\"active_stable_ids_sha256\":\"" << row.active_stable_ids_sha256 << "\""
      << ",\"base_path\":\"" << row.base_path << "\""
      << ",\"receipt_before_leaf_materialization\":"
      << (row.receipt_before_leaf_materialization ? "true" : "false")
      << ",\"range_receipt_before_leaf_materialization\":"
      << (row.range_receipt_before_leaf_materialization ? "true" : "false")
      << ",\"id_list_capacity\":" << row.id_list_capacity
      << ",\"range_predicate_mirror_branch_aligned\":"
      << (row.range_predicate_mirror_branch_aligned ? "true" : "false")
      << ",\"exact_full_immutable_base_range_fallback\":"
      << (row.exact_full_immutable_base_range_fallback ? "true" : "false")
      << ",\"full_immutable_base_candidate_count\":"
      << row.full_immutable_base_candidate_count
      << ",\"native_final_res_ids_cross_checked\":"
      << (row.native_final_res_ids_cross_checked ? "true" : "false")
      << ",\"direct_sidecars_exact_range_filtered\":"
      << (row.direct_sidecars_exact_range_filtered ? "true" : "false")
      << ",\"production_full_live_scan\":"
      << (row.production_full_live_scan ? "true" : "false")
      << ",\"full_frozen_base_snapshot_integrity_audit_on_operation\":"
      << (row.full_frozen_base_snapshot_integrity_audit_on_operation ? "true" : "false")
      << ",\"control_plane_partition_metadata_audit_on_operation\":"
      << (row.control_plane_partition_metadata_audit_on_operation ? "true" : "false")
      << ",\"runner_single_image_admission_required\":"
      << (row.runner_single_image_admission_required ? "true" : "false")
      << ",\"post_rebuild_oracle_required\":"
      << (row.post_rebuild_oracle_required ? "true" : "false")
      << ",\"post_rebuild_verification_ticket_issued\":"
      << (row.post_rebuild_verification_ticket_issued ? "true" : "false")
      << ",\"post_rebuild_issuance_nonce\":" << row.post_rebuild_issuance_nonce
      << ",\"post_rebuild_receipt_sha256\":\"" << row.post_rebuild_receipt_sha256 << "\""
      << ",\"post_rebuild_result_sha256\":\"" << row.post_rebuild_result_sha256 << "\""
      << ",\"post_rebuild_binding_sha256\":\"" << row.post_rebuild_binding_sha256 << "\""
      << ",\"raw_receipt_leaf_pair_count\":" << row.raw_receipt_leaf_pair_count
      << ",\"unique_receipt_leaf_count\":" << row.unique_receipt_leaf_count
      << ",\"gts_visited_leaf_ids\":";
  json_int_array(&out, row.gts_visited_leaf_ids);
  out << ",\"receipt_leaf_spans\":";
  json_receipt_leaf_spans(&out, row.receipt_leaf_spans);
  out << ",\"base_receipt_rows\":";
  json_receipt_base_rows(&out, row.base_receipt_rows);
  out << ",\"native_final_res_ids\":";
  json_int_array(&out, row.native_final_res_ids);
  out << ",\"sidecar_source_leaf_pairs\":";
  json_sidecar_leaf_pairs(&out, row.sidecar_source_leaf_pairs);
  out << ",\"sidecar_candidate_ids\":";
  json_stable_array(&out, row.sidecar_candidate_ids);
  out << ",\"delta_candidate_ids\":";
  json_stable_array(&out, row.delta_candidate_ids);
  out << ",\"base_results\":";
  json_stable_distance_array(&out, row.base_results);
  out << ",\"sidecar_results\":";
  json_stable_distance_array(&out, row.sidecar_results);
  out << ",\"delta_results\":";
  json_stable_distance_array(&out, row.delta_results);
  out << ",\"results\":";
  json_stable_distance_array(&out, row.results);
  out << "}";
  return out.str();
}

inline std::string serialize_rebuild_jsonl_record(int op_index, const RebuildReceipt& row) {
  std::ostringstream out;
  out << "{\"record\":\"rebuild\",\"op_index\":" << op_index
      << ",\"tree_version\":" << row.tree_version
      << ",\"tree_height\":" << row.tree_height
      << ",\"node_capacity\":" << row.node_capacity
      << ",\"nonempty_node_count\":" << row.nonempty_node_count
      << ",\"fanout\":" << row.fanout
      << ",\"live_ids_sha256\":\"" << row.live_ids_sha256 << "\""
      << ",\"immutable_base_stable_ids_sha256\":\"" << row.immutable_base_stable_ids_sha256 << "\""
      << ",\"immutable_base_payload_sha256\":\"" << row.immutable_base_payload_sha256 << "\""
      << ",\"native_tree_bytes_sha256\":\"" << row.native_tree_bytes_sha256 << "\""
      << ",\"local_to_stable_sha256\":\"" << row.local_to_stable_sha256_value << "\""
      << ",\"logical_leaf_stable_ids_sha256\":\"" << row.logical_leaf_stable_ids_sha256 << "\""
      << ",\"tree_payload_sha256\":\"" << row.tree_payload_sha256 << "\""
      << ",\"base_count\":" << row.base_count
      << ",\"sidecar_live\":" << row.sidecar_live
      << ",\"delta_live\":" << row.delta_live
      << ",\"compact_mapping_bijection_ok\":" << (row.compact_mapping_bijection_ok ? "true" : "false")
      << ",\"raw_leaf_rows_cover_compact_range\":" << (row.raw_leaf_rows_cover_compact_range ? "true" : "false")
      << ",\"destructive_fail_stop_contract\":" << (row.destructive_fail_stop_contract ? "true" : "false")
      << ",\"old_generation_retired_before_candidate_build\":" << (row.old_generation_retired_before_candidate_build ? "true" : "false")
      << ",\"dynamic_tiers_replaced_after_generation_publish\":" << (row.dynamic_tiers_replaced_after_generation_publish ? "true" : "false")
      << ",\"post_rebuild_knn_oracle_required\":" << (row.post_rebuild_knn_oracle_required ? "true" : "false")
      << ",\"post_rebuild_range_oracle_required\":" << (row.post_rebuild_range_oracle_required ? "true" : "false")
      << "}";
  return out.str();
}

// Compile-only/audit schema strings.  A later runner must turn QueryExport and
// RebuildReceipt into JSONL under this exact field contract; it must not claim
// runtime results until native fixture execution actually occurs.
constexpr const char* kG3QueryReceiptSchema =
    "kind,query_id,query_dimension,requested_k,tree_version,state_epoch,"
    "tree_payload_sha256,native_tree_bytes_sha256,immutable_base_stable_ids_sha256,"
    "immutable_base_payload_sha256,local_to_stable_sha256,query_vector_sha256,"
    "active_stable_ids_sha256,base_path,receipt_before_leaf_materialization,"
    "range_receipt_before_leaf_materialization,range_predicate_mirror_branch_aligned,"
    "exact_full_immutable_base_range_fallback,full_immutable_base_candidate_count,"
    "id_list_capacity,native_final_res_ids_cross_checked,direct_sidecars_exact_range_filtered,"
    "production_full_live_scan,full_frozen_base_snapshot_integrity_audit_on_operation,"
    "control_plane_partition_metadata_audit_on_operation,"
    "runner_single_image_admission_required,post_rebuild_oracle_required,"
    "post_rebuild_verification_ticket_issued,post_rebuild_issuance_nonce,"
    "post_rebuild_receipt_sha256,post_rebuild_result_sha256,post_rebuild_binding_sha256,"
    "raw_receipt_leaf_pair_count,unique_receipt_leaf_count,gts_visited_leaf_ids,"
    "receipt_leaf_spans,base_receipt_rows,native_final_res_ids,"
    "sidecar_source_leaf_pairs,sidecar_candidate_ids,delta_candidate_ids,base_results,"
    "sidecar_results,delta_results,results";
constexpr const char* kG3RebuildReceiptSchema =
    "tree_version,tree_height,node_capacity,nonempty_node_count,fanout,live_ids_sha256,immutable_base_stable_ids_sha256,"
    "immutable_base_payload_sha256,native_tree_bytes_sha256,local_to_stable_sha256,"
    "logical_leaf_stable_ids_sha256,tree_payload_sha256,base_count,sidecar_live,delta_live,"
    "compact_mapping_bijection_ok,raw_leaf_rows_cover_compact_range,"
    "destructive_fail_stop_contract,old_generation_retired_before_candidate_build,"
    "dynamic_tiers_replaced_after_generation_publish,post_rebuild_knn_oracle_required,"
    "post_rebuild_range_oracle_required";

}  // namespace safe_c1_g3

void upload_rp_constants(float* h_alpha, float* h_beta, float* h_gamma,
                         int /*num_levels*/, float* h_lut_breaks,
                         float* h_lut_slopes, float* h_lut_intercepts,
                         int lut_size, int mode) {
  G3_CUDA(cudaMemcpyToSymbol(c_rp_alpha, h_alpha, RP_MAX_LEVELS * sizeof(float), 0,
                             cudaMemcpyHostToDevice));
  G3_CUDA(cudaMemcpyToSymbol(c_rp_beta, h_beta, RP_MAX_LEVELS * sizeof(float), 0,
                             cudaMemcpyHostToDevice));
  G3_CUDA(cudaMemcpyToSymbol(c_rp_gamma, h_gamma, RP_MAX_LEVELS * sizeof(float), 0,
                             cudaMemcpyHostToDevice));
  if (h_lut_breaks && h_lut_slopes && h_lut_intercepts) {
    G3_CUDA(cudaMemcpyToSymbol(c_lut_breaks, h_lut_breaks, RP_LUT_SIZE * sizeof(float), 0,
                               cudaMemcpyHostToDevice));
    G3_CUDA(cudaMemcpyToSymbol(c_lut_slopes, h_lut_slopes, RP_LUT_SIZE * sizeof(float), 0,
                               cudaMemcpyHostToDevice));
    G3_CUDA(cudaMemcpyToSymbol(c_lut_intercepts, h_lut_intercepts, RP_LUT_SIZE * sizeof(float), 0,
                               cudaMemcpyHostToDevice));
  }
  G3_CUDA(cudaMemcpyToSymbol(c_lut_num_segments, &lut_size, sizeof(int), 0,
                             cudaMemcpyHostToDevice));
  G3_CUDA(cudaMemcpyToSymbol(c_rp_mode, &mode, sizeof(int), 0,
                             cudaMemcpyHostToDevice));
}
