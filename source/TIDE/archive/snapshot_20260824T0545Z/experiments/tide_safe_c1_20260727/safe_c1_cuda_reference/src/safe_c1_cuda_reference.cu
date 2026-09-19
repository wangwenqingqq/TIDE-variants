// CUDA Safe-C1 reference trace executor.
//
// Scope: correctness bridge for the revised Safe-C1 state contract.  It uses an
// immutable vector pool, stable IDs, a frozen radial routing snapshot, bounded
// per-leaf sidecars, and a separate exact delta tier.  Query answers are
// obtained by an exact CUDA scan of the live stable-ID set, so this program is
// deliberately NOT a latency result and NOT a claim that the archived GTS
// incremental updater is correct.  It never includes or calls the legacy
// incremental_insert/update code.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr char kMagic[8] = {'E', '1', 'G', 'T', 'R', 'C', '0', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr float kStrictEps = 1.0e-6F;

enum OpCode : std::uint8_t { kInsert = 1, kDelete = 2, kKnn = 3, kRange = 4 };

enum class Mode { kSafe, kBuffer };
enum class Placement : std::uint8_t { kBase, kDirect, kDelta, kDeleted };

#pragma pack(push, 1)
struct TraceHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t dimension;
  std::uint32_t base_n;
  std::uint32_t reservoir_n;
  std::uint32_t pool_n;
  std::uint32_t query_n;
  std::uint32_t k;
  float radius;
  std::uint64_t event_count;
};
struct TraceEvent {
  std::uint32_t op_index;
  std::uint8_t op_code;
  std::uint8_t reserved[3];
  std::int32_t argument;
};
#pragma pack(pop)
static_assert(sizeof(TraceHeader) == 48, "trace header ABI mismatch");
static_assert(sizeof(TraceEvent) == 12, "trace event ABI mismatch");

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void cuda_check(cudaError_t status, const char* expression, const char* file, int line) {
  if (status != cudaSuccess) {
    std::ostringstream out;
    out << "CUDA failure " << file << ':' << line << " for " << expression << ": "
        << cudaGetErrorString(status) << " (" << static_cast<int>(status) << ')';
    fail(out.str());
  }
}
#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

__global__ void live_l2_kernel(const float* pool, const float* query,
                               const std::uint8_t* active, float* distances,
                               int pool_n, int dimension) {
  const int id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= pool_n) return;
  if (active[id] == 0) {
    // Avoid relying on a toolkit-specific CUDART_INF_F macro.
    distances[id] = __int_as_float(0x7f800000);
    return;
  }
  float sum = 0.0F;
  const float* vector = pool + static_cast<std::size_t>(id) * dimension;
  for (int d = 0; d < dimension; ++d) {
    const float diff = vector[d] - query[d];
    sum += diff * diff;
  }
  distances[id] = sqrtf(sum);
}

struct Args {
  std::string bundle;
  std::string trace;
  std::string output;
  std::string summary;
  Mode mode = Mode::kSafe;
  int leaves = 16;
  int leaf_capacity = 64;
};

std::string mode_name(Mode mode) { return mode == Mode::kSafe ? "safe" : "buffer"; }

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    auto need = [&]() -> std::string {
      if (++i >= argc) fail("missing value after " + key);
      return std::string(argv[i]);
    };
    if (key == "--bundle") args.bundle = need();
    else if (key == "--trace") args.trace = need();
    else if (key == "--out") args.output = need();
    else if (key == "--summary") args.summary = need();
    else if (key == "--mode") {
      const std::string value = need();
      if (value == "safe") args.mode = Mode::kSafe;
      else if (value == "buffer") args.mode = Mode::kBuffer;
      else fail("--mode must be safe or buffer");
    } else if (key == "--leaves") args.leaves = std::stoi(need());
    else if (key == "--leaf-capacity") args.leaf_capacity = std::stoi(need());
    else if (key == "--help") {
      std::cout << "usage: safe_c1_cuda_reference --bundle DIR --trace FILE --out JSONL --summary JSON "
                   "--mode safe|buffer [--leaves N] [--leaf-capacity N]\n";
      std::exit(0);
    } else {
      fail("unknown argument: " + key);
    }
  }
  if (args.bundle.empty() || args.trace.empty() || args.output.empty() || args.summary.empty()) {
    fail("--bundle, --trace, --out, and --summary are required");
  }
  if (args.leaves < 2 || args.leaf_capacity < 1) fail("invalid leaves or leaf capacity");
  return args;
}

template <typename T>
std::vector<T> read_exact(const std::string& path, std::size_t count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open " + path);
  std::vector<T> values(count);
  input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (input.gcount() != static_cast<std::streamsize>(count * sizeof(T))) {
    fail("truncated or incorrectly sized input: " + path);
  }
  char extra = 0;
  if (input.read(&extra, 1)) fail("unexpected extra bytes in input: " + path);
  return values;
}

std::pair<TraceHeader, std::vector<TraceEvent>> read_trace(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open trace " + path);
  TraceHeader header{};
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (input.gcount() != static_cast<std::streamsize>(sizeof(header))) fail("truncated trace header");
  if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0) fail("trace magic mismatch");
  if (header.version != kVersion) fail("unsupported trace version");
  if (header.dimension == 0 || header.base_n == 0 || header.pool_n < header.base_n ||
      header.reservoir_n != header.pool_n - header.base_n || header.query_n == 0 || header.k == 0) {
    fail("invalid trace header dimensions/counts");
  }
  std::vector<TraceEvent> events(header.event_count);
  input.read(reinterpret_cast<char*>(events.data()),
             static_cast<std::streamsize>(events.size() * sizeof(TraceEvent)));
  if (input.gcount() != static_cast<std::streamsize>(events.size() * sizeof(TraceEvent))) {
    fail("truncated trace events");
  }
  char extra = 0;
  if (input.read(&extra, 1)) fail("unexpected extra bytes in trace");
  for (std::size_t i = 0; i < events.size(); ++i) {
    const TraceEvent& event = events[i];
    if (event.op_index != i || event.op_code < kInsert || event.op_code > kRange) {
      fail("invalid or non-contiguous trace event at index " + std::to_string(i));
    }
  }
  return {header, std::move(events)};
}

float l2(const std::vector<float>& pool, int dimension, int left, int right) {
  double sum = 0.0;
  const float* a = pool.data() + static_cast<std::size_t>(left) * dimension;
  const float* b = pool.data() + static_cast<std::size_t>(right) * dimension;
  for (int d = 0; d < dimension; ++d) {
    const double diff = static_cast<double>(a[d]) - static_cast<double>(b[d]);
    sum += diff * diff;
  }
  return static_cast<float>(std::sqrt(sum));
}

struct FrozenLeaf {
  float low = 0.0F;
  float high = 0.0F;
};

struct FrozenRadialSnapshot {
  int pivot = 0;
  std::vector<FrozenLeaf> leaves;

  static FrozenRadialSnapshot build(const std::vector<float>& pool, int dimension,
                                    int base_n, int leaves_requested) {
    if (base_n < leaves_requested * 2) fail("base_n too small for frozen radial leaves");
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(base_n - 1);
    for (int id = 1; id < base_n; ++id) ranked.emplace_back(l2(pool, dimension, 0, id), id);
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
      return a.first != b.first ? a.first < b.first : a.second < b.second;
    });
    FrozenRadialSnapshot snapshot;
    snapshot.leaves.reserve(leaves_requested);
    for (int leaf = 0; leaf < leaves_requested; ++leaf) {
      const int begin = static_cast<int>((static_cast<std::int64_t>(leaf) * ranked.size()) / leaves_requested);
      const int end = static_cast<int>((static_cast<std::int64_t>(leaf + 1) * ranked.size()) / leaves_requested) - 1;
      if (begin < 0 || end < begin || end >= static_cast<int>(ranked.size())) fail("radial partition error");
      snapshot.leaves.push_back(FrozenLeaf{ranked[begin].first, ranked[end].first});
    }
    return snapshot;
  }

  int certify_leaf(const std::vector<float>& pool, int dimension, int id) const {
    const float distance = l2(pool, dimension, pivot, id);
    int match = -1;
    for (int leaf = 0; leaf < static_cast<int>(leaves.size()); ++leaf) {
      const FrozenLeaf& interval = leaves[leaf];
      if (distance > interval.low + kStrictEps && distance < interval.high - kStrictEps) {
        if (match != -1) return -1;  // overlap is rejected, never arbitrarily routed.
        match = leaf;
      }
    }
    return match;
  }
};

std::string json_number(float value) {
  std::ostringstream out;
  out << std::setprecision(9) << value;
  return out.str();
}

void emit_update(std::ofstream& output, const TraceEvent& event, Placement placement) {
  const char* op = event.op_code == kInsert ? "insert" : "delete";
  const char* placement_name = placement == Placement::kDirect ? "direct" :
                               placement == Placement::kDelta ? "delta" :
                               placement == Placement::kDeleted ? "deleted" : "base";
  output << "{\"record\":\"update\",\"op_index\":" << event.op_index
         << ",\"op\":\"" << op << "\",\"stable_id\":" << event.argument
         << ",\"placement\":\"" << placement_name << "\"}\n";
}

void emit_query(std::ofstream& output, const TraceEvent& event, const char* kind,
                const std::vector<std::pair<float, int>>& rows) {
  output << "{\"record\":\"query\",\"op_index\":" << event.op_index
         << ",\"kind\":\"" << kind << "\",\"query_id\":" << event.argument
         << ",\"results\":[";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (i) output << ',';
    output << '[' << rows[i].second << ',' << json_number(rows[i].first) << ']';
  }
  output << "]}\n";
}

const char* placement_name(Placement placement) {
  switch (placement) {
    case Placement::kBase: return "base";
    case Placement::kDirect: return "direct";
    case Placement::kDelta: return "delta";
    case Placement::kDeleted: return "deleted";
  }
  return "unknown";
}

struct Counters {
  int inserts = 0;
  int deletes = 0;
  int knn = 0;
  int range = 0;
  int direct_inserts = 0;
  int delta_inserts = 0;
  int certificate_rejects = 0;
  int capacity_rejects = 0;
  std::uint64_t candidate_distances = 0;
};

void write_summary(const std::string& path, const Args& args, const TraceHeader& header,
                   const cudaDeviceProp& prop, const Counters& counters,
                   int final_active, int direct_live, int delta_live) {
  std::ofstream output(path);
  if (!output) fail("cannot write summary " + path);
  output << "{\n"
         << "  \"schema\": \"e1g-safe-c1-cuda-reference-v1\",\n"
         << "  \"status\": \"PASS_REFERENCE_CORRECTNESS\",\n"
         << "  \"result_scope\": \"CUDA exact-live-set Safe-C1 reference correctness; not GTS dynamic integration and not a latency result\",\n"
         << "  \"legacy_incremental_updater_used\": false,\n"
         << "  \"archived_gts_dynamic_code_used\": false,\n"
         << "  \"cuda_query_method\": \"exact full active-set scan; frozen tree is used only for Safe-C1 direct-vs-delta placement\",\n"
         << "  \"mode\": \"" << mode_name(args.mode) << "\",\n"
         << "  \"dimension\": " << header.dimension << ",\n"
         << "  \"base_n\": " << header.base_n << ",\n"
         << "  \"reservoir_n\": " << header.reservoir_n << ",\n"
         << "  \"pool_n\": " << header.pool_n << ",\n"
         << "  \"query_n\": " << header.query_n << ",\n"
         << "  \"k\": " << header.k << ",\n"
         << "  \"radius\": " << std::setprecision(9) << header.radius << ",\n"
         << "  \"frozen_routing\": {\"leaf_count\": " << args.leaves
         << ", \"leaf_sidecar_capacity\": " << args.leaf_capacity
         << ", \"certificate\": \"unique strict interior radial interval; otherwise global delta\"},\n"
         << "  \"operations\": {\"insert\": " << counters.inserts
         << ", \"delete\": " << counters.deletes << ", \"knn\": " << counters.knn
         << ", \"range\": " << counters.range << "},\n"
         << "  \"placement\": {\"direct_inserts\": " << counters.direct_inserts
         << ", \"delta_inserts\": " << counters.delta_inserts
         << ", \"certificate_rejects\": " << counters.certificate_rejects
         << ", \"capacity_rejects\": " << counters.capacity_rejects
         << ", \"direct_live_final\": " << direct_live
         << ", \"delta_live_final\": " << delta_live << "},\n"
         << "  \"exact_scan_candidate_distances\": " << counters.candidate_distances << ",\n"
         << "  \"final_active_count\": " << final_active << ",\n"
         << "  \"gpu\": {\"name\": \"" << prop.name << "\", \"compute_capability\": \""
         << prop.major << '.' << prop.minor << "\", \"sm_count\": " << prop.multiProcessorCount << "}\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  float* d_pool = nullptr;
  float* d_queries = nullptr;
  float* d_distances = nullptr;
  std::uint8_t* d_active = nullptr;
  try {
    const Args args = parse_args(argc, argv);
    const auto [header, events] = read_trace(args.trace);
    const std::string pool_path = args.bundle + "/pool.f32";
    const std::string queries_path = args.bundle + "/queries.f32";
    std::vector<float> pool = read_exact<float>(pool_path,
        static_cast<std::size_t>(header.pool_n) * header.dimension);
    std::vector<float> queries = read_exact<float>(queries_path,
        static_cast<std::size_t>(header.query_n) * header.dimension);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const FrozenRadialSnapshot frozen = FrozenRadialSnapshot::build(
        pool, static_cast<int>(header.dimension), static_cast<int>(header.base_n), args.leaves);
    std::vector<std::uint8_t> active(header.pool_n, 0);
    std::vector<Placement> placement(header.pool_n, Placement::kDeleted);
    for (std::uint32_t id = 0; id < header.base_n; ++id) {
      active[id] = 1;
      placement[id] = Placement::kBase;
    }
    std::vector<int> sidecar_count(args.leaves, 0);

    CUDA_CHECK(cudaMalloc(&d_pool, pool.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_queries, queries.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_active, active.size() * sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_distances, active.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pool, pool.data(), pool.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, queries.data(), queries.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_active, active.data(), active.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice));

    std::ofstream output(args.output);
    if (!output) fail("cannot write output " + args.output);
    output << "{\"record\":\"meta\",\"schema\":\"e1g-safe-c1-cuda-reference-v1\","
           << "\"scope\":\"reference correctness only; not dynamic GTS integration\","
           << "\"mode\":\"" << mode_name(args.mode) << "\"}\n";

    Counters counters;
    std::vector<float> host_distances(header.pool_n);
    const int threads = 256;
    const int blocks = (static_cast<int>(header.pool_n) + threads - 1) / threads;

    for (const TraceEvent& event : events) {
      const int argument = event.argument;
      if (event.op_code == kInsert) {
        if (argument < static_cast<int>(header.base_n) || argument >= static_cast<int>(header.pool_n)) {
          fail("insert stable ID is outside disjoint reservoir at op " + std::to_string(event.op_index));
        }
        if (active[argument] != 0) fail("insert targets an active stable ID at op " + std::to_string(event.op_index));
        active[argument] = 1;
        Placement selected = Placement::kDelta;
        if (args.mode == Mode::kSafe) {
          const int leaf = frozen.certify_leaf(pool, static_cast<int>(header.dimension), argument);
          if (leaf < 0) {
            ++counters.certificate_rejects;
          } else if (sidecar_count[leaf] >= args.leaf_capacity) {
            ++counters.capacity_rejects;
          } else {
            selected = Placement::kDirect;
            ++sidecar_count[leaf];
          }
        }
        placement[argument] = selected;
        if (selected == Placement::kDirect) ++counters.direct_inserts;
        else ++counters.delta_inserts;
        CUDA_CHECK(cudaMemcpy(d_active + argument, &active[argument], sizeof(std::uint8_t), cudaMemcpyHostToDevice));
        emit_update(output, event, selected);
        ++counters.inserts;
      } else if (event.op_code == kDelete) {
        if (argument < 0 || argument >= static_cast<int>(header.pool_n) || active[argument] == 0) {
          fail("delete targets inactive/out-of-range stable ID at op " + std::to_string(event.op_index));
        }
        const Placement before = placement[argument];
        if (before == Placement::kDirect) {
          const int leaf = frozen.certify_leaf(pool, static_cast<int>(header.dimension), argument);
          if (leaf >= 0 && sidecar_count[leaf] > 0) --sidecar_count[leaf];
          // The placement leaf is intentionally not recomputed for routing: it only
          // decrements capacity when the original strict certificate remains valid.
        }
        active[argument] = 0;
        placement[argument] = Placement::kDeleted;
        CUDA_CHECK(cudaMemcpy(d_active + argument, &active[argument], sizeof(std::uint8_t), cudaMemcpyHostToDevice));
        emit_update(output, event, Placement::kDeleted);
        ++counters.deletes;
      } else {
        if (argument < 0 || argument >= static_cast<int>(header.query_n)) {
          fail("query ID out of range at op " + std::to_string(event.op_index));
        }
        const float* query = d_queries + static_cast<std::size_t>(argument) * header.dimension;
        live_l2_kernel<<<blocks, threads>>>(d_pool, query, d_active, d_distances,
                                             static_cast<int>(header.pool_n), static_cast<int>(header.dimension));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host_distances.data(), d_distances,
                              host_distances.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<std::pair<float, int>> rows;
        rows.reserve(header.pool_n);
        for (int id = 0; id < static_cast<int>(header.pool_n); ++id) {
          if (active[id] != 0) rows.emplace_back(host_distances[id], id);
        }
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
          return a.first != b.first ? a.first < b.first : a.second < b.second;
        });
        counters.candidate_distances += rows.size();
        if (event.op_code == kKnn) {
          if (rows.size() > header.k) rows.resize(header.k);
          emit_query(output, event, "knn", rows);
          ++counters.knn;
        } else if (event.op_code == kRange) {
          const auto keep = std::upper_bound(rows.begin(), rows.end(), header.radius,
              [](float radius, const std::pair<float, int>& value) { return radius < value.first; });
          rows.erase(keep, rows.end());
          emit_query(output, event, "range", rows);
          ++counters.range;
        } else {
          fail("unknown query opcode");
        }
      }
      if (!output) fail("failed while writing JSONL output");
    }
    output.close();

    int final_active = 0, direct_live = 0, delta_live = 0;
    for (std::size_t id = 0; id < active.size(); ++id) {
      if (active[id] == 0) continue;
      ++final_active;
      if (placement[id] == Placement::kDirect) ++direct_live;
      else if (placement[id] == Placement::kDelta) ++delta_live;
    }
    write_summary(args.summary, args, header, prop, counters, final_active, direct_live, delta_live);

    CUDA_CHECK(cudaFree(d_pool)); d_pool = nullptr;
    CUDA_CHECK(cudaFree(d_queries)); d_queries = nullptr;
    CUDA_CHECK(cudaFree(d_active)); d_active = nullptr;
    CUDA_CHECK(cudaFree(d_distances)); d_distances = nullptr;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "safe_c1_cuda_reference: " << error.what() << '\n';
    if (d_pool) cudaFree(d_pool);
    if (d_queries) cudaFree(d_queries);
    if (d_active) cudaFree(d_active);
    if (d_distances) cudaFree(d_distances);
    return 2;
  }
}
