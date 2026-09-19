// Isolated C1 four-variant microbenchmark runner.
// This translation unit intentionally reuses the current isolated implementation
// (main.cu + update.cuh) and changes only the two pre-registered C1 gates.
// It is not a general dynamic-update/C3 benchmark: its input must contain type-2
// queries only and its tree-invariance/result-equivalence artifacts are mandatory.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Reuse the isolated production TU as a library in this executable, avoiding a
// second hand-copied implementation of GTS setup/query semantics.
#define main gts_legacy_main_unused
#include "main.cu"
#undef main

namespace fs = std::filesystem;
namespace c1c = gtspp_c1_counter;

namespace {

struct Config {
  fs::path base;
  fs::path trace;
  fs::path out;
  float radius = 0.0f;
  int replicate = -1;
  std::string variant;
  bool smoke = false;
  bool profile_only = false;
  int trace_limit = 1024;
};

struct CanonicalPair {
  std::int32_t id;
  std::uint32_t distance_bits;
};

struct QueryResult {
  int qid = -1;
  int count = 0;
  std::uint64_t canonical_fnv1a64 = 0;
  double wall_ms = 0.0;
  float gpu_ms = 0.0f;
  c1c::Snapshot alloc_delta{};
  int buffered_result_count = 0;
  // Both exported capacity fields are a single immutable local snapshot
  // captured before an ephemeral workspace can reset the global capacity.
  int update_result_capacity_slots = 0;
  int total_result_capacity_slots = 0;
  std::string capacity_snapshot_stage;
  std::vector<CanonicalPair> canonical;
};

struct TreeFingerprint {
  std::uint64_t fnv1a64 = 0;
  int node_count = 0;
  int id_slots = 0;
  int tree_height = 0;
};

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

[[noreturn]] void die(const std::string& message) {
  std::cerr << "C1Microbench error: " << message << std::endl;
  std::exit(2);
}

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " --base PATH --trace PATH --radius R --out DIR --replicate N --variant NAME"
      << " [--smoke --trace-limit N] [--profile-only]\n"
      << "Measured mode requires exactly the pre-registered 1024 type-2 operations.\n"
      << "--smoke marks a shorter, unmeasured execution and is never a result run.\n"
      << "--profile-only runs the full frozen trace for Nsight allocation evidence only; it is never timing data.\n";
}

int parse_int(const char* text, const char* field) {
  try {
    std::size_t pos = 0;
    const int value = std::stoi(text, &pos);
    if (text[pos] != '\0') throw std::invalid_argument("trailing characters");
    return value;
  } catch (const std::exception&) {
    die(std::string("invalid integer for ") + field + ": " + text);
  }
}

float parse_float(const char* text, const char* field) {
  try {
    std::size_t pos = 0;
    const float value = std::stof(text, &pos);
    if (text[pos] != '\0') throw std::invalid_argument("trailing characters");
    return value;
  } catch (const std::exception&) {
    die(std::string("invalid float for ") + field + ": " + text);
  }
}

Config parse_args(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    usage(argv[0]);
    std::exit(0);
  }
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto require_value = [&](const char* name) -> const char* {
      if (++i >= argc) die(std::string("missing value for ") + name);
      return argv[i];
    };
    if (arg == "--base") cfg.base = require_value("--base");
    else if (arg == "--trace") cfg.trace = require_value("--trace");
    else if (arg == "--radius") cfg.radius = parse_float(require_value("--radius"), "--radius");
    else if (arg == "--out") cfg.out = require_value("--out");
    else if (arg == "--replicate") cfg.replicate = parse_int(require_value("--replicate"), "--replicate");
    else if (arg == "--variant") cfg.variant = require_value("--variant");
    else if (arg == "--smoke") cfg.smoke = true;
    else if (arg == "--profile-only") cfg.profile_only = true;
    else if (arg == "--trace-limit") cfg.trace_limit = parse_int(require_value("--trace-limit"), "--trace-limit");
    else die("unknown argument: " + arg);
  }
  if (cfg.base.empty() || cfg.trace.empty() || cfg.out.empty() || cfg.variant.empty() || cfg.replicate < 1) {
    die("--base, --trace, --radius, --out, --replicate, and --variant are all required");
  }
  if (!(cfg.radius > 0.0f) || !std::isfinite(cfg.radius)) die("radius must be finite and > 0");
  if (cfg.trace_limit <= 0 || cfg.trace_limit > 1024) die("--trace-limit must be in [1,1024]");
  if (!cfg.smoke && cfg.trace_limit != 1024) {
    die("a non-smoke execution must use the pre-registered 1024-operation trace");
  }
  if (cfg.smoke && cfg.profile_only) die("--smoke and --profile-only are mutually exclusive");
  if (cfg.profile_only && cfg.trace_limit != 1024) die("--profile-only must use the frozen 1024-operation trace");
  if (!fs::is_regular_file(cfg.base) || !fs::is_regular_file(cfg.trace)) die("base or trace path is not a regular file");
  return cfg;
}

const char* compiled_variant_name() {
#if C1_PERSISTENT_WORKSPACE
#if C1_ONE_QUERY_FASTPATH
  return "P_F_full_C1";
#else
  return "P_G_workspace_only";
#endif
#else
#if C1_ONE_QUERY_FASTPATH
  return "E_F_fastpath_only";
#else
  return "E_G_c1_off_reference";
#endif
#endif
}

const char* run_mode(const Config& cfg) {
  if (cfg.profile_only) return "UNMEASURED_NSYS_PROFILE_DO_NOT_USE";
  return cfg.smoke ? "UNMEASURED_SMOKE_DO_NOT_USE" : "MEASURED_PROTOCOL";
}

void fnv_add_byte(std::uint64_t& h, std::uint8_t b) {
  h ^= b;
  h *= kFnvPrime;
}

template <typename T>
void fnv_add_scalar(std::uint64_t& h, T value) {
  static_assert(std::is_trivially_copyable<T>::value, "scalar must be trivially copyable");
  std::array<std::uint8_t, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  for (std::uint8_t b : bytes) fnv_add_byte(h, b);
}

void fnv_add_bytes(std::uint64_t& h, const void* data, std::size_t bytes) {
  const auto* ptr = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < bytes; ++i) fnv_add_byte(h, ptr[i]);
}

std::string hex64(std::uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << value;
  return out.str();
}


std::uint64_t canonical_hash(const std::vector<CanonicalPair>& pairs) {
  std::uint64_t h = kFnvOffset;
  fnv_add_scalar(h, static_cast<std::uint64_t>(pairs.size()));
  for (const CanonicalPair& p : pairs) {
    fnv_add_scalar(h, p.id);
    fnv_add_scalar(h, p.distance_bits);
  }
  return h;
}

std::vector<CanonicalPair> canonicalize_current_result(int total_count) {
  if (total_count < 0) die("negative result count");
  std::vector<CanonicalPair> pairs;
  pairs.reserve(static_cast<std::size_t>(total_count));
  for (int i = 0; i < total_count; ++i) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &total_result_dis[i], sizeof(bits));
    pairs.push_back(CanonicalPair{static_cast<std::int32_t>(total_result_id[i]), bits});
  }
  std::sort(pairs.begin(), pairs.end(), [](const CanonicalPair& a, const CanonicalPair& b) {
    if (a.id != b.id) return a.id < b.id;
    return a.distance_bits < b.distance_bits;
  });
  return pairs;
}

int padded_id_slots_from_host_nodes(const std::vector<TN>& nodes, const std::vector<int>& empty) {
  long long maximum = 0;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (empty[i] == 0 && nodes[i].is_leaf == 1) {
      const long long end = static_cast<long long>(nodes[i].lid) + nodes[i].size + LEAF_PAD_SLOTS;
      maximum = std::max(maximum, end);
    }
  }
  if (maximum <= 0 || maximum > std::numeric_limits<int>::max()) die("invalid padded id-list extent");
  return static_cast<int>(maximum);
}

TreeFingerprint tree_fingerprint() {
  CHECK(cudaDeviceSynchronize());
  if (max_node_num == nullptr || node_list == nullptr || empty_list == nullptr || max_dis_d == nullptr || id_list == nullptr) {
    die("tree fingerprint requested before static tree construction");
  }
  const int n = max_node_num[0];
  if (n <= 0) die("invalid max_node_num for fingerprint");
  std::vector<TN> nodes(static_cast<std::size_t>(n));
  std::vector<int> empty(static_cast<std::size_t>(n));
  std::vector<float> max_dis(static_cast<std::size_t>(n));
  CHECK(cudaMemcpy(nodes.data(), node_list, nodes.size() * sizeof(TN), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(empty.data(), empty_list, empty.size() * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(max_dis.data(), max_dis_d, max_dis.size() * sizeof(float), cudaMemcpyDeviceToHost));
  const int id_slots = padded_id_slots_from_host_nodes(nodes, empty);
  std::vector<int> ids(static_cast<std::size_t>(id_slots));
  CHECK(cudaMemcpy(ids.data(), id_list, ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> deleted;
  if (is_delete != nullptr && data_info != nullptr && data_info[1] > 0) {
    deleted.resize(static_cast<std::size_t>(data_info[1]));
    CHECK(cudaMemcpy(deleted.data(), is_delete, deleted.size() * sizeof(int), cudaMemcpyDeviceToHost));
  }
  std::uint64_t h = kFnvOffset;
  fnv_add_scalar(h, n);
  fnv_add_scalar(h, tree_h);
  fnv_add_scalar(h, TREE_ORDER);
  fnv_add_scalar(h, MAX_SIZE);
  fnv_add_scalar(h, id_slots);
  fnv_add_bytes(h, nodes.data(), nodes.size() * sizeof(TN));
  fnv_add_bytes(h, empty.data(), empty.size() * sizeof(int));
  fnv_add_bytes(h, max_dis.data(), max_dis.size() * sizeof(float));
  fnv_add_bytes(h, ids.data(), ids.size() * sizeof(int));
  if (!deleted.empty()) fnv_add_bytes(h, deleted.data(), deleted.size() * sizeof(int));
  return TreeFingerprint{h, n, id_slots, tree_h};
}

void append_tree_fingerprint(std::ofstream& out, const char* label, const TreeFingerprint& fp) {
  out << "{\"label\":\"" << label << "\",\"fnv1a64\":\"" << hex64(fp.fnv1a64)
      << "\",\"node_count\":" << fp.node_count << ",\"id_slots\":" << fp.id_slots
      << ",\"tree_height\":" << fp.tree_height << "}\n";
  out.flush();
}

void write_u32(std::ofstream& out, std::uint32_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void write_i32(std::ofstream& out, std::int32_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void write_canonical_record(std::ofstream& out, int ordinal, const QueryResult& result) {
  write_u32(out, static_cast<std::uint32_t>(ordinal));
  write_i32(out, static_cast<std::int32_t>(result.qid));
  write_u32(out, static_cast<std::uint32_t>(result.canonical.size()));
  for (const CanonicalPair& p : result.canonical) {
    write_i32(out, p.id);
    write_u32(out, p.distance_bits);
  }
  if (!out) die("failed writing canonical result artifact");
}

void write_query_json(std::ofstream& out, int ordinal, const char* phase, const QueryResult& result) {
  const c1c::Snapshot& a = result.alloc_delta;
  out << std::fixed << std::setprecision(6)
      << "{\"phase\":\"" << phase << "\",\"ordinal\":" << ordinal
      << ",\"qid\":" << result.qid << ",\"result_count\":" << result.count
      << ",\"canonical_hash_algorithm\":\"fnv1a64_sorted_id_distance_bits\""
      << ",\"canonical_hash\":\"" << hex64(result.canonical_fnv1a64) << "\""
      << ",\"end_to_end_wall_ms\":" << result.wall_ms
      << ",\"gpu_event_ms\":" << result.gpu_ms
      << ",\"cuda_status\":\"success\""
      << ",\"buffered_result_count\":" << result.buffered_result_count
      << ",\"update_result_capacity_slots\":" << result.update_result_capacity_slots
      << ",\"total_result_capacity_slots\":" << result.total_result_capacity_slots
      << ",\"capacity_snapshot_stage\":\"" << result.capacity_snapshot_stage << "\""
      << ",\"allocation_delta\":{\"cudaMalloc_calls\":" << a.malloc_calls
      << ",\"cudaMallocManaged_calls\":" << a.managed_calls
      << ",\"cudaFree_calls\":" << a.free_calls
      << ",\"cudaMalloc_requested_bytes\":" << a.requested_malloc_bytes
      << ",\"cudaMallocManaged_requested_bytes\":" << a.requested_managed_bytes
      << ",\"failed_calls\":" << a.failed_calls
      << ",\"api_host_ms\":" << a.api_host_ms << "}}\n";
}

std::string cuda_uuid_string(const cudaUUID_t& uuid) {
  const auto* b = reinterpret_cast<const unsigned char*>(uuid.bytes);
  bool any_nonzero = false;
  for (int i = 0; i < 16; ++i) any_nonzero = any_nonzero || b[i] != 0;
  if (!any_nonzero) die("CUDA UUID unavailable");
  char text[sizeof("GPU-CONFIGURE-ARCHIVE-DEVICE")] = {};
  const int written = std::snprintf(
      text, sizeof(text),
      "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      static_cast<unsigned int>(b[0]), static_cast<unsigned int>(b[1]),
      static_cast<unsigned int>(b[2]), static_cast<unsigned int>(b[3]),
      static_cast<unsigned int>(b[4]), static_cast<unsigned int>(b[5]),
      static_cast<unsigned int>(b[6]), static_cast<unsigned int>(b[7]),
      static_cast<unsigned int>(b[8]), static_cast<unsigned int>(b[9]),
      static_cast<unsigned int>(b[10]), static_cast<unsigned int>(b[11]),
      static_cast<unsigned int>(b[12]), static_cast<unsigned int>(b[13]),
      static_cast<unsigned int>(b[14]), static_cast<unsigned int>(b[15]));
  if (written != 40) die("CUDA UUID formatting failed");
  return std::string(text);
}

void write_run_card(const fs::path& out, const Config& cfg, int cuda_runtime_version,
                    int cuda_driver_version, const cudaDeviceProp& device_prop,
                    const std::string& pci_bus_id, const std::string& device_uuid) {
  std::ofstream f(out / "run_card.json");
  if (!f) die("cannot create run_card.json");
  f << "{\n"
    << "  \"schema\": \"gtspp-c1-microbench-run-card-v7\",\n"
    << "  \"run_mode\": \"" << run_mode(cfg) << "\",\n"
    << "  \"compiled_variant\": \"" << compiled_variant_name() << "\",\n"
    << "  \"requested_variant\": \"" << cfg.variant << "\",\n"
    << "  \"replicate\": " << cfg.replicate << ",\n"
    << "  \"base\": \"" << cfg.base.string() << "\",\n"
    << "  \"trace\": \"" << cfg.trace.string() << "\",\n"
    << "  \"radius\": " << cfg.radius << ",\n"
    << "  \"trace_limit\": " << cfg.trace_limit << ",\n"
    << "  \"profile_only\": " << (cfg.profile_only ? "true" : "false") << ",\n"
    << "  \"cuda_runtime_version\": " << cuda_runtime_version << ",\n"
    << "  \"cuda_driver_version\": " << cuda_driver_version << ",\n"
    << "  \"visible_cuda_device_name\": \"" << device_prop.name << "\",\n"
    << "  \"visible_cuda_device_pci_bus_id\": \"" << pci_bus_id << "\",\n"
    << "  \"visible_cuda_device_uuid\": \"" << device_uuid << "\",\n"
    << "  \"visible_cuda_device_major\": " << device_prop.major << ",\n"
    << "  \"visible_cuda_device_minor\": " << device_prop.minor << ",\n"
    << "  \"C1_PERSISTENT_WORKSPACE\": " << C1_PERSISTENT_WORKSPACE << ",\n"
    << "  \"C1_ONE_QUERY_FASTPATH\": " << C1_ONE_QUERY_FASTPATH << ",\n"
    << "  \"C2_residual_mode\": 0,\n"
    << "  \"timing_boundary\": \"wall: query-id write through final CUDA synchronization, result materialization, and ephemeral C1 free when enabled; GPU event: selected engine work before host-only release; canonical sorting/hash/file I/O excluded\"\n"
    << "}\n";
}

void upload_c2_mode_zero() {
  float h_alpha[32] = {0};
  float h_beta[32] = {0};
  float h_gamma[32];
  for (int l = 0; l < 32; ++l) h_gamma[l] = (l <= 2) ? 1.0f : 10.0f;
  upload_rp_constants(h_alpha, h_beta, h_gamma, tree_h, nullptr, nullptr, nullptr, 0, 0);
}

void setup_query_only_state() {
  // C1 is query-only: allocate only the immutable-query state needed by
  // searchIndexRnnUpdate.  In particular, do not initialize incremental-insert,
  // overflow, delete-prefix, or rebuild state; no C3 mutation API is reachable.
  CHECK(cudaMallocManaged((void **)&is_delete, data_info[1] * sizeof(int)));
  CHECK(cudaMallocManaged((void **)&qid_list, sizeof(int)));
  CHECK(cudaMemset(is_delete, 0, data_info[1] * sizeof(int)));
  in_size = 0;
  rnum[0] = 0;
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
}

QueryResult run_query(int query_id, c1c::Phase phase, cudaEvent_t event_start, cudaEvent_t event_stop) {
  c1c::set_phase(phase);
  const c1c::Snapshot alloc_before = c1c::snapshot(phase);
  const auto wall_begin = std::chrono::steady_clock::now();
  CHECK(cudaEventRecord(event_start));

  // This is the current update.cuh type-2 branch with the query-only invariant
  // in_size==0.  C2 is already uploaded in mode 0; no C3 mutation is allowed.
  qid_list[0] = query_id;
  rnum[0] = 0;
  searchIndexRnnUpdate(data_d, node_list, id_list, max_node_num, qid_list, 1, r, tree_h, data_info, empty_list,
                       qresult_count, qresult_count_prefix, result_id, result_dis, data_s, size_s);
  if (in_size != 0) die("query-only C1 runner observed nonzero insert buffer");
  if (rnum[0] != 0) die("query-only C1 runner observed nonzero buffered-result count");
  if (qresult_count == nullptr) die("query-only C1 runner observed null result-count workspace");
  if (qresult_count[0] < 0 || qresult_count[0] > update_result_ws_cap) {
    die("query-only C1 runner observed update-result capacity/overflow violation");
  }
  // rnum is required to remain zero in this query-only protocol, so the
  // search workspace already contains the complete result.  Do not enter the
  // update/delete merge path, which is C3/session machinery rather than C1.
  total_result_num = qresult_count[0];
  if (total_result_num < 0 || total_result_num > update_result_ws_cap) {
    die("query-only result count outside C1 workspace capacity");
  }
  total_result_id = result_id;
  total_result_dis = result_dis;
  CHECK(cudaEventRecord(event_stop));
  CHECK(cudaEventSynchronize(event_stop));
  float gpu_ms = 0.0f;
  CHECK(cudaEventElapsedTime(&gpu_ms, event_start, event_stop));
  CHECK(cudaGetLastError());

  // Copy result materialization required by the selected path before ephemeral
  // release.  Sorting/hash/file I/O happen only after the wall boundary.
  std::vector<CanonicalPair> raw;
  raw.reserve(static_cast<std::size_t>(total_result_num));
  for (int i = 0; i < total_result_num; ++i) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &total_result_dis[i], sizeof(bits));
    raw.push_back(CanonicalPair{static_cast<std::int32_t>(total_result_id[i]), bits});
  }

  // Capture capacity while the current query workspace still owns it.  In the
  // ephemeral variants releaseC1QueryWorkspace resets update_result_ws_cap to
  // zero, so no result/provenance field may read that global after release.
  const int c1_capacity_snapshot_pre_release = update_result_ws_cap;

#if !C1_PERSISTENT_WORKSPACE
  releaseC1QueryWorkspace(qresult_count, qresult_count_prefix, result_id, result_dis);
#endif
  CHECK(cudaGetLastError());
  const auto wall_end = std::chrono::steady_clock::now();
  const c1c::Snapshot alloc_after = c1c::snapshot(phase);

  std::sort(raw.begin(), raw.end(), [](const CanonicalPair& a, const CanonicalPair& b) {
    if (a.id != b.id) return a.id < b.id;
    return a.distance_bits < b.distance_bits;
  });
  QueryResult result;
  result.qid = query_id;
  result.count = total_result_num;
  result.canonical = std::move(raw);
  result.canonical_fnv1a64 = canonical_hash(result.canonical);
  result.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
  result.gpu_ms = gpu_ms;
  result.alloc_delta = c1c::subtract(alloc_after, alloc_before);
  result.buffered_result_count = rnum[0];
  // In the no-buffer/no-C3 path the update-result workspace is the total
  // result workspace.  Both fields must derive from the same pre-release
  // local snapshot, never from the resettable global workspace capacity.
  result.update_result_capacity_slots = c1_capacity_snapshot_pre_release;
  result.total_result_capacity_slots = c1_capacity_snapshot_pre_release;
  result.capacity_snapshot_stage = "pre_ephemeral_release";
  return result;
}

void run_phase(const char* phase_name, c1c::Phase phase, int count, const Config& cfg,
               cudaEvent_t event_start, cudaEvent_t event_stop) {
  const fs::path json_path = cfg.out / (std::string("phase_") + phase_name + "_ops.jsonl");
  const fs::path canonical_path = cfg.out / (std::string("phase_") + phase_name + "_canonical.bin");
  std::ofstream json_out(json_path);
  std::ofstream canonical_out(canonical_path, std::ios::binary);
  if (!json_out || !canonical_out) die("cannot create phase output under " + cfg.out.string());
  for (int i = 0; i < count; ++i) {
    if (update_list[i].update_flag != 2) die("trace contains non-type-2 operation in C1 runner");
    const QueryResult result = run_query(update_list[i].update_id, phase, event_start, event_stop);
    write_query_json(json_out, i, phase_name, result);
    write_canonical_record(canonical_out, i, result);
  }
  json_out.flush();
  canonical_out.flush();
}

void write_phase_counter(const fs::path& out) {
  c1c::write_json((out / "allocation_by_phase.json").string());
}

void write_completion(const fs::path& out, const Config& cfg, const TreeFingerprint& initial,
                      const TreeFingerprint& cold, const TreeFingerprint& provision,
                      const TreeFingerprint& warmup, const TreeFingerprint& steady) {
  std::ofstream f(out / "completion.json");
  const bool invariant = initial.fnv1a64 == cold.fnv1a64 && initial.fnv1a64 == provision.fnv1a64 &&
                         initial.fnv1a64 == warmup.fnv1a64 && initial.fnv1a64 == steady.fnv1a64;
  f << "{\n"
    << "  \"schema\": \"gtspp-c1-microbench-completion-v7\",\n"
    << "  \"run_mode\": \"" << run_mode(cfg) << "\",\n"
    << "  \"compiled_variant\": \"" << compiled_variant_name() << "\",\n"
    << "  \"tree_invariance_pass\": " << (invariant ? "true" : "false") << ",\n"
    << "  \"initial_tree_fingerprint\": \"" << hex64(initial.fnv1a64) << "\",\n"
    << "  \"cold_tree_fingerprint\": \"" << hex64(cold.fnv1a64) << "\",\n"
    << "  \"provision_tree_fingerprint\": \"" << hex64(provision.fnv1a64) << "\",\n"
    << "  \"warmup_tree_fingerprint\": \"" << hex64(warmup.fnv1a64) << "\",\n"
    << "  \"steady_tree_fingerprint\": \"" << hex64(steady.fnv1a64) << "\"\n"
    << "}\n";
  if (!invariant) die("tree invariance gate failed; output is invalid for C1 timing comparison");
}

}  // namespace

int main(int argc, char** argv) {
  const Config cfg = parse_args(argc, argv);
  if (cfg.variant != compiled_variant_name()) {
    die("requested variant does not match compile-time gates: requested=" + cfg.variant +
        " compiled=" + compiled_variant_name());
  }
  fs::create_directories(cfg.out);

  // GPU work begins only after all CLI/variant/output validation has passed.
  int cuda_runtime_version = 0;
  int cuda_driver_version = 0;
  cudaDeviceProp device_prop{};
  char pci_bus_id[32] = {0};
  CHECK(cudaRuntimeGetVersion(&cuda_runtime_version));
  CHECK(cudaDriverGetVersion(&cuda_driver_version));
  CHECK(cudaGetDeviceProperties(&device_prop, 0));
  CHECK(cudaDeviceGetPCIBusId(pci_bus_id, static_cast<int>(sizeof(pci_bus_id)), 0));
  const std::string visible_cuda_device_uuid = cuda_uuid_string(device_prop.uuid);
  write_run_card(cfg.out, cfg, cuda_runtime_version, cuda_driver_version, device_prop,
                 std::string(pci_bus_id), visible_cuda_device_uuid);
  file = const_cast<char*>(cfg.base.c_str());
  load(file, data_info, data_d, data_s, size_s);
  CHECK(cudaGetLastError());
  file_u = const_cast<char*>(cfg.trace.c_str());
  loadUpdate(file_u, update_list, update_num);
  CHECK(cudaGetLastError());
  if (update_num != 1024) die("trace must declare exactly 1024 operations");
  if (cfg.trace_limit > update_num) die("trace-limit exceeds loaded trace");
  for (int i = 0; i < cfg.trace_limit; ++i) {
    if (update_list[i].update_flag != 2) die("C1 trace is not query-only");
  }
  qnum = 1;
  r = cfg.radius;
  search_type = 1;

  AdvancedGPUTimer index_timer("C1 microbenchmark index construction");
  index_timer.start();
  indexConstru(data_d, data_s, size_s, data_info, id_list, node_list, max_node_num, tree_h, empty_list);
  CHECK(cudaGetLastError());
  index_timer.add_measurement();
  upload_c2_mode_zero();
  CHECK(cudaGetLastError());
  setup_query_only_state();
  CHECK(cudaGetLastError());

  c1c::reset_all();
  c1c::set_phase(c1c::Phase::kSetup);
  std::ofstream tree_out(cfg.out / "tree_fingerprints.jsonl");
  if (!tree_out) die("cannot create tree fingerprint output");
  const TreeFingerprint initial = tree_fingerprint();
  append_tree_fingerprint(tree_out, "after_build_and_session_setup", initial);

  cudaEvent_t event_start = nullptr;
  cudaEvent_t event_stop = nullptr;
  CHECK(cudaEventCreate(&event_start));
  CHECK(cudaEventCreate(&event_stop));

  const int first128 = std::min(128, cfg.trace_limit);
  run_phase("cold", c1c::Phase::kCold, first128, cfg, event_start, event_stop);
  const TreeFingerprint cold = tree_fingerprint();
  append_tree_fingerprint(tree_out, "after_cold", cold);

  run_phase("provision", c1c::Phase::kProvision, cfg.trace_limit, cfg, event_start, event_stop);
  const TreeFingerprint provision = tree_fingerprint();
  append_tree_fingerprint(tree_out, "after_provision", provision);

  run_phase("warmup", c1c::Phase::kWarmup, first128, cfg, event_start, event_stop);
  const TreeFingerprint warmup = tree_fingerprint();
  append_tree_fingerprint(tree_out, "after_warmup", warmup);

  // The sole primary latency distribution is phase_steady_ops.jsonl.
  run_phase("steady", c1c::Phase::kSteady, cfg.trace_limit, cfg, event_start, event_stop);
  const TreeFingerprint steady = tree_fingerprint();
  append_tree_fingerprint(tree_out, "after_steady", steady);
  tree_out.close();

  c1c::set_phase(c1c::Phase::kTeardown);
  releaseC1QueryWorkspace(qresult_count, qresult_count_prefix, result_id, result_dis);
  if (qid_list != nullptr) { CHECK(cudaFree(qid_list)); qid_list = nullptr; }
  if (is_delete != nullptr) { CHECK(cudaFree(is_delete)); is_delete = nullptr; }
  CHECK(cudaGetLastError());
  CHECK(cudaEventDestroy(event_start));
  CHECK(cudaEventDestroy(event_stop));
  write_phase_counter(cfg.out);
  write_completion(cfg.out, cfg, initial, cold, provision, warmup, steady);
  std::cout << "C1Microbench completed variant=" << compiled_variant_name()
            << " mode=" << run_mode(cfg) << " out=" << cfg.out << std::endl;
  return 0;
}
