// E1 frozen-base latency--quality trade-off timing runner.
//
// This new adapter includes the existing byte-bound Safe-C1/GTS core but never
// changes src/ or existing correctness/recall runners. Scope: frozen immutable
// base + external insertion-only delta; native query_knn versus the CURRENT
// exact query_range(UINT64_MAX) fallback; host steady_clock public-API timing.
//
// The APIs are deliberately NOT equivalent: query_knn returns KNN while range
// returns all immutable-base rows and this adapter truncates only after API
// timing. Artifacts therefore say not_same_api=true and
// timing_claim=latency_quality_tradeoff_only, never fair speedup.
//
// dry-run is CPU-only. timing is wired only for a future external guard.

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Exactly one GTS implementation image in this TU.  The copied source is
// byte-identical to v24; provenance/v24_unmodified_source_closure.sha256 binds
// it to its archived origin.
#include "../src/g3_safe_c1_native_matrix.cu"

namespace fair_latency_tradeoff {
namespace fs = std::filesystem;

using safe_c1_g3::DistanceSq;
using safe_c1_g3::EngineMode;
using safe_c1_g3::IssuedQuery;
using safe_c1_g3::NativeSafeC1Matrix;
using safe_c1_g3::StableDistance;
using safe_c1_g3::StableId;

constexpr std::array<char, 8> kTraceMagic{{'E', '1', 'G', 'T', 'R', 'C', '0', '1'}};
constexpr std::uint32_t kTraceVersion = 1;
constexpr char kProjectionSchema[] = "e1-frozen-base-knn-projection-manifest-v1";
constexpr char kMetadataSchema[] = "e1-frozen-base-knn-projection-bundle-v1";
constexpr char kAdmissionSchema[] = "e1-frozen-base-knn-projection-admission-v2";

enum class OpCode : std::uint8_t { kInsert = 1, kDelete = 2, kKnn = 3, kRange = 4 };

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
static_assert(sizeof(TraceHeader) == 48, "E1 trace header ABI changed");
static_assert(sizeof(TraceEvent) == 12, "E1 trace event ABI changed");
static_assert(!safe_c1_g3::G3_DIRECT_SIDECAR_RUNTIME_GUARD_OPEN,
              "delta-only runner refuses an open direct-sidecar core");

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("fair_latency_tradeoff fail-stop: " + message);
}
void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

struct Args {
  fs::path bundle;
  fs::path preflight;
  fs::path output;
  fs::path summary;
  std::string mode;
  int warmup_passes = -1;
  int measured_passes = -1;
  std::string timing_guard_nonce;
};

int parse_nonnegative_decimal(const std::string& text, const std::string& label) {
  require(!text.empty(), "empty " + label);
  for (const unsigned char ch : text) require(std::isdigit(ch), "nondecimal " + label);
  try {
    const unsigned long long value = std::stoull(text);
    require(value <= static_cast<unsigned long long>(std::numeric_limits<int>::max()),
            label + " exceeds int range");
    return static_cast<int>(value);
  } catch (const std::exception&) {
    fail("invalid " + label);
  }
}

Args parse_args(int argc, char** argv) {
  Args args;
  bool saw_warmup = false;
  bool saw_measured = false;
  for (int i = 1; i < argc; ++i) {
    const std::string flag(argv[i]);
    const auto path_value = [&]() -> fs::path {
      if (++i >= argc) fail("missing value after " + flag);
      return fs::path(argv[i]);
    };
    const auto text_value = [&]() -> std::string {
      if (++i >= argc) fail("missing value after " + flag);
      return std::string(argv[i]);
    };
    if (flag == "--bundle") args.bundle = path_value();
    else if (flag == "--preflight") args.preflight = path_value();
    else if (flag == "--out") args.output = path_value();
    else if (flag == "--summary") args.summary = path_value();
    else if (flag == "--mode") args.mode = text_value();
    else if (flag == "--warmup-passes") {
      require(!saw_warmup, "duplicate --warmup-passes");
      args.warmup_passes = parse_nonnegative_decimal(text_value(), "warmup passes");
      saw_warmup = true;
    } else if (flag == "--measured-passes") {
      require(!saw_measured, "duplicate --measured-passes");
      args.measured_passes = parse_nonnegative_decimal(text_value(), "measured passes");
      saw_measured = true;
    } else if (flag == "--timing-guard-nonce") {
      args.timing_guard_nonce = text_value();
    } else if (flag == "--help") {
      std::cout << "usage: fair_safe_c1_latency_tradeoff_e1_runner"
                   " --bundle DIR --preflight ADMISSION.env --out JSONL --summary JSON"
                   " --mode dry-run|timing [--warmup-passes N --measured-passes N"
                   " --timing-guard-nonce NONCE]\n";
      std::exit(0);
    } else {
      fail("unknown argument: " + flag);
    }
  }
  require(!args.bundle.empty() && !args.preflight.empty() && !args.output.empty() &&
              !args.summary.empty() && !args.mode.empty(),
          "--bundle, --preflight, --out, --summary, and --mode are required");
  require(args.mode == "dry-run" || args.mode == "timing", "mode must be dry-run or timing");
  if (args.mode == "dry-run") {
    require(!saw_warmup && !saw_measured && args.timing_guard_nonce.empty(),
            "dry-run accepts no timing-pass or guard-nonce options");
  } else {
    require(saw_warmup && saw_measured && args.warmup_passes >= 0 &&
                args.measured_passes >= 1 && args.warmup_passes <= 3 &&
                args.measured_passes <= 3,
            "timing requires explicit warmup [0,3] and measured [1,3] passes");
    require(!args.timing_guard_nonce.empty(), "timing requires a guard nonce");
    const char* nonce = std::getenv("SAFE_C1_LATENCY_TRADEOFF_GUARD_NONCE");
    const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
    require(nonce != nullptr && args.timing_guard_nonce == nonce,
            "timing requires matching guard-provided nonce");
    require(visible != nullptr && *visible != '\0',
            "timing requires guard-pinned CUDA_VISIBLE_DEVICES");
  }
  return args;
}

std::string read_text(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "cannot open " + path.string());
  std::ostringstream buffer;
  buffer << input.rdbuf();
  require(input.good() || input.eof(), "cannot read " + path.string());
  return buffer.str();
}

std::unordered_map<std::string, std::string> read_env(const fs::path& path) {
  std::unordered_map<std::string, std::string> values;
  std::istringstream input(read_text(path));
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    const std::size_t split = line.find('=');
    require(split != std::string::npos && split > 0, "malformed admission line");
    const std::string key = line.substr(0, split);
    const std::string value = line.substr(split + 1);
    require(!key.empty() && !value.empty(), "empty admission key/value");
    require(values.emplace(key, value).second, "duplicate admission key: " + key);
  }
  return values;
}

const std::string& required_env(const std::unordered_map<std::string, std::string>& env,
                                const std::string& key) {
  const auto found = env.find(key);
  if (found == env.end()) fail("admission omits " + key);
  return found->second;
}

std::optional<std::string> json_string_field(const std::string& text, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  std::size_t at = text.find(needle);
  if (at == std::string::npos) return std::nullopt;
  at = text.find(':', at + needle.size());
  if (at == std::string::npos) return std::nullopt;
  ++at;
  while (at < text.size() && std::isspace(static_cast<unsigned char>(text[at]))) ++at;
  if (at >= text.size() || text[at] != '"') return std::nullopt;
  ++at;
  std::string out;
  while (at < text.size() && text[at] != '"') {
    // The required schema string contains no escapes.  Rejecting an escaped
    // schema is safer than writing a partial JSON parser.
    if (text[at] == '\\') return std::nullopt;
    out.push_back(text[at++]);
  }
  if (at >= text.size()) return std::nullopt;
  return out;
}

std::string sha256_file(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "cannot hash missing file " + path.string());
  safe_c1_g3::Sha256 digest;
  std::array<char, 1 << 16> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize got = input.gcount();
    if (got > 0) digest.update(buffer.data(), static_cast<std::size_t>(got));
  }
  require(input.eof(), "hash read failed for " + path.string());
  return digest.final_hex();
}

std::string sha256_bytes(const void* address, std::size_t bytes) {
  safe_c1_g3::Sha256 digest;
  if (bytes != 0) {
    require(address != nullptr, "cannot hash null nonempty byte span");
    digest.update(address, bytes);
  }
  return digest.final_hex();
}

bool is_lower_sha256(const std::string& value) {
  if (value.size() != 64) return false;
  for (const unsigned char ch : value) {
    if (!(std::isdigit(ch) || (ch >= 'a' && ch <= 'f'))) return false;
  }
  return true;
}

std::uint64_t parse_unsigned_env(const std::unordered_map<std::string, std::string>& env,
                                 const std::string& key) {
  const std::string& value = required_env(env, key);
  require(!value.empty(), "empty admission integer " + key);
  for (const unsigned char ch : value) {
    require(std::isdigit(ch), "nondecimal admission integer " + key);
  }
  try {
    return static_cast<std::uint64_t>(std::stoull(value));
  } catch (const std::exception&) {
    fail("invalid admission integer " + key);
  }
}

int parse_int_env(const std::unordered_map<std::string, std::string>& env,
                  const std::string& key) {
  const std::uint64_t value = parse_unsigned_env(env, key);
  require(value <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
          "admission integer exceeds int range: " + key);
  return static_cast<int>(value);
}

struct AdmissionWitness {
  int dimension = 0;
  int base_n = 0;
  int pool_n = 0;
  int query_n = 0;
  int k = 0;
  int event_count = 0;
  int insert_count = 0;
  int knn_count = 0;
  std::uint64_t final_active_count = 0;
  std::string metadata_sha256;
  std::string trace_sha256;
  std::string projection_event_stream_sha256;
  std::string final_active_set_sha256;
};

void validate_output_target(const fs::path& target, const std::string& label) {
  require(!target.empty(), label + " path is empty");
  require(!fs::exists(target), label + " already exists; runner refuses overwrite");
  const fs::path parent = target.has_parent_path() ? target.parent_path() : fs::path(".");
  require(fs::is_directory(parent) && !fs::is_symlink(parent),
          label + " parent must be an existing non-symlink directory");
}

void validate_output_targets_before_cuda(const Args& args) {
  validate_output_target(args.output, "output");
  validate_output_target(args.summary, "summary");
  const fs::path output_absolute = fs::absolute(args.output).lexically_normal();
  const fs::path summary_absolute = fs::absolute(args.summary).lexically_normal();
  require(output_absolute != summary_absolute, "output and summary paths alias");
}

AdmissionWitness validate_preflight_and_bundle(const Args& args) {
  require(fs::is_directory(args.bundle) && !fs::is_symlink(args.bundle),
          "bundle path is not a real directory");
  require(fs::is_regular_file(args.preflight) && !fs::is_symlink(args.preflight),
          "preflight admission is missing or unsafe");
  const auto env = read_env(args.preflight);
  require(required_env(env, "schema") == kAdmissionSchema, "wrong admission schema");
  require(required_env(env, "status") == "PASS", "preflight admission is not PASS");
  require(required_env(env, "projection_manifest_schema") == kProjectionSchema,
          "admission manifest schema drift");
  require(required_env(env, "projection_metadata_schema") == kMetadataSchema,
          "admission metadata schema drift");
  require(required_env(env, "ops") == "insert,knn" &&
              required_env(env, "excluded_ops") == "delete,range",
          "admission operation policy drift");
  require(required_env(env, "base_immutable") == "true" &&
              required_env(env, "direct_sidecar_allowed") == "false" &&
              required_env(env, "legacy_routing_allowed") == "false",
          "admission Safe-C1 policy drift");
  require(fs::weakly_canonical(args.bundle).string() == required_env(env, "bundle_realpath"),
          "admission bundle path differs from supplied bundle");

  AdmissionWitness witness;
  witness.dimension = parse_int_env(env, "dimension");
  witness.base_n = parse_int_env(env, "base_n");
  witness.pool_n = parse_int_env(env, "pool_n");
  witness.query_n = parse_int_env(env, "query_n");
  witness.k = parse_int_env(env, "k");
  witness.event_count = parse_int_env(env, "event_count");
  witness.insert_count = parse_int_env(env, "insert_count");
  witness.knn_count = parse_int_env(env, "knn_count");
  witness.final_active_count = parse_unsigned_env(env, "final_active_count");
  require(parse_unsigned_env(env, "source_event_count") >
              static_cast<std::uint64_t>(witness.event_count),
          "source provenance event count is not larger than projection");
  witness.metadata_sha256 = required_env(env, "metadata_sha256");
  witness.trace_sha256 = required_env(env, "trace_sha256");
  witness.projection_event_stream_sha256 =
      required_env(env, "projection_event_stream_sha256");
  witness.final_active_set_sha256 = required_env(env, "final_active_set_sha256");
  const std::array<const std::string*, 11> admission_hashes{{
      &witness.metadata_sha256, &witness.trace_sha256,
      &witness.projection_event_stream_sha256, &witness.final_active_set_sha256,
      &required_env(env, "manifest_sha256"), &required_env(env, "pool_sha256"),
      &required_env(env, "queries_sha256"), &required_env(env, "mapping_sha256"),
      &required_env(env, "base_ids_sha256"), &required_env(env, "source_trace_sha256"),
      &required_env(env, "source_event_stream_sha256")}};
  for (const std::string* hash : admission_hashes) {
    require(is_lower_sha256(*hash), "admission contains malformed SHA-256");
  }

  const fs::path manifest = args.bundle / "manifest.json";
  const fs::path metadata = args.bundle / "metadata.json";
  require(fs::is_regular_file(manifest) && !fs::is_symlink(manifest),
          "bundle manifest.json is missing");
  require(fs::is_regular_file(metadata) && !fs::is_symlink(metadata),
          "bundle metadata.json is missing");
  const auto manifest_schema = json_string_field(read_text(manifest), "schema");
  const auto metadata_schema = json_string_field(read_text(metadata), "schema");
  require(manifest_schema.has_value() && *manifest_schema == kProjectionSchema,
          "manifest schema is not frozen-base KNN projection v1");
  require(metadata_schema.has_value() && *metadata_schema == kMetadataSchema,
          "metadata schema is not frozen-base KNN projection bundle v1");

  const std::array<std::pair<const char*, const char*>, 7> files{{
      {"manifest.json", "manifest_sha256"},
      {"metadata.json", "metadata_sha256"},
      {"pool.i16", "pool_sha256"},
      {"queries.i16", "queries_sha256"},
      {"stable_id_to_pool_row.i32", "mapping_sha256"},
      {"initial_base_stable_ids.i32", "base_ids_sha256"},
      {"trace.e1gtrc", "trace_sha256"},
  }};
  for (const auto& entry : files) {
    const fs::path file = args.bundle / entry.first;
    require(fs::is_regular_file(file) && !fs::is_symlink(file),
            "required non-symlink bundle file missing: " + file.string());
    require(sha256_file(file) == required_env(env, entry.second),
            "bundle bytes drifted after CPU preflight: " + file.string());
  }
  return witness;
}

template <typename T>
std::vector<T> read_exact(const fs::path& path, std::size_t count) {
  const std::uintmax_t expected_bytes =
      static_cast<std::uintmax_t>(count) * static_cast<std::uintmax_t>(sizeof(T));
  require(fs::file_size(path) == expected_bytes, "unexpected byte size: " + path.string());
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "cannot open " + path.string());
  std::vector<T> values(count);
  if (!values.empty()) {
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
  }
  require(input.good() || input.eof(), "truncated binary input: " + path.string());
  return values;
}

struct HostPool {
  int dimension = 0;
  std::vector<std::int16_t> values;
  std::vector<int> stable_to_pool_row;

  const std::int16_t* vector_for(StableId stable) const {
    require(stable >= 0 && stable < static_cast<int>(stable_to_pool_row.size()),
            "stable ID is outside mapping");
    const int row = stable_to_pool_row[static_cast<std::size_t>(stable)];
    require(row >= 0 && row * dimension + dimension <= static_cast<int>(values.size()),
            "mapping points outside pool");
    return values.data() + static_cast<std::size_t>(row) * dimension;
  }

  DistanceSq l2sq(StableId stable, const std::int16_t* query) const {
    const std::int16_t* value = vector_for(stable);
    DistanceSq total = 0;
    for (int dim = 0; dim < dimension; ++dim) {
      const std::int64_t delta = static_cast<std::int64_t>(value[dim]) -
                                 static_cast<std::int64_t>(query[dim]);
      total += static_cast<DistanceSq>(delta * delta);
    }
    return total;
  }
};

struct Bundle {
  TraceHeader header{};
  std::vector<TraceEvent> events;
  HostPool pool;
  std::vector<std::int16_t> queries;
  std::vector<StableId> base_ids;
  std::string event_stream_sha256;
};

Bundle load_bundle(const Args& args) {
  Bundle bundle;
  const fs::path trace_path = args.bundle / "trace.e1gtrc";
  {
    std::ifstream input(trace_path, std::ios::binary);
    require(static_cast<bool>(input), "cannot open trace");
    input.read(reinterpret_cast<char*>(&bundle.header), sizeof(bundle.header));
    require(input.gcount() == static_cast<std::streamsize>(sizeof(bundle.header)),
            "truncated trace header");
    require(std::memcmp(bundle.header.magic, kTraceMagic.data(), kTraceMagic.size()) == 0,
            "trace magic mismatch");
    require(bundle.header.version == kTraceVersion, "unsupported trace version");
    require(bundle.header.dimension > 0 && bundle.header.base_n > 0 &&
                bundle.header.pool_n >= bundle.header.base_n &&
                bundle.header.reservoir_n == bundle.header.pool_n - bundle.header.base_n &&
                bundle.header.query_n > 0 && bundle.header.k > 0,
            "invalid frozen-base trace header");
    bundle.events.resize(static_cast<std::size_t>(bundle.header.event_count));
    if (!bundle.events.empty()) {
      input.read(reinterpret_cast<char*>(bundle.events.data()),
                 static_cast<std::streamsize>(bundle.events.size() * sizeof(TraceEvent)));
    }
    require(input.gcount() == static_cast<std::streamsize>(bundle.events.size() * sizeof(TraceEvent)),
            "truncated trace events");
    char extra = 0;
    require(!input.read(&extra, 1), "trace has unexpected trailing bytes");
    bundle.event_stream_sha256 = sha256_bytes(
        bundle.events.empty() ? nullptr : static_cast<const void*>(bundle.events.data()),
        bundle.events.size() * sizeof(TraceEvent));
  }

  bundle.pool.dimension = static_cast<int>(bundle.header.dimension);
  bundle.pool.values = read_exact<std::int16_t>(
      args.bundle / "pool.i16",
      static_cast<std::size_t>(bundle.header.pool_n) * bundle.header.dimension);
  bundle.pool.stable_to_pool_row =
      read_exact<int>(args.bundle / "stable_id_to_pool_row.i32", bundle.header.pool_n);
  bundle.queries = read_exact<std::int16_t>(
      args.bundle / "queries.i16",
      static_cast<std::size_t>(bundle.header.query_n) * bundle.header.dimension);
  bundle.base_ids = read_exact<int>(args.bundle / "initial_base_stable_ids.i32",
                                    bundle.header.base_n);

  std::vector<unsigned char> physical_seen(bundle.header.pool_n, 0);
  std::vector<unsigned char> live(bundle.header.pool_n, 0);
  for (int row : bundle.pool.stable_to_pool_row) {
    require(row >= 0 && row < static_cast<int>(bundle.header.pool_n),
            "stableID->pool-row mapping leaves bundle bounds");
    require(physical_seen[static_cast<std::size_t>(row)] == 0,
            "stableID->pool-row mapping is not bijective");
    physical_seen[static_cast<std::size_t>(row)] = 1;
  }
  for (StableId stable : bundle.base_ids) {
    require(stable >= 0 && stable < static_cast<int>(bundle.header.pool_n),
            "initial base stable ID outside mapping");
    require(live[static_cast<std::size_t>(stable)] == 0, "duplicate initial base stable ID");
    live[static_cast<std::size_t>(stable)] = 1;
  }

  for (std::size_t index = 0; index < bundle.events.size(); ++index) {
    const TraceEvent& event = bundle.events[index];
    require(event.op_index == index, "trace event index is not contiguous");
    require(event.op_code == static_cast<std::uint8_t>(OpCode::kInsert) ||
                event.op_code == static_cast<std::uint8_t>(OpCode::kKnn),
            "projection trace contains delete/range/unknown operation");
    if (event.op_code == static_cast<std::uint8_t>(OpCode::kInsert)) {
      require(event.argument >= 0 && event.argument < static_cast<std::int32_t>(bundle.header.pool_n),
              "insert stable ID outside mapping");
      require(live[static_cast<std::size_t>(event.argument)] == 0,
              "projection insert repeats a base or prior delta stable ID");
      live[static_cast<std::size_t>(event.argument)] = 1;
    } else {
      require(event.argument >= 0 && event.argument < static_cast<std::int32_t>(bundle.header.query_n),
              "KNN query ID outside query matrix");
    }
  }
  return bundle;
}

// Trace state with exactly two tiers: immutable base and one global delta.
// It has no routing data structure and no delete/rebuild API by construction.
class DeltaOnlyState final {
 public:
  DeltaOnlyState(int stable_capacity, const std::vector<StableId>& base_ids)
      : active_(static_cast<std::size_t>(stable_capacity), 0),
        immutable_base_(static_cast<std::size_t>(stable_capacity), 0) {
    require(stable_capacity > 0, "invalid state capacity");
    for (StableId stable : base_ids) {
      require(stable >= 0 && stable < stable_capacity, "base ID outside state");
      require(active_[static_cast<std::size_t>(stable)] == 0, "duplicate base ID in state");
      active_[static_cast<std::size_t>(stable)] = 1;
      immutable_base_[static_cast<std::size_t>(stable)] = 1;
    }
  }

  void insert_to_global_delta(StableId stable) {
    require(stable >= 0 && stable < static_cast<int>(active_.size()), "delta insert outside state");
    require(active_[static_cast<std::size_t>(stable)] == 0, "delta insert targets live stable ID");
    require(immutable_base_[static_cast<std::size_t>(stable)] == 0,
            "attempt to insert an immutable base stable ID");
    active_[static_cast<std::size_t>(stable)] = 1;
    global_delta_.push_back(stable);
  }

  const std::vector<unsigned char>& active() const { return active_; }
  const std::vector<unsigned char>& immutable_base() const { return immutable_base_; }
  const std::vector<StableId>& global_delta() const { return global_delta_; }

  std::uint64_t active_count() const {
    std::uint64_t count = 0;
    for (unsigned char live : active_) count += live != 0 ? 1ULL : 0ULL;
    return count;
  }

  std::string active_set_sha256_lf() const {
    safe_c1_g3::Sha256 digest;
    for (StableId stable = 0; stable < static_cast<StableId>(active_.size()); ++stable) {
      if (active_[static_cast<std::size_t>(stable)] == 0) continue;
      const std::string line = std::to_string(stable) + "\n";
      digest.update(line);
    }
    return digest.final_hex();
  }

 private:
  std::vector<unsigned char> active_;
  std::vector<unsigned char> immutable_base_;
  std::vector<StableId> global_delta_;
};

std::vector<StableDistance> exact_topk(const HostPool& pool,
                                       const std::vector<unsigned char>& active,
                                       const std::int16_t* query, int k) {
  require(k > 0, "nonpositive k");
  std::vector<StableDistance> rows;
  for (StableId stable = 0; stable < static_cast<StableId>(active.size()); ++stable) {
    if (active[static_cast<std::size_t>(stable)] != 0) {
      rows.push_back(StableDistance{stable, pool.l2sq(stable, query)});
    }
  }
  std::sort(rows.begin(), rows.end(), safe_c1_g3::distance_then_stable);
  if (static_cast<int>(rows.size()) > k) rows.resize(static_cast<std::size_t>(k));
  return rows;
}

std::vector<StableDistance> merge_topk(std::vector<StableDistance> base,
                                       std::vector<StableDistance> delta, int k) {
  base.insert(base.end(), delta.begin(), delta.end());
  std::sort(base.begin(), base.end(), safe_c1_g3::distance_then_stable);
  for (std::size_t index = 1; index < base.size(); ++index) {
    require(base[index - 1].stable_id != base[index].stable_id,
            "base and global delta overlap in merge");
  }
  if (static_cast<int>(base.size()) > k) base.resize(static_cast<std::size_t>(k));
  return base;
}

std::size_t stable_overlap_count(const std::vector<StableDistance>& actual,
                                 const std::vector<StableDistance>& expected) {
  std::size_t count = 0;
  for (const StableDistance& row : actual) {
    for (const StableDistance& oracle : expected) {
      if (row.stable_id == oracle.stable_id) {
        ++count;
        break;
      }
    }
  }
  return count;
}

void require_equal(const std::vector<StableDistance>& actual,
                   const std::vector<StableDistance>& expected,
                   const std::string& label) {
  if (actual == expected) return;
  std::ostringstream out;
  out << label << " mismatch actual=" << actual.size() << " expected=" << expected.size();
  const std::size_t shared = std::min(actual.size(), expected.size());
  std::size_t first = 0;
  while (first < shared && actual[first] == expected[first]) ++first;
  out << " first_diff_index=" << first;
  if (first < actual.size()) {
    out << " actual=" << actual[first].stable_id << ":" << actual[first].distance_sq;
  }
  if (first < expected.size()) {
    out << " expected=" << expected[first].stable_id << ":" << expected[first].distance_sq;
  }
  fail(out.str());
}

void write_results(std::ostream& output, const std::vector<StableDistance>& rows) {
  output << '[';
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (i) output << ',';
    output << '[' << rows[i].stable_id << ',' << rows[i].distance_sq << ']';
  }
  output << ']';
}

void bootstrap_read_only_base(NativeSafeC1Matrix* engine, const Bundle& bundle,
                              const DeltaOnlyState& state) {
  require(engine != nullptr, "null engine");
  const std::int16_t* query0 = bundle.queries.data();
  const auto base_knn = exact_topk(bundle.pool, state.immutable_base(), query0,
                                   static_cast<int>(bundle.header.k));
  IssuedQuery first_knn = engine->query_knn(0, query0, static_cast<int>(bundle.header.k));
  require_equal(first_knn.export_data.results, base_knn, "native base bootstrap KNN");
  require(first_knn.post_rebuild_ticket.has_value(), "core omitted bootstrap KNN ticket");
  engine->verify_first_post_rebuild_query(std::move(*first_knn.post_rebuild_ticket), base_knn);

  // This private v24 admission gate is not a range service path.  It is
  // required only to move the unmodified core to Ready before E1 KNN replay.
  std::vector<StableDistance> base_all =
      exact_topk(bundle.pool, state.immutable_base(), query0,
                 static_cast<int>(bundle.header.base_n));
  IssuedQuery first_range =
      engine->query_range(0, query0, std::numeric_limits<DistanceSq>::max());
  require_equal(first_range.export_data.results, base_all, "private bootstrap range");
  require(first_range.post_rebuild_ticket.has_value(), "core omitted bootstrap range ticket");
  engine->verify_first_post_rebuild_query(std::move(*first_range.post_rebuild_ticket), base_all);
  require(engine->mode() == EngineMode::kReady, "core did not reach read-only Ready mode");
}


struct QueryCase {
  int ordinal = -1;
  std::uint32_t op_index = 0;
  int query_id = -1;
  std::vector<StableId> external_delta_ids;
  std::vector<StableDistance> exact_active_topk;
};

std::vector<StableDistance> exact_topk_for_ids(const HostPool& pool,
                                                const std::vector<StableId>& ids,
                                                const std::int16_t* query, int k) {
  std::vector<StableDistance> rows;
  rows.reserve(ids.size());
  for (StableId stable : ids) {
    require(stable >= 0 && stable < static_cast<int>(pool.stable_to_pool_row.size()),
            "external delta stable ID outside pool");
    rows.push_back(StableDistance{stable, pool.l2sq(stable, query)});
  }
  std::sort(rows.begin(), rows.end(), safe_c1_g3::distance_then_stable);
  for (std::size_t i = 1; i < rows.size(); ++i) {
    require(rows[i - 1].stable_id != rows[i].stable_id, "duplicate external delta ID");
  }
  if (static_cast<int>(rows.size()) > k) rows.resize(static_cast<std::size_t>(k));
  return rows;
}

std::vector<QueryCase> build_query_cases(const Bundle& bundle, const AdmissionWitness& admission) {
  DeltaOnlyState state(static_cast<int>(bundle.header.pool_n), bundle.base_ids);
  std::vector<QueryCase> cases;
  cases.reserve(static_cast<std::size_t>(admission.knn_count));
  int inserts = 0;
  for (const TraceEvent& event : bundle.events) {
    if (event.op_code == static_cast<std::uint8_t>(OpCode::kInsert)) {
      state.insert_to_global_delta(event.argument);
      ++inserts;
      continue;
    }
    require(event.op_code == static_cast<std::uint8_t>(OpCode::kKnn),
            "case builder observed nonprojected operation");
    const std::int16_t* query = bundle.queries.data() +
        static_cast<std::size_t>(event.argument) * bundle.pool.dimension;
    QueryCase item;
    item.ordinal = static_cast<int>(cases.size());
    item.op_index = event.op_index;
    item.query_id = event.argument;
    item.external_delta_ids = state.global_delta();
    item.exact_active_topk = exact_topk(bundle.pool, state.active(), query,
                                        static_cast<int>(bundle.header.k));
    cases.push_back(std::move(item));
  }
  require(inserts == admission.insert_count && static_cast<int>(cases.size()) == admission.knn_count,
          "case count drift from admission");
  require(state.active_count() == admission.final_active_count,
          "case builder final active count drift");
  require(state.active_set_sha256_lf() == admission.final_active_set_sha256,
          "case builder final active hash drift");
  return cases;
}

void json_string(std::ostream& output, const std::string& value) {
  output << '"';
  for (const unsigned char ch : value) {
    switch (ch) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (ch < 0x20U) {
          static const char digits[] = "0123456789abcdef";
          output << "\\u00" << digits[(ch >> 4U) & 0x0fU] << digits[ch & 0x0fU];
        } else {
          output << static_cast<char>(ch);
        }
    }
  }
  output << '"';
}

struct Observation {
  std::string condition;
  int pass = -1;
  int case_ordinal = -1;
  std::uint32_t op_index = 0;
  int query_id = -1;
  int external_delta_live = 0;
  std::uint64_t api_host_ns = 0;
  int api_result_count_before_adapter_truncation = 0;
  int base_candidate_count = 0;
  int visited_leaf_count = 0;
  int full_immutable_base_candidate_count = 0;
  bool exact_full_immutable_base_fallback = false;
  std::string base_path;
  std::vector<StableDistance> merged;
  std::string merged_sha256;
  std::size_t merged_overlap = 0;
  bool merged_exact = false;
};

using SteadyClock = std::chrono::steady_clock;

Observation measure_one(NativeSafeC1Matrix* engine, const Bundle& bundle,
                        const QueryCase& item, bool native, int pass) {
  require(engine != nullptr, "null timing engine");
  const std::int16_t* query = bundle.queries.data() +
      static_cast<std::size_t>(item.query_id) * bundle.pool.dimension;
  const int k = static_cast<int>(bundle.header.k);
  const auto t0 = SteadyClock::now();
  IssuedQuery issued = native
      ? engine->query_knn(item.query_id, query, k)
      : engine->query_range(item.query_id, query, std::numeric_limits<DistanceSq>::max());
  const auto t1 = SteadyClock::now();

  Observation output;
  output.condition = native ? "native_query_knn_candidate" : "exact_query_range_full_base_fallback";
  output.pass = pass;
  output.case_ordinal = item.ordinal;
  output.op_index = item.op_index;
  output.query_id = item.query_id;
  output.external_delta_live = static_cast<int>(item.external_delta_ids.size());
  output.api_host_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
  output.api_result_count_before_adapter_truncation = static_cast<int>(issued.export_data.results.size());
  output.base_candidate_count = static_cast<int>(issued.export_data.base_results.size());
  output.visited_leaf_count = static_cast<int>(issued.export_data.gts_visited_leaf_ids.size());
  output.full_immutable_base_candidate_count = issued.export_data.full_immutable_base_candidate_count;
  output.exact_full_immutable_base_fallback = issued.export_data.exact_full_immutable_base_range_fallback;
  output.base_path = issued.export_data.base_path;

  std::vector<StableDistance> base_topk = std::move(issued.export_data.results);
  if (native) {
    require(issued.export_data.kind == "knn" && !output.exact_full_immutable_base_fallback &&
                output.full_immutable_base_candidate_count == 0 &&
                static_cast<int>(base_topk.size()) == k && output.base_candidate_count >= k &&
                output.base_candidate_count <= static_cast<int>(bundle.header.base_n),
            "native timing call lost candidate-KNN contract");
  } else {
    require(issued.export_data.kind == "range" && output.exact_full_immutable_base_fallback &&
                output.full_immutable_base_candidate_count == static_cast<int>(bundle.header.base_n) &&
                output.base_path == "exact_full_immutable_base_range_fallback_no_gts_receipt" &&
                static_cast<int>(base_topk.size()) == static_cast<int>(bundle.header.base_n),
            "fallback timing call lost full-base range-fallback contract");
    // Deliberately after API timing: this is a wrapper-shaped KNN response, not
    // an attempt to make the two timed APIs equal.
    base_topk.resize(static_cast<std::size_t>(k));
  }
  // The exact external-delta scan/merge is intentionally outside api_host_ns.
  const std::vector<StableDistance> exact_delta =
      exact_topk_for_ids(bundle.pool, item.external_delta_ids, query, k);
  output.merged = merge_topk(std::move(base_topk), exact_delta, k);
  // Oracle/quality accounting and hashing are outside both time windows.
  output.merged_overlap = stable_overlap_count(output.merged, item.exact_active_topk);
  output.merged_exact = output.merged == item.exact_active_topk;
  if (!native) require(output.merged_exact, "exact fallback lost exact full-active top-k");
  output.merged_sha256 = safe_c1_g3::stable_distance_vector_sha256(
      output.merged, "fair-safe-c1-latency-tradeoff-e1-merged-result-v1");
  require(engine->mode() == EngineMode::kReady, "timing call drifted engine mode");
  return output;
}

void run_phase(NativeSafeC1Matrix* engine, const Bundle& bundle,
               const std::vector<QueryCase>& cases, int pass_count, bool retain,
               std::vector<Observation>* observations) {
  require(engine != nullptr && observations != nullptr, "null timing phase argument");
  for (int pass = 0; pass < pass_count; ++pass) {
    for (const QueryCase& item : cases) {
      // Adjacent cases alternate N,F,F,N; the next pass reverses that ordering.
      const bool native_first = ((pass + item.ordinal) % 2) == 0;
      Observation first = measure_one(engine, bundle, item, native_first, pass);
      Observation second = measure_one(engine, bundle, item, !native_first, pass);
      if (retain) {
        observations->push_back(std::move(first));
        observations->push_back(std::move(second));
      }
    }
  }
}

std::uint64_t percentile_ns(std::vector<std::uint64_t> values, int numerator, int denominator) {
  require(!values.empty() && numerator >= 0 && numerator <= denominator && denominator > 0,
          "invalid percentile input");
  std::sort(values.begin(), values.end());
  const std::size_t rank = static_cast<std::size_t>(
      (static_cast<unsigned long long>(numerator) * (values.size() - 1U)) /
      static_cast<unsigned long long>(denominator));
  return values[rank];
}

void write_distribution(std::ostream& output, const std::vector<Observation>& observations,
                        const std::string& condition) {
  std::vector<std::uint64_t> values;
  for (const Observation& row : observations) {
    if (row.condition == condition) values.push_back(row.api_host_ns);
  }
  require(!values.empty(), "empty timing distribution");
  unsigned long long sum = 0;
  for (const std::uint64_t value : values) sum += value;
  output << "{\"n\":" << values.size()
         << ",\"min_ns\":" << *std::min_element(values.begin(), values.end())
         << ",\"p50_ns\":" << percentile_ns(values, 50, 100)
         << ",\"p95_ns\":" << percentile_ns(values, 95, 100)
         << ",\"max_ns\":" << *std::max_element(values.begin(), values.end())
         << ",\"mean_ns\":" << (sum / values.size()) << '}';
}

void write_observation(std::ostream& output, const Observation& row) {
  output << "{\"schema\":\"fair-safe-c1-latency-tradeoff-e1-v1\""
         << ",\"record\":\"measurement\""
         << ",\"not_same_api\":true"
         << ",\"timing_claim\":\"latency_quality_tradeoff_only\""
         << ",\"condition\":";
  json_string(output, row.condition);
  output << ",\"pass\":" << row.pass
         << ",\"case_ordinal\":" << row.case_ordinal
         << ",\"op_index\":" << row.op_index
         << ",\"query_id\":" << row.query_id
         << ",\"external_global_delta_live\":" << row.external_delta_live
         << ",\"api_host_ns\":" << row.api_host_ns
         << ",\"post_api_external_delta_merge_excluded_from_api_host_ns\":true"
         << ",\"api_result_count_before_adapter_truncation\":"
         << row.api_result_count_before_adapter_truncation
         << ",\"base_candidate_count\":" << row.base_candidate_count
         << ",\"visited_leaf_count\":" << row.visited_leaf_count
         << ",\"full_immutable_base_candidate_count\":"
         << row.full_immutable_base_candidate_count
         << ",\"exact_full_immutable_base_fallback\":"
         << (row.exact_full_immutable_base_fallback ? "true" : "false")
         << ",\"base_path\":";
  json_string(output, row.base_path);
  output << ",\"merged_overlap_at_k\":" << row.merged_overlap
         << ",\"merged_exact_match\":" << (row.merged_exact ? "true" : "false")
         << ",\"merged_result_sha256\":";
  json_string(output, row.merged_sha256);
  output << ",\"merged\":";
  write_results(output, row.merged);
  output << "}\n";
}

void write_dry_run(const Args& args, const Bundle& bundle, const AdmissionWitness& admission,
                   const std::vector<QueryCase>& cases) {
  std::ofstream events(args.output, std::ios::out);
  require(static_cast<bool>(events), "cannot create dry-run output");
  events << "{\"schema\":\"fair-safe-c1-latency-tradeoff-e1-v1\""
         << ",\"record\":\"dry_run\""
         << ",\"gpu_used\":false"
         << ",\"not_same_api\":true"
         << ",\"timing_claim\":\"latency_quality_tradeoff_only\""
         << ",\"case_count\":" << cases.size()
         << ",\"event_count\":" << bundle.events.size()
         << ",\"insert_count\":" << admission.insert_count
         << ",\"knn_count\":" << admission.knn_count
         << "}\n";
  events.close();
  require(static_cast<bool>(events), "failed writing dry-run output");
  std::ofstream summary(args.summary, std::ios::out);
  require(static_cast<bool>(summary), "cannot create dry-run summary");
  summary << "{\"schema\":\"fair-safe-c1-latency-tradeoff-e1-v1\""
          << ",\"mode\":\"dry-run\""
          << ",\"status\":\"PASS_CPU_ONLY_PLAN\""
          << ",\"gpu_used\":false"
          << ",\"not_same_api\":true"
          << ",\"timing_claim\":\"latency_quality_tradeoff_only\""
          << ",\"scope\":\"frozen immutable base plus external insertion-only global delta; no CUDA engine instantiated\""
          << ",\"base_n\":" << bundle.header.base_n
          << ",\"dimension\":" << bundle.header.dimension
          << ",\"k\":" << bundle.header.k
          << ",\"query_cases\":" << cases.size()
          << ",\"final_active_count\":" << admission.final_active_count
          << "}\n";
  summary.close();
  require(static_cast<bool>(summary), "failed writing dry-run summary");
}

int run(const Args& args) {
  validate_output_targets_before_cuda(args);
  const AdmissionWitness admission = validate_preflight_and_bundle(args);
  Bundle bundle = load_bundle(args);
  require(static_cast<int>(bundle.header.dimension) == admission.dimension &&
              static_cast<int>(bundle.header.base_n) == admission.base_n &&
              static_cast<int>(bundle.header.pool_n) == admission.pool_n &&
              static_cast<int>(bundle.header.query_n) == admission.query_n &&
              static_cast<int>(bundle.header.k) == admission.k &&
              static_cast<int>(bundle.events.size()) == admission.event_count,
          "trace header/count drifted from admission");
  require(bundle.event_stream_sha256 == admission.projection_event_stream_sha256,
          "trace event stream drifted from admission");
  const std::vector<QueryCase> cases = build_query_cases(bundle, admission);
  if (args.mode == "dry-run") {
    write_dry_run(args, bundle, admission, cases);
    return 0;
  }

  // This point is reachable only after a future external guard passes a fresh
  // nonce and pins one CUDA-visible device. No current task launches this path.
  DeltaOnlyState base_state(static_cast<int>(bundle.header.pool_n), bundle.base_ids);
  NativeSafeC1Matrix engine(/*sidecar_leaf_capacity=*/1, static_cast<int>(bundle.header.k));
  engine.initialize_immutable_pool(bundle.pool.dimension, bundle.pool.values,
                                   bundle.pool.stable_to_pool_row);
  (void)engine.build_initial_base(bundle.base_ids);
  bootstrap_read_only_base(&engine, bundle, base_state);
  std::vector<Observation> observations;
  observations.reserve(static_cast<std::size_t>(args.measured_passes) * cases.size() * 2U);
  run_phase(&engine, bundle, cases, args.warmup_passes, false, &observations);
  run_phase(&engine, bundle, cases, args.measured_passes, true, &observations);
  require(engine.mode() == EngineMode::kReady, "engine mode drift after timing phase");

  std::ofstream events(args.output, std::ios::out);
  require(static_cast<bool>(events), "cannot create timing output");
  for (const Observation& row : observations) write_observation(events, row);
  events.close();
  require(static_cast<bool>(events), "failed writing timing output");
  std::size_t native_exact = 0, native_overlap = 0, fallback_exact = 0;
  for (const Observation& row : observations) {
    if (row.condition == "native_query_knn_candidate") {
      native_exact += row.merged_exact ? 1U : 0U;
      native_overlap += row.merged_overlap;
    } else {
      fallback_exact += row.merged_exact ? 1U : 0U;
    }
  }
  std::ofstream summary(args.summary, std::ios::out);
  require(static_cast<bool>(summary), "cannot create timing summary");
  summary << "{\"schema\":\"fair-safe-c1-latency-tradeoff-e1-v1\""
          << ",\"mode\":\"timing\""
          << ",\"status\":\"PASS_TRADEOFF_TIMING\""
          << ",\"not_same_api\":true"
          << ",\"timing_claim\":\"latency_quality_tradeoff_only\""
          << ",\"scope\":\"native query_knn candidate API versus current exact query_range(UINT64_MAX) full-result immutable-base fallback; not a fair same-API speedup\""
          << ",\"warmup_passes_discarded\":" << args.warmup_passes
          << ",\"measured_passes\":" << args.measured_passes
          << ",\"measurement_records\":" << observations.size()
          << ",\"post_api_external_delta_merge_excluded_from_api_host_ns\":true"
          << ",\"json_and_oracle_excluded_from_api_host_ns\":true"
          << ",\"native_api_host_ns\":";
  write_distribution(summary, observations, "native_query_knn_candidate");
  summary << ",\"fallback_api_host_ns\":";
  write_distribution(summary, observations, "exact_query_range_full_base_fallback");
  summary << ",\"quality\":{\"native_exact_set_count\":" << native_exact
          << ",\"native_overlap_sum\":" << native_overlap
          << ",\"fallback_exact_set_count\":" << fallback_exact
          << ",\"fallback_expected_exact_set_count\":"
          << (static_cast<std::size_t>(args.measured_passes) * cases.size()) << "}"
          << ",\"limitations\":[\"not_same_api\",\"no CUDA event/kernel timing\",\"external delta is CPU wrapper work\",\"no delete/range-workload/rebuild/direct-sidecar claim\"]"
          << "}\n";
  summary.close();
  require(static_cast<bool>(summary), "failed writing timing summary");
  return 0;
}

}  // namespace fair_latency_tradeoff

int main(int argc, char** argv) {
  try {
    return fair_latency_tradeoff::run(fair_latency_tradeoff::parse_args(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
