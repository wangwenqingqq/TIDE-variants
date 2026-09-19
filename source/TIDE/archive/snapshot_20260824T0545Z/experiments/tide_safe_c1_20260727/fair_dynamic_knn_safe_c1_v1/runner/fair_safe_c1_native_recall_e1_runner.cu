// Independent E1 frozen-base native-GTS candidate-recall probe.
//
// Quality-diagnostic scope only.  The frozen base is constructed once through
// the byte-identical v24 compact GTS core, and every trace insertion is retained
// only in DeltaOnlyState::global_delta_.  This wrapper never calls the core's
// insert(), erase_mutable(), or rebuild_from_current_live() APIs: no old routing
// certificate, sidecar, direct C3 path, or native tree mutation is part of the
// trace semantics.
//
// This probe executes the archived receipt-selected GTS candidate path for each
// projected KNN, rescoring only its selected base rows, then merges an exact
// global delta.  It independently records overlap against exhaustive int64
// canonical top-k.  It is deliberately a quality/recall diagnostic: it makes
// no exactness, fallback, traversal-completeness, or performance claim.

// --mode recall-probe is the only implemented mode.  --mode timing fails
// before any CUDA operation; a separately reviewed hot path is required before
// collecting any performance number.

#include <algorithm>
#include <array>
#include <cctype>
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

namespace fair_native_recall {
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
  throw std::runtime_error("fair_native_recall fail-stop: " + message);
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
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag(argv[i]);
    const auto value = [&]() -> fs::path {
      if (++i >= argc) fail("missing value after " + flag);
      return fs::path(argv[i]);
    };
    if (flag == "--bundle") args.bundle = value();
    else if (flag == "--preflight") args.preflight = value();
    else if (flag == "--out") args.output = value();
    else if (flag == "--summary") args.summary = value();
    else if (flag == "--mode") {
      if (++i >= argc) fail("missing value after --mode");
      args.mode = argv[i];
    } else if (flag == "--help") {
      std::cout << "usage: fair_safe_c1_native_recall_e1_runner"
                   " --bundle DIR --preflight ADMISSION.env --out JSONL --summary JSON"
                   " --mode recall-probe\n";
      std::exit(0);
    } else {
      fail("unknown argument: " + flag);
    }
  }
  require(!args.bundle.empty() && !args.preflight.empty() && !args.output.empty() &&
              !args.summary.empty() && !args.mode.empty(),
          "--bundle, --preflight, --out, --summary, and --mode are required");
  // Timing mode intentionally stops before constructing a CUDA engine.
  require(args.mode == "recall-probe",
          "only recall-probe is implemented; timing is refused without a separately admitted hot path");
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

int run(const Args& args) {
  // These checks happen before the v24 core can allocate CUDA state.
  validate_output_targets_before_cuda(args);
  const AdmissionWitness admission = validate_preflight_and_bundle(args);
  Bundle bundle = load_bundle(args);
  require(static_cast<int>(bundle.header.dimension) == admission.dimension &&
              static_cast<int>(bundle.header.base_n) == admission.base_n &&
              static_cast<int>(bundle.header.pool_n) == admission.pool_n &&
              static_cast<int>(bundle.header.query_n) == admission.query_n &&
              static_cast<int>(bundle.header.k) == admission.k &&
              static_cast<int>(bundle.events.size()) == admission.event_count,
          "trace header/count drifted from sealed admission");
  require(bundle.event_stream_sha256 == admission.projection_event_stream_sha256,
          "trace event-stream hash drifted from sealed admission");
  DeltaOnlyState state(static_cast<int>(bundle.header.pool_n), bundle.base_ids);

  // The immutable pool is duplicated only so the runner's independent CPU
  // oracle retains its stableID->pool-row view.  No trace insertion is passed
  // into this core instance.
  NativeSafeC1Matrix engine(/*sidecar_leaf_capacity=*/1, static_cast<int>(bundle.header.k));
  engine.initialize_immutable_pool(bundle.pool.dimension, bundle.pool.values,
                                   bundle.pool.stable_to_pool_row);
  (void)engine.build_initial_base(bundle.base_ids);
  bootstrap_read_only_base(&engine, bundle, state);

  std::ostringstream output;
  std::size_t knn_count = 0;
  std::size_t insert_count = 0;
  std::size_t base_exact_match_count = 0;
  std::size_t merged_exact_match_count = 0;
  std::size_t base_overlap_sum = 0;
  std::size_t merged_overlap_sum = 0;
  for (const TraceEvent& event : bundle.events) {
    if (event.op_code == static_cast<std::uint8_t>(OpCode::kInsert)) {
      state.insert_to_global_delta(event.argument);
      ++insert_count;
      output << "{\"record\":\"update\",\"op_index\":" << event.op_index
             << ",\"op\":\"insert\",\"stable_id\":" << event.argument
             << ",\"placement\":\"global_delta\",\"base_mutated\":false}\n";
      continue;
    }

    const std::int16_t* query =
        bundle.queries.data() + static_cast<std::size_t>(event.argument) * bundle.pool.dimension;
    // Candidate-restricted GTS path: record quality against the exhaustive
    // oracle rather than treating selected-leaf rescoring as a generic proof.
    IssuedQuery native =
        engine.query_knn(event.argument, query, static_cast<int>(bundle.header.k));
    require(!native.post_rebuild_ticket.has_value(),
            "unexpected post-rebuild ticket during recall probe");
    require(native.export_data.kind == "knn" &&
                !native.export_data.exact_full_immutable_base_range_fallback &&
                native.export_data.full_immutable_base_candidate_count == 0 &&
                static_cast<int>(native.export_data.results.size()) ==
                    static_cast<int>(bundle.header.k) &&
                native.export_data.base_results.size() >= bundle.header.k &&
                native.export_data.base_results.size() <= bundle.header.base_n,
            "native candidate probe received malformed base result");

    const auto expected_base =
        exact_topk(bundle.pool, state.immutable_base(), query, static_cast<int>(bundle.header.k));
    const std::size_t base_overlap = stable_overlap_count(native.export_data.results, expected_base);
    const bool base_exact_match = native.export_data.results == expected_base;

    const auto exact_delta =
        exact_topk(bundle.pool,
                   [&state, &bundle]() {
                     std::vector<unsigned char> only_delta(bundle.header.pool_n, 0);
                     for (StableId id : state.global_delta()) only_delta[static_cast<std::size_t>(id)] = 1;
                     return only_delta;
                   }(),
                   query, static_cast<int>(bundle.header.k));
    const auto merged = merge_topk(native.export_data.results, exact_delta,
                                   static_cast<int>(bundle.header.k));
    const auto expected_full =
        exact_topk(bundle.pool, state.active(), query, static_cast<int>(bundle.header.k));
    const std::size_t merged_overlap = stable_overlap_count(merged, expected_full);
    const bool merged_exact_match = merged == expected_full;
    require(engine.mode() == EngineMode::kReady, "read-only core mode drifted");
    base_exact_match_count += base_exact_match ? 1U : 0U;
    merged_exact_match_count += merged_exact_match ? 1U : 0U;
    base_overlap_sum += base_overlap;
    merged_overlap_sum += merged_overlap;
    ++knn_count;

    output << "{\"record\":\"knn\",\"op_index\":" << event.op_index
           << ",\"query_id\":" << event.argument
           << ",\"k\":" << bundle.header.k
           << ",\"native_base_candidate_path\":\"" << native.export_data.base_path << "\""
           << ",\"native_base_candidate_count\":" << native.export_data.base_results.size()
           << ",\"native_visited_leaf_count\":" << native.export_data.gts_visited_leaf_ids.size()
           << ",\"native_gts_candidate_path_used_for_trace\":true"
           << ",\"typed_exact_full_immutable_base_fallback_used_for_trace\":false"
           << ",\"base_immutable\":true,\"direct_sidecar_used\":false"
           << ",\"global_delta_live\":" << state.global_delta().size()
           << ",\"base_overlap_count\":" << base_overlap
           << ",\"merged_overlap_count\":" << merged_overlap
           << ",\"base_exact_match\":" << (base_exact_match ? "true" : "false")
           << ",\"merged_exact_match\":" << (merged_exact_match ? "true" : "false")
           << ",\"base_candidate_topk\":";
    write_results(output, native.export_data.results);
    output << ",\"merged\":";
    write_results(output, merged);
    output << "}\n";
  }

  require(static_cast<int>(insert_count) == admission.insert_count &&
              static_cast<int>(knn_count) == admission.knn_count,
          "replayed operation counts drifted from sealed admission");
  require(state.active_count() == admission.final_active_count,
          "replayed final active count drifted from sealed admission");
  const std::string final_active_hash = state.active_set_sha256_lf();
  require(final_active_hash == admission.final_active_set_sha256,
          "replayed final active LF hash drifted from sealed admission");

  // Out/summary were proven absent before CUDA initialization.  Emit only after
  // the entire trace and final-state witness have passed.
  {
    std::ofstream result_file(args.output, std::ios::out);
    require(static_cast<bool>(result_file), "cannot create output JSONL");
    result_file << output.str();
    result_file.close();
    require(static_cast<bool>(result_file), "failed writing output JSONL");
  }
  {
    std::ofstream summary(args.summary, std::ios::out);
    require(static_cast<bool>(summary), "cannot create summary");
    summary << "{\"schema\":\"fair-safe-c1-native-candidate-recall-probe-v1\""
            << ",\"mode\":\"recall_probe\""
            << ",\"event_count\":" << bundle.events.size()
            << ",\"insert_count\":" << insert_count
            << ",\"knn_count\":" << knn_count
            << ",\"global_delta_live\":" << state.global_delta().size()
            << ",\"final_active_count\":" << state.active_count()
            << ",\"final_active_set_sha256\":\"" << final_active_hash << "\""
            << ",\"metadata_sha256\":\"" << admission.metadata_sha256 << "\""
            << ",\"trace_sha256\":\"" << admission.trace_sha256 << "\""
            << ",\"projection_event_stream_sha256\":\""
            << admission.projection_event_stream_sha256 << "\""
            << ",\"base_exact_match_count\":" << base_exact_match_count
            << ",\"merged_exact_match_count\":" << merged_exact_match_count
            << ",\"base_overlap_sum\":" << base_overlap_sum
            << ",\"merged_overlap_sum\":" << merged_overlap_sum
            << ",\"base_immutable\":true"
            << ",\"native_gts_candidate_path_used_for_trace\":true"
            << ",\"typed_exact_full_immutable_base_fallback_used_for_trace\":false"
            << ",\"direct_sidecar_used\":false"
            << ",\"legacy_routing_used\":false"
            << ",\"timing_claim\":false"
            << ",\"status\":\"PASS_PROBE\"}\n";
    summary.close();
    require(static_cast<bool>(summary), "failed writing summary");
  }
  return 0;
}

}  // namespace fair_native_recall

int main(int argc, char** argv) {
  try {
    return fair_native_recall::run(fair_native_recall::parse_args(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}

