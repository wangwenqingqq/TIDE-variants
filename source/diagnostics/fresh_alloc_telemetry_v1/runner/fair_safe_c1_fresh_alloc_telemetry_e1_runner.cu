// E1 frozen-base Stage-0 fresh-allocation telemetry control.
//
// This isolated diagnostic preserves the current Safe-C1/GTS execution path:
// frozen immutable base plus external insertion-only delta; native query_knn
// and the existing exact full-base range fallback are deterministic
// conditions in an ABBA semantic control.  It records correctness and
// fresh-allocation witnesses only; it makes no performance conclusion.
//
// dry-run is CPU-only. semantic-control mode is callable only through the
// Stage-0 guard.

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <optional>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Exactly one isolated diagnostic GTS implementation image in this TU.
// Its reviewed Stage-0 telemetry diff and source closure are bound by the
// fresh-allocation provenance manifest; control v1b sources remain untouched.
#include "../src/g3_safe_c1_native_matrix_fresh_alloc_telemetry.cu"

namespace fresh_allocation_semantic_control {
namespace fs = std::filesystem;

using safe_c1_g3::DistanceSq;
using safe_c1_g3::EngineMode;
using safe_c1_g3::IssuedQuery;
using safe_c1_g3::NativeSafeC1Matrix;
using safe_c1_g3::StableDistance;
using safe_c1_g3::StableId;
using safe_c1_g3::LocalRow;
using safe_c1_g3::ReceiptBaseRow;
using safe_c1_g3::ReceiptLeafSpan;

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
  throw std::runtime_error("fresh_allocation_semantic_control fail-stop: " + message);
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
  fs::path witness;
  fs::path source_manifest;
  std::string run_id;
  std::string fresh_alloc_guard_nonce;
};

bool is_stage0_run_name(const std::string& value) {
  if (value.empty() || value.size() > 64U) return false;
  const auto ascii_alnum = [](unsigned char ch) {
    return (ch >= static_cast<unsigned char>('A') && ch <= static_cast<unsigned char>('Z')) ||
           (ch >= static_cast<unsigned char>('a') && ch <= static_cast<unsigned char>('z')) ||
           (ch >= static_cast<unsigned char>('0') && ch <= static_cast<unsigned char>('9'));
  };
  if (!ascii_alnum(static_cast<unsigned char>(value.front()))) return false;
  for (unsigned char ch : value) {
    if (!ascii_alnum(ch) && ch != static_cast<unsigned char>('_') &&
        ch != static_cast<unsigned char>('-')) return false;
  }
  return true;
}

bool is_lower_hex_nonce64(const std::string& value) {
  if (value.size() != 64U) return false;
  for (unsigned char ch : value) {
    if (!((ch >= static_cast<unsigned char>('0') && ch <= static_cast<unsigned char>('9')) ||
          (ch >= static_cast<unsigned char>('a') && ch <= static_cast<unsigned char>('f')))) {
      return false;
    }
  }
  return true;
}

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
  std::set<std::string> seen_flags;
  for (int i = 1; i < argc; ++i) {
    const std::string flag(argv[i]);
    if (flag != "--help") {
      require(seen_flags.insert(flag).second, "duplicate option: " + flag);
    }
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
    } else if (flag == "--witness") {
      args.witness = path_value();
    } else if (flag == "--source-manifest") {
      args.source_manifest = path_value();
    } else if (flag == "--run-id") {
      args.run_id = text_value();
    } else if (flag == "--fresh-alloc-guard-nonce") {
      args.fresh_alloc_guard_nonce = text_value();
    } else if (flag == "--help") {
      std::cout << "usage: fair_safe_c1_fresh_alloc_telemetry_e1_runner"
                   " --bundle DIR --preflight ADMISSION.env --out JSONL --summary JSON"
                   " --mode dry-run|semantic-control [--warmup-passes 1 --measured-passes 1"
                   " --witness JSONL --source-manifest JSON --run-id NAME"
                   " --fresh-alloc-guard-nonce NONCE]\n";
      std::exit(0);
    } else {
      fail("unknown argument: " + flag);
    }
  }
  require(!args.bundle.empty() && !args.preflight.empty() && !args.output.empty() &&
              !args.summary.empty() && !args.mode.empty(),
          "--bundle, --preflight, --out, --summary, and --mode are required");
  require(args.mode == "dry-run" || args.mode == "semantic-control",
          "mode must be dry-run or semantic-control");
  if (args.mode == "dry-run") {
    require(!saw_warmup && !saw_measured && args.witness.empty() &&
                args.source_manifest.empty() && args.run_id.empty() &&
                args.fresh_alloc_guard_nonce.empty(),
            "dry-run accepts no scheduled-pass or guard-nonce options");
  } else {
    require(saw_warmup && saw_measured && args.warmup_passes >= 0 &&
                args.measured_passes >= 1 && args.warmup_passes <= 3 &&
                args.measured_passes <= 3,
            "semantic-control requires explicit warmup [0,3] and measured [1,3] passes");
    require(args.warmup_passes == 1 && args.measured_passes == 1,
            "Stage-0 witness contract fixes one initialization and one semantic ABBA pass");
    const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
    const char* fresh_nonce = std::getenv("SAFE_C1_FRESH_ALLOC_TELEMETRY_GUARD_NONCE");
    require(!args.witness.empty() && !args.source_manifest.empty() && !args.run_id.empty() &&
                !args.fresh_alloc_guard_nonce.empty(),
            "semantic-control requires Stage-0 witness/manifest/run-id/fresh nonce");
    require(is_stage0_run_name(args.run_id),
            "semantic-control run-id must match the Stage-0 ASCII NAME_RE contract");
    require(is_lower_hex_nonce64(args.fresh_alloc_guard_nonce),
            "semantic-control fresh-allocation guard nonce must be 64 lowercase hex characters");
    require(fresh_nonce != nullptr && args.fresh_alloc_guard_nonce == fresh_nonce,
            "semantic-control requires matching fresh-allocation guard nonce");
    require(visible != nullptr && *visible != '\0',
            "semantic-control requires guard-pinned CUDA_VISIBLE_DEVICES");
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

// Stage-0 accepts only the generated JSON forms used by its frozen inputs.
// Escaped strings are deliberately outside this tiny parser's contract: an
// input containing one fails before CUDA rather than being partially parsed.
std::optional<std::size_t> unique_json_member_value_offset(
    const std::string& text, const std::string& key) {
  if (text.find('\\') != std::string::npos) return std::nullopt;
  std::optional<std::size_t> value_offset;
  for (std::size_t at = 0; at < text.size();) {
    if (text[at] != '"') {
      ++at;
      continue;
    }
    const std::size_t end = text.find('"', at + 1U);
    if (end == std::string::npos) return std::nullopt;
    std::size_t after = end + 1U;
    while (after < text.size() && std::isspace(static_cast<unsigned char>(text[after]))) ++after;
    if (text.substr(at + 1U, end - at - 1U) == key && after < text.size() && text[after] == ':') {
      if (value_offset.has_value()) return std::nullopt;  // duplicate JSON member
      ++after;
      while (after < text.size() && std::isspace(static_cast<unsigned char>(text[after]))) ++after;
      value_offset = after;
    }
    at = end + 1U;
  }
  return value_offset;
}

std::optional<std::string> json_string_field(const std::string& text, const std::string& key) {
  const std::optional<std::size_t> value_offset = unique_json_member_value_offset(text, key);
  if (!value_offset.has_value() || *value_offset >= text.size() || text[*value_offset] != '"') {
    return std::nullopt;
  }
  const std::size_t value_begin = *value_offset + 1U;
  const std::size_t value_end = text.find('"', value_begin);
  if (value_end == std::string::npos) return std::nullopt;
  return text.substr(value_begin, value_end - value_begin);
}

std::optional<std::size_t> json_object_end_without_escapes(
    const std::string& text, std::size_t object_begin) {
  if (object_begin >= text.size() || text[object_begin] != '{' ||
      text.find('\\') != std::string::npos) {
    return std::nullopt;
  }
  bool in_string = false;
  int depth = 0;
  for (std::size_t at = object_begin; at < text.size(); ++at) {
    const char ch = text[at];
    if (ch == '"') {
      in_string = !in_string;
      continue;
    }
    if (in_string) continue;
    if (ch == '{') {
      ++depth;
    } else if (ch == '}') {
      if (--depth == 0) return at;
      if (depth < 0) return std::nullopt;
    }
  }
  return std::nullopt;
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
  if (args.mode == "semantic-control") {
    validate_output_target(args.witness, "Stage-0 witness");
    const fs::path witness_absolute = fs::absolute(args.witness).lexically_normal();
    require(witness_absolute != output_absolute && witness_absolute != summary_absolute,
            "Stage-0 witness path aliases a legacy output");
    require(fs::is_regular_file(args.source_manifest) && !fs::is_symlink(args.source_manifest),
            "Stage-0 source manifest is missing or unsafe");
    const auto manifest_schema = json_string_field(read_text(args.source_manifest), "schema");
    require(manifest_schema.has_value() &&
                *manifest_schema == "fair-safe-c1-fresh-allocation-telemetry-source-manifest-v1",
            "Stage-0 source manifest schema drift");
  }
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


// A deliberately tiny JSON value tree for the Stage-0 sidecar only.  The
// existing engine.jsonl / summary.json serializers below remain byte-for-byte
// on their legacy v1b path.  This tree provides a language-independent record
// chain canonicalization matching the CPU-only validator.
struct WitnessJson {
  enum class Kind { kNull, kBool, kInt, kString, kArray, kObject };
  Kind kind = Kind::kNull;
  bool boolean = false;
  std::int64_t integer = 0;
  std::string text;
  std::vector<WitnessJson> array;
  std::vector<std::pair<std::string, WitnessJson>> object;
};

WitnessJson witness_null() { return WitnessJson{}; }
WitnessJson witness_bool(bool value) {
  WitnessJson output; output.kind = WitnessJson::Kind::kBool; output.boolean = value; return output;
}
WitnessJson witness_int(std::int64_t value) {
  WitnessJson output; output.kind = WitnessJson::Kind::kInt; output.integer = value; return output;
}
WitnessJson witness_uint(std::uint64_t value) {
  require(value <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
          "Stage-0 witness integer exceeds signed JSON contract");
  return witness_int(static_cast<std::int64_t>(value));
}
WitnessJson witness_string(std::string value) {
  WitnessJson output; output.kind = WitnessJson::Kind::kString; output.text = std::move(value); return output;
}
WitnessJson witness_array(std::vector<WitnessJson> value) {
  WitnessJson output; output.kind = WitnessJson::Kind::kArray; output.array = std::move(value); return output;
}
WitnessJson witness_object(std::vector<std::pair<std::string, WitnessJson>> value) {
  for (std::size_t i = 0; i < value.size(); ++i) {
    require(!value[i].first.empty(), "empty Stage-0 witness object key");
    for (std::size_t j = 0; j < i; ++j) {
      require(value[j].first != value[i].first, "duplicate Stage-0 witness object key");
    }
  }
  WitnessJson output; output.kind = WitnessJson::Kind::kObject; output.object = std::move(value); return output;
}
WitnessJson witness_object(std::initializer_list<std::pair<std::string, WitnessJson>> value) {
  return witness_object(std::vector<std::pair<std::string, WitnessJson>>(value));
}

std::string witness_canonical_value(const WitnessJson& value) {
  switch (value.kind) {
    case WitnessJson::Kind::kNull: return "N;";
    case WitnessJson::Kind::kBool: return value.boolean ? "B1;" : "B0;";
    case WitnessJson::Kind::kInt: return "I" + std::to_string(value.integer) + ";";
    case WitnessJson::Kind::kString:
      return "S" + std::to_string(value.text.size()) + ":" + value.text + ";";
    case WitnessJson::Kind::kArray: {
      std::string output = "A" + std::to_string(value.array.size()) + "[";
      for (const WitnessJson& item : value.array) output += witness_canonical_value(item);
      output += "]";
      return output;
    }
    case WitnessJson::Kind::kObject: {
      std::vector<const std::pair<std::string, WitnessJson>*> fields;
      fields.reserve(value.object.size());
      for (const auto& item : value.object) fields.push_back(&item);
      std::sort(fields.begin(), fields.end(), [](const auto* left, const auto* right) {
        return left->first < right->first;
      });
      std::string output = "O" + std::to_string(fields.size()) + "{";
      for (const auto* item : fields) {
        output += witness_canonical_value(witness_string(item->first));
        output += witness_canonical_value(item->second);
      }
      output += "}";
      return output;
    }
  }
  fail("unknown Stage-0 witness JSON kind");
}

void write_witness_json(std::ostream& output, const WitnessJson& value) {
  switch (value.kind) {
    case WitnessJson::Kind::kNull: output << "null"; return;
    case WitnessJson::Kind::kBool: output << (value.boolean ? "true" : "false"); return;
    case WitnessJson::Kind::kInt: output << value.integer; return;
    case WitnessJson::Kind::kString: json_string(output, value.text); return;
    case WitnessJson::Kind::kArray:
      output << '[';
      for (std::size_t i = 0; i < value.array.size(); ++i) {
        if (i) output << ',';
        write_witness_json(output, value.array[i]);
      }
      output << ']';
      return;
    case WitnessJson::Kind::kObject:
      output << '{';
      for (std::size_t i = 0; i < value.object.size(); ++i) {
        if (i) output << ',';
        json_string(output, value.object[i].first);
        output << ':';
        write_witness_json(output, value.object[i].second);
      }
      output << '}';
      return;
  }
  fail("unknown Stage-0 witness JSON kind");
}


struct FreshWitnessData {
  bool applicable = false;
  std::string reason;
  SafeC1FreshAllocTelemetry telemetry;
  int raw_receipt_leaf_pair_count = 0;
  int unique_receipt_leaf_count = 0;
  int id_list_capacity = 0;
  // Legacy raw res_ids are an attribution-only diagnostic.  These fields make
  // their observed completeness explicit without changing the receipt-defined
  // Safe-C1 candidate/result contract.
  int native_result_required_count = 0;
  bool native_result_complete = false;
  bool native_final_res_ids_cross_checked = false;
  std::string tree_payload_sha256;
  std::vector<int> visited_leaf_ids;
  std::vector<ReceiptLeafSpan> receipt_leaf_spans;
  std::vector<ReceiptBaseRow> candidate_rows;
  std::vector<SafeC1VisitedLeafPair> ordered_leaf_pairs;
  std::vector<LocalRow> native_result_ids;
  std::vector<float> native_result_distances;
  std::string exact_candidate_stable_distance_sha256;
  std::string api_result_stable_distance_sha256;
  std::string candidate_stable_set_sha256;
};


struct Observation {
  std::string condition;
  int pass = -1;
  int case_ordinal = -1;
  std::uint32_t op_index = 0;
  int query_id = -1;
  int external_delta_live = 0;
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
  std::string phase;
  int phase_pass = -1;
  int phase_slot = -1;
  FreshWitnessData fresh_witness;
};


std::uint32_t f32_bits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "float bit width drift");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::string f32_bits_hex(float value) {
  std::ostringstream output;
  output << std::hex << std::nouppercase << std::setw(8) << std::setfill('0') << f32_bits(value);
  return output.str();
}

std::string stage0_hash(const std::string& domain, const std::string& body) {
  return safe_c1_g3::sha256_text(domain + "\n" + body);
}

std::string witness_record_digest(const WitnessJson& unsealed_record) {
  return stage0_hash("fair-safe-c1-stage0/witness-record/v1",
                     witness_canonical_value(unsealed_record));
}

void witness_append_and_seal(std::ostream& output, WitnessJson record,
                             const WitnessJson& previous_digest,
                             std::string* digest_out) {
  require(record.kind == WitnessJson::Kind::kObject && digest_out != nullptr,
          "invalid Stage-0 witness record sealing input");
  record.object.emplace_back("prev_record_sha256", previous_digest);
  const std::string digest = witness_record_digest(record);
  record.object.emplace_back("record_sha256", witness_string(digest));
  write_witness_json(output, record);
  output << '\n';
  *digest_out = digest;
}

std::string hash_stack_events(const SafeC1FreshAllocTelemetry& telemetry) {
  std::ostringstream out;
  out << "count=" << telemetry.stack_events.size() << '\n';
  for (std::size_t i = 0; i < telemetry.stack_events.size(); ++i) {
    const auto& e = telemetry.stack_events[i];
    out << "i=" << i << ";kind=" << e.kind << ";qs=" << e.qs << ";qe=" << e.qe
        << ";level=" << e.cur_level << ";qnum_up=" << e.qnum_up
        << ";offset_n=" << e.offset_n << ";qs_up=" << e.qs_up
        << ";size_a=" << e.size_a << ";depth=" << e.depth_after << '\n';
  }
  return stage0_hash("fair-safe-c1-stage0/stack-schedule/v1", out.str());
}

std::string hash_size_list_writes(const SafeC1FreshAllocTelemetry& telemetry) {
  std::ostringstream out;
  out << "count=" << telemetry.size_list_writes.size() << '\n';
  for (std::size_t i = 0; i < telemetry.size_list_writes.size(); ++i) {
    const auto& w = telemetry.size_list_writes[i];
    out << "i=" << i << ";level=" << w.level << ";value=" << w.value << '\n';
  }
  return stage0_hash("fair-safe-c1-stage0/size-list-writes/v1", out.str());
}

std::string hash_traversal_steps(const SafeC1FreshAllocTelemetry& telemetry) {
  std::ostringstream out;
  out << "count=" << telemetry.traversal_steps.size() << '\n';
  for (std::size_t i = 0; i < telemetry.traversal_steps.size(); ++i) {
    const auto& e = telemetry.traversal_steps[i];
    out << "i=" << i << ";qs=" << e.qs << ";qe=" << e.qe
        << ";level=" << e.cur_level << ";qnum_up=" << e.qnum_up
        << ";offset_n=" << e.offset_n << ";qs_up=" << e.qs_up
        << ";size_a=" << e.size_a << ";qnum_l=" << e.qnum_l
        << ";offset_p=" << e.offset_p << ";offset_up_p=" << e.offset_up_p
        << ";nnum_l=" << e.nnum_l << ";pnum=" << e.pnum_level
        << ";pnum_total=" << e.pnum_level_total << ";lnum=" << e.leaf_lnum
        << ";receipt_stride=" << e.receipt_stride
        << ";update_before=" << (e.update_disk_before ? 1 : 0)
        << ";label=" << (e.used_label_cnode ? 1 : 0)
        << ";update_after=" << (e.update_disk_after ? 1 : 0) << '\n';
  }
  return stage0_hash("fair-safe-c1-stage0/traversal-steps/v1", out.str());
}

std::string hash_ordered_leaf_pairs(const std::vector<SafeC1VisitedLeafPair>& rows,
                                    const std::string& tree, int qnum) {
  std::ostringstream out;
  out << "tree=" << tree << "\nqnum=" << qnum << "\ncount=" << rows.size() << '\n';
  for (std::size_t i = 0; i < rows.size(); ++i) {
    out << "i=" << i << ";q=" << rows[i].query_id << ";leaf=" << rows[i].leaf_id << '\n';
  }
  return stage0_hash("fair-safe-c1-stage0/ordered-leaf-pairs/v1", out.str());
}

std::string hash_unique_leaf_set(const std::vector<int>& rows, const std::string& tree) {
  std::ostringstream out;
  out << "tree=" << tree << "\ncount=" << rows.size() << '\n';
  for (std::size_t i = 0; i < rows.size(); ++i) out << "i=" << i << ";leaf=" << rows[i] << '\n';
  return stage0_hash("fair-safe-c1-stage0/unique-leaf-set/v1", out.str());
}

std::string hash_receipt_spans(const std::vector<ReceiptLeafSpan>& rows,
                               const std::string& tree, int capacity) {
  std::ostringstream out;
  out << "tree=" << tree << "\nid_list_capacity=" << capacity << "\ncount=" << rows.size() << '\n';
  for (std::size_t i = 0; i < rows.size(); ++i) {
    out << "i=" << i << ";leaf=" << rows[i].leaf_id << ";lid=" << rows[i].id_list_lid
        << ";size=" << rows[i].size << '\n';
  }
  return stage0_hash("fair-safe-c1-stage0/receipt-spans/v1", out.str());
}

std::string hash_candidate_rows(const std::vector<ReceiptBaseRow>& rows,
                                const std::string& tree, int capacity) {
  std::ostringstream out;
  out << "tree=" << tree << "\nid_list_capacity=" << capacity << "\ncount=" << rows.size() << '\n';
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    out << "i=" << i << ";leaf=" << r.leaf_id << ";slot=" << r.id_list_slot
        << ";local=" << r.local_row << ";stable=" << r.stable_id << '\n';
  }
  return stage0_hash("fair-safe-c1-stage0/receipt-candidate-rows/v1", out.str());
}

std::string hash_candidate_stable_set(const std::vector<ReceiptBaseRow>& rows) {
  std::vector<StableId> stable;
  stable.reserve(rows.size());
  for (const auto& row : rows) stable.push_back(row.stable_id);
  std::sort(stable.begin(), stable.end());
  std::ostringstream out;
  out << "count=" << stable.size() << '\n';
  for (std::size_t i = 0; i < stable.size(); ++i) out << "i=" << i << ";stable=" << stable[i] << '\n';
  return stage0_hash("fair-safe-c1-stage0/candidate-stable-set/v1", out.str());
}

std::string hash_native_result_slots(const std::vector<LocalRow>& ids,
                                     const std::vector<float>& distances,
                                     int qnum, int k) {
  require(ids.size() == distances.size(), "native result diagnostic slot count mismatch");
  std::ostringstream out;
  out << "qnum=" << qnum << "\nk=" << k << "\ncount=" << ids.size() << '\n';
  for (std::size_t i = 0; i < ids.size(); ++i) {
    out << "i=" << i << ";local=" << ids[i] << ";distance_f32_bits="
        << f32_bits_hex(distances[i]) << '\n';
  }
  return stage0_hash("fair-safe-c1-stage0/native-result-slots/v1", out.str());
}


WitnessJson witness_stable_distance_rows(const std::vector<StableDistance>& rows) {
  std::vector<WitnessJson> output;
  output.reserve(rows.size());
  for (const StableDistance& row : rows) {
    output.push_back(witness_array({
        witness_int(static_cast<std::int64_t>(row.stable_id)),
        witness_uint(static_cast<std::uint64_t>(row.distance_sq)),
    }));
  }
  return witness_array(std::move(output));
}

WitnessJson witness_stack_events(const SafeC1FreshAllocTelemetry& telemetry) {
  std::vector<WitnessJson> output;
  output.reserve(telemetry.stack_events.size());
  for (std::size_t index = 0; index < telemetry.stack_events.size(); ++index) {
    const auto& event = telemetry.stack_events[index];
    output.push_back(witness_object({
        {"index", witness_uint(index)}, {"kind", witness_int(event.kind)},
        {"qs", witness_int(event.qs)}, {"qe", witness_int(event.qe)},
        {"cur_level", witness_int(event.cur_level)}, {"qnum_up", witness_int(event.qnum_up)},
        {"offset_n", witness_int(event.offset_n)}, {"qs_up", witness_int(event.qs_up)},
        {"size_a", witness_int(event.size_a)}, {"depth_after", witness_uint(event.depth_after)},
    }));
  }
  return witness_array(std::move(output));
}

WitnessJson witness_size_list_writes(const SafeC1FreshAllocTelemetry& telemetry) {
  std::vector<WitnessJson> output;
  output.reserve(telemetry.size_list_writes.size());
  for (std::size_t index = 0; index < telemetry.size_list_writes.size(); ++index) {
    const auto& write = telemetry.size_list_writes[index];
    output.push_back(witness_object({
        {"index", witness_uint(index)}, {"level", witness_int(write.level)},
        {"value", witness_int(write.value)},
    }));
  }
  return witness_array(std::move(output));
}

WitnessJson witness_traversal_steps(const SafeC1FreshAllocTelemetry& telemetry) {
  std::vector<WitnessJson> output;
  output.reserve(telemetry.traversal_steps.size());
  for (std::size_t index = 0; index < telemetry.traversal_steps.size(); ++index) {
    const auto& step = telemetry.traversal_steps[index];
    output.push_back(witness_object({
        {"index", witness_uint(index)}, {"qs", witness_int(step.qs)}, {"qe", witness_int(step.qe)},
        {"cur_level", witness_int(step.cur_level)}, {"qnum_up", witness_int(step.qnum_up)},
        {"offset_n", witness_int(step.offset_n)}, {"qs_up", witness_int(step.qs_up)},
        {"size_a", witness_int(step.size_a)}, {"qnum_l", witness_int(step.qnum_l)},
        {"offset_p", witness_int(step.offset_p)}, {"offset_up_p", witness_int(step.offset_up_p)},
        {"nnum_l", witness_int(step.nnum_l)}, {"pnum_level", witness_int(step.pnum_level)},
        {"pnum_level_total", witness_int(step.pnum_level_total)},
        {"leaf_lnum", witness_int(step.leaf_lnum)}, {"receipt_stride", witness_int(step.receipt_stride)},
        {"update_disk_before", witness_bool(step.update_disk_before)},
        {"used_label_cnode", witness_bool(step.used_label_cnode)},
        {"update_disk_after", witness_bool(step.update_disk_after)},
    }));
  }
  return witness_array(std::move(output));
}

WitnessJson witness_ordered_leaf_pairs(const std::vector<SafeC1VisitedLeafPair>& rows) {
  std::vector<WitnessJson> output;
  output.reserve(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    output.push_back(witness_object({
        {"index", witness_uint(index)}, {"query_id", witness_int(rows[index].query_id)},
        {"leaf_id", witness_int(rows[index].leaf_id)},
    }));
  }
  return witness_array(std::move(output));
}

WitnessJson witness_leaf_ids(const std::vector<int>& rows) {
  std::vector<WitnessJson> output;
  output.reserve(rows.size());
  for (const int leaf : rows) output.push_back(witness_int(leaf));
  return witness_array(std::move(output));
}

WitnessJson witness_receipt_spans(const std::vector<ReceiptLeafSpan>& rows) {
  std::vector<WitnessJson> output;
  output.reserve(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    const auto& row = rows[index];
    output.push_back(witness_object({
        {"index", witness_uint(index)}, {"leaf_id", witness_int(row.leaf_id)},
        {"id_list_lid", witness_int(row.id_list_lid)}, {"size", witness_int(row.size)},
    }));
  }
  return witness_array(std::move(output));
}

WitnessJson witness_candidate_rows(const std::vector<ReceiptBaseRow>& rows) {
  std::vector<WitnessJson> output;
  output.reserve(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    const auto& row = rows[index];
    output.push_back(witness_object({
        {"index", witness_uint(index)}, {"leaf_id", witness_int(row.leaf_id)},
        {"id_list_slot", witness_int(row.id_list_slot)}, {"local_row", witness_int(row.local_row)},
        {"stable_id", witness_int(row.stable_id)},
    }));
  }
  return witness_array(std::move(output));
}

WitnessJson witness_native_result_slots(const std::vector<LocalRow>& ids,
                                        const std::vector<float>& distances) {
  require(ids.size() == distances.size(), "Stage-0 native result slot vector mismatch");
  std::vector<WitnessJson> output;
  output.reserve(ids.size());
  for (std::size_t index = 0; index < ids.size(); ++index) {
    output.push_back(witness_object({
        {"index", witness_uint(index)}, {"local_row", witness_int(ids[index])},
        {"distance_f32_bits", witness_string(f32_bits_hex(distances[index]))},
    }));
  }
  return witness_array(std::move(output));
}

const char* fresh_direct_slot_name(int slot) {
  switch (slot) {
    case SAFE_C1_FRESH_ALLOC_LOCAL_RESULT_IDS: return "local_result_ids";
    case SAFE_C1_FRESH_ALLOC_RES_DIS: return "res_dis";
    case SAFE_C1_FRESH_ALLOC_SIZE_LIST: return "size_list";
    case SAFE_C1_FRESH_ALLOC_DISK: return "disk";
    case SAFE_C1_FRESH_ALLOC_P_LIST_K: return "p_list_k";
  }
  fail("unknown Stage-0 direct CUDA allocation slot");
}

const char* fresh_direct_operation_name(int operation) {
  switch (operation) {
    case SAFE_C1_FRESH_ALLOC_DIRECT_ALLOC: return "alloc";
    case SAFE_C1_FRESH_ALLOC_DIRECT_FREE: return "free";
  }
  fail("unknown Stage-0 direct CUDA allocation operation");
}

void require_direct_event_ledger(const SafeC1FreshAllocTelemetry& t) {
  struct ExpectedEvent { int slot; int operation; std::uint64_t bytes; };
  const std::array<ExpectedEvent, 10> expected{{
      {SAFE_C1_FRESH_ALLOC_LOCAL_RESULT_IDS, SAFE_C1_FRESH_ALLOC_DIRECT_ALLOC,
       t.local_result_ids_requested_bytes},
      {SAFE_C1_FRESH_ALLOC_RES_DIS, SAFE_C1_FRESH_ALLOC_DIRECT_ALLOC, t.res_dis_requested_bytes},
      {SAFE_C1_FRESH_ALLOC_SIZE_LIST, SAFE_C1_FRESH_ALLOC_DIRECT_ALLOC, t.size_list_requested_bytes},
      {SAFE_C1_FRESH_ALLOC_DISK, SAFE_C1_FRESH_ALLOC_DIRECT_ALLOC, t.disk_requested_bytes},
      {SAFE_C1_FRESH_ALLOC_P_LIST_K, SAFE_C1_FRESH_ALLOC_DIRECT_ALLOC, t.p_list_requested_bytes},
      {SAFE_C1_FRESH_ALLOC_P_LIST_K, SAFE_C1_FRESH_ALLOC_DIRECT_FREE, t.p_list_requested_bytes},
      {SAFE_C1_FRESH_ALLOC_SIZE_LIST, SAFE_C1_FRESH_ALLOC_DIRECT_FREE, t.size_list_requested_bytes},
      {SAFE_C1_FRESH_ALLOC_DISK, SAFE_C1_FRESH_ALLOC_DIRECT_FREE, t.disk_requested_bytes},
      {SAFE_C1_FRESH_ALLOC_RES_DIS, SAFE_C1_FRESH_ALLOC_DIRECT_FREE, t.res_dis_requested_bytes},
      {SAFE_C1_FRESH_ALLOC_LOCAL_RESULT_IDS, SAFE_C1_FRESH_ALLOC_DIRECT_FREE,
       t.local_result_ids_requested_bytes},
  }};
  require(t.direct_allocation_events.size() == expected.size(),
          "Stage-0 direct CUDA event ledger cardinality");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const SafeC1FreshAllocDirectCudaEvent& actual = t.direct_allocation_events[index];
    require(actual.slot == expected[index].slot && actual.operation == expected[index].operation &&
                actual.requested_bytes == expected[index].bytes,
            "Stage-0 direct CUDA event ledger order/value drift");
  }
}

int direct_event_rank(const SafeC1FreshAllocTelemetry& t, int slot, int operation) {
  require_direct_event_ledger(t);
  int rank = 0;
  for (const SafeC1FreshAllocDirectCudaEvent& event : t.direct_allocation_events) {
    if (event.operation != operation) continue;
    ++rank;
    if (event.slot == slot) return rank;
  }
  fail("Stage-0 direct CUDA event slot absent");
}

WitnessJson witness_direct_events(const SafeC1FreshAllocTelemetry& t) {
  require_direct_event_ledger(t);
  std::vector<WitnessJson> output;
  output.reserve(t.direct_allocation_events.size());
  for (std::size_t index = 0; index < t.direct_allocation_events.size(); ++index) {
    const SafeC1FreshAllocDirectCudaEvent& event = t.direct_allocation_events[index];
    output.push_back(witness_object({
        {"index", witness_uint(index)}, {"slot", witness_string(fresh_direct_slot_name(event.slot))},
        {"operation", witness_string(fresh_direct_operation_name(event.operation))},
        {"requested_bytes", witness_uint(event.requested_bytes)},
    }));
  }
  return witness_array(std::move(output));
}

WitnessJson witness_fresh_alloc(const FreshWitnessData& witness, bool native) {
  if (!native) {
    return witness_object({
        {"applicable", witness_bool(false)},
        {"reason", witness_string("exact_full_immutable_base_range_fallback")},
        {"allocations", witness_array({})},
        {"direct_events", witness_array({})},
        {"entry_named_global_scratch_ptrs_null", witness_null()},
    });
  }
  const SafeC1FreshAllocTelemetry& t = witness.telemetry;
  require(t.entry_stack_empty && t.entry_named_global_scratch_ptrs_null &&
              t.receipt_cleared && !t.update_disk_at_header_entry &&
              t.normal_return && t.exit_stack_empty && t.res_dis_allocated &&
              t.size_list_allocated && t.disk_allocated && t.p_list_allocated &&
              t.p_list_freed && t.size_list_freed && t.disk_freed &&
              t.res_dis_freed_by_adapter && t.local_result_ids_allocated &&
              t.local_result_ids_freed_by_adapter,
          "incomplete Stage-0 named fresh-allocation lifecycle");
  require_direct_event_ledger(t);
  const auto allocation = [&t](std::string slot, std::string allocator, int direct_slot,
                               std::uint64_t bytes) {
    return witness_object({
        {"slot", witness_string(std::move(slot))}, {"allocator", witness_string(std::move(allocator))},
        {"requested_bytes", witness_uint(bytes)},
        {"alloc_call_index",
         witness_int(direct_event_rank(t, direct_slot, SAFE_C1_FRESH_ALLOC_DIRECT_ALLOC))},
        {"alloc_status", witness_string("success")},
        {"free_call_index",
         witness_int(direct_event_rank(t, direct_slot, SAFE_C1_FRESH_ALLOC_DIRECT_FREE))},
        {"free_status", witness_string("success")},
    });
  };
  return witness_object({
      {"applicable", witness_bool(true)},
      {"scope", witness_string("named_direct_cuda_allocation_lifecycle_v1")},
      {"allocation_epoch", witness_uint(t.allocation_epoch)},
      {"entry_named_global_scratch_ptrs_null",
       witness_bool(t.entry_named_global_scratch_ptrs_null)},
      {"direct_events", witness_direct_events(t)},
      {"entry", witness_object({
          {"stack_empty", witness_bool(t.entry_stack_empty)},
          {"receipt_cleared", witness_bool(t.receipt_cleared)},
          {"update_disk_set_false", witness_bool(!t.update_disk_at_header_entry)},
      })},
      {"memory", witness_object({
          {"qnum", witness_int(t.qnum)}, {"k", witness_int(t.k)},
          {"tree_height", witness_int(t.tree_h)},
          {"free_bytes_after_fixed_allocs", witness_uint(t.raw_cuda_mem_avail_bytes)},
          {"total_bytes", witness_uint(t.cuda_mem_total_bytes)},
          {"capacity_policy", witness_string("cap_4GiB_if_free_gt_8GiB_else_half")},
          {"policy_bytes", witness_uint(t.policy_available_bytes)},
          {"p_list_elements", witness_uint(t.p_list_elements)},
          {"p_list_requested_bytes", witness_uint(t.p_list_requested_bytes)},
      })},
      {"allocations", witness_array({
          allocation("local_result_ids", "cudaMallocManaged",
                     SAFE_C1_FRESH_ALLOC_LOCAL_RESULT_IDS, t.local_result_ids_requested_bytes),
          allocation("res_dis", "cudaMallocManaged",
                     SAFE_C1_FRESH_ALLOC_RES_DIS, t.res_dis_requested_bytes),
          allocation("size_list", "cudaMallocManaged",
                     SAFE_C1_FRESH_ALLOC_SIZE_LIST, t.size_list_requested_bytes),
          allocation("disk", "cudaMalloc",
                     SAFE_C1_FRESH_ALLOC_DISK, t.disk_requested_bytes),
          allocation("p_list_k", "cudaMalloc",
                     SAFE_C1_FRESH_ALLOC_P_LIST_K, t.p_list_requested_bytes),
      })},
      {"exit", witness_object({
          {"stack_empty", witness_bool(t.exit_stack_empty)},
          {"all_tracked_freed", witness_bool(true)},
      })},
  });
}

WitnessJson witness_traversal(const FreshWitnessData& witness, bool native) {
  if (!native) {
    return witness_object({
        {"stack_push_count", witness_int(0)}, {"stack_pop_count", witness_int(0)},
        {"max_stack_depth", witness_int(0)}, {"stack_schedule_sha256", witness_null()},
        {"size_list_write_count", witness_int(0)}, {"size_list_write_sha256", witness_null()},
        {"traversal_step_count", witness_int(0)}, {"traversal_steps_sha256", witness_null()},
        {"raw_leaf_pair_count", witness_int(0)}, {"ordered_leaf_pair_sha256", witness_null()},
        {"unique_leaf_count", witness_int(0)}, {"unique_leaf_set_sha256", witness_null()},
        {"id_list_capacity", witness_int(0)},
        {"receipt_span_count", witness_int(0)}, {"receipt_span_sha256", witness_null()},
        {"candidate_row_count", witness_int(0)}, {"ordered_candidate_row_sha256", witness_null()},
        {"candidate_stable_set_sha256", witness_null()},
        {"native_result_slot_count", witness_int(0)},
        {"native_result_non_sentinel_count", witness_int(0)},
        {"native_result_required_count", witness_null()},
        {"native_result_complete", witness_null()},
        {"native_result_slots_sha256", witness_null()},
        {"native_final_res_ids_cross_checked", witness_bool(false)},
        {"stack_events", witness_array({})}, {"size_list_writes", witness_array({})},
        {"traversal_steps", witness_array({})}, {"ordered_leaf_pairs", witness_array({})},
        {"visited_leaf_ids", witness_array({})}, {"receipt_leaf_spans", witness_array({})},
        {"candidate_rows", witness_array({})}, {"native_result_slots", witness_array({})},
    });
  }
  const SafeC1FreshAllocTelemetry& t = witness.telemetry;
  require(witness.raw_receipt_leaf_pair_count == static_cast<int>(witness.ordered_leaf_pairs.size()) &&
              witness.unique_receipt_leaf_count == static_cast<int>(witness.visited_leaf_ids.size()) &&
              witness.unique_receipt_leaf_count == static_cast<int>(witness.receipt_leaf_spans.size()) &&
              witness.native_result_required_count >= 0 &&
              static_cast<int>(witness.candidate_rows.size()) >= witness.native_result_required_count &&
              witness.native_result_ids.size() == witness.native_result_distances.size() &&
              t.stack_push_count == t.stack_pop_count && t.stack_push_count > 0 &&
              t.stack_events.size() == t.stack_push_count + t.stack_pop_count &&
              witness.native_final_res_ids_cross_checked,
          "inconsistent Stage-0 traversal telemetry shape");
  std::size_t non_sentinel = 0;
  for (const LocalRow row : witness.native_result_ids) if (row != -1) ++non_sentinel;
  require(witness.native_result_complete ==
              (static_cast<int>(non_sentinel) >= witness.native_result_required_count),
          "legacy native res_ids completeness diagnostic drift");
  const std::string stable_set = hash_candidate_stable_set(witness.candidate_rows);
  require(witness.candidate_stable_set_sha256 == stable_set,
          "candidate stable-set hash drift before witness serialization");
  return witness_object({
      {"stack_push_count", witness_uint(t.stack_push_count)},
      {"stack_pop_count", witness_uint(t.stack_pop_count)},
      {"max_stack_depth", witness_uint(t.max_stack_depth)},
      {"stack_schedule_sha256", witness_string(hash_stack_events(t))},
      {"size_list_write_count", witness_uint(t.size_list_writes.size())},
      {"size_list_write_sha256", witness_string(hash_size_list_writes(t))},
      {"traversal_step_count", witness_uint(t.traversal_steps.size())},
      {"traversal_steps_sha256", witness_string(hash_traversal_steps(t))},
      {"raw_leaf_pair_count", witness_int(witness.raw_receipt_leaf_pair_count)},
      {"ordered_leaf_pair_sha256",
       witness_string(hash_ordered_leaf_pairs(witness.ordered_leaf_pairs,
                                               witness.tree_payload_sha256, t.qnum))},
      {"unique_leaf_count", witness_int(witness.unique_receipt_leaf_count)},
      {"unique_leaf_set_sha256",
       witness_string(hash_unique_leaf_set(witness.visited_leaf_ids, witness.tree_payload_sha256))},
      {"id_list_capacity", witness_int(witness.id_list_capacity)},
      {"receipt_span_count", witness_uint(witness.receipt_leaf_spans.size())},
      {"receipt_span_sha256",
       witness_string(hash_receipt_spans(witness.receipt_leaf_spans,
                                          witness.tree_payload_sha256, witness.id_list_capacity))},
      {"candidate_row_count", witness_uint(witness.candidate_rows.size())},
      {"ordered_candidate_row_sha256",
       witness_string(hash_candidate_rows(witness.candidate_rows,
                                           witness.tree_payload_sha256, witness.id_list_capacity))},
      {"candidate_stable_set_sha256", witness_string(stable_set)},
      {"native_result_slot_count", witness_uint(witness.native_result_ids.size())},
      {"native_result_non_sentinel_count", witness_uint(non_sentinel)},
      {"native_result_required_count", witness_int(witness.native_result_required_count)},
      {"native_result_complete", witness_bool(witness.native_result_complete)},
      {"native_result_slots_sha256",
       witness_string(hash_native_result_slots(witness.native_result_ids,
                                                witness.native_result_distances, t.qnum, t.k))},
      {"native_final_res_ids_cross_checked",
       witness_bool(witness.native_final_res_ids_cross_checked)},
      {"stack_events", witness_stack_events(t)},
      {"size_list_writes", witness_size_list_writes(t)},
      {"traversal_steps", witness_traversal_steps(t)},
      {"ordered_leaf_pairs", witness_ordered_leaf_pairs(witness.ordered_leaf_pairs)},
      {"visited_leaf_ids", witness_leaf_ids(witness.visited_leaf_ids)},
      {"receipt_leaf_spans", witness_receipt_spans(witness.receipt_leaf_spans)},
      {"candidate_rows", witness_candidate_rows(witness.candidate_rows)},
      {"native_result_slots",
       witness_native_result_slots(witness.native_result_ids, witness.native_result_distances)},
  });
}

WitnessJson witness_results_block(const Observation& row) {
  require(!row.fresh_witness.exact_candidate_stable_distance_sha256.empty() &&
              !row.fresh_witness.api_result_stable_distance_sha256.empty(),
          "missing Stage-0 result commitments");
  return witness_object({
      {"base_candidate_count", witness_int(row.base_candidate_count)},
      {"exact_candidate_stable_distance_sha256",
       witness_string(row.fresh_witness.exact_candidate_stable_distance_sha256)},
      {"api_result_count_before_adapter_truncation",
       witness_int(row.api_result_count_before_adapter_truncation)},
      {"api_result_stable_distance_sha256",
       witness_string(row.fresh_witness.api_result_stable_distance_sha256)},
      {"merged", witness_stable_distance_rows(row.merged)},
      {"merged_result_sha256", witness_string(row.merged_sha256)},
      {"merged_overlap_at_k", witness_uint(row.merged_overlap)},
      {"merged_exact_match", witness_bool(row.merged_exact)},
  });
}

std::string manifest_field_or_fail(const fs::path& manifest, const std::string& key) {
  const std::optional<std::string> value = json_string_field(read_text(manifest), key);
  require(value.has_value() && is_lower_sha256(*value), "invalid Stage-0 manifest SHA field: " + key);
  return *value;
}

std::string manifest_artifact_sha_or_fail(const fs::path& manifest, const std::string& artifact) {
  const std::string text = read_text(manifest);
  const std::optional<std::size_t> object_begin =
      unique_json_member_value_offset(text, artifact);
  require(object_begin.has_value() && *object_begin < text.size() && text[*object_begin] == '{',
          "source manifest omits or duplicates artifact object: " + artifact);
  const std::optional<std::size_t> object_end =
      json_object_end_without_escapes(text, *object_begin);
  require(object_end.has_value(), "unterminated source manifest artifact: " + artifact);
  const std::optional<std::string> value = json_string_field(
      text.substr(*object_begin, *object_end - *object_begin + 1U), "sha256");
  require(value.has_value() && is_lower_sha256(*value),
          "source manifest artifact SHA malformed: " + artifact);
  return *value;
}

WitnessJson witness_bundle_hashes(const fs::path& bundle) {
  static const std::array<const char*, 7> kNames{{
      "manifest.json", "metadata.json", "trace.e1gtrc", "pool.i16", "queries.i16",
      "stable_id_to_pool_row.i32", "initial_base_stable_ids.i32",
  }};
  std::vector<std::pair<std::string, WitnessJson>> output;
  output.reserve(kNames.size());
  for (const char* name : kNames) {
    output.emplace_back(name, witness_string(sha256_file(bundle / name)));
  }
  return witness_object(std::move(output));
}

WitnessJson witness_hash_contract() {
  return witness_object({
      {"ordered_leaf_pair", witness_string("fair-safe-c1-stage0/ordered-leaf-pairs/v1")},
      {"unique_leaf_set", witness_string("fair-safe-c1-stage0/unique-leaf-set/v1")},
      {"receipt_span", witness_string("fair-safe-c1-stage0/receipt-spans/v1")},
      {"ordered_candidate_row", witness_string("fair-safe-c1-stage0/receipt-candidate-rows/v1")},
      {"candidate_stable_set", witness_string("fair-safe-c1-stage0/candidate-stable-set/v1")},
      {"native_result_slot", witness_string("fair-safe-c1-stage0/native-result-slots/v1")},
      {"stack_schedule", witness_string("fair-safe-c1-stage0/stack-schedule/v1")},
      {"size_list_write", witness_string("fair-safe-c1-stage0/size-list-writes/v1")},
      {"traversal_steps", witness_string("fair-safe-c1-stage0/traversal-steps/v1")},
      {"exact_candidate_stable_distance",
       witness_string("fair-safe-c1-stage0/exact-candidate-stable-distance/v1")},
      {"api_result_stable_distance",
       witness_string("fair-safe-c1-stage0/api-result-stable-distance/v1")},
  });
}

WitnessJson witness_run_start(const Args& args, const Bundle& bundle,
                              const std::vector<QueryCase>& cases,
                              const std::string& tree_payload_sha256) {
  const fs::path bundle_path = fs::canonical(args.bundle);
  const std::string manifest_sha = sha256_file(args.source_manifest);
  const std::string closure_sha = manifest_field_or_fail(args.source_manifest,
                                                          "full_source_closure_sha256");
  const std::string runner_sha = manifest_artifact_sha_or_fail(args.source_manifest, "runner_source");
  const std::string binary_sha = sha256_file(fs::read_symlink("/proc/self/exe"));
  return witness_object({
      {"schema", witness_string("fair-safe-c1-stage0-fresh-allocation-witness-v1")},
      {"record", witness_string("run_start")},
      {"diagnostic_variant", witness_string("fresh_allocation_telemetry_control")},
      {"publication_eligible", witness_bool(false)},
      {"run_id", witness_string(args.run_id)},
      {"source_manifest_sha256", witness_string(manifest_sha)},
      {"artifact_hashes", witness_object({
          {"source_closure_sha256", witness_string(closure_sha)},
          {"runner_sha256", witness_string(runner_sha)},
          {"binary_sha256", witness_string(binary_sha)},
      })},
      {"control_engine_schema", witness_string("fresh-allocation-telemetry-e1-v1")},
      {"bundle_path", witness_string(bundle_path.string())},
      {"bundle_input_hashes", witness_bundle_hashes(bundle_path)},
      {"admission_sha256", witness_string(sha256_file(args.preflight))},
      {"tree_payload_sha256", witness_string(tree_payload_sha256)},
      {"schedule", witness_object({
          {"initialization_passes", witness_int(args.warmup_passes)},
          {"semantic_passes", witness_int(args.measured_passes)},
          {"case_count", witness_uint(cases.size())},
          {"query_witness_count", witness_uint(cases.size() * 4U)},
          {"native_query_count", witness_uint(cases.size() * 2U)},
          {"fallback_query_count", witness_uint(cases.size() * 2U)},
          {"ordering",
           witness_string("ABBA_native_first_if_(global_phase_pass_plus_case_ordinal)_mod_2_is_0")},
      })},
      {"hash_contract", witness_hash_contract()},
      {"record_index", witness_int(0)},
  });
}

WitnessJson witness_query_record(const Observation& row, std::uint64_t record_index,
                                 const std::string& tree_payload_sha256) {
  const bool native = row.condition == "native_query_knn_candidate";
  require(native || row.condition == "exact_query_range_full_base_fallback",
          "unknown Stage-0 witness condition");
  require(row.phase == "initialization" || row.phase == "semantic",
          "unknown Stage-0 witness phase");
  require(((row.phase == "initialization" && row.phase_pass == 0) ||
           (row.phase == "semantic" && row.phase_pass == 1)) &&
              row.phase_slot >= 0 && row.fresh_witness.applicable == native &&
              row.fresh_witness.tree_payload_sha256 == tree_payload_sha256,
          "incomplete Stage-0 query witness binding");
  return witness_object({
      {"schema", witness_string("fair-safe-c1-stage0-fresh-allocation-witness-v1")},
      {"record", witness_string("query_witness")},
      {"diagnostic_variant", witness_string("fresh_allocation_telemetry_control")},
      {"publication_eligible", witness_bool(false)},
      {"record_index", witness_uint(record_index)},
      {"phase", witness_string(row.phase)},
      {"phase_pass", witness_int(row.phase_pass)},
      {"phase_slot", witness_int(row.phase_slot)},
      {"condition", witness_string(row.condition)},
      {"case_ordinal", witness_int(row.case_ordinal)},
      {"op_index", witness_uint(row.op_index)},
      {"query_id", witness_int(row.query_id)},
      {"external_global_delta_live", witness_int(row.external_delta_live)},
      {"tree_payload_sha256", witness_string(tree_payload_sha256)},
      {"base_path", witness_string(row.base_path)},
      {"fresh_alloc", witness_fresh_alloc(row.fresh_witness, native)},
      {"traversal", witness_traversal(row.fresh_witness, native)},
      {"results", witness_results_block(row)},
      {"query_status", witness_string("PASS")},
      {"engine_mode_before", witness_string("ready")},
      {"engine_mode_after", witness_string("ready")},
      {"exception_stage", witness_null()},
  });
}

void write_fresh_allocation_witness(const Args& args, const Bundle& bundle,
                                    const std::vector<QueryCase>& cases,
                                    const std::vector<Observation>& witnesses) {
  require(args.mode == "semantic-control" && witnesses.size() == cases.size() * 4U &&
              !witnesses.empty(), "Stage-0 witness cardinality");
  const std::string tree_payload_sha256 = witnesses.front().fresh_witness.tree_payload_sha256;
  require(is_lower_sha256(tree_payload_sha256), "first Stage-0 witness lacks tree SHA");
  for (const Observation& row : witnesses) {
    require(row.fresh_witness.tree_payload_sha256 == tree_payload_sha256,
            "tree payload drift across Stage-0 witness schedule");
  }
  std::ofstream output(args.witness, std::ios::out);
  require(static_cast<bool>(output), "cannot create Stage-0 fresh-allocation witness");
  std::string previous;
  witness_append_and_seal(output, witness_run_start(args, bundle, cases, tree_payload_sha256),
                          witness_null(), &previous);
  std::size_t native_count = 0;
  std::size_t fallback_count = 0;
  std::uint64_t record_index = 1;
  for (const Observation& row : witnesses) {
    if (row.condition == "native_query_knn_candidate") ++native_count;
    else ++fallback_count;
    witness_append_and_seal(output, witness_query_record(row, record_index, tree_payload_sha256),
                            witness_string(previous), &previous);
    ++record_index;
  }
  require(native_count == cases.size() * 2U && fallback_count == cases.size() * 2U,
          "Stage-0 witness condition cardinality");
  witness_append_and_seal(output, witness_object({
      {"schema", witness_string("fair-safe-c1-stage0-fresh-allocation-witness-v1")},
      {"record", witness_string("run_end")},
      {"diagnostic_variant", witness_string("fresh_allocation_telemetry_control")},
      {"publication_eligible", witness_bool(false)},
      {"record_index", witness_uint(record_index)},
      {"status", witness_string("PASS_STAGE0_FRESH_ALLOCATION_WITNESS")},
      {"query_witness_count", witness_uint(witnesses.size())},
      {"native_query_count", witness_uint(native_count)},
      {"fallback_query_count", witness_uint(fallback_count)},
      {"exception_count", witness_int(0)},
      {"last_record_sha256", witness_string(previous)},
  }), witness_string(previous), &previous);
  output.close();
  require(static_cast<bool>(output), "failed writing Stage-0 fresh-allocation witness");
}

Observation execute_semantic_query(NativeSafeC1Matrix* engine, const Bundle& bundle,
                                   const QueryCase& item, bool native, int pass) {
  require(engine != nullptr, "null semantic-control engine");
  const std::int16_t* query = bundle.queries.data() +
      static_cast<std::size_t>(item.query_id) * bundle.pool.dimension;
  const int k = static_cast<int>(bundle.header.k);
  IssuedQuery issued = native
      ? engine->query_knn(item.query_id, query, k)
      : engine->query_range(item.query_id, query, std::numeric_limits<DistanceSq>::max());

  Observation output;
  output.condition = native ? "native_query_knn_candidate" : "exact_query_range_full_base_fallback";
  output.pass = pass;
  output.case_ordinal = item.ordinal;
  output.op_index = item.op_index;
  output.query_id = item.query_id;
  output.external_delta_live = static_cast<int>(item.external_delta_ids.size());
  output.api_result_count_before_adapter_truncation = static_cast<int>(issued.export_data.results.size());
  output.base_candidate_count = static_cast<int>(issued.export_data.base_results.size());
  output.visited_leaf_count = static_cast<int>(issued.export_data.gts_visited_leaf_ids.size());
  output.full_immutable_base_candidate_count = issued.export_data.full_immutable_base_candidate_count;
  output.exact_full_immutable_base_fallback = issued.export_data.exact_full_immutable_base_range_fallback;
  output.base_path = issued.export_data.base_path;

  // Stage-0 telemetry is copied only after the public API has returned.
  const auto& export_data = issued.export_data;
  output.fresh_witness.applicable = native;
  output.fresh_witness.tree_payload_sha256 = export_data.tree_payload_sha256;
  output.fresh_witness.exact_candidate_stable_distance_sha256 =
      safe_c1_g3::stable_distance_vector_sha256(
          export_data.base_results, "fair-safe-c1-stage0/exact-candidate-stable-distance/v1");
  output.fresh_witness.api_result_stable_distance_sha256 =
      safe_c1_g3::stable_distance_vector_sha256(
          export_data.results, "fair-safe-c1-stage0/api-result-stable-distance/v1");
  if (native) {
    output.fresh_witness.telemetry = export_data.fresh_allocation_telemetry;
    output.fresh_witness.raw_receipt_leaf_pair_count = export_data.raw_receipt_leaf_pair_count;
    output.fresh_witness.unique_receipt_leaf_count = export_data.unique_receipt_leaf_count;
    output.fresh_witness.id_list_capacity = export_data.id_list_capacity;
    output.fresh_witness.native_result_required_count =
        export_data.native_result_required_count;
    output.fresh_witness.native_result_complete = export_data.native_result_complete;
    output.fresh_witness.native_final_res_ids_cross_checked =
        export_data.native_final_res_ids_cross_checked;
    output.fresh_witness.visited_leaf_ids = export_data.gts_visited_leaf_ids;
    output.fresh_witness.receipt_leaf_spans = export_data.receipt_leaf_spans;
    output.fresh_witness.candidate_rows = export_data.base_receipt_rows;
    output.fresh_witness.ordered_leaf_pairs = export_data.ordered_raw_leaf_pairs;
    output.fresh_witness.native_result_ids = export_data.native_final_res_ids;
    output.fresh_witness.native_result_distances = export_data.native_final_res_distances;
    output.fresh_witness.candidate_stable_set_sha256 =
        hash_candidate_stable_set(output.fresh_witness.candidate_rows);
  } else {
    output.fresh_witness.reason = "exact_full_immutable_base_range_fallback";
  }

  std::vector<StableDistance> base_topk = std::move(issued.export_data.results);
  if (native) {
    require(issued.export_data.kind == "knn" && !output.exact_full_immutable_base_fallback &&
                output.full_immutable_base_candidate_count == 0 &&
                static_cast<int>(base_topk.size()) == k && output.base_candidate_count >= k &&
                output.base_candidate_count <= static_cast<int>(bundle.header.base_n),
            "native semantic call lost candidate-KNN contract");
  } else {
    require(issued.export_data.kind == "range" && output.exact_full_immutable_base_fallback &&
                output.full_immutable_base_candidate_count == static_cast<int>(bundle.header.base_n) &&
                output.base_path == "exact_full_immutable_base_range_fallback_no_gts_receipt" &&
                static_cast<int>(base_topk.size()) == static_cast<int>(bundle.header.base_n),
            "fallback semantic call lost full-base range-fallback contract");
    // The full-result fallback is shaped into its sealed top-k response.
    base_topk.resize(static_cast<std::size_t>(k));
  }
  // The exact external-delta scan/merge is part of the sealed semantic check.
  const std::vector<StableDistance> exact_delta =
      exact_topk_for_ids(bundle.pool, item.external_delta_ids, query, k);
  output.merged = merge_topk(std::move(base_topk), exact_delta, k);
  // Oracle/quality accounting and hashing are part of the semantic check.
  output.merged_overlap = stable_overlap_count(output.merged, item.exact_active_topk);
  output.merged_exact = output.merged == item.exact_active_topk;
  if (!native) require(output.merged_exact, "exact fallback lost exact full-active top-k");
  output.merged_sha256 = safe_c1_g3::stable_distance_vector_sha256(
      output.merged, "fair-safe-c1-semantic-control-e1-merged-result-v1");
  require(engine->mode() == EngineMode::kReady, "semantic call drifted engine mode");
  return output;
}

Observation semantic_only_observation(const Observation& source) {
  Observation output;
  output.condition = source.condition;
  output.pass = source.pass;
  output.case_ordinal = source.case_ordinal;
  output.op_index = source.op_index;
  output.query_id = source.query_id;
  output.external_delta_live = source.external_delta_live;
  output.api_result_count_before_adapter_truncation =
      source.api_result_count_before_adapter_truncation;
  output.base_candidate_count = source.base_candidate_count;
  output.visited_leaf_count = source.visited_leaf_count;
  output.full_immutable_base_candidate_count = source.full_immutable_base_candidate_count;
  output.exact_full_immutable_base_fallback = source.exact_full_immutable_base_fallback;
  output.base_path = source.base_path;
  output.merged = source.merged;
  output.merged_sha256 = source.merged_sha256;
  output.merged_overlap = source.merged_overlap;
  output.merged_exact = source.merged_exact;
  output.phase = source.phase;
  output.phase_pass = source.phase_pass;
  output.phase_slot = source.phase_slot;
  return output;
}

void run_phase(NativeSafeC1Matrix* engine, const Bundle& bundle,
               const std::vector<QueryCase>& cases, int pass_count, bool retain,
               const std::string& phase, int global_pass_base, int global_slot_base,
               std::vector<Observation>* observations, std::vector<Observation>* witnesses) {
  require(engine != nullptr && observations != nullptr && witnesses != nullptr,
          "null Stage-0 phase argument");
  require(cases.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max() / 2),
          "Stage-0 case count overflows global witness slots");
  const int slots_per_pass = static_cast<int>(cases.size()) * 2;
  for (int pass = 0; pass < pass_count; ++pass) {
    const int global_pass = global_pass_base + pass;
    const int pass_slot_base = global_slot_base + pass * slots_per_pass;
    for (const QueryCase& item : cases) {
      // Across warmup then measured, each case executes N,F,F,N (or F,N,N,F).
      // `pass` remains phase-local in legacy engine JSON; sidecar fields below
      // carry the global scheduling coordinates reconstructed by the validator.
      const bool native_first = ((global_pass + item.ordinal) % 2) == 0;
      Observation first = execute_semantic_query(engine, bundle, item, native_first, pass);
      Observation second = execute_semantic_query(engine, bundle, item, !native_first, pass);
      first.phase = phase;
      first.phase_pass = global_pass;
      first.phase_slot = pass_slot_base + item.ordinal * 2;
      second.phase = phase;
      second.phase_pass = global_pass;
      second.phase_slot = pass_slot_base + item.ordinal * 2 + 1;
      if (retain) {
        observations->push_back(semantic_only_observation(first));
        observations->push_back(semantic_only_observation(second));
      }
      witnesses->push_back(std::move(first));
      witnesses->push_back(std::move(second));
    }
  }
}

void write_semantic_observation(std::ostream& output, const Observation& row) {
  output << "{\"schema\":\"fresh-allocation-telemetry-e1-v1\""
         << ",\"record\":\"semantic_observation\""
         << ",\"diagnostic_variant\":\"fresh_allocation_telemetry_control\""
         << ",\"publication_eligible\":false"
         << ",\"condition\":";
  json_string(output, row.condition);
  output << ",\"semantic_pass\":" << row.pass
         << ",\"schedule_phase\":";
  json_string(output, row.phase);
  output << ",\"schedule_phase_pass\":" << row.phase_pass
         << ",\"schedule_phase_slot\":" << row.phase_slot
         << ",\"case_ordinal\":" << row.case_ordinal
         << ",\"op_index\":" << row.op_index
         << ",\"query_id\":" << row.query_id
         << ",\"external_global_delta_live\":" << row.external_delta_live
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
  events << "{\"schema\":\"fresh-allocation-telemetry-e1-v1\""
         << ",\"record\":\"dry_run_semantic_plan\""
         << ",\"diagnostic_variant\":\"fresh_allocation_telemetry_control\""
         << ",\"publication_eligible\":false"
         << ",\"gpu_used\":false"
         << ",\"case_count\":" << cases.size()
         << ",\"event_count\":" << bundle.events.size()
         << ",\"insert_count\":" << admission.insert_count
         << ",\"knn_count\":" << admission.knn_count
         << "}\n";
  events.close();
  require(static_cast<bool>(events), "failed writing dry-run output");
  std::ofstream summary(args.summary, std::ios::out);
  require(static_cast<bool>(summary), "cannot create dry-run summary");
  summary << "{\"schema\":\"fresh-allocation-telemetry-e1-v1\""
          << ",\"mode\":\"dry-run\""
          << ",\"status\":\"PASS_CPU_ONLY_SEMANTIC_PLAN\""
          << ",\"diagnostic_variant\":\"fresh_allocation_telemetry_control\""
          << ",\"publication_eligible\":false"
          << ",\"claim_scope\":\"semantic_fresh_allocation_control_only\""
          << ",\"gpu_used\":false"
          << ",\"scope\":\"sealed frozen-base semantic plan; no CUDA engine instantiated\""
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
  // Bootstrap is outside the declared Stage-0 ABBA schedule.  Reset only the
  // copied-header diagnostic counter after Ready is established; no traversal,
  // memory, kernel, allocator, or engine state is changed by this reset.
  safe_c1_fresh_alloc_epoch_counter = 0;
  safe_c1_last_fresh_alloc_telemetry = SafeC1FreshAllocTelemetry{};
  std::vector<Observation> observations;
  std::vector<Observation> witnesses;
  observations.reserve(static_cast<std::size_t>(args.measured_passes) * cases.size() * 2U);
  witnesses.reserve(static_cast<std::size_t>(args.warmup_passes + args.measured_passes) *
                    cases.size() * 2U);
  const int slots_per_phase_pass = static_cast<int>(cases.size()) * 2;
  run_phase(&engine, bundle, cases, args.warmup_passes, false, "initialization",
            /*global_pass_base=*/0, /*global_slot_base=*/0, &observations, &witnesses);
  run_phase(&engine, bundle, cases, args.measured_passes, true, "semantic",
            /*global_pass_base=*/args.warmup_passes,
            /*global_slot_base=*/args.warmup_passes * slots_per_phase_pass,
            &observations, &witnesses);
  require(engine.mode() == EngineMode::kReady, "engine mode drift after semantic phase");

  std::ofstream events(args.output, std::ios::out);
  require(static_cast<bool>(events), "cannot create semantic output");
  for (const Observation& row : observations) write_semantic_observation(events, row);
  events.close();
  require(static_cast<bool>(events), "failed writing semantic output");

  std::size_t native_exact = 0, native_overlap = 0, fallback_exact = 0;
  std::size_t native_records = 0, fallback_records = 0;
  for (const Observation& row : observations) {
    if (row.condition == "native_query_knn_candidate") {
      ++native_records;
      native_exact += row.merged_exact ? 1U : 0U;
      native_overlap += row.merged_overlap;
    } else {
      ++fallback_records;
      fallback_exact += row.merged_exact ? 1U : 0U;
    }
  }
  std::ofstream summary(args.summary, std::ios::out);
  require(static_cast<bool>(summary), "cannot create semantic summary");
  summary << "{\"schema\":\"fresh-allocation-telemetry-e1-v1\""
          << ",\"mode\":\"semantic-control\""
          << ",\"status\":\"PASS_SEMANTIC_CONTROL\""
          << ",\"diagnostic_variant\":\"fresh_allocation_telemetry_control\""
          << ",\"publication_eligible\":false"
          << ",\"claim_scope\":\"semantic_fresh_allocation_control_only\""
          << ",\"scope\":\"sealed-input semantic/correctness control with fresh-allocation witness sidecar\""
          << ",\"schedule\":{\"initialization_passes\":" << args.warmup_passes
          << ",\"semantic_passes\":" << args.measured_passes
          << ",\"semantic_records\":" << observations.size()
          << ",\"ordering\":\"ABBA_native_first_if_(global_phase_pass_plus_case_ordinal)_mod_2_is_0\"}"
          << ",\"conditions\":{\"native_query_knn_candidate\":" << native_records
          << ",\"exact_query_range_full_base_fallback\":" << fallback_records << "}"
          << ",\"quality\":{\"native_exact_set_count\":" << native_exact
          << ",\"native_overlap_sum\":" << native_overlap
          << ",\"fallback_exact_set_count\":" << fallback_exact
          << ",\"fallback_expected_exact_set_count\":"
          << (static_cast<std::size_t>(args.measured_passes) * cases.size()) << "}"
          << ",\"fresh_witness\":{\"query_records\":" << witnesses.size()
          << ",\"native_query_records\":" << (cases.size() * 2U)
          << ",\"fallback_query_records\":" << (cases.size() * 2U) << "}"
          << ",\"limitations\":[\"semantic_control_only\","
             "\"conditions_have_distinct_semantic_contracts\","
             "\"fresh_allocation_lifecycle_is_checked_in_sealed_witness\","
             "\"no_delete_range_workload_rebuild_or_direct_sidecar_claim\","
             "\"no_performance_conclusion\"]"
          << "}\n";
  summary.close();
  require(static_cast<bool>(summary), "failed writing semantic summary");
  // Sidecar construction follows every public API call and seals the
  // fresh-allocation lifecycle independently from the engine observations.
  write_fresh_allocation_witness(args, bundle, cases, witnesses);
  return 0;
}

}  // namespace fresh_allocation_semantic_control

int main(int argc, char** argv) {
  try {
    return fresh_allocation_semantic_control::run(fresh_allocation_semantic_control::parse_args(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
