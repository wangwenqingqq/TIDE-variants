// Controlled, single-translation-unit Safe-C1 G3 4K correctness pilot.
//
// This wrapper is intentionally a correctness gate, not a benchmark.  It
// records no timing/throughput/latency statistic and emits no performance
// conclusion.  It is fixed to the archived 4K fixture, validates every
// fixture/bootstrap/source-closure byte hash before Matrix construction, and
// maintains an independent host active set plus exact signed-int64 squared-L2
// oracle.  The oracle is runner control-plane code only: it is never exposed
// to the Matrix answer path except as the required independent expectation for
// opaque post-rebuild tickets.
//
// The only native-engine entry points below are NativeSafeC1Matrix public APIs.
// This wrapper makes no raw GTS traversal/build call and no raw residual-
// pruning upload/read call.  It must be built as exactly one wrapper TU that
// includes the implementation below; do not link a second legacy GTS TU/DSO
// or allow external legacy GTS/RP calls in the controlled process.
//
// Range scope is deliberately narrow: the Matrix range path is a
// branch-aligned vector predicate mirror with exact result filtering.  This
// runner does not call it archive-native, does not claim bitwise equivalence
// to the archive RNN entry point, and does not make any direct-range claim.
// Direct sidecars are statically disabled; every mutable insertion in this
// pilot must return delta placement.

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <string_view>
#include <system_error>
#include <type_traits>

// Unique implementation inclusion: do not separately compile/link this .cu.
#include "../src/g3_safe_c1_native_matrix.cu"

namespace g3_4k_pilot {
namespace fs = std::filesystem;

using safe_c1_g3::DistanceSq;
using safe_c1_g3::EngineMode;
using safe_c1_g3::IssuedQuery;
using safe_c1_g3::NativeSafeC1Matrix;
using safe_c1_g3::Placement;
using safe_c1_g3::PlacementKind;
using safe_c1_g3::QueryExport;
using safe_c1_g3::RebuildReceipt;
using safe_c1_g3::StableDistance;
using safe_c1_g3::StableId;

constexpr char kReleaseRoot[] =
    "/workspace/experiments/tide_safe_c1_20260727/"
    "safe_c1_native_matrix_g3_v4_lightweight_identity_pilot";
constexpr char kFixtureRelative[] = "inputs/g3_sift4096_branchstress_l2_d3";
constexpr char kBootstrapRelative[] =
    "preflight/bootstrap_oracles/g3_sift4096_branchstress_l2_d3";
constexpr int kDimension = 128;
constexpr int kPoolRows = 6144;
constexpr int kQueryRows = 131;
constexpr int kBaseCount = 4096;
constexpr int kK = 10;
constexpr int kExpectedTreeHeight = 4;
constexpr int kExpectedFanout = 10;
constexpr DistanceSq kBootstrapRangeRadius = 427400000ULL;
constexpr DistanceSq kTracePostRebuildRangeRadius = 1516660000ULL;
constexpr char kInitialLiveSha[] =
    "2cf645aec1ff09ceac94895976db7d23ae80271c8af1e11cf353f416f09ad77e";
constexpr char kPostRebuildLiveSha[] =
    "5bba15c93adaf8b81b13288d7617374cf27688748c6c48ef9faf0a09d33f8c6d";

// This is the v3 source after only the RebuildReceipt topology extension.  A
// later build/launch guard must additionally bind the produced executable and
// its single-TU build command; runtime source checking alone cannot attest a
// previously built binary.
struct ExpectedFile {
  const char* relative;
  const char* sha256;
  std::size_t exact_bytes;  // zero means textual size is not independently fixed.
};

constexpr ExpectedFile kSourceClosure[] = {
    {"src/g3_safe_c1_native_matrix.cu",
     "c22a7682f5aa56e4d96ae2a912348eaf4bcb379971950a21bd15fd89f5c5ca81", 0},
    {"src/g3_safe_search_v2.cuh",
     "e52ae9a14db32967ef0816d880a6fb46ed7793e86ebc85175b9f96986f46b4f0", 0},
    {"reference/include/tree.cuh",
     "f812c385d254d77e84bbd515e4055e94cc8163ee7a7d57216f3659c09f569c2b", 0},
    {"reference/include/file.cuh",
     "b8be03246ec82cadd3228b75a6dfd91c552ccfa3124d2ca9470e28e0031ffd8a", 0},
    {"reference/include/config.cuh",
     "622d0977e49d80d5791364bc12463de9bce5aaca324d6a681512004c20d100cb", 0},
    {"reference/include/mlp_constant.cuh",
     "cf6545624c0cbc51744b978298e6d138591521c7b600ece8812b6195f735e559", 0},
    {"reference/include/residual_pruning.cuh",
     "745a4bd5564b8085c40a48abac625f24dcbacb8a996d29e4cba311dd936e33b2", 0},
};

constexpr ExpectedFile kFixtureFiles[] = {
    {"manifest.json",
     "1ee0e125c13242b1aae201e71954d1e2117234ec2252f5a15a0008967ec4145d", 0},
    {"metadata.json",
     "db549ae25907603865d330bab171945b52cbad5a49b714989723551a7416ff99", 0},
    {"initial_base_stable_ids.i32",
     "6b0751ba5e64fc9c13ddfb44778fa7d6a1f7d7aa9d6a5e38a1f0a1502c3fb9e3",
     static_cast<std::size_t>(kBaseCount) * sizeof(std::int32_t)},
    {"pool.i16",
     "899adaa59b265ee788841f1a48b667b7da39df166972ba9568c4e94727b31170",
     static_cast<std::size_t>(kPoolRows) * kDimension * sizeof(std::int16_t)},
    {"queries.i16",
     "f92cdb94b02e7d920b7d54023961a823de29b8a962bdde606efc2acc1501d6f5",
     static_cast<std::size_t>(kQueryRows) * kDimension * sizeof(std::int16_t)},
    {"stable_id_to_pool_row.i32",
     "93710cce11c994b6b1934713842c93cfcec76a3563fc47574abf419137f4c5c8",
     static_cast<std::size_t>(kPoolRows) * sizeof(std::int32_t)},
    {"oracle_expected.jsonl",
     "9a3a5ba48ffedad2432cfe03196082543f0fd8f80b1befc2d25515f1b6f3c571", 0},
    {"selection_receipt.json",
     "42c23134749776b44abe4dcd1408143a94c8b9312064cd4c16c15d5d016abe3d", 0},
    {"trace.jsonl",
     "e520e256333d1c91e8ca7beb2e72d7651f63d60f6a712cb91a8d5d98a3c063d8", 0},
};

constexpr ExpectedFile kBootstrapFiles[] = {
    {"bootstrap_oracle.json",
     "4c74995c19f6a9a1309d75bb1611850ceec877b5c66f813f7f0e120e42e0ad1f", 0},
    {"bootstrap_oracle_validation.json",
     "8f5ea4416ca20ece3c960b683420844e2aa84931c6990ac958120ab2e1e4e851", 0},
};

[[noreturn]] void runner_fail(const std::string& reason) {
  throw std::runtime_error("G3 controlled 4K pilot refused: " + reason);
}

void require(bool condition, const std::string& reason) {
  if (!condition) runner_fail(reason);
}

std::string sha256_bytes(const std::vector<std::uint8_t>& bytes) {
  safe_c1_g3::Sha256 digest;
  if (!bytes.empty()) digest.update(bytes.data(), bytes.size());
  return digest.final_hex();
}

// The runner intentionally owns canonicalization and result/stable-set
// serialization for its independent oracle.  It shares only a small SHA-256
// primitive with the TU, not the GTS candidate selection, state, sort helper,
 // or production result validator.
std::string runner_sha256_text(const std::string& text) {
  safe_c1_g3::Sha256 digest;
  digest.update(text);
  return digest.final_hex();
}

bool runner_distance_then_stable(const StableDistance& left,
                                 const StableDistance& right) {
  return left.distance_sq != right.distance_sq
             ? left.distance_sq < right.distance_sq
             : left.stable_id < right.stable_id;
}

void require_runner_canonical(const std::vector<StableDistance>& rows,
                              const std::string& label) {
  std::set<StableId> seen;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    require(rows[i].stable_id >= 0, label + " has negative stable ID");
    require(seen.insert(rows[i].stable_id).second, label + " has duplicate stable ID");
    if (i != 0) {
      require(runner_distance_then_stable(rows[i - 1], rows[i]),
              label + " is not in local (distance_sq,stable_id) order");
    }
  }
}

void local_sort_and_require_canonical(std::vector<StableDistance>* rows,
                                      const std::string& label) {
  require(rows != nullptr, label + " has null rows");
  std::sort(rows->begin(), rows->end(), runner_distance_then_stable);
  require_runner_canonical(*rows, label);
}

std::string local_stable_set_sha(const std::vector<StableId>& ids) {
  std::vector<StableId> canonical = ids;
  std::sort(canonical.begin(), canonical.end());
  require(std::adjacent_find(canonical.begin(), canonical.end()) == canonical.end(),
          "local stable-set digest received duplicate ID");
  std::ostringstream bytes;
  for (StableId id : canonical) bytes << id << '\n';
  return runner_sha256_text(bytes.str());
}

std::string local_result_sha(const std::vector<StableDistance>& rows) {
  require_runner_canonical(rows, "local result digest");
  std::ostringstream bytes;
  bytes << "safe-c1-g3-controlled-4k-local-result-v1\n";
  for (const StableDistance& row : rows) {
    bytes << row.stable_id << ':' << static_cast<unsigned long long>(row.distance_sq) << '\n';
  }
  return runner_sha256_text(bytes.str());
}

std::vector<std::uint8_t> read_regular_file_strict(const fs::path& path) {
  const fs::file_status status = fs::symlink_status(path);
  require(!fs::is_symlink(status), "refusing symlinked input: " + path.string());
  require(fs::is_regular_file(status), "required regular file is missing: " + path.string());
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "cannot open input: " + path.string());
  std::vector<char> raw((std::istreambuf_iterator<char>(input)),
                        std::istreambuf_iterator<char>());
  require(input.eof(), "cannot fully read input: " + path.string());
  std::vector<std::uint8_t> bytes(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>(static_cast<unsigned char>(raw[i]));
  }
  return bytes;
}

std::string text_from_bytes(const std::vector<std::uint8_t>& bytes,
                            const std::string& label) {
  for (std::uint8_t byte : bytes) require(byte != 0, label + " unexpectedly contains NUL");
  return std::string(bytes.begin(), bytes.end());
}

void require_file_hash(const fs::path& root, const ExpectedFile& expected) {
  const fs::path file = root / expected.relative;
  const std::vector<std::uint8_t> bytes = read_regular_file_strict(file);
  if (expected.exact_bytes != 0) {
    require(bytes.size() == expected.exact_bytes,
            std::string("unexpected byte length for ") + expected.relative);
  }
  require(sha256_bytes(bytes) == expected.sha256,
          std::string("SHA-256 mismatch for ") + expected.relative);
}

void require_all_hashes(const fs::path& root, const ExpectedFile* files,
                        std::size_t file_count) {
  for (std::size_t i = 0; i < file_count; ++i) require_file_hash(root, files[i]);
}

void require_contains(const std::string& text, const std::string& token,
                      const std::string& label) {
  require(text.find(token) != std::string::npos,
          label + " lacks required token: " + token);
}

void validate_manifest_and_bootstrap_contract(const fs::path& fixture,
                                              const fs::path& bootstrap) {
  const std::string manifest = text_from_bytes(read_regular_file_strict(fixture / "manifest.json"),
                                               "manifest.json");
  require_contains(manifest, "\"schema\": \"safe-c1-g3-bundle-manifest-v1\"",
                   "manifest.json");
  require_contains(manifest, "\"status\": \"CPU_PREPARED_NOT_NATIVE_EXECUTED\"",
                   "manifest.json");
  for (const ExpectedFile& file : kFixtureFiles) {
    if (std::string(file.relative) == "manifest.json") continue;
    require_contains(manifest,
                     std::string("\"") + file.relative + "\": \"" + file.sha256 + "\"",
                     "manifest.json");
  }

  const std::string metadata = text_from_bytes(read_regular_file_strict(fixture / "metadata.json"),
                                               "metadata.json");
  require_contains(metadata, "\"base_n\": 4096", "metadata.json");
  require_contains(metadata, "\"dim\": 128", "metadata.json");
  require_contains(metadata, "\"event_count\": 21", "metadata.json");
  require_contains(metadata, "\"height\": 4", "metadata.json");
  require_contains(metadata, "\"metric\": \"L2\"", "metadata.json");
  require_contains(metadata, "\"oracle_distance\": \"int64 squared L2\"", "metadata.json");

  const std::string bootstrap_json =
      text_from_bytes(read_regular_file_strict(bootstrap / "bootstrap_oracle.json"),
                      "bootstrap_oracle.json");
  require_contains(bootstrap_json, "\"schema\": \"safe-c1-g3-bootstrap-oracle-v1\"",
                   "bootstrap_oracle.json");
  require_contains(bootstrap_json, "\"status\": \"CPU_ONLY_NOT_RUN\"",
                   "bootstrap_oracle.json");
  require_contains(bootstrap_json, "\"query_id\": 1", "bootstrap_oracle.json");
  require_contains(bootstrap_json, "\"query_id\": 2", "bootstrap_oracle.json");
  require_contains(bootstrap_json, "\"radius_sq\": 427400000",
                   "bootstrap_oracle.json");
  require_contains(bootstrap_json, kInitialLiveSha, "bootstrap_oracle.json");
  require_contains(bootstrap_json,
                   "\"must_not_append_to_fixture_or_engine_trace\": true",
                   "bootstrap_oracle.json");

  const std::string bootstrap_validation =
      text_from_bytes(read_regular_file_strict(bootstrap / "bootstrap_oracle_validation.json"),
                      "bootstrap_oracle_validation.json");
  require_contains(bootstrap_validation, "PASS_CPU_ONLY_NOT_RUN",
                   "bootstrap_oracle_validation.json");
}

std::uint16_t decode_u16_le(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
  require(offset + 2 <= bytes.size(), "truncated little-endian u16");
  return static_cast<std::uint16_t>(bytes[offset]) |
         (static_cast<std::uint16_t>(bytes[offset + 1]) << 8U);
}

std::int16_t decode_i16_le(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
  return static_cast<std::int16_t>(decode_u16_le(bytes, offset));
}

std::uint32_t decode_u32_le(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
  require(offset + 4 <= bytes.size(), "truncated little-endian u32");
  return static_cast<std::uint32_t>(bytes[offset]) |
         (static_cast<std::uint32_t>(bytes[offset + 1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[offset + 2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[offset + 3]) << 24U);
}

std::int32_t decode_i32_le(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
  const std::uint32_t raw = decode_u32_le(bytes, offset);
  if (raw <= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
    return static_cast<std::int32_t>(raw);
  }
  const std::int64_t signed_value = static_cast<std::int64_t>(raw) - (1LL << 32);
  return static_cast<std::int32_t>(signed_value);
}

std::vector<std::int16_t> decode_i16_file(const fs::path& path, std::size_t expected_values) {
  const std::vector<std::uint8_t> bytes = read_regular_file_strict(path);
  require(bytes.size() == expected_values * sizeof(std::int16_t),
          "unexpected i16 input length: " + path.string());
  std::vector<std::int16_t> values(expected_values);
  for (std::size_t i = 0; i < expected_values; ++i) {
    values[i] = decode_i16_le(bytes, i * sizeof(std::int16_t));
  }
  return values;
}

std::vector<int> decode_i32_file_to_int(const fs::path& path, std::size_t expected_values) {
  const std::vector<std::uint8_t> bytes = read_regular_file_strict(path);
  require(bytes.size() == expected_values * sizeof(std::int32_t),
          "unexpected i32 input length: " + path.string());
  std::vector<int> values(expected_values);
  for (std::size_t i = 0; i < expected_values; ++i) {
    const std::int32_t value = decode_i32_le(bytes, i * sizeof(std::int32_t));
    require(value >= std::numeric_limits<int>::min() &&
                value <= std::numeric_limits<int>::max(),
            "i32 fixture value outside host int range");
    values[i] = static_cast<int>(value);
  }
  return values;
}

struct FixtureData {
  std::vector<std::int16_t> pool;
  std::vector<std::int16_t> queries;
  std::vector<int> stable_to_pool_row;
  std::vector<StableId> initial_base;
};

FixtureData load_and_validate_binary_fixture(const fs::path& fixture) {
  FixtureData data;
  data.pool = decode_i16_file(fixture / "pool.i16",
                              static_cast<std::size_t>(kPoolRows) * kDimension);
  data.queries = decode_i16_file(fixture / "queries.i16",
                                 static_cast<std::size_t>(kQueryRows) * kDimension);
  data.stable_to_pool_row =
      decode_i32_file_to_int(fixture / "stable_id_to_pool_row.i32", kPoolRows);
  data.initial_base =
      decode_i32_file_to_int(fixture / "initial_base_stable_ids.i32", kBaseCount);

  std::vector<int> physical_seen(kPoolRows, 0);
  for (int row : data.stable_to_pool_row) {
    require(row >= 0 && row < kPoolRows, "stable-to-pool mapping leaves immutable pool");
    require(++physical_seen[static_cast<std::size_t>(row)] == 1,
            "stable-to-pool mapping is not bijective for this fixed fixture");
  }
  for (int seen : physical_seen) require(seen == 1, "stable-to-pool mapping misses a pool row");

  std::set<StableId> initial_seen;
  for (StableId stable : data.initial_base) {
    require(stable >= 0 && stable < kPoolRows, "initial stable ID outside fixture capacity");
    require(initial_seen.insert(stable).second, "duplicate initial stable ID");
  }
  require(static_cast<int>(initial_seen.size()) == kBaseCount, "initial base cardinality drift");
  require(local_stable_set_sha(data.initial_base) == kInitialLiveSha,
          "initial base stable-ID digest drifted");
  return data;
}

std::size_t field_value_start(const std::string& json, const std::string& key,
                              const std::string& label) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = json.find(needle);
  require(key_pos != std::string::npos, label + " has no field " + key);
  require(json.find(needle, key_pos + needle.size()) == std::string::npos,
          label + " duplicates scalar field " + key);
  std::size_t value = json.find(':', key_pos + needle.size());
  require(value != std::string::npos, label + " has malformed field " + key);
  ++value;
  while (value < json.size() &&
         (json[value] == ' ' || json[value] == '\t' || json[value] == '\r' ||
          json[value] == '\n')) {
    ++value;
  }
  require(value < json.size(), label + " has empty field " + key);
  return value;
}

std::string json_string_field(const std::string& json, const std::string& key,
                              const std::string& label) {
  std::size_t pos = field_value_start(json, key, label);
  require(json[pos] == '"', label + " field " + key + " is not a JSON string");
  ++pos;
  std::string output;
  while (pos < json.size() && json[pos] != '"') {
    require(json[pos] != '\\', label + " field " + key + " uses unsupported escape");
    output.push_back(json[pos++]);
  }
  require(pos < json.size(), label + " field " + key + " is unterminated");
  return output;
}

std::int64_t parse_signed_decimal(const std::string& json, std::size_t* pos,
                                  const std::string& label) {
  require(pos != nullptr && *pos < json.size(), label + " has missing signed integer");
  bool negative = false;
  if (json[*pos] == '-') {
    negative = true;
    ++(*pos);
  }
  require(*pos < json.size() && json[*pos] >= '0' && json[*pos] <= '9',
          label + " has malformed signed integer");
  std::uint64_t magnitude = 0;
  while (*pos < json.size() && json[*pos] >= '0' && json[*pos] <= '9') {
    const std::uint64_t digit = static_cast<std::uint64_t>(json[*pos] - '0');
    require(magnitude <= (std::numeric_limits<std::uint64_t>::max() - digit) / 10ULL,
            label + " signed integer overflow");
    magnitude = magnitude * 10ULL + digit;
    ++(*pos);
  }
  if (!negative) {
    require(magnitude <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
            label + " signed integer exceeds int64");
    return static_cast<std::int64_t>(magnitude);
  }
  require(magnitude <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1ULL,
          label + " signed integer underflows int64");
  if (magnitude == static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1ULL) {
    return std::numeric_limits<std::int64_t>::min();
  }
  return -static_cast<std::int64_t>(magnitude);
}

std::uint64_t parse_unsigned_decimal(const std::string& json, std::size_t* pos,
                                     const std::string& label) {
  require(pos != nullptr && *pos < json.size() && json[*pos] >= '0' && json[*pos] <= '9',
          label + " has malformed unsigned integer");
  std::uint64_t value = 0;
  while (*pos < json.size() && json[*pos] >= '0' && json[*pos] <= '9') {
    const std::uint64_t digit = static_cast<std::uint64_t>(json[*pos] - '0');
    require(value <= (std::numeric_limits<std::uint64_t>::max() - digit) / 10ULL,
            label + " unsigned integer overflow");
    value = value * 10ULL + digit;
    ++(*pos);
  }
  return value;
}

std::int64_t json_int_field(const std::string& json, const std::string& key,
                            const std::string& label) {
  std::size_t pos = field_value_start(json, key, label);
  return parse_signed_decimal(json, &pos, label + " field " + key);
}

std::optional<std::int64_t> json_optional_int_field(const std::string& json,
                                                     const std::string& key,
                                                     const std::string& label) {
  std::size_t pos = field_value_start(json, key, label);
  if (json.compare(pos, 4, "null") == 0) return std::nullopt;
  return parse_signed_decimal(json, &pos, label + " field " + key);
}

void skip_ws(const std::string& input, std::size_t* pos) {
  while (*pos < input.size() &&
         (input[*pos] == ' ' || input[*pos] == '\t' || input[*pos] == '\r' ||
          input[*pos] == '\n')) {
    ++(*pos);
  }
}

std::vector<StableDistance> json_distance_pairs_field(const std::string& json,
                                                       const std::string& key,
                                                       const std::string& label) {
  std::size_t pos = field_value_start(json, key, label);
  require(json[pos] == '[', label + " result field is not an array");
  ++pos;
  skip_ws(json, &pos);
  std::vector<StableDistance> output;
  if (pos < json.size() && json[pos] == ']') {
    ++pos;
    return output;
  }
  while (true) {
    require(pos < json.size() && json[pos] == '[',
            label + " result row is not [stable_id,distance_sq]");
    ++pos;
    skip_ws(json, &pos);
    const std::int64_t stable = parse_signed_decimal(json, &pos, label + " result stable ID");
    require(stable >= 0 && stable <= std::numeric_limits<int>::max(),
            label + " result stable ID outside int range");
    skip_ws(json, &pos);
    require(pos < json.size() && json[pos] == ',', label + " result row lacks comma");
    ++pos;
    skip_ws(json, &pos);
    const std::uint64_t distance = parse_unsigned_decimal(json, &pos, label + " result distance");
    skip_ws(json, &pos);
    require(pos < json.size() && json[pos] == ']', label + " result row lacks closing bracket");
    ++pos;
    output.push_back(StableDistance{static_cast<int>(stable), distance});
    skip_ws(json, &pos);
    require(pos < json.size(), label + " result list is truncated");
    if (json[pos] == ']') {
      ++pos;
      break;
    }
    require(json[pos] == ',', label + " result list lacks comma");
    ++pos;
    skip_ws(json, &pos);
  }
  require_runner_canonical(output, label);
  return output;
}

std::vector<std::string> nonempty_lines(const std::string& text,
                                        const std::string& label) {
  std::vector<std::string> lines;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t end = text.find('\n', begin);
    const std::string line = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (!line.empty()) {
      require(line.back() != '\r', label + " uses unsupported CRLF line form");
      lines.push_back(line);
    }
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return lines;
}

struct TraceOp {
  int op_index = -1;
  std::string op;
  StableId stable_id = -1;
  int query_id = -1;
  DistanceSq radius_sq = 0;
  std::string fixture_role;
};

struct FixedTraceExpected {
  int op_index;
  const char* op;
  int stable_id;
  int query_id;
  DistanceSq radius_sq;
};

constexpr FixedTraceExpected kFixedTrace[] = {
    {0, "insert", 4876, -1, 0},
    {1, "insert", 4898, -1, 0},
    {2, "knn", -1, 128, 0},
    {3, "range", -1, 128, 1},
    {4, "insert", 5303, -1, 0},
    {5, "knn", -1, 129, 0},
    {6, "range", -1, 129, 1},
    {7, "delete", 4876, -1, 0},
    {8, "knn", -1, 128, 0},
    {9, "range", -1, 128, 1},
    {10, "delete", 5303, -1, 0},
    {11, "knn", -1, 129, 0},
    {12, "range", -1, 129, 1},
    {13, "insert", 4097, -1, 0},
    {14, "knn", -1, 130, 0},
    {15, "range", -1, 130, 1},
    {16, "insert", 4099, -1, 0},
    {17, "insert", 4102, -1, 0},
    {18, "rebuild", -1, -1, 0},
    {19, "knn", -1, 0, 0},
    {20, "range", -1, 0, kTracePostRebuildRangeRadius},
};

std::string optional_role_from_trace(const std::string& line) {
  if (line.find("\"expected_role\"") != std::string::npos) {
    return json_string_field(line, "expected_role", "trace event");
  }
  if (line.find("\"expected_prior\"") != std::string::npos) {
    return std::string("prior_") + json_string_field(line, "expected_prior", "trace event");
  }
  return "";
}

std::vector<TraceOp> parse_and_validate_fixed_trace(const fs::path& fixture) {
  const std::string trace =
      text_from_bytes(read_regular_file_strict(fixture / "trace.jsonl"), "trace.jsonl");
  const std::vector<std::string> lines = nonempty_lines(trace, "trace.jsonl");
  require(lines.size() == std::size(kFixedTrace), "trace must contain exactly op_index 0..20");
  std::vector<TraceOp> result;
  result.reserve(lines.size());
  for (std::size_t i = 0; i < lines.size(); ++i) {
    const FixedTraceExpected& expected = kFixedTrace[i];
    const std::string label = "trace event " + std::to_string(i);
    require(json_int_field(lines[i], "op_index", label) == expected.op_index,
            label + " has wrong op_index");
    const std::string op = json_string_field(lines[i], "op", label);
    require(op == expected.op, label + " has wrong operation");
    TraceOp current;
    current.op_index = expected.op_index;
    current.op = op;
    current.fixture_role = optional_role_from_trace(lines[i]);
    if (op == "insert" || op == "delete") {
      const std::int64_t id = json_int_field(lines[i], "stable_id", label);
      require(id == expected.stable_id, label + " has wrong stable ID");
      current.stable_id = static_cast<StableId>(id);
    } else if (op == "knn") {
      const std::int64_t query_id = json_int_field(lines[i], "query_id", label);
      require(query_id == expected.query_id, label + " has wrong query ID");
      current.query_id = static_cast<int>(query_id);
    } else if (op == "range") {
      const std::int64_t query_id = json_int_field(lines[i], "query_id", label);
      const std::int64_t radius = json_int_field(lines[i], "radius_sq", label);
      require(query_id == expected.query_id && radius >= 0 &&
                  static_cast<DistanceSq>(radius) == expected.radius_sq,
              label + " has wrong range contract");
      current.query_id = static_cast<int>(query_id);
      current.radius_sq = static_cast<DistanceSq>(radius);
    } else if (op == "rebuild") {
      require(expected.op_index == 18, "unexpected rebuild position");
      require(json_string_field(lines[i], "expected_live_ids_sha256", label) == kPostRebuildLiveSha,
              "rebuild trace live-ID digest drift");
      // This is a pinned legacy source-role datum (two historical direct
      // insertions left one direct survivor plus three deltas), not the safe
      // v3 runtime policy.  v3 validates it for input provenance only and
      // separately asserts its four surviving mutable IDs are all delta.
      require(json_int_field(lines[i], "expected_delta_live", label) == 3,
              "rebuild trace legacy delta-count datum drift");
    } else {
      runner_fail(label + " uses unknown operation");
    }
    result.push_back(std::move(current));
  }
  return result;
}

struct OracleRecord {
  int op_index = -1;
  std::string kind;
  int query_id = -1;
  DistanceSq radius_sq = 0;
  std::string active_ids_sha256;
  std::vector<StableDistance> results;
};

std::map<int, OracleRecord> parse_and_validate_trace_oracle(
    const fs::path& fixture, const std::vector<TraceOp>& trace) {
  const std::string oracle =
      text_from_bytes(read_regular_file_strict(fixture / "oracle_expected.jsonl"),
                      "oracle_expected.jsonl");
  const std::vector<std::string> lines = nonempty_lines(oracle, "oracle_expected.jsonl");
  std::map<int, OracleRecord> output;
  for (const std::string& line : lines) {
    const std::string label = "oracle_expected record";
    require(json_string_field(line, "record", label) == "oracle_query",
            "oracle_expected record kind drift");
    const std::int64_t op_index = json_int_field(line, "op_index", label);
    require(op_index >= 0 && op_index <= 20, "oracle op index outside fixed trace");
    OracleRecord record;
    record.op_index = static_cast<int>(op_index);
    record.kind = json_string_field(line, "kind", label);
    const std::int64_t query_id = json_int_field(line, "query_id", label);
    require(query_id >= 0 && query_id < kQueryRows, "oracle query ID outside query input");
    record.query_id = static_cast<int>(query_id);
    const std::optional<std::int64_t> radius =
        json_optional_int_field(line, "radius_sq", label);
    if (record.kind == "knn") {
      require(!radius.has_value(), "KNN oracle must carry null radius");
    } else if (record.kind == "range") {
      require(radius.has_value() && *radius >= 0, "range oracle has invalid radius");
      record.radius_sq = static_cast<DistanceSq>(*radius);
    } else {
      runner_fail("oracle uses unknown query kind");
    }
    record.active_ids_sha256 = json_string_field(line, "active_ids_sha256", label);
    record.results = json_distance_pairs_field(line, "results", label);
    require(output.emplace(record.op_index, std::move(record)).second,
            "oracle has duplicate op index");
  }

  std::size_t expected_queries = 0;
  for (const TraceOp& event : trace) {
    if (event.op == "knn" || event.op == "range") {
      ++expected_queries;
      const auto it = output.find(event.op_index);
      require(it != output.end(), "oracle lacks a trace query op");
      require(it->second.kind == event.op && it->second.query_id == event.query_id,
              "oracle query kind/ID does not match trace");
      if (event.op == "range") {
        require(it->second.radius_sq == event.radius_sq, "oracle range radius does not match trace");
      }
    }
  }
  require(output.size() == expected_queries, "oracle contains unexpected query records");
  return output;
}

std::vector<StableDistance> exact_oracle(const FixtureData& data,
                                         const std::set<StableId>& active,
                                         int query_id, bool is_knn,
                                         int requested_k, DistanceSq radius_sq) {
  require(query_id >= 0 && query_id < kQueryRows, "independent oracle query ID outside fixture");
  require(!active.empty(), "independent oracle refuses an empty active set");
  const std::int16_t* query =
      data.queries.data() + static_cast<std::size_t>(query_id) * kDimension;
  std::vector<StableDistance> all;
  all.reserve(active.size());
  for (StableId stable : active) {
    require(stable >= 0 && stable < static_cast<int>(data.stable_to_pool_row.size()),
            "independent oracle stable ID outside mapping");
    const int physical_row = data.stable_to_pool_row[static_cast<std::size_t>(stable)];
    require(physical_row >= 0 && physical_row < kPoolRows,
            "independent oracle physical row outside pool");
    const std::int16_t* point =
        data.pool.data() + static_cast<std::size_t>(physical_row) * kDimension;
    std::int64_t sum = 0;
    for (int dim = 0; dim < kDimension; ++dim) {
      const std::int64_t diff = static_cast<std::int64_t>(point[dim]) -
                                static_cast<std::int64_t>(query[dim]);
      const std::int64_t square = diff * diff;
      require(sum <= std::numeric_limits<std::int64_t>::max() - square,
              "independent signed-int64 L2 accumulator overflow");
      sum += square;
    }
    all.push_back(StableDistance{stable, static_cast<DistanceSq>(sum)});
  }
  local_sort_and_require_canonical(&all, "independent signed-int64 L2 oracle");
  if (is_knn) {
    require(requested_k > 0, "independent KNN oracle requires positive k");
    const std::size_t count = std::min<std::size_t>(requested_k, all.size());
    all.resize(count);
  } else {
    all.erase(std::remove_if(all.begin(), all.end(),
                             [radius_sq](const StableDistance& row) {
                               return row.distance_sq > radius_sq;
                             }),
              all.end());
  }
  require_runner_canonical(all, "independent signed-int64 L2 oracle output");
  return all;
}

void require_equal_results(const std::vector<StableDistance>& actual,
                           const std::vector<StableDistance>& expected,
                           const std::string& label) {
  if (actual != expected) {
    const std::string actual_sha = local_result_sha(actual);
    const std::string expected_sha = local_result_sha(expected);
    runner_fail(label + " mismatch; actual=" + actual_sha + " independent=" + expected_sha);
  }
}

std::vector<StableDistance> bootstrap_knn_expected_literal() {
  return {{2781, 659730000ULL}, {2492, 691140000ULL}, {1322, 691300000ULL},
          {3136, 701890000ULL}, {1038, 727480000ULL}, {925, 768220000ULL},
          {3998, 768250000ULL}, {2183, 769360000ULL}, {1533, 773060000ULL},
          {145, 773980000ULL}};
}

std::vector<StableDistance> bootstrap_range_expected_literal() {
  return {{2707, 427400000ULL}};
}

const std::int16_t* query_ptr(const FixtureData& data, int query_id) {
  require(query_id >= 0 && query_id < kQueryRows, "query pointer ID outside fixture");
  return data.queries.data() + static_cast<std::size_t>(query_id) * kDimension;
}

std::string json_escape(const std::string& value) {
  std::ostringstream escaped;
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      escaped << '\\' << static_cast<char>(c);
    } else if (c >= 0x20U) {
      escaped << static_cast<char>(c);
    } else {
      escaped << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(c) << std::dec << std::setfill(' ');
    }
  }
  return escaped.str();
}

std::string result_sha_for_attestation(const std::vector<StableDistance>& values) {
  return local_result_sha(values);
}

std::string ticket_attestation_json(const std::string& phase, int op_index,
                                    const QueryExport& query,
                                    const std::vector<StableDistance>& expected) {
  require(query.post_rebuild_verification_ticket_issued,
          "cannot attest a query without a private issued ticket");
  require(!query.post_rebuild_binding_sha256.empty() &&
              !query.post_rebuild_receipt_sha256.empty() &&
              !query.post_rebuild_result_sha256.empty(),
          "issued ticket query lacks binding diagnostics");
  std::ostringstream out;
  out << "{\"record\":\"ticket_attestation\",\"phase\":\"" << json_escape(phase)
      << "\",\"op_index\":" << op_index
      << ",\"kind\":\"" << json_escape(query.kind) << "\""
      << ",\"query_id\":" << query.query_id
      << ",\"tree_version\":" << query.tree_version
      << ",\"state_epoch\":" << query.state_epoch
      << ",\"ticket_issuance_nonce\":" << query.post_rebuild_issuance_nonce
      << ",\"ticket_consumed\":true"
      << ",\"engine_result_sha256\":\"" << query.post_rebuild_result_sha256 << "\""
      << ",\"independent_result_sha256\":\"" << result_sha_for_attestation(expected) << "\""
      << ",\"receipt_sha256\":\"" << query.post_rebuild_receipt_sha256 << "\""
      << ",\"binding_sha256\":\"" << query.post_rebuild_binding_sha256 << "\""
      << "}";
  return out.str();
}

std::string mutation_attestation_json(const TraceOp& event, const Placement& placement,
                                      const std::set<StableId>& active) {
  // Keep the established CPU validator's update schema exactly.  The extra
  // certificate diagnostics distinguish the native frozen-tree certificate
  // from the closed runtime placement policy: a true certificate still must
  // produce a delta row while the direct guard is false.
  require(event.op == "insert" || event.op == "delete",
          "mutation serializer received a non-mutation event");
  const char* placement_name = safe_c1_g3::placement_name(placement.kind);
  std::ostringstream out;
  out << "{\"record\":\"update\",\"op\":\"" << event.op
      << "\",\"op_index\":" << event.op_index
      << ",\"stable_id\":" << event.stable_id
      << ",\"fixture_role\":\"" << json_escape(event.fixture_role) << "\""
      << ",\"placement\":\"" << placement_name << "\""
      << ",\"actual_placement\":\"" << placement_name << "\""
      << ",\"sidecar_leaf_id\":" << placement.sidecar_leaf_id
      << ",\"certificate_ok\":" << (placement.certificate.ok ? "true" : "false")
      << ",\"certificate_sidecar_leaf_id\":" << placement.certificate.sidecar_leaf_id
      << ",\"certificate_failure_reason\":\""
      << json_escape(placement.certificate.failure_reason) << "\""
      << ",\"certificate_matching_children\":" << placement.certificate.matching_children
      << ",\"certificate_children\":[";
  for (std::size_t i = 0; i < placement.certificate.levels.size(); ++i) {
    if (i) out << ',';
    out << placement.certificate.levels[i].child;
  }
  out << "]"
      << ",\"safe_policy_requires_delta\":true"
      << ",\"runtime_role\":\"guard_closed_delta\""
      << ",\"fixture_role_source_mirror_only\":true"
      << ",\"direct_runtime_guard_open\":false"
      << ",\"active_ids_sha256\":\""
      << local_stable_set_sha(
             std::vector<StableId>(active.begin(), active.end()))
      << "\"";
  if (event.op == "insert") {
    out << ",\"fallback_reason\":\"" << json_escape(placement.fallback_reason) << "\"";
  } else {
    out << ",\"prior_placement\":\"" << placement_name << "\"";
  }
  out << "}";
  return out.str();
}

void require_native_certificate_contract(const TraceOp& event,
                                         const Placement& placement) {
  // These checks consume the certificate returned from the actual frozen native
  // snapshot, not the source-mirror selection record.  They deliberately
  // preserve the safe runtime policy: every row remains delta while direct is
  // closed, including the fixture's historical capacity-labelled candidate.
  require(placement.kind == PlacementKind::kDelta && placement.sidecar_leaf_id == -1,
          "closed direct guard must leave every insertion in global delta");
  switch (event.op_index) {
    case 0:
    case 1:
    case 4: {
      constexpr int expected_children[] = {1, 13, 131};
      constexpr int expected_parents[] = {0, 1, 13};
      require(placement.certificate.ok &&
                  placement.certificate.every_predicate_branch_mirrored &&
                  placement.certificate.all_include_branch_not_relied_on &&
                  placement.certificate.sidecar_leaf_id == 131,
              "native strict certificate failed its fixed sibling-min witness contract");
      require(placement.certificate.levels.size() == std::size(expected_children),
              "native strict certificate has wrong branch-witness depth");
      for (std::size_t i = 0; i < std::size(expected_children); ++i) {
        const auto& level = placement.certificate.levels[i];
        require(level.parent_leaf_or_internal == expected_parents[i] &&
                    level.child == expected_children[i] &&
                    !level.last_child &&
                    level.next_sibling == expected_children[i] + 1 &&
                    level.predicate_branch ==
                        safe_c1_g3::GtsKnnPruningBranch::kPredicateNonLastSibling,
                "native strict certificate has wrong sibling-min branch path");
      }
      require(placement.fallback_reason == "direct_visibility_runtime_guard_closed",
              "closed direct guard did not produce its explicit fallback reason");
      break;
    }
    case 13:
    case 16:
    case 17:
      require(!placement.certificate.ok &&
                  placement.certificate.matching_children == 0 &&
                  placement.certificate.failure_reason == "native_sibling_gap_or_boundary" &&
                  placement.fallback_reason == "native_sibling_gap_or_boundary",
              "boundary/gap fixture did not fail through the precise native certificate path");
      break;
    default:
      runner_fail("unexpected insertion op_index in fixed certificate contract");
  }
}

void emit_engine_record(std::ofstream* output, int expected_op_index,
                        const std::string& json) {
  require(output != nullptr && static_cast<bool>(*output), "engine trace stream is unavailable");
  const std::string needle = "\"op_index\":" + std::to_string(expected_op_index);
  require(json.find(needle) != std::string::npos,
          "engine record does not bind expected op index");
  *output << json << '\n';
  require(static_cast<bool>(*output), "failed while writing engine trace");
}

void write_text_file(const fs::path& path, const std::string& text) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  require(static_cast<bool>(output), "cannot create output: " + path.string());
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  output.flush();
  require(static_cast<bool>(output), "cannot write output: " + path.string());
}

struct RunPaths final {
  fs::path final_dir;
  fs::path staging_dir;
};

RunPaths create_staging_directory_or_fail(int argc, char** argv) {
  require(argc == 3 && std::string_view(argv[1]) == "--run-dir",
          "usage is exactly: g3_4k_pilot_wrapper --run-dir <absolute-v4-runs-child>");
  const fs::path root = fs::weakly_canonical(fs::path(kReleaseRoot));
  require(root == fs::path(kReleaseRoot), "release root must not be symlinked/relocated");
  const fs::path runs_root = root / "runs";
  const fs::file_status runs_status = fs::symlink_status(runs_root);
  require(!fs::is_symlink(runs_status) && fs::is_directory(runs_status),
          "release runs directory must be a real directory");
  const fs::path final_dir = fs::absolute(fs::path(argv[2])).lexically_normal();
  require(final_dir.parent_path() == runs_root,
          "run directory must be one new direct child of the fixed v4 runs directory");
  require(!final_dir.filename().empty() && final_dir.filename() != "." &&
              final_dir.filename() != "..",
          "run directory name is invalid");
  require(!fs::exists(final_dir), "final run directory already exists; no reuse/retry is allowed");
  // Archive CHECK() can call exit(1), bypassing the C++ catch below.  Therefore
  // all intermediate bytes live only in a hidden sibling staging directory;
  // final_dir does not exist unless this process closes every output and
  // atomically renames the whole directory at success.
  const fs::path staging_dir = runs_root / (".staging-" + final_dir.filename().string());
  require(!fs::exists(staging_dir), "staging directory already exists; inspect it, do not overwrite");
  require(fs::create_directory(staging_dir), "cannot create an exclusive hidden staging directory");
  return RunPaths{final_dir, staging_dir};
}

void require_topology_receipt(const RebuildReceipt& receipt, const std::string& phase,
                              const std::set<StableId>& active,
                              const std::string& expected_live_sha) {
  require(receipt.tree_version > 0, phase + " receipt has zero tree version");
  require(receipt.tree_height == kExpectedTreeHeight,
          phase + " actual tree height does not equal fixed source-height contract");
  require(receipt.node_capacity > 0 && receipt.nonempty_node_count > 0 &&
              receipt.nonempty_node_count <= receipt.node_capacity,
          phase + " receipt has invalid native capacity/occupancy facts");
  require(receipt.fanout == kExpectedFanout,
          phase + " actual fanout does not equal pinned GTS order");
  require(receipt.base_count == static_cast<int>(active.size()),
          phase + " base count differs from independently maintained active set");
  require(receipt.sidecar_live == 0 && receipt.delta_live == 0,
          phase + " rebuild did not clear dynamic tiers");
  require(receipt.dynamic_tiers_replaced_after_generation_publish,
          phase + " receipt does not attest cleared dynamic tiers");
  require(receipt.live_ids_sha256 == expected_live_sha,
          phase + " receipt live-ID hash mismatch");
  require(receipt.live_ids_sha256 ==
              local_stable_set_sha(std::vector<StableId>(active.begin(), active.end())),
          phase + " receipt live-ID hash differs from independent active set");
}

void compare_trace_query_to_independent_oracle(
    const TraceOp& event, const QueryExport& actual, const FixtureData& fixture,
    const std::set<StableId>& active, const std::map<int, OracleRecord>& persisted_oracle) {
  const bool is_knn = event.op == "knn";
  const std::vector<StableDistance> independent = exact_oracle(
      fixture, active, event.query_id, is_knn, kK, event.radius_sq);
  const auto persisted = persisted_oracle.find(event.op_index);
  require(persisted != persisted_oracle.end(), "trace query is absent from persisted CPU oracle");
  require(persisted->second.active_ids_sha256 ==
              local_stable_set_sha(std::vector<StableId>(active.begin(), active.end())),
          "persisted oracle active-set digest differs from independent active set");
  require_equal_results(independent, persisted->second.results,
                        "independent recomputation versus persisted CPU oracle");
  require_equal_results(actual.results, independent,
                        "native Matrix query versus independent signed-int64 oracle");
  require(actual.active_stable_ids_sha256 == persisted->second.active_ids_sha256,
          "Matrix query active-set digest differs from independently maintained state");
  require(!actual.full_frozen_base_snapshot_integrity_audit_on_operation &&
              actual.control_plane_partition_metadata_audit_on_operation &&
              actual.runner_single_image_admission_required,
          "Matrix query integrity scope drifted from lightweight-identity/single-image contract");
  if (is_knn) {
    require(actual.kind == "knn" && actual.requested_k == kK && actual.radius_sq == 0,
            "Matrix KNN query contract drift");
  } else {
    require(actual.kind == "range" && actual.requested_k == 0 &&
                actual.radius_sq == event.radius_sq,
            "Matrix range query contract drift");
    // This asserts result filtering only.  It makes no archive-native or
    // direct-range assertion; direct guard remains closed for the entire run.
    require(actual.range_predicate_mirror_branch_aligned,
            "Matrix range path did not identify its branch-mirror scope");
  }
}

void consume_post_rebuild_ticket_or_fail(NativeSafeC1Matrix* matrix,
                                         IssuedQuery* issued,
                                         const std::vector<StableDistance>& independent,
                                         const std::string& phase,
                                         int op_index,
                                         std::vector<std::string>* ticket_records) {
  require(matrix != nullptr && issued != nullptr && ticket_records != nullptr,
          "ticket consumption received null runner state");
  require(issued->post_rebuild_ticket.has_value(),
          "required post-rebuild opaque ticket was not issued");
  const QueryExport& export_data = issued->export_data;
  auto ticket = std::move(*issued->post_rebuild_ticket);
  matrix->verify_first_post_rebuild_query(std::move(ticket), independent);
  ticket_records->push_back(ticket_attestation_json(phase, op_index, export_data, independent));
}

}  // namespace g3_4k_pilot

int main(int argc, char** argv) {
  using g3_4k_pilot::EngineMode;
  using g3_4k_pilot::IssuedQuery;
  using g3_4k_pilot::NativeSafeC1Matrix;
  using g3_4k_pilot::Placement;
  using g3_4k_pilot::PlacementKind;
  using g3_4k_pilot::RebuildReceipt;
  using g3_4k_pilot::StableDistance;
  using g3_4k_pilot::StableId;
  g3_4k_pilot::RunPaths run_paths;
  try {
    // All strict byte/source/fixture validation is done before the Matrix
    // constructor and before any CUDA/native API can be reached.
    const std::filesystem::path root = std::filesystem::weakly_canonical(std::filesystem::path(g3_4k_pilot::kReleaseRoot));
    g3_4k_pilot::require(root == std::filesystem::path(g3_4k_pilot::kReleaseRoot),
               "release root must have the fixed canonical absolute path");
    g3_4k_pilot::require_all_hashes(root, g3_4k_pilot::kSourceClosure, std::size(g3_4k_pilot::kSourceClosure));
    const std::filesystem::path fixture = root / g3_4k_pilot::kFixtureRelative;
    const std::filesystem::path bootstrap = root / g3_4k_pilot::kBootstrapRelative;
    g3_4k_pilot::require_all_hashes(fixture, g3_4k_pilot::kFixtureFiles, std::size(g3_4k_pilot::kFixtureFiles));
    g3_4k_pilot::require_all_hashes(bootstrap, g3_4k_pilot::kBootstrapFiles, std::size(g3_4k_pilot::kBootstrapFiles));
    g3_4k_pilot::validate_manifest_and_bootstrap_contract(fixture, bootstrap);
    const g3_4k_pilot::FixtureData fixture_data = g3_4k_pilot::load_and_validate_binary_fixture(fixture);
    const std::vector<g3_4k_pilot::TraceOp> trace = g3_4k_pilot::parse_and_validate_fixed_trace(fixture);
    const std::map<int, g3_4k_pilot::OracleRecord> persisted_oracle =
        g3_4k_pilot::parse_and_validate_trace_oracle(fixture, trace);
    g3_4k_pilot::require(!safe_c1_g3::G3_DIRECT_SIDECAR_RUNTIME_GUARD_OPEN,
               "controlled pilot refuses any direct-sidecar runtime guard");
    g3_4k_pilot::require(!safe_c1_g3::G3_NATIVE_RANGE_RECEIPT_IMPLEMENTED &&
                   safe_c1_g3::G3_RANGE_BRANCH_ALIGNED_VECTOR_PREDICATE_MIRROR_IMPLEMENTED,
               "controlled range scope drifted from branch-mirror-only contract");
    g3_4k_pilot::require(safe_c1_g3::G3_LIGHTWEIGHT_GENERATION_IDENTITY_GUARD_IMPLEMENTED &&
                   !safe_c1_g3::G3_FULL_FROZEN_BASE_SNAPSHOT_ON_EACH_OPERATION &&
                   safe_c1_g3::G3_RUNNER_SINGLE_IMAGE_ADMISSION_REQUIRED,
               "controlled pilot requires lightweight identity guard plus single-image admission");

    run_paths = g3_4k_pilot::create_staging_directory_or_fail(argc, argv);
    const std::filesystem::path engine_tmp = run_paths.staging_dir / ".engine.jsonl.tmp";
    const std::filesystem::path bootstrap_tmp = run_paths.staging_dir / ".bootstrap_attestation.json.tmp";
    const std::filesystem::path tickets_tmp = run_paths.staging_dir / ".ticket_attestation.jsonl.tmp";
    std::ofstream engine(engine_tmp, std::ios::binary | std::ios::trunc);
    g3_4k_pilot::require(static_cast<bool>(engine), "cannot create temporary engine trace");

    // Independent state is deliberately owned by this wrapper, not read from
    // Matrix internals.  It is the sole source for CPU exact expectations.
    std::set<StableId> active(fixture_data.initial_base.begin(), fixture_data.initial_base.end());
    g3_4k_pilot::require(static_cast<int>(active.size()) == g3_4k_pilot::kBaseCount,
               "independent initial active set cardinality drift");
    g3_4k_pilot::require(local_stable_set_sha(
                   std::vector<StableId>(active.begin(), active.end())) == g3_4k_pilot::kInitialLiveSha,
               "independent initial active digest drift");

    NativeSafeC1Matrix matrix(/*sidecar_leaf_capacity=*/2, /*requested_k=*/g3_4k_pilot::kK);
    matrix.initialize_immutable_pool(g3_4k_pilot::kDimension, fixture_data.pool,
                                     fixture_data.stable_to_pool_row);

    // Bootstrap does not carry an engine op_index and must never be appended to
    // engine.jsonl.  It gates the initial build exactly as the op18 rebuild
    // later gates its new generation.
    const RebuildReceipt initial_build = matrix.build_initial_base(fixture_data.initial_base);
    g3_4k_pilot::require_topology_receipt(initial_build, "initial bootstrap build", active, g3_4k_pilot::kInitialLiveSha);
    g3_4k_pilot::require(matrix.post_rebuild_oracle_pending(),
               "initial build did not require KNN/range ticket verification");

    std::vector<std::string> ticket_records;
    const std::vector<StableDistance> bootstrap_knn_independent =
        g3_4k_pilot::exact_oracle(fixture_data, active, /*query_id=*/1, /*is_knn=*/true, g3_4k_pilot::kK, 0);
    g3_4k_pilot::require_equal_results(bootstrap_knn_independent, g3_4k_pilot::bootstrap_knn_expected_literal(),
                             "bootstrap KNN independent oracle");
    IssuedQuery bootstrap_knn = matrix.query_knn(1, g3_4k_pilot::query_ptr(fixture_data, 1), g3_4k_pilot::kK);
    g3_4k_pilot::require_equal_results(bootstrap_knn.export_data.results, bootstrap_knn_independent,
                             "bootstrap KNN Matrix result");
    g3_4k_pilot::consume_post_rebuild_ticket_or_fail(&matrix, &bootstrap_knn,
                                           bootstrap_knn_independent, "bootstrap", -1,
                                           &ticket_records);
    g3_4k_pilot::require(matrix.post_rebuild_oracle_pending(),
               "range ticket unexpectedly absent after only bootstrap KNN verification");

    const std::vector<StableDistance> bootstrap_range_independent =
        g3_4k_pilot::exact_oracle(fixture_data, active, /*query_id=*/2, /*is_knn=*/false, 0,
                        g3_4k_pilot::kBootstrapRangeRadius);
    g3_4k_pilot::require_equal_results(bootstrap_range_independent, g3_4k_pilot::bootstrap_range_expected_literal(),
                             "bootstrap range independent oracle");
    IssuedQuery bootstrap_range =
        matrix.query_range(2, g3_4k_pilot::query_ptr(fixture_data, 2), g3_4k_pilot::kBootstrapRangeRadius);
    g3_4k_pilot::require_equal_results(bootstrap_range.export_data.results, bootstrap_range_independent,
                             "bootstrap range Matrix result");
    g3_4k_pilot::consume_post_rebuild_ticket_or_fail(&matrix, &bootstrap_range,
                                           bootstrap_range_independent, "bootstrap", -1,
                                           &ticket_records);
    g3_4k_pilot::require(!matrix.post_rebuild_oracle_pending() && matrix.mode() == EngineMode::kReady,
               "initial bootstrap tickets did not return Matrix to Ready");

    std::ostringstream bootstrap_attestation;
    bootstrap_attestation
        << "{\"record\":\"bootstrap_attestation\",\"engine_trace_excludes_bootstrap\":true,"
        << "\"range_scope\":\"branch_aligned_predicate_mirror_exact_filtered_not_archive_native\","
        << "\"direct_runtime_guard_open\":false,\"initial_rebuild\":"
        << safe_c1_g3::serialize_rebuild_jsonl_record(-1, initial_build)
        << ",\"initial_active_ids_sha256\":\"" << g3_4k_pilot::kInitialLiveSha << "\""
        << ",\"knn_query_id\":1,\"knn_result_sha256\":\""
        << g3_4k_pilot::result_sha_for_attestation(bootstrap_knn_independent) << "\""
        << ",\"range_query_id\":2,\"range_radius_sq\":"
        << static_cast<unsigned long long>(g3_4k_pilot::kBootstrapRangeRadius)
        << ",\"range_result_sha256\":\""
        << g3_4k_pilot::result_sha_for_attestation(bootstrap_range_independent) << "\""
        << "}";

    for (const g3_4k_pilot::TraceOp& event : trace) {
      if (event.op == "insert") {
        g3_4k_pilot::require(active.find(event.stable_id) == active.end(),
                   "independent active set already contains insertion ID");
        const Placement placement = matrix.insert(event.stable_id);
        g3_4k_pilot::require_native_certificate_contract(event, placement);
        active.insert(event.stable_id);
        g3_4k_pilot::emit_engine_record(&engine, event.op_index,
                              g3_4k_pilot::mutation_attestation_json(event, placement, active));
      } else if (event.op == "delete") {
        g3_4k_pilot::require(active.find(event.stable_id) != active.end(),
                   "independent active set lacks deletion ID");
        const Placement prior = matrix.erase_mutable(event.stable_id);
        g3_4k_pilot::require(prior.kind == PlacementKind::kDelta,
                   "safe policy requires every current mutable deletion to remove delta");
        active.erase(event.stable_id);
        g3_4k_pilot::emit_engine_record(&engine, event.op_index,
                              g3_4k_pilot::mutation_attestation_json(event, prior, active));
      } else if (event.op == "knn" || event.op == "range") {
        if (event.op == "knn") {
          IssuedQuery issued = matrix.query_knn(event.query_id,
                                                g3_4k_pilot::query_ptr(fixture_data, event.query_id), g3_4k_pilot::kK);
          g3_4k_pilot::compare_trace_query_to_independent_oracle(
              event, issued.export_data, fixture_data, active, persisted_oracle);
          if (event.op_index == 19) {
            const std::vector<StableDistance> independent =
                g3_4k_pilot::exact_oracle(fixture_data, active, event.query_id, true, g3_4k_pilot::kK, 0);
            g3_4k_pilot::consume_post_rebuild_ticket_or_fail(&matrix, &issued, independent,
                                                   "post_rebuild", event.op_index,
                                                   &ticket_records);
          } else {
            g3_4k_pilot::require(!issued.post_rebuild_ticket.has_value(),
                       "non-post-rebuild trace KNN unexpectedly issued a ticket");
          }
          g3_4k_pilot::emit_engine_record(&engine, event.op_index,
                                safe_c1_g3::serialize_query_jsonl_record(event.op_index,
                                                                          issued.export_data));
        } else {
          IssuedQuery issued = matrix.query_range(event.query_id,
                                                  g3_4k_pilot::query_ptr(fixture_data, event.query_id),
                                                  event.radius_sq);
          g3_4k_pilot::compare_trace_query_to_independent_oracle(
              event, issued.export_data, fixture_data, active, persisted_oracle);
          if (event.op_index == 20) {
            const std::vector<StableDistance> independent =
                g3_4k_pilot::exact_oracle(fixture_data, active, event.query_id, false, 0, event.radius_sq);
            g3_4k_pilot::consume_post_rebuild_ticket_or_fail(&matrix, &issued, independent,
                                                   "post_rebuild", event.op_index,
                                                   &ticket_records);
            g3_4k_pilot::require(!matrix.post_rebuild_oracle_pending() && matrix.mode() == EngineMode::kReady,
                       "post-rebuild KNN/range tickets did not return Matrix to Ready");
          } else {
            g3_4k_pilot::require(!issued.post_rebuild_ticket.has_value(),
                       "non-post-rebuild trace range unexpectedly issued a ticket");
          }
          g3_4k_pilot::emit_engine_record(&engine, event.op_index,
                                safe_c1_g3::serialize_query_jsonl_record(event.op_index,
                                                                          issued.export_data));
        }
      } else if (event.op == "rebuild") {
        // The archived trace's old direct-role selection expected three delta
        // rows before rebuild.  In this safer pilot direct is disabled, so all
        // four surviving mutable IDs are delta before Matrix rebuild.  That
        // legacy source-role count is intentionally not treated as a success
        // criterion; only the independently maintained live set is.
        g3_4k_pilot::require(active.size() == 4100,
                   "safe-policy trace must have 4100 live IDs before rebuild");
        const RebuildReceipt rebuilt = matrix.rebuild_from_current_live();
        g3_4k_pilot::require_topology_receipt(rebuilt, "trace rebuild", active, g3_4k_pilot::kPostRebuildLiveSha);
        g3_4k_pilot::require(matrix.post_rebuild_oracle_pending(),
                   "trace rebuild did not require new KNN/range ticket verification");
        g3_4k_pilot::emit_engine_record(&engine, event.op_index,
                              safe_c1_g3::serialize_rebuild_jsonl_record(event.op_index, rebuilt));
      } else {
        g3_4k_pilot::runner_fail("fixed trace dispatch encountered unknown operation");
      }
    }

    engine.flush();
    g3_4k_pilot::require(static_cast<bool>(engine), "cannot flush temporary engine trace");
    engine.close();
    g3_4k_pilot::require(static_cast<bool>(engine), "cannot close temporary engine trace");
    g3_4k_pilot::require(ticket_records.size() == 4,
               "must consume exactly two bootstrap and two post-rebuild tickets");

    std::ostringstream tickets;
    for (const std::string& record : ticket_records) tickets << record << '\n';
    g3_4k_pilot::write_text_file(bootstrap_tmp, bootstrap_attestation.str());
    g3_4k_pilot::write_text_file(tickets_tmp, tickets.str());
    std::filesystem::rename(engine_tmp, run_paths.staging_dir / "engine.jsonl");
    std::filesystem::rename(bootstrap_tmp, run_paths.staging_dir / "bootstrap_attestation.json");
    std::filesystem::rename(tickets_tmp, run_paths.staging_dir / "ticket_attestation.jsonl");
    std::filesystem::rename(run_paths.staging_dir, run_paths.final_dir);
    return 0;
  } catch (const std::exception& error) {
    // A C++ exception never publishes a final result; archive CHECK exit(1)
    // may leave hidden staging bytes for inspection, but cannot create final_dir.
    if (!run_paths.staging_dir.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(run_paths.staging_dir, ignored);
    }
    std::cerr << error.what() << '\n';
    return 70;
  }
}
