// Safe-C2 v3 speculative gamma traversal plus gamma-only fallback skeleton.
// This isolated implementation never mutates v2.  V2 sealed test material is
// historical provenance only and is explicitly forbidden for v3 tuning/runs.
// V3 binds a separate compact workload manifest before any executable run.
//
// Scope boundary:
//   * Safe-C2 only: static vector-KNN GTS with a corrected Eq. (1) interval
//     traversal and Eq. (2) per-level gamma scaling.
//   * This is NOT the submitted implementation: the submitted/legacy vector
//     kernel used the next sibling min boundary rather than max_dis_d.
//   * No update code, incremental insert, range exporter, or dynamic exact smoke;
//     neither original archive is written.
//   * gamma>1 is empirical recall-controlled pruning, not a universal theorem.
//
// This file and headers live only under c2_speculative_fallback_v3. `search_v3.cuh`
// retains the corrected two-sided interval test and adds diagnostics for every
// node kept by the unscaled test but removed by a gamma-scaled speculative test.
// A flagged query is recomputed with all gamma=1 on the same frozen snapshot and
// replaces its result before final accounting.  This is a guarded empirical
// design, not a proof that gamma>1 is a metric lower bound.

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <type_traits>
#include <vector>

#define RP_DEFINE_CONSTANTS
#include "residual_pruning.cuh"
#include "tree.cuh"
#include "search_v3.cuh"

// `config.cuh` deliberately maps the legacy token `short` to float. Safe-C2
// freezes that effective representation: raw fvec float32 is copied without a
// cast/quantization into all tree and vector-KNN APIs.
using SafeC2EffectiveCoordinate = short;
static_assert(std::is_same<SafeC2EffectiveCoordinate, float>::value,
              "Safe-C2 requires config.cuh effective coordinate type float32");

// The archived headers declare this host routine but define it only in main.cu.
// Keep a local implementation to upload a fully explicit C2 gamma vector.
void upload_rp_constants(float* h_alpha, float* h_beta, float* h_gamma,
                         int /*num_levels*/, float* h_lut_breaks,
                         float* h_lut_slopes, float* h_lut_intercepts,
                         int lut_size, int mode) {
  CHECK(cudaMemcpyToSymbol(c_rp_alpha, h_alpha, RP_MAX_LEVELS * sizeof(float), 0,
                           cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(c_rp_beta, h_beta, RP_MAX_LEVELS * sizeof(float), 0,
                           cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(c_rp_gamma, h_gamma, RP_MAX_LEVELS * sizeof(float), 0,
                           cudaMemcpyHostToDevice));
  if (h_lut_breaks != nullptr && h_lut_slopes != nullptr && h_lut_intercepts != nullptr) {
    CHECK(cudaMemcpyToSymbol(c_lut_breaks, h_lut_breaks, RP_LUT_SIZE * sizeof(float), 0,
                             cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(c_lut_slopes, h_lut_slopes, RP_LUT_SIZE * sizeof(float), 0,
                             cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(c_lut_intercepts, h_lut_intercepts, RP_LUT_SIZE * sizeof(float), 0,
                             cudaMemcpyHostToDevice));
  }
  CHECK(cudaMemcpyToSymbol(c_lut_num_segments, &lut_size, sizeof(int), 0,
                           cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(c_rp_mode, &mode, sizeof(int), 0,
                           cudaMemcpyHostToDevice));
}

namespace safe_c2 {

constexpr int kDimension = 128;
constexpr int kBaseN = 1'000'000;
constexpr int kQueryN = 10'000;
constexpr int kGtWidth = 100;
constexpr int kK = 10;
constexpr int kFreezeLevel = 2;  // levels 0..2 exact baseline
constexpr int kBinaryIterations = 16;
constexpr float kCandidateLo = 1.0F;
constexpr float kCandidateHi = 5.0F;
constexpr float kMu = 0.75F;
constexpr std::size_t kWorkspaceCapBytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr float kDistanceAbsFloor = 2.0e-3F;
constexpr int kDistanceMaxUlps = 2;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("SAFE-C2: " + message);
}

void cuda_or_die(cudaError_t status, const char* expression, const char* file, int line) {
  if (status == cudaSuccess) return;
  std::ostringstream out;
  out << "CUDA " << expression << " failed at " << file << ':' << line << ": "
      << cudaGetErrorString(status);
  fail(out.str());
}
#define C2_CUDA(expr) ::safe_c2::cuda_or_die((expr), #expr, __FILE__, __LINE__)

struct Args {
  std::string mode;  // calibrate or evaluate
  std::string base_fvecs;
  std::string query_fvecs;
  std::string groundtruth_ivecs;
  std::string query_ids;
  std::string output_dir;
  std::string gamma_vector_file;
  std::string stage;
  std::string workload_manifest;
  int warmup_reps = 1;
  int timed_reps = 1;
  long long baseline_target_correct = -1;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    auto next = [&]() -> std::string {
      if (++i >= argc) fail("missing value after " + key);
      return argv[i];
    };
    if (key == "--mode") args.mode = next();
    else if (key == "--base-fvecs") args.base_fvecs = next();
    else if (key == "--query-fvecs") args.query_fvecs = next();
    else if (key == "--groundtruth-ivecs") args.groundtruth_ivecs = next();
    else if (key == "--query-ids") args.query_ids = next();
    else if (key == "--out") args.output_dir = next();
    else if (key == "--gamma-vector-file") args.gamma_vector_file = next();
    else if (key == "--stage") args.stage = next();
    else if (key == "--workload-manifest") args.workload_manifest = next();
    else if (key == "--warmup-reps") args.warmup_reps = std::stoi(next());
    else if (key == "--timed-reps") args.timed_reps = std::stoi(next());
    else if (key == "--baseline-target-correct") args.baseline_target_correct = std::stoll(next());
    else if (key == "--help") {
      std::cout
          << "usage: GTS_c2_perlevel_sift1m --mode calibrate|evaluate --base-fvecs BASE "
          << "--query-fvecs QUERY --groundtruth-ivecs GT --query-ids IDS --out DIR "
          << "--stage calibration|validation|test --workload-manifest MANIFEST "
          << "[--gamma-vector-file FILE] "
          << "[--warmup-reps 1] [--timed-reps N]\n";
      std::exit(0);
    } else {
      fail("unknown argument " + key);
    }
  }
  if (args.mode != "calibrate" && args.mode != "evaluate") fail("--mode must be calibrate or evaluate");
  if (args.base_fvecs.empty() || args.query_fvecs.empty() || args.groundtruth_ivecs.empty() ||
      args.query_ids.empty() || args.output_dir.empty() || args.stage.empty() || args.workload_manifest.empty()) {
    fail("missing required input/output argument");
  }
  if (args.warmup_reps < 0 || args.timed_reps <= 0) fail("invalid repetition counts");
  if (args.mode == "calibrate") {
    if (args.stage != "calibration") fail("calibrate mode requires --stage calibration");
    if (!args.gamma_vector_file.empty() || args.baseline_target_correct != -1) {
      fail("calibrate mode derives its own all-ones baseline/vector");
    }
  } else {
    if (args.gamma_vector_file.empty()) {
      fail("evaluate mode requires --gamma-vector-file");
    }
    if (args.baseline_target_correct != -1) {
      fail("paired evaluate derives a same-split all-ones baseline; do not import a calibration target");
    }
    if (args.stage != "validation" && args.stage != "test") {
      fail("evaluate mode requires validation or test stage");
    }
  }
  return args;
}

// Small self-contained SHA-256 implementation used only to bind a run to the
// independently frozen compact workload manifest.  Do not substitute a shell
// hash command: a missing/aliased utility must fail closed rather than weaken
// the experimental protocol.
class Sha256 {
 public:
  Sha256() { reset(); }

  void update(const unsigned char* input, std::size_t length) {
    for (std::size_t i = 0; i < length; ++i) {
      block_[block_length_++] = input[i];
      if (block_length_ == block_.size()) {
        transform();
        bit_length_ += 512;
        block_length_ = 0;
      }
    }
  }

  std::array<unsigned char, 32> final() {
    const std::uint64_t total_bits = bit_length_ + static_cast<std::uint64_t>(block_length_) * 8ULL;
    std::size_t i = block_length_;
    block_[i++] = 0x80U;
    if (i > 56) {
      while (i < 64) block_[i++] = 0;
      transform();
      i = 0;
    }
    while (i < 56) block_[i++] = 0;
    for (int byte = 7; byte >= 0; --byte) {
      block_[i++] = static_cast<unsigned char>((total_bits >> (byte * 8)) & 0xffU);
    }
    transform();
    std::array<unsigned char, 32> output{};
    for (std::size_t word = 0; word < state_.size(); ++word) {
      output[word * 4] = static_cast<unsigned char>((state_[word] >> 24) & 0xffU);
      output[word * 4 + 1] = static_cast<unsigned char>((state_[word] >> 16) & 0xffU);
      output[word * 4 + 2] = static_cast<unsigned char>((state_[word] >> 8) & 0xffU);
      output[word * 4 + 3] = static_cast<unsigned char>(state_[word] & 0xffU);
    }
    reset();
    return output;
  }

 private:
  static std::uint32_t rotr(std::uint32_t value, std::uint32_t shift) {
    return (value >> shift) | (value << (32U - shift));
  }

  void reset() {
    block_length_ = 0;
    bit_length_ = 0;
    state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
              0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  }

  void transform() {
    static constexpr std::array<std::uint32_t, 64> k = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i) {
      w[i] = (static_cast<std::uint32_t>(block_[i * 4]) << 24) |
             (static_cast<std::uint32_t>(block_[i * 4 + 1]) << 16) |
             (static_cast<std::uint32_t>(block_[i * 4 + 2]) << 8) |
             static_cast<std::uint32_t>(block_[i * 4 + 3]);
    }
    for (std::size_t i = 16; i < w.size(); ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (std::size_t i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t choose = (e & f) ^ ((~e) & g);
      const std::uint32_t t1 = h + s1 + choose + k[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = s0 + majority;
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
    state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
  }

  std::array<unsigned char, 64> block_{};
  std::array<std::uint32_t, 8> state_{};
  std::size_t block_length_ = 0;
  std::uint64_t bit_length_ = 0;
};

std::string sha256_hex_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open workload-bound input " + path);
  Sha256 sha;
  std::array<char, 65536> buffer{};
  while (input.good()) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      sha.update(reinterpret_cast<const unsigned char*>(buffer.data()), static_cast<std::size_t>(count));
    }
  }
  if (!input.eof()) fail("failed while hashing workload-bound input " + path);
  const auto digest = sha.final();
  static constexpr char kHex[] = "0123456789abcdef";
  std::string text;
  text.reserve(digest.size() * 2);
  for (unsigned char byte : digest) {
    text.push_back(kHex[(byte >> 4) & 0x0fU]);
    text.push_back(kHex[byte & 0x0fU]);
  }
  return text;
}

std::string read_text_file_strict(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open workload manifest " + path);
  std::ostringstream out;
  out << input.rdbuf();
  if (!input.good() && !input.eof()) fail("cannot read workload manifest " + path);
  return out.str();
}

std::string manifest_string_field(const std::string& text, const std::string& field) {
  const std::string marker = "\"" + field + "\"";
  const std::size_t marker_pos = text.find(marker);
  if (marker_pos == std::string::npos) fail("workload manifest lacks string field " + field);
  std::size_t pos = text.find(':', marker_pos + marker.size());
  if (pos == std::string::npos) fail("malformed workload manifest field " + field);
  ++pos;
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
  if (pos >= text.size() || text[pos] != '\"') fail("workload manifest field is not a JSON string " + field);
  ++pos;
  const std::size_t end = text.find('\"', pos);
  if (end == std::string::npos) fail("unterminated workload manifest field " + field);
  return text.substr(pos, end - pos);
}

bool manifest_true_field(const std::string& text, const std::string& field) {
  const std::string marker = "\"" + field + "\"";
  const std::size_t marker_pos = text.find(marker);
  if (marker_pos == std::string::npos) return false;
  std::size_t pos = text.find(':', marker_pos + marker.size());
  if (pos == std::string::npos) return false;
  ++pos;
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
  return text.compare(pos, 4, "true") == 0;
}

bool is_sha256_hex(const std::string& text) {
  if (text.size() != 64) return false;
  for (unsigned char c : text) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  }
  return true;
}

bool references_v2_sealed_root(const std::string& path) {
  return path.find("/c2_interval_bound_safe_v2_tiefree/") != std::string::npos ||
         path.find("/c2_interval_bound_safe_v2_tiefree") != std::string::npos;
}

// This sealed digest is copied from v2 provenance.  The path guard rejects
// direct v2 use and the digest guard also rejects a copied old test-ID file.
constexpr const char* kV2SealedTestIdsSha256 = "50ccb28263bf23e499b9c50e4d3e9800fca3949aa5b5ea8a454237d5987701ce";

void validate_workload_binding(const Args& args) {
  const std::array<std::pair<const char*, const std::string*>, 5> input_paths = {{
      {"base_fvecs", &args.base_fvecs},
      {"query_fvecs", &args.query_fvecs},
      {"groundtruth_ivecs", &args.groundtruth_ivecs},
      {"query_ids", &args.query_ids},
      {"gamma_vector", &args.gamma_vector_file},
  }};
  for (const auto& entry : input_paths) {
    if (entry.second->empty()) continue;
    if (references_v2_sealed_root(*entry.second)) {
      fail(std::string("v2 sealed root is forbidden as v3 ") + entry.first + " input");
    }
  }
  if (!args.gamma_vector_file.empty() &&
      args.gamma_vector_file.find("/c2_speculative_fallback_v3/") == std::string::npos) {
    fail("evaluate gamma vector must be produced/stored under the v3 root");
  }
  if (references_v2_sealed_root(args.workload_manifest) ||
      args.workload_manifest.find("/c2_speculative_fallback_v3/") == std::string::npos) {
    fail("workload manifest must reside under the isolated v3 root, never v2");
  }
  if (references_v2_sealed_root(args.output_dir) ||
      args.output_dir.find("/c2_speculative_fallback_v3/") == std::string::npos) {
    fail("v3 output directory must reside under the isolated v3 root");
  }
  const std::string manifest = read_text_file_strict(args.workload_manifest);
  if (manifest_string_field(manifest, "schema") != "gts-v3-compact-learn-workload-v1") {
    fail("unexpected workload manifest schema");
  }
  if (!manifest_true_field(manifest, "sealed_v2_test_forbidden")) {
    fail("workload manifest must explicitly forbid v2 sealed test reuse");
  }
  if (manifest_string_field(manifest, "v2_sealed_test_ids_sha256_forbidden") !=
      kV2SealedTestIdsSha256) {
    fail("workload manifest v2 sealed-test digest binding mismatch");
  }
  const std::array<std::pair<const char*, const std::string*>, 3> stable_hashes = {{
      {"mapping_sha256", nullptr},
      {"calibration_ids_sha256", nullptr},
      {"validation_ids_sha256", nullptr},
  }};
  for (const auto& entry : stable_hashes) {
    if (!is_sha256_hex(manifest_string_field(manifest, entry.first))) {
      fail(std::string("workload manifest has invalid ") + entry.first);
    }
  }
  if (!is_sha256_hex(manifest_string_field(manifest, "test_ids_sha256"))) {
    fail("workload manifest has invalid test_ids_sha256");
  }
  const std::string mapping_path = manifest_string_field(manifest, "mapping_path");
  const std::string mapping_expected = manifest_string_field(manifest, "mapping_sha256");
  if (mapping_path.find("/c2_speculative_fallback_v3/") == std::string::npos ||
      references_v2_sealed_root(mapping_path) || !is_sha256_hex(mapping_expected)) {
    fail("workload mapping path/digest is not an isolated v3 binding");
  }
  if (sha256_hex_file(mapping_path) != mapping_expected) {
    fail("workload local-to-original mapping digest mismatch");
  }
  const std::string stage_field = args.stage + "_ids_sha256";
  const std::array<std::pair<std::string, const std::string*>, 4> expected_hashes = {{
      {"base_fvecs_sha256", &args.base_fvecs},
      {"query_fvecs_sha256", &args.query_fvecs},
      {"groundtruth_ivecs_sha256", &args.groundtruth_ivecs},
      {stage_field, &args.query_ids},
  }};
  for (const auto& entry : expected_hashes) {
    const std::string expected = manifest_string_field(manifest, entry.first);
    if (!is_sha256_hex(expected)) fail("workload manifest has invalid digest field " + entry.first);
    const std::string observed = sha256_hex_file(*entry.second);
    if (observed != expected) fail("workload input digest mismatch for " + entry.first);
    if (observed == kV2SealedTestIdsSha256) {
      fail("v2 sealed test IDs digest is forbidden for every v3 stage");
    }
  }
}


struct Fvecs {
  int dimension = 0;
  int count = 0;
  std::vector<float> values;
  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();
};

Fvecs read_fvecs(const std::string& path, int required_count, int required_dimension) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open fvecs " + path);
  input.seekg(0, std::ios::end);
  const std::streamoff bytes = input.tellg();
  input.seekg(0, std::ios::beg);
  int dimension = 0;
  input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
  if (!input || dimension != required_dimension) fail("unexpected fvecs dimension in " + path);
  const std::streamoff record_bytes = static_cast<std::streamoff>((dimension + 1) * sizeof(float));
  if (bytes <= 0 || bytes % record_bytes != 0 || bytes / record_bytes != required_count) {
    fail("unexpected fvecs count/layout in " + path);
  }
  input.seekg(0, std::ios::beg);
  Fvecs out;
  out.dimension = dimension;
  out.count = required_count;
  out.values.resize(static_cast<std::size_t>(required_count) * dimension);
  for (int row = 0; row < required_count; ++row) {
    int row_dimension = 0;
    input.read(reinterpret_cast<char*>(&row_dimension), sizeof(row_dimension));
    if (!input || row_dimension != dimension) fail("inconsistent fvecs row header at " + std::to_string(row));
    float* destination = out.values.data() + static_cast<std::size_t>(row) * dimension;
    input.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(dimension * sizeof(float)));
    if (!input) fail("truncated fvecs payload at " + std::to_string(row));
    for (int d = 0; d < dimension; ++d) {
      if (!std::isfinite(destination[d])) fail("non-finite fvecs value");
      out.min_value = std::min(out.min_value, destination[d]);
      out.max_value = std::max(out.max_value, destination[d]);
    }
  }
  char extra = 0;
  if (input.read(&extra, 1)) fail("extra fvecs payload");
  return out;
}

std::vector<int> read_ivecs(const std::string& path, int required_count, int required_width) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open ivecs " + path);
  input.seekg(0, std::ios::end);
  const std::streamoff bytes = input.tellg();
  input.seekg(0, std::ios::beg);
  int width = 0;
  input.read(reinterpret_cast<char*>(&width), sizeof(width));
  if (!input || width != required_width) fail("unexpected ivecs width");
  const std::streamoff record_bytes = static_cast<std::streamoff>((width + 1) * sizeof(int));
  if (bytes <= 0 || bytes % record_bytes != 0 || bytes / record_bytes != required_count) {
    fail("unexpected ivecs count/layout");
  }
  input.seekg(0, std::ios::beg);
  std::vector<int> out(static_cast<std::size_t>(required_count) * width);
  for (int row = 0; row < required_count; ++row) {
    int row_width = 0;
    input.read(reinterpret_cast<char*>(&row_width), sizeof(row_width));
    if (!input || row_width != width) fail("inconsistent ivecs row header at " + std::to_string(row));
    int* destination = out.data() + static_cast<std::size_t>(row) * width;
    input.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(width * sizeof(int)));
    if (!input) fail("truncated ivecs payload at " + std::to_string(row));
    std::unordered_set<int> seen;
    for (int j = 0; j < width; ++j) {
      if (destination[j] < 0 || destination[j] >= kBaseN) fail("out-of-range GT id");
      if (!seen.insert(destination[j]).second) fail("duplicate GT id in row " + std::to_string(row));
    }
  }
  char extra = 0;
  if (input.read(&extra, 1)) fail("extra ivecs payload");
  return out;
}

std::vector<int> read_query_ids(const std::string& path) {
  std::ifstream input(path);
  if (!input) fail("cannot open query IDs " + path);
  std::vector<int> ids;
  std::unordered_set<int> seen;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::size_t consumed = 0;
    int id = 0;
    try { id = std::stoi(line, &consumed); } catch (...) { fail("invalid query ID line"); }
    if (consumed != line.size() || id < 0 || id >= kQueryN || !seen.insert(id).second) {
      fail("invalid/duplicate query ID");
    }
    ids.push_back(id);
  }
  if (ids.empty()) fail("empty query ID list");
  return ids;
}

std::array<float, RP_MAX_LEVELS> read_gamma_vector(const std::string& path) {
  std::ifstream input(path);
  if (!input) fail("cannot open gamma vector " + path);
  std::array<float, RP_MAX_LEVELS> gamma{};
  std::string tag;
  for (int level = 0; level < RP_MAX_LEVELS; ++level) {
    int found_level = -1;
    if (!(input >> found_level >> gamma[level]) || found_level != level || !std::isfinite(gamma[level]) || gamma[level] < 1.0F) {
      fail("invalid gamma vector file");
    }
  }
  if (input >> tag) fail("extra gamma vector data");
  return gamma;
}

void write_gamma_vector(const std::string& path, const std::array<float, RP_MAX_LEVELS>& gamma) {
  std::ofstream out(path);
  if (!out) fail("cannot write gamma vector");
  out << std::setprecision(9);
  for (int level = 0; level < RP_MAX_LEVELS; ++level) out << level << ' ' << gamma[level] << '\n';
  if (!out) fail("failed writing gamma vector");
}

struct MetricPlan {
  long double bbox_squared = 0.0L;
  long double bbox_l2 = 0.0L;
  int infi_dis = 10000;
};

MetricPlan metric_plan(const Fvecs& base, const Fvecs& queries, const std::vector<int>& selected_ids) {
  std::vector<float> lo(kDimension, std::numeric_limits<float>::infinity());
  std::vector<float> hi(kDimension, -std::numeric_limits<float>::infinity());
  const auto absorb = [&](const float* vector) {
    for (int d = 0; d < kDimension; ++d) { lo[d] = std::min(lo[d], vector[d]); hi[d] = std::max(hi[d], vector[d]); }
  };
  for (int i = 0; i < base.count; ++i) absorb(base.values.data() + static_cast<std::size_t>(i) * kDimension);
  for (int qid : selected_ids) absorb(queries.values.data() + static_cast<std::size_t>(qid) * kDimension);
  MetricPlan plan;
  for (int d = 0; d < kDimension; ++d) { const long double delta = static_cast<long double>(hi[d]) - lo[d]; plan.bbox_squared += delta * delta; }
  plan.bbox_l2 = std::sqrt(plan.bbox_squared);
  const long double with_margin = std::ceil(plan.bbox_l2) + 1024.0L;
  if (with_margin > static_cast<long double>(std::numeric_limits<int>::max())) fail("infi_dis overflow");
  plan.infi_dis = std::max(10000, static_cast<int>(with_margin));
  return plan;
}

struct Runtime {
  int* data_info = nullptr;
  float* data_d = nullptr;
  int* id_list = nullptr;
  TN* node_list = nullptr;
  int* max_node_num = nullptr;
  int* empty_list = nullptr;
  int tree_height = 0;
};

void release_tree(Runtime& runtime) {
  if (runtime.id_list) { cudaFree(runtime.id_list); runtime.id_list = nullptr; }
  if (runtime.node_list) { cudaFree(runtime.node_list); runtime.node_list = nullptr; }
  if (runtime.max_node_num) { cudaFree(runtime.max_node_num); runtime.max_node_num = nullptr; }
  if (runtime.empty_list) { cudaFree(runtime.empty_list); runtime.empty_list = nullptr; }
  if (runtime.data_d) { cudaFree(runtime.data_d); runtime.data_d = nullptr; }
  if (runtime.data_info) { cudaFree(runtime.data_info); runtime.data_info = nullptr; }
  if (max_dis_d) { cudaFree(max_dis_d); max_dis_d = nullptr; }
  if (split_list) { cudaFree(split_list); split_list = nullptr; }
  if (pid_list) { cudaFree(pid_list); pid_list = nullptr; }
  if (dis_list) { cudaFree(dis_list); dis_list = nullptr; }
  if (split_num) { cudaFree(split_num); split_num = nullptr; }
}

std::uint64_t fnv1a(const void* address, std::size_t bytes, std::uint64_t seed = 1469598103934665603ULL) {
  const auto* data = static_cast<const unsigned char*>(address);
  std::uint64_t h = seed;
  for (std::size_t i = 0; i < bytes; ++i) { h ^= static_cast<std::uint64_t>(data[i]); h *= 1099511628211ULL; }
  return h;
}

struct FrozenTreeSnapshot {
  int node_count = 0;
  std::uint64_t hash = 0;
  void capture(const Runtime& runtime) {
    if (!runtime.max_node_num || !runtime.node_list || !runtime.id_list || !runtime.empty_list || !max_dis_d) fail("tree not initialized");
    node_count = runtime.max_node_num[0];
    if (node_count <= 0) fail("invalid tree node count");
    std::vector<TN> nodes(static_cast<std::size_t>(node_count));
    std::vector<int> empty(static_cast<std::size_t>(node_count));
    std::vector<float> maxd(static_cast<std::size_t>(node_count));
    C2_CUDA(cudaMemcpy(nodes.data(), runtime.node_list, nodes.size() * sizeof(TN), cudaMemcpyDeviceToHost));
    C2_CUDA(cudaMemcpy(empty.data(), runtime.empty_list, empty.size() * sizeof(int), cudaMemcpyDeviceToHost));
    C2_CUDA(cudaMemcpy(maxd.data(), max_dis_d, maxd.size() * sizeof(float), cudaMemcpyDeviceToHost));
    // Tree construction pads only leaves. Hash the complete padded id_list rather
    // than the first base_n entries so any static-query mutation is detected.
    int leaf_count = 0;
    for (int nid = 1; nid < node_count; ++nid) {
      if (empty[nid] == 0 && nodes[nid].is_leaf == 1) ++leaf_count;
    }
    if (leaf_count <= 0) fail("frozen snapshot found no leaves");
    const std::size_t id_capacity = static_cast<std::size_t>(kBaseN) +
        static_cast<std::size_t>(leaf_count) * static_cast<std::size_t>(LEAF_PAD_SLOTS);
    std::vector<int> ids(id_capacity);
    C2_CUDA(cudaMemcpy(ids.data(), runtime.id_list, ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::uint64_t h = fnv1a(&node_count, sizeof(node_count));
    h = fnv1a(nodes.data(), nodes.size() * sizeof(TN), h);
    h = fnv1a(empty.data(), empty.size() * sizeof(int), h);
    h = fnv1a(maxd.data(), maxd.size() * sizeof(float), h);
    h = fnv1a(ids.data(), ids.size() * sizeof(int), h);
    hash = h;
  }
  void assert_unchanged(const Runtime& runtime) const {
    FrozenTreeSnapshot other; other.capture(runtime);
    if (other.node_count != node_count || other.hash != hash) fail("frozen static tree mutated during C2 query");
  }
};

struct Timing {
  double wall_ms = 0.0;
  double gpu_ms = 0.0;
};

Timing elapsed_cuda_event(cudaEvent_t start, cudaEvent_t stop, std::chrono::steady_clock::time_point wall_start) {
  C2_CUDA(cudaEventRecord(stop));
  C2_CUDA(cudaEventSynchronize(stop));
  float gpu = 0.0F;
  C2_CUDA(cudaEventElapsedTime(&gpu, start, stop));
  Timing out;
  out.gpu_ms = gpu;
  out.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
  return out;
}

struct BuildTiming {
  Timing base_h2d;
  Timing tree_build;
  double input_load_wall_ms = 0.0;
  double snapshot_wall_ms = 0.0;
};

Runtime build_tree(const Fvecs& base, const MetricPlan& metric, BuildTiming* timing) {
  Runtime runtime;
  C2_CUDA(cudaMallocManaged(&runtime.data_info, 3 * sizeof(int)));
  runtime.data_info[0] = kDimension;
  runtime.data_info[1] = kBaseN;
  runtime.data_info[2] = 2;
  C2_CUDA(cudaMalloc(&runtime.data_d, base.values.size() * sizeof(float)));
  cudaEvent_t begin = nullptr, end = nullptr;
  C2_CUDA(cudaEventCreate(&begin)); C2_CUDA(cudaEventCreate(&end));
  auto h2d_start = std::chrono::steady_clock::now();
  C2_CUDA(cudaEventRecord(begin));
  C2_CUDA(cudaMemcpyAsync(runtime.data_d, base.values.data(), base.values.size() * sizeof(float), cudaMemcpyHostToDevice));
  timing->base_h2d = elapsed_cuda_event(begin, end, h2d_start);
  C2_CUDA(cudaEventDestroy(begin)); C2_CUDA(cudaEventDestroy(end)); begin = end = nullptr;
  DIS_CODE = 100;
  INFI_DIS = metric.infi_dis;
  C2_CUDA(cudaEventCreate(&begin)); C2_CUDA(cudaEventCreate(&end));
  auto tree_start = std::chrono::steady_clock::now();
  C2_CUDA(cudaEventRecord(begin));
  indexConstru(runtime.data_d, nullptr, nullptr, runtime.data_info, runtime.id_list, runtime.node_list,
               runtime.max_node_num, runtime.tree_height, runtime.empty_list);
  C2_CUDA(cudaDeviceSynchronize());
  C2_CUDA(cudaGetLastError());
  timing->tree_build = elapsed_cuda_event(begin, end, tree_start);
  C2_CUDA(cudaEventDestroy(begin)); C2_CUDA(cudaEventDestroy(end));
  // The archived constructor uses these global temporary pointers internally.
  // They have already been freed in this code path; null them to avoid a false
  // double-free in isolated cleanup.  Tree outputs remain owned by Runtime.
  dis_list = nullptr; split_list = nullptr; split_num = nullptr; pid_list = nullptr;
  if (!runtime.id_list || !runtime.node_list || !runtime.max_node_num || !runtime.empty_list || runtime.tree_height <= 0) {
    fail("GTS tree build did not produce a valid runtime");
  }
  return runtime;
}

bool close_distance(float observed, float reference) {
  const float ulp = std::fabs(std::nextafterf(reference, std::numeric_limits<float>::infinity()) - reference);
  const float tolerance = std::max(kDistanceAbsFloor, static_cast<float>(kDistanceMaxUlps) * ulp);
  return std::fabs(observed - reference) <= tolerance;
}

// This host-only evaluator is deliberately scalar/ordered.  `volatile` prevents
// reassociation or contraction of the accumulation, so the CPU-only admission
// verifier can reproduce the exact source-level fp32 recurrence.  It is outside
// the timed GPU API interval and does not alter traversal, Eq. (1), or Eq. (2).
float raw_l2(const float* a, const float* b) {
  volatile float sum = 0.0F;
  for (int d = 0; d < kDimension; ++d) {
    const float delta = a[d] - b[d];
    const float term = delta * delta;
    sum = sum + term;
  }
  return std::sqrt(static_cast<float>(sum));
}

// Safe-C2 computes a mathematical L2 certificate on host and widens it
// outward before copying it to the traversal arrays. The fixed envelope is
// intentionally conservative for 128-term fp32 GPU accumulation; it makes the
// stored min no larger and stored max no smaller than the recomputed cover.
constexpr double kIntervalRelEnvelope = 1.0e-4;
constexpr double kIntervalAbsEnvelope = 1.0e-4;

struct IntervalCertificate {
  int node_count = 0;
  int active_nonroot_nodes = 0;
  int leaf_nodes = 0;
  long long membership_observations = 0;
  int original_min_above_actual = 0;
  int original_max_below_actual = 0;
  int min_outward_repaired = 0;
  int max_outward_repaired = 0;
  std::uint64_t before_interval_hash = 0;
  std::uint64_t after_interval_hash = 0;
  bool post_reverify_passed = false;
  double wall_ms = 0.0;
  double max_exact_l2 = 0.0;
};

double raw_l2_double(const float* a, const float* b) {
  long double sum = 0.0L;
  for (int d = 0; d < kDimension; ++d) {
    const long double delta = static_cast<long double>(a[d]) - static_cast<long double>(b[d]);
    sum += delta * delta;
  }
  return static_cast<double>(std::sqrt(sum));
}

float outward_lower(double exact) {
  if (!std::isfinite(exact) || exact < 0.0) fail("non-finite/negative exact interval minimum");
  const double lowered = std::max(0.0, exact * (1.0 - kIntervalRelEnvelope) - kIntervalAbsEnvelope);
  if (lowered == 0.0) return 0.0F;
  const float rounded = static_cast<float>(lowered);
  const float result = std::nextafterf(rounded, -std::numeric_limits<float>::infinity());
  if (!std::isfinite(result) || static_cast<double>(result) > exact) fail("lower outward rounding was not safe");
  return result;
}

float outward_upper(double exact) {
  if (!std::isfinite(exact) || exact < 0.0) fail("non-finite/negative exact interval maximum");
  const double raised = exact * (1.0 + kIntervalRelEnvelope) + kIntervalAbsEnvelope;
  const float rounded = static_cast<float>(raised);
  const float result = std::nextafterf(rounded, std::numeric_limits<float>::infinity());
  if (!std::isfinite(result) || static_cast<double>(result) < exact) fail("upper outward rounding was not safe");
  return result;
}

std::uint64_t active_interval_hash(const std::vector<TN>& nodes, const std::vector<int>& empty,
                                   const std::vector<float>& max_values) {
  if (nodes.size() != empty.size() || nodes.size() != max_values.size()) fail("interval hash array size mismatch");
  const std::size_t count = nodes.size();
  std::uint64_t h = fnv1a(&count, sizeof(count));
  for (std::size_t nid = 1; nid < nodes.size(); ++nid) {
    if (empty[nid] != 0) continue;
    h = fnv1a(&nid, sizeof(nid), h);
    h = fnv1a(&nodes[nid], sizeof(TN), h);
    h = fnv1a(&max_values[nid], sizeof(float), h);
  }
  return h;
}

// The GTS padding pass relocates leaf lists, so internal node lids are no longer
// contiguous membership ranges. Reconstruct each internal cover by propagating
// every leaf member to all heap ancestors. This is deliberately a pre-query
// certification step, not query time and not a submitted-paper timing claim.
IntervalCertificate certify_and_outward_repair_intervals(const Runtime& runtime, const Fvecs& base) {
  const auto begin = std::chrono::steady_clock::now();
  if (!runtime.max_node_num || !runtime.node_list || !runtime.id_list || !runtime.empty_list || !max_dis_d) {
    fail("cannot certify missing tree/max interval arrays");
  }
  if (TREE_ORDER != 10) fail("Safe-C2 certificate requires the frozen tree order 10");
  const int node_count = runtime.max_node_num[0];
  if (node_count <= 1) fail("invalid node count for interval certificate");
  std::vector<TN> nodes(static_cast<std::size_t>(node_count));
  std::vector<int> empty(static_cast<std::size_t>(node_count));
  std::vector<float> max_values(static_cast<std::size_t>(node_count));
  IntervalCertificate certificate;
  certificate.node_count = node_count;
  C2_CUDA(cudaMemcpy(nodes.data(), runtime.node_list, nodes.size() * sizeof(TN), cudaMemcpyDeviceToHost));
  C2_CUDA(cudaMemcpy(empty.data(), runtime.empty_list, empty.size() * sizeof(int), cudaMemcpyDeviceToHost));
  C2_CUDA(cudaMemcpy(max_values.data(), max_dis_d, max_values.size() * sizeof(float), cudaMemcpyDeviceToHost));
  certificate.before_interval_hash = active_interval_hash(nodes, empty, max_values);

  int leaf_nodes = 0;
  for (int nid = 1; nid < node_count; ++nid) {
    if (empty[nid] == 0 && nodes[nid].is_leaf == 1) ++leaf_nodes;
  }
  if (leaf_nodes <= 0) fail("no non-empty leaves for interval certificate");
  const std::size_t id_capacity = static_cast<std::size_t>(kBaseN) +
      static_cast<std::size_t>(leaf_nodes) * static_cast<std::size_t>(LEAF_PAD_SLOTS);
  std::vector<int> ids(id_capacity);
  C2_CUDA(cudaMemcpy(ids.data(), runtime.id_list, ids.size() * sizeof(int), cudaMemcpyDeviceToHost));

  std::vector<double> exact_min(static_cast<std::size_t>(node_count), std::numeric_limits<double>::infinity());
  std::vector<double> exact_max(static_cast<std::size_t>(node_count), 0.0);
  std::vector<int> observed_members(static_cast<std::size_t>(node_count), 0);
  certificate.leaf_nodes = leaf_nodes;

  for (int leaf = 1; leaf < node_count; ++leaf) {
    if (empty[leaf] != 0 || nodes[leaf].is_leaf != 1) continue;
    const TN leaf_node = nodes[leaf];
    if (leaf_node.size <= 0 || leaf_node.lid < 0 ||
        static_cast<std::size_t>(leaf_node.lid) + static_cast<std::size_t>(leaf_node.size) > ids.size()) {
      fail("invalid padded leaf membership range in interval certificate");
    }
    for (int slot = 0; slot < leaf_node.size; ++slot) {
      const int data_id = ids[static_cast<std::size_t>(leaf_node.lid) + static_cast<std::size_t>(slot)];
      if (data_id < 0 || data_id >= kBaseN) fail("invalid leaf payload id in interval certificate");
      int nid = leaf;
      while (nid != 0) {
        if (empty[nid] != 0 || nodes[nid].pid < 0 || nodes[nid].pid >= kBaseN) {
          fail("invalid ancestor/pivot while reconstructing Safe-C2 interval cover");
        }
        const double distance = raw_l2_double(
            base.values.data() + static_cast<std::size_t>(data_id) * kDimension,
            base.values.data() + static_cast<std::size_t>(nodes[nid].pid) * kDimension);
        if (!std::isfinite(distance)) fail("non-finite reconstructed L2 distance");
        exact_min[nid] = std::min(exact_min[nid], distance);
        exact_max[nid] = std::max(exact_max[nid], distance);
        ++observed_members[nid];
        ++certificate.membership_observations;
        nid = (nid - 1) / TREE_ORDER;
      }
    }
  }

  for (int nid = 1; nid < node_count; ++nid) {
    if (empty[nid] != 0) continue;
    ++certificate.active_nonroot_nodes;
    if (nodes[nid].size <= 0 || observed_members[nid] != nodes[nid].size ||
        !std::isfinite(exact_min[nid]) || !std::isfinite(exact_max[nid])) {
      fail("tree membership does not exactly reconstruct node interval cover");
    }
    const double exact_lo = exact_min[nid];
    const double exact_hi = exact_max[nid];
    if (nodes[nid].min_dis > exact_lo) ++certificate.original_min_above_actual;
    if (static_cast<double>(max_values[nid]) < exact_hi) ++certificate.original_max_below_actual;

    const float safe_lo = outward_lower(exact_lo);
    const float safe_hi = outward_upper(exact_hi);
    // Fail closed: after explicit outward repair no two-sided tolerance can
    // silently admit an invalid cover.
    if (static_cast<double>(safe_lo) > exact_lo || static_cast<double>(safe_hi) < exact_hi || safe_lo > safe_hi) {
      fail("outward-safe interval certificate invariant failed");
    }
    if (nodes[nid].min_dis != safe_lo) ++certificate.min_outward_repaired;
    if (max_values[nid] != safe_hi) ++certificate.max_outward_repaired;
    nodes[nid].min_dis = safe_lo;
    max_values[nid] = safe_hi;
    certificate.max_exact_l2 = std::max(certificate.max_exact_l2, exact_hi);
  }

  certificate.after_interval_hash = active_interval_hash(nodes, empty, max_values);
  C2_CUDA(cudaMemcpy(runtime.node_list, nodes.data(), nodes.size() * sizeof(TN), cudaMemcpyHostToDevice));
  C2_CUDA(cudaMemcpy(max_dis_d, max_values.data(), max_values.size() * sizeof(float), cudaMemcpyHostToDevice));
  C2_CUDA(cudaDeviceSynchronize());
  // Mandatory D2H re-verification after the repair: do not issue a query if a
  // device write was lost or if any stored directional cover is invalid.
  std::vector<TN> verified_nodes(static_cast<std::size_t>(node_count));
  std::vector<float> verified_max(static_cast<std::size_t>(node_count));
  C2_CUDA(cudaMemcpy(verified_nodes.data(), runtime.node_list, verified_nodes.size() * sizeof(TN), cudaMemcpyDeviceToHost));
  C2_CUDA(cudaMemcpy(verified_max.data(), max_dis_d, verified_max.size() * sizeof(float), cudaMemcpyDeviceToHost));
  if (active_interval_hash(verified_nodes, empty, verified_max) != certificate.after_interval_hash) {
    fail("post-repair D2H interval hash mismatch");
  }
  for (int nid = 1; nid < node_count; ++nid) {
    if (empty[nid] != 0) continue;
    if (static_cast<double>(verified_nodes[nid].min_dis) > exact_min[nid] ||
        static_cast<double>(verified_max[nid]) < exact_max[nid] ||
        verified_nodes[nid].min_dis > verified_max[nid]) {
      fail("post-repair directional interval cover verification failed");
    }
  }
  certificate.post_reverify_passed = true;
  certificate.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
  return certificate;
}

struct ReferenceCounters {
  long long correct = 0;
  long long total = 0;
  int invalid_output_ids = 0;
  int duplicate_output_ids = 0;
  int observed_distance_mismatches = 0;
  int reference_inconsistencies = 0;
  int boundary_ties = 0;
  std::vector<int> per_query_correct;
  std::string first_error;
  bool valid() const {
    return invalid_output_ids == 0 && duplicate_output_ids == 0 && observed_distance_mismatches == 0 &&
           reference_inconsistencies == 0 && boundary_ties == 0;
  }
};

struct PerQueryObservation {
  int global_qid = -1;
  int local_index = -1;
  int gt_overlap = 0;
  std::array<int, kK> gt_top10{};
  std::array<int, kK> returned_ids{};
  std::array<float, kK> returned_distances{};
};

void note_error(ReferenceCounters* c, const std::string& text) {
  if (c->first_error.empty()) c->first_error = text;
}

ReferenceCounters evaluate_results(const Fvecs& base, const Fvecs& all_queries, const std::vector<int>& gt,
                                   const std::vector<int>& selected_ids, const int* observed_ids,
                                   const float* observed_distances,
                                   std::vector<PerQueryObservation>* observations = nullptr) {
  ReferenceCounters result;
  result.total = static_cast<long long>(selected_ids.size()) * kK;
  result.per_query_correct.assign(selected_ids.size(), 0);
  if (observations != nullptr) {
    observations->clear();
    observations->reserve(selected_ids.size());
  }
  for (std::size_t local = 0; local < selected_ids.size(); ++local) {
    const int qid = selected_ids[local];
    const float* query = all_queries.values.data() + static_cast<std::size_t>(qid) * kDimension;
    const int* gt_row = gt.data() + static_cast<std::size_t>(qid) * kGtWidth;
    PerQueryObservation observation;
    observation.global_qid = qid;
    observation.local_index = static_cast<int>(local);
    std::unordered_set<int> gt_topk;
    std::vector<std::pair<float, int>> top100;
    top100.reserve(kGtWidth);
    for (int j = 0; j < kGtWidth; ++j) {
      const int id = gt_row[j];
      const float d = raw_l2(base.values.data() + static_cast<std::size_t>(id) * kDimension, query);
      top100.emplace_back(d, id);
      if (j < kK) {
        gt_topk.insert(id);
        observation.gt_top10[static_cast<std::size_t>(j)] = id;
      }
    }
    std::sort(top100.begin(), top100.end(), [](const auto& left, const auto& right) {
      return left.first != right.first ? left.first < right.first : left.second < right.second;
    });
    std::unordered_set<int> recomputed_topk;
    for (int j = 0; j < kK; ++j) recomputed_topk.insert(top100[j].second);
    if (recomputed_topk != gt_topk) {
      ++result.reference_inconsistencies;
      note_error(&result, "GT top-10 set disagrees with raw-distance sort within stored top-100 at q=" + std::to_string(qid));
    }
    if (top100[kK - 1].first == top100[kK].first) {
      ++result.boundary_ties;
      note_error(&result, "public GT has a top-k boundary tie at q=" + std::to_string(qid));
    }
    std::unordered_set<int> seen;
    int query_correct = 0;
    for (int rank = 0; rank < kK; ++rank) {
      const std::size_t offset = local * kK + rank;
      const int id = observed_ids[offset];
      const float reported = observed_distances[offset];
      observation.returned_ids[static_cast<std::size_t>(rank)] = id;
      observation.returned_distances[static_cast<std::size_t>(rank)] = reported;
      if (id < 0 || id >= kBaseN) {
        ++result.invalid_output_ids;
        note_error(&result, "GTS emitted invalid ID at q=" + std::to_string(qid));
        continue;
      }
      const bool first_occurrence = seen.insert(id).second;
      if (!first_occurrence) {
        ++result.duplicate_output_ids;
        note_error(&result, "GTS emitted duplicate top-k ID at q=" + std::to_string(qid));
      }
      const float exact_for_observed = raw_l2(base.values.data() + static_cast<std::size_t>(id) * kDimension, query);
      if (!std::isfinite(reported) || !close_distance(reported, exact_for_observed)) {
        ++result.observed_distance_mismatches;
        note_error(&result, "reported distance does not match raw L2 for emitted ID at q=" + std::to_string(qid));
      }
      if (first_occurrence && gt_topk.count(id) != 0) {
        ++result.correct;
        ++query_correct;
      }
    }
    result.per_query_correct[local] = query_correct;
    observation.gt_overlap = query_correct;
    if (observations != nullptr) observations->push_back(observation);
  }
  return result;
}

enum class QueryMode { kBaselineGammaOne, kSpeculativeGammaWithFallback };

struct FallbackStats {
  int flagged_queries = 0;
  std::uint64_t flagged_global_qids_fnv1a64 = 1469598103934665603ULL;
  double flag_d2h_wall_ms = 0.0;
  double fallback_query_h2d_wall_ms = 0.0;
  Timing fallback_static_api;
  double fallback_d2h_ids_wall_ms = 0.0;
  double fallback_d2h_distances_wall_ms = 0.0;
  double fallback_result_splice_d2d_ids_wall_ms = 0.0;
  std::vector<int> flagged_local_indices;
  std::vector<int> flagged_global_qids;
  std::vector<GammaOnlyPruneTrace> first_gamma_only_trace;
};

struct QueryRun {
  // The speculative traversal is diagnostic only.  The guarded pipeline includes
  // gamma-only flag transfer, gamma=1 fallback, and host-final-pair repair; it
  // is the only candidate latency figure eligible for a later paired comparison.
  // Contract: final_guarded_per_query is the complete IDs+distances result.
  // Legacy global res_dis is scratch, released after a subset fallback, and is
  // never claimed as an externally valid full-query distance output.
  Timing speculative_static_api;
  double speculative_d2h_ids_wall_ms = 0.0;
  double speculative_d2h_distances_wall_ms = 0.0;
  double raw_reference_evaluation_wall_ms = 0.0;
  double final_reference_evaluation_wall_ms = 0.0;
  double guarded_query_pipeline_wall_ms = 0.0;
  double guarded_total_runner_wall_ms = 0.0;
  double frozen_tree_snapshot_wall_ms = 0.0;
  int frozen_tree_snapshot_checks = 0;
  ReferenceCounters raw_speculative_reference;
  ReferenceCounters final_guarded_reference;
  std::vector<PerQueryObservation> raw_speculative_per_query;
  std::vector<PerQueryObservation> final_guarded_per_query;
  FallbackStats fallback;
};

void upload_gamma(const std::array<float, RP_MAX_LEVELS>& gamma) {
  float alpha[RP_MAX_LEVELS]{};
  float beta[RP_MAX_LEVELS]{};
  bool active = false;
  for (int level = 0; level < RP_MAX_LEVELS; ++level) {
    if (!std::isfinite(gamma[level]) || gamma[level] < 1.0F) fail("invalid gamma value");
    if (gamma[level] > 1.0F) active = true;
  }
  float gamma_copy[RP_MAX_LEVELS];
  for (int level = 0; level < RP_MAX_LEVELS; ++level) gamma_copy[level] = gamma[level];
  upload_rp_constants(alpha, beta, gamma_copy, RP_MAX_LEVELS, nullptr, nullptr, nullptr, 0, active ? 1 : 0);
}

std::array<float, RP_MAX_LEVELS> all_ones_gamma() {
  std::array<float, RP_MAX_LEVELS> out{};
  out.fill(1.0F);
  return out;
}

void assert_frozen_timed(const FrozenTreeSnapshot& frozen, const Runtime& runtime, QueryRun* run) {
  const auto begin = std::chrono::steady_clock::now();
  frozen.assert_unchanged(runtime);
  run->frozen_tree_snapshot_wall_ms +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
  ++run->frozen_tree_snapshot_checks;
}

Timing timed_baseline_search(const Runtime& runtime, float* queries_d, int* result_ids_d, int query_count) {
  update_disk = false;
  cudaEvent_t begin = nullptr, end = nullptr;
  C2_CUDA(cudaEventCreate(&begin)); C2_CUDA(cudaEventCreate(&end));
  const auto wall_start = std::chrono::steady_clock::now();
  C2_CUDA(cudaEventRecord(begin));
  searchIndexKnnV2(runtime.data_d, runtime.node_list, runtime.id_list, runtime.max_node_num,
                    queries_d, result_ids_d, query_count, kK, runtime.tree_height,
                    runtime.data_info, runtime.empty_list, nullptr, nullptr);
  const Timing timing = elapsed_cuda_event(begin, end, wall_start);
  C2_CUDA(cudaEventDestroy(begin)); C2_CUDA(cudaEventDestroy(end));
  if (res_dis == nullptr) fail("baseline GTS query did not publish res_dis");
  return timing;
}

Timing timed_speculative_search(const Runtime& runtime, float* queries_d, int* result_ids_d, int query_count,
                                int* gamma_only_pruned_d, GammaOnlyPruneTrace* gamma_only_trace_d) {
  update_disk = false;
  cudaEvent_t begin = nullptr, end = nullptr;
  C2_CUDA(cudaEventCreate(&begin)); C2_CUDA(cudaEventCreate(&end));
  const auto wall_start = std::chrono::steady_clock::now();
  C2_CUDA(cudaEventRecord(begin));
  searchIndexKnnV3Speculative(runtime.data_d, runtime.node_list, runtime.id_list, runtime.max_node_num,
                               queries_d, result_ids_d, query_count, kK, runtime.tree_height,
                               runtime.data_info, runtime.empty_list, nullptr, nullptr,
                               gamma_only_pruned_d, gamma_only_trace_d);
  const Timing timing = elapsed_cuda_event(begin, end, wall_start);
  C2_CUDA(cudaEventDestroy(begin)); C2_CUDA(cudaEventDestroy(end));
  if (res_dis == nullptr) fail("speculative GTS query did not publish res_dis");
  return timing;
}

void release_res_dis() {
  if (res_dis != nullptr) {
    C2_CUDA(cudaFree(res_dis));
    res_dis = nullptr;
  }
}

std::uint64_t fnv1a_int_vector(const std::vector<int>& values) {
  if (values.empty()) return fnv1a(nullptr, 0);
  return fnv1a(values.data(), values.size() * sizeof(int));
}

void write_int_json_array(std::ostream& out, const std::array<int, kK>& values) {
  out << '[';
  for (int i = 0; i < kK; ++i) { if (i) out << ','; out << values[static_cast<std::size_t>(i)]; }
  out << ']';
}

void write_json_float(std::ostream& out, float value) {
  // Reference validation marks non-finite distances invalid; serialize them as
  // JSON null rather than emitting non-standard NaN/Inf tokens in an audit file.
  if (std::isfinite(value)) out << std::setprecision(10) << value;
  else out << "null";
}

void write_float_json_array(std::ostream& out, const std::array<float, kK>& values) {
  out << '[';
  for (int i = 0; i < kK; ++i) {
    if (i) out << ',';
    write_json_float(out, values[static_cast<std::size_t>(i)]);
  }
  out << ']';
}

std::uint64_t write_per_query_jsonl(const std::string& path, const std::string& role, int timed_rep,
                                     const QueryRun& run) {
  if (run.raw_speculative_per_query.size() != run.final_guarded_per_query.size()) {
    fail("per-query raw/final record count mismatch");
  }
  std::ofstream out(path);
  if (!out) fail("cannot write v3 per-query JSONL");
  for (std::size_t i = 0; i < run.final_guarded_per_query.size(); ++i) {
    const auto& raw = run.raw_speculative_per_query[i];
    const auto& final = run.final_guarded_per_query[i];
    const bool fallback = std::binary_search(run.fallback.flagged_local_indices.begin(),
                                             run.fallback.flagged_local_indices.end(), final.local_index);
    out << std::setprecision(10)
        << "{\"schema\":\"safe-c2-v3-per-query-v1\",\"role\":\"" << role
        << "\",\"timed_rep\":" << timed_rep
        << ",\"local_index\":" << final.local_index
        << ",\"global_qid\":" << final.global_qid
        << ",\"gamma_only_fallback_applied\":" << (fallback ? "true" : "false")
        << ",\"gt_top10\":";
    write_int_json_array(out, final.gt_top10);
    out << ",\"raw_speculative\":{\"overlap\":" << raw.gt_overlap << ",\"ids\":";
    write_int_json_array(out, raw.returned_ids);
    out << ",\"distances\":";
    write_float_json_array(out, raw.returned_distances);
    out << "},\"final_guarded\":{\"overlap\":" << final.gt_overlap << ",\"ids\":";
    write_int_json_array(out, final.returned_ids);
    out << ",\"distances\":";
    write_float_json_array(out, final.returned_distances);
    out << '}';
    if (fallback) {
      const auto it = std::lower_bound(run.fallback.flagged_local_indices.begin(),
                                       run.fallback.flagged_local_indices.end(), final.local_index);
      const std::size_t offset = static_cast<std::size_t>(it - run.fallback.flagged_local_indices.begin());
      const auto& trace = run.fallback.first_gamma_only_trace[offset];
      out << ",\"first_gamma_only_prune\":{\"nid\":" << trace.nid
          << ",\"level\":" << trace.level
          << ",\"lb_tri\":";
      write_json_float(out, trace.lb_tri);
      out << ",\"lb_eff\":";
      write_json_float(out, trace.lb_eff);
      out << ",\"disk_at_decision\":";
      write_json_float(out, trace.disk_at_decision);
      out << '}';
    }
    out << "}\n";
  }
  if (!out) fail("failed writing v3 per-query JSONL");
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot reopen v3 per-query JSONL");
  std::uint64_t hash = 1469598103934665603ULL;
  char c = 0;
  while (input.get(c)) hash = fnv1a(&c, 1, hash);
  return hash;
}

QueryRun execute_once(const Runtime& runtime, const FrozenTreeSnapshot& frozen,
                      float* query_d, int query_count, const Fvecs& base,
                      const Fvecs& all_queries, const std::vector<int>& gt,
                      const std::vector<int>& selected_ids, int* result_ids_d,
                      int* result_ids_h, float* result_distances_h,
                      QueryMode mode, const std::array<float, RP_MAX_LEVELS>& active_gamma,
                      bool collect_results) {
  QueryRun run;
  const auto total_start = std::chrono::steady_clock::now();
  const auto pipeline_start = total_start;
  int* flags_d = nullptr;
  GammaOnlyPruneTrace* traces_d = nullptr;
  int* fallback_ids_d = nullptr;
  int* fallback_ids_h = nullptr;
  float* fallback_distances_h = nullptr;
  float* fallback_queries_d = nullptr;
  float* fallback_queries_h = nullptr;
  try {
    if (mode == QueryMode::kSpeculativeGammaWithFallback) {
      C2_CUDA(cudaMalloc(&flags_d, static_cast<std::size_t>(query_count) * sizeof(int)));
      C2_CUDA(cudaMalloc(&traces_d, static_cast<std::size_t>(query_count) * sizeof(GammaOnlyPruneTrace)));
      run.speculative_static_api = timed_speculative_search(runtime, query_d, result_ids_d, query_count, flags_d, traces_d);
    } else {
      run.speculative_static_api = timed_baseline_search(runtime, query_d, result_ids_d, query_count);
    }
    // The fallback must see the exact same static tree as the speculative pass.
    assert_frozen_timed(frozen, runtime, &run);

    if (collect_results) {
      const auto begin = std::chrono::steady_clock::now();
      C2_CUDA(cudaMemcpy(result_ids_h, result_ids_d, static_cast<std::size_t>(query_count) * kK * sizeof(int), cudaMemcpyDeviceToHost));
      run.speculative_d2h_ids_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
      const auto dist_begin = std::chrono::steady_clock::now();
      C2_CUDA(cudaMemcpy(result_distances_h, res_dis, static_cast<std::size_t>(query_count) * kK * sizeof(float), cudaMemcpyDefault));
      run.speculative_d2h_distances_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dist_begin).count();
      const auto raw_eval_begin = std::chrono::steady_clock::now();
      run.raw_speculative_reference = evaluate_results(base, all_queries, gt, selected_ids, result_ids_h, result_distances_h,
                                                       &run.raw_speculative_per_query);
      run.raw_reference_evaluation_wall_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - raw_eval_begin).count();
    }

    if (mode == QueryMode::kSpeculativeGammaWithFallback) {
      std::vector<int> flags(static_cast<std::size_t>(query_count));
      std::vector<GammaOnlyPruneTrace> traces(static_cast<std::size_t>(query_count));
      const auto flag_begin = std::chrono::steady_clock::now();
      C2_CUDA(cudaMemcpy(flags.data(), flags_d, flags.size() * sizeof(int), cudaMemcpyDeviceToHost));
      C2_CUDA(cudaMemcpy(traces.data(), traces_d, traces.size() * sizeof(GammaOnlyPruneTrace), cudaMemcpyDeviceToHost));
      run.fallback.flag_d2h_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - flag_begin).count();
      for (int local = 0; local < query_count; ++local) {
        if (flags[static_cast<std::size_t>(local)] != 0) {
          if (traces[static_cast<std::size_t>(local)].nid < 0) fail("gamma-only flag lacks trace");
          run.fallback.flagged_local_indices.push_back(local);
          run.fallback.flagged_global_qids.push_back(selected_ids[static_cast<std::size_t>(local)]);
          run.fallback.first_gamma_only_trace.push_back(traces[static_cast<std::size_t>(local)]);
        }
      }
      run.fallback.flagged_queries = static_cast<int>(run.fallback.flagged_local_indices.size());
      run.fallback.flagged_global_qids_fnv1a64 = fnv1a_int_vector(run.fallback.flagged_global_qids);
      release_res_dis();

      if (run.fallback.flagged_queries > 0) {
        const int fallback_count = run.fallback.flagged_queries;
        C2_CUDA(cudaMallocHost(&fallback_queries_h, static_cast<std::size_t>(fallback_count) * kDimension * sizeof(float)));
        for (int j = 0; j < fallback_count; ++j) {
          const int global_qid = run.fallback.flagged_global_qids[static_cast<std::size_t>(j)];
          std::memcpy(fallback_queries_h + static_cast<std::size_t>(j) * kDimension,
                      all_queries.values.data() + static_cast<std::size_t>(global_qid) * kDimension,
                      kDimension * sizeof(float));
        }
        C2_CUDA(cudaMalloc(&fallback_queries_d, static_cast<std::size_t>(fallback_count) * kDimension * sizeof(float)));
        const auto h2d_begin = std::chrono::steady_clock::now();
        C2_CUDA(cudaMemcpy(fallback_queries_d, fallback_queries_h, static_cast<std::size_t>(fallback_count) * kDimension * sizeof(float), cudaMemcpyHostToDevice));
        run.fallback.fallback_query_h2d_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h2d_begin).count();
        C2_CUDA(cudaMalloc(&fallback_ids_d, static_cast<std::size_t>(fallback_count) * kK * sizeof(int)));
        if (collect_results) {
          C2_CUDA(cudaMallocHost(&fallback_ids_h, static_cast<std::size_t>(fallback_count) * kK * sizeof(int)));
          C2_CUDA(cudaMallocHost(&fallback_distances_h, static_cast<std::size_t>(fallback_count) * kK * sizeof(float)));
        }
        // Gamma is never used as a safety certificate: every flagged query is
        // recomputed with the original unscaled traversal before final output.
        upload_gamma(all_ones_gamma());
        run.fallback.fallback_static_api = timed_baseline_search(runtime, fallback_queries_d, fallback_ids_d, fallback_count);
        if (collect_results) {
          const auto ids_begin = std::chrono::steady_clock::now();
          C2_CUDA(cudaMemcpy(fallback_ids_h, fallback_ids_d, static_cast<std::size_t>(fallback_count) * kK * sizeof(int), cudaMemcpyDeviceToHost));
          run.fallback.fallback_d2h_ids_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ids_begin).count();
          const auto distances_begin = std::chrono::steady_clock::now();
          C2_CUDA(cudaMemcpy(fallback_distances_h, res_dis, static_cast<std::size_t>(fallback_count) * kK * sizeof(float), cudaMemcpyDefault));
          run.fallback.fallback_d2h_distances_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - distances_begin).count();
          for (int j = 0; j < fallback_count; ++j) {
            const int local = run.fallback.flagged_local_indices[static_cast<std::size_t>(j)];
            const std::size_t target = static_cast<std::size_t>(local) * kK;
            const std::size_t source = static_cast<std::size_t>(j) * kK;
            std::copy_n(fallback_ids_h + source, kK, result_ids_h + target);
            std::copy_n(fallback_distances_h + source, kK, result_distances_h + target);
          }
        }
        // Preserve the runner's full-query device ID buffer for callers that use
        // IDs.  The legacy global res_dis is a scratch allocation with only the
        // fallback subset after this point; it is deliberately not exported as a
        // full-query distance output.  The complete final IDs+distances contract
        // is the host pair below, checked by evaluate_results and persisted JSONL.
        const auto splice_begin = std::chrono::steady_clock::now();
        for (int j = 0; j < fallback_count; ++j) {
          const int local = run.fallback.flagged_local_indices[static_cast<std::size_t>(j)];
          const std::size_t target = static_cast<std::size_t>(local) * kK;
          const std::size_t source = static_cast<std::size_t>(j) * kK;
          C2_CUDA(cudaMemcpy(result_ids_d + target, fallback_ids_d + source,
                             kK * sizeof(int), cudaMemcpyDeviceToDevice));
        }
        run.fallback.fallback_result_splice_d2d_ids_wall_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - splice_begin).count();
        release_res_dis();
        // Restoring makes the next speculative call independent of fallback.
        upload_gamma(active_gamma);
        assert_frozen_timed(frozen, runtime, &run);
      }
    }

    // Exclude CPU oracle validation from query-pipeline latency, but retain a
    // full runner wall time separately.  Output copies and every fallback action
    // remain included in the guarded pipeline figure.
    const auto pipeline_end = std::chrono::steady_clock::now();
    run.guarded_query_pipeline_wall_ms = std::chrono::duration<double, std::milli>(
        pipeline_end - pipeline_start).count() - run.raw_reference_evaluation_wall_ms -
        run.frozen_tree_snapshot_wall_ms;
    if (run.guarded_query_pipeline_wall_ms < 0.0) fail("negative guarded pipeline timing");
    if (collect_results) {
      const auto final_eval_begin = std::chrono::steady_clock::now();
      run.final_guarded_reference = evaluate_results(base, all_queries, gt, selected_ids, result_ids_h, result_distances_h,
                                                     &run.final_guarded_per_query);
      run.final_reference_evaluation_wall_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - final_eval_begin).count();
      if (mode == QueryMode::kBaselineGammaOne) {
        run.raw_speculative_reference = run.final_guarded_reference;
        run.raw_speculative_per_query = run.final_guarded_per_query;
      }
    }
    release_res_dis();
    if (flags_d) { C2_CUDA(cudaFree(flags_d)); flags_d = nullptr; }
    if (traces_d) { C2_CUDA(cudaFree(traces_d)); traces_d = nullptr; }
    if (fallback_ids_d) { C2_CUDA(cudaFree(fallback_ids_d)); fallback_ids_d = nullptr; }
    if (fallback_ids_h) { C2_CUDA(cudaFreeHost(fallback_ids_h)); fallback_ids_h = nullptr; }
    if (fallback_distances_h) { C2_CUDA(cudaFreeHost(fallback_distances_h)); fallback_distances_h = nullptr; }
    if (fallback_queries_d) { C2_CUDA(cudaFree(fallback_queries_d)); fallback_queries_d = nullptr; }
    if (fallback_queries_h) { C2_CUDA(cudaFreeHost(fallback_queries_h)); fallback_queries_h = nullptr; }
    run.guarded_total_runner_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();
    return run;
  } catch (...) {
    if (res_dis) { cudaFree(res_dis); res_dis = nullptr; }
    if (flags_d) cudaFree(flags_d);
    if (traces_d) cudaFree(traces_d);
    if (fallback_ids_d) cudaFree(fallback_ids_d);
    if (fallback_ids_h) cudaFreeHost(fallback_ids_h);
    if (fallback_distances_h) cudaFreeHost(fallback_distances_h);
    if (fallback_queries_d) cudaFree(fallback_queries_d);
    if (fallback_queries_h) cudaFreeHost(fallback_queries_h);
    throw;
  }
}

struct PerQueryArtifact {
  std::string path;
  std::uint64_t fnv1a64 = 0;
  std::string role;
  int timed_rep = -1;
};

struct StageResult {
  int warmups = 0;
  int timed_reps = 0;
  double speculative_static_api_wall_ms_sum = 0.0;
  double speculative_static_api_gpu_ms_sum = 0.0;
  double speculative_d2h_ids_wall_ms_sum = 0.0;
  double speculative_d2h_distances_wall_ms_sum = 0.0;
  double flag_d2h_wall_ms_sum = 0.0;
  double fallback_query_h2d_wall_ms_sum = 0.0;
  double fallback_static_api_wall_ms_sum = 0.0;
  double fallback_static_api_gpu_ms_sum = 0.0;
  double fallback_d2h_ids_wall_ms_sum = 0.0;
  double fallback_d2h_distances_wall_ms_sum = 0.0;
  double fallback_result_splice_d2d_ids_wall_ms_sum = 0.0;
  double raw_reference_evaluation_wall_ms_sum = 0.0;
  double final_reference_evaluation_wall_ms_sum = 0.0;
  double guarded_query_pipeline_wall_ms_sum = 0.0;
  double guarded_total_runner_wall_ms_sum = 0.0;
  double frozen_tree_snapshot_wall_ms_sum = 0.0;
  int frozen_tree_snapshot_checks_sum = 0;
  long long gamma_only_flagged_queries_sum = 0;
  std::uint64_t gamma_only_flagged_global_qids_fnv1a64 = 1469598103934665603ULL;
  ReferenceCounters raw_speculative_reference_sum;
  ReferenceCounters final_guarded_reference_sum;
  std::vector<ReferenceCounters> raw_speculative_repetitions;
  std::vector<ReferenceCounters> final_guarded_repetitions;
  std::vector<PerQueryArtifact> per_query_artifacts;
};

void add_counters(ReferenceCounters* target, const ReferenceCounters& source) {
  target->correct += source.correct; target->total += source.total;
  target->invalid_output_ids += source.invalid_output_ids;
  target->duplicate_output_ids += source.duplicate_output_ids;
  target->observed_distance_mismatches += source.observed_distance_mismatches;
  target->reference_inconsistencies += source.reference_inconsistencies;
  target->boundary_ties += source.boundary_ties;
  if (target->first_error.empty()) target->first_error = source.first_error;
}

void append_timed_run(StageResult* target, const QueryRun& one) {
  target->speculative_static_api_wall_ms_sum += one.speculative_static_api.wall_ms;
  target->speculative_static_api_gpu_ms_sum += one.speculative_static_api.gpu_ms;
  target->speculative_d2h_ids_wall_ms_sum += one.speculative_d2h_ids_wall_ms;
  target->speculative_d2h_distances_wall_ms_sum += one.speculative_d2h_distances_wall_ms;
  target->flag_d2h_wall_ms_sum += one.fallback.flag_d2h_wall_ms;
  target->fallback_query_h2d_wall_ms_sum += one.fallback.fallback_query_h2d_wall_ms;
  target->fallback_static_api_wall_ms_sum += one.fallback.fallback_static_api.wall_ms;
  target->fallback_static_api_gpu_ms_sum += one.fallback.fallback_static_api.gpu_ms;
  target->fallback_d2h_ids_wall_ms_sum += one.fallback.fallback_d2h_ids_wall_ms;
  target->fallback_d2h_distances_wall_ms_sum += one.fallback.fallback_d2h_distances_wall_ms;
  target->fallback_result_splice_d2d_ids_wall_ms_sum += one.fallback.fallback_result_splice_d2d_ids_wall_ms;
  target->raw_reference_evaluation_wall_ms_sum += one.raw_reference_evaluation_wall_ms;
  target->final_reference_evaluation_wall_ms_sum += one.final_reference_evaluation_wall_ms;
  target->guarded_query_pipeline_wall_ms_sum += one.guarded_query_pipeline_wall_ms;
  target->guarded_total_runner_wall_ms_sum += one.guarded_total_runner_wall_ms;
  target->frozen_tree_snapshot_wall_ms_sum += one.frozen_tree_snapshot_wall_ms;
  target->frozen_tree_snapshot_checks_sum += one.frozen_tree_snapshot_checks;
  target->gamma_only_flagged_queries_sum += one.fallback.flagged_queries;
  target->gamma_only_flagged_global_qids_fnv1a64 = fnv1a(&one.fallback.flagged_global_qids_fnv1a64,
                                                          sizeof(one.fallback.flagged_global_qids_fnv1a64),
                                                          target->gamma_only_flagged_global_qids_fnv1a64);
  add_counters(&target->raw_speculative_reference_sum, one.raw_speculative_reference);
  add_counters(&target->final_guarded_reference_sum, one.final_guarded_reference);
  target->raw_speculative_repetitions.push_back(one.raw_speculative_reference);
  target->final_guarded_repetitions.push_back(one.final_guarded_reference);
}

StageResult run_stage(const Runtime& runtime, const FrozenTreeSnapshot& frozen, float* query_d, int query_count,
                      const Fvecs& base, const Fvecs& all_queries, const std::vector<int>& gt,
                      const std::vector<int>& selected_ids, int warmups, int timed_reps,
                      QueryMode mode, const std::array<float, RP_MAX_LEVELS>& gamma,
                      const std::string& output_dir, const std::string& role,
                      bool persist_per_query) {
  int* ids_d = nullptr; int* ids_h = nullptr; float* distances_h = nullptr;
  const std::size_t result_count = static_cast<std::size_t>(query_count) * kK;
  C2_CUDA(cudaMalloc(&ids_d, result_count * sizeof(int)));
  C2_CUDA(cudaMallocHost(&ids_h, result_count * sizeof(int)));
  C2_CUDA(cudaMallocHost(&distances_h, result_count * sizeof(float)));
  StageResult result; result.warmups = warmups; result.timed_reps = timed_reps;
  try {
    for (int i = 0; i < warmups; ++i) {
      upload_gamma(gamma);
      (void)execute_once(runtime, frozen, query_d, query_count, base, all_queries, gt, selected_ids, ids_d, ids_h, distances_h,
                         mode, gamma, false);
    }
    for (int i = 0; i < timed_reps; ++i) {
      upload_gamma(gamma);
      QueryRun one = execute_once(runtime, frozen, query_d, query_count, base, all_queries, gt, selected_ids, ids_d, ids_h, distances_h,
                                  mode, gamma, true);
      append_timed_run(&result, one);
      if (persist_per_query) {
        const std::string path = output_dir + "/per_query_" + role + "_rep_" + std::to_string(i) + ".jsonl";
        const std::uint64_t hash = write_per_query_jsonl(path, role, i, one);
        result.per_query_artifacts.push_back(PerQueryArtifact{path, hash, role, i});
      }
    }
  } catch (...) {
    if (res_dis) { cudaFree(res_dis); res_dis = nullptr; }
    cudaFree(ids_d); cudaFreeHost(ids_h); cudaFreeHost(distances_h); throw;
  }
  C2_CUDA(cudaFree(ids_d)); C2_CUDA(cudaFreeHost(ids_h)); C2_CUDA(cudaFreeHost(distances_h));
  return result;
}

struct PairedAbbaStageResult {
  StageResult baseline;
  StageResult speculative_guarded;
  int shared_warmup_pairs = 0;
  std::vector<std::string> timed_pair_order;
};

PairedAbbaStageResult run_paired_abba_stage(
    const Runtime& runtime, const FrozenTreeSnapshot& frozen, float* query_d, int query_count,
    const Fvecs& base, const Fvecs& all_queries, const std::vector<int>& gt,
    const std::vector<int>& selected_ids, const std::array<float, RP_MAX_LEVELS>& gamma,
    int warmups, int timed_reps, const std::string& output_dir) {
  int* ids_d = nullptr; int* ids_h = nullptr; float* distances_h = nullptr;
  const std::size_t result_count = static_cast<std::size_t>(query_count) * kK;
  C2_CUDA(cudaMalloc(&ids_d, result_count * sizeof(int)));
  C2_CUDA(cudaMallocHost(&ids_h, result_count * sizeof(int)));
  C2_CUDA(cudaMallocHost(&distances_h, result_count * sizeof(float)));
  PairedAbbaStageResult result;
  result.baseline.warmups = warmups; result.baseline.timed_reps = timed_reps;
  result.speculative_guarded.warmups = warmups; result.speculative_guarded.timed_reps = timed_reps;
  const auto all_ones = all_ones_gamma();
  auto execute = [&](const std::array<float, RP_MAX_LEVELS>& configuration, QueryMode mode, bool timed,
                     StageResult* target, const std::string& role, int rep) {
    upload_gamma(configuration);
    QueryRun one = execute_once(runtime, frozen, query_d, query_count, base, all_queries, gt, selected_ids,
                                ids_d, ids_h, distances_h, mode, configuration, timed);
    if (timed) {
      append_timed_run(target, one);
      const std::string path = output_dir + "/per_query_" + role + "_rep_" + std::to_string(rep) + ".jsonl";
      const std::uint64_t hash = write_per_query_jsonl(path, role, rep, one);
      target->per_query_artifacts.push_back(PerQueryArtifact{path, hash, role, rep});
    }
  };
  try {
    for (int warmup = 0; warmup < warmups; ++warmup) {
      execute(all_ones, QueryMode::kBaselineGammaOne, false, nullptr, "baseline_warmup", warmup);
      execute(gamma, QueryMode::kSpeculativeGammaWithFallback, false, nullptr, "guarded_warmup", warmup);
      ++result.shared_warmup_pairs;
    }
    for (int rep = 0; rep < timed_reps; ++rep) {
      if ((rep % 2) == 0) {
        result.timed_pair_order.push_back("baseline_then_speculative_guarded");
        execute(all_ones, QueryMode::kBaselineGammaOne, true, &result.baseline, "baseline", rep);
        execute(gamma, QueryMode::kSpeculativeGammaWithFallback, true, &result.speculative_guarded, "speculative_guarded", rep);
      } else {
        result.timed_pair_order.push_back("speculative_guarded_then_baseline");
        execute(gamma, QueryMode::kSpeculativeGammaWithFallback, true, &result.speculative_guarded, "speculative_guarded", rep);
        execute(all_ones, QueryMode::kBaselineGammaOne, true, &result.baseline, "baseline", rep);
      }
    }
  } catch (...) {
    if (res_dis) { cudaFree(res_dis); res_dis = nullptr; }
    cudaFree(ids_d); cudaFreeHost(ids_h); cudaFreeHost(distances_h); throw;
  }
  C2_CUDA(cudaFree(ids_d)); C2_CUDA(cudaFreeHost(ids_h)); C2_CUDA(cudaFreeHost(distances_h));
  return result;
}

struct PerQueryGateStats {
  bool passed = false;
  int query_count = 0;
  int violating_queries = 0;
  int min_per_query_delta = 0;
};

PerQueryGateStats per_query_no_regression(const ReferenceCounters& candidate,
                                          const ReferenceCounters& baseline) {
  PerQueryGateStats out;
  if (!candidate.valid() || !baseline.valid() ||
      candidate.per_query_correct.empty() ||
      candidate.per_query_correct.size() != baseline.per_query_correct.size()) {
    out.violating_queries = -1;  // malformed/invalid reference: fail closed.
    out.min_per_query_delta = -kK;
    return out;
  }
  out.query_count = static_cast<int>(candidate.per_query_correct.size());
  out.min_per_query_delta = kK;
  for (std::size_t i = 0; i < candidate.per_query_correct.size(); ++i) {
    const int delta = candidate.per_query_correct[i] - baseline.per_query_correct[i];
    out.min_per_query_delta = std::min(out.min_per_query_delta, delta);
    if (delta < 0) ++out.violating_queries;
  }
  out.passed = out.violating_queries == 0;
  return out;
}

bool same_per_query_overlap(const ReferenceCounters& left, const ReferenceCounters& right) {
  return left.valid() && right.valid() && left.per_query_correct == right.per_query_correct;
}

struct StageGateStats {
  bool passed = false;
  int repetitions_checked = 0;
  int query_count = 0;
  int violating_queries_sum = 0;
  int max_violating_queries = 0;
  int min_per_query_delta = 0;
};

// Gate the *final guarded* results only.  Raw speculative output is retained as
// a diagnostic/selection trace and cannot be advertised as the final answer.
StageGateStats stage_final_guarded_gate(const StageResult& stage, const ReferenceCounters& baseline) {
  StageGateStats result;
  if (!baseline.valid() || baseline.per_query_correct.empty() ||
      stage.final_guarded_repetitions.empty()) return result;
  result.min_per_query_delta = kK;
  for (const auto& counters : stage.final_guarded_repetitions) {
    const PerQueryGateStats one = per_query_no_regression(counters, baseline);
    ++result.repetitions_checked;
    result.query_count = one.query_count;
    result.min_per_query_delta = std::min(result.min_per_query_delta, one.min_per_query_delta);
    if (one.violating_queries < 0) {
      result.violating_queries_sum = -1;
      result.max_violating_queries = -1;
      return result;
    }
    result.violating_queries_sum += one.violating_queries;
    result.max_violating_queries = std::max(result.max_violating_queries, one.violating_queries);
    if (!one.passed) return result;
  }
  result.passed = true;
  return result;
}

std::string json_escape(const std::string& raw) {
  std::ostringstream out;
  for (unsigned char c : raw) {
    if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
    else if (c == '\n') out << "\\n";
    else if (c < 0x20) out << ' ';
    else out << static_cast<char>(c);
  }
  return out.str();
}

std::string hex_u64(std::uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

void write_gamma_json(std::ostream& out, const std::array<float, RP_MAX_LEVELS>& gamma) {
  out << '[';
  for (int l = 0; l < RP_MAX_LEVELS; ++l) { if (l) out << ','; out << std::setprecision(8) << gamma[l]; }
  out << ']';
}

void write_reference_json(std::ostream& out, const ReferenceCounters& c) {
  out << "{\"correct\":" << c.correct << ",\"total\":" << c.total
      << ",\"recall_at_10\":" << std::setprecision(10)
      << (c.total ? static_cast<double>(c.correct) / static_cast<double>(c.total) : 0.0)
      << ",\"invalid_output_ids\":" << c.invalid_output_ids
      << ",\"duplicate_output_ids\":" << c.duplicate_output_ids
      << ",\"observed_distance_mismatches\":" << c.observed_distance_mismatches
      << ",\"reference_inconsistencies\":" << c.reference_inconsistencies
      << ",\"boundary_ties\":" << c.boundary_ties
      << ",\"first_error\":\"" << json_escape(c.first_error) << "\"}";
}

void write_reference_repetitions_json(std::ostream& out, const std::vector<ReferenceCounters>& repetitions) {
  out << '[';
  for (std::size_t i = 0; i < repetitions.size(); ++i) {
    if (i) out << ',';
    write_reference_json(out, repetitions[i]);
  }
  out << ']';
}

void write_per_query_artifacts_json(std::ostream& out, const std::vector<PerQueryArtifact>& artifacts) {
  out << '[';
  for (std::size_t i = 0; i < artifacts.size(); ++i) {
    if (i) out << ',';
    const auto& artifact = artifacts[i];
    out << "{\"path\":\"" << json_escape(artifact.path)
        << "\",\"fnv1a64\":\"" << hex_u64(artifact.fnv1a64)
        << "\",\"role\":\"" << json_escape(artifact.role)
        << "\",\"timed_rep\":" << artifact.timed_rep << '}';
  }
  out << ']';
}

void write_stage_json(std::ostream& out, const StageResult& stage, int query_count) {
  const int denominator = std::max(1, stage.timed_reps * std::max(1, query_count));
  const int rep_denominator = std::max(1, stage.timed_reps);
  const double flagged_rate = static_cast<double>(stage.gamma_only_flagged_queries_sum) /
      static_cast<double>(denominator);
  out << "{\"warmup_reps\":" << stage.warmups
      << ",\"timed_reps\":" << stage.timed_reps
      << ",\"query_count\":" << query_count
      << ",\"final_output_policy\":\"gamma=1 fallback overwrites every gamma-only-pruned query\""
      << ",\"raw_speculative_is_diagnostic_only\":true"
      << ",\"speculative_static_api\":{\"wall_ms_sum\":" << stage.speculative_static_api_wall_ms_sum
      << ",\"gpu_ms_sum\":" << stage.speculative_static_api_gpu_ms_sum
      << ",\"wall_ms_per_query\":" << (stage.speculative_static_api_wall_ms_sum / denominator)
      << ",\"gpu_ms_per_query\":" << (stage.speculative_static_api_gpu_ms_sum / denominator) << '}'
      << ",\"copies_and_fallback\":{\"speculative_result_ids_d2h_wall_ms_sum\":" << stage.speculative_d2h_ids_wall_ms_sum
      << ",\"speculative_result_distances_d2h_wall_ms_sum\":" << stage.speculative_d2h_distances_wall_ms_sum
      << ",\"gamma_only_flag_d2h_wall_ms_sum\":" << stage.flag_d2h_wall_ms_sum
      << ",\"fallback_query_h2d_wall_ms_sum\":" << stage.fallback_query_h2d_wall_ms_sum
      << ",\"fallback_static_api_wall_ms_sum\":" << stage.fallback_static_api_wall_ms_sum
      << ",\"fallback_static_api_gpu_ms_sum\":" << stage.fallback_static_api_gpu_ms_sum
      << ",\"fallback_result_ids_d2h_wall_ms_sum\":" << stage.fallback_d2h_ids_wall_ms_sum
      << ",\"fallback_result_distances_d2h_wall_ms_sum\":" << stage.fallback_d2h_distances_wall_ms_sum
      << ",\"fallback_result_splice_d2d_ids_wall_ms_sum\":" << stage.fallback_result_splice_d2d_ids_wall_ms_sum << '}'
      << ",\"gamma_only_fallback\":{\"flagged_queries_sum\":" << stage.gamma_only_flagged_queries_sum
      << ",\"flagged_query_rate\":" << flagged_rate
      << ",\"flagged_global_qids_fnv1a64\":\"" << hex_u64(stage.gamma_only_flagged_global_qids_fnv1a64)
      << "\",\"per_timed_rep_mean\":" << (static_cast<double>(stage.gamma_only_flagged_queries_sum) / rep_denominator) << '}'
      << ",\"guarded_timing\":{\"query_pipeline_wall_ms_sum\":" << stage.guarded_query_pipeline_wall_ms_sum
      << ",\"query_pipeline_wall_ms_per_query\":" << (stage.guarded_query_pipeline_wall_ms_sum / denominator)
      << ",\"full_runner_wall_ms_sum\":" << stage.guarded_total_runner_wall_ms_sum
      << ",\"raw_reference_evaluation_wall_ms_sum\":" << stage.raw_reference_evaluation_wall_ms_sum
      << ",\"final_reference_evaluation_wall_ms_sum\":" << stage.final_reference_evaluation_wall_ms_sum
      << ",\"frozen_tree_snapshot_wall_ms_sum_excluded_from_pipeline\":" << stage.frozen_tree_snapshot_wall_ms_sum
      << ",\"frozen_tree_snapshot_checks\":" << stage.frozen_tree_snapshot_checks_sum << '}'
      << ",\"raw_speculative_reference_sum\":";
  write_reference_json(out, stage.raw_speculative_reference_sum);
  out << ",\"final_guarded_reference_sum\":";
  write_reference_json(out, stage.final_guarded_reference_sum);
  out << ",\"raw_speculative_per_repetition\":";
  write_reference_repetitions_json(out, stage.raw_speculative_repetitions);
  out << ",\"final_guarded_per_repetition\":";
  write_reference_repetitions_json(out, stage.final_guarded_repetitions);
  out << ",\"per_query_artifacts\":";
  write_per_query_artifacts_json(out, stage.per_query_artifacts);
  out << '}';
}

// This diagnostic is deliberately written before fail-closed baseline abort.
// It cannot create a gamma candidate and it never reads v2 sealed test material.
void write_baseline_invalid_diagnostic(const std::string& path, const Args& args,
                                       const StageResult& baseline) {
  std::ofstream out(path);
  if (!out) fail("cannot write baseline-invalid diagnostic");
  const int query_count = static_cast<int>(
      baseline.final_guarded_reference_sum.total / std::max(1, baseline.timed_reps * kK));
  const ReferenceCounters& first = baseline.final_guarded_repetitions.empty()
      ? baseline.final_guarded_reference_sum : baseline.final_guarded_repetitions.front();
  out << std::setprecision(10)
      << "{\n"
      << "  \"schema\": \"safe-c2-v3-baseline-invalid-diagnostic-v1\",\n"
      << "  \"status\": \"FAIL_BASELINE_REFERENCE_BEFORE_GAMMA\",\n"
      << "  \"v3_speculative_fallback\": true,\n"
      << "  \"v2_sealed_test_policy\": \"consumed historical provenance; forbidden for v3 input/tuning\",\n"
      << "  \"mode\": \"" << args.mode << "\", \"stage\": \"" << args.stage << "\",\n"
      << "  \"workload_manifest\": \"" << json_escape(args.workload_manifest) << "\",\n"
      << "  \"reason\": \"gamma=1 calibration baseline reference check failed\",\n"
      << "  \"no_gamma_candidate_was_issued\": true,\n"
      << "  \"first_timed_final_guarded_reference\": ";
  write_reference_json(out, first);
  out << ",\n  \"baseline_stage\": ";
  write_stage_json(out, baseline, query_count);
  out << "\n}\n";
  if (!out) fail("failed writing baseline-invalid diagnostic");
}

struct CalibrationRecord {
  int level = -1;
  int iteration = -1;
  float candidate = 1.0F;
  long long raw_speculative_correct = 0;
  bool raw_speculative_reference_valid = false;
  long long gamma_only_flagged_queries = 0;
  int per_query_violating_queries = -1;
  int min_per_query_delta = -kK;
  bool accepted = false;
};

void write_summary(const std::string& path, const Args& args, const BuildTiming& build, const Runtime& runtime,
                   const MetricPlan& metric, const Timing& query_h2d, const IntervalCertificate& interval_certificate,
                   const std::array<float, RP_MAX_LEVELS>& gamma,
                   const StageResult& result, const StageResult* paired_baseline,
                   long long baseline_target, const StageGateStats& gate_stats,
                   const std::vector<std::string>& abba_timed_pair_order, int abba_shared_warmup_pairs,
                   const std::vector<CalibrationRecord>& calibration, double calibration_wall_ms,
                   const std::string& status, const std::string& gpu_name) {
  std::ofstream out(path);
  if (!out) fail("cannot write summary");
  const int result_query_count = static_cast<int>(result.final_guarded_reference_sum.total /
      std::max(1, result.timed_reps * kK));
  const bool speed_eligible = !abba_timed_pair_order.empty() && gate_stats.passed;
  out << std::setprecision(10);
  out << "{\n"
      << "  \"schema\": \"safe-c2-speculative-fallback-v3-run-v1\",\n"
      << "  \"status\": \"" << status << "\",\n"
      << "  \"scope\": \"v3 isolated Safe-C2 speculative gamma traversal with gamma-only fallback on an independently frozen compact workload; empirical guarded-output evaluation only\",\n"
      << "  \"implementation_identity\": \"v3 speculative-fallback skeleton; not v2 and not the submitted legacy C2 kernel\",\n"
      << "  \"v3_speculative_fallback\": true,\n"
      << "  \"v2_sealed_test_policy\": \"consumed historical provenance; forbidden for v3 gamma tuning, calibration, evaluation, retry, or input binding\",\n"
      << "  \"workload_binding\": {\"manifest\": \"" << json_escape(args.workload_manifest)
      << "\", \"schema\": \"gts-v3-compact-learn-workload-v1\", \"sha256_verified_in_runner\": true},\n"
      << "  \"mode\": \"" << args.mode << "\", \"stage\": \"" << args.stage << "\",\n"
      << "  \"input_paths\": {\"base_fvecs\":\"" << json_escape(args.base_fvecs)
      << "\",\"query_fvecs\":\"" << json_escape(args.query_fvecs)
      << "\",\"groundtruth_ivecs\":\"" << json_escape(args.groundtruth_ivecs)
      << "\",\"stage_query_ids\":\"" << json_escape(args.query_ids) << "\"},\n"
      << "  \"final_output_policy\": {\"gamma_is_speculative_not_metric_lower_bound\": true, \"gamma_only_condition\": \"unscaled_keep && !scaled_keep\", \"fallback\": \"re-query each flagged local query with all gamma=1 on the same FrozenTreeSnapshot; repair the host final IDs+distances pair and splice device IDs\", \"host_final_pair_contract\": \"evaluate_results validates final IDs+distances before JSONL; legacy res_dis is scratch and never claimed as a full-query output after subset fallback\", \"primary_reference\": \"final_guarded only\"},\n"
      << "  \"archived_incremental_updater_used\": false,\n"
      << "  \"dynamic_exact_smoke_used\": false,\n"
      << "  \"workspace_cap_bytes\": " << kWorkspaceCapBytes << ",\n"
      << "  \"tree\": {\"base_n\": " << kBaseN << ", \"dimension\": " << kDimension
      << ", \"height\": " << runtime.tree_height << ", \"metric\": \"raw-float32-L2\", \"effective_coordinate_type\": \"float32 (config.cuh macro short->float)\", \"infi_dis\": " << metric.infi_dis
      << ", \"bbox_l2_upper_bound\": " << static_cast<double>(metric.bbox_l2) << "},\n"
      << "  \"gamma_vector\": ";
  write_gamma_json(out, gamma);
  out << ",\n  \"c2_algorithm1\": {\"freeze_level\": " << kFreezeLevel
      << ", \"binary_iterations\": " << kBinaryIterations << ", \"candidate_lo\": " << kCandidateLo
      << ", \"candidate_hi\": " << kCandidateHi << ", \"mu\": " << kMu
      << ", \"order\": \"h-1 down to 3; raw speculative calibration trace only\"},\n"
      << "  \"timing\": {\"base_h2d_wall_ms\": " << build.base_h2d.wall_ms
      << ", \"base_h2d_gpu_ms\": " << build.base_h2d.gpu_ms
      << ", \"tree_build_wall_ms\": " << build.tree_build.wall_ms
      << ", \"tree_build_gpu_ms\": " << build.tree_build.gpu_ms
      << ", \"query_h2d_wall_ms\": " << query_h2d.wall_ms
      << ", \"query_h2d_gpu_ms\": " << query_h2d.gpu_ms
      << ", \"tree_snapshot_wall_ms\": " << build.snapshot_wall_ms
      << ", \"interval_certificate_wall_ms\": " << interval_certificate.wall_ms
      << ", \"calibration_wall_ms\": " << calibration_wall_ms << ", \"stage\": ";
  write_stage_json(out, result, result_query_count);
  out << "},\n  \"interval_certificate\": {\"status\": \"PASS_OUTWARD_SAFE_REPAIRED_AND_REVERIFIED\", \"corrected_after_submit\": true, \"repair_policy\": \"post-build D2H reconstruction then outward H2D replacement; block if post-repair directional verification fails\", \"method\": \"current padded leaf memberships propagated to heap ancestors\", \"node_count\": " << interval_certificate.node_count
      << ", \"active_nonroot_nodes\": " << interval_certificate.active_nonroot_nodes
      << ", \"leaf_nodes\": " << interval_certificate.leaf_nodes
      << ", \"membership_observations\": " << interval_certificate.membership_observations
      << ", \"original_min_above_actual_count\": " << interval_certificate.original_min_above_actual
      << ", \"original_max_below_actual_count\": " << interval_certificate.original_max_below_actual
      << ", \"min_outward_repaired_count\": " << interval_certificate.min_outward_repaired
      << ", \"max_outward_repaired_count\": " << interval_certificate.max_outward_repaired
      << ", \"before_interval_hash\": \"" << hex_u64(interval_certificate.before_interval_hash)
      << "\", \"after_interval_hash\": \"" << hex_u64(interval_certificate.after_interval_hash)
      << "\", \"post_reverify_passed\": " << (interval_certificate.post_reverify_passed ? "true" : "false")
      << ", \"relative_envelope\": " << kIntervalRelEnvelope
      << ", \"absolute_envelope\": " << kIntervalAbsEnvelope
      << ", \"max_exact_l2\": " << interval_certificate.max_exact_l2 << "},\n"
      << "  \"baseline_target_raw_speculative_correct\": " << baseline_target
      << ", \"final_guarded_per_query_no_regression_gate\": {\"passed\": " << (gate_stats.passed ? "true" : "false")
      << ", \"repetitions_checked\": " << gate_stats.repetitions_checked
      << ", \"query_count\": " << gate_stats.query_count
      << ", \"violating_queries_sum\": " << gate_stats.violating_queries_sum
      << ", \"max_violating_queries\": " << gate_stats.max_violating_queries
      << ", \"min_per_query_delta\": " << gate_stats.min_per_query_delta << "},\n"
      << "  \"timing_comparison\": {\"eligible_for_speed_comparison\": " << (speed_eligible ? "true" : "false")
      << ", \"metric_if_eligible\": \"guarded_timing.query_pipeline_wall_ms; never speculative_static_api alone\""
      << ", \"scheme\": \"" << (!abba_timed_pair_order.empty() ? "heldout ABBA: alternate baseline->speculative_guarded and speculative_guarded->baseline after shared warmup" : "calibration correctness-only; not eligible for speed comparison") << "\""
      << ", \"shared_warmup_pairs\": " << abba_shared_warmup_pairs << ", \"timed_pair_order\":[";
  for (std::size_t i = 0; i < abba_timed_pair_order.size(); ++i) {
    if (i) out << ',';
    out << "\"" << abba_timed_pair_order[i] << "\"";
  }
  out << "]},\n";
  if (paired_baseline != nullptr) {
    const int baseline_query_count = static_cast<int>(paired_baseline->final_guarded_reference_sum.total /
        std::max(1, paired_baseline->timed_reps * kK));
    out << "  \"paired_same_split_baseline\": ";
    write_stage_json(out, *paired_baseline, baseline_query_count);
    const double baseline_recall = paired_baseline->final_guarded_reference_sum.total
        ? static_cast<double>(paired_baseline->final_guarded_reference_sum.correct) /
              paired_baseline->final_guarded_reference_sum.total : 0.0;
    const double final_recall = result.final_guarded_reference_sum.total
        ? static_cast<double>(result.final_guarded_reference_sum.correct) /
              result.final_guarded_reference_sum.total : 0.0;
    const double raw_recall = result.raw_speculative_reference_sum.total
        ? static_cast<double>(result.raw_speculative_reference_sum.correct) /
              result.raw_speculative_reference_sum.total : 0.0;
    out << ",\n  \"paired_final_guarded_recall_delta_minus_baseline\": " << (final_recall - baseline_recall)
        << ",\n  \"paired_raw_speculative_recall_delta_minus_baseline_diagnostic\": " << (raw_recall - baseline_recall) << ",\n";
  }
  out << "  \"calibration_trace\": [";
  for (std::size_t i = 0; i < calibration.size(); ++i) {
    const auto& r = calibration[i]; if (i) out << ',';
    out << "{\"level\":" << r.level << ",\"iteration\":" << r.iteration << ",\"candidate\":" << r.candidate
        << ",\"raw_speculative_correct\":" << r.raw_speculative_correct
        << ",\"raw_speculative_reference_valid\":" << (r.raw_speculative_reference_valid ? "true" : "false")
        << ",\"gamma_only_flagged_queries\":" << r.gamma_only_flagged_queries
        << ",\"per_query_violating_queries\":" << r.per_query_violating_queries
        << ",\"min_per_query_delta\":" << r.min_per_query_delta
        << ",\"accepted\":" << (r.accepted ? "true" : "false") << '}';
  }
  out << "],\n  \"gpu\": {\"name\": \"" << json_escape(gpu_name) << "\"},\n"
      << "  \"not_established\": [\"universal no-false-negative safety for gamma>1\", \"that unflagged speculative output is identical to gamma=1 without the heldout gate\", \"dynamic update/C3 behavior\", \"drift/recalibration deployment\", \"performance equivalence to the submitted artifact\", \"equivalence to v2 or submitted legacy C2 implementation\"]\n"
      << "}\n";
  if (!out) fail("failed writing summary");
}

int safe_c2_main(int argc, char** argv) {
  Runtime runtime;
  float* selected_queries_h = nullptr;
  float* selected_queries_d = nullptr;
  try {
    const Args args = parse_args(argc, argv);
    // Bind every run to the independently frozen compact workload before any
    // dataset/tree allocation.  Direct paths and copied v2 sealed test IDs are
    // both rejected; no v3 calibration can silently reuse v2 material.
    validate_workload_binding(args);
    // Require caller-created empty output directory, preventing accidental
    // overwrites of a frozen stage result.
    { std::ifstream probe(args.output_dir + "/summary.json"); if (probe.good()) fail("summary already exists in --out"); }
    const auto load_start = std::chrono::steady_clock::now();
    const Fvecs base = read_fvecs(args.base_fvecs, kBaseN, kDimension);
    const Fvecs all_queries = read_fvecs(args.query_fvecs, kQueryN, kDimension);
    const std::vector<int> gt = read_ivecs(args.groundtruth_ivecs, kQueryN, kGtWidth);
    const std::vector<int> selected_ids = read_query_ids(args.query_ids);
    const MetricPlan metric = metric_plan(base, all_queries, selected_ids);
    BuildTiming build;
    build.input_load_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
    runtime = build_tree(base, metric, &build);
    if (runtime.tree_height < 4 || runtime.tree_height > RP_MAX_LEVELS) {
      fail("runtime tree height outside frozen Safe-C2 protocol range");
    }
    const IntervalCertificate interval_certificate = certify_and_outward_repair_intervals(runtime, base);
    const auto snapshot_start = std::chrono::steady_clock::now();
    FrozenTreeSnapshot frozen; frozen.capture(runtime);
    build.snapshot_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - snapshot_start).count();

    const int selected_count = static_cast<int>(selected_ids.size());
    C2_CUDA(cudaMallocHost(&selected_queries_h, static_cast<std::size_t>(selected_count) * kDimension * sizeof(float)));
    for (int local = 0; local < selected_count; ++local) {
      const float* source = all_queries.values.data() + static_cast<std::size_t>(selected_ids[local]) * kDimension;
      std::memcpy(selected_queries_h + static_cast<std::size_t>(local) * kDimension, source, kDimension * sizeof(float));
    }
    C2_CUDA(cudaMalloc(&selected_queries_d, static_cast<std::size_t>(selected_count) * kDimension * sizeof(float)));
    cudaEvent_t h2d_begin = nullptr, h2d_end = nullptr;
    C2_CUDA(cudaEventCreate(&h2d_begin)); C2_CUDA(cudaEventCreate(&h2d_end));
    const auto query_h2d_start = std::chrono::steady_clock::now();
    C2_CUDA(cudaEventRecord(h2d_begin));
    C2_CUDA(cudaMemcpyAsync(selected_queries_d, selected_queries_h,
                             static_cast<std::size_t>(selected_count) * kDimension * sizeof(float), cudaMemcpyHostToDevice));
    const Timing query_h2d = elapsed_cuda_event(h2d_begin, h2d_end, query_h2d_start);
    C2_CUDA(cudaEventDestroy(h2d_begin)); C2_CUDA(cudaEventDestroy(h2d_end));

    cudaDeviceProp prop{}; int device = 0; C2_CUDA(cudaGetDevice(&device)); C2_CUDA(cudaGetDeviceProperties(&prop, device));
    std::array<float, RP_MAX_LEVELS> gamma{}; gamma.fill(1.0F);
    StageResult stage_result;
    StageResult paired_baseline_result;
    bool has_paired_baseline = false;
    long long baseline_target = args.baseline_target_correct;
    std::vector<CalibrationRecord> calibration_trace;
    double calibration_wall_ms = 0.0;
    StageGateStats gate_stats;
    std::vector<std::string> abba_timed_pair_order;
    int abba_shared_warmup_pairs = 0;
    bool passed = false;

    if (args.mode == "calibrate") {
      const auto calibration_start = std::chrono::steady_clock::now();
      const auto all_ones = all_ones_gamma();
      // Calibration split only: gamma=1 establishes the reference.  The v3
      // manifest and argument parser reject v2 sealed test material before this.
      const StageResult baseline = run_stage(
          runtime, frozen, selected_queries_d, selected_count, base, all_queries, gt,
          selected_ids, args.warmup_reps, 1, QueryMode::kBaselineGammaOne, all_ones,
          args.output_dir, "calibration_baseline", true);
      if (baseline.final_guarded_repetitions.size() != 1 ||
          !baseline.final_guarded_repetitions[0].valid()) {
        write_baseline_invalid_diagnostic(args.output_dir + "/baseline_invalid_diagnostic.json", args, baseline);
        fail("calibration gamma=1 baseline reference check failed");
      }
      const ReferenceCounters calibration_baseline = baseline.final_guarded_repetitions[0];
      baseline_target = calibration_baseline.correct;
      paired_baseline_result = baseline;
      has_paired_baseline = true;
      // Algorithm 1 selection remains intentionally conservative: it consults
      // raw speculative calibration output only.  Final guarded output cannot
      // make a candidate appear better by hiding raw loss with fallback.
      for (int level = runtime.tree_height - 1; level >= kFreezeLevel + 1; --level) {
        float lo = kCandidateLo, hi = kCandidateHi, best = 1.0F;
        for (int iter = 0; iter < kBinaryIterations; ++iter) {
          const float mid = (lo + hi) / 2.0F;
          gamma[level] = mid;
          const std::string role = "calibration_candidate_l" + std::to_string(level) +
              "_i" + std::to_string(iter);
          const StageResult candidate = run_stage(
              runtime, frozen, selected_queries_d, selected_count, base, all_queries, gt,
              selected_ids, 0, 1, QueryMode::kSpeculativeGammaWithFallback, gamma,
              args.output_dir, role, true);
          const ReferenceCounters& raw = candidate.raw_speculative_repetitions.at(0);
          const PerQueryGateStats candidate_gate = per_query_no_regression(raw, calibration_baseline);
          const bool accepted = raw.valid() && raw.correct >= baseline_target && candidate_gate.passed;
          calibration_trace.push_back(CalibrationRecord{
              level, iter, mid, raw.correct, raw.valid(), candidate.gamma_only_flagged_queries_sum,
              candidate_gate.violating_queries, candidate_gate.min_per_query_delta, accepted});
          if (accepted) { best = mid; lo = mid; } else { hi = mid; }
        }
        gamma[level] = 1.0F + kMu * (best - 1.0F);
        // Conservative post-margin confirmation, again using raw speculative
        // calibration output.  It is not a held-out claim.
        const std::string confirm_role = "calibration_confirm_l" + std::to_string(level);
        const StageResult confirm = run_stage(
            runtime, frozen, selected_queries_d, selected_count, base, all_queries, gt,
            selected_ids, 0, 1, QueryMode::kSpeculativeGammaWithFallback, gamma,
            args.output_dir, confirm_role, true);
        const ReferenceCounters& raw = confirm.raw_speculative_repetitions.at(0);
        const PerQueryGateStats confirm_gate = per_query_no_regression(raw, calibration_baseline);
        const bool confirmed = raw.valid() && raw.correct >= baseline_target && confirm_gate.passed;
        calibration_trace.push_back(CalibrationRecord{
            level, kBinaryIterations, gamma[level], raw.correct, raw.valid(),
            confirm.gamma_only_flagged_queries_sum, confirm_gate.violating_queries,
            confirm_gate.min_per_query_delta, confirmed});
        if (!confirmed) gamma[level] = 1.0F;
      }
      calibration_wall_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - calibration_start).count();
      // Frozen final vector: this measurement is not used for selection.  Its
      // final guarded result is checked against the same gamma=1 baseline.
      stage_result = run_stage(
          runtime, frozen, selected_queries_d, selected_count, base, all_queries, gt,
          selected_ids, 0, 1, QueryMode::kSpeculativeGammaWithFallback, gamma,
          args.output_dir, "calibration_final_guarded", true);
      gate_stats = stage_final_guarded_gate(stage_result, calibration_baseline);
      passed = gate_stats.passed;
      write_gamma_vector(args.output_dir + "/final_v3_speculative_gamma_vector.txt", gamma);
    } else {
      gamma = read_gamma_vector(args.gamma_vector_file);
      for (int level = 0; level <= kFreezeLevel; ++level) {
        if (gamma[level] != 1.0F) fail("gamma vector violates frozen shallow levels");
      }
      // Fixed-vector held-out control: this stage cannot alter gamma, mu, order,
      // or workload membership.  Its paired timing output includes fallback cost.
      const PairedAbbaStageResult paired = run_paired_abba_stage(
          runtime, frozen, selected_queries_d, selected_count, base, all_queries, gt, selected_ids,
          gamma, args.warmup_reps, args.timed_reps, args.output_dir);
      paired_baseline_result = paired.baseline;
      stage_result = paired.speculative_guarded;
      abba_timed_pair_order = paired.timed_pair_order;
      abba_shared_warmup_pairs = paired.shared_warmup_pairs;
      has_paired_baseline = true;
      if (paired_baseline_result.final_guarded_repetitions.empty()) {
        fail("missing paired ABBA gamma=1 baseline result");
      }
      const ReferenceCounters heldout_baseline = paired_baseline_result.final_guarded_repetitions.front();
      baseline_target = heldout_baseline.correct;
      for (const auto& counters : paired_baseline_result.final_guarded_repetitions) {
        if (!counters.valid() || counters.correct != baseline_target ||
            !same_per_query_overlap(counters, heldout_baseline)) {
          fail("paired ABBA gamma=1 baseline is invalid or per-query nondeterministic; v3 heldout stage is blocked");
        }
      }
      gate_stats = stage_final_guarded_gate(stage_result, heldout_baseline);
      passed = gate_stats.passed;
    }

    // Query H2D is independent of GTS static API timing; attach it to build so
    // summary consumers cannot accidentally fold it into per-query latency.
    build.input_load_wall_ms += 0.0;  // retained separately in future schema revisions
    const std::string status = passed ? (args.mode == "calibrate" ? "PASS_V3_CALIBRATION_GUARDED" : "PASS_V3_HELDOUT_GUARDED")
                                      : (args.mode == "calibrate" ? "FAIL_V3_CALIBRATION_GUARDED" : "FAIL_V3_HELDOUT_GUARDED");
    // Store query H2D in an adjacent compact file; summary also gets it patched below.
    write_summary(args.output_dir + "/summary.json", args, build, runtime, metric, query_h2d, interval_certificate, gamma, stage_result,
                  has_paired_baseline ? &paired_baseline_result : nullptr,
                  baseline_target, gate_stats, abba_timed_pair_order, abba_shared_warmup_pairs,
                  calibration_trace, calibration_wall_ms, status, prop.name);
    {
      std::ofstream qh2d(args.output_dir + "/query_h2d.json");
      qh2d << std::setprecision(10) << "{\"query_count\":" << selected_count << ",\"query_h2d_wall_ms\":"
           << query_h2d.wall_ms << ",\"query_h2d_gpu_ms\":" << query_h2d.gpu_ms
           << ",\"note\":\"separate from static GTS query API timing\"}\n";
    }
    C2_CUDA(cudaFree(selected_queries_d)); selected_queries_d = nullptr;
    C2_CUDA(cudaFreeHost(selected_queries_h)); selected_queries_h = nullptr;
    release_tree(runtime);
    return passed ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    if (res_dis) { cudaFree(res_dis); res_dis = nullptr; }
    if (selected_queries_d) cudaFree(selected_queries_d);
    if (selected_queries_h) cudaFreeHost(selected_queries_h);
    release_tree(runtime);
    return 2;
  }
}

}  // namespace safe_c2

int main(int argc, char** argv) {
  return safe_c2::safe_c2_main(argc, argv);
}
