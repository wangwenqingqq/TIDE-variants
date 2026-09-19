#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

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

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <link.h>
#include <map>
#include <optional>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <linux/fs.h>
#include <vector>

extern char** environ;

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
    "safe_c1_native_matrix_g3_v10_nvml_pci_bufferfix_pilot";
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

// Source closure bytes below are runtime source-audit inputs only.  They do
// not attest the already executing binary or its link image; a later external
// build/launch admission must bind wrapper/source hashes, exact one-TU command,
// executable SHA-256, and dynamic-link inspection before any native launch.
struct ExpectedFile {
  const char* relative;
  const char* sha256;
  std::size_t exact_bytes;  // zero means textual size is not independently fixed.
};

constexpr ExpectedFile kSourceClosure[] = {
    {"src/g3_safe_c1_native_matrix.cu",
     "12ef95e115c92ed2ebc86fc116aa441aacdbadfe850676a684c0b9119aa8b138", 0},
    {"src/g3_safe_search_v2.cuh",
     "65688fedcbb05a1fff680555e3139b6e640bfdd72288b93dfccddcec90dfbf01", 0},
    {"reference/include/tree.cuh",
     "c1324bef173358c31a8371e1f050d4372cd1c92cd98c71af3832b2d19c769fd6", 0},
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
  // Hash and parsing must consume the same immutable in-memory bytes.  Open
  // the final component with O_NOFOLLOW and retain the FD while reading, rather
  // than symlink_status()+ifstream followed by a second path read.
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    const int saved_errno = errno;
    runner_fail("cannot securely open input: " + path.string() + ": " +
                std::strerror(saved_errno));
  }
  struct stat status {};
  if (::fstat(fd, &status) != 0) {
    const int saved_errno = errno;
    (void)::close(fd);
    runner_fail("cannot fstat input: " + path.string() + ": " +
                std::strerror(saved_errno));
  }
  if (!S_ISREG(status.st_mode) || status.st_size < 0 ||
      static_cast<std::uintmax_t>(status.st_size) >
          static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
    (void)::close(fd);
    runner_fail("required input is not a safely bounded regular file: " + path.string());
  }
  std::vector<std::uint8_t> bytes;
  bytes.reserve(static_cast<std::size_t>(status.st_size));
  std::array<std::uint8_t, 65536> chunk{};
  while (true) {
    const ssize_t count = ::read(fd, chunk.data(), chunk.size());
    if (count == 0) break;
    if (count < 0) {
      if (errno == EINTR) continue;
      const int saved_errno = errno;
      (void)::close(fd);
      runner_fail("cannot fully read input: " + path.string() + ": " +
                  std::strerror(saved_errno));
    }
    bytes.insert(bytes.end(), chunk.begin(), chunk.begin() + count);
  }
  if (::close(fd) != 0) {
    const int saved_errno = errno;
    runner_fail("cannot close input FD: " + path.string() + ": " +
                std::strerror(saved_errno));
  }
  return bytes;
}

std::string text_from_bytes(const std::vector<std::uint8_t>& bytes,
                            const std::string& label) {
  for (std::uint8_t byte : bytes) require(byte != 0, label + " unexpectedly contains NUL");
  return std::string(bytes.begin(), bytes.end());
}

std::string source_closure_descriptor_sha256() {
  std::ostringstream descriptor;
  descriptor << "safe-c1-g3-source-closure-v1\n";
  for (const ExpectedFile& file : kSourceClosure) {
    descriptor << file.relative << ':' << file.sha256 << '\n';
  }
  return runner_sha256_text(descriptor.str());
}

bool canonical_nvml_driver_version(const std::string& value) {
  if (value.empty() || value.size() > 63) return false;
  bool saw_dot = false;
  bool previous_dot = true;
  for (char character : value) {
    if (character == '.') {
      if (previous_dot) return false;
      saw_dot = true;
      previous_dot = true;
    } else if (character >= '0' && character <= '9') {
      previous_dot = false;
    } else {
      return false;
    }
  }
  return saw_dot && !previous_dot;
}

std::string current_executable_path_or_fail() {
  std::array<char, 4096> buffer{};
  const ssize_t length = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  require(length > 0 && static_cast<std::size_t>(length) < buffer.size() - 1,
          "cannot resolve the executing binary through /proc/self/exe");
  return std::string(buffer.data(), static_cast<std::size_t>(length));
}

// This is a pre-native host-ELF snapshot. It deliberately does not make a
// claim about CUDA-driver/JIT images that may appear after the first CUDA API.
// Ordinary images are bound through /proc/self/map_files to an FD for the
// backing object that is actually mapped in this process. The only pseudo
// image accepted is the kernel vDSO.
struct PreNativeHostLoaderDescriptor final {
  std::string text;
  std::string sha256;
};

bool canonical_gpu_uuid(const std::string& value) {
  if (value.size() != 40 || value.rfind("GPU-", 0) != 0) return false;
  constexpr std::array<std::size_t, 4> hyphens = {12, 17, 22, 27};
  for (std::size_t index = 4; index < value.size(); ++index) {
    const bool must_be_hyphen =
        std::find(hyphens.begin(), hyphens.end(), index) != hyphens.end();
    if (must_be_hyphen) {
      if (value[index] != '-') return false;
    } else if (!((value[index] >= '0' && value[index] <= '9') ||
                 (value[index] >= 'a' && value[index] <= 'f') ||
                 (value[index] >= 'A' && value[index] <= 'F'))) {
      return false;
    }
  }
  return true;
}

std::string require_controlled_environment_or_fail(bool require_explicit_gpu) {
  std::size_t path_count = 0;
  std::size_t home_count = 0;
  std::size_t gpu_count = 0;
  std::string gpu_uuid;
  for (char** entry = ::environ; entry != nullptr && *entry != nullptr; ++entry) {
    const std::string value(*entry);
    if (value == "PATH=/usr/bin:/bin") {
      ++path_count;
    } else if (value == "HOME=/nonexistent") {
      ++home_count;
    } else if (value.rfind("CUDA_VISIBLE_DEVICES=", 0) == 0) {
      const std::string candidate =
          value.substr(std::string("CUDA_VISIBLE_DEVICES=").size());
      require(canonical_gpu_uuid(candidate),
              "controlled run launcher must use exactly one canonical GPU UUID");
      ++gpu_count;
      gpu_uuid = candidate;
    } else {
      runner_fail("controlled launcher environment has an unapproved variable: " +
                  value.substr(0, value.find('=')));
    }
  }
  require(path_count == 1 && home_count == 1,
          "controlled launcher must provide one fixed PATH and HOME through env -i");
  require(!require_explicit_gpu || gpu_count == 1,
          "controlled run launcher must provide one canonical CUDA_VISIBLE_DEVICES UUID");
  require(require_explicit_gpu || gpu_count == 0,
          "pre-native control descriptor must not carry a CUDA device selection");
  return gpu_uuid;
}

void require_clean_loader_environment_or_fail() {
  (void)require_controlled_environment_or_fail(false);
}

std::string require_clean_run_environment_or_fail() {
  return require_controlled_environment_or_fail(true);
}

std::uintptr_t parse_proc_hex_address_or_fail(const std::string& text,
                                              const std::string& label) {
  require(!text.empty() &&
              std::all_of(text.begin(), text.end(), [](unsigned char c) {
                return std::isxdigit(c) != 0;
              }),
          label + " is not hexadecimal");
  std::size_t consumed = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &consumed, 16);
  } catch (const std::exception&) {
    runner_fail(label + " cannot be parsed");
  }
  require(consumed == text.size() &&
              value <= static_cast<unsigned long long>(
                           std::numeric_limits<std::uintptr_t>::max()),
          label + " is outside uintptr_t range");
  return static_cast<std::uintptr_t>(value);
}

struct ProcMapping final {
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
  std::string pathname;
};

std::vector<ProcMapping> proc_self_mappings_or_fail() {
  const std::string maps = text_from_bytes(
      read_regular_file_strict(fs::path("/proc/self/maps")), "/proc/self/maps");
  std::vector<ProcMapping> output;
  std::istringstream stream(maps);
  std::string line;
  while (std::getline(stream, line)) {
    require(!line.empty() && line.back() != '\r',
            "/proc/self/maps has malformed line");
    std::istringstream fields(line);
    std::string range;
    std::string permissions;
    std::string offset;
    std::string device;
    std::string inode;
    require(static_cast<bool>(fields >> range >> permissions >> offset >> device >> inode),
            "/proc/self/maps record lacks required fields");
    std::string pathname;
    std::getline(fields, pathname);
    while (!pathname.empty() && pathname.front() == ' ') pathname.erase(pathname.begin());
    const std::size_t dash = range.find('-');
    require(dash != std::string::npos && dash != 0 && dash + 1 < range.size() &&
                range.find('-', dash + 1) == std::string::npos,
            "/proc/self/maps address range is malformed");
    const std::uintptr_t begin = parse_proc_hex_address_or_fail(
        range.substr(0, dash), "/proc/self/maps start address");
    const std::uintptr_t end = parse_proc_hex_address_or_fail(
        range.substr(dash + 1), "/proc/self/maps end address");
    require(begin < end, "/proc/self/maps range is empty or reversed");
    output.push_back(ProcMapping{begin, end, std::move(pathname)});
  }
  require(!output.empty(), "/proc/self/maps has no mappings");
  return output;
}

fs::path proc_map_file_path(const ProcMapping& mapping) {
  std::ostringstream name;
  name << std::hex << mapping.begin << '-' << mapping.end;
  return fs::path("/proc/self/map_files") / name.str();
}

struct MappedFileBytes final {
  std::vector<std::uint8_t> bytes;
  std::uintmax_t device = 0;
  std::uintmax_t inode = 0;
};

MappedFileBytes read_actual_mapped_file_or_fail(const ProcMapping& mapping) {
  const fs::path map_file = proc_map_file_path(mapping);
  // map_files is a kernel-owned link to the backing object for this mapping.
  // Opening it (without O_NOFOLLOW) is intentional: the FD binds the hash to
  // the mapped object even if a pathname was replaced or deleted.
  const int fd = ::open(map_file.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    const int saved_errno = errno;
    runner_fail("cannot open mapped ELF backing file: " + map_file.string() + ": " +
                std::strerror(saved_errno));
  }
  struct stat status {};
  if (::fstat(fd, &status) != 0) {
    const int saved_errno = errno;
    (void)::close(fd);
    runner_fail("cannot fstat mapped ELF backing file: " + map_file.string() + ": " +
                std::strerror(saved_errno));
  }
  if (!S_ISREG(status.st_mode) || status.st_size < 0 ||
      static_cast<std::uintmax_t>(status.st_size) >
          static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
    (void)::close(fd);
    runner_fail("mapped ELF backing object is not a bounded regular file: " +
                map_file.string());
  }
  std::vector<std::uint8_t> bytes;
  bytes.reserve(static_cast<std::size_t>(status.st_size));
  std::array<std::uint8_t, 65536> chunk{};
  while (true) {
    const ssize_t count = ::read(fd, chunk.data(), chunk.size());
    if (count == 0) break;
    if (count < 0) {
      if (errno == EINTR) continue;
      const int saved_errno = errno;
      (void)::close(fd);
      runner_fail("cannot read mapped ELF backing file: " + map_file.string() + ": " +
                  std::strerror(saved_errno));
    }
    bytes.insert(bytes.end(), chunk.begin(), chunk.begin() + count);
  }
  if (::close(fd) != 0) {
    const int saved_errno = errno;
    runner_fail("cannot close mapped ELF backing-file FD: " + map_file.string() + ": " +
                std::strerror(saved_errno));
  }
  return MappedFileBytes{std::move(bytes), static_cast<std::uintmax_t>(status.st_dev),
                         static_cast<std::uintmax_t>(status.st_ino)};
}

struct LoaderImage final {
  std::string raw_name;
  std::uintptr_t load_bias = 0;
  std::vector<ElfW(Phdr)> headers;
};

struct LoaderImageCollection final {
  std::vector<LoaderImage> images;
  bool failed = false;
};

int collect_loader_image(struct dl_phdr_info* info, std::size_t, void* opaque) noexcept {
  auto* collection = static_cast<LoaderImageCollection*>(opaque);
  try {
    if (info == nullptr || info->dlpi_phdr == nullptr || info->dlpi_phnum == 0) {
      collection->failed = true;
      return 1;
    }
    LoaderImage image;
    image.raw_name = info->dlpi_name == nullptr ? "" : info->dlpi_name;
    image.load_bias = static_cast<std::uintptr_t>(info->dlpi_addr);
    image.headers.assign(info->dlpi_phdr, info->dlpi_phdr + info->dlpi_phnum);
    collection->images.push_back(std::move(image));
    return 0;
  } catch (...) {
    collection->failed = true;
    return 1;
  }
}

bool allowed_vdso_name(const std::string& value) {
  return value == "linux-vdso.so.1" || value == "linux-gate.so.1";
}

const ProcMapping& mapping_for_loader_image_or_fail(
    const LoaderImage& image, const std::vector<ProcMapping>& mappings) {
  for (const ElfW(Phdr)& header : image.headers) {
    if (header.p_type != PT_LOAD || header.p_filesz == 0) continue;
    const std::uintmax_t vaddr = static_cast<std::uintmax_t>(header.p_vaddr);
    const std::uintmax_t memsz = static_cast<std::uintmax_t>(header.p_memsz);
    require(vaddr <= static_cast<std::uintmax_t>(
                         std::numeric_limits<std::uintptr_t>::max()) &&
                memsz <= static_cast<std::uintmax_t>(
                         std::numeric_limits<std::uintptr_t>::max()) &&
                image.load_bias <= std::numeric_limits<std::uintptr_t>::max() -
                                       static_cast<std::uintptr_t>(vaddr),
            "loader PT_LOAD address is outside controlled uintptr_t range");
    const std::uintptr_t segment_begin =
        image.load_bias + static_cast<std::uintptr_t>(vaddr);
    require(memsz <= static_cast<std::uintmax_t>(
                         std::numeric_limits<std::uintptr_t>::max() - segment_begin),
            "loader PT_LOAD size overflows controlled uintptr_t range");
    const std::uintptr_t segment_end =
        segment_begin + static_cast<std::uintptr_t>(memsz);
    for (const ProcMapping& mapping : mappings) {
      if (mapping.begin < segment_end && segment_begin < mapping.end) return mapping;
    }
  }
  runner_fail("cannot bind a loader image to a /proc/self/maps file mapping");
}

PreNativeHostLoaderDescriptor pre_native_host_loader_descriptor_or_fail() {
  LoaderImageCollection collection;
  const int iterate_result = ::dl_iterate_phdr(collect_loader_image, &collection);
  require(iterate_result == 0 && !collection.failed && !collection.images.empty(),
          "cannot enumerate pre-native host loader images");
  const std::vector<ProcMapping> mappings = proc_self_mappings_or_fail();
  std::vector<std::string> rows;
  rows.reserve(collection.images.size());
  for (const LoaderImage& image : collection.images) {
    const ProcMapping& mapping = mapping_for_loader_image_or_fail(image, mappings);
    if (mapping.pathname == "[vdso]") {
      require(allowed_vdso_name(image.raw_name),
              "kernel vDSO mapping has an unrecognized loader name");
      rows.push_back("pseudo=" + image.raw_name);
      continue;
    }
    // A regular ELF may use a bare/relative SONAME equal to a vDSO spelling.
    // It is not a pseudo image: map_files binds and hashes its real backing FD.
    const MappedFileBytes mapped = read_actual_mapped_file_or_fail(mapping);
    rows.push_back("file;device=" + std::to_string(mapped.device) +
                   ";inode=" + std::to_string(mapped.inode) +
                   ";sha256=" + sha256_bytes(mapped.bytes));
  }
  std::sort(rows.begin(), rows.end());
  require(std::adjacent_find(rows.begin(), rows.end()) == rows.end(),
          "pre-native host loader image has duplicate mapped identities");
  std::ostringstream descriptor;
  descriptor << "schema=safe-c1-g3-pre-native-host-loader-image-v2\n";
  for (const std::string& row : rows) descriptor << "object=" << row << '\n';
  const std::string text = descriptor.str();
  return PreNativeHostLoaderDescriptor{text, runner_sha256_text(text)};
}

std::map<std::string, std::string> parse_strict_admission_kv(const std::string& text,
                                                              const std::string& label) {
  std::map<std::string, std::string> output;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t end = text.find('\n', begin);
    if (begin == text.size()) break;  // permit one conventional final LF
    const std::string line = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    require(!line.empty() && line.back() != '\r', label + " has an empty or CRLF record");
    const std::size_t equals = line.find('=');
    require(equals != std::string::npos && equals != 0 && equals + 1 < line.size() &&
                line.find('=', equals + 1) == std::string::npos,
            label + " has malformed key=value record");
    const std::string key = line.substr(0, equals);
    const std::string value = line.substr(equals + 1);
    require(key.find_first_of(" \t\r\n") == std::string::npos &&
                value.find_first_of("\r\n") == std::string::npos,
            label + " has whitespace/control in key or value");
    require(output.emplace(key, value).second, label + " duplicates an admission key");
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return output;
}

const std::string& admission_value(const std::map<std::string, std::string>& fields,
                                   const char* key) {
  const auto found = fields.find(key);
  require(found != fields.end(), std::string("admission lacks required key: ") + key);
  return found->second;
}

void require_sha256_hex(const std::string& value, const std::string& label) {
  require(value.size() == 64 &&
              std::all_of(value.begin(), value.end(), [](char c) {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
              }),
          label + " is not lowercase SHA-256 hex");
}

struct LaunchArguments final {
  fs::path final_run_dir;
  fs::path admission_path;
};

LaunchArguments parse_launch_arguments_or_fail(int argc, char** argv) {
  require(argc == 5 && std::string_view(argv[1]) == "--run-dir" &&
              std::string_view(argv[3]) == "--admission",
          "usage is exactly: g3_4k_pilot_wrapper --run-dir <absolute-v6-runs-child> --admission <absolute-v6-admission>");
  return LaunchArguments{fs::absolute(fs::path(argv[2])).lexically_normal(),
                         fs::absolute(fs::path(argv[4])).lexically_normal()};
}

std::string lowercase_ascii(std::string value) {
  for (char& character : value) {
    if (character >= 'A' && character <= 'Z') {
      character = static_cast<char>(character - 'A' + 'a');
    }
  }
  return value;
}

void require_dynamic_loader_calls_absent(const std::string& text,
                                         const std::string& label) {
  // The spellings are assembled so this source can inspect the wrapper itself
  // without matching this checker implementation.
  const std::array<std::string, 4> forbidden = {
      std::string("dl") + "open", std::string("dl") + "sym",
      std::string("dlm") + "open", std::string("dlv") + "sym"};
  for (const std::string& symbol : forbidden) {
    require(text.find(symbol) == std::string::npos,
            label + " contains forbidden dynamic-loader symbol: " + symbol);
  }
}

bool valid_sm_arch(const std::string& value) {
  return value.size() >= 5 && value.rfind("sm_", 0) == 0 &&
         std::all_of(value.begin() + 3, value.end(), [](unsigned char c) {
           return c >= static_cast<unsigned char>('0') &&
                  c <= static_cast<unsigned char>('9');
         });
}

std::vector<std::string> parse_dynamic_needed_sonames_or_fail(
    const std::string& readelf_text) {
  std::vector<std::string> output;
  std::istringstream lines(readelf_text);
  std::string line;
  constexpr char kMarker[] = "Shared library: [";
  while (std::getline(lines, line)) {
    if (line.find("(NEEDED)") == std::string::npos) continue;
    const std::size_t marker = line.find(kMarker);
    require(marker != std::string::npos, "readelf NEEDED record lacks a SONAME");
    const std::size_t begin = marker + std::char_traits<char>::length(kMarker);
    const std::size_t end = line.find(']', begin);
    require(end != std::string::npos && end > begin,
            "readelf NEEDED SONAME is malformed");
    const std::string soname = line.substr(begin, end - begin);
    require(soname.find_first_of(" \t\r\n/") == std::string::npos,
            "readelf NEEDED SONAME has unsafe characters");
    output.push_back(soname);
  }
  std::sort(output.begin(), output.end());
  require(std::adjacent_find(output.begin(), output.end()) == output.end(),
          "readelf has duplicate NEEDED SONAME records");
  return output;
}

void require_needed_allowlist_or_fail(const std::vector<std::string>& sonames) {
  constexpr std::array<std::string_view, 8> kAllowed = {
      "libc.so.6", "libm.so.6", "libpthread.so.0", "librt.so.1",
      "libdl.so.2", "libstdc++.so.6", "libgcc_s.so.1",
      "ld-linux-x86-64.so.2"};
  for (const std::string& soname : sonames) {
    require(std::find(kAllowed.begin(), kAllowed.end(), std::string_view(soname)) !=
                kAllowed.end(),
            "readelf NEEDED SONAME is outside fixed static-cudart allowlist: " + soname);
  }
}

std::string needed_sonames_sha256(const std::vector<std::string>& sonames) {
  std::ostringstream descriptor;
  descriptor << "safe-c1-g3-needed-sonames-v1\n";
  for (const std::string& soname : sonames) descriptor << soname << '\n';
  return runner_sha256_text(descriptor.str());
}

void validate_link_command_manifest_or_fail(
    const fs::path& root, const fs::path& self,
    const std::string& expected_wrapper_sha256,
    const std::vector<std::uint8_t>& command_bytes) {
  const std::map<std::string, std::string> fields = parse_strict_admission_kv(
      text_from_bytes(command_bytes, "link command manifest"), "link command manifest");
  constexpr const char* required[] = {
      "schema", "compiler_path", "host_compiler_path", "translation_unit",
      "source_snapshot_path", "source_snapshot_descriptor_sha256",
      "final_binary_path", "language_standard", "gpu_arch", "link_mode",
      "translation_unit_count", "object_or_archive_inputs", "explicit_link_libraries",
      "cudart_linkage", "raw_legacy_gts_rp_calls_in_wrapper",
      "dynamic_loading_calls_in_controlled_closure", "host_elf_loader_enumeration",
      "compile_only_contract_sha256", "compile_witness_sha256",
      "compile_only_object_sha256", "compile_only_gate_role",
      "build_launcher_path", "build_launcher_sha256", "build_body_path", "build_body_sha256",
      "sanitized_build_environment"};
  require(fields.size() == std::size(required),
          "link command manifest has unrecognized/missing keys");
  for (const char* key : required) (void)admission_value(fields, key);
  const fs::path wrapper = root / "runner" / "g3_4k_pilot_wrapper.cu";
  const fs::path launcher =
      root / "tools" / "build_and_attest_g3_4k_single_tu.sh";
  const fs::path body =
      root / "tools" / ".build_and_attest_g3_4k_single_tu.body.sh";
  const fs::path snapshot = root / "preflight" / "sealed_source";
  const fs::path snapshot_wrapper =
      snapshot / "runner" / "g3_4k_pilot_wrapper.cu";
  require(admission_value(fields, "schema") == "safe-c1-g3-link-command-manifest-v3" &&
              admission_value(fields, "compiler_path") == "/usr/local/cuda-13.1/bin/nvcc" &&
              admission_value(fields, "host_compiler_path") == "/usr/bin/g++" &&
              admission_value(fields, "translation_unit") == snapshot_wrapper.string() &&
              admission_value(fields, "source_snapshot_path") == snapshot.string() &&
              admission_value(fields, "source_snapshot_descriptor_sha256") ==
                  source_closure_descriptor_sha256() &&
              admission_value(fields, "final_binary_path") == self.string() &&
              admission_value(fields, "language_standard") == "c++17" &&
              admission_value(fields, "link_mode") == "executable" &&
              admission_value(fields, "translation_unit_count") == "1" &&
              admission_value(fields, "object_or_archive_inputs") == "absent" &&
              admission_value(fields, "explicit_link_libraries") == "dl" &&
              admission_value(fields, "cudart_linkage") == "static" &&
              admission_value(fields, "raw_legacy_gts_rp_calls_in_wrapper") == "absent" &&
              admission_value(fields, "dynamic_loading_calls_in_controlled_closure") == "absent" &&
              admission_value(fields, "host_elf_loader_enumeration") ==
                  "dl_iterate_phdr_pre_native_only" &&
              admission_value(fields, "compile_only_gate_role") ==
                  "sealed_source_pre_link_compile_not_link_input" &&
              admission_value(fields, "build_launcher_path") == launcher.string() &&
              admission_value(fields, "build_body_path") == body.string() &&
              admission_value(fields, "sanitized_build_environment") ==
                  "env_i_private_body_trampoline_v2" &&
              valid_sm_arch(admission_value(fields, "gpu_arch")),
          "link command manifest violates fixed controlled single-TU contract");
  require_sha256_hex(admission_value(fields, "source_snapshot_descriptor_sha256"),
                     "link command manifest source-snapshot descriptor SHA-256");
  require_sha256_hex(admission_value(fields, "build_launcher_sha256"),
                     "link command manifest build-launcher SHA-256");
  require_sha256_hex(admission_value(fields, "build_body_sha256"),
                     "link command manifest build-body SHA-256");
  for (const char* key : {"compile_only_contract_sha256", "compile_witness_sha256",
                          "compile_only_object_sha256"}) {
    require_sha256_hex(admission_value(fields, key),
                       std::string("link command manifest ") + key);
  }
  require(sha256_bytes(read_regular_file_strict(launcher)) ==
              admission_value(fields, "build_launcher_sha256") &&
              sha256_bytes(read_regular_file_strict(body)) ==
                  admission_value(fields, "build_body_sha256"),
          "current controlled build launcher/body differs from admission");
  require(fs::weakly_canonical(snapshot) == snapshot && fs::is_directory(snapshot),
          "sealed source snapshot is not the fixed real preflight directory");
  require(sha256_bytes(read_regular_file_strict(snapshot_wrapper)) ==
              expected_wrapper_sha256,
          "sealed snapshot wrapper differs from admission-bound wrapper");
  for (const ExpectedFile& expected : kSourceClosure) {
    require(sha256_bytes(read_regular_file_strict(snapshot / expected.relative)) ==
                expected.sha256,
            std::string("sealed source snapshot SHA mismatch for ") + expected.relative);
  }

  const fs::path compile_root = root / "preflight" / "compile_wrapper_only";
  const fs::path compile_snapshot = compile_root / "sealed_source";
  const fs::path compile_snapshot_wrapper =
      compile_snapshot / "runner" / "g3_4k_pilot_wrapper.cu";
  const fs::path compile_contract = compile_root / "compile_only_contract.txt";
  const fs::path compile_command = compile_root / "compile_command.txt";
  const fs::path compile_witness = compile_root / "compile_witness.txt";
  const fs::path compile_object = compile_root / "g3_4k_pilot_wrapper.compute_80.o";
  const fs::path compile_log = compile_root / "nvcc_compile.log";
  const fs::path nvcc_version = compile_root / "nvcc_version.txt";
  const fs::path host_cxx_version = compile_root / "host_cxx_version.txt";
  require(fs::weakly_canonical(compile_root) == compile_root &&
              fs::is_directory(compile_root) &&
              fs::weakly_canonical(compile_snapshot) == compile_snapshot &&
              fs::is_directory(compile_snapshot),
          "compile witness root/snapshot is not a fixed real preflight directory");
  require(sha256_bytes(read_regular_file_strict(compile_snapshot_wrapper)) ==
              expected_wrapper_sha256,
          "compile snapshot wrapper differs from admission-bound wrapper");
  for (const ExpectedFile& expected : kSourceClosure) {
    require(sha256_bytes(read_regular_file_strict(compile_snapshot / expected.relative)) ==
                expected.sha256,
            std::string("compile snapshot SHA mismatch for ") + expected.relative);
  }
  const std::vector<std::uint8_t> contract_bytes =
      read_regular_file_strict(compile_contract);
  const std::vector<std::uint8_t> command_witness_bytes =
      read_regular_file_strict(compile_command);
  const std::vector<std::uint8_t> witness_bytes =
      read_regular_file_strict(compile_witness);
  const std::vector<std::uint8_t> object_bytes =
      read_regular_file_strict(compile_object);
  const std::vector<std::uint8_t> log_bytes =
      read_regular_file_strict(compile_log);
  const std::vector<std::uint8_t> nvcc_version_bytes =
      read_regular_file_strict(nvcc_version);
  const std::vector<std::uint8_t> host_cxx_version_bytes =
      read_regular_file_strict(host_cxx_version);
  require(sha256_bytes(contract_bytes) ==
              admission_value(fields, "compile_only_contract_sha256") &&
              sha256_bytes(witness_bytes) ==
                  admission_value(fields, "compile_witness_sha256") &&
              sha256_bytes(object_bytes) ==
                  admission_value(fields, "compile_only_object_sha256"),
          "actual compile witness artifacts differ from link command");
  const std::map<std::string, std::string> witness_fields =
      parse_strict_admission_kv(text_from_bytes(witness_bytes, "compile witness"),
                                "compile witness");
  constexpr const char* witness_required[] = {
      "schema", "single_translation_unit", "link_or_runtime", "witness_role",
      "sanitized_build_environment", "compiler_path", "host_compiler_path",
      "translation_unit", "source_snapshot_path",
      "source_snapshot_descriptor_sha256", "wrapper_sha256",
      "compile_command_sha256", "compile_object_sha256", "compile_log_sha256",
      "nvcc_version_sha256", "host_cxx_version_sha256", "compile_launcher_path",
      "compile_launcher_sha256", "compile_body_path", "compile_body_sha256"};
  require(witness_fields.size() == std::size(witness_required),
          "compile witness has unrecognized/missing keys");
  for (const char* key : witness_required) (void)admission_value(witness_fields, key);
  require(admission_value(witness_fields, "schema") ==
                  "safe-c1-g3-compile-witness-v1" &&
              admission_value(witness_fields, "single_translation_unit") == "true" &&
              admission_value(witness_fields, "link_or_runtime") == "false" &&
              admission_value(witness_fields, "witness_role") ==
                  "sealed_source_pre_link_compile_not_link_input" &&
              admission_value(witness_fields, "sanitized_build_environment") ==
                  "env_i_private_body_trampoline_v2" &&
              admission_value(witness_fields, "compiler_path") ==
                  "/usr/local/cuda-13.1/bin/nvcc" &&
              admission_value(witness_fields, "host_compiler_path") == "/usr/bin/g++" &&
              admission_value(witness_fields, "translation_unit") ==
                  (compile_root / "sealed_source" / "runner" /
                   "g3_4k_pilot_wrapper.cu").string() &&
              admission_value(witness_fields, "source_snapshot_path") ==
                  (compile_root / "sealed_source").string() &&
              admission_value(witness_fields, "source_snapshot_descriptor_sha256") ==
                  source_closure_descriptor_sha256() &&
              admission_value(witness_fields, "wrapper_sha256") ==
                  expected_wrapper_sha256 &&
              admission_value(witness_fields, "compile_object_sha256") ==
                  admission_value(fields, "compile_only_object_sha256") &&
              admission_value(witness_fields, "compile_log_sha256") ==
                  sha256_bytes(log_bytes) &&
              admission_value(witness_fields, "compile_launcher_path") ==
                  (root / "tools" / "compile_g3_4k_wrapper_only.sh").string() &&
              admission_value(witness_fields, "compile_body_path") ==
                  (root / "tools" / ".compile_g3_4k_wrapper_only.body.sh").string(),
          "compile witness violates the fixed pre-link source-closure contract");
  for (const char* key : {"compile_command_sha256", "compile_object_sha256",
                          "compile_log_sha256", "nvcc_version_sha256",
                          "host_cxx_version_sha256", "compile_launcher_sha256",
                          "compile_body_sha256"}) {
    require_sha256_hex(admission_value(witness_fields, key),
                       std::string("compile witness ") + key);
  }
  require(admission_value(witness_fields, "compile_command_sha256") ==
                  sha256_bytes(command_witness_bytes) &&
              admission_value(witness_fields, "nvcc_version_sha256") ==
                  sha256_bytes(nvcc_version_bytes) &&
              admission_value(witness_fields, "host_cxx_version_sha256") ==
                  sha256_bytes(host_cxx_version_bytes),
          "compile witness does not bind the actual command/version evidence");

  const fs::path compile_launcher =
      root / "tools" / "compile_g3_4k_wrapper_only.sh";
  const fs::path compile_body =
      root / "tools" / ".compile_g3_4k_wrapper_only.body.sh";
  const std::string compile_launcher_sha256 =
      sha256_bytes(read_regular_file_strict(compile_launcher));
  const std::string compile_body_sha256 =
      sha256_bytes(read_regular_file_strict(compile_body));
  require(admission_value(witness_fields, "compile_launcher_sha256") ==
                  compile_launcher_sha256 &&
              admission_value(witness_fields, "compile_body_sha256") ==
                  compile_body_sha256,
          "compile witness launcher/body bytes differ from current controlled scripts");

  const std::map<std::string, std::string> compile_command_fields =
      parse_strict_admission_kv(
          text_from_bytes(command_witness_bytes, "compile command manifest"),
          "compile command manifest");
  constexpr const char* compile_command_required[] = {
      "schema", "single_translation_unit", "link_or_runtime",
      "sanitized_build_environment", "compiler_path", "host_compiler_path",
      "translation_unit", "source_snapshot_path",
      "source_snapshot_descriptor_sha256", "wrapper_sha256", "language_standard",
      "gpu_arch", "gpu_code", "source_input_count", "object_or_archive_inputs",
      "include_order", "compile_launcher_path", "compile_launcher_sha256",
      "compile_body_path", "compile_body_sha256"};
  require(compile_command_fields.size() == std::size(compile_command_required),
          "compile command manifest has unrecognized/missing keys");
  for (const char* key : compile_command_required) {
    (void)admission_value(compile_command_fields, key);
  }
  require(admission_value(compile_command_fields, "schema") ==
                  "safe-c1-g3-compile-command-v1" &&
              admission_value(compile_command_fields, "single_translation_unit") == "true" &&
              admission_value(compile_command_fields, "link_or_runtime") == "false" &&
              admission_value(compile_command_fields, "sanitized_build_environment") ==
                  "env_i_private_body_trampoline_v2" &&
              admission_value(compile_command_fields, "compiler_path") ==
                  "/usr/local/cuda-13.1/bin/nvcc" &&
              admission_value(compile_command_fields, "host_compiler_path") == "/usr/bin/g++" &&
              admission_value(compile_command_fields, "translation_unit") ==
                  compile_snapshot_wrapper.string() &&
              admission_value(compile_command_fields, "source_snapshot_path") ==
                  compile_snapshot.string() &&
              admission_value(compile_command_fields, "source_snapshot_descriptor_sha256") ==
                  source_closure_descriptor_sha256() &&
              admission_value(compile_command_fields, "wrapper_sha256") ==
                  expected_wrapper_sha256 &&
              admission_value(compile_command_fields, "language_standard") == "c++17" &&
              admission_value(compile_command_fields, "gpu_arch") == "compute_80" &&
              admission_value(compile_command_fields, "gpu_code") == "compute_80" &&
              admission_value(compile_command_fields, "source_input_count") == "1" &&
              admission_value(compile_command_fields, "object_or_archive_inputs") == "absent" &&
              admission_value(compile_command_fields, "include_order") ==
                  (compile_snapshot / "src").string() + ":" +
                      (compile_snapshot / "reference" / "include").string() &&
              admission_value(compile_command_fields, "compile_launcher_path") ==
                  compile_launcher.string() &&
              admission_value(compile_command_fields, "compile_launcher_sha256") ==
                  compile_launcher_sha256 &&
              admission_value(compile_command_fields, "compile_body_path") ==
                  compile_body.string() &&
              admission_value(compile_command_fields, "compile_body_sha256") ==
                  compile_body_sha256,
          "compile command manifest violates the sealed pre-link compile contract");
  for (const char* key : {"source_snapshot_descriptor_sha256", "wrapper_sha256",
                          "compile_launcher_sha256", "compile_body_sha256"}) {
    require_sha256_hex(admission_value(compile_command_fields, key),
                       std::string("compile command manifest ") + key);
  }

  const std::map<std::string, std::string> contract_fields =
      parse_strict_admission_kv(text_from_bytes(contract_bytes, "compile-only contract"),
                                "compile-only contract");
  constexpr const char* contract_required[] = {
      "schema", "single_translation_unit", "link_or_runtime",
      "sanitized_build_environment", "compiler_path", "host_compiler_path",
      "translation_unit", "source_snapshot_path",
      "source_snapshot_descriptor_sha256", "wrapper_sha256", "source_input_count",
      "object_or_archive_inputs", "include_order", "compile_command_sha256",
      "compile_witness_sha256", "compile_object_sha256", "compile_launcher_path",
      "compile_launcher_sha256", "compile_body_path", "compile_body_sha256"};
  require(contract_fields.size() == std::size(contract_required),
          "compile-only contract has unrecognized/missing keys");
  for (const char* key : contract_required) (void)admission_value(contract_fields, key);
  require(admission_value(contract_fields, "schema") ==
                  "safe-c1-g3-compile-only-manifest-v6" &&
              admission_value(contract_fields, "single_translation_unit") == "true" &&
              admission_value(contract_fields, "link_or_runtime") == "false" &&
              admission_value(contract_fields, "sanitized_build_environment") ==
                  "env_i_private_body_trampoline_v2" &&
              admission_value(contract_fields, "compiler_path") ==
                  "/usr/local/cuda-13.1/bin/nvcc" &&
              admission_value(contract_fields, "host_compiler_path") == "/usr/bin/g++" &&
              admission_value(contract_fields, "translation_unit") ==
                  compile_snapshot_wrapper.string() &&
              admission_value(contract_fields, "source_snapshot_path") ==
                  compile_snapshot.string() &&
              admission_value(contract_fields, "source_snapshot_descriptor_sha256") ==
                  source_closure_descriptor_sha256() &&
              admission_value(contract_fields, "wrapper_sha256") == expected_wrapper_sha256 &&
              admission_value(contract_fields, "source_input_count") == "1" &&
              admission_value(contract_fields, "object_or_archive_inputs") == "absent" &&
              admission_value(contract_fields, "include_order") ==
                  (compile_snapshot / "src").string() + ":" +
                      (compile_snapshot / "reference" / "include").string() &&
              admission_value(contract_fields, "compile_command_sha256") ==
                  sha256_bytes(command_witness_bytes) &&
              admission_value(contract_fields, "compile_witness_sha256") ==
                  sha256_bytes(witness_bytes) &&
              admission_value(contract_fields, "compile_object_sha256") ==
                  sha256_bytes(object_bytes) &&
              admission_value(contract_fields, "compile_launcher_path") ==
                  compile_launcher.string() &&
              admission_value(contract_fields, "compile_launcher_sha256") ==
                  compile_launcher_sha256 &&
              admission_value(contract_fields, "compile_body_path") == compile_body.string() &&
              admission_value(contract_fields, "compile_body_sha256") == compile_body_sha256,
          "compile-only contract does not bind the sealed command/witness/object closure");
  for (const char* key : {"source_snapshot_descriptor_sha256", "wrapper_sha256",
                          "compile_command_sha256", "compile_witness_sha256",
                          "compile_object_sha256", "compile_launcher_sha256",
                          "compile_body_sha256"}) {
    require_sha256_hex(admission_value(contract_fields, key),
                       std::string("compile-only contract ") + key);
  }
}

struct LaunchAdmission final {
  std::string admission_sha256;
  std::string admission_basename;
  std::string binary_sha256;
  std::string wrapper_sha256;
  std::string source_closure_descriptor_sha256;
  std::string link_image_report_sha256;
  std::string link_command_sha256;
  std::string pre_native_host_loader_descriptor_sha256;
  std::string loader_snapshot_scope;
  std::string run_launcher_sha256;
  std::string run_body_sha256;
  std::string nvml_snapshot_helper_sha256;
  std::string nvml_snapshot_schema;
  std::string nvml_python_realpath;
  std::string nvml_python_sha256;
  std::string nvml_library_realpath;
  std::string nvml_library_sha256;
};

LaunchAdmission validate_launch_admission_or_fail(
    const fs::path& root, const fs::path& supplied_admission) {
  require_clean_run_environment_or_fail();
  const fs::path admission_root = root / "preflight" / "admissions";
  const fs::path canonical_admission_root = fs::weakly_canonical(admission_root);
  require(canonical_admission_root == admission_root &&
              fs::is_directory(canonical_admission_root),
          "admission root must be fixed real v6 preflight/admissions directory");
  const fs::file_status supplied_status = fs::symlink_status(supplied_admission);
  require(!fs::is_symlink(supplied_status), "admission path must not be a symlink");
  const fs::path admission = fs::weakly_canonical(supplied_admission);
  require(admission.parent_path() == admission_root && admission.extension() == ".admission",
          "admission must be a fixed real child of v6 preflight/admissions");
  const std::vector<std::uint8_t> admission_bytes = read_regular_file_strict(admission);
  const std::map<std::string, std::string> fields = parse_strict_admission_kv(
      text_from_bytes(admission_bytes, "launch admission"), "launch admission");
  constexpr const char* required[] = {
      "schema", "binary_path", "binary_sha256", "wrapper_sha256",
      "source_closure_descriptor_sha256", "single_translation_unit",
      "wrapper_raw_legacy_gts_rp_calls_absent", "link_image_report_sha256",
      "link_command_sha256", "compile_witness_sha256", "compile_only_object_sha256",
      "pre_native_host_loader_descriptor_sha256", "loader_snapshot_scope",
      "sanitized_exec_environment", "run_launcher_path", "run_launcher_sha256",
      "run_body_path", "run_body_sha256", "nvml_snapshot_helper_path",
      "nvml_snapshot_helper_sha256", "nvml_snapshot_schema",
      "nvml_python_realpath", "nvml_python_sha256", "nvml_library_realpath",
      "nvml_library_sha256"};
  require(fields.size() == std::size(required),
          "launch admission has unrecognized/missing keys");
  for (const char* key : required) (void)admission_value(fields, key);
  require(admission_value(fields, "schema") == "safe-c1-g3-single-tu-admission-v4" &&
              admission_value(fields, "single_translation_unit") == "true" &&
              admission_value(fields, "wrapper_raw_legacy_gts_rp_calls_absent") == "true" &&
              admission_value(fields, "loader_snapshot_scope") ==
                  "pre_native_host_elf_only" &&
              admission_value(fields, "sanitized_exec_environment") ==
                  "env_i_path_home_cuda_uuid_v2",
          "launch admission schema/scope/sanitized-launch drift");
  require(admission_value(fields, "source_closure_descriptor_sha256") ==
              source_closure_descriptor_sha256(),
          "launch admission source-closure descriptor drift");
  for (const char* key : {"binary_sha256", "wrapper_sha256", "link_image_report_sha256",
                          "link_command_sha256", "compile_witness_sha256",
                          "compile_only_object_sha256",
                          "pre_native_host_loader_descriptor_sha256",
                          "run_launcher_sha256", "run_body_sha256",
                          "nvml_snapshot_helper_sha256", "nvml_python_sha256",
                          "nvml_library_sha256"}) {
    require_sha256_hex(admission_value(fields, key), std::string("admission ") + key);
  }

  const fs::path self = fs::weakly_canonical(fs::path(current_executable_path_or_fail()));
  const fs::path binary_root = root / "preflight" / "bin";
  require(fs::weakly_canonical(binary_root) == binary_root &&
              fs::is_directory(binary_root) && self.parent_path() == binary_root &&
              self == fs::path(admission_value(fields, "binary_path")),
          "executing binary does not equal admission-bound v6 preflight/bin path");
  require(sha256_bytes(read_regular_file_strict(self)) ==
              admission_value(fields, "binary_sha256"),
          "executing binary SHA-256 differs from admission");

  const fs::path wrapper = root / "runner" / "g3_4k_pilot_wrapper.cu";
  const fs::path run_launcher = root / "tools" / "run_g3_4k_single_tu.sh";
  const fs::path run_body = root / "tools" / ".run_g3_4k_single_tu.body.sh";
  const fs::path nvml_snapshot_helper = root / "tools" / ".g3_nvml_snapshot.py";
  require(admission_value(fields, "run_launcher_path") == run_launcher.string() &&
              admission_value(fields, "run_body_path") == run_body.string() &&
              admission_value(fields, "nvml_snapshot_helper_path") ==
                  nvml_snapshot_helper.string() &&
              admission_value(fields, "nvml_snapshot_schema") ==
                  "safe-c1-g3-nvml-snapshot-v1" &&
              admission_value(fields, "nvml_python_realpath") == "/usr/bin/python3.12" &&
              admission_value(fields, "nvml_library_realpath") ==
                  "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.590.48.01" &&
              sha256_bytes(read_regular_file_strict(run_launcher)) ==
                  admission_value(fields, "run_launcher_sha256") &&
              sha256_bytes(read_regular_file_strict(run_body)) ==
                  admission_value(fields, "run_body_sha256") &&
              sha256_bytes(read_regular_file_strict(nvml_snapshot_helper)) ==
                  admission_value(fields, "nvml_snapshot_helper_sha256"),
          "current controlled run launcher/body/NVML helper differs from admission");
  const std::vector<std::uint8_t> wrapper_bytes = read_regular_file_strict(wrapper);
  require_dynamic_loader_calls_absent(
      text_from_bytes(wrapper_bytes, "current wrapper"), "current wrapper");
  require(sha256_bytes(wrapper_bytes) == admission_value(fields, "wrapper_sha256"),
          "current wrapper SHA-256 differs from admission");

  const fs::path report = fs::path(admission.string() + ".link-image.txt");
  const fs::path command = fs::path(admission.string() + ".link-command.txt");
  const fs::path descriptor =
      fs::path(admission.string() + ".pre-native-host-loader.txt");
  const fs::path toolchain = fs::path(admission.string() + ".toolchain.txt");
  require(report.parent_path() == admission_root &&
              command.parent_path() == admission_root &&
              descriptor.parent_path() == admission_root &&
              toolchain.parent_path() == admission_root,
          "derived admission artifacts escape admission directory");
  const std::vector<std::uint8_t> report_bytes = read_regular_file_strict(report);
  const std::vector<std::uint8_t> command_bytes = read_regular_file_strict(command);
  const std::vector<std::uint8_t> descriptor_bytes =
      read_regular_file_strict(descriptor);
  const std::vector<std::uint8_t> toolchain_bytes = read_regular_file_strict(toolchain);
  require(sha256_bytes(report_bytes) == admission_value(fields, "link_image_report_sha256") &&
              sha256_bytes(command_bytes) ==
                  admission_value(fields, "link_command_sha256") &&
              sha256_bytes(descriptor_bytes) ==
                  admission_value(fields, "pre_native_host_loader_descriptor_sha256"),
          "admission-bound link/command/pre-native evidence SHA differs");
  require(text_from_bytes(descriptor_bytes, "pre-native host loader descriptor").rfind(
              "schema=safe-c1-g3-pre-native-host-loader-image-v2\n", 0) == 0,
          "pre-native host loader descriptor schema drift");
  require(text_from_bytes(toolchain_bytes, "toolchain report").rfind(
              "schema=safe-c1-g3-toolchain-v1\n", 0) == 0,
          "toolchain report schema drift");
  validate_link_command_manifest_or_fail(
      root, self, admission_value(fields, "wrapper_sha256"), command_bytes);

  const std::map<std::string, std::string> report_fields = parse_strict_admission_kv(
      text_from_bytes(report_bytes, "link-image report"), "link-image report");
  constexpr const char* report_required[] = {
      "schema", "single_translation_unit", "raw_legacy_gts_rp_calls_in_wrapper",
      "dynamic_loading_calls_in_controlled_closure", "host_elf_loader_enumeration", "legacy_gts_dso",
      "elf_rpath_or_runpath", "needed_allowlist", "needed_sonames_sha256",
      "binary_sha256", "wrapper_sha256", "source_closure_descriptor_sha256",
      "link_command_sha256", "readelf_dynamic_sha256", "ldd_sha256",
      "toolchain_report_sha256", "compile_witness_sha256",
      "compile_only_object_sha256"};
  require(report_fields.size() == std::size(report_required),
          "link-image report has unrecognized/missing keys");
  for (const char* key : report_required) (void)admission_value(report_fields, key);
  require(admission_value(report_fields, "schema") ==
                  "safe-c1-g3-link-image-report-v4" &&
              admission_value(report_fields, "single_translation_unit") == "true" &&
              admission_value(report_fields, "raw_legacy_gts_rp_calls_in_wrapper") ==
                  "absent" &&
              admission_value(report_fields, "dynamic_loading_calls_in_controlled_closure") ==
                  "absent" &&
              admission_value(report_fields, "host_elf_loader_enumeration") ==
                  "dl_iterate_phdr_pre_native_only" &&
              admission_value(report_fields, "legacy_gts_dso") == "absent" &&
              admission_value(report_fields, "elf_rpath_or_runpath") == "absent" &&
              admission_value(report_fields, "needed_allowlist") ==
                  "system_cxx_static_cudart_glibc_loader_v2" &&
              admission_value(report_fields, "binary_sha256") ==
                  admission_value(fields, "binary_sha256") &&
              admission_value(report_fields, "wrapper_sha256") ==
                  admission_value(fields, "wrapper_sha256") &&
              admission_value(report_fields, "source_closure_descriptor_sha256") ==
                  admission_value(fields, "source_closure_descriptor_sha256") &&
              admission_value(report_fields, "link_command_sha256") ==
                  admission_value(fields, "link_command_sha256") &&
              admission_value(report_fields, "compile_witness_sha256") ==
                  admission_value(fields, "compile_witness_sha256") &&
              admission_value(report_fields, "compile_only_object_sha256") ==
                  admission_value(fields, "compile_only_object_sha256"),
          "link-image report does not bind binary/source/command/compile witness");
  for (const char* key : {"needed_sonames_sha256", "readelf_dynamic_sha256",
                          "ldd_sha256", "toolchain_report_sha256",
                          "compile_witness_sha256", "compile_only_object_sha256"}) {
    require_sha256_hex(admission_value(report_fields, key),
                       std::string("link-image report ") + key);
  }
  require(sha256_bytes(toolchain_bytes) ==
              admission_value(report_fields, "toolchain_report_sha256"),
          "toolchain report SHA differs from link-image report");

  const fs::path readelf_report = fs::path(report.string() + ".readelf-dynamic.txt");
  const fs::path ldd_report = fs::path(report.string() + ".ldd.txt");
  require(readelf_report.parent_path() == admission_root &&
              ldd_report.parent_path() == admission_root,
          "dynamic-link report paths escape admission directory");
  const std::vector<std::uint8_t> readelf_bytes =
      read_regular_file_strict(readelf_report);
  const std::vector<std::uint8_t> ldd_bytes = read_regular_file_strict(ldd_report);
  require(sha256_bytes(readelf_bytes) ==
              admission_value(report_fields, "readelf_dynamic_sha256") &&
              sha256_bytes(ldd_bytes) == admission_value(report_fields, "ldd_sha256"),
          "actual readelf/ldd reports differ from admission evidence");
  const std::string readelf_text =
      text_from_bytes(readelf_bytes, "readelf dynamic report");
  const std::string ldd_text = text_from_bytes(ldd_bytes, "ldd report");
  const std::string readelf_lower = lowercase_ascii(readelf_text);
  const std::string ldd_lower = lowercase_ascii(ldd_text);
  require(readelf_text.find("Dynamic section") != std::string::npos &&
              readelf_lower.find("(rpath)") == std::string::npos &&
              readelf_lower.find("(runpath)") == std::string::npos &&
              readelf_lower.find("libgts") == std::string::npos &&
              ldd_lower.find("libgts") == std::string::npos &&
              ldd_lower.find("not found") == std::string::npos,
          "dynamic-link evidence has RPATH/RUNPATH, legacy GTS, or unresolved DSO");
  const std::vector<std::string> needed =
      parse_dynamic_needed_sonames_or_fail(readelf_text);
  require_needed_allowlist_or_fail(needed);
  require(needed_sonames_sha256(needed) ==
              admission_value(report_fields, "needed_sonames_sha256"),
          "readelf NEEDED set differs from admission-bound allowlist evidence");

  return LaunchAdmission{
      sha256_bytes(admission_bytes), admission.filename().string(),
      admission_value(fields, "binary_sha256"),
      admission_value(fields, "wrapper_sha256"),
      admission_value(fields, "source_closure_descriptor_sha256"),
      admission_value(fields, "link_image_report_sha256"),
      admission_value(fields, "link_command_sha256"),
      admission_value(fields, "pre_native_host_loader_descriptor_sha256"),
      admission_value(fields, "loader_snapshot_scope"),
      admission_value(fields, "run_launcher_sha256"),
      admission_value(fields, "run_body_sha256"),
      admission_value(fields, "nvml_snapshot_helper_sha256"),
      admission_value(fields, "nvml_snapshot_schema"),
      admission_value(fields, "nvml_python_realpath"),
      admission_value(fields, "nvml_python_sha256"),
      admission_value(fields, "nvml_library_realpath"),
      admission_value(fields, "nvml_library_sha256")};
}

void validate_pre_native_host_loader_snapshot_or_fail(
    const LaunchAdmission& admission) {
  require_clean_run_environment_or_fail();
  const PreNativeHostLoaderDescriptor actual =
      pre_native_host_loader_descriptor_or_fail();
  require(actual.sha256 == admission.pre_native_host_loader_descriptor_sha256,
          "actual pre-native host ELF image differs from admission descriptor");
}

void require_root_private_directory_or_fail(const fs::path& path,
                                            const std::string& label) {
  const fs::file_status status = fs::symlink_status(path);
  require(!fs::is_symlink(status) && fs::is_directory(status) &&
              fs::weakly_canonical(path) == path,
          label + " is not a fixed real directory");
  struct stat detail {};
  require(::stat(path.c_str(), &detail) == 0 && detail.st_uid == 0 &&
              detail.st_gid == 0 && (detail.st_mode & 0777) == 0700,
          label + " is not root-private 0700");
}

void require_root_private_file_or_fail(const fs::path& path,
                                       const std::string& label) {
  const fs::file_status status = fs::symlink_status(path);
  require(!fs::is_symlink(status), label + " must not be a symlink");
  struct stat detail {};
  require(::lstat(path.c_str(), &detail) == 0 && S_ISREG(detail.st_mode) &&
              detail.st_uid == 0 && detail.st_gid == 0 &&
              (detail.st_mode & 0777) == 0600 && detail.st_nlink == 1,
          label + " is not a root-private single-link 0600 regular file");
}

bool lower_hex(const std::string& value, std::size_t exact_size) {
  return value.size() == exact_size &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return (character >= static_cast<unsigned char>('0') &&
                   character <= static_cast<unsigned char>('9')) ||
                  (character >= static_cast<unsigned char>('a') &&
                   character <= static_cast<unsigned char>('f'));
         });
}

bool canonical_decimal(const std::string& value) {
  return !value.empty() &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return character >= static_cast<unsigned char>('0') &&
                  character <= static_cast<unsigned char>('9');
         }) &&
         (value.size() == 1 || value.front() != '0');
}

bool canonical_pci_bus_id(const std::string& value) {
  const std::size_t domain_width =
      value.size() == 12 ? 4 : (value.size() == 16 ? 8 : 0);
  if (domain_width == 0 || value[domain_width] != ':' ||
      value[domain_width + 3] != ':' || value[domain_width + 6] != '.') {
    return false;
  }
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (index == domain_width || index == domain_width + 3 ||
        index == domain_width + 6) {
      continue;
    }
    const char character = value[index];
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f') ||
          (character >= 'A' && character <= 'F'))) {
      return false;
    }
  }
  return true;
}

std::uint64_t parse_u64_token_field_or_fail(const std::string& value,
                                            const std::string& label) {
  require(canonical_decimal(value), label + " is not canonical unsigned decimal");
  std::uint64_t result = 0;
  for (char character : value) {
    const std::uint64_t digit = static_cast<std::uint64_t>(character - '0');
    require(result <= (std::numeric_limits<std::uint64_t>::max() - digit) / 10ULL,
            label + " overflows uint64");
    result = result * 10ULL + digit;
  }
  return result;
}

std::string read_single_line_system_file_or_fail(const fs::path& path,
                                                 const std::string& label) {
  std::string value = text_from_bytes(read_regular_file_strict(path), label);
  require(!value.empty() && value.find('\r') == std::string::npos,
          label + " is empty or CRLF");
  if (value.back() == '\n') value.pop_back();
  require(!value.empty() && value.find_first_of(" \t\n") == std::string::npos,
          label + " has whitespace");
  return value;
}

std::string current_host_name_or_fail() {
  std::array<char, 256> host{};
  require(::gethostname(host.data(), host.size() - 1) == 0 && host[0] != '\0',
          "cannot determine controlled host name");
  const std::string value(host.data());
  require(value.find_first_of(" \t\r\n=") == std::string::npos,
          "controlled host name is unsafe");
  return value;
}

std::uint64_t current_boottime_ns_or_fail() {
  struct timespec now {};
  require(::clock_gettime(CLOCK_BOOTTIME, &now) == 0 && now.tv_sec >= 0 &&
              now.tv_nsec >= 0 && now.tv_nsec < 1000000000L,
          "cannot read CLOCK_BOOTTIME");
  const std::uint64_t seconds = static_cast<std::uint64_t>(now.tv_sec);
  require(seconds <=
              (std::numeric_limits<std::uint64_t>::max() -
               static_cast<std::uint64_t>(now.tv_nsec)) /
                  1000000000ULL,
          "CLOCK_BOOTTIME nanoseconds overflow");
  return seconds * 1000000000ULL + static_cast<std::uint64_t>(now.tv_nsec);
}

void rename_noreplace_or_fail(const fs::path& source, const fs::path& destination,
                              const std::string& label) {
#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1U << 0)
#endif
  const long result = ::syscall(SYS_renameat2, AT_FDCWD, source.c_str(), AT_FDCWD,
                                destination.c_str(), RENAME_NOREPLACE);
  require(result == 0, label + " cannot be consumed without replacement: " +
                           std::strerror(errno));
}

void fsync_directory_or_fail(const fs::path& path, const std::string& label) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    runner_fail(label + " cannot be opened for fsync: " + std::strerror(errno));
  }
  const int fsync_result = ::fsync(fd);
  const int close_result = ::close(fd);
  require(fsync_result == 0 && close_result == 0, label + " fsync/close failed");
}

struct GpuLaunchAuthorization final {
  std::string token_sha256;
  std::string token_nonce;
  std::string gpu_uuid;
  std::string gpu_ordinal;
  std::string gpu_pci_bus_id;
  std::string host_boot_id;
  std::uint64_t expires_boottime_ns = 0;
  std::string prelaunch_idle_check_sha256;
  std::string nvml_snapshot_helper_sha256;
  std::string nvml_python_realpath;
  std::string nvml_python_sha256;
  std::string nvml_library_realpath;
  std::string nvml_library_sha256;
  std::string nvml_driver_version;
  std::string consumed_token_basename;
};

GpuLaunchAuthorization consume_gpu_launch_authorization_or_fail(
    const fs::path& root, const LaunchArguments& launch_args,
    const LaunchAdmission& admission) {
  const fs::path authorization_root = root / "preflight" / "gpu_authorizations";
  const fs::path pending = authorization_root / "pending";
  const fs::path consumed = authorization_root / "consumed";
  const fs::path token = pending / "g3_4k_pilot.token";
  const fs::path proof = pending / "g3_4k_pilot.prelaunch";
  require_root_private_directory_or_fail(authorization_root, "GPU authorization root");
  require_root_private_directory_or_fail(pending, "GPU authorization pending directory");
  require_root_private_directory_or_fail(consumed, "GPU authorization consumed directory");
  require_root_private_file_or_fail(token, "GPU launch token");
  require_root_private_file_or_fail(proof, "GPU prelaunch proof");

  const std::vector<std::uint8_t> token_bytes = read_regular_file_strict(token);
  const std::vector<std::uint8_t> proof_bytes = read_regular_file_strict(proof);
  const std::string token_sha256 = sha256_bytes(token_bytes);
  const std::map<std::string, std::string> token_fields =
      parse_strict_admission_kv(text_from_bytes(token_bytes, "GPU launch token"),
                                "GPU launch token");
  constexpr const char* token_required[] = {
      "schema", "token_nonce", "release_root", "host_name", "host_boot_id",
      "issued_boottime_ns", "expires_boottime_ns", "gpu_ordinal", "gpu_uuid",
      "gpu_pci_bus_id", "run_name", "run_dir", "admission_path",
      "admission_sha256", "binary_path", "binary_sha256", "wrapper_sha256",
      "source_closure_descriptor_sha256", "run_launcher_sha256",
      "run_body_sha256", "pre_native_host_loader_descriptor_sha256",
      "fixture_manifest_sha256", "bootstrap_oracle_sha256", "idle_sample_count",
      "idle_sample_interval_seconds", "idle_check_sha256", "issuer_approval_id",
      "issuer_script_sha256", "nvml_snapshot_schema", "nvml_snapshot_helper_path",
      "nvml_snapshot_helper_sha256", "nvml_python_realpath", "nvml_python_sha256",
      "issuer_nvml_library_realpath", "issuer_nvml_library_sha256",
      "issuer_nvml_driver_version", "purpose"};
  require(token_fields.size() == std::size(token_required),
          "GPU launch token has unrecognized/missing fields");
  for (const char* key : token_required) (void)admission_value(token_fields, key);

  const std::string current_gpu_uuid = require_clean_run_environment_or_fail();
  const fs::path self = fs::weakly_canonical(fs::path(current_executable_path_or_fail()));
  const fs::path run_launcher = root / "tools" / "run_g3_4k_single_tu.sh";
  const fs::path run_body = root / "tools" / ".run_g3_4k_single_tu.body.sh";
  const fs::path issuer = root / "tools" / "issue_g3_4k_gpu_token.sh";
  const fs::path nvml_snapshot_helper = root / "tools" / ".g3_nvml_snapshot.py";
  require(admission_value(token_fields, "schema") == "safe-c1-g3-gpu-launch-token-v2" &&
              admission_value(token_fields, "release_root") == root.string() &&
              admission_value(token_fields, "host_name") == current_host_name_or_fail() &&
              admission_value(token_fields, "host_boot_id") ==
                  read_single_line_system_file_or_fail(
                      "/proc/sys/kernel/random/boot_id", "kernel boot ID") &&
              admission_value(token_fields, "run_dir") == launch_args.final_run_dir.string() &&
              admission_value(token_fields, "run_name") ==
                  launch_args.final_run_dir.filename().string() &&
              admission_value(token_fields, "admission_path") ==
                  (root / "preflight" / "admissions" /
                   "g3_4k_pilot_wrapper.admission").string() &&
              admission_value(token_fields, "admission_sha256") ==
                  admission.admission_sha256 &&
              admission_value(token_fields, "binary_path") == self.string() &&
              admission_value(token_fields, "binary_sha256") == admission.binary_sha256 &&
              admission_value(token_fields, "wrapper_sha256") == admission.wrapper_sha256 &&
              admission_value(token_fields, "source_closure_descriptor_sha256") ==
                  admission.source_closure_descriptor_sha256 &&
              admission_value(token_fields, "run_launcher_sha256") ==
                  admission.run_launcher_sha256 &&
              admission_value(token_fields, "run_body_sha256") == admission.run_body_sha256 &&
              admission_value(token_fields, "nvml_snapshot_schema") ==
                  admission.nvml_snapshot_schema &&
              admission_value(token_fields, "nvml_snapshot_helper_path") ==
                  nvml_snapshot_helper.string() &&
              admission_value(token_fields, "nvml_snapshot_helper_sha256") ==
                  admission.nvml_snapshot_helper_sha256 &&
              admission_value(token_fields, "nvml_python_realpath") ==
                  admission.nvml_python_realpath &&
              admission_value(token_fields, "nvml_python_sha256") ==
                  admission.nvml_python_sha256 &&
              admission_value(token_fields, "issuer_nvml_library_realpath") ==
                  admission.nvml_library_realpath &&
              admission_value(token_fields, "issuer_nvml_library_sha256") ==
                  admission.nvml_library_sha256 &&
              admission_value(token_fields, "pre_native_host_loader_descriptor_sha256") ==
                  admission.pre_native_host_loader_descriptor_sha256 &&
              admission_value(token_fields, "fixture_manifest_sha256") ==
                  kFixtureFiles[0].sha256 &&
              admission_value(token_fields, "bootstrap_oracle_sha256") ==
                  kBootstrapFiles[0].sha256 &&
              admission_value(token_fields, "idle_sample_count") == "3" &&
              admission_value(token_fields, "idle_sample_interval_seconds") == "2" &&
              admission_value(token_fields, "purpose") ==
                  "safe_c1_g3_4k_correctness_pilot" &&
              sha256_bytes(read_regular_file_strict(run_launcher)) ==
                  admission.run_launcher_sha256 &&
              sha256_bytes(read_regular_file_strict(run_body)) == admission.run_body_sha256 &&
              sha256_bytes(read_regular_file_strict(issuer)) ==
                  admission_value(token_fields, "issuer_script_sha256"),
          "GPU launch token does not bind this controlled run");

  const std::string nonce = admission_value(token_fields, "token_nonce");
  const std::string gpu_ordinal = admission_value(token_fields, "gpu_ordinal");
  const std::string gpu_uuid = admission_value(token_fields, "gpu_uuid");
  const std::string pci_bus = admission_value(token_fields, "gpu_pci_bus_id");
  require(lower_hex(nonce, 64) && canonical_decimal(gpu_ordinal) &&
              canonical_gpu_uuid(gpu_uuid) && canonical_pci_bus_id(pci_bus) &&
              current_gpu_uuid == gpu_uuid,
          "GPU launch token/device environment identity is malformed or mismatched");
  for (const char* key : {"admission_sha256", "binary_sha256", "wrapper_sha256",
                          "source_closure_descriptor_sha256", "run_launcher_sha256",
                          "run_body_sha256", "nvml_snapshot_helper_sha256",
                          "nvml_python_sha256", "issuer_nvml_library_sha256",
                          "pre_native_host_loader_descriptor_sha256",
                          "fixture_manifest_sha256", "bootstrap_oracle_sha256",
                          "idle_check_sha256", "issuer_script_sha256"}) {
    require_sha256_hex(admission_value(token_fields, key),
                       std::string("GPU launch token ") + key);
  }
  require(canonical_nvml_driver_version(
              admission_value(token_fields, "issuer_nvml_driver_version")),
          "GPU launch token NVML driver version is malformed");
  const std::uint64_t issued = parse_u64_token_field_or_fail(
      admission_value(token_fields, "issued_boottime_ns"), "token issued_boottime_ns");
  const std::uint64_t expires = parse_u64_token_field_or_fail(
      admission_value(token_fields, "expires_boottime_ns"), "token expires_boottime_ns");
  const std::uint64_t now = current_boottime_ns_or_fail();
  require(issued <= expires && expires - issued <= 120ULL * 1000000000ULL &&
              now >= issued && now <= expires,
          "GPU launch token is outside its short CLOCK_BOOTTIME validity window");

  const std::map<std::string, std::string> proof_fields =
      parse_strict_admission_kv(text_from_bytes(proof_bytes, "GPU prelaunch proof"),
                                "GPU prelaunch proof");
  constexpr const char* proof_required[] = {
      "schema", "token_sha256", "token_nonce", "host_name", "host_boot_id",
      "observed_boottime_ns", "gpu_ordinal", "gpu_uuid", "gpu_pci_bus_id",
      "run_name", "run_dir", "prelaunch_idle_check_sha256", "nvml_snapshot_schema",
      "nvml_snapshot_helper_path", "nvml_snapshot_helper_sha256",
      "prelaunch_nvml_python_realpath", "prelaunch_nvml_python_sha256",
      "prelaunch_nvml_library_realpath", "prelaunch_nvml_library_sha256",
      "prelaunch_nvml_driver_version", "scope"};
  require(proof_fields.size() == std::size(proof_required),
          "GPU prelaunch proof has unrecognized/missing fields");
  for (const char* key : proof_required) (void)admission_value(proof_fields, key);
  require(admission_value(proof_fields, "schema") == "safe-c1-g3-prelaunch-proof-v2" &&
              admission_value(proof_fields, "token_sha256") == token_sha256 &&
              admission_value(proof_fields, "token_nonce") == nonce &&
              admission_value(proof_fields, "host_name") == current_host_name_or_fail() &&
              admission_value(proof_fields, "host_boot_id") ==
                  admission_value(token_fields, "host_boot_id") &&
              admission_value(proof_fields, "gpu_ordinal") == gpu_ordinal &&
              admission_value(proof_fields, "gpu_uuid") == gpu_uuid &&
              admission_value(proof_fields, "gpu_pci_bus_id") == pci_bus &&
              admission_value(proof_fields, "run_name") ==
                  admission_value(token_fields, "run_name") &&
              admission_value(proof_fields, "run_dir") == launch_args.final_run_dir.string() &&
              admission_value(proof_fields, "nvml_snapshot_schema") ==
                  admission_value(token_fields, "nvml_snapshot_schema") &&
              admission_value(proof_fields, "nvml_snapshot_helper_path") ==
                  admission_value(token_fields, "nvml_snapshot_helper_path") &&
              admission_value(proof_fields, "nvml_snapshot_helper_sha256") ==
                  admission_value(token_fields, "nvml_snapshot_helper_sha256") &&
              admission_value(proof_fields, "prelaunch_nvml_python_realpath") ==
                  admission_value(token_fields, "nvml_python_realpath") &&
              admission_value(proof_fields, "prelaunch_nvml_python_sha256") ==
                  admission_value(token_fields, "nvml_python_sha256") &&
              admission_value(proof_fields, "prelaunch_nvml_library_realpath") ==
                  admission_value(token_fields, "issuer_nvml_library_realpath") &&
              admission_value(proof_fields, "prelaunch_nvml_library_sha256") ==
                  admission_value(token_fields, "issuer_nvml_library_sha256") &&
              admission_value(proof_fields, "prelaunch_nvml_driver_version") ==
                  admission_value(token_fields, "issuer_nvml_driver_version") &&
              canonical_nvml_driver_version(
                  admission_value(proof_fields, "prelaunch_nvml_driver_version")) &&
              admission_value(proof_fields, "scope") ==
                  "issuer_and_run_body_prelaunch_sampling_not_hardware_reservation",
          "GPU prelaunch proof does not bind this token/run");
  require_sha256_hex(admission_value(proof_fields, "prelaunch_idle_check_sha256"),
                     "GPU prelaunch proof idle sample SHA-256");
  const std::uint64_t observed = parse_u64_token_field_or_fail(
      admission_value(proof_fields, "observed_boottime_ns"),
      "prelaunch proof observed_boottime_ns");
  require(observed >= issued && observed <= expires && now >= observed,
          "GPU prelaunch proof is outside token validity interval");

  const fs::path consumed_token = consumed / (nonce + ".token");
  const fs::path consumed_proof = consumed / (nonce + ".prelaunch");
  require(!fs::exists(consumed_token) && !fs::exists(consumed_proof),
          "GPU launch token nonce was already consumed");
  rename_noreplace_or_fail(token, consumed_token, "GPU launch token");
  fsync_directory_or_fail(consumed, "GPU authorization consumed directory");
  fsync_directory_or_fail(pending, "GPU authorization pending directory");
  rename_noreplace_or_fail(proof, consumed_proof, "GPU prelaunch proof");
  fsync_directory_or_fail(consumed, "GPU authorization consumed directory");
  fsync_directory_or_fail(pending, "GPU authorization pending directory");
  return GpuLaunchAuthorization{
      token_sha256, nonce, gpu_uuid, gpu_ordinal, pci_bus,
      admission_value(token_fields, "host_boot_id"), expires,
      admission_value(proof_fields, "prelaunch_idle_check_sha256"),
      admission_value(token_fields, "nvml_snapshot_helper_sha256"),
      admission_value(token_fields, "nvml_python_realpath"),
      admission_value(token_fields, "nvml_python_sha256"),
      admission_value(token_fields, "issuer_nvml_library_realpath"),
      admission_value(token_fields, "issuer_nvml_library_sha256"),
      admission_value(token_fields, "issuer_nvml_driver_version"),
      consumed_token.filename().string()};
}

void validate_expected_buffer(const ExpectedFile& expected,
                              const std::vector<std::uint8_t>& bytes) {
  if (expected.exact_bytes != 0) {
    require(bytes.size() == expected.exact_bytes,
            std::string("unexpected byte length for ") + expected.relative);
  }
  require(sha256_bytes(bytes) == expected.sha256,
          std::string("SHA-256 mismatch for ") + expected.relative);
}

using PinnedFileSet = std::map<std::string, std::vector<std::uint8_t>>;

PinnedFileSet bind_expected_files_once(const fs::path& root, const ExpectedFile* files,
                                       std::size_t file_count, const std::string& label) {
  PinnedFileSet output;
  for (std::size_t i = 0; i < file_count; ++i) {
    const ExpectedFile& expected = files[i];
    const fs::path relative(expected.relative);
    require(!relative.empty() && relative.is_relative() && relative.lexically_normal() == relative,
            label + " has unsafe expected relative path");
    std::vector<std::uint8_t> bytes = read_regular_file_strict(root / relative);
    validate_expected_buffer(expected, bytes);
    require(output.emplace(expected.relative, std::move(bytes)).second,
            label + " has duplicate expected path");
  }
  return output;
}

const std::vector<std::uint8_t>& pinned_bytes(const PinnedFileSet& files,
                                              const char* relative) {
  const auto found = files.find(relative);
  require(found != files.end(), std::string("missing pinned input bytes: ") + relative);
  return found->second;
}

void require_controlled_source_closure_dynamic_loader_calls_absent(
    const PinnedFileSet& files) {
  for (const ExpectedFile& expected : kSourceClosure) {
    require_dynamic_loader_calls_absent(
        text_from_bytes(pinned_bytes(files, expected.relative), expected.relative),
        std::string("controlled source closure ") + expected.relative);
  }
}

void require_contains(const std::string& text, const std::string& token,
                      const std::string& label) {
  require(text.find(token) != std::string::npos,
          label + " lacks required token: " + token);
}

void validate_manifest_and_bootstrap_contract(const PinnedFileSet& fixture,
                                              const PinnedFileSet& bootstrap) {
  const std::string manifest = text_from_bytes(pinned_bytes(fixture, "manifest.json"),
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

  const std::string metadata = text_from_bytes(pinned_bytes(fixture, "metadata.json"),
                                               "metadata.json");
  require_contains(metadata, "\"base_n\": 4096", "metadata.json");
  require_contains(metadata, "\"dim\": 128", "metadata.json");
  require_contains(metadata, "\"event_count\": 21", "metadata.json");
  require_contains(metadata, "\"height\": 4", "metadata.json");
  require_contains(metadata, "\"metric\": \"L2\"", "metadata.json");
  require_contains(metadata, "\"oracle_distance\": \"int64 squared L2\"", "metadata.json");

  const std::string bootstrap_json =
      text_from_bytes(pinned_bytes(bootstrap, "bootstrap_oracle.json"),
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
      text_from_bytes(pinned_bytes(bootstrap, "bootstrap_oracle_validation.json"),
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

std::vector<std::int16_t> decode_i16_buffer(const std::vector<std::uint8_t>& bytes,
                                            std::size_t expected_values,
                                            const std::string& label) {
  require(bytes.size() == expected_values * sizeof(std::int16_t),
          "unexpected i16 input length: " + label);
  std::vector<std::int16_t> values(expected_values);
  for (std::size_t i = 0; i < expected_values; ++i) {
    values[i] = decode_i16_le(bytes, i * sizeof(std::int16_t));
  }
  return values;
}

std::vector<int> decode_i32_buffer_to_int(const std::vector<std::uint8_t>& bytes,
                                          std::size_t expected_values,
                                          const std::string& label) {
  require(bytes.size() == expected_values * sizeof(std::int32_t),
          "unexpected i32 input length: " + label);
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

FixtureData load_and_validate_binary_fixture(const PinnedFileSet& fixture) {
  FixtureData data;
  data.pool = decode_i16_buffer(pinned_bytes(fixture, "pool.i16"),
                                static_cast<std::size_t>(kPoolRows) * kDimension, "pool.i16");
  data.queries = decode_i16_buffer(pinned_bytes(fixture, "queries.i16"),
                                   static_cast<std::size_t>(kQueryRows) * kDimension, "queries.i16");
  data.stable_to_pool_row =
      decode_i32_buffer_to_int(pinned_bytes(fixture, "stable_id_to_pool_row.i32"), kPoolRows,
                               "stable_id_to_pool_row.i32");
  data.initial_base =
      decode_i32_buffer_to_int(pinned_bytes(fixture, "initial_base_stable_ids.i32"), kBaseCount,
                               "initial_base_stable_ids.i32");

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

std::vector<TraceOp> parse_and_validate_fixed_trace(const std::vector<std::uint8_t>& trace_bytes) {
  const std::string trace = text_from_bytes(trace_bytes, "trace.jsonl");
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
      // v6 runtime policy.  v6 validates it for input provenance only and
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
    const std::vector<std::uint8_t>& oracle_bytes, const std::vector<TraceOp>& trace) {
  const std::string oracle = text_from_bytes(oracle_bytes, "oracle_expected.jsonl");
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
  // certificate diagnostics are a host replay over captured frozen-tree bytes,
  // not an actual native query traversal receipt.  Direct remains closed, so a
  // true diagnostic certificate still produces a delta row.
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
      << ",\"certificate_scope\":\"frozen_snapshot_host_replay_no_native_query_membership\""
      << ",\"native_query_membership_checked\":false"
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

void require_snapshot_certificate_diagnostic_contract(const TraceOp& event,
                                                      const Placement& placement) {
  // These checks exercise a host replay over captured frozen native-tree bytes,
  // not a native vector-query traversal receipt and not the source-mirror
  // selection record.  They deliberately preserve the safe runtime policy:
  // every row remains delta while direct is closed, including the fixture's
  // historical capacity-labelled candidate.
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
              "frozen-snapshot certificate failed its fixed sibling-min witness contract");
      require(placement.certificate.levels.size() == std::size(expected_children),
              "frozen-snapshot certificate has wrong branch-witness depth");
      for (std::size_t i = 0; i < std::size(expected_children); ++i) {
        const auto& level = placement.certificate.levels[i];
        require(level.parent_leaf_or_internal == expected_parents[i] &&
                    level.child == expected_children[i] &&
                    !level.last_child &&
                    level.next_sibling == expected_children[i] + 1 &&
                    level.predicate_branch ==
                        safe_c1_g3::GtsKnnPruningBranch::kPredicateNonLastSibling,
                "frozen-snapshot certificate has wrong sibling-min branch path");
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
              "boundary/gap fixture did not fail through the precise frozen-snapshot certificate path");
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

RunPaths create_staging_directory_or_fail(const fs::path& final_dir) {
  const fs::path root = fs::weakly_canonical(fs::path(kReleaseRoot));
  require(root == fs::path(kReleaseRoot), "release root must not be symlinked/relocated");
  const fs::path runs_root = root / "runs";
  const fs::file_status runs_status = fs::symlink_status(runs_root);
  require(!fs::is_symlink(runs_status) && fs::is_directory(runs_status),
          "release runs directory must be a real directory");
  require(final_dir.parent_path() == runs_root,
          "run directory must be one new direct child of the fixed v6 runs directory");
  require(!final_dir.filename().empty() && final_dir.filename() != "." &&
              final_dir.filename() != "..",
          "run directory name is invalid");
  require(!fs::exists(final_dir), "final run directory already exists; no reuse/retry is allowed");
  // The isolated tree builder now throws on checked failures; hidden sibling
  // staging still ensures no intermediate byte can become a final run directory.
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
    if (argc == 2 && std::string_view(argv[1]) == "--emit-runtime-image-descriptor") {
      // This control-only mode remains before every Matrix/CUDA entry. It emits
      // only the scoped pre-native host-ELF descriptor; it builds no tree,
      // uploads no data, and does not touch a GPU context.
      g3_4k_pilot::require_clean_loader_environment_or_fail();
      std::cout << g3_4k_pilot::pre_native_host_loader_descriptor_or_fail().text;
      return 0;
    }
    // All strict byte/source/fixture/admission validation is done before the
    // Matrix constructor and before any CUDA/native API can be reached.
    const g3_4k_pilot::LaunchArguments launch_args =
        g3_4k_pilot::parse_launch_arguments_or_fail(argc, argv);
    const std::filesystem::path root = std::filesystem::weakly_canonical(std::filesystem::path(g3_4k_pilot::kReleaseRoot));
    g3_4k_pilot::require(root == std::filesystem::path(g3_4k_pilot::kReleaseRoot),
               "release root must have the fixed canonical absolute path");
    const g3_4k_pilot::PinnedFileSet source_audit_bytes =
        g3_4k_pilot::bind_expected_files_once(root, g3_4k_pilot::kSourceClosure,
                                               std::size(g3_4k_pilot::kSourceClosure), "source closure");
    g3_4k_pilot::require_controlled_source_closure_dynamic_loader_calls_absent(
        source_audit_bytes);
    const std::filesystem::path fixture_root = root / g3_4k_pilot::kFixtureRelative;
    const std::filesystem::path bootstrap_root = root / g3_4k_pilot::kBootstrapRelative;
    const g3_4k_pilot::PinnedFileSet fixture_bytes =
        g3_4k_pilot::bind_expected_files_once(fixture_root, g3_4k_pilot::kFixtureFiles,
                                               std::size(g3_4k_pilot::kFixtureFiles), "fixture");
    const g3_4k_pilot::PinnedFileSet bootstrap_bytes =
        g3_4k_pilot::bind_expected_files_once(bootstrap_root, g3_4k_pilot::kBootstrapFiles,
                                               std::size(g3_4k_pilot::kBootstrapFiles), "bootstrap");
    g3_4k_pilot::validate_manifest_and_bootstrap_contract(fixture_bytes, bootstrap_bytes);
    const g3_4k_pilot::FixtureData fixture_data =
        g3_4k_pilot::load_and_validate_binary_fixture(fixture_bytes);
    const std::vector<g3_4k_pilot::TraceOp> trace =
        g3_4k_pilot::parse_and_validate_fixed_trace(
            g3_4k_pilot::pinned_bytes(fixture_bytes, "trace.jsonl"));
    const std::map<int, g3_4k_pilot::OracleRecord> persisted_oracle =
        g3_4k_pilot::parse_and_validate_trace_oracle(
            g3_4k_pilot::pinned_bytes(fixture_bytes, "oracle_expected.jsonl"), trace);
    g3_4k_pilot::require(!safe_c1_g3::G3_DIRECT_SIDECAR_RUNTIME_GUARD_OPEN,
               "controlled pilot refuses any direct-sidecar runtime guard");
    g3_4k_pilot::require(!safe_c1_g3::G3_NATIVE_RANGE_RECEIPT_IMPLEMENTED &&
                   safe_c1_g3::G3_RANGE_BRANCH_ALIGNED_VECTOR_PREDICATE_MIRROR_IMPLEMENTED,
               "controlled range scope drifted from branch-mirror-only contract");
    g3_4k_pilot::require(safe_c1_g3::G3_LIGHTWEIGHT_GENERATION_IDENTITY_GUARD_IMPLEMENTED &&
                   !safe_c1_g3::G3_FULL_FROZEN_BASE_SNAPSHOT_ON_EACH_OPERATION &&
                   safe_c1_g3::G3_RUNNER_SINGLE_IMAGE_ADMISSION_REQUIRED,
               "controlled pilot requires lightweight identity guard plus single-image admission");

    const g3_4k_pilot::LaunchAdmission launch_admission =
        g3_4k_pilot::validate_launch_admission_or_fail(root, launch_args.admission_path);
    run_paths = g3_4k_pilot::create_staging_directory_or_fail(launch_args.final_run_dir);
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
    g3_4k_pilot::require(g3_4k_pilot::local_stable_set_sha(
                   std::vector<StableId>(active.begin(), active.end())) == g3_4k_pilot::kInitialLiveSha,
               "independent initial active digest drift");

    // This snapshot is deliberately taken after all host input/staging setup
    // and immediately before the first Matrix/CUDA entry. CUDA driver/JIT
    // images that may appear after this boundary are outside its stated scope.
    g3_4k_pilot::validate_pre_native_host_loader_snapshot_or_fail(launch_admission);
    const g3_4k_pilot::GpuLaunchAuthorization gpu_authorization =
        g3_4k_pilot::consume_gpu_launch_authorization_or_fail(
            root, launch_args, launch_admission);
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
        << "\"direct_runtime_guard_open\":false"
        << ",\"launch_admission_sha256\":\"" << launch_admission.admission_sha256 << "\""
        << ",\"launch_binary_sha256\":\"" << launch_admission.binary_sha256 << "\""
        << ",\"launch_admission_basename\":\"" << launch_admission.admission_basename << "\""
        << ",\"launch_wrapper_sha256\":\"" << launch_admission.wrapper_sha256 << "\""
        << ",\"launch_source_closure_descriptor_sha256\":\""
        << launch_admission.source_closure_descriptor_sha256 << "\""
        << ",\"launch_link_image_report_sha256\":\""
        << launch_admission.link_image_report_sha256 << "\""
        << ",\"launch_link_command_sha256\":\""
        << launch_admission.link_command_sha256 << "\""
        << ",\"launch_pre_native_host_loader_descriptor_sha256\":\""
        << launch_admission.pre_native_host_loader_descriptor_sha256 << "\""
        << ",\"loader_snapshot_scope\":\"" << launch_admission.loader_snapshot_scope << "\""
        << ",\"launch_cuda_visible_devices\":\""
        << std::getenv("CUDA_VISIBLE_DEVICES") << "\""
        << ",\"gpu_authorization_token_sha256\":\""
        << gpu_authorization.token_sha256 << "\""
        << ",\"gpu_authorization_nonce\":\"" << gpu_authorization.token_nonce << "\""
        << ",\"gpu_uuid\":\"" << gpu_authorization.gpu_uuid << "\""
        << ",\"gpu_ordinal_issuer_mapping\":\"" << gpu_authorization.gpu_ordinal << "\""
        << ",\"gpu_pci_bus_id_issuer_mapping\":\"" << gpu_authorization.gpu_pci_bus_id << "\""
        << ",\"gpu_authorization_expires_boottime_ns\":"
        << static_cast<unsigned long long>(gpu_authorization.expires_boottime_ns)
        << ",\"prelaunch_idle_check_sha256\":\""
        << gpu_authorization.prelaunch_idle_check_sha256 << "\""
        << ",\"nvml_snapshot_schema\":\"safe-c1-g3-nvml-snapshot-v1\""
        << ",\"nvml_snapshot_helper_sha256\":\""
        << gpu_authorization.nvml_snapshot_helper_sha256 << "\""
        << ",\"nvml_python_realpath\":\""
        << gpu_authorization.nvml_python_realpath << "\""
        << ",\"nvml_python_sha256\":\""
        << gpu_authorization.nvml_python_sha256 << "\""
        << ",\"nvml_library_realpath\":\""
        << gpu_authorization.nvml_library_realpath << "\""
        << ",\"nvml_library_sha256\":\""
        << gpu_authorization.nvml_library_sha256 << "\""
        << ",\"nvml_driver_version\":\""
        << gpu_authorization.nvml_driver_version << "\""
        << ",\"consumed_gpu_token_basename\":\""
        << gpu_authorization.consumed_token_basename << "\""
        << ",\"gpu_uuid_binding_scope\":\"issuer_prelaunch_ordinal_mapping_not_wrapper_runtime_requery\""
        << ",\"initial_rebuild\":"
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
        g3_4k_pilot::require_snapshot_certificate_diagnostic_contract(event, placement);
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
    // A C++ exception never publishes a final result; hidden staging bytes may
    // remain only for inspection if a future process-level failure bypasses this catch.
    if (!run_paths.staging_dir.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(run_paths.staging_dir, ignored);
    }
    std::cerr << error.what() << '\n';
    return 70;
  }
}
