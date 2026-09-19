// CPU-only inspector for C1 canonical result files.  It never initializes CUDA.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace {
constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void fnv_byte(std::uint64_t& h, std::uint8_t b) {
  h ^= b;
  h *= kFnvPrime;
}
void fnv_le32(std::uint64_t& h, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) fnv_byte(h, static_cast<std::uint8_t>((v >> (8 * i)) & 0xffU));
}
void fnv_le64(std::uint64_t& h, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) fnv_byte(h, static_cast<std::uint8_t>((v >> (8 * i)) & 0xffU));
}
std::uint32_t le32(const std::uint8_t* b) {
  return static_cast<std::uint32_t>(b[0]) |
         (static_cast<std::uint32_t>(b[1]) << 8U) |
         (static_cast<std::uint32_t>(b[2]) << 16U) |
         (static_cast<std::uint32_t>(b[3]) << 24U);
}
std::int32_t as_i32(std::uint32_t u) {
  std::int32_t result = 0;
  std::memcpy(&result, &u, sizeof(result));
  return result;
}
[[noreturn]] void fail(const std::string& message) {
  std::cerr << "canonical_inspect_v5: " << message << '\n';
  std::exit(2);
}
bool read_exact(std::ifstream& in, std::uint8_t* out, std::size_t n) {
  in.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(n));
  return static_cast<std::size_t>(in.gcount()) == n;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) fail("usage: canonical_inspect_v5 PATH");
  std::ifstream in(argv[1], std::ios::binary);
  if (!in) fail("cannot open artifact");
  std::uint64_t records = 0, pairs = 0;
  for (;;) {
    std::uint8_t header[12];
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    const std::streamsize got = in.gcount();
    if (got == 0 && in.eof()) break;
    if (got != static_cast<std::streamsize>(sizeof(header))) fail("truncated record header");
    const std::uint32_t ordinal = le32(header);
    const std::int32_t qid = as_i32(le32(header + 4));
    const std::uint32_t count = le32(header + 8);
    std::uint64_t h = kFnvOffset;
    fnv_le64(h, static_cast<std::uint64_t>(count));
    bool sorted = true, have_previous = false;
    std::int32_t previous_id = 0;
    std::uint32_t previous_bits = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
      std::uint8_t pair_bytes[8];
      if (!read_exact(in, pair_bytes, sizeof(pair_bytes))) fail("truncated pair payload");
      const std::int32_t id = as_i32(le32(pair_bytes));
      const std::uint32_t bits = le32(pair_bytes + 4);
      fnv_le32(h, static_cast<std::uint32_t>(id));
      fnv_le32(h, bits);
      if (have_previous && (id < previous_id || (id == previous_id && bits < previous_bits))) sorted = false;
      previous_id = id; previous_bits = bits; have_previous = true;
    }
    if (!sorted) fail("canonical pairs are not sorted");
    if (pairs > std::numeric_limits<std::uint64_t>::max() - count) fail("pair count overflow");
    pairs += count;
    std::cout << "REC\t" << ordinal << '\t' << qid << '\t' << count << '\t'
              << std::hex << std::setfill('0') << std::setw(16) << h << std::dec << "\t1\n";
    ++records;
  }
  std::cout << "SUMMARY\t" << records << '\t' << pairs << '\n';
  return 0;
}

