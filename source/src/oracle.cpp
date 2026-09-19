// Independent unpruned CPU oracle: no TIDE bound or GPU source is reused.
#include <cstdint>
#include <algorithm>
extern "C" std::uint64_t p1_oracle(const std::uint64_t* fp,
    const std::uint64_t* ids, std::uint64_t n, const std::uint64_t* query,
    unsigned numerator, unsigned denominator, std::uint64_t* output) {
  std::uint64_t count = 0;
  for (std::uint64_t row=0; row<n; ++row) {
    if (ids[row] == query[0]) continue;
    unsigned intersection = 0, union_count = 0;
    for (int word=0; word<4; ++word) {
      intersection += __builtin_popcountll(fp[row*4+word] & query[word+1]);
      union_count += __builtin_popcountll(fp[row*4+word] | query[word+1]);
    }
    if (union_count && denominator*intersection >= numerator*union_count)
      output[count++] = ids[row];
  }
  std::sort(output, output+count);
  return count;
}
