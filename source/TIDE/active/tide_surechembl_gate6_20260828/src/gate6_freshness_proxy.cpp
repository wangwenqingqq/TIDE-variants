#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int WORDS = 4;
constexpr int ROW_WORDS = WORDS + 2;

struct Row {
  std::uint64_t id;
  std::array<std::uint64_t, WORDS> fp;
  std::uint16_t popcount;
};

std::vector<std::uint64_t> read_words(const std::filesystem::path &path) {
  const auto bytes = std::filesystem::file_size(path);
  if (bytes % sizeof(std::uint64_t))
    throw std::runtime_error("invalid u64 file size: " + path.string());
  std::vector<std::uint64_t> values(bytes / sizeof(std::uint64_t));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char *>(values.data()),
             static_cast<std::streamsize>(bytes));
  if (!input) throw std::runtime_error("read failed: " + path.string());
  return values;
}

std::vector<Row> read_rows(const std::filesystem::path &path,
                           bool require_sorted) {
  const auto packed = read_words(path);
  if (packed.size() % ROW_WORDS)
    throw std::runtime_error("invalid interleaved rows: " + path.string());
  std::vector<Row> rows(packed.size() / ROW_WORDS);
  std::uint16_t previous = 0;
  for (std::size_t index = 0; index < rows.size(); ++index) {
    auto &row = rows[index];
    row.id = packed[index * ROW_WORDS];
    unsigned actual = 0;
    for (int word = 0; word < WORDS; ++word) {
      row.fp[word] = packed[index * ROW_WORDS + 1 + word];
      actual += __builtin_popcountll(row.fp[word]);
    }
    const auto stored = packed[index * ROW_WORDS + ROW_WORDS - 1];
    if (stored > WORDS * 64 || stored != actual)
      throw std::runtime_error("population-count mismatch: " + path.string());
    row.popcount = static_cast<std::uint16_t>(stored);
    if (require_sorted && index && row.popcount < previous)
      throw std::runtime_error("rows are not population-count sorted: " +
                               path.string());
    previous = row.popcount;
  }
  return rows;
}

std::array<std::uint64_t, WORDS * 64 + 2> cumulative_bins(
    const std::vector<Row> &rows) {
  std::array<std::uint64_t, WORDS * 64 + 2> cumulative{};
  for (const auto &row : rows) cumulative[row.popcount + 1]++;
  for (std::size_t index = 1; index < cumulative.size(); ++index)
    cumulative[index] += cumulative[index - 1];
  return cumulative;
}

std::pair<std::uint64_t, std::uint64_t> exact_bounds(
    const std::array<std::uint64_t, WORDS * 64 + 2> &bins,
    std::uint16_t query_popcount) {
  constexpr int numerator = 7;
  constexpr int denominator = 10;
  const int lower =
      (numerator * static_cast<int>(query_popcount) + denominator - 1) /
      denominator;
  const int upper = denominator * static_cast<int>(query_popcount) / numerator;
  return {bins[std::max(0, lower)], bins[std::min(WORDS * 64, upper) + 1]};
}

std::uint64_t gained_rows(const std::vector<Row> &delta,
                          const std::array<std::uint64_t, WORDS * 64 + 2> &bins,
                          const Row &query) {
  const auto [begin, end] = exact_bounds(bins, query.popcount);
  std::uint64_t count = 0;
  for (std::uint64_t index = begin; index < end; ++index) {
    const auto &candidate = delta[index];
    if (candidate.id == query.id) continue;
    unsigned intersection = 0;
    for (int word = 0; word < WORDS; ++word)
      intersection +=
          __builtin_popcountll(query.fp[word] & candidate.fp[word]);
    unsigned union_count = static_cast<unsigned>(query.popcount) +
                           static_cast<unsigned>(candidate.popcount) -
                           intersection;
    if (union_count == 0) union_count = 1;
    if (intersection * 10u >= union_count * 7u) count++;
  }
  return count;
}

double quantile(std::vector<std::uint64_t> values, double q) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double position = q * static_cast<double>(values.size() - 1);
  const auto low = static_cast<std::size_t>(position);
  const auto high = std::min(values.size() - 1, low + 1);
  const double fraction = position - static_cast<double>(low);
  return static_cast<double>(values[low]) * (1.0 - fraction) +
         static_cast<double>(values[high]) * fraction;
}

std::string json_escape(const std::string &value) {
  std::string output;
  for (char character : value) {
    if (character == '\\' || character == '"') output.push_back('\\');
    output.push_back(character);
  }
  return output;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: " << argv[0] << " ROOT OUTPUT_JSON\n";
      return 64;
    }
    const std::filesystem::path root = argv[1];
    const std::filesystem::path output_path = argv[2];
    const std::array<std::string, 6> transitions{{
        "2026-06-01_to_2026-06-15", "2026-06-15_to_2026-07-01",
        "2026-07-01_to_2026-07-17", "2026-07-17_to_2026-08-04",
        "2026-08-04_to_2026-08-18", "2026-08-18_to_2026-08-25"}};

    struct Summary {
      std::string transition;
      std::uint64_t delta_rows;
      std::uint64_t queries;
      std::uint64_t queries_with_gain;
      std::vector<std::uint64_t> hits;
      double seconds;
    };
    std::vector<Summary> summaries;

    if (!output_path.parent_path().empty())
      std::filesystem::create_directories(output_path.parent_path());
    std::ofstream samples(output_path.string() + ".samples.csv");
    samples << "transition,query_index,query_id,new_self_excluded_hits\n";

    for (const auto &transition : transitions) {
      const auto begin = std::chrono::steady_clock::now();
      const auto directory = root / "data" / "prepared" /
                             "transition_queries" / transition;
      const auto delta = read_rows(directory / "delta_u64x6.bin", true);
      const auto queries = read_rows(directory / "queries_u64x6.bin", false);
      if (queries.empty() || queries.size() > 512)
        throw std::runtime_error("invalid frozen query count for " + transition);
      const auto bins = cumulative_bins(delta);
      std::vector<std::uint64_t> hits(queries.size());

#pragma omp parallel for schedule(dynamic)
      for (std::int64_t index = 0;
           index < static_cast<std::int64_t>(queries.size()); ++index)
        hits[index] = gained_rows(delta, bins, queries[index]);

      std::uint64_t with_gain = 0;
      for (std::size_t index = 0; index < queries.size(); ++index) {
        with_gain += hits[index] > 0;
        samples << transition << ',' << index << ',' << queries[index].id << ','
                << hits[index] << '\n';
      }
      summaries.push_back(Summary{
          transition, delta.size(), queries.size(), with_gain, hits,
          std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
              .count()});
    }

    const int qualifying = static_cast<int>(std::count_if(
        summaries.begin(), summaries.end(), [](const Summary &summary) {
          return summary.queries_with_gain * 2 >= summary.queries;
        }));
    std::ofstream output(output_path);
    output << "{\n"
           << "  \"experiment_id\": \"tide_20260828_gate6_freshness_proxy\",\n"
           << "  \"semantics\": \"exact rational 7/10; insert-only delta hits; query self-match excluded\",\n"
           << "  \"transitions\": [\n";
    for (std::size_t index = 0; index < summaries.size(); ++index) {
      const auto &summary = summaries[index];
      output << "    {\n"
             << "      \"transition\": \""
             << json_escape(summary.transition) << "\",\n"
             << "      \"delta_rows\": " << summary.delta_rows << ",\n"
             << "      \"queries\": " << summary.queries << ",\n"
             << "      \"queries_with_gain\": "
             << summary.queries_with_gain << ",\n"
             << "      \"fraction_with_gain\": " << std::setprecision(17)
             << static_cast<double>(summary.queries_with_gain) /
                    static_cast<double>(summary.queries)
             << ",\n"
             << "      \"new_hits_median\": "
             << quantile(summary.hits, 0.5) << ",\n"
             << "      \"new_hits_p95\": " << quantile(summary.hits, 0.95)
             << ",\n"
             << "      \"elapsed_s\": " << summary.seconds << "\n"
             << "    }" << (index + 1 == summaries.size() ? "\n" : ",\n");
    }
    output << "  ],\n"
           << "  \"transitions_at_least_half_gain\": " << qualifying << ",\n"
           << "  \"freshness_fraction_component_pass\": "
           << (qualifying >= 4 ? "true" : "false") << "\n"
           << "}\n";
    return qualifying >= 4 ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
