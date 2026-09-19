#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
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
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kWords = 4;
constexpr std::size_t kQueryWords = 6;

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

struct Group {
    std::uint64_t rows = 0;
    std::array<std::uint64_t, kWords> union_words{};
};

struct Query {
    std::uint64_t id = 0;
    std::array<std::uint64_t, kWords> words{};
    std::uint64_t popcount = 0;
};

struct Observation {
    std::uint64_t candidate_groups = 0;
    std::uint64_t candidate_rows = 0;
};

std::vector<Group> read_groups(const fs::path &path) {
    std::ifstream input(path);
    if(!input) fail("cannot open group statistics: " + path.string());
    std::string line;
    if(!std::getline(input, line) || line !=
       "group\trows\tunion_popcount\tword0\tword1\tword2\tword3") {
        fail("unexpected group-statistics header");
    }
    std::vector<Group> groups;
    std::uint64_t expected_group = 0;
    while(std::getline(input, line)) {
        std::istringstream stream(line);
        std::string field;
        std::array<std::string, 7> fields{};
        for(auto &value : fields) {
            if(!std::getline(stream, value, '\t')) fail("short group-statistics row");
        }
        if(std::getline(stream, field, '\t')) fail("long group-statistics row");
        if(std::stoull(fields[0]) != expected_group++) fail("non-consecutive group ID");
        Group group;
        group.rows = std::stoull(fields[1]);
        unsigned union_popcount = 0;
        for(std::size_t word = 0; word < kWords; ++word) {
            group.union_words[word] = std::stoull(fields[3 + word], nullptr, 16);
            union_popcount += static_cast<unsigned>(std::popcount(group.union_words[word]));
        }
        if(union_popcount != std::stoul(fields[2])) fail("union popcount mismatch");
        groups.push_back(group);
    }
    if(groups.empty()) fail("no groups were loaded");
    return groups;
}

std::vector<Query> read_queries(const fs::path &path) {
    const auto bytes = fs::file_size(path);
    if(bytes == 0 || bytes % (kQueryWords * sizeof(std::uint64_t)) != 0) {
        fail("invalid packed query file size");
    }
    std::ifstream input(path, std::ios::binary);
    if(!input) fail("cannot open packed queries: " + path.string());
    const auto count = bytes / (kQueryWords * sizeof(std::uint64_t));
    std::vector<Query> queries(count);
    for(auto &query : queries) {
        std::array<std::uint64_t, kQueryWords> packed{};
        input.read(reinterpret_cast<char *>(packed.data()), sizeof(packed));
        if(input.gcount() != static_cast<std::streamsize>(sizeof(packed))) {
            fail("short query read");
        }
        query.id = packed[0];
        unsigned popcount = 0;
        for(std::size_t word = 0; word < kWords; ++word) {
            query.words[word] = packed[1 + word];
            popcount += static_cast<unsigned>(std::popcount(query.words[word]));
        }
        query.popcount = packed[5];
        if(query.popcount == 0 || query.popcount != popcount) fail("query popcount mismatch");
    }
    return queries;
}

double percentile(std::vector<double> values, double probability) {
    if(values.empty()) fail("cannot summarize empty observations");
    std::sort(values.begin(), values.end());
    const double position = probability * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - lower;
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

void summarize(std::ofstream &output, const std::string &label,
               const std::vector<Observation> &observations,
               std::uint64_t total_groups, std::uint64_t total_rows) {
    std::vector<double> group_fractions;
    std::vector<double> row_fractions;
    group_fractions.reserve(observations.size());
    row_fractions.reserve(observations.size());
    double group_sum = 0;
    double row_sum = 0;
    std::uint64_t full_group_queries = 0;
    std::uint64_t full_row_queries = 0;
    for(const auto &observation : observations) {
        const double group_fraction = static_cast<double>(observation.candidate_groups) /
                                      static_cast<double>(total_groups);
        const double row_fraction = static_cast<double>(observation.candidate_rows) /
                                    static_cast<double>(total_rows);
        group_fractions.push_back(group_fraction);
        row_fractions.push_back(row_fraction);
        group_sum += group_fraction;
        row_sum += row_fraction;
        full_group_queries += observation.candidate_groups == total_groups;
        full_row_queries += observation.candidate_rows == total_rows;
    }
    output << std::setprecision(17)
           << label << ".mean_candidate_group_fraction=" << group_sum / observations.size() << '\n'
           << label << ".median_candidate_group_fraction=" << percentile(group_fractions, 0.5) << '\n'
           << label << ".p95_candidate_group_fraction=" << percentile(group_fractions, 0.95) << '\n'
           << label << ".mean_candidate_row_fraction=" << row_sum / observations.size() << '\n'
           << label << ".median_candidate_row_fraction=" << percentile(row_fractions, 0.5) << '\n'
           << label << ".p95_candidate_row_fraction=" << percentile(row_fractions, 0.95) << '\n'
           << label << ".full_group_queries=" << full_group_queries << '\n'
           << label << ".full_row_queries=" << full_row_queries << '\n';
}

}  // namespace

int main(int argc, char **argv) {
    try {
        if(argc != 4) {
            std::cerr << "usage: tgm_query_cost GROUPS_TSV QUERIES_U64X6 OUTPUT_PREFIX\n";
            return 2;
        }
        const auto groups = read_groups(argv[1]);
        const auto queries = read_queries(argv[2]);
        const fs::path prefix = argv[3];
        const fs::path detail_path = prefix.string() + ".queries.tsv";
        const fs::path summary_path = prefix.string() + ".summary.txt";
        if(fs::exists(detail_path) || fs::exists(summary_path)) {
            fail("refusing to replace query-cost output");
        }
        if(!prefix.parent_path().empty()) fs::create_directories(prefix.parent_path());
        std::uint64_t total_rows = 0;
        for(const auto &group : groups) total_rows += group.rows;
        struct Threshold { unsigned numerator; unsigned denominator; const char *label; };
        constexpr std::array<Threshold, 2> thresholds{{{7, 10, "7_10"}, {4, 5, "4_5"}}};
        std::array<std::vector<Observation>, thresholds.size()> observations;
        std::ofstream detail(detail_path);
        if(!detail) fail("cannot create query details");
        detail << "query\tid\tpopcount\tthreshold\tcandidate_groups\tcandidate_rows"
                  "\tcandidate_group_fraction\tcandidate_row_fraction\n";
        for(std::size_t query_index = 0; query_index < queries.size(); ++query_index) {
            const auto &query = queries[query_index];
            for(std::size_t threshold_index = 0; threshold_index < thresholds.size(); ++threshold_index) {
                const auto threshold = thresholds[threshold_index];
                Observation observation;
                for(const auto &group : groups) {
                    unsigned common = 0;
                    for(std::size_t word = 0; word < kWords; ++word) {
                        common += static_cast<unsigned>(
                            std::popcount(query.words[word] & group.union_words[word]));
                    }
                    if(static_cast<std::uint64_t>(threshold.denominator) * common >=
                       static_cast<std::uint64_t>(threshold.numerator) * query.popcount) {
                        ++observation.candidate_groups;
                        observation.candidate_rows += group.rows;
                    }
                }
                observations[threshold_index].push_back(observation);
                detail << query_index << '\t' << query.id << '\t' << query.popcount << '\t'
                       << threshold.numerator << '/' << threshold.denominator << '\t'
                       << observation.candidate_groups << '\t' << observation.candidate_rows << '\t'
                       << std::setprecision(17)
                       << static_cast<double>(observation.candidate_groups) / groups.size() << '\t'
                       << static_cast<double>(observation.candidate_rows) / total_rows << '\n';
            }
        }
        if(!detail) fail("failed to write query details");
        std::ofstream summary(summary_path);
        if(!summary) fail("cannot create query-cost summary");
        summary << "format=tgm_query_cost_v1\n"
                << "queries=" << queries.size() << "\n"
                << "groups=" << groups.size() << "\n"
                << "rows=" << total_rows << "\n";
        for(std::size_t index = 0; index < thresholds.size(); ++index) {
            summarize(summary, thresholds[index].label, observations[index], groups.size(), total_rows);
        }
        if(!summary) fail("failed to write query-cost summary");
        std::cout << "PASS queries=" << queries.size() << " groups=" << groups.size()
                  << " rows=" << total_rows << '\n';
        return 0;
    } catch(const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}

