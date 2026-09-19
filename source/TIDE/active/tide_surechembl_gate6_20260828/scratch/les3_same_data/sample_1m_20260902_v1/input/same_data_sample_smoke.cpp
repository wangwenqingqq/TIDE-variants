#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "LES3.h"

struct Row {
    int64_t id;
    std::multiset<int> tokens;
};

static std::multiset<int> parse_set(const std::string &line) {
    std::istringstream stream(line);
    std::multiset<int> tokens;
    int token;
    while(stream >> token) tokens.insert(token);
    return tokens;
}

static uint32_t intersection_size(const std::multiset<int> &left,
                                  const std::multiset<int> &right) {
    uint32_t count = 0;
    auto left_it = left.begin();
    auto right_it = right.begin();
    while(left_it != left.end() && right_it != right.end()) {
        if(*left_it < *right_it) ++left_it;
        else if(*right_it < *left_it) ++right_it;
        else {
            count++;
            ++left_it;
            ++right_it;
        }
    }
    return count;
}

static Row read_query(const std::string &sets_path,
                      const std::string &ids_path,
                      size_t query_row) {
    std::ifstream sets_input(sets_path);
    std::ifstream ids_input(ids_path);
    std::string line;
    int64_t id;
    for(size_t row = 0; std::getline(sets_input, line) && ids_input >> id; row++) {
        if(row == query_row) return {id, parse_set(line)};
    }
    throw std::runtime_error("query row is outside the sample");
}

static std::vector<int64_t> streaming_oracle(const std::string &sets_path,
                                             const std::string &ids_path,
                                             const std::multiset<int> &query,
                                             uint32_t numerator,
                                             uint32_t denominator) {
    std::ifstream sets_input(sets_path);
    std::ifstream ids_input(ids_path);
    std::string line;
    int64_t id;
    std::vector<int64_t> result;
    while(std::getline(sets_input, line) && ids_input >> id) {
        const auto candidate = parse_set(line);
        const uint32_t intersection = intersection_size(query, candidate);
        const uint32_t union_size = static_cast<uint32_t>(query.size() + candidate.size() - intersection);
        if(static_cast<uint64_t>(denominator) * intersection >=
           static_cast<uint64_t>(numerator) * union_size) {
            result.push_back(id);
        }
    }
    if(std::getline(sets_input, line) || ids_input >> id) {
        throw std::runtime_error("sample set and ID counts differ");
    }
    return result;
}

static void check(std::vector<int64_t> got,
                  std::vector<int64_t> expected,
                  const std::string &name) {
    std::sort(got.begin(), got.end());
    std::sort(expected.begin(), expected.end());
    if(got != expected) {
        std::cerr << "FAIL " << name << " expected=" << expected.size()
                  << " got=" << got.size() << '\n';
        std::exit(1);
    }
    std::cout << "PASS " << name << " matches=" << got.size() << '\n';
}

int main(int argc, char **argv) {
    if(argc != 5) {
        std::cerr << "usage: same_data_sample_smoke SETS IDS GROUPS QUERY_ROW\n";
        return 2;
    }
    const std::string sets_path = argv[1];
    const std::string ids_path = argv[2];
    const std::string groups_path = argv[3];
    const size_t query_row = std::stoull(argv[4]);
    const Row query = read_query(sets_path, ids_path, query_row);
    if(query.tokens.empty()) throw std::runtime_error("selected query is empty");

    LES3 les3(sets_path, groups_path, ids_path);
    check(les3.findDeltaNNExact(query.tokens, 7, 10),
          streaming_oracle(sets_path, ids_path, query.tokens, 7, 10),
          "same_data_sample_7_10");
    check(les3.findDeltaNNExact(query.tokens, 4, 5),
          streaming_oracle(sets_path, ids_path, query.tokens, 4, 5),
          "same_data_sample_4_5");

    constexpr int64_t inserted_id = 910000000000000001LL;
    les3.insertExact(query.tokens, inserted_id);
    auto expected = streaming_oracle(sets_path, ids_path, query.tokens, 7, 10);
    expected.push_back(inserted_id);
    check(les3.findDeltaNNExact(query.tokens, 7, 10), expected,
          "same_data_sample_post_insert_visibility");
    std::cout << "ALL_SAME_DATA_SAMPLE_GATES_PASS query_row=" << query_row
              << " query_id=" << query.id
              << " query_tokens=" << query.tokens.size() << '\n';
    return 0;
}
