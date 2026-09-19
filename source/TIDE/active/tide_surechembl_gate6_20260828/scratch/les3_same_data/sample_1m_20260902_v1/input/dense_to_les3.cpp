#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

struct Options {
    fs::path fingerprints;
    fs::path ids;
    fs::path popcounts;
    fs::path output_sets;
    fs::path output_ids;
    uint64_t expected_rows = 0;
    uint64_t limit = 0;
    uint64_t sample_count = 0;
};

static uint64_t parse_u64(const std::string &text, const char *name) {
    uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if(result.ec != std::errc() || result.ptr != text.data() + text.size()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return value;
}

static Options parse_options(int argc, char **argv) {
    if(argc < 6) {
        throw std::runtime_error(
            "usage: dense_to_les3 FP_U64X4 ID_I64 POPCNT_U16 OUT_SETS OUT_IDS "
            "[--expected-rows N] [--limit N | --sample-count N]");
    }
    Options options{argv[1], argv[2], argv[3], argv[4], argv[5]};
    for(int i = 6; i < argc; i += 2) {
        if(i + 1 >= argc) throw std::runtime_error("missing option value");
        const std::string flag = argv[i];
        if(flag == "--expected-rows") options.expected_rows = parse_u64(argv[i + 1], "expected rows");
        else if(flag == "--limit") options.limit = parse_u64(argv[i + 1], "limit");
        else if(flag == "--sample-count") options.sample_count = parse_u64(argv[i + 1], "sample count");
        else throw std::runtime_error("unknown option: " + flag);
    }
    if(options.limit != 0 && options.sample_count != 0) {
        throw std::runtime_error("--limit and --sample-count are mutually exclusive");
    }
    if(options.output_sets == options.output_ids) {
        throw std::runtime_error("set and ID outputs must differ");
    }
    return options;
}

static uint64_t rows_from_size(const fs::path &path, uint64_t row_bytes) {
    const uint64_t bytes = fs::file_size(path);
    if(bytes % row_bytes != 0) {
        throw std::runtime_error(path.string() + " has a partial row");
    }
    return bytes / row_bytes;
}

static void append_unsigned(std::string &output, uint64_t value) {
    std::array<char, 32> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if(result.ec != std::errc()) throw std::runtime_error("integer formatting failed");
    output.append(buffer.data(), result.ptr);
}

static void append_signed(std::string &output, int64_t value) {
    std::array<char, 32> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if(result.ec != std::errc()) throw std::runtime_error("ID formatting failed");
    output.append(buffer.data(), result.ptr);
}

static void append_set(std::string &output, const uint64_t *words) {
    bool first = true;
    for(uint64_t word_index = 0; word_index < 4; word_index++) {
        uint64_t bits = words[word_index];
        while(bits != 0) {
            const unsigned bit = static_cast<unsigned>(__builtin_ctzll(bits));
            if(!first) output.push_back(' ');
            append_unsigned(output, word_index * 64 + bit);
            first = false;
            bits &= bits - 1;
        }
    }
    output.push_back('\n');
}

static void flush_if_needed(std::ofstream &stream, std::string &buffer) {
    constexpr size_t flush_bytes = 8U << 20;
    if(buffer.size() >= flush_bytes) {
        stream.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        if(!stream) throw std::runtime_error("output write failed");
        buffer.clear();
    }
}

int main(int argc, char **argv) try {
    const Options options = parse_options(argc, argv);
    const uint64_t fp_rows = rows_from_size(options.fingerprints, 4 * sizeof(uint64_t));
    const uint64_t id_rows = rows_from_size(options.ids, sizeof(int64_t));
    const uint64_t popcount_rows = rows_from_size(options.popcounts, sizeof(uint16_t));
    if(fp_rows != id_rows || fp_rows != popcount_rows) {
        throw std::runtime_error("fingerprint, ID, and popcount row counts differ");
    }
    if(options.expected_rows != 0 && fp_rows != options.expected_rows) {
        throw std::runtime_error("source row count differs from --expected-rows");
    }
    if(options.limit > fp_rows || options.sample_count > fp_rows) {
        throw std::runtime_error("requested row count exceeds source rows");
    }
    if(fs::exists(options.output_sets) || fs::exists(options.output_ids)) {
        throw std::runtime_error("refusing to replace an existing output");
    }

    const uint64_t selected_rows = options.sample_count != 0
        ? options.sample_count : (options.limit != 0 ? options.limit : fp_rows);
    const std::string suffix = ".partial." + std::to_string(getpid());
    const fs::path partial_sets = options.output_sets.string() + suffix;
    const fs::path partial_ids = options.output_ids.string() + suffix;
    std::ifstream fp_input(options.fingerprints, std::ios::binary);
    std::ifstream id_input(options.ids, std::ios::binary);
    std::ifstream popcount_input(options.popcounts, std::ios::binary);
    std::ofstream set_output(partial_sets, std::ios::binary);
    std::ofstream id_output(partial_ids, std::ios::binary);
    if(!fp_input || !id_input || !popcount_input || !set_output || !id_output) {
        throw std::runtime_error("failed to open an input or output");
    }

    constexpr uint64_t chunk_rows = 1U << 16;
    std::vector<uint64_t> fingerprints(chunk_rows * 4);
    std::vector<int64_t> ids(chunk_rows);
    std::vector<uint16_t> popcounts(chunk_rows);
    std::string set_buffer;
    std::string id_buffer;
    set_buffer.reserve(9U << 20);
    id_buffer.reserve(9U << 20);

    uint64_t output_row = 0;
    uint64_t source_row = 0;
    uint64_t popcount_sum = 0;
    uint16_t popcount_min = std::numeric_limits<uint16_t>::max();
    uint16_t popcount_max = 0;
    uint64_t next_sample = options.sample_count == 0 ? 0 : 0;
    while(source_row < fp_rows && output_row < selected_rows) {
        const uint64_t remaining = fp_rows - source_row;
        const uint64_t rows = std::min(chunk_rows, remaining);
        fp_input.read(reinterpret_cast<char *>(fingerprints.data()), rows * 4 * sizeof(uint64_t));
        id_input.read(reinterpret_cast<char *>(ids.data()), rows * sizeof(int64_t));
        popcount_input.read(reinterpret_cast<char *>(popcounts.data()), rows * sizeof(uint16_t));
        if(!fp_input || !id_input || !popcount_input) {
            throw std::runtime_error("short read from dense input");
        }
        for(uint64_t local_row = 0; local_row < rows && output_row < selected_rows; local_row++) {
            const uint64_t absolute_row = source_row + local_row;
            const bool selected = options.sample_count == 0
                ? absolute_row < selected_rows : absolute_row == next_sample;
            if(!selected) continue;
            const uint64_t *words = fingerprints.data() + local_row * 4;
            const uint16_t computed = static_cast<uint16_t>(
                __builtin_popcountll(words[0]) + __builtin_popcountll(words[1]) +
                __builtin_popcountll(words[2]) + __builtin_popcountll(words[3]));
            if(computed != popcounts[local_row]) {
                throw std::runtime_error("stored popcount mismatch at source row " +
                                         std::to_string(absolute_row));
            }
            append_set(set_buffer, words);
            append_signed(id_buffer, ids[local_row]);
            id_buffer.push_back('\n');
            popcount_sum += computed;
            popcount_min = std::min(popcount_min, computed);
            popcount_max = std::max(popcount_max, computed);
            output_row++;
            if(options.sample_count != 0 && output_row < selected_rows) {
                next_sample = static_cast<uint64_t>(
                    (static_cast<__uint128_t>(output_row) * fp_rows) / selected_rows);
            }
            flush_if_needed(set_output, set_buffer);
            flush_if_needed(id_output, id_buffer);
        }
        source_row += rows;
    }
    if(output_row != selected_rows) throw std::runtime_error("selection produced the wrong row count");
    set_output.write(set_buffer.data(), static_cast<std::streamsize>(set_buffer.size()));
    id_output.write(id_buffer.data(), static_cast<std::streamsize>(id_buffer.size()));
    set_output.close();
    id_output.close();
    if(!set_output || !id_output) throw std::runtime_error("failed to finalize output");
    fs::rename(partial_sets, options.output_sets);
    fs::rename(partial_ids, options.output_ids);

    std::cout << "{\n"
              << "  \"source_rows\": " << fp_rows << ",\n"
              << "  \"selected_rows\": " << selected_rows << ",\n"
              << "  \"popcount_sum\": " << popcount_sum << ",\n"
              << "  \"popcount_min\": " << popcount_min << ",\n"
              << "  \"popcount_max\": " << popcount_max << ",\n"
              << "  \"output_sets_bytes\": " << fs::file_size(options.output_sets) << ",\n"
              << "  \"output_ids_bytes\": " << fs::file_size(options.output_ids) << "\n"
              << "}\n";
    return 0;
}
catch(const std::exception &error) {
    std::cerr << "dense_to_les3: " << error.what() << '\n';
    return 1;
}
