#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kWords = 4;
constexpr std::size_t kInput = 16;
constexpr std::size_t kHidden = 8;
constexpr std::size_t kParameters = 217;

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

std::uint64_t exact_file_size(const fs::path &path) {
    std::error_code error;
    const auto size = fs::file_size(path, error);
    if(error) fail("cannot stat " + path.string() + ": " + error.message());
    return size;
}

void require_absent(const fs::path &path) {
    if(fs::exists(path)) fail("refusing to replace existing output: " + path.string());
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

template <typename T>
class ReadOnlyMap {
public:
    ReadOnlyMap(const fs::path &path, std::uint64_t count) : count_(count) {
        const auto expected = count * sizeof(T);
        if(exact_file_size(path) != expected) {
            fail("byte size mismatch for " + path.string());
        }
        fd_ = ::open(path.c_str(), O_RDONLY);
        if(fd_ < 0) fail("cannot open " + path.string() + ": " + std::strerror(errno));
        if(expected != 0) {
            data_ = static_cast<const T *>(
                ::mmap(nullptr, static_cast<std::size_t>(expected), PROT_READ,
                       MAP_PRIVATE, fd_, 0));
            if(data_ == MAP_FAILED) {
                data_ = nullptr;
                fail("cannot mmap " + path.string() + ": " + std::strerror(errno));
            }
        }
    }

    ReadOnlyMap(const ReadOnlyMap &) = delete;
    ReadOnlyMap &operator=(const ReadOnlyMap &) = delete;

    ~ReadOnlyMap() {
        if(data_ != nullptr) {
            ::munmap(const_cast<T *>(data_), static_cast<std::size_t>(count_ * sizeof(T)));
        }
        if(fd_ >= 0) ::close(fd_);
    }

    const T &operator[](std::uint64_t index) const {
        if(index >= count_) fail("mapped read is out of bounds");
        return data_[index];
    }

    const T *data() const { return data_; }

private:
    int fd_ = -1;
    const T *data_ = nullptr;
    std::uint64_t count_ = 0;
};

struct Options {
    fs::path fp_path;
    fs::path ptr_path;
    fs::path order_path;
    fs::path ptr_meta_path;
    fs::path output_prefix;
    fs::path initial_offsets_path;
    std::uint64_t rows = 0;
    unsigned initial_level = 7;
    unsigned initial_groups = 128;
    unsigned target_level = 10;
    unsigned emit_from_level = 10;
    std::uint64_t seed = 20260902;
    std::size_t pairs_per_model = 40000;
    unsigned epochs = 3;
    std::size_t batch_size = 256;
    unsigned workers = 1;
    std::size_t stop_below = 50;
    bool write_text = false;
};

std::uint64_t parse_u64(const std::string &name, const std::string &value) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed);
    if(consumed != value.size()) fail("invalid " + name + ": " + value);
    return parsed;
}

unsigned parse_unsigned(const std::string &name, const std::string &value) {
    const auto parsed = parse_u64(name, value);
    if(parsed > std::numeric_limits<unsigned>::max()) {
        fail("value is too large for " + name + ": " + value);
    }
    return static_cast<unsigned>(parsed);
}

Options parse_options(int argc, char **argv) {
    if(argc < 6) {
        fail("usage: l2p_train FP PTR ORDER PTR_META OUTPUT_PREFIX [options]");
    }
    Options options;
    options.fp_path = argv[1];
    options.ptr_path = argv[2];
    options.order_path = argv[3];
    options.ptr_meta_path = argv[4];
    options.output_prefix = argv[5];
    for(int index = 6; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&](const std::string &name) -> std::string {
            if(index + 1 >= argc) fail("missing value after " + name);
            return argv[++index];
        };
        if(argument == "--rows") options.rows = parse_u64(argument, value(argument));
        else if(argument == "--initial-level") options.initial_level = parse_unsigned(argument, value(argument));
        else if(argument == "--initial-groups") options.initial_groups = parse_unsigned(argument, value(argument));
        else if(argument == "--initial-offsets") options.initial_offsets_path = value(argument);
        else if(argument == "--target-level") options.target_level = parse_unsigned(argument, value(argument));
        else if(argument == "--emit-from-level") options.emit_from_level = parse_unsigned(argument, value(argument));
        else if(argument == "--seed") options.seed = parse_u64(argument, value(argument));
        else if(argument == "--pairs-per-model") options.pairs_per_model = parse_u64(argument, value(argument));
        else if(argument == "--epochs") options.epochs = parse_unsigned(argument, value(argument));
        else if(argument == "--batch-size") options.batch_size = parse_u64(argument, value(argument));
        else if(argument == "--workers") options.workers = parse_unsigned(argument, value(argument));
        else if(argument == "--stop-below") options.stop_below = parse_u64(argument, value(argument));
        else if(argument == "--write-text") options.write_text = true;
        else fail("unknown argument: " + argument);
    }
    if(options.rows == 0 || options.rows > std::numeric_limits<std::uint32_t>::max()) {
        fail("--rows must be in [1, 2^32-1]");
    }
    if(options.initial_groups == 0 || options.initial_groups > options.rows) {
        fail("--initial-groups must be in [1, rows]");
    }
    if(options.initial_offsets_path.empty() &&
       (options.initial_level >= 63 ||
        options.initial_groups != (std::uint64_t{1} << options.initial_level))) {
        fail("without --initial-offsets, initial groups must equal 2^initial-level");
    }
    if(options.target_level < options.initial_level) fail("target level precedes initial level");
    if(options.emit_from_level < options.initial_level ||
       options.emit_from_level > options.target_level) {
        fail("emit-from level is outside the trained level range");
    }
    if(options.pairs_per_model == 0 || options.epochs == 0 ||
       options.batch_size == 0 || options.workers == 0 || options.stop_below < 2) {
        fail("training counts, workers, and stop threshold must be positive");
    }
    return options;
}

unsigned read_normalization_denominator(const fs::path &path,
                                        std::uint64_t expected_rows) {
    std::ifstream input(path);
    if(!input) fail("cannot open PTR metadata: " + path.string());
    std::unordered_map<std::string, std::string> values;
    std::string line;
    while(std::getline(input, line)) {
        const auto position = line.find('=');
        if(position != std::string::npos) {
            values.emplace(line.substr(0, position), line.substr(position + 1));
        }
    }
    if(values["format"] != "l2p_ptr_u8x16_v1") fail("unexpected PTR format");
    if(parse_u64("PTR rows", values["rows"]) != expected_rows) fail("PTR row metadata mismatch");
    if(parse_u64("PTR dims", values["ptr_dims"]) != kInput) fail("PTR dimension mismatch");
    const auto denominator = parse_u64("PTR normalization denominator",
                                       values["normalization_denominator"]);
    if(denominator == 0 || denominator > 255) fail("invalid PTR normalization denominator");
    return static_cast<unsigned>(denominator);
}

struct Group {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::uint64_t lineage = 0;
};

std::vector<Group> initial_groups(const Options &options) {
    std::vector<Group> groups;
    if(!options.initial_offsets_path.empty()) {
        const auto bytes = exact_file_size(options.initial_offsets_path);
        if(bytes % sizeof(std::uint64_t) != 0 || bytes < 2 * sizeof(std::uint64_t)) {
            fail("invalid initial offsets file");
        }
        const auto count = bytes / sizeof(std::uint64_t);
        ReadOnlyMap<std::uint64_t> offsets(options.initial_offsets_path, count);
        if(offsets[0] != 0 || offsets[count - 1] != options.rows) {
            fail("initial offsets do not span all rows");
        }
        groups.reserve(count - 1);
        for(std::uint64_t i = 0; i + 1 < count; ++i) {
            if(offsets[i] >= offsets[i + 1]) fail("initial groups must be non-empty");
            groups.push_back({offsets[i], offsets[i + 1], i});
        }
        if(groups.size() != options.initial_groups) {
            fail("--initial-groups disagrees with --initial-offsets");
        }
        return groups;
    }
    groups.reserve(options.initial_groups);
    for(std::uint64_t group = 0; group < options.initial_groups; ++group) {
        const auto begin = group * options.rows / options.initial_groups;
        const auto end = (group + 1) * options.rows / options.initial_groups;
        if(begin >= end) fail("balanced initialization created an empty group");
        groups.push_back({begin, end, group});
    }
    return groups;
}

struct DenseRow {
    const std::uint64_t *words;
};

std::uint32_t intersection(const DenseRow left, const DenseRow right) {
    unsigned count = 0;
    for(std::size_t word = 0; word < kWords; ++word) {
        count += static_cast<unsigned>(std::popcount(left.words[word] & right.words[word]));
    }
    return count;
}

std::uint32_t population(const DenseRow row) {
    unsigned count = 0;
    for(std::size_t word = 0; word < kWords; ++word) {
        count += static_cast<unsigned>(std::popcount(row.words[word]));
    }
    return count;
}

struct Pair {
    std::uint32_t left;
    std::uint32_t right;
    float distance;
};

constexpr std::size_t w1_index(std::size_t output, std::size_t input) {
    return output * kInput + input;
}
constexpr std::size_t b1_index(std::size_t output) { return 128 + output; }
constexpr std::size_t w2_index(std::size_t output, std::size_t input) {
    return 136 + output * kHidden + input;
}
constexpr std::size_t b2_index(std::size_t output) { return 200 + output; }
constexpr std::size_t w3_index(std::size_t input) { return 208 + input; }
constexpr std::size_t b3_index() { return 216; }

struct ForwardCache {
    std::array<double, kInput> x{};
    std::array<double, kHidden> z1{};
    std::array<double, kHidden> h1{};
    std::array<double, kHidden> z2{};
    std::array<double, kHidden> h2{};
    double output = 0;
};

class Model {
public:
    explicit Model(std::uint64_t seed) {
        std::mt19937_64 random(seed);
        initialize_layer(random, 0, 128, kInput);
        initialize_layer(random, 128, 8, kInput);
        initialize_layer(random, 136, 64, kHidden);
        initialize_layer(random, 200, 8, kHidden);
        initialize_layer(random, 208, 8, kHidden);
        initialize_layer(random, 216, 1, kHidden);
    }

    ForwardCache forward(const std::uint8_t *raw, double denominator) const {
        ForwardCache cache;
        for(std::size_t input = 0; input < kInput; ++input) {
            cache.x[input] = static_cast<double>(raw[input]) / denominator;
        }
        for(std::size_t output = 0; output < kHidden; ++output) {
            double value = parameters_[b1_index(output)];
            for(std::size_t input = 0; input < kInput; ++input) {
                value += parameters_[w1_index(output, input)] * cache.x[input];
            }
            cache.z1[output] = value;
            cache.h1[output] = std::max(0.0, value);
        }
        for(std::size_t output = 0; output < kHidden; ++output) {
            double value = parameters_[b2_index(output)];
            for(std::size_t input = 0; input < kHidden; ++input) {
                value += parameters_[w2_index(output, input)] * cache.h1[input];
            }
            cache.z2[output] = value;
            cache.h2[output] = std::max(0.0, value);
        }
        double logit = parameters_[b3_index()];
        for(std::size_t input = 0; input < kHidden; ++input) {
            logit += parameters_[w3_index(input)] * cache.h2[input];
        }
        if(logit >= 0) cache.output = 1.0 / (1.0 + std::exp(-logit));
        else {
            const double exponential = std::exp(logit);
            cache.output = exponential / (1.0 + exponential);
        }
        if(!std::isfinite(cache.output)) fail("non-finite model output");
        return cache;
    }

    void backward(const ForwardCache &cache, double output_gradient,
                  std::array<double, kParameters> &gradient) const {
        const double logit_gradient =
            output_gradient * cache.output * (1.0 - cache.output);
        gradient[b3_index()] += logit_gradient;
        std::array<double, kHidden> h2_gradient{};
        for(std::size_t input = 0; input < kHidden; ++input) {
            gradient[w3_index(input)] += logit_gradient * cache.h2[input];
            h2_gradient[input] = logit_gradient * parameters_[w3_index(input)];
        }
        std::array<double, kHidden> h1_gradient{};
        for(std::size_t output = 0; output < kHidden; ++output) {
            const double z2_gradient = cache.z2[output] > 0 ? h2_gradient[output] : 0.0;
            gradient[b2_index(output)] += z2_gradient;
            for(std::size_t input = 0; input < kHidden; ++input) {
                gradient[w2_index(output, input)] += z2_gradient * cache.h1[input];
                h1_gradient[input] += z2_gradient * parameters_[w2_index(output, input)];
            }
        }
        for(std::size_t output = 0; output < kHidden; ++output) {
            const double z1_gradient = cache.z1[output] > 0 ? h1_gradient[output] : 0.0;
            gradient[b1_index(output)] += z1_gradient;
            for(std::size_t input = 0; input < kInput; ++input) {
                gradient[w1_index(output, input)] += z1_gradient * cache.x[input];
            }
        }
    }

    void adam_step(const std::array<double, kParameters> &gradient,
                   std::uint64_t step) {
        constexpr double learning_rate = 1e-2;
        constexpr double beta1 = 0.9;
        constexpr double beta2 = 0.999;
        constexpr double epsilon = 1e-8;
        const double correction1 = 1.0 - std::pow(beta1, static_cast<double>(step));
        const double correction2 = 1.0 - std::pow(beta2, static_cast<double>(step));
        for(std::size_t index = 0; index < kParameters; ++index) {
            first_moment_[index] = beta1 * first_moment_[index] + (1.0 - beta1) * gradient[index];
            second_moment_[index] = beta2 * second_moment_[index] +
                                    (1.0 - beta2) * gradient[index] * gradient[index];
            const double first = first_moment_[index] / correction1;
            const double second = second_moment_[index] / correction2;
            parameters_[index] -= learning_rate * first / (std::sqrt(second) + epsilon);
            if(!std::isfinite(parameters_[index])) fail("non-finite model parameter");
        }
    }

private:
    void initialize_layer(std::mt19937_64 &random, std::size_t begin,
                          std::size_t count, std::size_t fan_in) {
        const double bound = 1.0 / std::sqrt(static_cast<double>(fan_in));
        std::uniform_real_distribution<double> distribution(-bound, bound);
        for(std::size_t index = begin; index < begin + count; ++index) {
            parameters_[index] = distribution(random);
        }
    }

    std::array<double, kParameters> parameters_{};
    std::array<double, kParameters> first_moment_{};
    std::array<double, kParameters> second_moment_{};
};

struct SplitResult {
    std::uint64_t lineage = 0;
    std::uint64_t input_size = 0;
    std::uint64_t left_size = 0;
    bool split = false;
    bool fallback = false;
    std::array<double, 3> epoch_loss{
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()};
};

class Trainer {
public:
    Trainer(const Options &options, const ReadOnlyMap<std::uint64_t> &fingerprints,
            const ReadOnlyMap<std::uint8_t> &ptr, double denominator)
        : options_(options), fingerprints_(fingerprints), ptr_(ptr),
          denominator_(denominator) {}

    SplitResult split(const Group &group, unsigned level,
                      const std::vector<std::uint32_t> &current,
                      std::vector<std::uint32_t> &next) const {
        const std::uint64_t size = group.end - group.begin;
        if(size < options_.stop_below) {
            std::copy(current.begin() + static_cast<std::ptrdiff_t>(group.begin),
                      current.begin() + static_cast<std::ptrdiff_t>(group.end),
                      next.begin() + static_cast<std::ptrdiff_t>(group.begin));
            return {.lineage = group.lineage, .input_size = size, .left_size = size,
                    .split = false, .fallback = false};
        }
        const std::uint64_t model_seed = splitmix64(
            options_.seed ^ (static_cast<std::uint64_t>(level) << 56U) ^ group.lineage);
        std::mt19937_64 random(model_seed);
        std::uniform_int_distribution<std::uint64_t> member(0, size - 1);
        std::uniform_int_distribution<std::uint64_t> other(0, size - 2);
        std::vector<Pair> pairs;
        pairs.reserve(options_.pairs_per_model);
        for(std::size_t index = 0; index < options_.pairs_per_model; ++index) {
            const auto left_local = member(random);
            auto right_local = other(random);
            if(right_local >= left_local) ++right_local;
            const auto left_row = current[group.begin + left_local];
            const auto right_row = current[group.begin + right_local];
            const DenseRow left{fingerprints_.data() + left_row * kWords};
            const DenseRow right{fingerprints_.data() + right_row * kWords};
            const auto denominator = population(left) + population(right);
            if(denominator == 0) fail("empty pair encountered during training");
            const double distance = 1.0 - 2.0 * intersection(left, right) /
                                                static_cast<double>(denominator);
            pairs.push_back({static_cast<std::uint32_t>(left_local),
                             static_cast<std::uint32_t>(right_local),
                             static_cast<float>(distance)});
        }

        Model model(splitmix64(model_seed));
        SplitResult result;
        result.lineage = group.lineage;
        result.input_size = size;
        result.split = true;
        std::uint64_t step = 0;
        for(unsigned epoch = 0; epoch < options_.epochs; ++epoch) {
            std::shuffle(pairs.begin(), pairs.end(), random);
            double total_loss = 0;
            for(std::size_t begin = 0; begin < pairs.size(); begin += options_.batch_size) {
                const auto end = std::min(begin + options_.batch_size, pairs.size());
                const double batch_inverse = 1.0 / static_cast<double>(end - begin);
                std::array<double, kParameters> gradient{};
                for(std::size_t pair_index = begin; pair_index < end; ++pair_index) {
                    const auto &pair = pairs[pair_index];
                    const auto left_row = current[group.begin + pair.left];
                    const auto right_row = current[group.begin + pair.right];
                    const auto left = model.forward(ptr_.data() + left_row * kInput, denominator_);
                    const auto right = model.forward(ptr_.data() + right_row * kInput, denominator_);
                    const bool same_side = (left.output < 0.5 && right.output < 0.5) ||
                                           (left.output >= 0.5 && right.output >= 0.5);
                    if(!same_side) continue;
                    const double difference = left.output - right.output;
                    total_loss += pair.distance * (0.5 - std::abs(difference));
                    if(difference == 0) continue;
                    const double sign = difference > 0 ? 1.0 : -1.0;
                    const double left_gradient = -pair.distance * sign * batch_inverse;
                    model.backward(left, left_gradient, gradient);
                    model.backward(right, -left_gradient, gradient);
                }
                model.adam_step(gradient, ++step);
            }
            const double average = total_loss / static_cast<double>(pairs.size());
            if(!std::isfinite(average)) fail("non-finite epoch loss");
            if(epoch < result.epoch_loss.size()) result.epoch_loss[epoch] = average;
        }

        std::vector<std::uint8_t> side(size);
        std::uint64_t left_size = 0;
        for(std::uint64_t local = 0; local < size; ++local) {
            const auto row = current[group.begin + local];
            const bool right = model.forward(ptr_.data() + row * kInput, denominator_).output >= 0.5;
            side[local] = static_cast<std::uint8_t>(right);
            if(!right) ++left_size;
        }
        if(left_size == 0 || left_size == size) {
            result.fallback = true;
            std::vector<std::uint64_t> positions(size);
            std::iota(positions.begin(), positions.end(), 0);
            std::shuffle(positions.begin(), positions.end(), random);
            std::fill(side.begin(), side.end(), 1);
            left_size = size / 2;
            for(std::uint64_t index = 0; index < left_size; ++index) {
                side[positions[index]] = 0;
            }
        }
        if(left_size == 0 || left_size == size) fail("split did not create two non-empty groups");
        auto left_output = group.begin;
        auto right_output = group.begin + left_size;
        for(std::uint64_t local = 0; local < size; ++local) {
            const auto row = current[group.begin + local];
            if(side[local] == 0) next[left_output++] = row;
            else next[right_output++] = row;
        }
        if(left_output != group.begin + left_size || right_output != group.end) {
            fail("split output boundary mismatch");
        }
        result.left_size = left_size;
        return result;
    }

private:
    const Options &options_;
    const ReadOnlyMap<std::uint64_t> &fingerprints_;
    const ReadOnlyMap<std::uint8_t> &ptr_;
    double denominator_;
};

void validate_coverage(const std::vector<std::uint32_t> &order,
                       const std::vector<Group> &groups, std::uint64_t rows) {
    if(groups.empty() || groups.front().begin != 0 || groups.back().end != rows) {
        fail("group boundaries do not span every row");
    }
    for(std::size_t index = 0; index < groups.size(); ++index) {
        if(groups[index].begin >= groups[index].end) fail("empty group");
        if(index != 0 && groups[index - 1].end != groups[index].begin) {
            fail("group boundaries are not contiguous");
        }
    }
    std::vector<std::uint8_t> seen((rows + 7) / 8, 0);
    for(const auto row : order) {
        if(row >= rows) fail("partition contains an out-of-range row");
        const auto byte = row >> 3U;
        const auto mask = static_cast<std::uint8_t>(1U << (row & 7U));
        if((seen[byte] & mask) != 0) fail("partition contains a duplicate row");
        seen[byte] |= mask;
    }
    for(std::uint64_t row = 0; row < rows; ++row) {
        if((seen[row >> 3U] & static_cast<std::uint8_t>(1U << (row & 7U))) == 0) {
            fail("partition is missing a row");
        }
    }
}

void atomic_rename(const fs::path &source, const fs::path &destination) {
    std::error_code error;
    fs::rename(source, destination, error);
    if(error) fail("cannot rename " + source.string() + " to " +
                   destination.string() + ": " + error.message());
}

void emit_level(const Options &options, unsigned level,
                const std::vector<std::uint32_t> &order,
                const std::vector<Group> &groups,
                const std::vector<SplitResult> &split_results,
                unsigned source_level) {
    const fs::path stem = options.output_prefix.string() + ".level" + std::to_string(level);
    const fs::path order_path = stem.string() + ".order_u32.bin";
    const fs::path offsets_path = stem.string() + ".offsets_u64.bin";
    const fs::path metrics_path = stem.string() + ".metrics.tsv";
    const fs::path meta_path = stem.string() + ".meta.txt";
    const fs::path text_path = stem.string() + ".groups.txt";
    require_absent(order_path);
    require_absent(offsets_path);
    require_absent(metrics_path);
    require_absent(meta_path);
    if(options.write_text) require_absent(text_path);
    if(!options.output_prefix.parent_path().empty()) {
        fs::create_directories(options.output_prefix.parent_path());
    }
    const std::string suffix = ".tmp." + std::to_string(::getpid());
    const fs::path order_temp = order_path.string() + suffix;
    const fs::path offsets_temp = offsets_path.string() + suffix;
    const fs::path metrics_temp = metrics_path.string() + suffix;
    const fs::path meta_temp = meta_path.string() + suffix;
    const fs::path text_temp = text_path.string() + suffix;

    {
        std::ofstream output(order_temp, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char *>(order.data()),
                     static_cast<std::streamsize>(order.size() * sizeof(order[0])));
        if(!output) fail("failed to write compact partition order");
    }
    {
        std::ofstream output(offsets_temp, std::ios::binary | std::ios::trunc);
        const std::uint64_t zero = 0;
        output.write(reinterpret_cast<const char *>(&zero), sizeof(zero));
        for(const auto &group : groups) {
            output.write(reinterpret_cast<const char *>(&group.end), sizeof(group.end));
        }
        if(!output) fail("failed to write compact partition offsets");
    }
    {
        std::ofstream output(metrics_temp, std::ios::trunc);
        output << "source_level\tsource_group\tlineage\tinput_size\tsplit\tleft_size\tright_size"
                  "\tfallback\tepoch1_loss\tepoch2_loss\tepoch3_loss\n";
        for(std::size_t index = 0; index < split_results.size(); ++index) {
            const auto &result = split_results[index];
            output << source_level << '\t' << index << '\t' << result.lineage << '\t'
                   << result.input_size << '\t' << (result.split ? 1 : 0) << '\t'
                   << result.left_size << '\t';
            if(result.split) output << (result.input_size - result.left_size);
            else output << 0;
            output << '\t' << (result.fallback ? 1 : 0);
            for(const auto loss : result.epoch_loss) {
                if(std::isfinite(loss)) output << '\t' << std::setprecision(17) << loss;
                else output << "\tNA";
            }
            output << '\n';
        }
        if(!output) fail("failed to write split metrics");
    }
    if(options.write_text) {
        std::ofstream output(text_temp, std::ios::trunc);
        for(const auto &group : groups) {
            for(auto position = group.begin; position < group.end; ++position) {
                if(position != group.begin) output << ' ';
                output << order[position];
            }
            output << '\n';
        }
        if(!output) fail("failed to write text group file");
    }
    std::uint64_t min_group = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_group = 0;
    for(const auto &group : groups) {
        const auto size = group.end - group.begin;
        min_group = std::min(min_group, size);
        max_group = std::max(max_group, size);
    }
    std::uint64_t fallback_count = 0;
    std::uint64_t trained_count = 0;
    for(const auto &result : split_results) {
        fallback_count += result.fallback;
        trained_count += result.split;
    }
    {
        std::ofstream output(meta_temp, std::ios::trunc);
        output << "format=l2p_partition_v1\n"
               << "rows=" << options.rows << "\n"
               << "level=" << level << "\n"
               << "groups=" << groups.size() << "\n"
               << "min_group_size=" << min_group << "\n"
               << "max_group_size=" << max_group << "\n"
               << "source_level=" << source_level << "\n"
               << "source_models_trained=" << trained_count << "\n"
               << "source_collapse_fallbacks=" << fallback_count << "\n"
               << "seed=" << options.seed << "\n"
               << "pairs_per_model=" << options.pairs_per_model << "\n"
               << "epochs=" << options.epochs << "\n"
               << "batch_size=" << options.batch_size << "\n"
               << "workers=" << options.workers << "\n"
               << "stop_below=" << options.stop_below << "\n"
               << "text_groups=" << (options.write_text ? 1 : 0) << "\n";
        if(!output) fail("failed to write partition metadata");
    }

    atomic_rename(order_temp, order_path);
    atomic_rename(offsets_temp, offsets_path);
    atomic_rename(metrics_temp, metrics_path);
    atomic_rename(meta_temp, meta_path);
    if(options.write_text) atomic_rename(text_temp, text_path);
}

}  // namespace

int main(int argc, char **argv) {
    try {
        if constexpr(std::endian::native != std::endian::little) {
            fail("only little-endian hosts are supported");
        }
        const Options options = parse_options(argc, argv);
        const auto denominator = read_normalization_denominator(options.ptr_meta_path,
                                                                 options.rows);
        ReadOnlyMap<std::uint64_t> fingerprints(options.fp_path, options.rows * kWords);
        ReadOnlyMap<std::uint8_t> ptr(options.ptr_path, options.rows * kInput);
        ReadOnlyMap<std::uint32_t> initial_order(options.order_path, options.rows);
        std::vector<std::uint32_t> current(initial_order.data(),
                                           initial_order.data() + options.rows);
        auto groups = initial_groups(options);
        validate_coverage(current, groups, options.rows);

        Trainer trainer(options, fingerprints, ptr, denominator);
        for(unsigned source_level = options.initial_level;
            source_level < options.target_level; ++source_level) {
            std::vector<std::uint32_t> next(options.rows);
            std::vector<SplitResult> results(groups.size());
            std::atomic<std::size_t> next_group{0};
            std::atomic<bool> cancelled{false};
            std::mutex error_mutex;
            std::exception_ptr worker_error;
            auto worker = [&]() {
                try {
                    while(!cancelled.load(std::memory_order_relaxed)) {
                        const auto index = next_group.fetch_add(1);
                        if(index >= groups.size()) break;
                        results[index] = trainer.split(groups[index], source_level, current, next);
                    }
                } catch(...) {
                    cancelled.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if(!worker_error) worker_error = std::current_exception();
                }
            };
            std::vector<std::thread> threads;
            threads.reserve(options.workers);
            for(unsigned index = 0; index < options.workers; ++index) threads.emplace_back(worker);
            for(auto &thread : threads) thread.join();
            if(worker_error) std::rethrow_exception(worker_error);

            std::vector<Group> next_groups;
            next_groups.reserve(groups.size() * 2);
            for(std::size_t index = 0; index < groups.size(); ++index) {
                const auto &group = groups[index];
                const auto &result = results[index];
                if(!result.split) {
                    next_groups.push_back({group.begin, group.end, group.lineage});
                } else {
                    const auto middle = group.begin + result.left_size;
                    next_groups.push_back({group.begin, middle, group.lineage << 1U});
                    next_groups.push_back({middle, group.end, (group.lineage << 1U) | 1U});
                }
            }
            current.swap(next);
            groups.swap(next_groups);
            validate_coverage(current, groups, options.rows);
            const unsigned completed_level = source_level + 1;
            std::uint64_t fallback_count = 0;
            std::uint64_t trained_count = 0;
            for(const auto &result : results) {
                fallback_count += result.fallback;
                trained_count += result.split;
            }
            std::cout << "PASS level=" << completed_level
                      << " groups=" << groups.size()
                      << " models_trained=" << trained_count
                      << " collapse_fallbacks=" << fallback_count << '\n';
            if(completed_level >= options.emit_from_level) {
                emit_level(options, completed_level, current, groups, results, source_level);
            }
        }
        std::cout << "ALL_L2P_TRAINING_GATES_PASS target_level=" << options.target_level
                  << " groups=" << groups.size() << " rows=" << options.rows << '\n';
        return 0;
    } catch(const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
