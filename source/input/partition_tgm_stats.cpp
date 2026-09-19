#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kWords = 4;
constexpr std::size_t kTokens = 256;

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

template <typename T>
class ReadOnlyMap {
public:
    ReadOnlyMap(const fs::path &path, std::uint64_t count) : count_(count) {
        if(exact_file_size(path) != count * sizeof(T)) {
            fail("byte size mismatch for " + path.string());
        }
        fd_ = ::open(path.c_str(), O_RDONLY);
        if(fd_ < 0) fail("cannot open " + path.string() + ": " + std::strerror(errno));
        if(count != 0) {
            data_ = static_cast<const T *>(
                ::mmap(nullptr, static_cast<std::size_t>(count * sizeof(T)), PROT_READ,
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
        if(index >= count_) fail("mapped access is out of bounds");
        return data_[index];
    }
    const T *data() const { return data_; }

private:
    int fd_ = -1;
    const T *data_ = nullptr;
    std::uint64_t count_ = 0;
};

void rename_exact(const fs::path &source, const fs::path &destination) {
    std::error_code error;
    fs::rename(source, destination, error);
    if(error) fail("cannot rename " + source.string() + " to " +
                   destination.string() + ": " + error.message());
}

}  // namespace

int main(int argc, char **argv) {
    try {
        if(argc != 7 || std::string(argv[5]) != "--rows") {
            std::cerr << "usage: partition_tgm_stats FP_U64X4 ORDER_U32 OFFSETS_U64 "
                         "OUTPUT_PREFIX --rows ROWS\n";
            return 2;
        }
        if constexpr(std::endian::native != std::endian::little) {
            fail("only little-endian hosts are supported");
        }
        const fs::path fp_path = argv[1];
        const fs::path order_path = argv[2];
        const fs::path offsets_path = argv[3];
        const fs::path prefix = argv[4];
        const auto rows = std::stoull(argv[6]);
        if(rows == 0 || rows > std::numeric_limits<std::uint32_t>::max()) {
            fail("row count must be in [1, 2^32-1]");
        }
        const auto offsets_bytes = exact_file_size(offsets_path);
        if(offsets_bytes < 2 * sizeof(std::uint64_t) ||
           offsets_bytes % sizeof(std::uint64_t) != 0) {
            fail("invalid offsets file size");
        }
        const auto offsets_count = offsets_bytes / sizeof(std::uint64_t);
        const auto groups = offsets_count - 1;
        ReadOnlyMap<std::uint64_t> fingerprints(fp_path, rows * kWords);
        ReadOnlyMap<std::uint32_t> order(order_path, rows);
        ReadOnlyMap<std::uint64_t> offsets(offsets_path, offsets_count);
        if(offsets[0] != 0 || offsets[offsets_count - 1] != rows) {
            fail("offsets do not span all rows");
        }

        const fs::path groups_path = prefix.string() + ".groups.tsv";
        const fs::path summary_path = prefix.string() + ".summary.txt";
        require_absent(groups_path);
        require_absent(summary_path);
        if(!prefix.parent_path().empty()) fs::create_directories(prefix.parent_path());
        const std::string suffix = ".tmp." + std::to_string(::getpid());
        const fs::path groups_temp = groups_path.string() + suffix;
        const fs::path summary_temp = summary_path.string() + suffix;

        std::ofstream group_output(groups_temp, std::ios::trunc);
        if(!group_output) fail("cannot create group-statistics output");
        group_output << "group\trows\tunion_popcount\tword0\tword1\tword2\tword3\n";
        std::array<std::uint64_t, kTokens> token_group_presence{};
        std::uint64_t fully_saturated = 0;
        std::uint64_t min_union = kTokens;
        std::uint64_t max_union = 0;
        std::uint64_t min_group_size = rows;
        std::uint64_t max_group_size = 0;
        std::vector<std::uint8_t> seen((rows + 7) / 8, 0);

        for(std::uint64_t group = 0; group < groups; ++group) {
            const auto begin = offsets[group];
            const auto end = offsets[group + 1];
            if(begin >= end) {
                fail("empty or invalid group offsets");
            }
            std::array<std::uint64_t, kWords> union_words{};
            for(auto position = begin; position < end; ++position) {
                const auto row = order[position];
                if(row >= rows) fail("partition contains an out-of-range row");
                const auto byte = row >> 3U;
                const auto mask = static_cast<std::uint8_t>(1U << (row & 7U));
                if((seen[byte] & mask) != 0) fail("partition contains a duplicate row");
                seen[byte] |= mask;
                for(std::size_t word = 0; word < kWords; ++word) {
                    union_words[word] |= fingerprints[row * kWords + word];
                }
            }
            unsigned union_popcount = 0;
            for(std::size_t word = 0; word < kWords; ++word) {
                union_popcount += static_cast<unsigned>(std::popcount(union_words[word]));
                auto bits = union_words[word];
                while(bits != 0) {
                    const auto bit = static_cast<unsigned>(std::countr_zero(bits));
                    ++token_group_presence[word * 64 + bit];
                    bits &= bits - 1;
                }
            }
            const auto group_size = end - begin;
            fully_saturated += union_popcount == kTokens;
            min_union = std::min<std::uint64_t>(min_union, union_popcount);
            max_union = std::max<std::uint64_t>(max_union, union_popcount);
            min_group_size = std::min(min_group_size, group_size);
            max_group_size = std::max(max_group_size, group_size);
            group_output << group << '\t' << group_size << '\t' << union_popcount;
            for(const auto word : union_words) group_output << '\t' << std::hex << word << std::dec;
            group_output << '\n';
        }
        if(!group_output) fail("failed to write group statistics");
        group_output.close();
        for(std::uint64_t row = 0; row < rows; ++row) {
            if((seen[row >> 3U] & static_cast<std::uint8_t>(1U << (row & 7U))) == 0) {
                fail("partition is missing a row");
            }
        }
        const auto min_token_presence =
            *std::min_element(token_group_presence.begin(), token_group_presence.end());
        const auto max_token_presence =
            *std::max_element(token_group_presence.begin(), token_group_presence.end());
        std::uint64_t universal_tokens = 0;
        for(const auto count : token_group_presence) universal_tokens += count == groups;

        {
            std::ofstream summary(summary_temp, std::ios::trunc);
            if(!summary) fail("cannot create TGM summary");
            summary << "format=partition_tgm_stats_v1\n"
                    << "rows=" << rows << "\n"
                    << "groups=" << groups << "\n"
                    << "min_group_size=" << min_group_size << "\n"
                    << "max_group_size=" << max_group_size << "\n"
                    << "min_union_popcount=" << min_union << "\n"
                    << "max_union_popcount=" << max_union << "\n"
                    << "fully_saturated_groups=" << fully_saturated << "\n"
                    << "fully_saturated_group_fraction=" << std::setprecision(17)
                    << static_cast<double>(fully_saturated) / groups << "\n"
                    << "min_token_group_presence=" << min_token_presence << "\n"
                    << "max_token_group_presence=" << max_token_presence << "\n"
                    << "universal_tokens=" << universal_tokens << "\n"
                    << "token_group_presence=";
            for(std::size_t token = 0; token < kTokens; ++token) {
                if(token != 0) summary << ',';
                summary << token_group_presence[token];
            }
            summary << '\n';
            if(!summary) fail("failed to write TGM summary");
        }
        rename_exact(groups_temp, groups_path);
        rename_exact(summary_temp, summary_path);
        std::cout << "PASS rows=" << rows << " groups=" << groups
                  << " min_union_popcount=" << min_union
                  << " max_union_popcount=" << max_union
                  << " fully_saturated_groups=" << fully_saturated
                  << " universal_tokens=" << universal_tokens << '\n';
        return 0;
    } catch(const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
