#include <array>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kWords = 4;
constexpr std::size_t kTokens = 256;
constexpr std::size_t kPtrDims = 16;

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

void read_exact(std::ifstream &input, char *data, std::size_t size,
                const std::string &label) {
    input.read(data, static_cast<std::streamsize>(size));
    if(input.gcount() != static_cast<std::streamsize>(size)) {
        fail("short read from " + label);
    }
}

std::uint16_t row_popcount(const std::array<std::uint64_t, kWords> &words) {
    unsigned count = 0;
    for(const auto word : words) count += static_cast<unsigned>(std::popcount(word));
    if(count > std::numeric_limits<std::uint16_t>::max()) fail("popcount overflow");
    return static_cast<std::uint16_t>(count);
}

std::uint8_t minimum_token(const std::array<std::uint64_t, kWords> &words) {
    for(unsigned word_index = 0; word_index < kWords; ++word_index) {
        const auto word = words[word_index];
        if(word != 0) {
            return static_cast<std::uint8_t>(
                word_index * 64 + static_cast<unsigned>(std::countr_zero(word)));
        }
    }
    fail("empty set is outside the frozen L2P contract");
}

std::array<std::uint8_t, kPtrDims>
ptr_counts(const std::array<std::uint64_t, kWords> &words) {
    std::array<std::uint8_t, kPtrDims> result{};
    for(unsigned word_index = 0; word_index < kWords; ++word_index) {
        auto word = words[word_index];
        while(word != 0) {
            const unsigned bit = static_cast<unsigned>(std::countr_zero(word));
            const unsigned token = word_index * 64 + bit;
            for(unsigned depth = 0; depth < 8; ++depth) {
                const unsigned branch = (token >> (7U - depth)) & 1U;
                auto &entry = result[2U * depth + branch];
                if(entry == std::numeric_limits<std::uint8_t>::max()) {
                    fail("PTR coordinate overflow");
                }
                ++entry;
            }
            word &= word - 1;
        }
    }
    return result;
}

class MappedU32Output {
public:
    MappedU32Output(const fs::path &path, std::uint64_t count) : count_(count) {
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        if(fd_ < 0) fail("cannot create " + path.string() + ": " + std::strerror(errno));
        const std::uint64_t bytes = count * sizeof(std::uint32_t);
        if(bytes > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
            fail("permutation file is too large");
        }
        if(::ftruncate(fd_, static_cast<off_t>(bytes)) != 0) {
            fail("cannot size " + path.string() + ": " + std::strerror(errno));
        }
        if(bytes != 0) {
            data_ = static_cast<std::uint32_t *>(
                ::mmap(nullptr, static_cast<std::size_t>(bytes), PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd_, 0));
            if(data_ == MAP_FAILED) {
                data_ = nullptr;
                fail("cannot mmap " + path.string() + ": " + std::strerror(errno));
            }
        }
    }

    MappedU32Output(const MappedU32Output &) = delete;
    MappedU32Output &operator=(const MappedU32Output &) = delete;

    ~MappedU32Output() {
        const std::uint64_t bytes = count_ * sizeof(std::uint32_t);
        if(data_ != nullptr) {
            ::msync(data_, static_cast<std::size_t>(bytes), MS_SYNC);
            ::munmap(data_, static_cast<std::size_t>(bytes));
        }
        if(fd_ >= 0) ::close(fd_);
    }

    std::uint32_t &operator[](std::uint64_t index) {
        if(index >= count_) fail("permutation write is out of bounds");
        return data_[index];
    }

private:
    int fd_ = -1;
    std::uint32_t *data_ = nullptr;
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
        if(argc != 6 || std::string(argv[4]) != "--expected-rows") {
            std::cerr << "usage: l2p_build_ptr FP_U64X4 POPCNT_U16 OUTPUT_PREFIX "
                         "--expected-rows ROWS\n";
            return 2;
        }
        if constexpr(std::endian::native != std::endian::little) {
            fail("only little-endian hosts are supported");
        }

        const fs::path fp_path = argv[1];
        const fs::path pop_path = argv[2];
        const fs::path prefix = argv[3];
        const std::uint64_t rows = std::stoull(argv[5]);
        if(rows == 0 || rows > std::numeric_limits<std::uint32_t>::max()) {
            fail("row count must be in [1, 2^32-1]");
        }
        if(exact_file_size(fp_path) != rows * kWords * sizeof(std::uint64_t)) {
            fail("fingerprint byte size does not match the expected row count");
        }
        if(exact_file_size(pop_path) != rows * sizeof(std::uint16_t)) {
            fail("popcount byte size does not match the expected row count");
        }

        const fs::path ptr_path = prefix.string() + ".ptr_u8x16.bin";
        const fs::path perm_path = prefix.string() + ".min_token_order_u32.bin";
        const fs::path meta_path = prefix.string() + ".ptr_meta.txt";
        require_absent(ptr_path);
        require_absent(perm_path);
        require_absent(meta_path);
        if(!prefix.parent_path().empty()) fs::create_directories(prefix.parent_path());

        const std::string suffix = ".tmp." + std::to_string(::getpid());
        const fs::path ptr_temp = ptr_path.string() + suffix;
        const fs::path perm_temp = perm_path.string() + suffix;
        const fs::path meta_temp = meta_path.string() + suffix;
        require_absent(ptr_temp);
        require_absent(perm_temp);
        require_absent(meta_temp);

        std::array<std::uint64_t, kTokens> min_counts{};
        std::uint8_t max_ptr_count = 0;
        std::uint64_t popcount_sum = 0;
        std::uint16_t min_popcount = std::numeric_limits<std::uint16_t>::max();
        std::uint16_t max_popcount = 0;

        {
            std::ifstream fp(fp_path, std::ios::binary);
            std::ifstream pop(pop_path, std::ios::binary);
            std::ofstream ptr(ptr_temp, std::ios::binary | std::ios::trunc);
            if(!fp || !pop || !ptr) fail("cannot open first-pass input or output");
            for(std::uint64_t row = 0; row < rows; ++row) {
                std::array<std::uint64_t, kWords> words{};
                std::uint16_t stored_popcount = 0;
                read_exact(fp, reinterpret_cast<char *>(words.data()), sizeof(words),
                           fp_path.string());
                read_exact(pop, reinterpret_cast<char *>(&stored_popcount),
                           sizeof(stored_popcount), pop_path.string());
                const auto observed_popcount = row_popcount(words);
                if(observed_popcount != stored_popcount) {
                    fail("stored popcount mismatch at row " + std::to_string(row));
                }
                const auto counts = ptr_counts(words);
                for(const auto count : counts) max_ptr_count = std::max(max_ptr_count, count);
                ptr.write(reinterpret_cast<const char *>(counts.data()),
                          static_cast<std::streamsize>(counts.size()));
                if(!ptr) fail("PTR output write failed");
                ++min_counts[minimum_token(words)];
                popcount_sum += observed_popcount;
                min_popcount = std::min(min_popcount, observed_popcount);
                max_popcount = std::max(max_popcount, observed_popcount);
            }
            if(fp.peek() != std::ifstream::traits_type::eof() ||
               pop.peek() != std::ifstream::traits_type::eof()) {
                fail("input contains trailing bytes");
            }
        }

        std::array<std::uint64_t, kTokens> bucket_offsets{};
        std::array<std::uint64_t, kTokens> bucket_next{};
        std::uint64_t running = 0;
        for(std::size_t token = 0; token < kTokens; ++token) {
            bucket_offsets[token] = running;
            bucket_next[token] = running;
            running += min_counts[token];
        }
        if(running != rows) fail("minimum-token histogram does not cover all rows");

        {
            std::ifstream fp(fp_path, std::ios::binary);
            if(!fp) fail("cannot reopen fingerprint input");
            MappedU32Output permutation(perm_temp, rows);
            for(std::uint64_t row = 0; row < rows; ++row) {
                std::array<std::uint64_t, kWords> words{};
                read_exact(fp, reinterpret_cast<char *>(words.data()), sizeof(words),
                           fp_path.string());
                const auto token = minimum_token(words);
                permutation[bucket_next[token]++] = static_cast<std::uint32_t>(row);
            }
        }
        for(std::size_t token = 0; token < kTokens; ++token) {
            if(bucket_next[token] != bucket_offsets[token] + min_counts[token]) {
                fail("permutation bucket size mismatch");
            }
        }

        {
            std::ofstream meta(meta_temp, std::ios::trunc);
            if(!meta) fail("cannot create PTR metadata");
            meta << "format=l2p_ptr_u8x16_v1\n"
                 << "rows=" << rows << "\n"
                 << "tokens=" << kTokens << "\n"
                 << "ptr_dims=" << kPtrDims << "\n"
                 << "normalization_denominator=" << static_cast<unsigned>(max_ptr_count) << "\n"
                 << "popcount_sum=" << popcount_sum << "\n"
                 << "min_popcount=" << min_popcount << "\n"
                 << "max_popcount=" << max_popcount << "\n"
                 << "minimum_token_counts=";
            for(std::size_t token = 0; token < kTokens; ++token) {
                if(token != 0) meta << ',';
                meta << min_counts[token];
            }
            meta << '\n';
            if(!meta) fail("PTR metadata write failed");
        }

        rename_exact(ptr_temp, ptr_path);
        rename_exact(perm_temp, perm_path);
        rename_exact(meta_temp, meta_path);

        std::cout << "PASS rows=" << rows
                  << " ptr_dims=" << kPtrDims
                  << " normalization_denominator=" << static_cast<unsigned>(max_ptr_count)
                  << " popcount_sum=" << popcount_sum
                  << " min_popcount=" << min_popcount
                  << " max_popcount=" << max_popcount << '\n';
        std::cout << "ptr=" << ptr_path << '\n'
                  << "permutation=" << perm_path << '\n'
                  << "metadata=" << meta_path << '\n';
        return 0;
    } catch(const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
