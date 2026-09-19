#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kReaders = 4;
constexpr std::uint64_t kPublications = 64;

struct RunDescriptor {
  std::uint64_t epoch;
  std::uint64_t rows;
  std::uint64_t checksum;
};

struct Snapshot {
  std::uint64_t epoch;
  std::vector<RunDescriptor> runs;
};

std::uint64_t descriptor_checksum(std::uint64_t epoch) {
  std::uint64_t value = 1469598103934665603ULL;
  value ^= epoch;
  value *= 1099511628211ULL;
  value ^= 291 + (epoch * 37) % 82;
  value *= 1099511628211ULL;
  return value;
}

bool valid_snapshot(const Snapshot &snapshot) {
  if (snapshot.epoch > kPublications) return false;
  if (snapshot.runs.size() != snapshot.epoch + 1) return false;
  for (std::size_t index = 0; index < snapshot.runs.size(); ++index) {
    const auto &run = snapshot.runs[index];
    if (run.epoch != index) return false;
    if (run.rows != 291 + (index * 37) % 82) return false;
    if (run.checksum != descriptor_checksum(index)) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: gate5_snapshot_tsan OUTPUT");
    auto initial = std::make_shared<Snapshot>();
    initial->epoch = 0;
    initial->runs.push_back(
        RunDescriptor{0, 291, descriptor_checksum(0)});
    std::shared_ptr<Snapshot> active = initial;
    std::atomic<bool> start{false};
    std::atomic<bool> done{false};
    std::atomic<int> ready{0};
    std::atomic<std::uint64_t> captures{0};
    std::atomic<std::uint64_t> mismatches{0};
    std::array<std::atomic<bool>, kPublications + 1> observed;
    for (auto &slot : observed) slot.store(false, std::memory_order_relaxed);

    std::vector<std::thread> readers;
    for (int reader = 0; reader < kReaders; ++reader) {
      readers.emplace_back([&] {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        do {
          const auto snapshot =
              std::atomic_load_explicit(&active, std::memory_order_acquire);
          if (!valid_snapshot(*snapshot))
            mismatches.fetch_add(1, std::memory_order_relaxed);
          observed[snapshot->epoch].store(true, std::memory_order_release);
          captures.fetch_add(1, std::memory_order_relaxed);
        } while (!done.load(std::memory_order_acquire));
      });
    }

    std::thread writer([&] {
      while (ready.load(std::memory_order_acquire) != kReaders)
        std::this_thread::yield();
      start.store(true, std::memory_order_release);
      observed[0].store(true, std::memory_order_release);
      auto current = initial;
      for (std::uint64_t epoch = 1; epoch <= kPublications; ++epoch) {
        auto next = std::make_shared<Snapshot>(*current);
        next->epoch = epoch;
        next->runs.push_back(RunDescriptor{
            epoch, 291 + (epoch * 37) % 82, descriptor_checksum(epoch)});
        std::atomic_store_explicit(&active, next, std::memory_order_release);
        while (!observed[epoch].load(std::memory_order_acquire))
          std::this_thread::yield();
        current = std::move(next);
      }
      while (captures.load(std::memory_order_acquire) < 10000)
        std::this_thread::yield();
      done.store(true, std::memory_order_release);
    });

    for (auto &reader : readers) reader.join();
    writer.join();
    std::uint64_t observed_count = 0;
    for (const auto &slot : observed)
      observed_count += slot.load(std::memory_order_acquire) ? 1 : 0;
    const bool pass = mismatches.load() == 0 &&
                      observed_count == kPublications + 1 &&
                      captures.load() >= 10000;
    std::ofstream output(argv[1]);
    output << "{\n"
           << "  \"reader_threads\": " << kReaders << ",\n"
           << "  \"publications\": " << kPublications << ",\n"
           << "  \"observed_epoch_count\": " << observed_count << ",\n"
           << "  \"snapshot_captures\": " << captures.load() << ",\n"
           << "  \"mismatches\": " << mismatches.load() << ",\n"
           << "  \"G5_SAN_HOST\": " << (pass ? "true" : "false") << "\n"
           << "}\n";
    if (!pass) return 2;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
