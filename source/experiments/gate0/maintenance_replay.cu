// New sequential maintenance harness; the frozen query kernel is shared by all arms.
#include "../../vendor/tide/frozen_query_runtime.cuh"
#include <map>
#include <set>
#include <sys/resource.h>

namespace {
using Clock = std::chrono::steady_clock;
using DB = OwnedHostDB<4>;
using View = HostDBView<4>;
using Q = Query<4>;
constexpr std::uint64_t BUDGET = 256ull << 20;
constexpr std::uint64_t RUNTIME_BYTES = sizeof(DeviceQuery<4>) + MAX_RUNS * sizeof(DeviceSlice)
    + OUTPUT_CAPACITY * sizeof(DeviceHit) + 2 * sizeof(std::uint32_t);
std::uint64_t live_bytes = RUNTIME_BYTES, peak_bytes = RUNTIME_BYTES;

double elapsed(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

struct Resident {
  std::shared_ptr<DB> host;
  std::shared_ptr<DeviceRun<4>> device;
  std::uint64_t bytes;
  double upload_ms;
  explicit Resident(std::shared_ptr<DB> data) : host(std::move(data)), bytes(host->ids.size() * 42) {
    if (live_bytes + bytes > BUDGET) throw std::runtime_error("tracked VRAM budget exceeded before allocation");
    const auto start = Clock::now();
    device = std::make_shared<DeviceRun<4>>(0, host->view());
    upload_ms = elapsed(start);
    live_bytes += bytes;
    peak_bytes = std::max(peak_bytes, live_bytes);
  }
  ~Resident() { device.reset(); live_bytes -= bytes; }
  Resident(const Resident &) = delete;
};

struct OwnedEpoch {
  int epoch = 0;
  std::vector<std::shared_ptr<Resident>> runs;
  Snapshot<4> snapshot() const {
    Snapshot<4> result;
    for (const auto &run : runs) result.runs.push_back({run->device, run->host->view()});
    return result;
  }
  std::uint64_t rows() const {
    std::uint64_t count = 0;
    for (const auto &run : runs) count += run->host->ids.size();
    return count;
  }
};

std::shared_ptr<DB> merge_runs(const std::vector<std::shared_ptr<Resident>> &runs) {
  // Ordinary stable population-count bucket merge: linear host copies, no novel policy.
  std::array<std::uint64_t, 257> counts{}, next{};
  std::uint64_t n = 0;
  for (const auto &run : runs) {
    n += run->host->ids.size();
    for (auto pc : run->host->pc) counts.at(pc)++;
  }
  for (int pc = 1; pc <= 256; ++pc) next[pc] = next[pc - 1] + counts[pc - 1];
  auto out = std::make_shared<DB>();
  out->ids.resize(n); out->fp.resize(n * 4); out->pc.resize(n);
  for (const auto &run : runs) {
    const auto &src = *run->host;
    for (std::size_t row = 0; row < src.ids.size(); ++row) {
      const auto pos = next[src.pc[row]]++;
      out->ids[pos] = src.ids[row]; out->pc[pos] = src.pc[row];
      std::copy_n(src.fp.data() + row * 4, 4, out->fp.data() + pos * 4);
    }
  }
  return out;
}

struct Maintenance {
  double total_ms = 0, host_build_ms = 0, upload_ms = 0;
  std::uint64_t uploaded_bytes = 0, merge_read_write_bytes = 0, peak = 0;
  int merges = 0;
};

std::shared_ptr<OwnedEpoch> publish(const std::shared_ptr<OwnedEpoch> &old,
                                  std::shared_ptr<DB> delta, const std::string &policy,
                                  int epoch, Maintenance &m) {
  const auto start = Clock::now();
  peak_bytes = live_bytes;
  auto next = std::make_shared<OwnedEpoch>();
  next->epoch = epoch;
  if (old) next->runs = old->runs;
  auto added = std::make_shared<Resident>(std::move(delta));
  m.upload_ms += added->upload_ms; m.uploaded_bytes += added->bytes;
  next->runs.push_back(std::move(added));
  auto compact_suffix = [&](std::size_t first) {
    std::vector<std::shared_ptr<Resident>> selected(next->runs.begin() + first, next->runs.end());
    const auto begin_build = Clock::now();
    auto merged_host = merge_runs(selected);
    m.host_build_ms += elapsed(begin_build);
    auto merged = std::make_shared<Resident>(std::move(merged_host));
    m.upload_ms += merged->upload_ms;
    m.uploaded_bytes += merged->bytes;
    m.merge_read_write_bytes += 2 * merged->bytes;
    m.merges++;
    next->runs.erase(next->runs.begin() + first, next->runs.end());
    next->runs.push_back(std::move(merged));
  };
  if (epoch > 0 && (policy == "compact" || (policy == "periodic2" && epoch % 2 == 0))) {
    compact_suffix(0);
  } else if (policy == "size_tiered") {
    while (next->runs.size() >= 2) {
      const auto n = next->runs.size();
      if (next->runs[n - 2]->host->ids.size() > 2 * next->runs[n - 1]->host->ids.size()) break;
      compact_suffix(n - 2);
    }
  }
  // DeviceRun uses blocking H2D completion; publication happens only after it returns.
  m.total_ms = elapsed(start); m.peak = peak_bytes;
  return next;
}

std::vector<std::shared_ptr<DB>> load_history(const std::string &root) {
  std::vector<std::shared_ptr<DB>> runs;
  std::set<std::uint64_t> ids;
  for (int epoch = 0; epoch < 7; ++epoch) {
    MappedFile file(root + "/run" + std::to_string(epoch) + ".bin");
    auto run = std::make_shared<DB>(load_interleaved<4>(file));
    if (run->ids.empty()) throw std::runtime_error("empty sampled arrival cohort");
    for (auto id : run->ids)
      if (!ids.insert(id).second) throw std::runtime_error("duplicate ID in cumulative history");
    runs.push_back(std::move(run));
  }
  return runs;
}

std::vector<HostHit> independent_scan(const std::vector<std::shared_ptr<DB>> &runs,
                                    int epoch, const Q &q, int num, int den) {
  // No population-count bound, no stored-popcount arithmetic, no GPU code reuse.
  std::vector<HostHit> result;
  for (int run = 0; run <= epoch; ++run) {
    const auto &db = *runs[run];
    for (std::size_t row = 0; row < db.ids.size(); ++row) {
      unsigned intersection = 0, unions = 0;
      for (int word = 0; word < 4; ++word) {
        intersection += __builtin_popcountll(q.fp[word] & db.fp[row * 4 + word]);
        unions += __builtin_popcountll(q.fp[word] | db.fp[row * 4 + word]);
      }
      if (den * intersection >= num * unions)
        result.push_back({db.ids[row], static_cast<std::uint16_t>(intersection),
                          static_cast<std::uint16_t>(unions)});
    }
  }
  sort_hits(result);
  return result;
}

bool equal_hits(const std::vector<HostHit> &a, const std::vector<HostHit> &b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i].id != b[i].id || a[i].intersection != b[i].intersection || a[i].union_count != b[i].union_count)
      return false;
  return true;
}

std::string oracle_path(const std::string &root, int epoch, int threshold, int q) {
  return root + "/oracle/e" + std::to_string(epoch) + "_t" + std::to_string(threshold)
      + "_q" + std::to_string(q) + ".bin";
}

void save_oracle(const std::string &path, const std::vector<HostHit> &hits) {
  if (std::filesystem::exists(path)) throw std::runtime_error("refusing existing oracle file");
  std::ofstream out(path, std::ios::binary);
  for (const auto &hit : hits) {
    DeviceHit record{hit.id, hit.intersection, hit.union_count, 0};
    out.write(reinterpret_cast<const char *>(&record), sizeof(record));
  }
  if (!out) throw std::runtime_error("oracle write failed");
}

std::vector<HostHit> read_oracle(const std::string &path) {
  MappedFile file(path);
  std::vector<HostHit> result;
  if (!file.bytes()) return result;
  const auto *records = file.as<DeviceHit>();
  for (std::size_t i = 0; i < file.bytes() / sizeof(DeviceHit); ++i)
    result.push_back({records[i].id, records[i].intersection, records[i].union_count});
  return result;
}

RunResult query(QueryRuntime<4> &runtime, std::shared_ptr<const OwnedEpoch> acquired,
                const Q &q, int num, int den) {
  const auto snapshot = acquired->snapshot();
  return runtime.run(snapshot, q, num, den, true, false);
}

std::vector<std::string> order(int rotation) {
  std::vector<std::string> policies{"all_delta", "periodic2", "size_tiered", "compact"};
  std::rotate(policies.begin(), policies.begin() + rotation, policies.end());
  return policies;
}

void run(const std::string &root, const std::string &output, const std::string &mode, int rotation) {
  const auto load_start = Clock::now();
  auto history = load_history(root);
  MappedFile qfile(root + "/queries.bin");
  QueryStore<4> queries(qfile);
  if (queries.size() != 32) throw std::runtime_error("exactly 32 queries required");
  for (int q = 0; q < 32; ++q) if (!queries[q].pc) throw std::runtime_error("empty query rejected");
  const double load_ms = elapsed(load_start);
  if (mode == "oracle") {
    if (!std::filesystem::create_directory(root + "/oracle"))
      throw std::runtime_error("oracle directory exists; do not overwrite prior evidence");
    const auto begin = Clock::now();
    std::uint64_t records = 0;
    for (int epoch = 0; epoch < 7; ++epoch)
      for (int threshold : {70, 80}) for (int q = 0; q < 32; ++q) {
        auto hits = independent_scan(history, epoch, queries[q], threshold, 100);
        records += hits.size();
        save_oracle(oracle_path(root, epoch, threshold, q), hits);
      }
    std::cout << "oracle_requests=448 records=" << records << " load_ms=" << load_ms
              << " oracle_ms=" << elapsed(begin) << std::endl;
    return;
  }
  using Key = std::tuple<int, int, int>;
  std::map<Key, std::vector<HostHit>> oracles;
  for (int epoch = 0; epoch < 7; ++epoch)
    for (int t : {70, 80}) for (int q = 0; q < 32; ++q)
      oracles[{epoch, t, q}] = read_oracle(oracle_path(root, epoch, t, q));
  CUDA_CHECK(cudaSetDevice(0));
  QueryRuntime<4> runtime(0);
  std::ofstream out(output + ".queries.csv"), maint(output + ".maintenance.csv"), warm(output + ".warmup.csv");
  if (!out || !maint || !warm) throw std::runtime_error("cannot create result files");
  out << "policy,epoch,threshold,query,rows,runs,service_ms,kernel_ms,candidate_rows,scan_fp_pc_bytes,"
         "tail_threads,active_slices,descriptor_bytes,hits,output_bytes,result_hash,cpu_complete_match,"
         "oracle_compare_ms,tracked_live_device_bytes,observed_hits,overflow\n";
  warm << "policy,epoch,threshold,query,service_ms,kernel_ms,hits,overflow,cpu_complete_match\n";
  maint << "policy,epoch,rows,runs,merges,total_ms,host_build_ms,upload_ms,uploaded_bytes,"
           "merge_host_read_write_bytes,tracked_peak_device_bytes,old_epoch_logical_bytes,"
           "cuda_observed_used_bytes,host_maxrss_kib,old_epoch_complete_checks,post_release_device_bytes\n";
  std::uint64_t checked = 0, old_checked = 0;
  for (const auto &policy : order(rotation)) {
    std::shared_ptr<OwnedEpoch> current;
    std::uint64_t expected_rows = 0;
    for (int epoch = 0; epoch < 7; ++epoch) {
      auto old = current;
      Maintenance m;
      current = publish(old, history[epoch], policy, epoch, m);
      expected_rows += history[epoch]->ids.size();
      if (current->rows() != expected_rows) throw std::runtime_error("cumulative row balance mismatch");
      int old_checks = 0;
      if (old) for (int t : {70, 80}) {
        const auto r = query(runtime, old, queries[0], t, 100);
        if (r.overflow || !equal_hits(r.hits, oracles.at({epoch - 1, t, 0})))
          throw std::runtime_error("held old epoch changed after publication");
        old_checks++; old_checked++;
      }
      std::size_t free = 0, total = 0;
      CUDA_CHECK(cudaMemGetInfo(&free, &total));
      struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);
      const auto old_logical = old ? old->rows() * 42 : 0;
      old.reset();
      maint << policy << ',' << epoch << ',' << current->rows() << ',' << current->runs.size() << ','
            << m.merges << ',' << std::setprecision(17) << m.total_ms << ',' << m.host_build_ms << ','
            << m.upload_ms << ',' << m.uploaded_bytes << ',' << m.merge_read_write_bytes << ','
            << m.peak << ',' << old_logical << ',' << total - free << ',' << usage.ru_maxrss << ','
            << old_checks << ',' << live_bytes << '\n';
      for (int t : {70, 80}) {
        if (mode == "bench") for (int w = 0; w < 8; ++w) {
          const auto r = query(runtime, current, queries[w], t, 100);
          const bool match = equal_hits(r.hits, oracles.at({epoch, t, w}));
          warm << policy << ',' << epoch << ',' << t << ',' << w << ',' << std::setprecision(17)
               << r.service_ms << ',' << r.kernel_ms << ',' << r.hits.size() << ',' << r.overflow << ',' << match << '\n';
          if (r.overflow || !match)
            throw std::runtime_error("warmup correctness failed");
        }
        for (int q = 0; q < 32; ++q) {
          const auto r = query(runtime, current, queries[q], t, 100);
          const auto compare_begin = Clock::now();
          const bool match = equal_hits(r.hits, oracles.at({epoch, t, q}));
          const double compare_ms = elapsed(compare_begin);
          out << policy << ',' << epoch << ',' << t << ',' << q << ',' << current->rows() << ','
              << current->runs.size() << ',' << std::setprecision(17) << r.service_ms << ',' << r.kernel_ms << ','
              << r.candidate_rows << ',' << r.candidate_rows * 34 << ','
              << r.launched_blocks * BLOCK - r.candidate_rows << ',' << r.active_slices << ','
              << r.active_slices * sizeof(DeviceSlice) << ',' << r.hits.size() << ','
              << r.hits.size() * sizeof(DeviceHit) << ',' << r.result_hash << ',' << match << ','
              << compare_ms << ',' << live_bytes << ',' << r.observed_hits << ',' << r.overflow << '\n';
          if (r.overflow || !match) {
            out.flush(); maint.flush();
            throw std::runtime_error("overflow or full CPU-vector equality failure");
          }
          checked++;
        }
      }
      out.flush(); maint.flush();
    }
    current.reset();
    if (live_bytes != RUNTIME_BYTES) throw std::runtime_error("resident accounting leak after policy");
  }
  if (!out || !maint || !warm) throw std::runtime_error("result write failed");
  std::cout << "completed_requests=" << checked << " old_epoch_checks=" << old_checked
            << " load_ms=" << load_ms << " tracked_budget_bytes=" << BUDGET << std::endl;
}

void guard_tests() {
  CUDA_CHECK(cudaSetDevice(0));
  QueryRuntime<4> runtime(0);
  auto data = std::make_shared<DB>();
  const std::size_t rows = OUTPUT_CAPACITY + 1;
  data->ids.resize(rows); data->fp.assign(rows * 4, 0); data->pc.assign(rows, 1);
  for (std::size_t i = 0; i < rows; ++i) { data->ids[i] = i; data->fp[i * 4] = 1; }
  auto epoch = std::make_shared<OwnedEpoch>();
  epoch->runs.push_back(std::make_shared<Resident>(data));
  Q q; q.fp[0] = 1; q.pc = 1;
  const auto overflow = query(runtime, epoch, q, 1, 1);
  if (!overflow.overflow || overflow.observed_hits != rows || !overflow.hits.empty())
    throw std::runtime_error("overflow must fail explicitly, never return a truncated vector");
  q.fp[0] = 3; q.pc = 2;
  const auto empty = query(runtime, epoch, q, 1, 1);
  if (empty.overflow || !empty.hits.empty() || empty.candidate_rows != 0)
    throw std::runtime_error("empty bounded slice is incorrect");
  std::cout << "overflow_guard_pass=1 empty_slice_pass=1 requested_hits=" << rows << std::endl;
}
}  // namespace

int main(int argc, char **argv) {
  try {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) throw std::runtime_error("missing argument value");
      const std::string key(argv[i]);
      if (key != "--root" && key != "--output" && key != "--mode" && key != "--rotation")
        throw std::runtime_error("unknown argument");
      if (!args.emplace(key, argv[i + 1]).second) throw std::runtime_error("duplicate argument");
    }
    const auto mode = args.at("--mode");
    if (mode == "guards") { guard_tests(); return 0; }
    if (mode != "oracle" && mode != "check" && mode != "bench") throw std::runtime_error("invalid mode");
    const int rotation = args.count("--rotation") ? std::stoi(args.at("--rotation")) : 0;
    if (rotation < 0 || rotation > 3) throw std::runtime_error("rotation out of range");
    const auto output = mode == "oracle" ? "" : args.at("--output");
    if (!output.empty() && (std::filesystem::exists(output + ".queries.csv") ||
                          std::filesystem::exists(output + ".maintenance.csv")))
      throw std::runtime_error("refusing existing output prefix");
    run(args.at("--root"), output, mode, rotation);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << std::endl;
    return 2;
  }
}
