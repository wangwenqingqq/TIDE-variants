// Gate 0 native-tree comparison for exact Levenshtein range search.
//
// This intentionally reuses the same archived C2-off/static GTS host that was
// frozen for TIDE-G1.  It builds once, warms persistent workspaces, and then
// compares repeated native traversal against an exact two-row GPU scan.  The
// scan materializes one byte per object/query so an independent result-set
// oracle can reject count-preserving omissions/duplicates.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "update.cuh"

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kMaxWordLen = 40;
constexpr int kMaxQueries = 32;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "FAIL: " << message << "\n";
  std::exit(2);
}

void cuda_ok(cudaError_t status, const char *where) {
  if (status != cudaSuccess) {
    std::ostringstream out;
    out << where << ": " << cudaGetErrorString(status);
    fail(out.str());
  }
}

double mean(const std::vector<double> &values) {
  return values.empty() ? 0.0
                        : std::accumulate(values.begin(), values.end(), 0.0) /
                              values.size();
}

double percentile(std::vector<double> values, double q) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  size_t pos = static_cast<size_t>(std::ceil(q * values.size())) - 1;
  return values[std::min(pos, values.size() - 1)];
}

struct HostStrings {
  int declared_max = 0;
  int metric = -1;
  size_t n = 0;
  std::vector<char> padded;
  std::vector<int> lengths;
};

HostStrings load_strings(const char *path) {
  std::ifstream in(path);
  if (!in) fail(std::string("open dataset: ") + path);
  HostStrings d;
  size_t declared_n = 0;
  in >> d.declared_max >> declared_n >> d.metric;
  std::string line;
  std::getline(in, line);
  if (d.metric != 6 || d.declared_max > kMaxWordLen)
    fail("dataset is not the frozen Words/edit-distance workload");
  d.padded.reserve(declared_n * static_cast<size_t>(M));
  d.lengths.reserve(declared_n);
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.size() > static_cast<size_t>(kMaxWordLen)) fail("word too long");
    size_t base = d.padded.size();
    d.padded.resize(base + M, 0);
    std::copy(line.begin(), line.end(), d.padded.begin() + base);
    d.lengths.push_back(static_cast<int>(line.size()));
  }
  d.n = d.lengths.size();
  if (d.n != declared_n) fail("dataset row count mismatch");
  return d;
}

std::vector<int> load_query_ids(const char *path, size_t n) {
  std::ifstream in(path);
  if (!in) fail(std::string("open query ids: ") + path);
  int count = 0;
  in >> count;
  if (count < kMaxQueries) fail("query file has fewer than 32 IDs");
  std::vector<int> ids(kMaxQueries);
  for (int &id : ids) {
    if (!(in >> id) || id < 0 || static_cast<size_t>(id) >= n)
      fail("invalid query ID");
  }
  return ids;
}

std::vector<int> parse_radii(const std::string &csv) {
  std::vector<int> out;
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) out.push_back(std::stoi(token));
  if (out.empty()) fail("empty radius list");
  return out;
}

__device__ __forceinline__ uint8_t edit_distance_scan(const char *a, int n,
                                                       const char *b, int m) {
  if (m > n) {
    const char *tp = a;
    a = b;
    b = tp;
    int tn = n;
    n = m;
    m = tn;
  }
  uint16_t row0[kMaxWordLen + 1];
  uint16_t row1[kMaxWordLen + 1];
  uint16_t *prev = row0;
  uint16_t *curr = row1;
  for (int j = 0; j <= m; ++j) prev[j] = static_cast<uint16_t>(j);
  for (int i = 1; i <= n; ++i) {
    curr[0] = static_cast<uint16_t>(i);
    for (int j = 1; j <= m; ++j) {
      uint16_t del = static_cast<uint16_t>(prev[j] + 1);
      uint16_t ins = static_cast<uint16_t>(curr[j - 1] + 1);
      uint16_t sub = static_cast<uint16_t>(prev[j - 1] +
                                           (a[i - 1] == b[j - 1] ? 0 : 1));
      uint16_t best = del < ins ? del : ins;
      curr[j] = best < sub ? best : sub;
    }
    uint16_t *tmp = prev;
    prev = curr;
    curr = tmp;
  }
  return static_cast<uint8_t>(prev[m]);
}

__global__ void dense_flags_kernel(const char *data, const int *lengths,
                                   size_t n, const int *query_ids, int nq,
                                   int radius, uint8_t *flags) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = n * static_cast<size_t>(nq);
  if (idx >= total) return;
  int q = static_cast<int>(idx / n);
  size_t obj = idx - static_cast<size_t>(q) * n;
  int qid = query_ids[q];
  flags[idx] = edit_distance_scan(data + obj * M, lengths[obj],
                                  data + static_cast<size_t>(qid) * M,
                                  lengths[qid]) <= radius;
}

struct Timing {
  std::vector<double> device_ms;
  std::vector<double> wall_ms;
};

template <typename Launch>
Timing benchmark(Launch launch, int warmup, int repeats) {
  for (int i = 0; i < warmup; ++i) {
    launch();
    cuda_ok(cudaDeviceSynchronize(), "warmup sync");
  }
  cudaEvent_t start = nullptr, stop = nullptr;
  cuda_ok(cudaEventCreate(&start), "create start event");
  cuda_ok(cudaEventCreate(&stop), "create stop event");
  Timing t;
  for (int i = 0; i < repeats; ++i) {
    auto wb = Clock::now();
    cuda_ok(cudaEventRecord(start), "record start");
    launch();
    cuda_ok(cudaEventRecord(stop), "record stop");
    cuda_ok(cudaEventSynchronize(stop), "event sync");
    auto we = Clock::now();
    float ms = 0.0f;
    cuda_ok(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    t.device_ms.push_back(ms);
    t.wall_ms.push_back(
        std::chrono::duration<double, std::milli>(we - wb).count());
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return t;
}

struct CaseResult {
  int radius = 0;
  int batch = 0;
  Timing tree;
  Timing scan;
  int oracle_set_mismatches = 0;
  std::vector<int> result_counts;
  uint64_t selected_leaf_slots = 0;
};

CaseResult run_case(int radius, int nq, int repeats, size_t n, char *data_s,
                    int *size_s, int *data_info, TN *node_list, int *id_list,
                    int *max_node_num, int tree_h, int *&empty_list,
                    int *query_ids, int *&qresult_count,
                    int *&qresult_prefix, int *&result_id,
                    float *&result_dis, uint8_t *dense_flags) {
  auto tree_launch = [&] {
    searchIndexRnnUpdate(nullptr, node_list, id_list, max_node_num, query_ids,
                         nq, static_cast<float>(radius), tree_h, data_info,
                         empty_list, qresult_count, qresult_prefix, result_id,
                         result_dis, data_s, size_s);
  };
  Timing tree_time = benchmark(tree_launch, 3, repeats);
  tree_launch();
  cuda_ok(cudaDeviceSynchronize(), "tree correctness sync");
  uint64_t selected_slots = static_cast<uint64_t>(search_num[0]) * MAX_SIZE;

  size_t total = n * static_cast<size_t>(nq);
  const int threads = 128;
  auto scan_launch = [&] {
    dense_flags_kernel<<<static_cast<unsigned int>((total + threads - 1) / threads),
                         threads>>>(data_s, size_s, n, query_ids, nq, radius,
                                   dense_flags);
  };
  Timing scan_time = benchmark(scan_launch, 3, repeats);
  scan_launch();
  cuda_ok(cudaDeviceSynchronize(), "scan correctness sync");
  std::vector<uint8_t> host_flags(total);
  cuda_ok(cudaMemcpy(host_flags.data(), dense_flags, total,
                     cudaMemcpyDeviceToHost),
          "copy exact flags");

  CaseResult out;
  out.radius = radius;
  out.batch = nq;
  out.tree = std::move(tree_time);
  out.scan = std::move(scan_time);
  out.selected_leaf_slots = selected_slots;
  out.result_counts.resize(nq);
  for (int q = 0; q < nq; ++q) {
    std::vector<int> expected;
    const uint8_t *row = host_flags.data() + static_cast<size_t>(q) * n;
    for (size_t i = 0; i < n; ++i) {
      if (row[i]) expected.push_back(static_cast<int>(i));
    }
    std::vector<int> observed;
    int begin = qresult_prefix[q];
    int count = qresult_count[q];
    observed.assign(result_id + begin, result_id + begin + count);
    std::sort(observed.begin(), observed.end());
    observed.erase(std::unique(observed.begin(), observed.end()), observed.end());
    out.result_counts[q] = static_cast<int>(expected.size());
    if (expected != observed) ++out.oracle_set_mismatches;
  }
  return out;
}

template <typename T>
void print_array(std::ostream &os, const std::vector<T> &v) {
  os << '[';
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) os << ',';
    os << v[i];
  }
  os << ']';
}

void print_timing(std::ostream &os, const Timing &t) {
  os << "{\"device_ms\":";
  print_array(os, t.device_ms);
  os << ",\"wall_ms\":";
  print_array(os, t.wall_ms);
  os << ",\"device_median_ms\":" << percentile(t.device_ms, 0.5)
     << ",\"device_p95_ms\":" << percentile(t.device_ms, 0.95)
     << ",\"wall_median_ms\":" << percentile(t.wall_ms, 0.5)
     << ",\"wall_p95_ms\":" << percentile(t.wall_ms, 0.95) << '}';
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: native_gts_edit_probe WORDS QUERY_IDS RADII_CSV "
                 "REPEATS SUMMARY_JSON EXPECTED_N\n";
    return 2;
  }
  const char *data_path = argv[1];
  const char *query_path = argv[2];
  std::vector<int> radii = parse_radii(argv[3]);
  int repeats = std::atoi(argv[4]);
  const char *summary_path = argv[5];
  size_t expected_n = std::strtoull(argv[6], nullptr, 10);
  if (repeats <= 0) fail("invalid repeats");

  cuda_ok(cudaSetDevice(0), "select visible GPU");
  cudaDeviceProp prop{};
  cuda_ok(cudaGetDeviceProperties(&prop, 0), "properties");
  HostStrings host = load_strings(data_path);
  if (host.n != expected_n) fail("unexpected dataset cardinality");
  std::vector<int> host_query_ids = load_query_ids(query_path, host.n);

  char *data_s = nullptr;
  int *size_s = nullptr;
  int *data_info = nullptr;
  int *id_list = nullptr;
  TN *node_list = nullptr;
  int *max_node_num = nullptr;
  int *empty_list = nullptr;
  int tree_h = 0;
  cuda_ok(cudaMallocManaged(&data_s, host.padded.size()), "allocate strings");
  cuda_ok(cudaMallocManaged(&size_s, host.lengths.size() * sizeof(int)),
          "allocate lengths");
  cuda_ok(cudaMallocManaged(&data_info, 3 * sizeof(int)), "allocate info");
  cuda_ok(cudaMemcpy(data_s, host.padded.data(), host.padded.size(),
                     cudaMemcpyHostToDevice),
          "copy strings");
  cuda_ok(cudaMemcpy(size_s, host.lengths.data(),
                     host.lengths.size() * sizeof(int), cudaMemcpyHostToDevice),
          "copy lengths");
  data_info[0] = host.declared_max;
  data_info[1] = static_cast<int>(host.n);
  data_info[2] = 6;

  cudaEvent_t ib, ie;
  cudaEventCreate(&ib);
  cudaEventCreate(&ie);
  cudaEventRecord(ib);
  indexConstru(nullptr, data_s, size_s, data_info, id_list, node_list,
               max_node_num, tree_h, empty_list);
  cudaEventRecord(ie);
  cudaEventSynchronize(ie);
  float index_build_ms = 0.0f;
  cudaEventElapsedTime(&index_build_ms, ib, ie);
  cudaEventDestroy(ib);
  cudaEventDestroy(ie);
  cuda_ok(cudaGetLastError(), "index build");

  cuda_ok(cudaMallocManaged(&is_delete, host.n * sizeof(int)),
          "allocate delete bitmap");
  cuda_ok(cudaMemset(is_delete, 0, host.n * sizeof(int)), "clear deletes");
  int *query_ids = nullptr;
  cuda_ok(cudaMallocManaged(&query_ids, kMaxQueries * sizeof(int)),
          "allocate queries");
  std::copy(host_query_ids.begin(), host_query_ids.end(), query_ids);
  cuda_ok(cudaDeviceSynchronize(), "publish queries");

  int *qresult_count = nullptr;
  int *qresult_prefix = nullptr;
  int *result_id = nullptr;
  float *result_dis = nullptr;
  uint8_t *dense_flags = nullptr;
  cuda_ok(cudaMalloc(&dense_flags, host.n * kMaxQueries),
          "allocate exact flags");

  std::vector<CaseResult> results;
  for (int radius : radii) {
    results.push_back(run_case(radius, 1, repeats, host.n, data_s, size_s,
                               data_info, node_list, id_list, max_node_num,
                               tree_h, empty_list, query_ids, qresult_count,
                               qresult_prefix, result_id, result_dis,
                               dense_flags));
    results.push_back(run_case(radius, 32, repeats, host.n, data_s, size_s,
                               data_info, node_list, id_list, max_node_num,
                               tree_h, empty_list, query_ids, qresult_count,
                               qresult_prefix, result_id, result_dis,
                               dense_flags));
  }

  int total_mismatches = 0;
  for (const auto &r : results) total_mismatches += r.oracle_set_mismatches;
  std::ofstream out(summary_path);
  if (!out) fail("create summary");
  out << std::setprecision(10)
      << "{\"schema\":\"gts-gate0-native-edit-v1\","
      << "\"dataset\":\"Words\",\"n\":" << host.n
      << ",\"gpu_name\":\"" << prop.name << "\",\"tree_height\":"
      << tree_h << ",\"max_nodes\":" << max_node_num[0]
      << ",\"index_build_ms\":" << index_build_ms
      << ",\"query_ids\":";
  print_array(out, host_query_ids);
  out << ",\"radii\":";
  print_array(out, radii);
  out << ",\"total_oracle_set_mismatches\":" << total_mismatches
      << ",\"results\":[";
  for (size_t i = 0; i < results.size(); ++i) {
    if (i) out << ',';
    const auto &r = results[i];
    out << "{\"radius\":" << r.radius << ",\"batch\":" << r.batch
        << ",\"tree\":";
    print_timing(out, r.tree);
    out << ",\"scan\":";
    print_timing(out, r.scan);
    out << ",\"result_counts\":";
    print_array(out, r.result_counts);
    out << ",\"selected_leaf_slots\":" << r.selected_leaf_slots
        << ",\"oracle_set_mismatches\":" << r.oracle_set_mismatches << '}';
  }
  out << "]}\n";
  out.close();

  std::cout << "NATIVE_EDIT_DONE n=" << host.n << " tree_h=" << tree_h
            << " build_ms=" << index_build_ms
            << " oracle_set_mismatches=" << total_mismatches << "\n";
  return total_mismatches == 0 ? 0 : 3;
}

