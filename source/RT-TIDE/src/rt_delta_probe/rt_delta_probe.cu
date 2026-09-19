#include <cuda_runtime.h>
#include <optix_function_table_definition.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rtspatial/spatial_index.cuh"
#include "rtspatial/utils/queue.h"
#include "rtspatial/utils/shared_value.h"
#include "rtspatial/utils/stream.h"

namespace {
using point_t = rtspatial::Point<float, 2>;
using envelope_t = rtspatial::Envelope<point_t>;
using pair_t = thrust::pair<uint32_t, uint32_t>;

std::vector<float> read_f32(const std::string& path, size_t expected) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  auto bytes = static_cast<size_t>(in.tellg());
  if (bytes != expected * sizeof(float)) {
    throw std::runtime_error("bad byte size for " + path + ": got " +
                             std::to_string(bytes) + ", expected " +
                             std::to_string(expected * sizeof(float)));
  }
  in.seekg(0);
  std::vector<float> out(expected);
  in.read(reinterpret_cast<char*>(out.data()), bytes);
  if (!in) throw std::runtime_error("short read " + path);
  return out;
}

void cuda_check(cudaError_t code, const char* what) {
  if (code != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(code));
  }
}

uint64_t key(uint32_t geom, uint32_t query) {
  return (static_cast<uint64_t>(query) << 32) | geom;
}

std::vector<uint64_t> expected_keys(const std::vector<float>& points,
                                    const std::vector<float>& boxes,
                                    size_t n_points, size_t n_queries,
                                    float point_eps,
                                    const std::vector<uint8_t>& deleted) {
  std::vector<uint64_t> out;
  out.reserve(n_points * n_queries / 2);
  for (uint32_t qi = 0; qi < n_queries; ++qi) {
    const float qminx = boxes[4 * qi + 0];
    const float qminy = boxes[4 * qi + 1];
    const float qmaxx = boxes[4 * qi + 2];
    const float qmaxy = boxes[4 * qi + 3];
    for (uint32_t gi = 0; gi < n_points; ++gi) {
      if (deleted[gi]) continue;
      const float x = points[2 * gi + 0];
      const float y = points[2 * gi + 1];
      const bool hit = (x - point_eps <= qmaxx) && (x + point_eps >= qminx) &&
                       (y - point_eps <= qmaxy) && (y + point_eps >= qminy);
      if (hit) out.push_back(key(gi, qi));
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<uint64_t> actual_keys(rtspatial::Queue<pair_t>& results,
                                  rtspatial::Stream& stream) {
  const size_t n = results.size(stream.cuda_stream());
  std::vector<pair_t> pairs(n);
  if (n) {
    cuda_check(cudaMemcpyAsync(pairs.data(), results.data(), n * sizeof(pair_t),
                               cudaMemcpyDeviceToHost, stream.cuda_stream()),
               "cudaMemcpyAsync(results)");
    stream.Sync();
  }
  std::vector<uint64_t> out;
  out.reserve(n);
  for (const auto& p : pairs) out.push_back(key(p.first, p.second));
  std::sort(out.begin(), out.end());
  return out;
}

struct Comparison {
  bool equal = false;
  size_t expected = 0;
  size_t actual = 0;
  size_t duplicate_actual = 0;
  size_t missing = 0;
  size_t extra = 0;
};

Comparison compare(const std::vector<uint64_t>& expected,
                   const std::vector<uint64_t>& actual) {
  Comparison c;
  c.expected = expected.size();
  c.actual = actual.size();
  c.duplicate_actual = 0;
  for (size_t d = 1; d < actual.size(); ++d) {
    c.duplicate_actual += static_cast<size_t>(actual[d] == actual[d - 1]);
  }
  size_t i = 0, j = 0;
  while (i < expected.size() && j < actual.size()) {
    if (expected[i] == actual[j]) { ++i; ++j; }
    else if (expected[i] < actual[j]) { ++c.missing; ++i; }
    else { ++c.extra; ++j; }
  }
  c.missing += expected.size() - i;
  c.extra += actual.size() - j;
  c.equal = c.missing == 0 && c.extra == 0 && c.duplicate_actual == 0;
  return c;
}

void print_comparison(const char* name, const Comparison& c) {
  std::cout << "\"" << name << "\":{\"equal\":"
            << (c.equal ? "true" : "false")
            << ",\"expected\":" << c.expected
            << ",\"actual\":" << c.actual
            << ",\"missing\":" << c.missing
            << ",\"extra\":" << c.extra
            << ",\"duplicates\":" << c.duplicate_actual << "}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 6 || argc > 9) {
      std::cerr << "usage: rt_delta_probe POINTS QUERY_BOXES N_POINTS N_QUERIES "
                   "BATCH_SIZE [REPEATS=7] [DELETE_STRIDE=100] [POINT_EPS=1e-4]\n";
      return 2;
    }
    const std::string points_path = argv[1];
    const std::string boxes_path = argv[2];
    const size_t n_points = std::stoull(argv[3]);
    const size_t n_queries = std::stoull(argv[4]);
    const size_t batch_size = std::stoull(argv[5]);
    const int repeats = argc >= 7 ? std::stoi(argv[6]) : 7;
    const size_t delete_stride = argc >= 8 ? std::stoull(argv[7]) : 100;
    const float point_eps = argc >= 9 ? std::stof(argv[8]) : 1e-4f;
    if (!n_points || !n_queries || !batch_size || repeats < 1) {
      throw std::runtime_error("all sizes and repeats must be positive");
    }
    if (n_points * n_queries > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("result queue capacity exceeds uint32_t");
    }

    const auto points = read_f32(points_path, n_points * 2);
    const auto boxes = read_f32(boxes_path, n_queries * 4);
    std::vector<envelope_t> h_points;
    h_points.reserve(n_points);
    for (size_t i = 0; i < n_points; ++i) {
      const float x = points[2 * i + 0];
      const float y = points[2 * i + 1];
      h_points.emplace_back(point_t(x - point_eps, y - point_eps),
                            point_t(x + point_eps, y + point_eps));
    }
    std::vector<envelope_t> h_queries;
    h_queries.reserve(n_queries);
    for (size_t i = 0; i < n_queries; ++i) {
      h_queries.emplace_back(point_t(boxes[4 * i + 0], boxes[4 * i + 1]),
                             point_t(boxes[4 * i + 2], boxes[4 * i + 3]));
    }

    thrust::device_vector<envelope_t> d_points = h_points;
    thrust::device_vector<envelope_t> d_queries = h_queries;
    rtspatial::SpatialIndex<float, 2> index;
    rtspatial::Config config;
    config.ptx_root = PTX_ROOT;
    config.max_geometries = n_points;
    config.max_queries = n_queries;
    config.preallocate = true;
    config.prefer_fast_build_geom = false;
    config.prefer_fast_build_query = false;
    rtspatial::Stream stream;

    auto t0 = std::chrono::steady_clock::now();
    index.Init(config);
    stream.Sync();
    auto t1 = std::chrono::steady_clock::now();
    for (size_t begin = 0; begin < n_points; begin += batch_size) {
      const size_t count = std::min(batch_size, n_points - begin);
      index.Insert(rtspatial::ArrayView<envelope_t>(
                       thrust::raw_pointer_cast(d_points.data()) + begin, count),
                   stream.cuda_stream());
      stream.Sync();
    }
    auto t2 = std::chrono::steady_clock::now();

    rtspatial::Queue<pair_t> results;
    results.Init(static_cast<uint32_t>(n_points * n_queries));
    rtspatial::SharedValue<rtspatial::Queue<pair_t>::device_t> d_results;
    d_results.set(stream.cuda_stream(), results.DeviceObject());
    stream.Sync();

    std::vector<double> query_ms;
    for (int rep = 0; rep < repeats; ++rep) {
      results.Clear(stream.cuda_stream());
      stream.Sync();
      auto qs = std::chrono::steady_clock::now();
      index.Query(rtspatial::Predicate::kIntersects,
                  rtspatial::ArrayView<envelope_t>(d_queries), d_results.data(),
                  stream.cuda_stream());
      stream.Sync();
      auto qe = std::chrono::steady_clock::now();
      query_ms.push_back(std::chrono::duration<double, std::milli>(qe - qs).count());
    }
    auto actual_before = actual_keys(results, stream);
    std::vector<uint8_t> deleted(n_points, 0);
    auto expected_before = expected_keys(points, boxes, n_points, n_queries,
                                         point_eps, deleted);
    auto cmp_before = compare(expected_before, actual_before);

    std::vector<size_t> delete_ids;
    if (delete_stride) {
      for (size_t i = 0; i < n_points; i += delete_stride) {
        delete_ids.push_back(i);
        deleted[i] = 1;
      }
    }
    double delete_ms = 0.0;
    Comparison cmp_after;
    if (!delete_ids.empty()) {
      thrust::device_vector<size_t> d_delete = delete_ids;
      auto ds = std::chrono::steady_clock::now();
      index.Delete(rtspatial::ArrayView<size_t>(d_delete), stream.cuda_stream());
      stream.Sync();
      auto de = std::chrono::steady_clock::now();
      delete_ms = std::chrono::duration<double, std::milli>(de - ds).count();
      results.Clear(stream.cuda_stream());
      index.Query(rtspatial::Predicate::kIntersects,
                  rtspatial::ArrayView<envelope_t>(d_queries), d_results.data(),
                  stream.cuda_stream());
      stream.Sync();
      auto actual_after = actual_keys(results, stream);
      auto expected_after = expected_keys(points, boxes, n_points, n_queries,
                                          point_eps, deleted);
      cmp_after = compare(expected_after, actual_after);
    } else {
      cmp_after = cmp_before;
    }

    std::sort(query_ms.begin(), query_ms.end());
    const double qsum = std::accumulate(query_ms.begin(), query_ms.end(), 0.0);
    const double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double insert_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const size_t p95_idx = static_cast<size_t>(std::ceil(0.95 * query_ms.size())) - 1;

    std::cout << std::setprecision(10) << "{";
    std::cout << "\"n_points\":" << n_points
              << ",\"n_queries\":" << n_queries
              << ",\"batch_size\":" << batch_size
              << ",\"repeats\":" << repeats
              << ",\"point_eps\":" << point_eps
              << ",\"init_ms\":" << init_ms
              << ",\"insert_total_ms\":" << insert_ms
              << ",\"query_mean_ms\":" << qsum / query_ms.size()
              << ",\"query_min_ms\":" << query_ms.front()
              << ",\"query_p95_ms\":" << query_ms[p95_idx]
              << ",\"candidate_ratio\":"
              << static_cast<double>(cmp_before.actual) / (n_points * n_queries)
              << ",\"deleted_count\":" << delete_ids.size()
              << ",\"delete_ms\":" << delete_ms << ",";
    print_comparison("before_delete", cmp_before);
    std::cout << ",";
    print_comparison("after_delete", cmp_after);
    std::cout << "}\n";
    return (cmp_before.equal && cmp_after.equal) ? 0 : 3;
  } catch (const std::exception& e) {
    std::cerr << "rt_delta_probe error: " << e.what() << "\n";
    return 1;
  }
}
