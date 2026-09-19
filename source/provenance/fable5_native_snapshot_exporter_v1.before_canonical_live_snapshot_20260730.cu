// Fable5 native E0 frozen-tree snapshot exporter (v1).
//
// This is intentionally separate from fable5_matched_gpu_runner_v1.cu so the
// matched-replay binary and its hash remain stable.  It builds exactly one
// seeded E0 tree, captures the native host snapshot, and writes the metadata a
// CPU-only trace selector needs.  It performs no Safe-C1/buffer-only update
// replay, no dynamic query, no timing, and no C2/C3 operation.
//
// This file is prepared/static-audited only in this change.  It must be built
// and invoked later only through its dedicated guarded execution wrapper.

#define main fable5_archived_g2_entry_not_invoked
#include "safe_c1_dynamic_gts.cu"
#undef main

#include <cctype>
#include <cstring>

namespace fable5_snapshot_exporter {

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("Fable5 native snapshot exporter: " + message);
}

struct Args {
  std::string bundle;
  std::string header_trace;
  std::string output;
  std::string trace_sha256;
  std::string exporter_source_sha256;
  std::string core_source_sha256;
  std::string matched_runner_source_sha256;
  std::string pool_sha256;
  std::string queries_sha256;
  std::string initial_base_sha256;
  std::string stable_map_sha256;
};

bool is_sha256_hex(const std::string& value) {
  return value.size() == 64U && std::all_of(value.begin(), value.end(), [](unsigned char c) {
    return std::isxdigit(c) != 0;
  });
}

void require_sha256_hex(const char* label, const std::string& value) {
  if (!is_sha256_hex(value)) fail(std::string(label) + " must be a 64-character hexadecimal SHA-256");
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string key(argv[index]);
    auto value = [&]() -> std::string {
      if (++index >= argc) fail("missing value after " + key);
      return argv[index];
    };
    if (key == "--bundle") args.bundle = value();
    else if (key == "--header-trace") args.header_trace = value();
    else if (key == "--out") args.output = value();
    else if (key == "--trace-sha256") args.trace_sha256 = value();
    else if (key == "--exporter-source-sha256") args.exporter_source_sha256 = value();
    else if (key == "--core-source-sha256") args.core_source_sha256 = value();
    else if (key == "--matched-runner-source-sha256") args.matched_runner_source_sha256 = value();
    else if (key == "--pool-sha256") args.pool_sha256 = value();
    else if (key == "--queries-sha256") args.queries_sha256 = value();
    else if (key == "--initial-base-sha256") args.initial_base_sha256 = value();
    else if (key == "--stable-map-sha256") args.stable_map_sha256 = value();
    else if (key == "--help") {
      std::cout
          << "usage: GTS_fable5_native_snapshot_exporter_v1 --bundle BUNDLE "
          << "--header-trace E1GTRC02.bin --out SNAPSHOT.json --trace-sha256 SHA256 "
          << "--exporter-source-sha256 SHA256 --core-source-sha256 SHA256 "
          << "--matched-runner-source-sha256 SHA256 --pool-sha256 SHA256 "
          << "--queries-sha256 SHA256 --initial-base-sha256 SHA256 "
          << "--stable-map-sha256 SHA256\n";
      std::exit(0);
    } else {
      fail("unknown argument " + key);
    }
  }
  if (args.bundle.empty() || args.header_trace.empty() || args.output.empty()) {
    fail("--bundle, --header-trace, and --out are required");
  }
  require_sha256_hex("--trace-sha256", args.trace_sha256);
  require_sha256_hex("--exporter-source-sha256", args.exporter_source_sha256);
  require_sha256_hex("--core-source-sha256", args.core_source_sha256);
  require_sha256_hex("--matched-runner-source-sha256", args.matched_runner_source_sha256);
  require_sha256_hex("--pool-sha256", args.pool_sha256);
  require_sha256_hex("--queries-sha256", args.queries_sha256);
  require_sha256_hex("--initial-base-sha256", args.initial_base_sha256);
  require_sha256_hex("--stable-map-sha256", args.stable_map_sha256);
  return args;
}

// The header trace is immutable-input metadata only.  The exporter accepts the
// archived nine-op header source before the CPU selector can make a new Fable5
// twelve-op trace, and deliberately does not inspect/replay its events.
void validate_export_header(const TraceHeader& header) {
  if (header.dimension == 0 || header.base_n == 0 || header.pool_n < header.base_n ||
      header.reservoir_n != header.pool_n - header.base_n || header.query_n == 0 ||
      header.k == 0 || header.k > header.base_n || !std::isfinite(header.radius) ||
      header.radius < 0.0F) {
    fail("invalid immutable header for snapshot export");
  }
}

std::uint32_t float_bits(float value) {
  static_assert(sizeof(float) == sizeof(std::uint32_t), "snapshot export requires IEEE float32 bits");
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void emit_int_array(std::ostream& output, const std::vector<int>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) output << ',';
    output << values[index];
  }
  output << ']';
}

struct LogicalLeafPayload {
  int node_id = -1;
  std::vector<int> ids;
};

std::vector<LogicalLeafPayload> capture_leaf_payload(
    const safe_c1::BaseTreeRuntime& runtime,
    const safe_c1::FrozenTreeSnapshot& snapshot) {
  if (runtime.id_list == nullptr || runtime.max_node_num == nullptr ||
      snapshot.nodes.size() != snapshot.empty.size() ||
      static_cast<std::size_t>(runtime.max_node_num[0]) != snapshot.nodes.size()) {
    fail("inconsistent runtime/snapshot during logical leaf export");
  }
  std::vector<LogicalLeafPayload> payloads;
  std::vector<int> layout;
  int logical_count = 0;
  for (int node_id = 0; node_id < static_cast<int>(snapshot.nodes.size()); ++node_id) {
    if (snapshot.empty[static_cast<std::size_t>(node_id)] != 0 ||
        snapshot.nodes[static_cast<std::size_t>(node_id)].is_leaf != 1) {
      continue;
    }
    const TN& node = snapshot.nodes[static_cast<std::size_t>(node_id)];
    if (node.lid < 0 || node.size < 0) fail("negative frozen leaf range");
    LogicalLeafPayload item;
    item.node_id = node_id;
    item.ids.resize(static_cast<std::size_t>(node.size));
    if (!item.ids.empty()) {
      G2_CUDA(cudaMemcpy(item.ids.data(), runtime.id_list + node.lid,
                         item.ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }
    layout.push_back(node_id);
    layout.push_back(node.size);
    layout.insert(layout.end(), item.ids.begin(), item.ids.end());
    if (logical_count > std::numeric_limits<int>::max() - node.size) {
      fail("logical leaf count overflow");
    }
    logical_count += node.size;
    payloads.push_back(std::move(item));
  }
  const std::uint64_t layout_hash = safe_c1::FrozenTreeSnapshot::fnv1a(
      layout.empty() ? nullptr : layout.data(), layout.size() * sizeof(int));
  if (logical_count != snapshot.logical_leaf_id_count ||
      layout_hash != snapshot.logical_leaf_id_hash) {
    fail("logical leaf payload does not match captured FrozenTreeSnapshot hash");
  }
  return payloads;
}

void validate_exact_e0_leaf_set(const std::vector<LogicalLeafPayload>& payloads,
                                const std::vector<int>& initial_ids) {
  std::vector<int> flattened;
  for (const LogicalLeafPayload& payload : payloads) {
    flattened.insert(flattened.end(), payload.ids.begin(), payload.ids.end());
  }
  std::sort(flattened.begin(), flattened.end());
  if (flattened != initial_ids) {
    fail("exported frozen E0 leaf IDs are not exactly the immutable initial base");
  }
}

void write_snapshot(const Args& args, const TraceHeader& header,
                    const safe_c1::FrozenTreeSnapshot& snapshot,
                    const std::vector<LogicalLeafPayload>& payloads,
                    std::uint64_t immutable_pool_fnv1a) {
  if (sizeof(TN) != 20U || snapshot.nodes.size() != snapshot.empty.size() ||
      snapshot.nodes.size() != snapshot.max_distance.size() ||
      snapshot.compute_hash() != snapshot.hash) {
    fail("refusing to export unknown or internally inconsistent frozen TN snapshot");
  }
  std::ofstream output(args.output, std::ios::out | std::ios::trunc);
  if (!output) fail("cannot open snapshot output");
  output << "{\n"
         << "  \"schema\": \"fable5-native-frozen-snapshot-v1\",\n"
         << "  \"status\": \"NATIVE_E0_CAPTURED_PENDING_CPU_SELECTOR\",\n"
         << "  \"scope\": \"one native E0 metadata capture only; no policy replay, update trace, dynamic query, timing, C2, C3, or deployment claim\",\n"
         << "  \"capture_mode\": \"explicit standalone exporter; no matched runner replay\",\n"
         << "  \"residual_pruning\": {\"mode\":0,\"meaning\":\"baseline metric lower-bound traversal constants uploaded; no C2 calibration/drift or C3 scheduling\",\"query_executed\":false},\n"
         << "  \"immutable_input\": {\n"
         << "    \"header_trace_sha256\": \"" << args.trace_sha256 << "\",\n"
         << "    \"pool_i16_sha256\": \"" << args.pool_sha256 << "\",\n"
         << "    \"queries_i16_sha256\": \"" << args.queries_sha256 << "\",\n"
         << "    \"initial_base_stable_ids_i32_sha256\": \"" << args.initial_base_sha256 << "\",\n"
         << "    \"stable_id_to_pool_row_i32_sha256\": \"" << args.stable_map_sha256 << "\",\n"
         << "    \"exporter_source_sha256\": \"" << args.exporter_source_sha256 << "\",\n"
         << "    \"core_source_sha256\": \"" << args.core_source_sha256 << "\",\n"
         << "    \"matched_runner_source_sha256\": \"" << args.matched_runner_source_sha256 << "\"\n"
         << "  },\n"
         << "  \"header\": {\"dimension\":" << header.dimension
         << ",\"base_n\":" << header.base_n << ",\"reservoir_n\":" << header.reservoir_n
         << ",\"pool_n\":" << header.pool_n << ",\"query_n\":" << header.query_n
         << ",\"k\":" << header.k << ",\"radius\":" << std::setprecision(9) << header.radius << "},\n"
         << "  \"tree_height\": " << snapshot.tree_height << ",\n"
         << "  \"fanout\": " << snapshot.fanout << ",\n"
         << "  \"tn_abi\": \"little-endian:i32,f32bits,i32,i32,i32\",\n"
         << "  \"tn_size_bytes\": " << sizeof(TN) << ",\n"
         << "  \"metric_encoding\": {\"infi_dis\":" << INFI_DIS << ",\"dis_code\":" << DIS_CODE << "},\n"
         << "  \"native_hashes\": {\"frozen_tree_fnv1a64\":" << snapshot.hash
         << ",\"logical_leaf_id_fnv1a64\":" << snapshot.logical_leaf_id_hash
         << ",\"logical_leaf_id_count\":" << snapshot.logical_leaf_id_count
         << ",\"immutable_pool_fnv1a64\":" << immutable_pool_fnv1a << "},\n"
         << "  \"nodes\": [";
  for (std::size_t index = 0; index < snapshot.nodes.size(); ++index) {
    if (index) output << ',';
    const TN& node = snapshot.nodes[index];
    output << "{\"pid\":" << node.pid << ",\"min_dis_f32_bits\":" << float_bits(node.min_dis)
           << ",\"size\":" << node.size << ",\"lid\":" << node.lid
           << ",\"is_leaf\":" << node.is_leaf << '}';
  }
  output << "],\n  \"empty\": [";
  for (std::size_t index = 0; index < snapshot.empty.size(); ++index) {
    if (index) output << ',';
    output << snapshot.empty[index];
  }
  output << "],\n  \"max_distance_f32_bits\": [";
  for (std::size_t index = 0; index < snapshot.max_distance.size(); ++index) {
    if (index) output << ',';
    output << float_bits(snapshot.max_distance[index]);
  }
  output << "],\n  \"logical_leaf_payload\": [";
  for (std::size_t index = 0; index < payloads.size(); ++index) {
    if (index) output << ',';
    const LogicalLeafPayload& payload = payloads[index];
    const TN& node = snapshot.nodes[static_cast<std::size_t>(payload.node_id)];
    output << "{\"node_id\":" << payload.node_id << ",\"lid\":" << node.lid
           << ",\"size\":" << node.size << ",\"ids\":";
    emit_int_array(output, payload.ids);
    output << '}';
  }
  output << "]\n}\n";
  output.flush();
  if (!output) fail("failed while writing native frozen snapshot");
}

int execute(const Args& args) {
  safe_c1::BaseTreeRuntime runtime;
  try {
    const auto trace = read_trace(args.header_trace);
    const TraceHeader& header = trace.first;
    validate_export_header(header);
    const std::vector<int> initial_ids = require_identity_layout(args.bundle, header);
    const std::size_t pool_values = static_cast<std::size_t>(header.pool_n) * header.dimension;
    const std::size_t query_values = static_cast<std::size_t>(header.query_n) * header.dimension;
    const std::vector<std::int16_t> pool_i16 =
        read_exact_binary<std::int16_t>(args.bundle + "/pool.i16", pool_values);
    const std::vector<std::int16_t> queries_i16 =
        read_exact_binary<std::int16_t>(args.bundle + "/queries.i16", query_values);
    const MetricEncodingPlan metric = make_metric_encoding_plan(
        pool_i16, queries_i16, static_cast<int>(header.dimension));
    std::vector<GtsScalar> immutable_pool(pool_values);
    for (std::size_t index = 0; index < pool_values; ++index) {
      immutable_pool[index] = static_cast<GtsScalar>(pool_i16[index]);
    }
    const std::uint64_t immutable_pool_fnv1a = safe_c1::FrozenTreeSnapshot::fnv1a(
        immutable_pool.data(), immutable_pool.size() * sizeof(GtsScalar));

    DIS_CODE = metric.dis_code;
    INFI_DIS = metric.infi_dis;
    G2_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&runtime.data_info), 3 * sizeof(int)));
    runtime.data_info[0] = static_cast<int>(header.dimension);
    runtime.data_info[1] = static_cast<int>(header.base_n);
    runtime.data_info[2] = 2;
    G2_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&runtime.data_d),
                              pool_values * sizeof(GtsScalar)));
    for (std::size_t index = 0; index < pool_values; ++index) {
      runtime.data_d[index] = immutable_pool[index];
    }

    // Mirror matched runner mode=0 exactly.  This exporter does not launch a
    // query; the record only pins the baseline traversal configuration.
    float alpha[RP_MAX_LEVELS]{};
    float beta[RP_MAX_LEVELS]{};
    float gamma[RP_MAX_LEVELS];
    std::fill(std::begin(gamma), std::end(gamma), 1.0F);
    upload_rp_constants(alpha, beta, gamma, RP_MAX_LEVELS, nullptr, nullptr, nullptr, 0, 0);

    build_seeded_epoch(runtime, initial_ids);
    safe_c1::FrozenTreeSnapshot snapshot;
    snapshot.capture(runtime);
    if (snapshot.logical_leaf_id_count != static_cast<int>(initial_ids.size())) {
      fail("captured E0 leaf count does not equal immutable initial base");
    }
    const std::vector<LogicalLeafPayload> payloads = capture_leaf_payload(runtime, snapshot);
    validate_exact_e0_leaf_set(payloads, initial_ids);
    write_snapshot(args, header, snapshot, payloads, immutable_pool_fnv1a);
    release_tree_full(runtime);
    return 0;
  } catch (...) {
    release_tree_full(runtime);
    throw;
  }
}

}  // namespace fable5_snapshot_exporter

int main(int argc, char** argv) {
  try {
    return fable5_snapshot_exporter::execute(fable5_snapshot_exporter::parse_args(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
