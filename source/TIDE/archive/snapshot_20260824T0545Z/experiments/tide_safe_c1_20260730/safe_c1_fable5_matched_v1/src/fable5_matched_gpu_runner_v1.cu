// Fable5 matched Safe-C1 vs buffer-only GPU runner (v1).
//
// This source is deliberately isolated from the archived G2 witness.  It is a
// minimal correctness runner, not a performance harness: it consumes one
// binary E1GTRC02 trace per policy, derives every certificate from the frozen
// native GTS sibling boundaries, records the traversal receipt, and checks
// every dynamic query against the same full-active-set exact oracle.
//
// It is intentionally NOT compiled or executed by this change.  The only
// permitted execution path is the separate NVML-only GPU1 guard.

// Reuse the archived, pinned GTS integration support in this *translation unit*
// without invoking its historical nine-operation entry point.  The archived
// source remains unmodified and this runner has its own main below.
#define main fable5_archived_g2_entry_not_invoked
#include "safe_c1_dynamic_gts.cu"
#undef main
#include <cctype>

namespace fable5_matched {

enum class Policy { kSafeC1, kBufferOnly };
enum class Fallback { kNone, kCertificateReject, kCapacityFull, kBufferOnlyPolicy };

const char* policy_label(Policy policy) {
  return policy == Policy::kSafeC1 ? "safe_c1" : "buffer_only";
}

const char* fallback_label(Fallback fallback) {
  switch (fallback) {
    case Fallback::kNone: return "none";
    case Fallback::kCertificateReject: return "certificate_reject";
    case Fallback::kCapacityFull: return "capacity_full";
    case Fallback::kBufferOnlyPolicy: return "buffer_only_policy";
  }
  return "unknown";
}

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("Fable5 matched runner: " + message);
}

struct Args {
  std::string bundle;
  std::string trace;
  std::string policy;
  std::string output;
  std::string summary;
  std::string trace_sha256;
  int leaf_capacity = -1;
  bool require_coverage = true;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    auto value = [&]() -> std::string {
      if (++i >= argc) fail("missing value after " + key);
      return argv[i];
    };
    if (key == "--bundle") args.bundle = value();
    else if (key == "--trace") args.trace = value();
    else if (key == "--policy") args.policy = value();
    else if (key == "--out") args.output = value();
    else if (key == "--summary") args.summary = value();
    else if (key == "--trace-sha256") args.trace_sha256 = value();
    else if (key == "--leaf-capacity") args.leaf_capacity = std::stoi(value());
    else if (key == "--require-coverage") {
      const std::string raw = value();
      if (raw != "0" && raw != "1") fail("--require-coverage must be 0 or 1");
      args.require_coverage = raw == "1";
    } else if (key == "--help") {
      std::cout
          << "usage: GTS_fable5_matched_v1 --bundle BUNDLE --trace TRACE.e1gtrc --policy "
          << "safe_c1|buffer_only --out RESULTS.jsonl --summary SUMMARY.json "
          << "--trace-sha256 SHA256 --leaf-capacity N [--require-coverage 0|1]\n";
      std::exit(0);
    } else {
      fail("unknown argument " + key);
    }
  }
  if (args.bundle.empty() || args.trace.empty() || args.policy.empty() || args.output.empty() ||
      args.summary.empty() || args.trace_sha256.size() != 64U || args.leaf_capacity <= 0) {
    fail("--bundle, --trace, --policy, --out, --summary, 64-char --trace-sha256, and positive --leaf-capacity are required");
  }
  if (args.policy != "safe_c1" && args.policy != "buffer_only") fail("unsupported --policy");
  if (!std::all_of(args.trace_sha256.begin(), args.trace_sha256.end(), [](unsigned char c) {
        return std::isxdigit(c) != 0;
      })) {
    fail("--trace-sha256 is not hexadecimal");
  }
  return args;
}

Policy parse_policy(const std::string& value) {
  if (value == "safe_c1") return Policy::kSafeC1;
  if (value == "buffer_only") return Policy::kBufferOnly;
  fail("invalid policy");
}

// The runner deliberately uses the existing E1GTRC02 binary trace ABI.  A
// future trace producer must select stable IDs from the immutable pool, but it
// never supplies a leaf label: this runner computes it from frozen native GTS
// metadata at runtime.
void validate_matched_trace(const TraceHeader& header, const std::vector<TraceEvent>& events) {
  if (header.event_count != events.size() || header.dimension == 0 || header.base_n == 0 ||
      header.pool_n < header.base_n || header.reservoir_n != header.pool_n - header.base_n ||
      header.query_n == 0 || header.k == 0 || header.k > header.base_n) {
    fail("invalid matched trace header");
  }
  bool has_insert = false, has_delete = false, has_query_before = false, has_query_after = false;
  int rebuild_index = -1;
  for (std::size_t i = 0; i < events.size(); ++i) {
    const TraceEvent& event = events[i];
    if (event.op_index != i || event.op < kInsert || event.op > kRebuild || event.op == kRange) {
      fail("invalid/non-matched event at op " + std::to_string(i));
    }
    if (event.op == kInsert) {
      has_insert = true;
      if (event.argument < static_cast<int>(header.base_n) || event.argument >= static_cast<int>(header.pool_n)) {
        fail("insert ID is not a fresh reservoir stable ID");
      }
    } else if (event.op == kDelete) {
      has_delete = true;
      if (event.argument < 0 || event.argument >= static_cast<int>(header.pool_n)) fail("delete ID outside pool");
    } else if (event.op == kKnn) {
      if (event.argument < 0 || event.argument >= static_cast<int>(header.query_n)) fail("query ID outside trace query pool");
      if (rebuild_index < 0) has_query_before = true;
      else has_query_after = true;
    } else if (event.op == kRebuild) {
      if (event.argument != 0 || rebuild_index >= 0) fail("matched v1 requires exactly one zero-argument rebuild");
      rebuild_index = static_cast<int>(i);
    }
  }
  if (!has_insert || !has_delete || rebuild_index < 0 || !has_query_before || !has_query_after) {
    fail("matched v1 trace must cover insert/delete/query both sides of one rebuild");
  }
}

struct InsertReceipt {
  safe_c1::PlacementKind placement = safe_c1::PlacementKind::kUnknown;
  int native_certificate_leaf = -1;
  int sidecar_leaf = -1;
  Fallback fallback = Fallback::kNone;
};

struct DeleteReceipt {
  safe_c1::PlacementKind prior = safe_c1::PlacementKind::kUnknown;
  int sidecar_leaf = -1;
  int native_certificate_leaf = -1;
  Fallback fallback = Fallback::kNone;
};

// Host-side state only.  Direct entries remain in an external per-leaf
// sidecar.  No path here writes legacy TN/id_list after epoch construction.
class MatchedState {
 public:
  MatchedState(Policy policy, int pool_n, const std::vector<int>& initial_base, int leaf_capacity)
      : policy_(policy),
        active_(static_cast<std::size_t>(pool_n), 0),
        placement_(static_cast<std::size_t>(pool_n), safe_c1::PlacementKind::kDeleted),
        sidecar_leaf_(static_cast<std::size_t>(pool_n), -1),
        certificate_leaf_(static_cast<std::size_t>(pool_n), -1),
        fallback_(static_cast<std::size_t>(pool_n), Fallback::kNone),
        leaf_capacity_(leaf_capacity) {
    if (pool_n <= 0 || leaf_capacity <= 0 || !std::is_sorted(initial_base.begin(), initial_base.end()) ||
        std::adjacent_find(initial_base.begin(), initial_base.end()) != initial_base.end()) {
      fail("invalid matched state initialization");
    }
    for (int id : initial_base) {
      check(id);
      if (active_[static_cast<std::size_t>(id)] != 0) fail("duplicate initial base ID");
      active_[static_cast<std::size_t>(id)] = 1;
      placement_[static_cast<std::size_t>(id)] = safe_c1::PlacementKind::kBase;
    }
  }

  InsertReceipt insert(int id, const EpochFrozen& frozen, const safe_c1::HostVectorPool& pool) {
    require_not_rebuild_pending("insert");
    check(id);
    if (active_[static_cast<std::size_t>(id)] != 0) fail("insert targets active stable ID");
    active_[static_cast<std::size_t>(id)] = 1;

    // This is the native strict sibling-boundary certificate.  It is evaluated
    // for BOTH policies; buffer-only ignores placement but records the exact
    // same certificate receipt for matched comparison.
    const int certificate = frozen.snapshot.certify_leaf(pool, id);
    certificate_leaf_[static_cast<std::size_t>(id)] = certificate;
    ++certificate_evaluations_;
    if (certificate >= 0) ++certificate_accepts_;
    else ++certificate_rejects_;
    sidecar_leaf_[static_cast<std::size_t>(id)] = -1;

    if (policy_ == Policy::kBufferOnly) {
      place_delta(id, Fallback::kBufferOnlyPolicy);
      return {safe_c1::PlacementKind::kDelta, certificate, -1, Fallback::kBufferOnlyPolicy};
    }
    if (certificate < 0) {
      place_delta(id, Fallback::kCertificateReject);
      ++safe_certificate_fallbacks_;
      return {safe_c1::PlacementKind::kDelta, certificate, -1, Fallback::kCertificateReject};
    }
    auto& values = sidecars_[certificate];
    if (static_cast<int>(values.size()) >= leaf_capacity_) {
      place_delta(id, Fallback::kCapacityFull);
      ++safe_capacity_fallbacks_;
      return {safe_c1::PlacementKind::kDelta, certificate, -1, Fallback::kCapacityFull};
    }
    values.push_back(id);
    placement_[static_cast<std::size_t>(id)] = safe_c1::PlacementKind::kDirect;
    sidecar_leaf_[static_cast<std::size_t>(id)] = certificate;
    fallback_[static_cast<std::size_t>(id)] = Fallback::kNone;
    ++direct_inserts_;
    return {safe_c1::PlacementKind::kDirect, certificate, certificate, Fallback::kNone};
  }

  DeleteReceipt erase(int id) {
    require_not_rebuild_pending("delete");
    check(id);
    if (active_[static_cast<std::size_t>(id)] == 0) fail("delete targets inactive stable ID");
    const safe_c1::PlacementKind prior = placement_[static_cast<std::size_t>(id)];
    const int leaf = sidecar_leaf_[static_cast<std::size_t>(id)];
    const int certificate = certificate_leaf_[static_cast<std::size_t>(id)];
    const Fallback reason = fallback_[static_cast<std::size_t>(id)];
    if (prior == safe_c1::PlacementKind::kDirect) {
      auto found = sidecars_.find(leaf);
      if (found == sidecars_.end()) fail("direct placement missing its sidecar");
      auto& ids = found->second;
      ids.erase(std::remove(ids.begin(), ids.end(), id), ids.end());
      ++direct_deletes_;
    } else if (prior == safe_c1::PlacementKind::kDelta) {
      delta_.erase(std::remove(delta_.begin(), delta_.end(), id), delta_.end());
      ++delta_deletes_;
    } else if (prior == safe_c1::PlacementKind::kBase) {
      rebuild_required_ = true;
      ++base_deletes_;
    } else {
      fail("invalid active placement on delete");
    }
    active_[static_cast<std::size_t>(id)] = 0;
    placement_[static_cast<std::size_t>(id)] = safe_c1::PlacementKind::kDeleted;
    sidecar_leaf_[static_cast<std::size_t>(id)] = -1;
    return {prior, leaf, certificate, reason};
  }

  void require_not_rebuild_pending(const char* operation) const {
    if (rebuild_required_) fail(std::string("base deletion requires immediate REBUILD before ") + operation);
  }
  void require_rebuild_pending() const {
    if (!rebuild_required_) fail("REBUILD appears without a preceding base deletion");
  }

  void after_rebuild(const std::vector<int>& newly_seeded_live_ids) {
    require_rebuild_pending();
    if (live_ids() != newly_seeded_live_ids) fail("rebuild seed does not preserve exact active set");
    sidecars_.clear();
    delta_.clear();
    for (int id = 0; id < static_cast<int>(active_.size()); ++id) {
      const std::size_t pos = static_cast<std::size_t>(id);
      placement_[pos] = active_[pos] ? safe_c1::PlacementKind::kBase : safe_c1::PlacementKind::kDeleted;
      sidecar_leaf_[pos] = -1;
      certificate_leaf_[pos] = -1;
      fallback_[pos] = Fallback::kNone;
    }
    rebuild_required_ = false;
    ++rebuilds_;
  }

  std::vector<int> sidecar_candidates_for(const std::vector<int>& visited_leaf_ids) const {
    std::vector<int> selected;
    for (int leaf : visited_leaf_ids) {
      const auto found = sidecars_.find(leaf);
      if (found == sidecars_.end()) continue;
      for (int id : found->second) {
        if (active_[static_cast<std::size_t>(id)] &&
            placement_[static_cast<std::size_t>(id)] == safe_c1::PlacementKind::kDirect) {
          selected.push_back(id);
        }
      }
    }
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    return selected;
  }

  const std::vector<std::uint8_t>& active() const { return active_; }
  const std::vector<int>& delta_ids() const { return delta_; }
  std::vector<int> live_ids() const {
    std::vector<int> ids;
    for (int id = 0; id < static_cast<int>(active_.size()); ++id)
      if (active_[static_cast<std::size_t>(id)]) ids.push_back(id);
    return ids;
  }
  safe_c1::PlacementKind placement_of(int id) const { check(id); return placement_[static_cast<std::size_t>(id)]; }
  bool rebuild_required() const { return rebuild_required_; }
  int direct_inserts() const { return direct_inserts_; }
  int direct_deletes() const { return direct_deletes_; }
  int delta_deletes() const { return delta_deletes_; }
  int base_deletes() const { return base_deletes_; }
  int rebuilds() const { return rebuilds_; }
  int certificate_evaluations() const { return certificate_evaluations_; }
  int certificate_accepts() const { return certificate_accepts_; }
  int certificate_rejects() const { return certificate_rejects_; }
  int safe_capacity_fallbacks() const { return safe_capacity_fallbacks_; }
  int safe_certificate_fallbacks() const { return safe_certificate_fallbacks_; }
  std::uint64_t active_hash() const { return fnv_active(active_); }

 private:
  void check(int id) const {
    if (id < 0 || id >= static_cast<int>(active_.size())) fail("stable ID outside immutable pool");
  }
  void place_delta(int id, Fallback reason) {
    delta_.push_back(id);
    placement_[static_cast<std::size_t>(id)] = safe_c1::PlacementKind::kDelta;
    fallback_[static_cast<std::size_t>(id)] = reason;
  }

  Policy policy_;
  std::vector<std::uint8_t> active_;
  std::vector<safe_c1::PlacementKind> placement_;
  std::vector<int> sidecar_leaf_;
  std::vector<int> certificate_leaf_;
  std::vector<Fallback> fallback_;
  int leaf_capacity_ = 0;
  std::unordered_map<int, std::vector<int>> sidecars_;
  std::vector<int> delta_;
  bool rebuild_required_ = false;
  int direct_inserts_ = 0;
  int direct_deletes_ = 0;
  int delta_deletes_ = 0;
  int base_deletes_ = 0;
  int rebuilds_ = 0;
  int certificate_evaluations_ = 0;
  int certificate_accepts_ = 0;
  int certificate_rejects_ = 0;
  int safe_capacity_fallbacks_ = 0;
  int safe_certificate_fallbacks_ = 0;
};

// NB: run_query is GPU-integrated: it invokes real native GTS traversal,
// consumes its visited-leaf receipt for sidecar visibility, merges the global
// exact delta, and validates the result against the independent full-active-set
// oracle.  It records no timing measurements.
void run_query(const TraceEvent& event, const TraceHeader& header, MatchedState& state,
               const EpochFrozen& epoch, safe_c1::BaseTreeRuntime& runtime,
               safe_c1::HostVectorPool& pool, float* query_d, int* candidate_ids_d,
               double* candidate_distances_d, std::ofstream& out) {
  state.require_not_rebuild_pending("query");
  const int qid = event.argument;
  float* query = query_d + static_cast<std::size_t>(qid) * header.dimension;
  const safe_c1::TopKWithTraversalReceipt base = safe_c1::run_gts_base_topk_with_receipt(
      runtime, query, 1, static_cast<int>(header.k));
  epoch.assert_unchanged(runtime);
  std::vector<int> sidecars = state.sidecar_candidates_for(base.visited_leaf_ids);
  std::vector<int> extra = sidecars;
  extra.insert(extra.end(), state.delta_ids().begin(), state.delta_ids().end());
  std::sort(extra.begin(), extra.end());
  if (std::adjacent_find(extra.begin(), extra.end()) != extra.end()) fail("sidecar/delta overlap");
  for (int id : extra) {
    if (!state.active()[static_cast<std::size_t>(id)] ||
        state.placement_of(id) == safe_c1::PlacementKind::kBase) {
      fail("invalid nonbase dynamic candidate");
    }
  }
  std::vector<double> distances(extra.size());
  if (!extra.empty()) {
    G2_CUDA(cudaMemcpy(candidate_ids_d, extra.data(), extra.size() * sizeof(int), cudaMemcpyHostToDevice));
    exact_candidate_l2<<<(static_cast<int>(extra.size()) + 255) / 256, 256>>>(
        runtime.data_d, query, candidate_ids_d, candidate_distances_d,
        static_cast<int>(extra.size()), static_cast<int>(header.dimension));
    G2_CUDA(cudaGetLastError());
    G2_CUDA(cudaDeviceSynchronize());
    G2_CUDA(cudaMemcpy(distances.data(), candidate_distances_d, distances.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  }
  std::vector<std::pair<double, int>> merged;
  merged.reserve(static_cast<std::size_t>(header.k) + extra.size());
  for (int id : base.topk.ids) {
    if (id < 0 || id >= static_cast<int>(header.pool_n) ||
        !state.active()[static_cast<std::size_t>(id)] ||
        state.placement_of(id) != safe_c1::PlacementKind::kBase) {
      fail("native GTS emitted nonbase/deleted result");
    }
    const float* vector = pool.at(id);
    double squared = 0.0;
    for (int d = 0; d < pool.dimension; ++d) {
      const double diff = static_cast<double>(vector[d]) - static_cast<double>(query[d]);
      squared += diff * diff;
    }
    merged.emplace_back(std::sqrt(squared), id);
  }
  for (std::size_t i = 0; i < extra.size(); ++i) merged.emplace_back(distances[i], extra[i]);
  std::sort(merged.begin(), merged.end(), [](const auto& left, const auto& right) {
    return left.first != right.first ? left.first < right.first : left.second < right.second;
  });
  if (merged.size() > header.k) merged.resize(header.k);
  const auto oracle = exact_active_oracle(pool, query, state.active(), static_cast<int>(header.k));
  require_same_exact_rows(merged, oracle);

  out << "{\"record\":\"query\",\"op_index\":" << event.op_index
      << ",\"query_id\":" << qid << ",\"active_hash\":" << state.active_hash()
      << ",\"gts_visited_leaf_ids\":";
  emit_int_array(out, base.visited_leaf_ids);
  out << ",\"sidecar_ids\":";
  emit_int_array(out, sidecars);
  out << ",\"delta_ids\":";
  emit_int_array(out, state.delta_ids());
  out << ",\"oracle_full_active_set_checked\":true,\"results\":[";
  for (std::size_t i = 0; i < merged.size(); ++i) {
    if (i) out << ',';
    out << '[' << merged[i].second << ',' << json_float(merged[i].first) << ']';
  }
  out << "]}\n";
}

int execute(const Args& args) {
  const Policy policy = parse_policy(args.policy);
  safe_c1::BaseTreeRuntime runtime;
  float* query_d = nullptr;
  int* candidate_ids_d = nullptr;
  double* candidate_distances_d = nullptr;
  try {
    const auto trace = read_trace(args.trace);
    const TraceHeader& header = trace.first;
    const std::vector<TraceEvent>& events = trace.second;
    validate_matched_trace(header, events);

    // Full CPU file/trace validation happens before any CUDA allocation.
    const std::vector<int> initial_ids = require_identity_layout(args.bundle, header);
    const std::size_t pool_values = static_cast<std::size_t>(header.pool_n) * header.dimension;
    const std::size_t query_values = static_cast<std::size_t>(header.query_n) * header.dimension;
    const std::vector<std::int16_t> pool_i16 =
        read_exact_binary<std::int16_t>(args.bundle + "/pool.i16", pool_values);
    const std::vector<std::int16_t> queries_i16 =
        read_exact_binary<std::int16_t>(args.bundle + "/queries.i16", query_values);
    const MetricEncodingPlan metric = make_metric_encoding_plan(pool_i16, queries_i16,
                                                                  static_cast<int>(header.dimension));
    TieAudit ties;
    require_static_epoch_tie_free(initial_ids, pool_i16, queries_i16,
                                  static_cast<int>(header.dimension), static_cast<int>(header.k), &ties);

    safe_c1::HostVectorPool pool;
    pool.dimension = static_cast<int>(header.dimension);
    pool.vectors.resize(pool_values);
    for (std::size_t i = 0; i < pool_values; ++i) pool.vectors[i] = static_cast<float>(pool_i16[i]);
    std::vector<float> query_host(query_values);
    for (std::size_t i = 0; i < query_values; ++i) query_host[i] = static_cast<float>(queries_i16[i]);
    std::vector<GtsScalar> immutable_pool_encoded(pool_values);
    for (std::size_t i = 0; i < pool_values; ++i) immutable_pool_encoded[i] = static_cast<GtsScalar>(pool_i16[i]);
    const std::uint64_t host_pool_hash = safe_c1::FrozenTreeSnapshot::fnv1a(
        immutable_pool_encoded.data(), immutable_pool_encoded.size() * sizeof(GtsScalar));

    DIS_CODE = metric.dis_code;
    INFI_DIS = metric.infi_dis;
    G2_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&runtime.data_info), 3 * sizeof(int)));
    runtime.data_info[0] = static_cast<int>(header.dimension);
    runtime.data_info[1] = static_cast<int>(header.base_n);
    runtime.data_info[2] = 2;
    G2_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&runtime.data_d), pool_values * sizeof(GtsScalar)));
    for (std::size_t i = 0; i < pool_values; ++i) runtime.data_d[i] = immutable_pool_encoded[i];
    float alpha[RP_MAX_LEVELS]{};
    float beta[RP_MAX_LEVELS]{};
    float gamma[RP_MAX_LEVELS];
    std::fill(std::begin(gamma), std::end(gamma), 1.0F);
    upload_rp_constants(alpha, beta, gamma, RP_MAX_LEVELS, nullptr, nullptr, nullptr, 0, 0);
    G2_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&query_d), query_values * sizeof(float)));
    std::copy(query_host.begin(), query_host.end(), query_d);
    G2_CUDA(cudaMalloc(reinterpret_cast<void**>(&candidate_ids_d), header.pool_n * sizeof(int)));
    G2_CUDA(cudaMalloc(reinterpret_cast<void**>(&candidate_distances_d), header.pool_n * sizeof(double)));

    auto device_pool_hash = [&]() {
      std::vector<GtsScalar> copy(pool_values);
      G2_CUDA(cudaMemcpy(copy.data(), runtime.data_d, copy.size() * sizeof(GtsScalar), cudaMemcpyDeviceToHost));
      return safe_c1::FrozenTreeSnapshot::fnv1a(copy.data(), copy.size() * sizeof(GtsScalar));
    };
    if (device_pool_hash() != host_pool_hash) fail("immutable full pool mismatch before E0");

    build_seeded_epoch(runtime, initial_ids);
    EpochFrozen epoch0;
    epoch0.capture(runtime, initial_ids);
    validate_real_static_probe(runtime, query_d, initial_ids, pool_i16, queries_i16,
                               static_cast<int>(header.dimension), static_cast<int>(header.k));
    epoch0.assert_unchanged(runtime);

    MatchedState state(policy, static_cast<int>(header.pool_n), initial_ids, args.leaf_capacity);
    std::ofstream out(args.output);
    if (!out) fail("cannot open output JSONL");
    out << "{\"record\":\"meta\",\"schema\":\"fable5-matched-gpu-runner-v1\","
        << "\"policy\":\"" << policy_label(policy) << "\",\"trace_sha256\":\"" << args.trace_sha256
        << "\",\"scope\":\"matched correctness receipt only; no timing, throughput, latency, C2, C3, or deployment claim\","
        << "\"legacy_incremental_updater_used\":false,\"direct_storage\":\"external_receipt_indexed_sidecar\","
        << "\"rebuild_policy\":\"base_delete_immediate\"}\n";
    out << "{\"record\":\"epoch\",\"epoch\":0,\"tree_hash\":" << epoch0.snapshot.hash
        << ",\"leaf_stable_id_hash\":" << epoch0.exact_leaf_hash
        << ",\"pivot_stable_id_hash\":" << epoch0.exact_pivot_hash
        << ",\"leaf_set_exact\":true,\"pivot_set_subset_of_live\":true}\n";

    EpochFrozen epoch1;
    const EpochFrozen* current_epoch = &epoch0;
    bool have_epoch1 = false;
    std::uint64_t pre_rebuild_active_hash = 0;
    std::uint64_t post_rebuild_active_hash = 0;
    std::uint64_t post_rebuild_pool_hash = 0;
    for (const TraceEvent& event : events) {
      if (event.op == kInsert) {
        const InsertReceipt receipt = state.insert(event.argument, *current_epoch, pool);
        current_epoch->assert_unchanged(runtime);
        out << "{\"record\":\"update\",\"op_index\":" << event.op_index
            << ",\"op\":\"insert\",\"stable_id\":" << event.argument
            << ",\"placement\":\"" << placement_label(receipt.placement)
            << "\",\"native_certificate_leaf\":" << receipt.native_certificate_leaf
            << ",\"sidecar_leaf_id\":" << receipt.sidecar_leaf
            << ",\"fallback_reason\":\"" << fallback_label(receipt.fallback) << "\"}\n";
      } else if (event.op == kDelete) {
        const DeleteReceipt receipt = state.erase(event.argument);
        current_epoch->assert_unchanged(runtime);
        out << "{\"record\":\"update\",\"op_index\":" << event.op_index
            << ",\"op\":\"delete\",\"stable_id\":" << event.argument
            << ",\"prior_placement\":\"" << placement_label(receipt.prior)
            << "\",\"native_certificate_leaf\":" << receipt.native_certificate_leaf
            << ",\"sidecar_leaf_id\":" << receipt.sidecar_leaf
            << ",\"fallback_reason\":\"" << fallback_label(receipt.fallback) << "\"}\n";
      } else if (event.op == kKnn) {
        run_query(event, header, state, *current_epoch, runtime, pool, query_d,
                  candidate_ids_d, candidate_distances_d, out);
      } else if (event.op == kRebuild) {
        state.require_rebuild_pending();
        current_epoch->assert_unchanged(runtime);
        const std::vector<int> live = state.live_ids();
        pre_rebuild_active_hash = state.active_hash();
        const std::uint64_t before_pool_hash = device_pool_hash();
        if (before_pool_hash != host_pool_hash) fail("immutable pool mutated before rebuild");
        release_tree_metadata_only(runtime);
        build_seeded_epoch(runtime, live);
        epoch1.capture(runtime, live);
        require_static_epoch_tie_free(live, pool_i16, queries_i16,
                                      static_cast<int>(header.dimension), static_cast<int>(header.k), &ties);
        validate_real_static_probe(runtime, query_d, live, pool_i16, queries_i16,
                                   static_cast<int>(header.dimension), static_cast<int>(header.k));
        epoch1.assert_unchanged(runtime);
        post_rebuild_pool_hash = device_pool_hash();
        if (post_rebuild_pool_hash != host_pool_hash) fail("metadata-only rebuild mutated immutable pool");
        state.after_rebuild(live);
        post_rebuild_active_hash = state.active_hash();
        if (pre_rebuild_active_hash != post_rebuild_active_hash || state.rebuild_required()) {
          fail("rebuild did not preserve active set/clear barrier");
        }
        current_epoch = &epoch1;
        have_epoch1 = true;
        out << "{\"record\":\"rebuild\",\"op_index\":" << event.op_index
            << ",\"trigger\":\"base_delete_immediate\",\"metadata_only_release\":true,"
            << "\"pre_rebuild_active_hash\":" << pre_rebuild_active_hash
            << ",\"post_rebuild_active_hash\":" << post_rebuild_active_hash
            << ",\"immutable_pool_hash_before\":" << before_pool_hash
            << ",\"immutable_pool_hash_after\":" << post_rebuild_pool_hash
            << ",\"E1_tree_hash\":" << epoch1.snapshot.hash
            << ",\"E1_leaf_set_exact\":true,\"E1_pivot_set_subset_of_live\":true}\n";
      } else {
        fail("unsupported matched trace opcode");
      }
      if (!out) fail("failed to write JSONL receipt");
    }
    if (!have_epoch1 || state.base_deletes() != 1 || state.rebuilds() != 1) {
      fail("matched trace did not realize exactly one base-delete/immediate-rebuild transition");
    }
    if (args.require_coverage) {
      if (policy == Policy::kSafeC1 &&
          (state.direct_inserts() == 0 || state.safe_capacity_fallbacks() == 0 ||
           state.safe_certificate_fallbacks() == 0 || state.direct_deletes() == 0 ||
           state.delta_deletes() == 0)) {
        fail("safe_c1 trace lacks required direct/capacity/certificate/delete coverage");
      }
      if (policy == Policy::kBufferOnly && state.direct_inserts() != 0) {
        fail("buffer-only policy unexpectedly admitted a direct sidecar");
      }
    }
    current_epoch->assert_unchanged(runtime);
    out.close();

    std::ofstream summary(args.summary);
    if (!summary) fail("cannot open summary");
    summary << "{\n"
            << "  \"schema\": \"fable5-matched-gpu-runner-v1\",\n"
            << "  \"status\": \"PASS_FABLE5_MATCHED_PENDING_INDEPENDENT_VALIDATOR\",\n"
            << "  \"policy\": \"" << policy_label(policy) << "\",\n"
            << "  \"trace_sha256\": \"" << args.trace_sha256 << "\",\n"
            << "  \"scope\": \"matched correctness receipt only; no timing/performance/C2/C3/deployment claim\",\n"
            << "  \"native_certificate\": {\"evaluations\": " << state.certificate_evaluations()
            << ", \"accepts\": " << state.certificate_accepts()
            << ", \"rejects\": " << state.certificate_rejects() << "},\n"
            << "  \"policy_routes\": {\"direct_inserts\": " << state.direct_inserts()
            << ", \"safe_capacity_fallbacks\": " << state.safe_capacity_fallbacks()
            << ", \"safe_certificate_fallbacks\": " << state.safe_certificate_fallbacks()
            << ", \"direct_deletes\": " << state.direct_deletes()
            << ", \"delta_deletes\": " << state.delta_deletes() << "},\n"
            << "  \"rebuild\": {\"trigger\": \"base_delete_immediate\", \"active_set_preserved\": true, \"immutable_pool_preserved\": true},\n"
            << "  \"not_established\": [\"timing\", \"throughput\", \"latency\", \"recall curve\", \"C2\", \"C3\", \"deployment\"]\n"
            << "}\n";
    summary.close();

    G2_CUDA(cudaFree(candidate_ids_d)); candidate_ids_d = nullptr;
    G2_CUDA(cudaFree(candidate_distances_d)); candidate_distances_d = nullptr;
    G2_CUDA(cudaFree(query_d)); query_d = nullptr;
    release_tree_full(runtime);
    return 0;
  } catch (...) {
    if (candidate_ids_d) cudaFree(candidate_ids_d);
    if (candidate_distances_d) cudaFree(candidate_distances_d);
    if (query_d) cudaFree(query_d);
    release_tree_full(runtime);
    throw;
  }
}

}  // namespace fable5_matched

int main(int argc, char** argv) {
  try {
    return fable5_matched::execute(fable5_matched::parse_args(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
