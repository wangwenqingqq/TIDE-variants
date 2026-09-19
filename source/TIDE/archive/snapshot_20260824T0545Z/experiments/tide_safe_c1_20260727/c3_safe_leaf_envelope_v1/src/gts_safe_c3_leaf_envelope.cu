// Safe-C3 single-candidate leaf-envelope correctness harness.
//
// This translation unit deliberately reuses only the native build/query skeleton
// from a frozen archived probe.  The executable entry point below is new and
// never invokes the archived probe's former main.  Its scope is deliberately
// narrow: one preloaded inactive raw-pool object (stable ID 4102), a strictly
// certified sibling-min route, one quiescent direct mutation, complete D2H
// structural re-verification, real native vector KNN, and an explicitly
// non-native exact full-live range fallback.
//
// It is NOT a generic C3 updater.  It does not cover deletion, rebuild,
// concurrency, merge/buffer paths, native range traversal, or performance.

#define main c3_archived_leaf_only_probe_main_disabled
#include "../reference/gts_c3_native_direct_insert_probe.cu"
#undef main

#include <array>
#include <cstdio>
#include <functional>
#include <set>

namespace {

constexpr int kSafeCandidateStableId = 4102;
constexpr int kSafeExpectedLeaf = 604;
constexpr std::array<int, 4> kSafeExpectedRoute{{0, 5, 60, 604}};
constexpr float kCoverageTolerance = 0.25F;

// The kernel receives a fully prepared plan and performs no search/fallback.
// It first checks every expected precondition, then applies exactly the
// field-level whitelist.  The harness is quiescent: there are no concurrent
// readers/writers during this test-only commit.
__global__ void safe_c3_commit_once_kernel(
    TN* node_list, float* max_distance, int* id_list,
    const int* route_nodes, const int* expected_sizes, int route_count,
    const int* max_nodes, const float* expected_max, const float* next_max,
    int max_count, int write_slot, int stable_id, int* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (id_list[write_slot] != -1) {
    *status = 1;
    return;
  }
  for (int i = 0; i < route_count; ++i) {
    if (node_list[route_nodes[i]].size != expected_sizes[i]) {
      *status = 2;
      return;
    }
  }
  for (int i = 0; i < max_count; ++i) {
    if (__float_as_uint(max_distance[max_nodes[i]]) !=
        __float_as_uint(expected_max[i])) {
      *status = 3;
      return;
    }
  }

  // Planned quiescent visibility order: payload, envelope, ancestor
  // cardinalities, then leaf cardinality.  A production concurrent protocol
  // would require separate synchronization semantics and is out of scope.
  id_list[write_slot] = stable_id;
  for (int i = 0; i < max_count; ++i) {
    max_distance[max_nodes[i]] = next_max[i];
  }
  for (int i = 0; i + 1 < route_count; ++i) {
    node_list[route_nodes[i]].size = expected_sizes[i] + 1;
  }
  node_list[route_nodes[route_count - 1]].size =
      expected_sizes[route_count - 1] + 1;
  *status = 0;
}

struct SafeArgs {
  std::string bundle;
  std::string output;
  std::string summary;
  std::string initial_snapshot;
  std::string final_snapshot;
  int candidate_id = -1;
  int expected_leaf = -1;
};

SafeArgs parse_safe_args(int argc, char** argv) {
  SafeArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    auto value = [&]() -> std::string {
      if (++i >= argc) fail("missing value after " + key);
      return argv[i];
    };
    if (key == "--bundle") args.bundle = value();
    else if (key == "--out") args.output = value();
    else if (key == "--summary") args.summary = value();
    else if (key == "--initial-snapshot") args.initial_snapshot = value();
    else if (key == "--final-snapshot") args.final_snapshot = value();
    else if (key == "--candidate-id") args.candidate_id = std::stoi(value());
    else if (key == "--expected-leaf") args.expected_leaf = std::stoi(value());
    else if (key == "--help") {
      std::cout
          << "usage: GTS_safe_c3_ancestor_fence --bundle BUNDLE --out ENGINE.jsonl "
          << "--summary SUMMARY.json --initial-snapshot INITIAL.json "
          << "--final-snapshot FINAL.json --candidate-id 4102 --expected-leaf 604\n";
      std::exit(0);
    } else {
      fail("unknown Safe-C3 argument " + key);
    }
  }
  if (args.bundle.empty() || args.output.empty() || args.summary.empty() ||
      args.initial_snapshot.empty() || args.final_snapshot.empty() ||
      args.candidate_id < 0 || args.expected_leaf < 0) {
    fail("all Safe-C3 arguments are required");
  }
  if (args.candidate_id != kSafeCandidateStableId ||
      args.expected_leaf != kSafeExpectedLeaf) {
    fail("this preregistered harness admits only candidate=4102 and leaf=604");
  }
  return args;
}

void write_atomic_no_overwrite(const std::string& path, const std::string& body) {
  if (path.empty()) fail("empty output path");
  {
    std::ifstream existing(path, std::ios::binary);
    if (existing.good()) fail("refusing to overwrite existing output " + path);
  }
  // The guarded launcher owns an otherwise-empty staging directory and publishes
  // that directory only after this runner and the independent validator pass.
  // This local temp-to-rename step prevents readers in that staging directory
  // from observing a partially written JSON receipt.
  const std::string temp = path + ".tmp_safe_c3";
  {
    std::ifstream stale(temp, std::ios::binary);
    if (stale.good()) fail("stale output staging file exists " + temp);
  }
  {
    std::ofstream staged(temp, std::ios::binary | std::ios::out);
    if (!staged) fail("cannot create output staging file " + temp);
    staged.write(body.data(), static_cast<std::streamsize>(body.size()));
    staged.flush();
    staged.close();
    if (!staged) fail("cannot finish output staging file " + temp);
  }
  if (std::rename(temp.c_str(), path.c_str()) != 0) {
    std::remove(temp.c_str());
    fail("cannot publish output receipt " + path);
  }
}

std::string safe_json_number(double value) {
  if (!std::isfinite(value)) fail("non-finite value cannot be emitted as JSON");
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

void emit_safe_float_array(std::ostream& out, const std::vector<float>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << safe_json_number(values[i]);
  }
  out << ']';
}

struct C3Snapshot {
  Snapshot logical;
  std::array<int, 3> data_info{};
  int id_capacity = 0;
  std::vector<int> id_slots;
  std::vector<short> data_rows;
};

C3Snapshot capture_c3_snapshot(const Runtime& runtime, std::size_t pool_values) {
  C3Snapshot result;
  result.logical = capture_snapshot(runtime);
  CUdeviceptr base = 0;
  std::size_t bytes = 0;
  const CUresult address_status = cuMemGetAddressRange(
      &base, &bytes, reinterpret_cast<CUdeviceptr>(runtime.id_list));
  if (address_status != CUDA_SUCCESS ||
      base != reinterpret_cast<CUdeviceptr>(runtime.id_list) ||
      bytes == 0 || bytes % sizeof(int) != 0) {
    fail("could not measure native id_list allocation exactly");
  }
  if (bytes / sizeof(int) > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    fail("id_list allocation exceeds int range");
  }
  result.id_capacity = static_cast<int>(bytes / sizeof(int));
  result.id_slots.resize(static_cast<std::size_t>(result.id_capacity));
  result.data_rows.resize(pool_values);
  C3_CUDA(cudaMemcpy(result.data_info.data(), runtime.data_info, 3 * sizeof(int),
                     cudaMemcpyDeviceToHost));
  C3_CUDA(cudaMemcpy(result.id_slots.data(), runtime.id_list,
                     result.id_slots.size() * sizeof(int), cudaMemcpyDeviceToHost));
  C3_CUDA(cudaMemcpy(result.data_rows.data(), runtime.data_d,
                     result.data_rows.size() * sizeof(short), cudaMemcpyDeviceToHost));
  return result;
}

std::vector<int> leaf_nodes_by_lid(const C3Snapshot& snapshot) {
  std::vector<int> leaves;
  for (int node = 0; node < static_cast<int>(snapshot.logical.nodes.size()); ++node) {
    if (snapshot.logical.empty[node] == 0 && snapshot.logical.nodes[node].is_leaf == 1) {
      leaves.push_back(node);
    }
  }
  std::sort(leaves.begin(), leaves.end(), [&](int a, int b) {
    const int la = snapshot.logical.nodes[a].lid;
    const int lb = snapshot.logical.nodes[b].lid;
    return la != lb ? la < lb : a < b;
  });
  return leaves;
}

int leaf_span_end(const C3Snapshot& snapshot, int leaf_id) {
  const auto leaves = leaf_nodes_by_lid(snapshot);
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    if (leaves[i] != leaf_id) continue;
    const int begin = snapshot.logical.nodes[leaf_id].lid;
    const int end = i + 1 < leaves.size()
                        ? snapshot.logical.nodes[leaves[i + 1]].lid
                        : snapshot.id_capacity;
    if (begin < 0 || end <= begin || end > snapshot.id_capacity) {
      fail("invalid measured physical leaf span");
    }
    return end;
  }
  fail("requested leaf is absent from physical layout");
}

void validate_leaf_spans(const C3Snapshot& snapshot, bool initial_layout) {
  const auto leaves = leaf_nodes_by_lid(snapshot);
  if (leaves.empty()) fail("tree has no live leaves");
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    const TN& leaf = snapshot.logical.nodes[leaves[i]];
    const int begin = leaf.lid;
    const int end = i + 1 < leaves.size()
                        ? snapshot.logical.nodes[leaves[i + 1]].lid
                        : snapshot.id_capacity;
    if (begin < 0 || end <= begin || end > snapshot.id_capacity ||
        leaf.size < 0 || leaf.size > end - begin) {
      fail("invalid leaf logical/physical span at node " + std::to_string(leaves[i]));
    }
    if (initial_layout && end - begin != leaf.size + LEAF_PAD_SLOTS) {
      fail("initial leaf padding ABI is not the archived fixed-span layout");
    }
  }
}

void validate_parent_cardinality(const C3Snapshot& snapshot) {
  const int order = snapshot.logical.tree_order;
  for (int node = 0; node < static_cast<int>(snapshot.logical.nodes.size()); ++node) {
    if (snapshot.logical.empty[node] != 0 || snapshot.logical.nodes[node].is_leaf == 1) continue;
    int child_count = 0;
    long long sum = 0;
    for (int slot = 0; slot < order; ++slot) {
      const int child = node * order + slot + 1;
      if (child >= static_cast<int>(snapshot.logical.nodes.size()) ||
          snapshot.logical.empty[child] != 0) {
        continue;
      }
      ++child_count;
      sum += snapshot.logical.nodes[child].size;
    }
    if (child_count == 0 ||
        sum != static_cast<long long>(snapshot.logical.nodes[node].size)) {
      fail("subtree cardinality invariant fails at node " + std::to_string(node));
    }
  }
}

void validate_membership(const C3Snapshot& snapshot, int base_n, int pool_n,
                         int candidate_id, bool candidate_must_be_live) {
  std::vector<int> counts(static_cast<std::size_t>(pool_n), 0);
  long long logical_total = 0;
  for (int leaf_id : leaf_nodes_by_lid(snapshot)) {
    const TN& leaf = snapshot.logical.nodes[leaf_id];
    const int end = leaf_span_end(snapshot, leaf_id);
    if (leaf.lid + leaf.size > end) fail("leaf exceeds its measured physical span");
    for (int did = 0; did < leaf.size; ++did) {
      const int id = snapshot.id_slots[static_cast<std::size_t>(leaf.lid + did)];
      if (id < 0 || id >= pool_n) {
        fail("invalid logical leaf payload ID at leaf " + std::to_string(leaf_id));
      }
      ++counts[static_cast<std::size_t>(id)];
      ++logical_total;
    }
  }
  const long long expected_total = static_cast<long long>(base_n) +
                                   (candidate_must_be_live ? 1LL : 0LL);
  if (logical_total != expected_total ||
      snapshot.logical.nodes[0].size != expected_total) {
    fail("root cardinality disagrees with reconstructed leaf membership");
  }
  for (int id = 0; id < base_n; ++id) {
    if (counts[static_cast<std::size_t>(id)] != 1) {
      fail("base ID multiplicity is not exactly one");
    }
  }
  for (int id = base_n; id < pool_n; ++id) {
    const int expected = candidate_must_be_live && id == candidate_id ? 1 : 0;
    if (counts[static_cast<std::size_t>(id)] != expected) {
      fail("inactive/candidate payload membership violates the single-object contract");
    }
  }
}

std::vector<std::vector<int>> reconstruct_subtree_members(
    const C3Snapshot& snapshot, int pool_n) {
  const int count = static_cast<int>(snapshot.logical.nodes.size());
  std::vector<std::vector<int>> members(static_cast<std::size_t>(count));
  std::vector<int> state(static_cast<std::size_t>(count), 0);
  std::function<void(int)> visit = [&](int node) {
    if (snapshot.logical.empty[node] != 0) return;
    if (state[static_cast<std::size_t>(node)] == 1) {
      fail("cycle encountered in heap tree reconstruction");
    }
    if (state[static_cast<std::size_t>(node)] == 2) return;
    state[static_cast<std::size_t>(node)] = 1;
    const TN& current = snapshot.logical.nodes[node];
    if (current.is_leaf == 1) {
      const int end = leaf_span_end(snapshot, node);
      if (current.lid < 0 || current.lid + current.size > end) {
        fail("invalid leaf during subtree reconstruction");
      }
      for (int did = 0; did < current.size; ++did) {
        const int id = snapshot.id_slots[static_cast<std::size_t>(current.lid + did)];
        if (id < 0 || id >= pool_n) fail("invalid payload during subtree reconstruction");
        members[static_cast<std::size_t>(node)].push_back(id);
      }
    } else {
      int children = 0;
      for (int slot = 0; slot < snapshot.logical.tree_order; ++slot) {
        const int child = node * snapshot.logical.tree_order + slot + 1;
        if (child >= count || snapshot.logical.empty[child] != 0) continue;
        ++children;
        visit(child);
        const auto& child_members = members[static_cast<std::size_t>(child)];
        members[static_cast<std::size_t>(node)].insert(
            members[static_cast<std::size_t>(node)].end(),
            child_members.begin(), child_members.end());
      }
      if (children == 0) fail("non-leaf has no live children during reconstruction");
    }
    if (static_cast<int>(members[static_cast<std::size_t>(node)].size()) != current.size) {
      fail("reconstructed subtree member count disagrees with TN.size");
    }
    state[static_cast<std::size_t>(node)] = 2;
  };
  visit(0);
  return members;
}

void validate_cover_and_sibling_fences(const C3Snapshot& snapshot,
                                       const std::vector<std::int16_t>& pool,
                                       int dimension, int pool_n,
                                       const std::vector<int>& strict_fence_nodes) {
  const auto members = reconstruct_subtree_members(snapshot, pool_n);
  std::set<int> strict_nodes(strict_fence_nodes.begin(), strict_fence_nodes.end());
  for (int node = 1; node < static_cast<int>(snapshot.logical.nodes.size()); ++node) {
    if (snapshot.logical.empty[node] != 0) continue;
    const TN& current = snapshot.logical.nodes[node];
    if (current.pid < 0 || current.pid >= pool_n) {
      fail("node pivot ID is invalid");
    }
    float actual_max = 0.0F;
    for (int id : members[static_cast<std::size_t>(node)]) {
      actual_max = std::max(actual_max, l2_routing_float(pool, dimension, id, current.pid));
    }
    const float stored = snapshot.logical.max_distance[static_cast<std::size_t>(node)];
    const float tolerance = std::max(kCoverageTolerance, std::fabs(actual_max) * 1.0e-6F);
    if (stored + tolerance < actual_max) {
      fail("stored max_dis_d fails recomputed subtree coverage at node " +
           std::to_string(node));
    }
  }

  for (int parent = 0; parent < static_cast<int>(snapshot.logical.nodes.size()); ++parent) {
    if (snapshot.logical.empty[parent] != 0) continue;
    for (int slot = 0; slot + 1 < snapshot.logical.tree_order; ++slot) {
      const int child = parent * snapshot.logical.tree_order + slot + 1;
      const int next = child + 1;
      if (child >= static_cast<int>(snapshot.logical.nodes.size()) ||
          next >= static_cast<int>(snapshot.logical.nodes.size()) ||
          snapshot.logical.empty[child] != 0 || snapshot.logical.empty[next] != 0) {
        continue;
      }
      const float left_max = snapshot.logical.max_distance[static_cast<std::size_t>(child)];
      const float right_min = snapshot.logical.nodes[next].min_dis;
      const float margin = right_min - left_max;
      if (margin < -kStrictEpsilon) {
        fail("stored max_dis_d crosses successor min_dis at child " +
             std::to_string(child));
      }
      if (strict_nodes.count(child) != 0 && !(margin > kStrictEpsilon)) {
        fail("updated max_dis_d lacks strict successor fence margin");
      }
    }
  }
}

struct SafeC3Hop {
  int parent = -1;
  int child = -1;
  int pivot = -1;
  int native_matching_children = 0;
  bool has_next = false;
  float distance = 0.0F;
  float min_before = 0.0F;
  float max_before = 0.0F;
  float next_min = 0.0F;
  float max_after = 0.0F;
  bool changes_max = false;
};

struct SafeC3Plan {
  int candidate_id = -1;
  int leaf_id = -1;
  int leaf_old_size = -1;
  int write_slot = -1;
  int physical_span_end = -1;
  std::vector<int> route_nodes;
  std::vector<int> expected_sizes;
  std::vector<SafeC3Hop> hops;
  std::vector<int> max_nodes;
  std::vector<float> max_before;
  std::vector<float> max_after;
};

std::vector<SafeC3Hop> derive_native_sibling_min_route(
    const C3Snapshot& snapshot, const std::vector<std::int16_t>& pool,
    int dimension, int candidate_id) {
  std::vector<SafeC3Hop> hops;
  int parent = 0;
  for (int level = 0; level < snapshot.logical.tree_height - 1; ++level) {
    int pivot = -1;
    for (int slot = 0; slot < snapshot.logical.tree_order; ++slot) {
      const int child = parent * snapshot.logical.tree_order + slot + 1;
      if (child >= static_cast<int>(snapshot.logical.nodes.size()) ||
          snapshot.logical.empty[child] != 0) {
        continue;
      }
      if (pivot < 0) pivot = snapshot.logical.nodes[child].pid;
      else if (pivot != snapshot.logical.nodes[child].pid) {
        fail("siblings do not share a routing pivot");
      }
    }
    if (pivot < 0 || pivot == candidate_id) fail("invalid candidate routing pivot");
    const float distance = l2_routing_float(pool, dimension, candidate_id, pivot);
    int selected = -1;
    int matches = 0;
    SafeC3Hop selected_hop;
    for (int slot = 0; slot < snapshot.logical.tree_order; ++slot) {
      const int child = parent * snapshot.logical.tree_order + slot + 1;
      if (child >= static_cast<int>(snapshot.logical.nodes.size()) ||
          snapshot.logical.empty[child] != 0) {
        continue;
      }
      const bool has_next = slot + 1 < snapshot.logical.tree_order;
      float next_min = 0.0F;
      if (has_next) {
        const int next = child + 1;
        if (next >= static_cast<int>(snapshot.logical.nodes.size()) ||
            snapshot.logical.empty[next] != 0 ||
            (next - 1) / snapshot.logical.tree_order != parent) {
          fail("native successor-min branch ABI is missing a sibling");
        }
        next_min = snapshot.logical.nodes[next].min_dis;
      }
      const float lower = snapshot.logical.nodes[child].min_dis;
      const float old_max = snapshot.logical.max_distance[static_cast<std::size_t>(child)];
      const bool native_route = distance > lower + kStrictEpsilon &&
          (!has_next || distance < next_min - kStrictEpsilon);
      if (!native_route) continue;
      ++matches;
      selected = child;
      selected_hop.parent = parent;
      selected_hop.child = child;
      selected_hop.pivot = pivot;
      selected_hop.has_next = has_next;
      selected_hop.distance = distance;
      selected_hop.min_before = lower;
      selected_hop.max_before = old_max;
      selected_hop.next_min = next_min;
      selected_hop.max_after = std::max(old_max, distance);
      selected_hop.changes_max =
          float_bits(selected_hop.max_before) != float_bits(selected_hop.max_after);
    }
    if (matches != 1 || selected < 0) {
      fail("candidate lacks a unique strict native sibling-min route");
    }
    selected_hop.native_matching_children = matches;
    if (selected_hop.has_next &&
        !(selected_hop.max_after < selected_hop.next_min - kStrictEpsilon)) {
      fail("candidate would violate successor min_dis fence after coverage update");
    }
    hops.push_back(selected_hop);
    if (snapshot.logical.nodes[selected].is_leaf == 1) break;
    parent = selected;
  }
  if (hops.empty() || snapshot.logical.nodes[hops.back().child].is_leaf != 1) {
    fail("native sibling-min route did not terminate at a leaf");
  }
  return hops;
}

SafeC3Plan prepare_safe_c3_plan(const C3Snapshot& initial,
                                const std::vector<std::int16_t>& pool,
                                int dimension, int base_n, int pool_n,
                                int candidate_id, int expected_leaf) {
  if (candidate_id != kSafeCandidateStableId || expected_leaf != kSafeExpectedLeaf ||
      candidate_id < base_n || candidate_id >= pool_n) {
    fail("candidate binding does not match preregistration");
  }
  for (int id = 0; id < base_n; ++id) {
    const std::size_t a = static_cast<std::size_t>(id) * dimension;
    const std::size_t b = static_cast<std::size_t>(candidate_id) * dimension;
    if (std::equal(pool.begin() + static_cast<std::ptrdiff_t>(a),
                   pool.begin() + static_cast<std::ptrdiff_t>(a + dimension),
                   pool.begin() + static_cast<std::ptrdiff_t>(b))) {
      fail("candidate vector has a zero-distance duplicate in the active base");
    }
  }

  SafeC3Plan plan;
  plan.candidate_id = candidate_id;
  plan.hops = derive_native_sibling_min_route(initial, pool, dimension, candidate_id);
  plan.route_nodes.push_back(0);
  for (const auto& hop : plan.hops) plan.route_nodes.push_back(hop.child);
  plan.leaf_id = plan.route_nodes.back();
  if (plan.leaf_id != expected_leaf ||
      plan.route_nodes.size() != kSafeExpectedRoute.size() ||
      !std::equal(plan.route_nodes.begin(), plan.route_nodes.end(),
                  kSafeExpectedRoute.begin())) {
    fail("frozen route differs from preregistered 0->5->60->604 contract");
  }

  const TN& leaf = initial.logical.nodes[plan.leaf_id];
  plan.leaf_old_size = leaf.size;
  plan.physical_span_end = leaf_span_end(initial, plan.leaf_id);
  plan.write_slot = leaf.lid + leaf.size;
  if (leaf.is_leaf != 1 || leaf.size < 0 || leaf.size + 1 > MAX_SIZE ||
      plan.write_slot < leaf.lid || plan.write_slot >= plan.physical_span_end ||
      initial.id_slots[static_cast<std::size_t>(plan.write_slot)] != -1) {
    fail("target leaf fails static MAX_SIZE or measured physical-slot admission");
  }
  for (int node : plan.route_nodes) {
    plan.expected_sizes.push_back(initial.logical.nodes[node].size);
  }
  for (const auto& hop : plan.hops) {
    if (hop.changes_max) {
      plan.max_nodes.push_back(hop.child);
      plan.max_before.push_back(hop.max_before);
      plan.max_after.push_back(hop.max_after);
    }
  }
  if (plan.max_nodes.size() != 1 || plan.max_nodes[0] != 604 ||
      float_bits(plan.max_before[0]) != 1193758447U ||
      float_bits(plan.max_after[0]) != 1193909109U ||
      plan.write_slot != 33564 || plan.physical_span_end != 33628 ||
      plan.expected_sizes != std::vector<int>({4096, 409, 49, 4})) {
    fail("precomputed Safe-C3 4102 mutation contract drifted");
  }
  return plan;
}

void commit_safe_c3_once(Runtime& runtime, const SafeC3Plan& plan) {
  int* route = nullptr;
  int* expected_sizes = nullptr;
  int* max_nodes = nullptr;
  float* max_before = nullptr;
  float* max_after = nullptr;
  int* status = nullptr;
  try {
    C3_CUDA(cudaMallocManaged(&route, plan.route_nodes.size() * sizeof(int)));
    C3_CUDA(cudaMallocManaged(&expected_sizes, plan.expected_sizes.size() * sizeof(int)));
    C3_CUDA(cudaMallocManaged(&max_nodes, plan.max_nodes.size() * sizeof(int)));
    C3_CUDA(cudaMallocManaged(&max_before, plan.max_before.size() * sizeof(float)));
    C3_CUDA(cudaMallocManaged(&max_after, plan.max_after.size() * sizeof(float)));
    C3_CUDA(cudaMallocManaged(&status, sizeof(int)));
    std::copy(plan.route_nodes.begin(), plan.route_nodes.end(), route);
    std::copy(plan.expected_sizes.begin(), plan.expected_sizes.end(), expected_sizes);
    std::copy(plan.max_nodes.begin(), plan.max_nodes.end(), max_nodes);
    std::copy(plan.max_before.begin(), plan.max_before.end(), max_before);
    std::copy(plan.max_after.begin(), plan.max_after.end(), max_after);
    *status = -1;
    C3_CUDA(cudaDeviceSynchronize());
    safe_c3_commit_once_kernel<<<1, 1>>>(
        runtime.node_list, max_dis_d, runtime.id_list,
        route, expected_sizes, static_cast<int>(plan.route_nodes.size()),
        max_nodes, max_before, max_after, static_cast<int>(plan.max_nodes.size()),
        plan.write_slot, plan.candidate_id, status);
    C3_CUDA(cudaDeviceSynchronize());
    C3_CUDA(cudaGetLastError());
    if (*status != 0) {
      fail("Safe-C3 device mutation precondition rejected with code " +
           std::to_string(*status));
    }
    C3_CUDA(cudaFree(route)); route = nullptr;
    C3_CUDA(cudaFree(expected_sizes)); expected_sizes = nullptr;
    C3_CUDA(cudaFree(max_nodes)); max_nodes = nullptr;
    C3_CUDA(cudaFree(max_before)); max_before = nullptr;
    C3_CUDA(cudaFree(max_after)); max_after = nullptr;
    C3_CUDA(cudaFree(status)); status = nullptr;
  } catch (...) {
    if (route) cudaFree(route);
    if (expected_sizes) cudaFree(expected_sizes);
    if (max_nodes) cudaFree(max_nodes);
    if (max_before) cudaFree(max_before);
    if (max_after) cudaFree(max_after);
    if (status) cudaFree(status);
    throw;
  }
}

struct D2HAudit {
  int changed_id_slots = 0;
  int changed_size_nodes = 0;
  int changed_max_nodes = 0;
  bool data_info_unchanged = false;
  bool data_rows_unchanged = false;
  bool topology_unchanged = false;
  bool membership_ok = false;
  bool cardinality_ok = false;
  bool coverage_and_fences_ok = false;
  float updated_fence_margin = 0.0F;
};

D2HAudit audit_postcommit(const C3Snapshot& before, const C3Snapshot& after,
                          const SafeC3Plan& plan,
                          const std::vector<std::int16_t>& pool,
                          int dimension, int base_n, int pool_n) {
  if (before.logical.nodes.size() != after.logical.nodes.size() ||
      before.id_capacity != after.id_capacity ||
      before.id_slots.size() != after.id_slots.size() ||
      before.logical.tree_height != after.logical.tree_height ||
      before.logical.tree_order != after.logical.tree_order) {
    fail("postcommit topology/allocation changed");
  }
  D2HAudit audit;
  audit.data_info_unchanged = before.data_info == after.data_info;
  audit.data_rows_unchanged = before.data_rows == after.data_rows;
  if (!audit.data_info_unchanged || !audit.data_rows_unchanged) {
    fail("Safe-C3 modified frozen base construction metadata or raw data pool");
  }

  const std::set<int> allowed_sizes(plan.route_nodes.begin(), plan.route_nodes.end());
  const std::set<int> allowed_max(plan.max_nodes.begin(), plan.max_nodes.end());
  for (std::size_t node = 0; node < before.logical.nodes.size(); ++node) {
    const TN& a = before.logical.nodes[node];
    const TN& b = after.logical.nodes[node];
    if (before.logical.empty[node] != after.logical.empty[node] ||
        a.pid != b.pid || float_bits(a.min_dis) != float_bits(b.min_dis) ||
        a.lid != b.lid || a.is_leaf != b.is_leaf) {
      fail("immutable topology field changed at node " + std::to_string(node));
    }
    const int expected_size = a.size + (allowed_sizes.count(static_cast<int>(node)) ? 1 : 0);
    if (b.size != expected_size) {
      fail("TN.size change violates the exact Safe-C3 path whitelist");
    }
    if (a.size != b.size) ++audit.changed_size_nodes;
    float expected_max = before.logical.max_distance[node];
    for (std::size_t j = 0; j < plan.max_nodes.size(); ++j) {
      if (plan.max_nodes[j] == static_cast<int>(node)) expected_max = plan.max_after[j];
    }
    if (float_bits(after.logical.max_distance[node]) != float_bits(expected_max)) {
      fail("max_dis_d change violates the exact Safe-C3 whitelist");
    }
    if (float_bits(before.logical.max_distance[node]) !=
        float_bits(after.logical.max_distance[node])) {
      ++audit.changed_max_nodes;
    }
  }
  audit.topology_unchanged = true;
  if (audit.changed_size_nodes != static_cast<int>(plan.route_nodes.size()) ||
      audit.changed_max_nodes != static_cast<int>(plan.max_nodes.size())) {
    fail("postcommit mutation count differs from the fixed one-object plan");
  }

  for (std::size_t slot = 0; slot < before.id_slots.size(); ++slot) {
    const int a = before.id_slots[slot];
    const int b = after.id_slots[slot];
    if (a == b) continue;
    ++audit.changed_id_slots;
    if (static_cast<int>(slot) != plan.write_slot || a != -1 ||
        b != plan.candidate_id) {
      fail("raw id_list difference violates the exact Safe-C3 slot whitelist");
    }
  }
  if (audit.changed_id_slots != 1) fail("Safe-C3 must change exactly one raw id_list slot");

  validate_leaf_spans(after, false);
  validate_membership(after, base_n, pool_n, plan.candidate_id, true);
  audit.membership_ok = true;
  validate_parent_cardinality(after);
  audit.cardinality_ok = true;
  validate_cover_and_sibling_fences(after, pool, dimension, pool_n, plan.max_nodes);
  audit.coverage_and_fences_ok = true;

  const auto route_after =
      derive_native_sibling_min_route(after, pool, dimension, plan.candidate_id);
  std::vector<int> after_nodes{0};
  for (const auto& hop : route_after) after_nodes.push_back(hop.child);
  if (after_nodes != plan.route_nodes) {
    fail("postcommit native sibling-min route differs from preregistered route");
  }
  audit.updated_fence_margin =
      after.logical.nodes[605].min_dis - after.logical.max_distance[604];
  if (!(audit.updated_fence_margin > kStrictEpsilon)) {
    fail("updated ancestor fence margin is not strictly positive");
  }
  return audit;
}

struct ExactRange {
  std::vector<int> ids;
  std::vector<std::int64_t> squared_distances;
};

ExactRange exact_full_live_range(const std::vector<std::int16_t>& pool, int dimension,
                                 int query_id, const std::vector<std::uint8_t>& active,
                                 std::int64_t radius_squared) {
  ExactRange result;
  const std::size_t qoff = static_cast<std::size_t>(query_id) * dimension;
  for (int id = 0; id < static_cast<int>(active.size()); ++id) {
    if (active[static_cast<std::size_t>(id)] == 0) continue;
    std::int64_t sum = 0;
    const std::size_t off = static_cast<std::size_t>(id) * dimension;
    for (int d = 0; d < dimension; ++d) {
      const std::int64_t delta =
          static_cast<std::int64_t>(pool[off + d]) - pool[qoff + d];
      sum += delta * delta;
    }
    if (sum <= radius_squared) {
      result.ids.push_back(id);
      result.squared_distances.push_back(sum);
    }
  }
  std::vector<std::size_t> order(result.ids.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    if (result.squared_distances[a] != result.squared_distances[b]) {
      return result.squared_distances[a] < result.squared_distances[b];
    }
    return result.ids[a] < result.ids[b];
  });
  ExactRange sorted;
  for (std::size_t pos : order) {
    sorted.ids.push_back(result.ids[pos]);
    sorted.squared_distances.push_back(result.squared_distances[pos]);
  }
  return sorted;
}

std::string snapshot_json(const C3Snapshot& snapshot) {
  std::ostringstream out;
  out << "{\"schema\":\"safe-c3-full-d2h-snapshot-v1\","
      << "\"tree_height\":" << snapshot.logical.tree_height
      << ",\"tree_order\":" << snapshot.logical.tree_order
      << ",\"max_size\":" << snapshot.logical.max_size
      << ",\"leaf_pad_slots\":" << snapshot.logical.leaf_pad_slots
      << ",\"data_info\":[" << snapshot.data_info[0] << ','
      << snapshot.data_info[1] << ',' << snapshot.data_info[2] << ']'
      << ",\"id_list_capacity\":" << snapshot.id_capacity
      << ",\"id_slots\":";
  emit_int_array(out, snapshot.id_slots);
  out << ",\"nodes\":[";
  for (std::size_t i = 0; i < snapshot.logical.nodes.size(); ++i) {
    if (i) out << ',';
    const TN& node = snapshot.logical.nodes[i];
    out << "{\"node_id\":" << i << ",\"empty\":" << snapshot.logical.empty[i]
        << ",\"pid\":" << node.pid
        << ",\"min_dis_bits\":" << float_bits(node.min_dis)
        << ",\"max_dis_bits\":" << float_bits(snapshot.logical.max_distance[i])
        << ",\"size\":" << node.size << ",\"lid\":" << node.lid
        << ",\"is_leaf\":" << node.is_leaf << '}';
  }
  out << "]}\n";
  return out.str();
}

void emit_safe_plan(std::ostream& out, const SafeC3Plan& plan) {
  out << "{\"record\":\"plan\",\"candidate_id\":" << plan.candidate_id
      << ",\"candidate_payload_mode\":\"preloaded_inactive_raw_pool\","
      << "\"direct_mutation_scope\":\"single_quiescent_native_tree_write\","
      << "\"route_nodes\":";
  emit_int_array(out, plan.route_nodes);
  out << ",\"expected_sizes_before\":";
  emit_int_array(out, plan.expected_sizes);
  out << ",\"expected_sizes_after\":[";
  for (std::size_t i = 0; i < plan.expected_sizes.size(); ++i) {
    if (i) out << ',';
    out << plan.expected_sizes[i] + 1;
  }
  out << "],\"leaf_id\":" << plan.leaf_id
      << ",\"leaf_old_size\":" << plan.leaf_old_size
      << ",\"write_slot\":" << plan.write_slot
      << ",\"physical_span_end\":" << plan.physical_span_end
      << ",\"max_update_nodes\":";
  emit_int_array(out, plan.max_nodes);
  out << ",\"max_before\":";
  emit_safe_float_array(out, plan.max_before);
  out << ",\"max_after\":";
  emit_safe_float_array(out, plan.max_after);
  out << ",\"hops\":[";
  for (std::size_t i = 0; i < plan.hops.size(); ++i) {
    if (i) out << ',';
    const auto& h = plan.hops[i];
    out << "{\"parent\":" << h.parent << ",\"child\":" << h.child
        << ",\"pivot\":" << h.pivot
        << ",\"native_matching_children\":" << h.native_matching_children
        << ",\"has_next_sibling\":" << (h.has_next ? "true" : "false")
        << ",\"distance\":" << safe_json_number(h.distance)
        << ",\"min_before\":" << safe_json_number(h.min_before)
        << ",\"max_before\":" << safe_json_number(h.max_before)
        << ",\"next_min\":" << safe_json_number(h.next_min)
        << ",\"max_after\":" << safe_json_number(h.max_after)
        << ",\"changes_max\":" << (h.changes_max ? "true" : "false") << '}';
  }
  out << "]}\n";
}

void emit_d2h_audit(std::ostream& out, const D2HAudit& audit) {
  out << "{\"record\":\"postcommit_d2h_audit\","
      << "\"changed_id_slots\":" << audit.changed_id_slots
      << ",\"changed_size_nodes\":" << audit.changed_size_nodes
      << ",\"changed_max_nodes\":" << audit.changed_max_nodes
      << ",\"data_info_unchanged\":"
      << (audit.data_info_unchanged ? "true" : "false")
      << ",\"data_rows_unchanged\":"
      << (audit.data_rows_unchanged ? "true" : "false")
      << ",\"topology_unchanged\":"
      << (audit.topology_unchanged ? "true" : "false")
      << ",\"membership_ok\":" << (audit.membership_ok ? "true" : "false")
      << ",\"cardinality_ok\":" << (audit.cardinality_ok ? "true" : "false")
      << ",\"coverage_and_fences_ok\":"
      << (audit.coverage_and_fences_ok ? "true" : "false")
      << ",\"updated_node\":604"
      << ",\"updated_fence_margin\":"
      << safe_json_number(audit.updated_fence_margin)
      << ",\"coverage_tolerance\":" << safe_json_number(kCoverageTolerance)
      << "}\n";
}

void emit_native_knn(std::ostream& out, int candidate_id, int active_count_value,
                     const GtsResult& gts, const ExactResult& exact) {
  out << "{\"record\":\"native_knn_4102\",\"query_stable_id\":" << candidate_id
      << ",\"native_api\":\"searchIndexKnnV2_vector_topk_ids\","
      << "\"active_count\":" << active_count_value << ",\"gts_ids\":";
  emit_int_array(out, gts.ids);
  out << ",\"gts_distances\":";
  emit_safe_float_array(out, gts.distances);
  out << ",\"independent_exact_ids\":";
  emit_int_array(out, exact.ids);
  out << ",\"independent_exact_squared_distances\":";
  emit_i64_array(out, exact.squared_distances);
  out << ",\"k_boundary_tie\":" << (exact.boundary_tie ? "true" : "false")
      << "}\n";
}

void emit_exact_range_fallback(std::ostream& out, int candidate_id,
                               const ExactRange& range) {
  out << "{\"record\":\"exact_full_live_range_fallback\","
      << "\"query_stable_id\":" << candidate_id
      << ",\"radius_squared\":0"
      << ",\"scope\":\"exact_full_live_range_fallback_no_gts_receipt\","
      << "\"native_gts_range_executed\":false,"
      << "\"native_range_correctness_claim\":false,"
      << "\"exact_ids\":";
  emit_int_array(out, range.ids);
  out << ",\"exact_squared_distances\":";
  emit_i64_array(out, range.squared_distances);
  out << "}\n";
}

std::string pass_summary_json(const TraceHeader& header, const SafeC3Plan& plan,
                              const D2HAudit& audit) {
  std::ostringstream out;
  out << "{\"schema\":\"safe-c3-leaf-envelope-run-v1\","
      << "\"status\":\"PASS\","
      << "\"scope\":\"one preloaded inactive raw-pool candidate; strict sibling-min admission; "
      << "path cardinality and required ancestor envelope update; D2H audit; native vector KNN\","
      << "\"candidate_id\":" << plan.candidate_id
      << ",\"base_n\":" << header.base_n
      << ",\"live_n\":" << (header.base_n + 1)
      << ",\"dimension\":" << header.dimension
      << ",\"k\":" << header.k
      << ",\"route_nodes\":";
  emit_int_array(out, plan.route_nodes);
  out << ",\"updated_max_nodes\":";
  emit_int_array(out, plan.max_nodes);
  out << ",\"updated_fence_margin\":"
      << safe_json_number(audit.updated_fence_margin)
      << ",\"native_range_correctness_claim\":false,"
      << "\"limitations\":["
      << "\"single candidate only\","
      << "\"preloaded inactive payload only\","
      << "\"no deletion\","
      << "\"no rebuild\","
      << "\"no concurrency\","
      << "\"no native range traversal claim\","
      << "\"no performance claim\","
      << "\"not a generic C3 correctness claim\""
      << "]}\n";
  return out.str();
}

std::string fail_summary_json(const std::string& detail) {
  // The detail originates from controlled local exceptions; avoid embedding it
  // in JSON to keep failure publication strict and independent of escaping.
  (void)detail;
  return "{\"schema\":\"safe-c3-leaf-envelope-run-v1\","
         "\"status\":\"FAIL_RUNNER_EXCEPTION\","
         "\"detail\":\"runner_failed_before_pass_publication\"}\n";
}

}  // namespace

int main(int argc, char** argv) {
  Runtime runtime;
  SafeArgs args;
  TraceHeader header{};
  MetricEncodingPlan metric_plan{};
  bool have_args = false;
  try {
    args = parse_safe_args(argc, argv);
    have_args = true;
    header = read_header(args.bundle + "/trace.e1gtrc");
    if (header.base_n != 4096 || header.pool_n != 6144) {
      fail("frozen bundle cardinality differs from preregistered Safe-C3 fixture");
    }
    const std::size_t pool_values =
        static_cast<std::size_t>(header.pool_n) * header.dimension;
    const std::size_t query_values =
        static_cast<std::size_t>(header.query_n) * header.dimension;
    const auto pool = read_exact_binary<std::int16_t>(
        args.bundle + "/pool.i16", pool_values);
    const auto queries = read_exact_binary<std::int16_t>(
        args.bundle + "/queries.i16", query_values);
    metric_plan = make_metric_encoding_plan(pool, queries,
                                            static_cast<int>(header.dimension));
    DIS_CODE = 100;
    INFI_DIS = metric_plan.infi_dis;

    // The full 6144-row raw pool is physically allocated before tree build,
    // while data_info[1] stays 4096 so the native tree is constructed from
    // exactly the frozen base.  Candidate 4102 is therefore an inactive
    // preloaded payload, not a synthetic post-build data allocation.
    C3_CUDA(cudaMallocManaged(&runtime.data_info, 3 * sizeof(int)));
    runtime.data_info[0] = static_cast<int>(header.dimension);
    runtime.data_info[1] = static_cast<int>(header.base_n);
    runtime.data_info[2] = 2;
    C3_CUDA(cudaMallocManaged(&runtime.data_d, pool_values * sizeof(short)));
    for (std::size_t i = 0; i < pool_values; ++i) {
      runtime.data_d[i] = static_cast<short>(pool[i]);
    }
    indexConstru(runtime.data_d, nullptr, nullptr, runtime.data_info,
                 runtime.id_list, runtime.node_list, runtime.max_node_num,
                 runtime.tree_height, runtime.empty_list);
    C3_CUDA(cudaDeviceSynchronize());
    C3_CUDA(cudaGetLastError());
    dis_list = nullptr;
    split_list = nullptr;
    split_num = nullptr;
    pid_list = nullptr;
    runtime.base_count = static_cast<int>(header.base_n);
    if (!runtime.ready()) fail("native base tree construction did not complete");

    install_baseline_residual_mode_zero();
    if (read_residual_mode() != 0) {
      fail("could not verify residual-pruning baseline mode zero");
    }

    const C3Snapshot initial = capture_c3_snapshot(runtime, pool_values);
    if (initial.data_info != std::array<int, 3>{
          static_cast<int>(header.dimension), static_cast<int>(header.base_n), 2} ||
        initial.data_rows != std::vector<short>(pool.begin(), pool.end())) {
      fail("frozen raw-pool or construction data_info snapshot mismatch");
    }
    validate_leaf_spans(initial, true);
    validate_membership(initial, static_cast<int>(header.base_n),
                        static_cast<int>(header.pool_n), args.candidate_id, false);
    validate_parent_cardinality(initial);
    std::string invariant_detail;
    if (!verify_search_upper_invariant(initial.logical, &invariant_detail)) {
      fail("precommit native successor-min invariant failure: " + invariant_detail);
    }

    const SafeC3Plan plan = prepare_safe_c3_plan(
        initial, pool, static_cast<int>(header.dimension), static_cast<int>(header.base_n),
        static_cast<int>(header.pool_n), args.candidate_id, args.expected_leaf);

    commit_safe_c3_once(runtime, plan);
    const C3Snapshot final_snapshot = capture_c3_snapshot(runtime, pool_values);
    const D2HAudit audit = audit_postcommit(
        initial, final_snapshot, plan, pool, static_cast<int>(header.dimension),
        static_cast<int>(header.base_n), static_cast<int>(header.pool_n));

    std::vector<std::uint8_t> active(static_cast<std::size_t>(header.pool_n), 0);
    std::fill(active.begin(), active.begin() + header.base_n,
              static_cast<std::uint8_t>(1));
    active[static_cast<std::size_t>(args.candidate_id)] = 1;
    const GtsResult gts = run_real_static_topk(
        runtime, pool, static_cast<int>(header.dimension), args.candidate_id,
        static_cast<int>(header.k));
    const ExactResult exact = exact_topk(
        pool, static_cast<int>(header.dimension), args.candidate_id, active,
        static_cast<int>(header.k));
    validate_real_query(gts, exact, args.candidate_id);
    const ExactRange range = exact_full_live_range(
        pool, static_cast<int>(header.dimension), args.candidate_id, active, 0);
    if (range.ids != std::vector<int>({args.candidate_id}) ||
        range.squared_distances != std::vector<std::int64_t>({0})) {
      fail("full-live exact range fallback did not isolate the unique candidate");
    }

    std::ostringstream engine;
    engine << "{\"record\":\"meta\","
           << "\"schema\":\"safe-c3-leaf-envelope-engine-v1\","
           << "\"scope\":\"single Safe-C3 candidate; no generic C3, range, delete, "
           << "rebuild, concurrency, or performance claim\","
           << "\"candidate_id\":" << args.candidate_id
           << ",\"expected_leaf\":" << args.expected_leaf
           << ",\"base_n\":" << header.base_n
           << ",\"pool_n\":" << header.pool_n
           << ",\"dimension\":" << header.dimension
           << ",\"k\":" << header.k
           << ",\"residual_pruning_mode\":0,"
           << "\"residual_pruning_runtime_verified\":true,"
           << "\"archived_incremental_updater_used\":false,"
           << "\"native_range_correctness_claim\":false}\n";
    emit_safe_plan(engine, plan);
    emit_d2h_audit(engine, audit);
    emit_native_knn(engine, args.candidate_id, active_count(active), gts, exact);
    emit_exact_range_fallback(engine, args.candidate_id, range);

    write_atomic_no_overwrite(args.initial_snapshot, snapshot_json(initial));
    write_atomic_no_overwrite(args.final_snapshot, snapshot_json(final_snapshot));
    write_atomic_no_overwrite(args.output, engine.str());
    write_atomic_no_overwrite(args.summary, pass_summary_json(header, plan, audit));
    release_runtime(runtime);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    if (have_args && !args.summary.empty()) {
      try {
        write_atomic_no_overwrite(args.summary, fail_summary_json(error.what()));
      } catch (...) {
      }
    }
    release_runtime(runtime);
    return 2;
  }
}

