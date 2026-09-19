// SVC-Q0 host adapter only. The copied native TIDE kernel/runtime is unchanged.
#define main svcq_unused_keeper_main
#include "../keeper/tide_query_keeper.cu"
#undef main

namespace svcq {
thread_local std::string error;
struct HostRun {
  MappedFile fp, ids, pc;
  HostDBView<4> host;
  explicit HostRun(const std::string& root)
      : fp(root + "/fp_u64x4.bin"), ids(root + "/id_i64.bin"),
        pc(root + "/popcnt_u16.bin"), host(make_host_view<4>(fp, ids, pc)) {}
};
struct Handle {
  std::vector<std::unique_ptr<HostRun>> hosts;
  Snapshot<4> snapshot;
  QueryRuntime<4> runtime;
  Handle(const std::string& root, const std::string& dataset,
         const std::string& layout) : runtime(0) {
    if (dataset != "real" && dataset != "fixture")
      throw std::runtime_error("invalid dataset");
    std::vector<std::string> parts;
    if (layout == "sharded") parts = {"base", "delta"};
    else if (layout == "base" || layout == "union") parts = {layout};
    else throw std::runtime_error("invalid layout");
    for (const auto& part : parts) {
      hosts.push_back(std::make_unique<HostRun>(root + "/" + dataset + "_" + part));
      const auto view = hosts.back()->host;
      snapshot.runs.push_back({std::make_shared<DeviceRun<4>>(0, view), view});
    }
  }
};
}
extern "C" const char* svcq_error() { return svcq::error.c_str(); }
extern "C" void* svcq_create(const char* root, const char* dataset, const char* layout) {
  try { return new svcq::Handle(root, dataset, layout); }
  catch (const std::exception& e) { svcq::error = e.what(); return nullptr; }
}
extern "C" void svcq_destroy(void* handle) { delete static_cast<svcq::Handle*>(handle); }
extern "C" std::int64_t svcq_search(void* handle, const std::uint64_t* packed,
    unsigned num, unsigned den, std::uint64_t* output, std::uint64_t capacity) {
  try {
    if (!handle || !packed || !output || !num || num > den || den > 65535)
      throw std::runtime_error("invalid query arguments");
    Query<4> query;
    query.id = packed[0]; query.pc = packed[5];
    if (!query.pc || packed[5] > 256)
      throw std::runtime_error("nonzero 256-bit query required");
    for (int i = 0; i < 4; ++i) query.fp[i] = packed[i + 1];
    auto& h = *static_cast<svcq::Handle*>(handle);
    auto result = h.runtime.run(h.snapshot, query, num, den, true);
    if (result.overflow || result.hits.size() > capacity)
      throw std::runtime_error("complete-output capacity exceeded");
    std::size_t count = 0;
    // Preserve self IDs and all distinct IDs with identical fingerprints.
    for (const auto& hit : result.hits) output[count++] = hit.id;
    return static_cast<std::int64_t>(count);
  } catch (const std::exception& e) { svcq::error = e.what(); return -1; }
}
