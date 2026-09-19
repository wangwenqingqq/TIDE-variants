// New host bridge; the included TIDE device kernel and runtime are unchanged.
#define main p1_unused_tide_main
#include "../keeper/tide_query_keeper.cu"
#undef main

namespace p1 {
thread_local std::string error;
struct Handle {
  MappedFile fp, ids, pc;
  HostDBView<4> host;
  Snapshot<4> snapshot;
  std::unique_ptr<QueryRuntime<4>> runtime;
  bool bounded;
  Handle(const std::string& root, bool use_bound)
      : fp(root+"/fp_u64x4.bin"), ids(root+"/id_i64.bin"),
        pc(root+"/popcnt_u16.bin"), host(make_host_view<4>(fp,ids,pc)),
        bounded(use_bound) {
    snapshot.runs.push_back({std::make_shared<DeviceRun<4>>(0,host),host});
    runtime = std::make_unique<QueryRuntime<4>>(0);
  }
};
}
extern "C" const char* p1_error() { return p1::error.c_str(); }
extern "C" void* p1_create(const char* root, int bounded) {
  try { return new p1::Handle(root,bounded!=0); }
  catch(const std::exception& e) { p1::error=e.what(); return nullptr; }
}
extern "C" void p1_destroy(void* handle) { delete static_cast<p1::Handle*>(handle); }
extern "C" std::int64_t p1_search(void* handle, const std::uint64_t* packed,
    unsigned num, unsigned den, std::uint64_t* output, std::uint64_t capacity) {
  try {
    auto& h=*static_cast<p1::Handle*>(handle);
    Query<4> q; q.id=packed[0]; q.pc=packed[5];
    if(!q.pc) throw std::runtime_error("zero query is outside the P1 contract");
    for(int i=0;i<4;++i) q.fp[i]=packed[i+1];
    auto result=h.runtime->run(h.snapshot,q,num,den,h.bounded);
    if(result.overflow) throw std::runtime_error("TIDE output overflow");
    std::uint64_t count=0;
    for(const auto& hit:result.hits) {
      if(hit.id==q.id) continue;
      if(count==capacity) throw std::runtime_error("bridge capacity overflow");
      output[count++]=hit.id;
    }
    return static_cast<std::int64_t>(count);
  } catch(const std::exception& e) { p1::error=e.what(); return -1; }
}
