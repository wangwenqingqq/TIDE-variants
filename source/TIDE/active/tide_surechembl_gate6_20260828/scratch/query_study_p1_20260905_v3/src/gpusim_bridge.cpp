// New host bridge around the pinned, unmodified GPUSimilarity core.
#define main p1_unused_gpusim_main
#include "../keeper/gpusim_adapter_keeper.cpp"
#undef main
#include <memory>
namespace p1 {
thread_local std::string error;
struct Handle {
  // Destroy Qt while the host runtime is still alive, after the database.
  std::unique_ptr<QCoreApplication> app;
  std::vector<char> id_arena;
  std::unique_ptr<gpusim::FingerprintDB> database;
  explicit Handle(const std::string& root) {
    static int argc=1;
    static char label[]="p1_gpusim";
    static char* argv[]={label,nullptr};
    if(!QCoreApplication::instance()) app=std::make_unique<QCoreApplication>(argc,argv);
    const auto n=exact_file_size(root+"/id_i64.bin")/8;
    if(n>static_cast<std::uint64_t>(std::numeric_limits<int>::max())) fail("row overflow");
    ReadOnlyMap<std::uint64_t> fp(root+"/fp_u64x4.bin",n*4);
    ReadOnlyMap<std::uint64_t> ids(root+"/id_i64.bin",n);
    std::vector<std::vector<char>> blocks(1);
    blocks[0].resize(n*32); std::memcpy(blocks[0].data(),fp.data(),n*32);
    id_arena.resize(n*kIdHexBytes);
    std::vector<char*> identifiers; identifiers.reserve(n);
    for(std::uint64_t i=0;i<n;++i) {
      char* ptr=id_arena.data()+i*kIdHexBytes; encode_hex_id(ids[i],ptr);
      identifiers.push_back(ptr);
    }
    // The core consumes both vectors with swap; they must be distinct objects.
    std::vector<char*> smiles=identifiers;
    database=std::make_unique<gpusim::FingerprintDB>(256,static_cast<int>(n),
                  QString(),blocks,smiles,identifiers);
    database->copyToGPU(1);
  }
};
}
extern "C" const char* p1_error() { return p1::error.c_str(); }
extern "C" void* p1_create(const char* root, int) {
  try { return new p1::Handle(root); }
  catch(const std::exception& e) { p1::error=e.what(); return nullptr; }
}
extern "C" void p1_destroy(void* handle) { delete static_cast<p1::Handle*>(handle); }
extern "C" std::int64_t p1_search(void* handle, const std::uint64_t* packed,
    unsigned num,unsigned den,std::uint64_t* output,std::uint64_t capacity) {
  try {
    if(!packed[5]) fail("zero query is outside the P1 contract");
    auto& h=*static_cast<p1::Handle*>(handle);
    gpusim::Fingerprint query; query.resize(8);
    std::memcpy(query.data(),packed+1,32);
    std::vector<char*> smiles, ids; std::vector<float> scores;
    unsigned long count=0;
    h.database->search(query,QString(),10000,static_cast<float>(num)/den,
                       smiles,ids,scores,count);
    if(count>=10000 || count!=ids.size() || scores.size()!=ids.size())
      fail("GPUSimilarity count/cap inconsistency");
    std::uint64_t kept=0;
    for(const auto* ptr:ids) {
      auto id=decode_hex_id(ptr); if(id==packed[0]) continue;
      if(kept==capacity) fail("bridge capacity overflow");
      output[kept++]=id;
    }
    return static_cast<std::int64_t>(kept);
  } catch(const std::exception& e) { p1::error=e.what(); return -1; }
}
