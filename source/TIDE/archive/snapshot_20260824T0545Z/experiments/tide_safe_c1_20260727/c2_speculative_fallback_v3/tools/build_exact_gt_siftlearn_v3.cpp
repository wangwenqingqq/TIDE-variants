// CPU-only independent oracle for Safe-C2 v3 compact SIFT-learn queries.
// It must not be used to tune gamma, traverse GTS, or issue a GPU workload.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
namespace {
constexpr int kDim = 128;
constexpr int kBaseN = 1'000'000;
constexpr int kQueryN = 10'000;
constexpr int kTop = 101;
constexpr int kGtWidth = 100;
constexpr int kReservedPrefix = 3072;
constexpr std::uint64_t kMagic = 0x31564f3354435347ULL; // "GTS C3OV1" opaque audit magic

[[noreturn]] void die(const std::string& m) { throw std::runtime_error("EXACT-GT-V3: " + m); }

struct Candidate {
  std::int32_t id = -1;
  std::int64_t d2 = std::numeric_limits<std::int64_t>::max();
  float fp = std::numeric_limits<float>::infinity();
};

struct ExactWorse { // priority_queue top = worst by (d2,id)
  bool operator()(const Candidate& a, const Candidate& b) const {
    if (a.d2 != b.d2) return a.d2 < b.d2;
    return a.id < b.id;
  }
};
struct FpWorse { // priority_queue top = worst by (fp,id)
  bool operator()(const Candidate& a, const Candidate& b) const {
    if (a.fp != b.fp) return a.fp < b.fp;
    return a.id < b.id;
  }
};

bool exact_better(const Candidate& a, const Candidate& b) {
  return a.d2 != b.d2 ? a.d2 < b.d2 : a.id < b.id;
}
bool fp_better(const Candidate& a, const Candidate& b) {
  return a.fp != b.fp ? a.fp < b.fp : a.id < b.id;
}

struct QueryResult {
  std::array<Candidate, kTop> exact{};
  std::array<Candidate, kTop> fp{};
  bool exact_boundary_tie = false;
  bool fp_boundary_tie = false;
  bool top10_set_agree = false;
  bool fp_top100_collision_different_d2 = false;
  bool eligible = false;
};

struct Args { fs::path base, query, out; };
Args parse(int argc, char** argv) {
  Args a;
  for (int i=1;i<argc;++i) {
    const std::string x=argv[i];
    if (x=="--help") {
      std::cout << "Usage: " << argv[0] << " --base BASE.fvecs --query COMPACT10K.fvecs --out FRESH_DIR\n";
      std::exit(0);
    }
    if (i+1>=argc) die("missing value for " + x);
    const char* v=argv[++i];
    if (x=="--base") a.base=v;
    else if (x=="--query") a.query=v;
    else if (x=="--out") a.out=v;
    else die("unknown arg " + x);
  }
  if (a.base.empty()||a.query.empty()||a.out.empty()) die("--base --query --out required");
  if (!fs::is_regular_file(a.base)||!fs::is_regular_file(a.query)) die("inputs must be regular files");
  if (fs::exists(a.out)||fs::is_symlink(a.out)) die("refuse overwrite/append output");
  return a;
}

std::vector<float> read_fvecs(const fs::path& p, int expected_rows, const char* label) {
  const std::uintmax_t expected = static_cast<std::uintmax_t>(expected_rows) * (4 + 4*kDim);
  if (fs::file_size(p) != expected) die(std::string(label)+" unexpected size");
  std::ifstream in(p, std::ios::binary);
  if (!in) die(std::string("cannot open ")+label);
  std::vector<float> out(static_cast<std::size_t>(expected_rows)*kDim);
  for (int row=0; row<expected_rows; ++row) {
    std::int32_t d=0; in.read(reinterpret_cast<char*>(&d), sizeof(d));
    if (!in || d!=kDim) die(std::string(label)+" bad dimension at row "+std::to_string(row));
    float* dst=out.data()+static_cast<std::size_t>(row)*kDim;
    in.read(reinterpret_cast<char*>(dst), sizeof(float)*kDim);
    if (!in) die(std::string(label)+" truncated at row "+std::to_string(row));
    for (int j=0;j<kDim;++j) {
      const float v=dst[j];
      if (!std::isfinite(v) || std::floor(v)!=v || std::fabs(v)>1.0e6F)
        die(std::string(label)+" non-finite/non-integral/out-of-range coordinate at row "+std::to_string(row));
    }
  }
  char extra=0; if (in.read(&extra,1)) die(std::string(label)+" trailing bytes");
  return out;
}

Candidate distance_to(const float* base, const float* q, int id) {
  std::int64_t d2=0;
  volatile float sum=0.0F;
  for (int d=0; d<kDim; ++d) {
    const int ai=static_cast<int>(base[d]);
    const int qi=static_cast<int>(q[d]);
    const int delta_i=ai-qi;
    d2 += static_cast<std::int64_t>(delta_i)*static_cast<std::int64_t>(delta_i);
    const float delta=base[d]-q[d];
    const float term=delta*delta;
    sum = sum + term;
  }
  const float fp=std::sqrt(static_cast<float>(sum));
  if (!std::isfinite(fp)) die("non-finite fp32 distance");
  return Candidate{static_cast<std::int32_t>(id),d2,fp};
}

template <typename Heap, typename Better>
void consider(Heap& h, const Candidate& c, Better better) {
  if (static_cast<int>(h.size()) < kTop) { h.push(c); return; }
  if (better(c,h.top())) { h.pop(); h.push(c); }
}

template <typename Heap, typename Better>
std::array<Candidate,kTop> sorted_heap(Heap h, Better better) {
  if (static_cast<int>(h.size())!=kTop) die("top heap wrong size");
  std::array<Candidate,kTop> out{};
  for (int i=0;i<kTop;++i) { out[i]=h.top(); h.pop(); }
  std::sort(out.begin(),out.end(),better);
  return out;
}

void finalize(QueryResult& out) {
  out.exact_boundary_tie = out.exact[9].d2 == out.exact[10].d2;
  out.fp_boundary_tie = out.fp[9].fp == out.fp[10].fp;
  std::array<int,kGtWidth> a{}, b{};
  for(int i=0;i<kGtWidth;++i) { a[i]=out.exact[i].id; b[i]=out.fp[i].id; }
  std::sort(a.begin(),a.begin()+10); std::sort(b.begin(),b.begin()+10);
  out.top10_set_agree = std::equal(a.begin(),a.begin()+10,b.begin());
  for(int i=0;i<kGtWidth;) {
    int j=i+1; while(j<kGtWidth && out.fp[j].fp==out.fp[i].fp) ++j;
    if(j-i>1) {
      const auto d=out.fp[i].d2;
      for(int z=i+1;z<j;++z) if(out.fp[z].d2!=d) out.fp_top100_collision_different_d2=true;
    }
    i=j;
  }
  out.eligible = !out.exact_boundary_tie && !out.fp_boundary_tie && out.top10_set_agree && !out.fp_top100_collision_different_d2;
}

void write_i32(std::ofstream& f, std::int32_t x) { f.write(reinterpret_cast<const char*>(&x),sizeof(x)); if(!f) die("write failed"); }
void write_i64(std::ofstream& f, std::int64_t x) { f.write(reinterpret_cast<const char*>(&x),sizeof(x)); if(!f) die("write failed"); }
void write_u64(std::ofstream& f, std::uint64_t x) { f.write(reinterpret_cast<const char*>(&x),sizeof(x)); if(!f) die("write failed"); }
void write_f32_bits(std::ofstream& f, float x) { std::uint32_t b=0; std::memcpy(&b,&x,sizeof(b)); f.write(reinterpret_cast<const char*>(&b),sizeof(b)); if(!f) die("write failed"); }

void fsync_file(const fs::path& p) {
  // std::ofstream flush is sufficient for this experiment artifact; the outer wrapper hashes after close.
  (void)p;
}

} // namespace

int main(int argc,char** argv) {
  try {
    const Args args=parse(argc,argv);
    const auto base=read_fvecs(args.base,kBaseN,"base");
    const auto queries=read_fvecs(args.query,kQueryN,"compact query");
    fs::create_directory(args.out);
    std::vector<QueryResult> all(kQueryN);
    std::atomic<int> done{0};
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
    for (int qid=0;qid<kQueryN;++qid) {
      std::priority_queue<Candidate,std::vector<Candidate>,ExactWorse> exact_h;
      std::priority_queue<Candidate,std::vector<Candidate>,FpWorse> fp_h;
      const float* q=queries.data()+static_cast<std::size_t>(qid)*kDim;
      for (int id=0;id<kBaseN;++id) {
        const Candidate c=distance_to(base.data()+static_cast<std::size_t>(id)*kDim,q,id);
        consider(exact_h,c,exact_better);
        consider(fp_h,c,fp_better);
      }
      all[qid].exact=sorted_heap(exact_h,exact_better);
      all[qid].fp=sorted_heap(fp_h,fp_better);
      finalize(all[qid]);
      const int n=++done;
      if ((n%100)==0) {
#ifdef _OPENMP
#pragma omp critical
#endif
        std::cerr << "EXACT_GT_PROGRESS completed_queries=" << n << "/" << kQueryN << "\n";
      }
    }

    std::ofstream gt(args.out/"learn_clean10k_gt_fp32_top100.ivecs",std::ios::binary);
    std::ofstream audit(args.out/"top101_int64_fp32_audit.bin",std::ios::binary);
    std::ofstream admission(args.out/"tie_admission.tsv");
    if(!gt||!audit||!admission) die("cannot create oracle outputs");
    write_u64(audit,kMagic); write_i32(audit,kDim); write_i32(audit,kBaseN); write_i32(audit,kQueryN); write_i32(audit,kTop);
    admission << "local_id\texact_boundary_tie\tfp32_boundary_tie\ttop10_idset_agree\tfp32_top100_collision_different_d2\teligible\n";
    int eligible_cal=0,eligible_val=0,eligible_test=0;
    std::ofstream cal(args.out/"calibration.ids"), val(args.out/"validation.ids"), test(args.out/"sealed_test.ids");
    if(!cal||!val||!test) die("cannot create stage IDs");
    for(int q=0;q<kQueryN;++q) {
      write_i32(gt,kGtWidth);
      for(int r=0;r<kGtWidth;++r) write_i32(gt,all[q].fp[r].id);
      write_i32(audit,q);
      for(int r=0;r<kTop;++r) {
        write_i32(audit,all[q].exact[r].id); write_i64(audit,all[q].exact[r].d2); write_f32_bits(audit,all[q].exact[r].fp);
        write_i32(audit,all[q].fp[r].id); write_i64(audit,all[q].fp[r].d2); write_f32_bits(audit,all[q].fp[r].fp);
      }
      const auto& x=all[q];
      admission << q << '\t' << int(x.exact_boundary_tie) << '\t' << int(x.fp_boundary_tie) << '\t'
                << int(x.top10_set_agree) << '\t' << int(x.fp_top100_collision_different_d2) << '\t' << int(x.eligible) << '\n';
      if(x.eligible) {
        if(q<2000) { cal<<q<<'\n'; ++eligible_cal; }
        else if(q<4000) { val<<q<<'\n'; ++eligible_val; }
        else { test<<q<<'\n'; ++eligible_test; }
      }
    }
    gt.close(); audit.close(); admission.close(); cal.close(); val.close(); test.close();
    std::ofstream summary(args.out/"oracle_summary.json");
    if(!summary) die("cannot write summary");
    summary << "{\n"
            << "  \"schema\": \"gts-v3-cpu-exact-fp32-int64-oracle-v1\",\n"
            << "  \"dimension\": 128,\n  \"base_count\": 1000000,\n  \"query_count\": 10000,\n"
            << "  \"top_width\": 101,\n  \"gt_width\": 100,\n"
            << "  \"fp32_semantics\": \"volatile float ordered sum of 128 delta*delta terms, then std::sqrt(float)\",\n"
            << "  \"exact_semantics\": \"int64 sum of squared integral-coordinate differences\",\n"
            << "  \"tie_admission\": \"exact rank10!=rank11; fp32 rank10!=rank11; exact/fp32 top10 ID sets agree; fp32 top100 contains no equal-fp32/different-int64-d2 collision; no replacement\",\n"
            << "  \"eligible_calibration\": " << eligible_cal << ",\n"
            << "  \"eligible_validation\": " << eligible_val << ",\n"
            << "  \"eligible_sealed_test\": " << eligible_test << "\n}\n";
    summary.close();
    std::cout << "EXACT_GT_COMPLETE eligible_calibration="<<eligible_cal<<" eligible_validation="<<eligible_val<<" eligible_sealed_test="<<eligible_test<<"\n";
    return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << "\n"; return 2; }
}
