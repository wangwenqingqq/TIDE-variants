// Independent CPU full-scan verifier for a fixed, result-independent 3x3
// cross-stage sample of the Safe-C2 v3 exact oracle. No GTS/GPU/gamma/recall.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
namespace {
constexpr int D=128,N=1000000,Q=10000,K=101;
constexpr std::uint64_t MAGIC=0x31564f3354435347ULL;
[[noreturn]] void die(const std::string&s){throw std::runtime_error("VERIFY-EXACT-GT-V3: "+s);} 
struct C{int32_t id;int64_t d2;float fp;};
bool eb(const C&a,const C&b){return a.d2!=b.d2?a.d2<b.d2:a.id<b.id;} bool fb(const C&a,const C&b){return a.fp!=b.fp?a.fp<b.fp:a.id<b.id;}
std::vector<float> readf(const fs::path&p,int rows){if(!fs::is_regular_file(p)||fs::file_size(p)!=uintmax_t(rows)*(4+D*4))die("bad fvec file "+p.string());std::ifstream f(p,std::ios::binary);std::vector<float>x(size_t(rows)*D);for(int r=0;r<rows;r++){int32_t d;f.read((char*)&d,4);if(!f||d!=D)die("bad fvec dim");f.read((char*)(x.data()+size_t(r)*D),D*4);if(!f)die("truncated fvec");for(int j=0;j<D;j++){float v=x[size_t(r)*D+j];if(!std::isfinite(v)||std::floor(v)!=v)die("nonintegral fvec");}}return x;}
C dist(const float*a,const float*b,int id){int64_t d2=0;volatile float s=0;for(int i=0;i<D;i++){int z=int(a[i])-int(b[i]);d2+=int64_t(z)*z;float t=a[i]-b[i];s=s+t*t;}return C{int32_t(id),d2,std::sqrt(static_cast<float>(s))};}
uint32_t bits(float x){uint32_t b;std::memcpy(&b,&x,4);return b;} int32_t ri32(std::ifstream&f){int32_t x;f.read((char*)&x,4);if(!f)die("audit EOF i32");return x;} int64_t ri64(std::ifstream&f){int64_t x;f.read((char*)&x,8);if(!f)die("audit EOF i64");return x;} uint64_t ru64(std::ifstream&f){uint64_t x;f.read((char*)&x,8);if(!f)die("audit EOF u64");return x;} float rf(std::ifstream&f){uint32_t b;f.read((char*)&b,4);if(!f)die("audit EOF f32");float x;std::memcpy(&x,&b,4);return x;}
struct A{std::array<C,K> e,f;};
A auditrow(std::ifstream&f,int q){constexpr std::streamoff header=24, stride=4+K*32;f.seekg(header+std::streamoff(q)*stride);if(!f)die("seek audit");if(ri32(f)!=q)die("audit q mismatch");A a;for(int i=0;i<K;i++){a.e[i]={ri32(f),ri64(f),rf(f)};a.f[i]={ri32(f),ri64(f),rf(f)};}return a;}
}
int main(int argc,char**argv){try{if(argc!=5)die("usage: VERIFY BASE QUERY AUDIT OUT_JSON");fs::path base=argv[1],query=argv[2],audit=argv[3],out=argv[4];if(fs::exists(out))die("output exists");auto b=readf(base,N),q=readf(query,Q);std::ifstream af(audit,std::ios::binary);if(!af)die("cannot open audit");if(ru64(af)!=MAGIC||ri32(af)!=D||ri32(af)!=N||ri32(af)!=Q||ri32(af)!=K)die("audit header");const std::array<int,9> sample={0,1000,1999,2000,3000,3999,4000,7000,9999};std::ofstream o(out);if(!o)die("output");o<<"{\n  \"schema\": \"gts-v3-independent-exact-oracle-check-v1\",\n  \"sample_local_ids\": [";for(size_t s=0;s<sample.size();s++){if(s)o<<',';o<<sample[s];}o<<"],\n  \"checks\": [\n";for(size_t si=0;si<sample.size();si++){int qi=sample[si];std::vector<C> all;all.reserve(N);const float*qq=q.data()+size_t(qi)*D;for(int id=0;id<N;id++)all.push_back(dist(b.data()+size_t(id)*D,qq,id));auto fp=all;std::partial_sort(all.begin(),all.begin()+K,all.end(),eb);std::partial_sort(fp.begin(),fp.begin()+K,fp.end(),fb);auto ar=auditrow(af,qi);for(int r=0;r<K;r++){if(all[r].id!=ar.e[r].id||all[r].d2!=ar.e[r].d2||bits(all[r].fp)!=bits(ar.e[r].fp))die("exact mismatch q="+std::to_string(qi)+" rank="+std::to_string(r));if(fp[r].id!=ar.f[r].id||fp[r].d2!=ar.f[r].d2||bits(fp[r].fp)!=bits(ar.f[r].fp))die("fp mismatch q="+std::to_string(qi)+" rank="+std::to_string(r));}o<<"    {\"local_id\":"<<qi<<",\"top101_exact_and_fp32_match\":true}"<<(si+1<sample.size()?",":"")<<"\n";}o<<"  ],\n  \"status\": \"PASS\"\n}\n";std::cout<<"INDEPENDENT_ORACLE_CHECK_PASS samples="<<sample.size()<<"\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}
