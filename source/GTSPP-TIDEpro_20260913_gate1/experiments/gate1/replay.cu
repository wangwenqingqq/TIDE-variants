#include "runtime.cuh"
#include <omp.h>

namespace g1 {
using History=std::vector<std::shared_ptr<DB>>;
History load(const std::string &root) {
  History h;
  for(int i=0;i<7;++i) { MappedFile f(root+"/run"+std::to_string(i)+".bin");
    h.push_back(std::make_shared<DB>(load_interleaved<4>(f))); require(!h.back()->ids.empty(),"empty cohort"); }
  return h;
}
std::vector<HostHit> independent(const History &h,int epoch,const Q &q,int t) {
  std::vector<HostHit> out;
  // Independent unbounded AND/OR scan: no stored pc, intervals, run descriptors or GPU arithmetic.
  for(int e=0;e<=epoch;++e) for(std::size_t i=0;i<h[e]->ids.size();++i) {
    unsigned in=0,un=0;
    for(int w=0;w<4;++w) { in+=__builtin_popcountll(q.fp[w]&h[e]->fp[i*4+w]);
      un+=__builtin_popcountll(q.fp[w]|h[e]->fp[i*4+w]); }
    if(in*100>=unsigned(t)*un) out.push_back({h[e]->ids[i],std::uint16_t(in),std::uint16_t(un)});
  }
  sort_hits(out); return out;
}
std::string opath(const std::string &root,int e,int t,int q) {
  return root+"/oracle/e"+std::to_string(e)+"_t"+std::to_string(t)+"_q"+std::to_string(q)+".bin";
}
void oracle(const History &h,const QueryStore<4> &qs,const std::string &root) {
  require(std::filesystem::create_directory(root+"/oracle"),"refusing existing oracle directory");
  double begin=now(); unsigned long long records=0;
  #pragma omp parallel for num_threads(4) reduction(+:records) schedule(dynamic)
  for(int index=0;index<896;++index) {
    int e=index/128,t=((index/64)%2)?80:70,q=index%64;
    auto result=independent(h,e,qs[q],t); records+=result.size();
    std::ofstream out(opath(root,e,t,q),std::ios::binary);
    for(auto r:result) {DeviceHit d{r.id,r.intersection,r.union_count,0}; out.write(reinterpret_cast<char*>(&d),sizeof(d));}
    if(!out) std::terminate();
  }
  std::cout<<"oracle_vectors=896 records="<<records<<" wall_ms="<<now()-begin<<std::endl;
}
struct Oracles {
  std::map<std::tuple<int,int,int>,std::vector<HostHit>> data;
  explicit Oracles(const std::string &root) {
    for(int e=0;e<7;++e) for(int t:{70,80}) for(int q=0;q<64;++q) {
      MappedFile f(opath(root,e,t,q)); auto &v=data[{e,t,q}];
      auto p=f.as<DeviceHit>(); for(std::size_t j=0;j<f.bytes()/sizeof(DeviceHit);++j)
        v.push_back({p[j].id,p[j].intersection,p[j].union_count});
    }
  }
  void check(const Result &r) const {
    require(!r.overflow,"full-output capacity exceeded; batch invalid");
    for(int j=0;j<r.batch;++j) {
      const auto &expected=data.at({r.epoch,r.threshold,(r.first+j)%64}); const auto &actual=r.hits[j];
      require(actual.size()==expected.size(),"complete oracle count mismatch");
      for(std::size_t i=0;i<actual.size();++i) require(actual[i].id==expected[i].id &&
        actual[i].intersection==expected[i].intersection && actual[i].union_count==expected[i].union_count,
        "complete oracle payload mismatch");
    }
  }
};
std::shared_ptr<DB> merge(const std::vector<std::shared_ptr<Run>> &runs,std::size_t first) {
  auto out=std::make_shared<DB>(); std::size_t n=0;
  for(std::size_t i=first;i<runs.size();++i) n+=runs[i]->host->ids.size();
  out->ids.resize(n); out->pc.resize(n); out->fp.resize(4*n);
  std::size_t pos=0;
  // Cached bucket boundaries give a stable linear contiguous-copy ordinary merge.
  for(int pc=0;pc<=256;++pc) for(std::size_t i=first;i<runs.size();++i) {
    const auto &r=*runs[i]; auto b=r.metadata.cumulative[pc],len=r.metadata.cumulative[pc+1]-b;
    std::copy_n(r.host->ids.data()+b,len,out->ids.data()+pos);
    std::copy_n(r.host->pc.data()+b,len,out->pc.data()+pos);
    std::copy_n(r.host->fp.data()+4*b,4*len,out->fp.data()+4*pos); pos+=len;
  }
  require(pos==n,"merge conservation error"); return out;
}
struct Maintenance {
  std::string policy,mode,status="published"; int epoch=0,batch=0,rotation=0,merges=0,deferred=0;
  double arrival=0,begin=0,ready=0,published=0,writer_done=0,build_ms=0,metadata_ms=0,stage_ms=0,h2d_ms=0;
  std::size_t uploaded=0,merge_rw=0,rows=0,runs=0,shared=0,current_only=0,retired_only=0;
  std::size_t reader_only_at_publish=0,unreferenced_at_publish=0,live_at_publish=0,other_held_at_publish=0;
  std::shared_ptr<EpochInfo> retired;
  std::vector<std::shared_ptr<Release>> exclusive_releases;
};
struct Writer {
  Arena &arena; cudaStream_t stream=nullptr;void *stage=nullptr;
  explicit Writer(Arena &a):arena(a) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    try {CUDA_CHECK(cudaMallocHost(&stage,8*MiB));}catch(...) {cudaStreamDestroy(stream);throw;}
  }
  ~Writer() {if(stream) cudaStreamSynchronize(stream);if(stage) cudaFreeHost(stage);if(stream) cudaStreamDestroy(stream);}
  EP build(const EP &old,std::shared_ptr<DB> delta,const std::string &policy,int epoch,Maintenance &m,int fault=0) {
    m.begin=now(); if(!m.arrival) m.arrival=m.begin; m.epoch=epoch; m.policy=policy;
    auto next=std::make_shared<Epoch>(); next->number=epoch; if(old) next->runs=old->runs;
    auto upload=[&](std::shared_ptr<DB> h,int injected) {
      auto r=std::make_shared<Run>(arena,h,stream,stage,injected);
      m.uploaded+=h->ids.size()*42; m.metadata_ms+=r->metadata_ms; m.stage_ms+=r->stage_ms; m.h2d_ms+=r->h2d_ms;
      return r;
    };
    if(!arena.fits(Run::footprint(delta->ids.size()))) {m.status="rejected_delta_budget";m.ready=now();return old;}
    next->runs.push_back(upload(delta,fault));
    auto compact=[&](std::size_t first) {
      std::size_t n=0; for(std::size_t i=first;i<next->runs.size();++i) n+=next->runs[i]->host->ids.size();
      if(!arena.fits(Run::footprint(n))) {m.deferred++;m.status="published_merge_deferred";return false;}
      double t=now(); auto host=merge(next->runs,first); m.build_ms+=now()-t;
      auto combined=upload(host,0); m.merges++; m.merge_rw+=n*84;
      next->runs.erase(next->runs.begin()+first,next->runs.end()); next->runs.push_back(combined); return true;
    };
    if(epoch>0 && (policy=="compact" || (policy=="periodic2" && epoch%2==0))) compact(0);
    else if(policy=="size_tiered") while(next->runs.size()>=2) {
      auto n=next->runs.size(); if(next->runs[n-2]->host->ids.size()>2*next->runs[n-1]->host->ids.size()) break;
      if(!compact(n-2)) break;
    }
    m.ready=now(); return next;
  }
  void publish(EP *slot,EP next,Maintenance &m) {
    auto old=std::atomic_load(slot); m.rows=next->rows();m.runs=next->runs.size();
    std::set<std::uint64_t> after;
    for(auto &r:next->runs) after.insert(r->uid);
    if(old && old!=next) {m.retired=old->info;
      for(auto &r:old->runs) if(!after.count(r->uid)) {m.retired_only+=r->bytes;m.exclusive_releases.push_back(r->released);}
    }
    {
      std::lock_guard<std::mutex> lock(publication_mutex);
      std::atomic_store(slot,next); m.published=now();
      auto reader=registered_reader.lock();std::set<std::uint64_t> read_ids;
      if(reader) for(auto &r:reader->runs) {
        read_ids.insert(r->uid);if(!after.count(r->uid)) m.reader_only_at_publish+=r->bytes;
      }
      for(auto &r:next->runs) {if(read_ids.count(r->uid)) m.shared+=r->bytes;else m.current_only+=r->bytes;}
      if(old && old!=next) for(auto &r:old->runs)
        if(!after.count(r->uid)&&!read_ids.count(r->uid)) m.unreferenced_at_publish+=r->bytes;
      m.live_at_publish=arena.live();
      auto classified=Runtime::bytes()+m.shared+m.current_only+m.reader_only_at_publish;
      require(m.live_at_publish>=classified,"unique-owner byte balance underflow");
      // Includes writer temporaries and, in shadow, the deliberately retained fixed base.
      m.other_held_at_publish=m.live_at_publish-classified;
    }
    old.reset();m.writer_done=now();arena.observe();
  }
};
std::vector<std::string> policies(int rotation) {
  std::vector<std::string> p{"all_delta","periodic2","size_tiered","compact"};
  std::rotate(p.begin(),p.begin()+rotation,p.end());return p;
}
struct Logs {
  std::ofstream requests,maint;
  Logs(const std::string &prefix):requests(prefix+".requests.csv"),maint(prefix+".maintenance.csv") {
    require(bool(requests)&&bool(maint),"cannot open logs");requests<<std::setprecision(12);maint<<std::setprecision(12);
    requests<<"rotation,mode,policy,phase,batch,epoch,threshold,first_query,rows,runs,begin_ms,end_ms,service_ms,kernel_ms,"
      "device_complete_ms,candidate_rows,scan_fp_pc_bytes,tail_threads,descriptors,descriptor_bytes,hits,output_bytes,result_hash,cpu_complete_match,query_queue_ms,maintenance_backlog\n";
    maint<<"rotation,mode,policy,batch,epoch,status,rows,runs,merges,deferred,begin_ms,ready_ms,published_ms,writer_done_ms,"
      "host_build_ms,metadata_ms,staging_ms,h2d_ms,uploaded_bytes,merge_host_rw_bytes,shared_bytes,current_only_bytes,"
      "retired_only_bytes,reader_only_at_publish_bytes,unreferenced_at_publish_bytes,live_at_publish_bytes,"
      "last_reader_device_done_ms,old_epoch_owner_released_ms,exclusive_actual_reclaim_ms,lifecycle_ms,arrival_ms,maintenance_queue_ms,other_held_at_publish_bytes\n";
  }
  void request(const Result &r,const std::string &p,const std::string &mode,int rotation,const std::string &phase,
      double queue=0,int backlog=0) {
    std::uint64_t hash=1469598103934665603ull,hits=0;
    for(auto &v:r.hits) {hash^=hash_hits(v);hash*=1099511628211ull;hits+=v.size();}
    requests<<rotation<<','<<mode<<','<<p<<','<<phase<<','<<r.batch<<','<<r.epoch<<','<<r.threshold<<','<<r.first<<','
      <<r.rows<<','<<r.runs<<','<<r.begin<<','<<r.end<<','<<r.end-r.begin<<','<<r.kernel_ms<<','<<r.device_complete<<','
      <<r.candidates<<','<<r.candidates*34<<','<<r.blocks*256-r.candidates<<','<<r.descriptors<<','<<r.descriptors*sizeof(Slice)<<','
      <<hits<<','<<r.output_bytes<<','<<hash<<",1,"<<queue<<','<<backlog<<'\n';
  }
  void maintenance(const Maintenance &m) {
    double last=m.retired?m.retired->last_reader_done.load():0;
    double owner=m.retired?m.retired->owner_released.load():m.writer_done;
    double reclaimed=m.writer_done;
    for(auto &r:m.exclusive_releases) {require(r->at.load()>0,"retired exclusive run never reclaimed");reclaimed=std::max(reclaimed,r->at.load());}
    if(m.retired) require(owner>0,"old epoch still owned at finalization");
    maint<<m.rotation<<','<<m.mode<<','<<m.policy<<','<<m.batch<<','<<m.epoch<<','<<m.status<<','<<m.rows<<','<<m.runs<<','
      <<m.merges<<','<<m.deferred<<','<<m.begin<<','<<m.ready<<','<<m.published<<','<<m.writer_done<<','<<m.build_ms<<','
      <<m.metadata_ms<<','<<m.stage_ms<<','<<m.h2d_ms<<','<<m.uploaded<<','<<m.merge_rw<<','<<m.shared<<','<<m.current_only<<','
      <<m.retired_only<<','<<m.reader_only_at_publish<<','<<m.unreferenced_at_publish<<','<<m.live_at_publish<<','<<last<<','
      <<owner<<','<<reclaimed<<','<<std::max({reclaimed,owner,m.writer_done})-m.begin<<','<<m.arrival<<','<<m.begin-m.arrival<<','<<m.other_held_at_publish<<'\n';
  }
};
void guards(Arena &a,Runtime &rt,Writer &writer,const History &h,const QueryStore<4> &qs,const Oracles &o) {
  require(Run::footprint(9461367)==Run::footprint(10279473),"large-run size class not stable");
  bool oversized=false;try {(void)Run::footprint(10500001);}catch(const std::runtime_error&){oversized=true;}
  require(oversized,"run capacity guard failed");
  auto baseline=a.live(); EP current; Maintenance initial;
  writer.publish(&current,writer.build(nullptr,h[0],"all_delta",0,initial),initial);
  auto steady=a.live();auto identity=current.get();
  for(int fault=1;fault<=4;++fault) {
    bool thrown=false;try {Maintenance m;auto next=writer.build(current,h[1],"all_delta",1,m,fault);(void)next;}
    catch(const std::runtime_error &e) {thrown=std::string(e.what()).find("injected")!=std::string::npos;}
    require(thrown && a.live()==steady && current.get()==identity,"failure rollback did not restore current/storage");
    o.check(rt.run(&current,qs,0,64,70));
  }
  {Maintenance m; auto next=writer.build(current,h[1],"compact",1,m);next.reset();
    require(a.live()==steady&&current.get()==identity,"before-publish rollback failed");}
  { bool refused=false;try {Buffer impossible(a,a.capacity+256);}catch(const std::bad_alloc&){refused=true;}
    require(refused&&a.live()==steady,"arena budget guard failed"); }
  {Buffer pressure(a,a.capacity-a.live()-256); Maintenance m;
    auto next=writer.build(current,h[1],"all_delta",1,m);
    require(next==current&&m.status=="rejected_delta_budget","delta rejection changed current");}
  require(a.live()==steady,"pressure guard leaked");
  // A low explicit append capacity forces a real GPU overflow, including native Q=64 routing.
  rt.begin(&current,qs,0,64,70,nullptr,1); auto overflow=rt.finish();
  require(overflow.overflow,"output overflow was not signaled");
  for(auto &v:overflow.hits) require(v.empty(),"partial result escaped on overflow");
  // Prepare ALL new GPU work before the host gate. Gate release depends only on CPU atomic publication.
  Maintenance m; auto next=writer.build(current,h[1],"compact",1,m);
  std::weak_ptr<Epoch> old=current; auto old_release=current->runs[0]->released;
  Gate gate;
  try {
    rt.begin(&current,qs,0,64,70,&gate);
    require(cudaEventQuery(rt.stop)==cudaErrorNotReady,"correctness gate failed to leave GPU work pending");
    // No CUDA API in the publication path while gated (including memory observation).
    std::atomic_store(&current,next);next.reset();
    require(!old.expired()&&old_release->at.load()==0,"old owner reclaimed before pending kernel/D2H");
    gate.open();auto result=rt.finish();o.check(result);
    require(result.epoch==0&&old.expired()&&old_release->at.load()>0,"old GPU ownership completion failed");
  } catch(...) {gate.open();rt.cancel();throw;}
  o.check(rt.run(&current,qs,0,64,70));
  // Exception-style cancellation synchronizes pending GPU work before releasing the acquired owner.
  Gate cancel_gate;
  try {rt.begin(&current,qs,0,64,80,&cancel_gate);std::weak_ptr<Epoch> cancelled=current;
    current.reset();require(!cancelled.expired(),"request did not retain epoch");rt.cancel();
    require(cancelled.expired()&&a.live()==baseline,"in-flight cancellation leaked/reclaimed early");
  } catch(...) {cancel_gate.open();rt.cancel();throw;}
  std::cout<<"GUARDS_PASS injected=5 budget_refusal=2 overflow=1 pending_GPU_publish=1 cancellation=1 arena_restored=1 large_class=1\n";
}
void sequential(Arena &a,Runtime &rt,Writer &w,const History &h,const QueryStore<4> &qs,const Oracles &o,
    Logs &logs,int rotation) {
  auto baseline=a.live(); std::size_t checked=0;
  for(int batch:{1,8,64}) for(const auto &policy:policies(rotation)) {
    EP current;std::vector<Maintenance> ledger;
    for(int e=0;e<7;++e) {
      Maintenance m;m.mode="sequential";m.rotation=rotation;m.batch=batch;
      w.publish(&current,w.build(std::atomic_load(&current),h[e],policy,e,m),m);ledger.push_back(m);
      require(current->number==e,"registered dataset cannot fit even append fallback");
      for(const std::string phase:{"warmup","measured"}) for(int t:{70,80}) for(int first=0;first<64;first+=batch) {
        auto r=rt.run(&current,qs,first,batch,t);o.check(r);logs.request(r,policy,"sequential",rotation,phase);checked+=batch;
      }
      a.observe();
    }
    current.reset();for(auto &m:ledger) logs.maintenance(m);
    require(a.live()==baseline,"sequential full-release arena balance failed");
  }
  std::cout<<"SEQUENTIAL_PASS checked_queries="<<checked<<" batches_Q=1/8/64\n";
}
void concurrent(Arena &a,Runtime &rt,Writer &w,const History &h,const QueryStore<4> &qs,const Oracles &o,
    Logs &logs,int rotation) {
  auto baseline=a.live();
  for(int batch:{1,8,64}) for(const auto &policy:policies(rotation))
    for(const std::string mode:{"static","shadow","growth"}) {
      EP visible;Maintenance initial;w.publish(&visible,w.build(nullptr,h[0],policy,0,initial),initial);
      EP writer_current=visible;std::vector<Maintenance> ledger;
      for(int j=0;j<64;j+=batch) {auto r=rt.run(&visible,qs,j,batch,70);o.check(r);logs.request(r,policy,mode,rotation,"warmup");}
      std::mutex mutex;std::condition_variable cv;int released=0;bool stop=false;std::exception_ptr failure;
      std::array<double,7> arrivals{};std::atomic<int> completed{0};
      std::thread thread;
      if(mode!="static") thread=std::thread([&] {
        try {
          CUDA_CHECK(cudaSetDevice(0));
          for(int e=1;e<7;++e) {
            Maintenance m;m.mode=mode;m.rotation=rotation;m.batch=batch;
            {std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return released>=e||stop;});if(stop) return;m.arrival=arrivals[e];}
            auto next=w.build(writer_current,h[e],policy,e,m);
            require(next->number==e&&m.deferred==0,"roomy concurrent run unexpectedly budget constrained");
            if(mode=="growth") w.publish(&visible,next,m);
            else w.publish(&writer_current,next,m); // exact same maintenance plan, fixed initial visible epoch
            writer_current=next;next.reset();ledger.push_back(m);completed.store(e);
          }
        } catch(...) {std::lock_guard<std::mutex> lock(mutex);failure=std::current_exception();stop=true;cv.notify_all();}
      });
      try {
        // Closed-loop single-reader fixed 256 batches; arrival triggers at 16,48,...,176 completions.
        // Six jobs may queue naturally. No sleeping / artificial upload prolongation.
        for(int i=0;i<256;++i) {
          if(i>=16 && (i-16)%32==0 && i<=176) {
            std::lock_guard<std::mutex> lock(mutex);released++;arrivals[released]=now();cv.notify_all();
          }
          {std::lock_guard<std::mutex> lock(mutex);if(failure) std::rethrow_exception(failure);}
          auto r=rt.run(&visible,qs,(i*batch)%64,batch,(i%2)?80:70);o.check(r);
          logs.request(r,policy,mode,rotation,"measured",0,mode=="static"?0:released-completed.load());
        }
      } catch(...) {
        {std::lock_guard<std::mutex> lock(mutex);stop=true;cv.notify_all();}
        if(thread.joinable()) thread.join();throw;
      }
      if(thread.joinable()) thread.join();if(failure) std::rethrow_exception(failure);
      if(mode=="growth") for(int t:{70,80}) {auto r=rt.run(&visible,qs,0,64,t);o.check(r);logs.request(r,policy,mode,rotation,"final_check");}
      visible.reset();writer_current.reset();for(auto &m:ledger) logs.maintenance(m);
      require(a.live()==baseline,"concurrent full-release arena balance failed");
    }
  std::cout<<"CONCURRENT_PASS closed_loop_batches_per_case=256 modes=static/shadow/growth\n";
}
} // namespace g1

int main(int argc,char **argv) {
  try {
    std::map<std::string,std::string> args;
    g1::require(argc%2==1,"expected key/value arguments");
    for(int i=1;i<argc;i+=2) {std::string k=argv[i];g1::require(k=="--root"||k=="--mode"||k=="--output"||k=="--cap-mib"||k=="--rotation","unknown argument");
      g1::require(args.emplace(k,argv[i+1]).second,"duplicate argument");}
    const auto root=args.at("--root"),mode=args.at("--mode");
    auto h=g1::load(root);MappedFile f(root+"/queries.bin");QueryStore<4> qs(f);g1::require(qs.size()==64,"expected 64 queries");
    if(mode=="oracle") {g1::oracle(h,qs,root);return 0;}
    g1::require(mode=="guards"||mode=="sequential"||mode=="concurrent","bad mode");
    auto prefix=args.at("--output");g1::require(!std::filesystem::exists(prefix+".arena.csv"),"refusing existing evidence");
    int cap=std::stoi(args.at("--cap-mib")),rotation=args.count("--rotation")?std::stoi(args.at("--rotation")):0;
    g1::require((cap==1280||cap==2048)&&rotation>=0&&rotation<4,"registered cap/rotation required");
    g1::Oracles oracles(root);g1::Arena arena(cap*g1::MiB,prefix+".arena.csv");
    {g1::Runtime rt(arena);g1::Writer writer(arena);g1::Logs logs(prefix);
      if(mode=="guards") g1::guards(arena,rt,writer,h,qs,oracles);
      else if(mode=="sequential") g1::sequential(arena,rt,writer,h,qs,oracles,logs,rotation);
      else g1::concurrent(arena,rt,writer,h,qs,oracles,logs,rotation);
    }
    g1::require(arena.live()==0,"arena must have no live allocations at shutdown");arena.observe();
    struct rusage rss{};getrusage(RUSAGE_SELF,&rss);
    std::ofstream summary(prefix+".memory.json");summary<<"{\"total_cap_bytes\":"<<arena.cap<<",\"baseline_observed_bytes\":"<<arena.baseline
      <<",\"fixed_arena_bytes\":"<<arena.capacity<<",\"peak_logical_arena_bytes\":"<<arena.peak<<",\"observed_whole_card_peak_bytes\":"<<arena.observed_peak
      <<",\"budget_refusals\":"<<arena.refused<<",\"final_live_arena_bytes\":0,\"host_maxrss_kib\":"<<rss.ru_maxrss<<"}\n";
    std::cout<<"PASS observed_peak="<<arena.observed_peak<<" total_cap="<<arena.cap<<" arena="<<arena.capacity<<" logical_peak="<<arena.peak<<std::endl;
    return 0;
  } catch(const std::exception &e) {std::cerr<<"ERROR: "<<e.what()<<std::endl;return 2;}
}
