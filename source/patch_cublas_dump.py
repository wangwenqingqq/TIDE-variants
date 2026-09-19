from pathlib import Path
p=Path('/workspace/RT-TIDE/experiments/large_batch_20260825/cublas_exact_scan_probe.cu')
s=p.read_text()
if 'CUBLAS_VALIDATE_OUT' not in s:
    s=s.replace('#include <cstdint>\n','#include <cstdint>\n#include <cstdlib>\n',1)
    old='''std::vector<uint32_t>hc(nq);ck(cudaMemcpy(hc.data(),cnt,nq*4,cudaMemcpyDeviceToHost),"cc");uint64_t sum='''
    new='''std::vector<uint32_t>hc(nq);ck(cudaMemcpy(hc.data(),cnt,nq*4,cudaMemcpyDeviceToHost),"cc");if(const char*op=std::getenv("CUBLAS_VALIDATE_OUT")){std::vector<float>ho(total);ck(cudaMemcpy(ho.data(),c,total*4,cudaMemcpyDeviceToHost),"copy validation output");std::ofstream of(op,std::ios::binary);of.write(reinterpret_cast<const char*>(ho.data()),total*4);if(!of)throw std::runtime_error("write validation output");}uint64_t sum='''
    assert old in s
    s=s.replace(old,new,1)
    p.write_text(s)
print(p)
