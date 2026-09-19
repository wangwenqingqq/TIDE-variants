import sys, faulthandler
from pathlib import Path
faulthandler.enable(all_threads=True)
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from run_backend import Native, validate_inputs
m,q,o=validate_inputs(root/'data/fixture')
print('validated CPU oracle',flush=True)
b=Native('gpusim',root,root/'data/fixture')
print('created GPU handle',flush=True)
for qi in range(len(q)):
 for p,d in [(7,10),(4,5)]:
  print('before search',qi,p,d,flush=True)
  r,_,_=b.run(q,[qi],p,d)
  print('after search',qi,len(r[0]),flush=True)
  import numpy as np
  assert np.array_equal(np.sort(r[0]),o[qi,p,d])
print('before close',flush=True);b.close();print('after close',flush=True)
