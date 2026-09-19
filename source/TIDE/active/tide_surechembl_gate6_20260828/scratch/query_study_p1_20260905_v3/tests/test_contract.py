import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
from make_oracle import load_oracle,python_oracle
from run_campaign import plan


class ContractTests(unittest.TestCase):
    def test_campaign_has_six_balanced_pairs_per_comparison(self):
        rows=plan()
        self.assertEqual(len(rows),48)
        for peer in ('nv_row','nv_column','gpusim','tide_unbounded'):
            selected=[r for r in rows if r['comparison']==peer]
            self.assertEqual(len(selected),12)
            self.assertEqual(sum(r['slot']==0 and r['backend']==peer for r in selected),3)
            self.assertEqual({r['pair'] for r in selected},set(range(1,7)))
    def test_complete_oracle_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'oracle.bin'
            p.write_bytes(b'P1ORCL01'+struct.pack('<QQQQQ',1,2,3,9,0))
            result=load_oracle(p)
            np.testing.assert_array_equal(result[0,7,10],[3,9])
            self.assertEqual(len(result[0,4,5]),0)
    def test_oracle_trailing_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'oracle.bin'; p.write_bytes(b'P1ORCL01'+struct.pack('<Q',0)+b'bad')
            with self.assertRaises(ValueError):load_oracle(p)
    def test_threshold_boundary_self_and_duplicate_fingerprint(self):
        ids=np.array([1,2,3],dtype=np.uint64)
        fp=np.array([[127,0,0,0],[1023,0,0,0],[127,0,0,0]],dtype=np.uint64)
        query=np.array([1,127,0,0,0,7],dtype=np.uint64)
        np.testing.assert_array_equal(python_oracle(ids,fp,query,7,10),[2,3])
        np.testing.assert_array_equal(python_oracle(ids,fp,query,4,5),[3])
    def test_guard_requires_admission_before_child_or_nvidia(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'must_not_exist'
            run=subprocess.run([sys.executable,str(SRC/'guarded_run.py'),
                                '--record',str(Path(td)/'record.json'),'--',
                                sys.executable,'-c',f'open({str(p)!r},"w").write("bad")'],
                                capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('explicit user-approved GPU 3 admission required',run.stderr)
            self.assertFalse(p.exists())


if __name__=='__main__':unittest.main()
