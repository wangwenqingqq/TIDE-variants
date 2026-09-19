import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare import quotas,hash_ids,records,pc


class PrepareTests(unittest.TestCase):
    def test_quota_exact(self):
        self.assertEqual(quotas([3,5,2],7).tolist(),[2,4,1])
    def test_zero_strata(self):
        self.assertEqual(quotas([0,10,0],4).tolist(),[0,4,0])
    def test_hash_independent_of_position(self):
        a=np.array([9,4,13,55],dtype=np.uint64)
        np.testing.assert_array_equal(hash_ids(a)[::-1],hash_ids(a[::-1]))
    def test_record_popcounts(self):
        fp=np.array([[0,1,3,7],[15,0,0,0]],dtype=np.uint64)
        np.testing.assert_array_equal(pc(fp),[6,4])
        q=records(np.array([91,92],dtype=np.uint64),fp)
        np.testing.assert_array_equal(q[:,5],[6,4])
    def test_invalid_quota(self):
        with self.assertRaises(ValueError): quotas([1,2],4)


if __name__=='__main__': unittest.main()
