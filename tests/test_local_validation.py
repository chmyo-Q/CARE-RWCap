import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT
from evaluate_transitions import preflight
from input_checks import validate_cpgr_counts


class LocalInputTests(unittest.TestCase):
    def test_cpgr_zero_visits_valid_but_cuda_errors_and_count_mismatch_rejected(self):
        p={'cuda_status':0,'samples':0,'masks':[0,0,0,0]}
        j={'cuda_status':0,'input_xyz_masks':[0]*8}
        validate_cpgr_counts(p,j)
        with self.assertRaisesRegex(ValueError,'CUDA error'):
            validate_cpgr_counts(dict(p,cuda_status=1),j)
        with self.assertRaisesRegex(ValueError,'counts disagree'):
            validate_cpgr_counts(dict(p,samples=1),j)

    def test_original_validation_indices_are_unique(self):
        c=json.loads((ROOT/'configs/transition_validation.json').read_text())
        self.assertEqual(len(set(c['poisson_validation_indices'])),10000)
        self.assertEqual((c['gradient_start'],c['gradient_end']),(90000,100000))

    def test_preflight_cannot_accept_another_dataset_by_size_alone(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'poisson.bin').write_bytes(b'abc')
            c={'data_bytes':{'poisson':3},'poisson_validation_indices':list(range(10000))}
            with self.assertRaisesRegex(ValueError,'Not the frozen'):
                preflight(p,'bpr',c)
            with patch('evaluate_transitions.DATA_IDENTITIES',{'poisson':hashlib.sha256(b'abc').hexdigest()}):
                self.assertTrue(preflight(p,'bpr',c)['poisson']['frozen_identity_verified'])
            with self.assertRaises(FileNotFoundError):preflight(p,'cpgr',c)


if __name__=='__main__':unittest.main()
