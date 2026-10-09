import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT
from evaluate_transitions import preflight, render_summary
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
                legacy=preflight(p,'bpr',c)
                self.assertTrue(legacy['poisson']['frozen_identity_verified'])
                self.assertEqual(preflight(p,'capr',c),legacy)
            with self.assertRaises(FileNotFoundError):preflight(p,'cpgr',c)

    def test_summary_pools_active_counts_not_full_validation_or_head_means(self):
        def head(n,raw,new):
            return dict(activated_samples=n,validation_samples=100,
                        raw_l2={'mean':raw},cpgr_l2={'mean':new},l2_win_rate=1,
                        exact_parity={'mean':0},raw_parity={'mean':.1},cpgr_parity={'mean':0},
                        all_validation_reference_only={'raw_l2':{'mean':99},'cpgr_l2':{'mean':88}})
        report={'cpgr':{'Gradient1':head(1,.1,.08),'Gradient2':head(3,.3,.2)}}
        text=render_summary(report,'test precision')
        self.assertIn('| Overall | 4/200 | 0.25 | 0.17 | -32.0000 | 100.0000 |',text)
        self.assertNotIn('| 99 |',text)
        self.assertIn('| Gradient1 | 0 | 0.1 | 0 |',text)

    def test_capr_summary_reports_tail_metrics_and_single_weight_metric(self):
        report={'bpr':{'raw_kl':{'mean':2,'q95':4,'q99':8},
                       'bpr_kl':{'mean':1,'q95':3,'q99':6},
                       'raw_action_error':{'mean':.1},'bpr_action_error':{'mean':.08},
                       'bpr_vs_p0_tv':{'mean':.02}},
                'weight':{'active':{'raw_error':{'mean':.5},'cpgr_error':{'mean':.4},'samples':4}}}
        text=render_summary(report,'test precision')
        self.assertIn('| Q95 KL | 4 | 3 | -25.0000 |',text)
        self.assertIn('| Q99 KL | 8 | 6 | -25.0000 |',text)
        self.assertIn('| Effective six-face mass NL1 | 0.5 | 0.4 | -20.0000 | 4 |',text)


if __name__=='__main__':unittest.main()
