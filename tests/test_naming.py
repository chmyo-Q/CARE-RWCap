"""Paper-facing names must preserve the archived experiment and statistics."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark import make_plan
from naming import normalize_arm
from summarize import save_summary


class NamingTest(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads((ROOT/'configs/paper_protocol.json').read_text())

    def test_aliases_preserve_exact_paper_plan(self):
        old=make_plan(self.protocol,'paper',['p0','bpr','full'])
        new=make_plan(self.protocol,'paper',['deeprwcap','capr','care-rwcap'])
        self.assertEqual(old,new)
        self.assertEqual(normalize_arm('capr'),'bpr')
        self.assertEqual(normalize_arm('bpr'),'bpr')
        self.assertEqual(normalize_arm('deeprwcap'),'p0')
        self.assertEqual(normalize_arm('care-rwcap'),'full')

    def test_same_arm_twice_via_alias_is_rejected(self):
        with self.assertRaises(ValueError):
            make_plan(self.protocol,'paper',['p0','deeprwcap'])
        with self.assertRaises(ValueError):
            make_plan(self.protocol,'paper',['bpr','capr'])

    def test_display_changes_do_not_rewrite_saved_data(self):
        report={'complete':True,'valid':3,'planned':3,'macro':{
            a:{'raw_macro_error_percent':1.0,'s24_macro_error_percent':2.0,
               'endpoint_macro_error_percent':3.0} for a in ['p0','bpr','full']},
            'module_comparisons':{'whole method: full CER - p0 raw':
                {'mean_difference_pp':0.0,'ci95_pp':None}},'resources':{},'scope':'Synthetic fixture'}
        with tempfile.TemporaryDirectory() as d:
            save_summary(d,report)
            self.assertEqual(json.loads((Path(d)/'summary.json').read_text()),report)
            text=(Path(d)/'summary.md').read_text(encoding='utf-8')
            self.assertIn('| DeepRWCap |',text)
            self.assertIn('| CARE-RWCap |',text)
            self.assertIn('| DeepRWCap + CAPR |',text)
            self.assertNotIn('BPR',text)
            self.assertIn('CARE-RWCap CER - DeepRWCap raw',text)
            self.assertNotIn('p0',text)


if __name__=='__main__':unittest.main()
