"""Exercise readout decisions without a reference influencing activation."""
import sys
import unittest
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT
from evaluate import evaluate
from readout.frozen import s24_value, logical


class ReadoutTest(unittest.TestCase):
    def test_complete_negative_couplings(self):
        self.assertEqual(s24_value({'A':4.,'B':-1.,'C':-2.}, {'A','B','C'}, 'A'),
                         (3.,'abs_coupling_sum'))

    def test_missing_and_extra(self):
        for values in [{'A':4.,'B':-1.}, {'A':4.,'B':-1.,'C':-2.,'D':-1.}]:
            self.assertEqual(s24_value(values,{'A','B','C'},'A'),
                             (4.,'identity_incomplete_or_extra_columns'))

    def test_sign_and_degenerate(self):
        for values in [{'A':4.,'B':1.}, {'A':4.,'B':0.}, {'A':4.}]:
            self.assertEqual(s24_value(values,set(values),'A'),(4.,'identity_sign_or_degenerate'))

    def test_logical_names(self):
        self.assertEqual(logical('42__B'),'B')
        self.assertEqual(logical('B_42'),'B_42')

    def test_duplicate_logical_columns_rejected_in_both_orders(self):
        f=ROOT/'tests/fixtures'
        original=(f/'complete.out').read_text()
        for before in ['Capacitance on 1__B', 'Capacitance on 2__C']:
            text=original.replace(before,'Capacitance on 3__B = -4.0e-15\n'+before)
            with tempfile.TemporaryDirectory() as directory:
                out=Path(directory)/'duplicate.out'
                out.write_text(text)
                with self.assertRaisesRegex(ValueError,'duplicate logical column'):
                    evaluate(f/'small.cap3d',out,4e-15)

    def test_duplicate_master_rejected(self):
        f=ROOT/'tests/fixtures'
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'duplicate.out'
            out.write_text((f/'complete.out').read_text()+'\nMaster A\n')
            with self.assertRaisesRegex(ValueError,'duplicate master block'):
                evaluate(f/'small.cap3d',out,4e-15)

    def test_parse_and_reference_independence(self):
        f=ROOT/'tests/fixtures'
        a=evaluate(f/'small.cap3d',f/'complete.out',4e-15)
        b=evaluate(f/'small.cap3d',f/'complete.out',3e-15)
        self.assertEqual(a['s24_capacitance_F'],b['s24_capacitance_F'])
        self.assertAlmostEqual(a['raw_self_error_percent'],0.)
        self.assertAlmostEqual(a['s24_self_error_percent'],25.)
        self.assertAlmostEqual(b['s24_self_error_percent'],0.)
        self.assertEqual(a['walks'],100)
        self.assertEqual(a['total_steps_approx'],250)
        with self.assertRaises(ValueError):
            evaluate(f/'small.cap3d',f/'complete.out',-1.)
        with self.assertRaises(ValueError):
            evaluate(f/'small.cap3d',f/'complete.out',master='B')


if __name__=='__main__': unittest.main()
