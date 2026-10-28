"""Test capacitance parsing and CER selection."""
import sys
import unittest
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT
from evaluate_readout import evaluate, coupling_error
from readout.cer import cer_value
from readout.parser import logical


class ReadoutTest(unittest.TestCase):
    def test_coupling_metric_uses_shared_magnitudes_and_excludes_self(self):
        with tempfile.TemporaryDirectory() as directory:
            ref=Path(directory)/'reference.dspf'
            ref.write_text('*|NET A 1000\n*|NET B 1\n*|NET C 4\n*|NET D 100\nC1 A B 1\nC2 A C 4\nC3 A D 100\n')
            result=coupling_error({'A':100.,'B':-2.,'C':5.},'A',ref)
            self.assertAlmostEqual(result['normalized_l1_percent'],40.)
            self.assertEqual(result['shared_offdiagonal_entries'],2)
            self.assertEqual(result['reference_mass_F'],5.)
            with self.assertRaisesRegex(ValueError,'shared coupling'):
                coupling_error({'A':100.},'A',ref)

    def test_coupling_spice_units_ground_and_upstream_averaging(self):
        with tempfile.TemporaryDirectory() as directory:
            ref=Path(directory)/'reference.spice'
            ref.write_text('** A 10fF (total)\nCp1 A 0 1fF\nCp2 A 0 3fF\n')
            result=coupling_error({'A':10e-15,'GROUND':-3e-15},'A',ref)
            self.assertAlmostEqual(result['normalized_l1_percent'],50.)
            with self.assertRaisesRegex(ValueError,'master missing'):
                coupling_error({'B':1.},'B',ref)

    def test_optional_coupling_reference_never_changes_self_readout(self):
        f=ROOT/'tests/fixtures'
        with tempfile.TemporaryDirectory() as directory:
            ref=Path(directory)/'reference.dspf'
            ref.write_text('*|NET A 4e-15\n*|NET B 1e-15\n*|NET C 2e-15\nC1 A B 1e-15\nC2 A C 2e-15\n')
            original=evaluate(f/'small.cap3d',f/'complete.out',4e-15)
            extended=evaluate(f/'small.cap3d',f/'complete.out',4e-15,coupling_reference=ref)
            self.assertAlmostEqual(extended.pop('coupling_error')['normalized_l1_percent'],0.)
            self.assertEqual(extended,original)

    def test_complete_negative_couplings(self):
        self.assertEqual(cer_value({'A':4.,'B':-1.,'C':-2.}, {'A','B','C'}, 'A'),
                         (3.,'abs_coupling_sum'))

    def test_missing_and_extra(self):
        for values in [{'A':4.,'B':-1.}, {'A':4.,'B':-1.,'C':-2.,'D':-1.}]:
            self.assertEqual(cer_value(values,{'A','B','C'},'A'),
                             (4.,'identity_incomplete_or_extra_columns'))

    def test_sign_and_degenerate(self):
        for values in [{'A':4.,'B':1.}, {'A':4.,'B':0.}, {'A':4.}]:
            self.assertEqual(cer_value(values,set(values),'A'),(4.,'identity_sign_or_degenerate'))

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
