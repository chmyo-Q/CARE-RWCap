"""CPU-only scheduling/statistics tests with explicitly synthetic observations."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT,dump
from benchmark import make_plan,execute
from summarize import summarize
from evaluate import evaluate
from readout.frozen import geometry


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads((ROOT/'configs/paper_protocol.json').read_text())

    def populate(self,directory,plan):
        dump(directory/'plan.json',plan)
        dump(directory/'status.json',{'state':'completed','failed':None})
        for item in plan['runs']:
            d=directory/item['directory'];d.mkdir(parents=True)
            # All numbers are artificial; no solver or performance claim.
            v=1.0 if item['arm']=='p0' else .8
            cfg=item['config'];master,nets=geometry(ROOT/cfg['geometry']);ref=cfg['reference_F']
            text=[f'Master {master}',f'Capacitance on {master} = {ref*(1+v/100):.17g}']
            text += [f'Capacitance on {net} = {-ref*(1+v/200)/(len(nets)-1):.17g}' for net in sorted(nets-{master})]
            text += ['RWCap has run 100 walks (2.5 hops/walk)','Elapsed time: 1.0sec, CPU time: 2.0sec']
            (d/'result.out').write_text('\n'.join(text)+'\n')
            row=evaluate(ROOT/cfg['geometry'],d/'result.out',ref,master)
            row.update(case=item['case'],arm=item['arm'],initial_seed=item['seed'])
            dump(d/'metrics.json',row)
            dump(d/'run.json',{'arm':item['arm'],'configuration':item['config']})
        for item in plan['warmups']:
            d=directory/item['directory'];d.mkdir(parents=True)
            dump(d/'metrics.json',{'synthetic_test_warmup':True})

    def test_paper_plan_is_complete_deterministic_and_balanced(self):
        p=make_plan(self.protocol,'paper')
        self.assertEqual(p,make_plan(self.protocol,'paper'))
        self.assertEqual(len(p['runs']),300)
        self.assertEqual(len(p['warmups']),3)
        keys={(r['case'],r['seed'],r['arm']) for r in p['runs']}
        self.assertEqual(len(keys),300)
        self.assertEqual({r['seed'] for r in p['runs']},set(range(2029,2039)))

    def test_complete_macro_and_same_readout_differences(self):
        p=make_plan(self.protocol,'paper')
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);self.populate(d,p)
            r=summarize(d)
            self.assertTrue(r['complete'])
            self.assertAlmostEqual(r['macro']['p0']['endpoint_macro_error_percent'],1.)
            self.assertAlmostEqual(r['macro']['full']['endpoint_macro_error_percent'],.4)
            self.assertAlmostEqual(r['differences']['full']['mean_difference_pp'],-.6)
            self.assertAlmostEqual(r['differences']['full']['s24_difference_vs_p0_s24_pp'],-.1)
            self.assertEqual(r['per_case'][0]['raw_error_percent']['repeat_sd'],0.)

    def test_missing_observation_withholds_macro(self):
        p=make_plan(self.protocol)
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);self.populate(d,p)
            (d/p['runs'][0]['directory']/'metrics.json').unlink()
            r=summarize(d)
            self.assertFalse(r['complete']);self.assertEqual(r['macro'],{})
            self.assertEqual(len(r['missing']),1)

    def test_wrong_seed_or_configuration_rejected(self):
        p=make_plan(self.protocol)
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);self.populate(d,p)
            file=d/p['runs'][0]['directory']/'metrics.json'
            row=json.loads(file.read_text());row['initial_seed']=9000;dump(file,row)
            r=summarize(d)
            self.assertFalse(r['complete']);self.assertEqual(len(r['invalid']),1)
            self.assertEqual(r['macro'],{})

    def test_failed_batch_does_not_become_complete_when_files_exist(self):
        p=make_plan(self.protocol)
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);self.populate(d,p)
            dump(d/'status.json',{'state':'failed','failed':{'message':'synthetic failure'}})
            r=summarize(d)
            self.assertFalse(r['complete']);self.assertEqual(r['differences'],{})

    def test_executor_preserves_failure_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            stub=root/'failing_test_runner.py';stub.write_text('raise SystemExit(17)\n')
            d=root/'batch'
            with self.assertRaises(subprocess.CalledProcessError):
                execute(make_plan(self.protocol),d,runner=stub)
            state=json.loads((d/'status.json').read_text())
            self.assertEqual(state['state'],'failed')
            self.assertEqual(len(list(d.glob('launch_*.log'))),1)
            self.assertEqual(len(summarize(d)['missing']),3)

    def test_edited_metrics_cannot_override_solver_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);p=make_plan(self.protocol);self.populate(d,p)
            f=d/p['runs'][0]['directory']/'metrics.json';row=json.loads(f.read_text())
            row['raw_self_error_percent']=0.;dump(f,row)
            r=summarize(d)
            self.assertFalse(r['complete']);self.assertEqual(r['macro'],{})
            self.assertIn('original solver output',r['invalid'][0]['reason'])

    def test_duplicate_plan_cannot_double_count_an_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);p=make_plan(self.protocol);self.populate(d,p)
            p['runs'][1]=copy.deepcopy(p['runs'][0]);dump(d/'plan.json',p)
            with self.assertRaisesRegex(ValueError,'Duplicate planned'):
                summarize(d)

    def test_one_seed_has_no_confidence_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);p=make_plan(self.protocol);self.populate(d,p)
            r=summarize(d)
            self.assertIsNone(r['module_comparisons']['whole method: full CER - p0 raw']['ci95_pp'])


if __name__=='__main__':unittest.main()
