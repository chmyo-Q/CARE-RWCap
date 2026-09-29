"""CPU baseline orchestration checks using explicitly synthetic solver output."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import baselines


class BaselineTest(unittest.TestCase):
    def test_full_plan_preserves_input_reference_and_explicit_seed(self):
        p = baselines.make_plan('fdm', 'paper')
        self.assertEqual(len(p['runs']), 100)
        self.assertEqual(len({(x['config']['case'], x['config']['seed']) for x in p['runs']}), 100)
        self.assertEqual(p['workers'], 16)
        for item in p['runs']:
            argv = baselines.command(p, item, Path('output'))
            self.assertEqual(argv[argv.index('--seed')+1], str(item['config']['seed']))
            self.assertEqual(argv[argv.index('--c-ratio')+1], str(item['config']['c_ratio']))
        with self.assertRaises(ValueError): baselines.make_plan('agf', workers=0)

    def test_failed_solve_preserved_without_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(baselines.os, 'access', return_value=True):
            out = Path(tmp)/'failed'
            def failure(*args, **kwargs): raise subprocess.CalledProcessError(7, args[0])
            with self.assertRaises(subprocess.CalledProcessError):
                baselines.execute(baselines.make_plan('agf'), out, runner=failure)
            self.assertEqual(json.loads((out/'status.json').read_text())['state'], 'failed')
            self.assertFalse((out/'summary.json').exists())
            self.assertTrue((out/'case8/seed2029/run.json').exists())

    def test_raw_endpoint_and_no_neural_environment_leak(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(baselines.os, 'access', return_value=True):
            out = Path(tmp)/'ok'; p = baselines.make_plan('microwalk')
            cfg = p['runs'][0]['config']; ref = cfg['reference_F']; master = cfg['master']
            def solver(*args, **kwargs):
                text = f'Master {master}\nCapacitance on {master} = {ref*1.01:.17g}\n'
                text += 'RWCap has run 100 walks (2.5 hops/walk)\nElapsed time: 1.0sec, CPU time: 2.0sec\n'
                (kwargs['cwd']/'result.out').write_text(text)
                self.assertNotIn('LD_PRELOAD', kwargs['env'])
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            with patch.dict(os.environ, {'LD_PRELOAD': '/example/cpgr.so', 'S29_GRADIENT_JOINT_ENABLE': '1'}):
                r = baselines.execute(p, out, runner=solver)
            self.assertAlmostEqual(r['raw_macro_self_error_percent'], 1.)
            self.assertNotIn('s24_capacitance_F', json.loads((out/'case8/seed2029/metrics.json').read_text()))
            self.assertTrue((out/'per_run.csv').is_file())
            self.assertIsNone(r['between_case_sd'])
            with self.assertRaises(FileExistsError): baselines.execute(p, out, runner=solver)


if __name__ == '__main__': unittest.main()
