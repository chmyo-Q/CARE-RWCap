"""Exercise the single-run command through result-file serialization."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import common


class RunCommandTest(unittest.TestCase):
    def test_solver_output_and_diagnostics_produce_json_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            output = folder / "run"
            config = folder / "config.json"
            config.write_text(json.dumps({
                "case": "fixture", "master": "A", "geometry": str(ROOT / "tests/fixtures/small.cap3d"),
                "reference_F": 4e-15, "seed": 2029, "workers": 8,
                "omp_threads": 1, "mkl_threads": 1, "p": .01, "c": .01, "c_ratio": .3,
            }), encoding="utf-8")

            def solver(command, **kwargs):
                run = kwargs["cwd"]
                (run / "result.out").write_bytes((ROOT / "tests/fixtures/complete.out").read_bytes())
                records = {
                    "SEED_AUDIT.json": {"requested_seed": 2029, "torch_manual_seed_called": True,
                                        "core_initial_seed_set": 2029, "core_set_seed_calls": 1},
                    "PROJECTION_COUNTS.json": {"cuda_status": 0, "calls": 1, "samples": 1, "masks": [0, 1, 0, 0]},
                    "JOINT_COUNTS.json": {"cuda_status": 0, "input_xyz_masks": [0, 1, 0, 0, 0, 0, 0, 0]},
                }
                for name, record in records.items():
                    (run / name).write_text(json.dumps(record), encoding="utf-8")

            argv = [str(ROOT / "scripts/run.py"), "--config", str(config),
                    "--arm", "care-rwcap", "--output", str(output)]
            with patch.object(sys, "argv", argv), patch.object(common, "check_files"), \
                 patch.object(common, "runtime_env", return_value={}) as environment, \
                 patch("subprocess.run", side_effect=solver), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(ROOT / "scripts/run.py"), run_name="__main__")
            result = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(result["arm"], "care-rwcap")
            self.assertEqual(result["initial_seed"], 2029)
            self.assertTrue(result["cer_applied"])
            self.assertEqual(environment.call_args.kwargs["diagnostics"], output.resolve())
            self.assertTrue((output / "run.json").is_file())


if __name__ == "__main__":
    unittest.main()