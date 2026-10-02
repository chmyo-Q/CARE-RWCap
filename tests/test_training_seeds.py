"""Guard the published supplement against incomplete or mismatched records."""
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from summarize_training_seeds import summarize


class TrainingSeedTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)/'data'
        shutil.copytree(ROOT/'results/training_seeds', self.data)
        self.protocol = json.loads((ROOT/'configs/paper_protocol.json').read_text())

    def mutate(self, filename, edit):
        path = self.data/filename
        with path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = list(reader)
        edit(rows)
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_archived_summary_and_all_cases_retained(self):
        cases, seeds, aggregate = summarize(self.data, self.protocol)
        self.assertEqual(len(cases), 30)
        self.assertEqual([r['wins'] for r in seeds], [7, 7, 7])
        self.assertAlmostEqual(aggregate['baseline_macro_percent'], 0.927750543349835, places=13)
        self.assertAlmostEqual(aggregate['mean_macro_selfcaperr_percent'], 0.8187972707470784, places=13)
        self.assertAlmostEqual(aggregate['sample_sd_macro_selfcaperr_percent'], 0.01991409174931329, places=13)

    def test_missing_observation_is_rejected(self):
        self.mutate('repeat_results.csv', lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, 'Incomplete Full'):
            summarize(self.data, self.protocol)

    def test_duplicate_observation_is_rejected(self):
        self.mutate('repeat_results.csv', lambda rows: rows.append(dict(rows[0])))
        with self.assertRaisesRegex(ValueError, 'duplicate Full'):
            summarize(self.data, self.protocol)

    def test_mismatched_reference_is_rejected(self):
        self.mutate('repeat_results.csv', lambda rows: rows[0].update(reference_F='1e-14'))
        with self.assertRaisesRegex(ValueError, 'Full reference'):
            summarize(self.data, self.protocol)

    def test_edited_capacitance_is_rejected(self):
        self.mutate('repeat_results.csv', lambda rows: rows[0].update(cer_F='1e-14'))
        with self.assertRaisesRegex(ValueError, 'CER error'):
            summarize(self.data, self.protocol)

    def test_missing_baseline_is_rejected(self):
        self.mutate('baseline_runs.csv', lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, 'Incomplete baseline'):
            summarize(self.data, self.protocol)


if __name__ == '__main__':
    unittest.main()
