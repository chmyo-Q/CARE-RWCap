"""Check the GGFT training path, frozen anchor, and source export."""

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install torch for CAPR training tests")
class CAPRTrainingTest(unittest.TestCase):
    def test_dataset_normalization_and_fixed_validation_indices(self):
        from capr.data import PoissonDataset, RECORD_VALUES, VOXELS, normalize_input, split_indices

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poisson.bin"
            records = np.ones((4, RECORD_VALUES), dtype="<f8")
            records[:, :VOXELS] *= 3
            np.concatenate([np.array([23.0, 1.0]), records.ravel()]).astype("<f8").tofile(path)
            dataset = PoissonDataset(path)
            x, y = dataset[0]
            self.assertEqual(x.shape, (1, 1, 23, 23, 23))
            torch.testing.assert_close(y.sum(), torch.tensor(1.0))
            torch.testing.assert_close(normalize_input(x), torch.ones_like(x))
            del dataset
            train, validation = split_indices(100000, 10000, 20260805)
            self.assertFalse(set(train) & set(validation))
            self.assertEqual(len(train), 90000)
            import json

            root = Path(__file__).resolve().parents[1]
            expected = json.loads((root / "configs/transition_validation.json").read_text())
            self.assertEqual(validation, expected["poisson_validation_indices"])

    def test_training_updates_only_adapter_and_matches_torchscript(self):
        from capr.model import build_model
        from capr.loss import capr_loss

        torch.set_num_threads(1)
        torch.manual_seed(2029)
        root = Path(__file__).resolve().parents[1]
        model = build_model(root / "models/checkpoints/deeprwcap/PoissonPredictor_best.pt")
        anchor_before = {k: v.clone() for k, v in model.anchor.state_dict().items()}
        before = model.adapter.residual.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=3e-4, weight_decay=1e-6)
        model.train()
        self.assertFalse(model.anchor.training)
        x = torch.rand(2, 1, 1, 23, 23, 23)
        target = torch.softmax(torch.randn(2, 529), dim=1).reshape(2, 23, 23)
        prediction = model(x)
        anchor, _, _ = model.components(x)
        loss = capr_loss(prediction, target, anchor)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.adapter.residual.weight))
        for key, value in model.anchor.state_dict().items():
            self.assertTrue(torch.equal(anchor_before[key], value), key)
        model.eval()
        scripted = torch.jit.script(model)
        with torch.inference_mode():
            torch.testing.assert_close(scripted(x), model(x), rtol=1e-5, atol=1e-7)
