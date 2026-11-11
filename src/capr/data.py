"""Memory-mapped GGFT Poisson records and the fixed CAPR split."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

N = 23
VOXELS = N**3
KERNEL_VALUES = 6 * N * N
RECORD_VALUES = VOXELS + 16 * 7 + KERNEL_VALUES


class PoissonDataset(Dataset):
    def __init__(self, path):
        path = Path(path)
        if path.stat().st_size < 16 or (path.stat().st_size - 16) % (RECORD_VALUES * 8):
            raise ValueError("Incomplete GGFT records")
        data = np.memmap(path, dtype="<f8", mode="r")
        if data[:2].tolist() != [float(N), 1.0]:
            raise ValueError("Expected GGFT header N=23, block_width=1")
        self.records = data[2:].reshape(-1, RECORD_VALUES)
        if not len(self.records):
            raise ValueError("Empty Poisson dataset")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        x = np.array(row[:VOXELS], dtype=np.float32).reshape(1, 1, N, N, N)
        # Normalize the FP64 reference before conversion, as in training.
        y = np.abs(row[-KERNEL_VALUES:].reshape(6, N, N)[0]) + 1e-10
        y = np.asarray(y / y.sum(), dtype=np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Nonfinite Poisson record: {index}")
        return torch.from_numpy(x), torch.from_numpy(y)


def normalize_input(x):
    maximum = x.amax(dim=(-3, -2, -1), keepdim=True)
    return x / maximum.masked_fill(maximum == 0, 1.0)


def split_indices(samples, validation_samples, seed):
    if not 0 < validation_samples < samples:
        raise ValueError("Validation count must be between zero and the dataset size")
    indices = np.random.default_rng(seed).permutation(samples)
    return indices[:-validation_samples].tolist(), indices[-validation_samples:].tolist()
