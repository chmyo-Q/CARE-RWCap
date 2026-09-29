#!/usr/bin/env python3
"""BPR: bounded residual Poisson predictor anchored to the paper P0."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

N = 23


class PosEmbed2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        y, x = torch.linspace(0, 1, N), torch.linspace(0, 1, N)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("pe", torch.stack((xx, yy), 0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, self.pe.expand_as(x[:, :2])), dim=1)


class DepthSepBlock(nn.Module):
    def __init__(self, ch: int, ch_out: int, dilation: int = 1) -> None:
        super().__init__()
        self.skip = ch == ch_out
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, groups=ch, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(), nn.Conv2d(ch, ch_out, 1, bias=False),
            nn.BatchNorm2d(ch_out), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FaceSolver(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pe_mode = "grid"
        self.pe = PosEmbed2D()
        self.z_weight = nn.Sequential(
            nn.Conv2d(N + 2, 16, 1, bias=False), nn.BatchNorm2d(16), nn.GELU())
        self.blocks = nn.Sequential(
            DepthSepBlock(16, 16), DepthSepBlock(16, 8),
            DepthSepBlock(8, 4, dilation=2), DepthSepBlock(4, 2, dilation=3))
        self.head = nn.Conv2d(2, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.z_weight(self.pe(x))))


class P0Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.name = "BeefedFacePredictor"
        self.solver = FaceSolver()
        self.head_mode = "relu"

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.solver(x.squeeze(1).squeeze(1))[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.forward_raw(x)) + 1.0e-10
        return value / value.sum(dim=[1, 2], keepdim=True)


class ResidualHead(nn.Module):
    """Small local residual and a state-dependent trust gate."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(N, 24, 1, bias=False), nn.BatchNorm2d(24), nn.GELU(),
            nn.Conv2d(24, 24, 3, padding=1, groups=24, bias=False),
            nn.BatchNorm2d(24), nn.GELU(),
            nn.Conv2d(24, 24, 3, padding=2, dilation=2, groups=24, bias=False),
            nn.BatchNorm2d(24), nn.GELU(),
            nn.Conv2d(24, 12, 1, bias=False), nn.BatchNorm2d(12), nn.GELU(),
        )
        self.residual = nn.Conv2d(12, 1, 1, bias=True)
        self.gate = nn.Linear(12, 1, bias=True)
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)

    def forward(self, face: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.features(face)
        residual = torch.tanh(self.residual(feature)[:, 0])
        pooled = feature.mean(dim=[2, 3])
        gate = torch.sigmoid(self.gate(pooled)).reshape(face.shape[0], 1, 1)
        return residual, gate


class ResidualFactorizedPredictor(nn.Module):
    """P0 conditional predictor plus a bounded multiplicative residual.

    The residual output layer is zero initialized, so this module initially recovers P0 up to the
    numerical effects of clamping and softmax normalization.
    """
    def __init__(self) -> None:
        super().__init__()
        self.max_log_dose = 0.125
        self.anchor = P0Predictor()
        self.adapter = ResidualHead()
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor.eval()
        return self

    def components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p0 = self.anchor(x)
        face = x.squeeze(1).squeeze(1)
        residual, gate = self.adapter(face)
        residual = residual - (p0 * residual).sum(dim=[1, 2], keepdim=True)
        dose = self.max_log_dose * gate
        return p0, residual, dose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p0, residual, dose = self.components(x)
        logits = torch.log(p0.clamp_min(1.0e-10)) + dose * residual
        flat = logits.reshape(logits.shape[0], 529)
        return torch.softmax(flat, dim=1).reshape(logits.shape[0], 23, 23)


def _extract_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "predictor"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and candidate:
                payload = candidate; break
    if not isinstance(payload, dict) or not payload:
        raise ValueError("P0 checkpoint does not contain a state_dict")
    state: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            continue
        name = str(key)
        for prefix in ("module.", "model.", "predictor."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name.rsplit(".", 1)[-1] not in {"total_ops", "total_params"}:
            state[name] = value
    return state


def load_p0(model: P0Predictor, checkpoint: Path) -> None:
    state = _extract_state(torch.load(checkpoint, map_location="cpu", weights_only=True))
    result = model.load_state_dict(state, strict=False)
    missing = [x for x in result.missing_keys if not x.endswith("num_batches_tracked")]
    unexpected = [x for x in result.unexpected_keys if not x.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(f"P0 checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def build_model(checkpoint: Path) -> ResidualFactorizedPredictor:
    model = ResidualFactorizedPredictor()
    load_p0(model.anchor, checkpoint)
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
