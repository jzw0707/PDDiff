"""SMSP mask perturbation and the TPA log-SNR sampling offset.

SMSP perturbs the target mask before Gaussian noising. The bounded translation,
local morphology, and soft mixing keep the original mask as the training target.

The TPA offset shifts the cosine log-SNR by a constant. It adds no learned
parameters and is shared by forward noising and reverse sampling.
"""

import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


class SMSP(nn.Module):
    """Corrupt a normalized binary diffusion source without changing its GT.

    With strength < 0.5, every original foreground pixel remains positive in
    [-1, 1], even for one-pixel or border targets. Empty masks stay empty.
    This module has no parameters/buffers and is inactive during evaluation.
    """

    def __init__(
        self,
        probability=0.5,
        strength=0.35,
        max_shift=1,
        kernel_size=3,
        patch_size=16,
        time_min=0.05,
        time_max=0.95,
    ):
        super().__init__()
        if not 0 <= probability <= 1 or not 0 <= strength <= 1:
            raise ValueError("SMSP probability and strength must be in [0, 1]")
        if not 0 <= time_min <= time_max <= 1:
            raise ValueError("SMSP requires 0 <= time_min <= time_max <= 1")
        for name, value, minimum in (
            ("max_shift", max_shift, 0),
            ("kernel_size", kernel_size, 1),
            ("patch_size", patch_size, 1),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"SMSP {name} must be an integer >= {minimum}")
        if kernel_size % 2 != 1:
            raise ValueError("SMSP kernel_size must be odd")
        self.probability = float(probability)
        self.strength = float(strength)
        self.max_shift = max_shift
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.time_min = float(time_min)
        self.time_max = float(time_max)

    def _translate(self, mask):
        if self.max_shift == 0:
            return mask
        batch, _, height, width = mask.shape
        shifts = torch.randint(
            -self.max_shift, self.max_shift + 1, (batch, 2), device=mask.device
        )
        rows = torch.arange(height, device=mask.device)[None, :, None] - shifts[:, 0, None, None]
        cols = torch.arange(width, device=mask.device)[None, None, :] - shifts[:, 1, None, None]
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        indices = rows.clamp(0, height - 1) * width + cols.clamp(0, width - 1)
        shifted = mask.flatten(2).gather(2, indices.flatten(1).unsqueeze(1))
        return shifted.view_as(mask) * valid[:, None].to(mask.dtype)

    @torch.no_grad()
    def forward(self, source, times):
        if not self.training or self.probability == 0 or self.strength == 0:
            return source
        if source.ndim != 4 or source.shape[1] != 1 or not source.is_floating_point():
            raise ValueError("SMSP requires a floating [B, 1, H, W] mask in [-1, 1]")
        if times.shape != (source.shape[0],):
            raise ValueError("SMSP requires one diffusion time per image")

        batch, _, height, width = source.shape
        apply = (torch.rand(batch, device=source.device) < self.probability)
        apply = apply & (times >= self.time_min) & (times <= self.time_max)
        mask = (source + 1) * 0.5
        shifted = self._translate(mask)
        radius = self.kernel_size // 2
        # Explicit background padding prevents target wraparound at image edges.
        padded = F.pad(shifted, (radius,) * 4, value=0)
        dilated = F.max_pool2d(padded, self.kernel_size, stride=1)
        eroded = -F.max_pool2d(-padded, self.kernel_size, stride=1)
        operation = torch.randint(0, 3, (batch, 1, 1, 1), device=source.device)
        candidate = torch.where(operation == 1, dilated, shifted)
        candidate = torch.where(operation == 2, eroded, candidate)

        # Coherent patches perturb only part of the contour, not independent
        # pixels throughout the background. No distant false targets are added.
        patches = torch.rand(
            batch, 1, math.ceil(height / self.patch_size), math.ceil(width / self.patch_size),
            device=source.device,
        )
        gate = F.interpolate(patches, size=(height, width), mode="nearest") > 0.5
        mixed = mask + self.strength * gate.to(mask.dtype) * (candidate - mask)
        return torch.where(apply[:, None, None, None], mixed * 2 - 1, source)


class TPALogSNRSchedule:
    def __init__(self, base_schedule, *, log_snr_offset):
        if (
            isinstance(log_snr_offset, bool)
            or not isinstance(log_snr_offset, Real)
            or not math.isfinite(log_snr_offset)
        ):
            raise ValueError("TPA log_snr_offset must be a finite real number")
        self.base_schedule = base_schedule
        self.log_snr_offset = float(log_snr_offset)

    def __call__(self, times):
        return self.base_schedule(times) + self.log_snr_offset


__all__ = ["SMSP", "TPALogSNRSchedule"]
