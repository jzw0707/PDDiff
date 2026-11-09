"""Dynamic Multiscale Spatial Projector (DMSP)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_blur_kernel1d(k=5, device="cpu", dtype=torch.float32):
    if k == 3:
        base = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
    elif k == 7:
        base = torch.tensor(
            [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
            device=device,
            dtype=dtype,
        )
    else:
        base = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    return base / base.sum()


class MSBlurDW(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        ker3 = torch.outer(_build_blur_kernel1d(3), _build_blur_kernel1d(3))[
            None, None
        ]
        ker5 = torch.outer(_build_blur_kernel1d(5), _build_blur_kernel1d(5))[
            None, None
        ]
        ker7 = torch.outer(_build_blur_kernel1d(7), _build_blur_kernel1d(7))[
            None, None
        ]
        self.register_buffer("w3", ker3.repeat(channels, 1, 1, 1))
        self.register_buffer("w5", ker5.repeat(channels, 1, 1, 1))
        self.register_buffer("w7", ker7.repeat(channels, 1, 1, 1))

        hid = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, 3 * channels, 1, bias=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        g = self.gate(x).view(B, 3, C, 1, 1)
        g = torch.softmax(g, dim=1)
        y3 = F.conv2d(x, self.w3, padding=1, groups=C)
        y5 = F.conv2d(x, self.w5, padding=2, groups=C)
        y7 = F.conv2d(x, self.w7, padding=3, groups=C)
        return g[:, 0] * y3 + g[:, 1] * y5 + g[:, 2] * y7


class DMSP(nn.Module):
    def __init__(self, in_ch, use_dw=True, reduction=16, with_proj=False):
        super().__init__()
        self.in_ch = in_ch
        self.with_proj = with_proj
        self.msblur = MSBlurDW(in_ch, reduction=reduction)
        self.unshuffle = nn.PixelUnshuffle(2)
        self.fold = nn.Conv2d(
            in_ch * 4, in_ch, kernel_size=1, groups=in_ch, bias=False
        )
        with torch.no_grad():
            w = torch.zeros(in_ch, 1, 1, 1)
            w[:] = 0.25
            self.fold.weight.copy_(w.repeat(1, 4, 1, 1))

        self.use_dw = use_dw
        if use_dw:
            self.dw = nn.Conv2d(
                in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False
            )
            self.bn_dw = nn.BatchNorm2d(in_ch)

        self.pw = nn.Conv2d(in_ch, in_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(2, 2)
        if with_proj:
            self.proj = nn.Conv2d(in_ch, in_ch, 1, 1, 0, bias=False)

        hid = max(in_ch // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, hid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, in_ch, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.msblur(x)
        y = self.unshuffle(y)
        y = self.fold(y)
        if self.use_dw:
            y = self.bn_dw(self.dw(y))
        y = self.bn(self.pw(y))

        s = self.pool(x)
        if self.with_proj:
            s = self.proj(s)

        out = self.act(y + s)
        return out * self.se(out)


def apply_dmsp(module, feature):
    """Run DMSP and restore the input spatial size."""
    height, width = feature.shape[-2:]
    even_height = height + (height % 2)
    even_width = width + (width % 2)
    resized = feature
    if (even_height, even_width) != (height, width):
        resized = F.interpolate(
            feature, size=(even_height, even_width), mode="bilinear", align_corners=False
        )
    projected = module(resized)
    if projected.shape[-2:] != (height, width):
        projected = F.interpolate(
            projected, size=(height, width), mode="bilinear", align_corners=False
        )
    return projected


__all__ = ["DMSP", "MSBlurDW", "apply_dmsp"]
