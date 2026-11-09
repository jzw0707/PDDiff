"""Memory-guided Spectral Alignment (MSA)."""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_size: int = 256,
        patch: int = 16,
        mem_slots: int = 256,
        mid_channels: int = 32,
        topk: int = 8,
        temperature: float = 1.0,
        use_overlap: bool = False,
        drop_out: float = 0.125,
        sim_mode: str = "fft",
        fft_crop_ratio: float = 1.0,
        fft_log: bool = True,
        align_shift: bool = True,
        phase_up: int = 1,
        padding_mode: str = "border",
    ):
        super().__init__()
        assert in_size % patch == 0
        self.in_size = in_size
        self.patch = patch
        self.topk = topk
        self.temperature = temperature
        self.use_overlap = use_overlap
        self.drop_out = drop_out
        self.sim_mode = sim_mode
        self.fft_crop_ratio = float(fft_crop_ratio)
        self.fft_log = bool(fft_log)
        self.align_shift = bool(align_shift)
        self.phase_up = int(max(1, phase_up))
        self.padding_mode = padding_mode

        self.q_proj = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.norm_q = nn.LayerNorm(mid_channels * patch * patch)

        D = mid_channels * patch * patch
        self.memory = nn.Parameter(torch.randn(mem_slots, D) * 0.02)
        self.mem_norm = nn.LayerNorm(D)

        p = patch
        hann = torch.hann_window(p, periodic=False).float()
        self.register_buffer("fft_window", torch.outer(hann, hann))

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + mid_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

    def _unfold(self, x):
        B, _, self.H, self.W = x.shape
        if self.use_overlap:
            stride = self.patch // 2
            padding = self.patch // 4
        else:
            stride = self.patch
            padding = 0
        tokens = F.unfold(
            x, kernel_size=self.patch, stride=stride, padding=padding
        )
        N = tokens.shape[-1]
        return tokens.transpose(1, 2), N, stride, padding

    def _fold(self, tokens, stride, padding):
        B, N, D = tokens.shape
        tokens = tokens.transpose(1, 2)
        out = F.fold(
            tokens,
            output_size=(self.H, self.W),
            kernel_size=self.patch,
            stride=stride,
            padding=padding,
        )
        if stride < self.patch:
            ones = torch.ones(
                B, 1, self.H, self.W, device=out.device, dtype=out.dtype
            )
            norm = F.fold(
                F.unfold(
                    ones,
                    kernel_size=self.patch,
                    stride=stride,
                    padding=padding,
                ),
                output_size=(self.H, self.W),
                kernel_size=self.patch,
                stride=stride,
                padding=padding,
            )
            out = out / norm.clamp_min(1e-6)
        return out

    def _tokens_4d(self, qtokens):
        B, N, D = qtokens.shape
        C, p = self.q_proj.out_channels, self.patch
        return qtokens.view(B * N, C, p, p), B, N

    def _fft_mag_vec(self, feat_bn):
        p = self.patch
        x = feat_bn * self.fft_window[None, None]
        spec = torch.fft.rfft2(x, norm="ortho")
        mag = spec.abs()
        h = max(1, int(round(p * self.fft_crop_ratio)))
        w = max(1, int(round((p // 2 + 1) * self.fft_crop_ratio)))
        mag = mag[:, :, :h, :w]
        if self.fft_log:
            mag = torch.log1p(mag)
        vec = mag.mean(dim=1).flatten(1)
        return F.normalize(vec, dim=-1)

    def _phase_corr_shift(self, q, m, up=1):
        p = self.patch
        w = self.fft_window[None, None]
        Hp, Wp = p * up, p * up
        Q = torch.fft.rfft2(q * w, s=(Hp, Wp), norm="ortho")
        M = torch.fft.rfft2(m * w, s=(Hp, Wp), norm="ortho")
        R = Q * torch.conj(M)
        R = R / R.abs().clamp_min(1e-8)
        corr = torch.fft.irfft2(R, s=(Hp, Wp), norm="ortho").real
        corr = corr.sum(dim=1, keepdim=False)
        gy = torch.linspace(0, Hp - 1, Hp, device=corr.device).view(1, Hp, 1)
        gx = torch.linspace(0, Wp - 1, Wp, device=corr.device).view(1, 1, Wp)
        prob = torch.softmax(corr.flatten(1), dim=-1).view(-1, Hp, Wp)
        y = (prob * gy).sum(dim=(1, 2))
        x = (prob * gx).sum(dim=(1, 2))
        y = torch.where(y > (Hp / 2), y - Hp, y) / up
        x = torch.where(x > (Wp / 2), x - Wp, x) / up
        return x, y

    def _shift2d(self, x, dx, dy):
        B, C, p, _ = x.shape
        device, dtype = x.device, x.dtype
        lin = torch.linspace(-1, 1, p, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        base = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        dxn = dx * (2.0 / (p - 1))
        dyn = dy * (2.0 / (p - 1))
        grid = base.clone()
        grid[..., 0] = grid[..., 0] + dxn.view(B, 1, 1)
        grid[..., 1] = grid[..., 1] + dyn.view(B, 1, 1)
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        qmap = self.q_proj(x)
        qtokens, N, stride, padding = self._unfold(qmap)
        qtokens = self.norm_q(qtokens)
        mem = self.mem_norm(self.memory)

        if self.sim_mode == "dot":
            logits = torch.matmul(qtokens, mem.t()) / sqrt(mem.shape[1])
        elif self.sim_mode == "fft":
            Cmid = self.q_proj.out_channels
            p = self.patch
            q_bn, Bb, Nb = self._tokens_4d(qtokens)
            m_mp = mem.view(mem.shape[0], Cmid, p, p)
            z_q = self._fft_mag_vec(q_bn)
            z_m = self._fft_mag_vec(m_mp)
            logits = (z_q @ z_m.t()).view(Bb, Nb, -1)
        else:
            raise ValueError(f"Unknown sim_mode: {self.sim_mode}")

        M = logits.size(-1)
        if self.training and getattr(self, "drop_out", 0.0) > 0.0:
            p = float(self.drop_out)
            keep = torch.rand(M, device=logits.device) > p
            need = max(0, self.topk - int(keep.sum().item()))
            if need > 0:
                off = (~keep).nonzero(as_tuple=True)[0]
                pick = off[torch.randperm(off.numel(), device=logits.device)[:need]]
                keep[pick] = True
            logits = logits.masked_fill(~keep.view(1, 1, M), float("-inf"))

        if self.topk is not None and self.topk < mem.shape[0]:
            topv, topi = torch.topk(logits, k=self.topk, dim=-1)
            attn = F.softmax(topv / self.temperature, dim=-1)
            K = self.topk
            idx = topi
        else:
            attn = F.softmax(logits / self.temperature, dim=-1)
            K = M
            idx = torch.arange(M, device=x.device)[None, None, :].expand(B, N, M)

        Cmid, p = self.q_proj.out_channels, self.patch
        mem_val = self.memory.view(M, Cmid, p, p)
        V_sel = mem_val[idx]

        if self.align_shift:
            q_bn, Bb, Nb = self._tokens_4d(qtokens)
            q_patch = (
                q_bn.view(B, N, Cmid, p, p)
                .unsqueeze(2)
                .expand(-1, -1, K, -1, -1, -1)
            )
            q_pair = q_patch.reshape(-1, Cmid, p, p)
            m_pair = V_sel.reshape(-1, Cmid, p, p)
            dx, dy = self._phase_corr_shift(q_pair, m_pair, up=self.phase_up)
            V_warp = self._shift2d(m_pair, dx, dy).view(B, N, K, Cmid, p, p)
        else:
            V_warp = V_sel

        read_patch = (
            attn.view(B, N, K, 1, 1, 1) * V_warp
        ).sum(dim=2)
        read_tokens = read_patch.flatten(3).view(B, N, -1)
        readout_map = self._fold(read_tokens, stride, padding)
        fused = self.fuse(torch.cat([x, readout_map], dim=1))
        return fused


def apply_msa(module, feature):
    """Run MSA on a patch-aligned grid and restore the input spatial size."""
    height, width = feature.shape[-2:]
    patch = module.patch
    target_height = max(patch, ((height + patch - 1) // patch) * patch)
    target_width = max(patch, ((width + patch - 1) // patch) * patch)
    resized = feature
    if (target_height, target_width) != (height, width):
        resized = F.interpolate(
            feature, size=(target_height, target_width), mode="bilinear", align_corners=False
        )
    aligned = module(resized)
    if aligned.shape[-2:] != (height, width):
        aligned = F.interpolate(
            aligned, size=(height, width), mode="bilinear", align_corners=False
        )
    return aligned


__all__ = ["MSA", "apply_msa"]
