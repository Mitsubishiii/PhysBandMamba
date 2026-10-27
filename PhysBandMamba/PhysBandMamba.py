"""PhysBandMamba (MidWidth-72) -- self-contained model definition.

This file contains ONLY the modules that are actually executed by the
MidWidth-72 configuration whose weights ship alongside it. Every branch that
the config disables has been removed rather than left behind a dead flag, so
what is written here is exactly what runs:

    frames [B,T,3,128,128]
      -> Fusion_Stem            (RhythmMamba's stem, at narrow width C/4)
      -> gradual widen          (C/4 -> 72 spatiotemporal conv, then 1x1 -> C)
                                 also halves T
      -> face mask + attention  (FaceMaskHead on the channel-mean logit map,
                                 Attention_mask for the pooling weights)
      -> SpectralChannelPurifier(mean-detrend, RMS-norm, cosine band-pass)
      -> attention-weighted spatial pooling  -> [B,C,T/2]
      -> ClassicMambaNet        (2 unidirectional Mamba blocks + Conv1d head)
      -> upsample x2 -> BVP [B,T]

Removed relative to the research file (neural_methods/model/PhysBandMamba.py):
PhysNetStem, RawStem, DiffFusionStem, Stem3Block, LightDiffBranch,
TemporalContextBlock, TemporalBridge, the background-orthogonalisation and
motion-compensation helpers, the bidirectional Mamba path, the
multi-temporal-parallel path, the linear-detrend path, the late width
projection, and every auxiliary-loss hook. None of them are reachable under
this configuration.

Verified: this file reproduces the research implementation bit-for-bit on the
shipped checkpoints (max |diff| = 0.0) -- see verify_clean_model.py.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba

EPS = 1e-6


# ---------------------------------------------------------------------------
# Stem
# ---------------------------------------------------------------------------

class Fusion_Stem(nn.Module):
    """RhythmMamba's stem, verbatim. Fuses the raw frame with a 4-tap temporal
    difference stack; outputs [B*T, dim, H/8, W/8]."""

    def __init__(self, apha=0.5, belta=0.5, dim=24):
        super().__init__()
        self.stem11 = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
        self.stem12 = nn.Sequential(
            nn.Conv2d(12, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
        self.stem21 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
        self.stem22 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
        self.apha = apha
        self.belta = belta

    def forward(self, x):  # x: [N,D,C,H,W]
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:, :1], x[:, :1], x[:, :D - 2]], 1)
        x2 = torch.cat([x[:, :1], x[:, :D - 1]], 1)
        x3 = x
        x4 = torch.cat([x[:, 1:], x[:, D - 1:]], 1)
        x5 = torch.cat([x[:, 2:], x[:, D - 1:], x[:, D - 1:]], 1)
        x_diff = self.stem12(torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 2).view(N * D, 12, H, W))
        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)
        x_path1 = self.stem21(self.apha * x + self.belta * x_diff)
        x_path2 = self.stem22(x_diff)
        return self.apha * x_path1 + self.belta * x_path2


class RhythmMambaStem(nn.Module):
    """Fusion_Stem at narrow width (C/narrow_ratio), then a gradual widen:
    a (2,5,5) spatiotemporal conv to `gradual_mid_width` -- which also halves
    T -- followed by a pointwise 1x1 conv to full width. Widening in two
    cheap steps instead of one wide conv roughly halves this stem's cost;
    mid-width 72 (rather than C/2) was selected because the narrower
    bottleneck showed markedly higher seed-to-seed variance."""

    def __init__(self, out_channels: int = 96, narrow_ratio: int = 4, gradual_mid_width: int = 72):
        super().__init__()
        narrow = max(1, out_channels // narrow_ratio)
        self.fusion_stem = Fusion_Stem(dim=narrow)
        self.stem3 = nn.Sequential(
            nn.Conv3d(narrow, gradual_mid_width, kernel_size=(2, 5, 5), stride=(2, 1, 1), padding=(0, 2, 2)),
            nn.BatchNorm3d(gradual_mid_width),
            nn.ReLU(inplace=True),
            nn.Conv3d(gradual_mid_width, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels))
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,T,C,H,W]
        B, T, C, H, W = x.shape
        feat = self.fusion_stem(x)
        _, Cn, Hs, Ws = feat.shape
        feat = feat.view(B, T, Cn, Hs, Ws).permute(0, 2, 1, 3, 4).contiguous()
        return self.stem3(feat)  # [B,out_channels,T//2,Hs,Ws]


# ---------------------------------------------------------------------------
# Attention / face mask
# ---------------------------------------------------------------------------

class Attention_mask(nn.Module):
    """RhythmMamba's per-frame spatial renormalisation of an attention map."""

    def forward(self, x):
        xsum = torch.sum(x, dim=3, keepdim=True)
        xsum = torch.sum(xsum, dim=4, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[3] * xshape[4] * 0.5


class FaceMaskHead(nn.Module):
    """Bimodal face/non-face mask from the channel-mean logit map: spatial
    z-score (so it cannot collapse to a constant) then a learnable, capped
    sharpening temperature."""

    def __init__(self, init_temperature: float = 2.0, max_temperature: float = 2.5):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(init_temperature))))
        self.max_temperature = max_temperature

    def forward(self, logit_map: torch.Tensor) -> torch.Tensor:  # [B,T,Hs,Ws]
        mean = logit_map.mean(dim=(-1, -2), keepdim=True)
        std = logit_map.std(dim=(-1, -2), keepdim=True, unbiased=False) + EPS
        temperature = torch.exp(self.log_temperature).clamp(max=self.max_temperature)
        return torch.sigmoid(temperature * (logit_map - mean) / std)


def mask_entropy(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean binary entropy of a [0,1] mask; low = confidently bimodal."""
    p = A.clamp(eps, 1.0 - eps)
    return (-(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))).mean()


# ---------------------------------------------------------------------------
# Spectral purifier
# ---------------------------------------------------------------------------

def soft_band_mask(freqs: torch.Tensor, trans_low=(0.50, 0.65), trans_high=(3.0, 3.30)) -> torch.Tensor:
    """Cosine-transition band-pass mask over `freqs` (Hz), values in [0,1].
    Full pass-band is [trans_low[1], trans_high[0]]; both edges ramp smoothly
    rather than being rectangular. Fixed and parameter-free."""
    lo0, lo1 = trans_low
    hi0, hi1 = trans_high

    def cosine_ramp(x, a, b):
        t = ((x - a) / max(b - a, 1e-8)).clamp(0.0, 1.0)
        return 0.5 - 0.5 * torch.cos(math.pi * t)

    return (cosine_ramp(freqs, lo0, lo1) * (1.0 - cosine_ramp(freqs, hi0, hi1))).clamp(0.0, 1.0)


class SpectralChannelPurifier(nn.Module):
    """Per-channel DC removal, RMS normalisation, and a fixed cosine band-pass
    in the physiological range. Carries NO learnable parameters."""

    def __init__(self, trans_low=(0.50, 0.65), trans_high=(3.0, 3.30)):
        super().__init__()
        self.trans_low = trans_low
        self.trans_high = trans_high

    def forward(self, Z_raw: torch.Tensor, fs: float) -> torch.Tensor:  # [B,C,T,Hs,Ws]
        T = Z_raw.shape[2]
        Z_centered = Z_raw - Z_raw.mean(dim=2, keepdim=True)
        rms = torch.sqrt((Z_centered ** 2).mean(dim=2, keepdim=True) + EPS).clamp(min=1e-4)
        Z_norm = Z_centered / rms
        with torch.autocast(device_type=Z_raw.device.type, enabled=False):
            Zf = torch.fft.rfft(Z_norm.float(), dim=2)
            freqs = torch.fft.rfftfreq(T, d=1.0 / fs, device=Z_raw.device)
            mask_c = soft_band_mask(freqs, self.trans_low, self.trans_high).view(1, 1, -1, 1, 1)
            return torch.fft.irfft(Zf * mask_c, n=T, dim=2).to(Z_raw.dtype)


# ---------------------------------------------------------------------------
# Temporal core
# ---------------------------------------------------------------------------

class ClassicMambaNet(nn.Module):
    """`depth` unidirectional Mamba blocks, each pre-normed and wrapped in a
    residual gated-conv MLP, followed by a pointwise head to the BVP trace."""

    def __init__(self, C=96, depth=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "norm_f": nn.LayerNorm(C),
                "mamba_f": Mamba(d_model=C, d_state=d_state, d_conv=d_conv, expand=expand),
                "mlp": nn.Sequential(nn.Conv1d(C, C * 2, 1), nn.GLU(1), nn.Conv1d(C, C, 1)),
            }))
        self.head = nn.Conv1d(C, 1, 1)

    def forward(self, X, return_hidden: bool = False):  # X: [B,C,T]
        h = X
        for blk in self.blocks:
            u = h.transpose(1, 2)                       # [B,T,C]
            y = blk["mamba_f"](blk["norm_f"](u)).transpose(1, 2)
            h = h + blk["mlp"](y)
        bvp = self.head(h).squeeze(1)
        return (bvp, h) if return_hidden else bvp


class TemporalEncoder(nn.Module):
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, depth: int = 2):
        super().__init__()
        self.classicmamba_net = ClassicMambaNet(C=channels, depth=depth, d_state=d_state,
                                                 d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor):
        return self.classicmamba_net(x, return_hidden=True)


# ---------------------------------------------------------------------------
# PhysBandMamba
# ---------------------------------------------------------------------------

class PhysBandMamba(nn.Module):
    """Defaults below ARE the MidWidth-72 configuration; constructing
    `PhysBandMamba()` with no arguments reproduces the shipped checkpoints."""

    def __init__(self,
                 c: int = 96,
                 band_trans_low=(0.50, 0.65),
                 band_trans_high=(3.0, 3.30),
                 mask_temperature_init: float = 2.0,
                 mask_temperature_max: float = 2.5,
                 stem_narrow_ratio: int = 4,
                 gradual_mid_width: int = 72,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 depth: int = 2):
        super().__init__()
        self.c = c
        self.stem = RhythmMambaStem(out_channels=c, narrow_ratio=stem_narrow_ratio,
                                     gradual_mid_width=gradual_mid_width)
        self.stem3_upsample = nn.Upsample(scale_factor=2)
        self.attn_mask_module = Attention_mask()
        self.face_mask_head = FaceMaskHead(init_temperature=mask_temperature_init,
                                            max_temperature=mask_temperature_max)
        self.purifier = SpectralChannelPurifier(trans_low=band_trans_low, trans_high=band_trans_high)
        self.temporal_encoder = TemporalEncoder(c, d_state=d_state, d_conv=d_conv,
                                                 expand=expand, depth=depth)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, fs: float = 30.0, epoch: Optional[int] = None,
                labels: Optional[torch.Tensor] = None, return_aux: bool = True) -> Dict:
        B, T, C, H, W = x.shape

        Z_raw = self.stem(x)                                  # [B,c,T/2,Hs,Ws]
        Tt = Z_raw.shape[2]
        t_ratio = T // Tt if Tt > 0 else 1
        fs_internal = fs / t_ratio                            # stem halved T, so halve fs

        # Attention. A_face01 is built in LOGIT space (channel-averaging
        # sigmoid outputs stays flat and never separates face from non-face);
        # A_raw is RhythmMamba's own per-channel renormalised mask.
        A_face01 = self.face_mask_head(Z_raw.mean(dim=1))     # [B,T/2,Hs,Ws]
        A_raw = self.attn_mask_module(torch.sigmoid(Z_raw))
        loss_mask_entropy = mask_entropy(A_face01)

        Z_phys = self.purifier(Z_raw, fs_internal)
        F_branch = (Z_phys * A_raw).mean(dim=(-1, -2))        # [B,c,T/2]

        bvp, h_temporal = self.temporal_encoder(F_branch)
        bvp = self.stem3_upsample(bvp.unsqueeze(1)).squeeze(1)  # restore T

        out = {'bvp': bvp, 'h_temporal': h_temporal,
               'A_face01': A_face01, 'loss_mask_entropy': loss_mask_entropy}
        if return_aux:
            out['diagnostics'] = {
                'A_raw_min': A_raw.min().detach(), 'A_raw_max': A_raw.max().detach(),
                'A_raw_mean': A_raw.mean().detach(), 'A_raw_std': A_raw.std(unbiased=False).detach(),
                'A_face01_min': A_face01.min().detach(), 'A_face01_max': A_face01.max().detach(),
                'A_face01_frac_above_0.5': (A_face01 > 0.5).float().mean().detach(),
                'mask_temperature': torch.exp(self.face_mask_head.log_temperature).detach(),
                'A_face_mass': A_face01.mean(dim=(-1, -2)).mean().detach(),
                'F_norm': F_branch.float().pow(2).mean().sqrt().detach(),
            }
        return out


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
