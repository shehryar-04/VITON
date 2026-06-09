"""
IUV Encoder
===========
Lightweight convolutional encoder that maps a 26-channel DensePose IUV
tensor to a feature tensor at latent resolution, ready to be concatenated
to the UNet input channels.

Input format (26 channels)
--------------------------
Channels  0..23 : one-hot encoding of body-part index I (24 classes, 0=background)
Channel   24    : U surface coordinate in [0, 1]
Channel   25    : V surface coordinate in [0, 1]

This replaces the previous 3-channel format where I was a single normalised
float.  One-hot encoding gives the model a clean, unambiguous signal for
each body part without imposing a false ordinal relationship between parts.

Architecture
------------
Input : (B, 26, H, W)
Output: (B, out_channels, H/8, W/8)  — matches VAE latent spatial size

Three strided conv layers (stride-2 each) downsample by 8×, matching the
VAE's spatial compression factor.  GroupNorm + SiLU throughout.

Usage
-----
    from model.iuv_encoder import IUVEncoder, IUV_LATENT_CHANNELS, IUV_IN_CHANNELS
    import torch

    encoder = IUVEncoder().to(device)
    iuv_tensor = torch.rand(1, IUV_IN_CHANNELS, 1024, 768)  # (B, 26, H, W)
    latent_feat = encoder(iuv_tensor)                         # (B, 8, 128, 96)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Input channels to IUVEncoder: 24 one-hot I classes + 1 U + 1 V = 26
IUV_IN_CHANNELS: int = 26

# Number of output channels from the IUV encoder.
# Must match the value in model/pipeline.py (IUV_LATENT_CHANNELS = 8).
IUV_LATENT_CHANNELS: int = 8


class IUVEncoder(nn.Module):
    """
    Maps a (B, 26, H, W) one-hot IUV tensor to (B, out_channels, H/8, W/8).

    Input channels (26):
        0..23  — one-hot body-part index (24 classes, produced by
                 torch.nn.functional.one_hot on the raw DensePose I map)
        24     — U surface coordinate in [0, 1]
        25     — V surface coordinate in [0, 1]

    Args:
        in_channels  (int): Input channels. Default 26 (IUV_IN_CHANNELS).
        mid_channels (int): Hidden feature width. Default 32.
        out_channels (int): Output channels to concatenate to UNet input.
                            Default 8 (IUV_LATENT_CHANNELS).
    """

    def __init__(
        self,
        in_channels: int = IUV_IN_CHANNELS,   # 26
        mid_channels: int = 32,
        out_channels: int = IUV_LATENT_CHANNELS,  # 4
    ) -> None:
        super().__init__()

        # 3 × stride-2 conv → 8× spatial downsampling
        self.encoder = nn.Sequential(
            # Block 1: (B, 26, H, W) → (B, mid, H/2, W/2)
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=mid_channels),
            nn.SiLU(inplace=True),

            # Block 2: (B, mid, H/2, W/2) → (B, mid*2, H/4, W/4)
            nn.Conv2d(mid_channels, mid_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=mid_channels * 2),
            nn.SiLU(inplace=True),

            # Block 3: (B, mid*2, H/4, W/4) → (B, out_channels, H/8, W/8)
            nn.Conv2d(mid_channels * 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, iuv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            iuv: (B, 26, H, W) float32 tensor.
                 Channels 0..23 = one-hot body-part index.
                 Channel  24    = U coordinate in [0, 1].
                 Channel  25    = V coordinate in [0, 1].

        Returns:
            (B, out_channels, H/8, W/8) float32 tensor.
        """
        return self.encoder(iuv)


def prepare_iuv_latent(
    iuv_numpy,          # np.ndarray (H, W, 26) float32
    target_h: int,
    target_w: int,
    encoder: IUVEncoder,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convenience function: numpy IUV26 → resized tensor → encoded latent.

    Args:
        iuv_numpy  : (H, W, 26) float32 array from DensePose.call_iuv().
                     Channels 0..23 = one-hot I, channel 24 = U, channel 25 = V.
        target_h   : Target image height (before VAE encoding), e.g. 1024.
        target_w   : Target image width,  e.g. 768.
        encoder    : IUVEncoder instance.
        device     : torch device.
        dtype      : torch dtype (fp16 / fp32).

    Returns:
        Tensor (1, out_channels, target_h/8, target_w/8).
    """
    import cv2
    import numpy as np

    n_ch = iuv_numpy.shape[2]  # 26

    # Resize each channel to target resolution.
    # One-hot channels (0..23): nearest-neighbour to keep binary values exact.
    # U, V channels (24, 25): bilinear for smooth interpolation.
    if iuv_numpy.shape[:2] != (target_h, target_w):
        iuv_resized = np.stack([
            cv2.resize(
                iuv_numpy[:, :, c],
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST if c < 24 else cv2.INTER_LINEAR,
            )
            for c in range(n_ch)   # range(26) — no loops over pixels, only over channels
        ], axis=2)
    else:
        iuv_resized = iuv_numpy

    # (H, W, 26) → (1, 26, H, W)
    iuv_tensor = torch.from_numpy(iuv_resized).permute(2, 0, 1).unsqueeze(0)
    iuv_tensor = iuv_tensor.to(device=device, dtype=dtype)

    with torch.no_grad():
        latent = encoder(iuv_tensor)

    return latent
