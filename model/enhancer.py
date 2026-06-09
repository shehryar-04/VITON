"""
Real-ESRGAN Image Enhancer
==========================
Self-contained Real-ESRGAN implementation for enhancing virtual try-on outputs,
specifically to preserve and sharpen cloth textures, folds, and fine details.

Why self-contained?
-------------------
The official `realesrgan` / `basicsr` pip packages import
`torchvision.transforms.functional_tensor`, which was REMOVED in torchvision
>= 0.17. That breaks installation on modern environments. This module
re-implements the RRDBNet generator architecture directly and loads the
official pretrained weights, avoiding the broken dependency chain entirely.

Models supported
----------------
- RealESRGAN_x4plus      : general 4x upscaler (23 RRDB blocks)
- RealESRGAN_x2plus      : general 2x upscaler (23 RRDB blocks)

Usage
-----
    from model.enhancer import RealESRGANEnhancer

    enhancer = RealESRGANEnhancer(scale=4, device="cuda")
    enhanced_pil = enhancer.enhance(result_pil)              # whole image
    enhanced_pil = enhancer.enhance(result_pil, outscale=2)  # custom output scale

    # Texture-focused: enhance only the garment region (via mask)
    enhanced_pil = enhancer.enhance_region(result_pil, mask_pil)
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Official pretrained weight URLs (Real-ESRGAN releases)
# ---------------------------------------------------------------------------
_WEIGHT_URLS = {
    4: (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.0/RealESRGAN_x4plus.pth"
    ),
    2: (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.2.1/RealESRGAN_x2plus.pth"
    ),
}


# ===========================================================================
# RRDBNet architecture  (matches the official Real-ESRGAN generator)
# ===========================================================================
class ResidualDenseBlock(nn.Module):
    """Residual Dense Block with 5 conv layers (growth channels = gc)."""

    def __init__(self, num_feat: int = 64, gc: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * gc, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically scaled residual (0.2) — matches official implementation
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat: int, gc: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, gc)
        self.rdb2 = ResidualDenseBlock(num_feat, gc)
        self.rdb3 = ResidualDenseBlock(num_feat, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    RRDBNet generator used by Real-ESRGAN.

    Args:
        num_in_ch:  input channels (3 for RGB)
        num_out_ch: output channels (3 for RGB)
        num_feat:   base feature channels (64)
        num_block:  number of RRDB blocks (23 for x4plus/x2plus)
        gc:         growth channels in each dense block (32)
        scale:      upscaling factor (2 or 4)
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        gc: int = 32,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.scale = scale

        # For x2, the input is pixel-unshuffled by 2 first (official trick),
        # so the first conv sees 4x the channels.
        first_in = num_in_ch
        if scale == 2:
            first_in = num_in_ch * 4
        elif scale == 1:
            first_in = num_in_ch * 16

        self.conv_first = nn.Conv2d(first_in, num_feat, 3, 1, 1)
        self.body = nn.ModuleList([RRDB(num_feat, gc) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling layers (one per 2x; x4 = two upsample steps)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _pixel_unshuffle(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        b, c, h, w = x.shape
        out_c = c * (scale ** 2)
        h2, w2 = h // scale, w // scale
        x = x.view(b, c, h2, scale, w2, scale)
        x = x.permute(0, 1, 3, 5, 2, 4).reshape(b, out_c, h2, w2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Official Real-ESRGAN trick: for scale 2/1 the input is pixel-unshuffled
        # first, then TWO 2x upsamples are ALWAYS applied. Net scale =
        #   x4: no unshuffle, *4  → 4x
        #   x2: unshuffle /2, *4  → 2x
        #   x1: unshuffle /4, *4  → 1x
        if self.scale == 2:
            feat = self._pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = self._pixel_unshuffle(x, 4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat

        # Two nearest-neighbour 2x upsample steps (always applied)
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ===========================================================================
# High-level enhancer wrapper
# ===========================================================================
class RealESRGANEnhancer:
    """
    Real-ESRGAN wrapper for enhancing try-on outputs with tiled inference
    (to bound VRAM) and optional garment-region-only enhancement.
    """

    def __init__(
        self,
        scale: int = 4,
        device: str = "cuda",
        weight_path: Optional[str] = None,
        half: bool = True,
        tile: int = 512,
        tile_pad: int = 32,
    ) -> None:
        """
        Args:
            scale:       model upscale factor (2 or 4)
            device:      torch device
            weight_path: local .pth path; if None, downloads official weights
            half:        use fp16 inference (faster, less VRAM)
            tile:        tile size in px for tiled inference (0 = no tiling)
            tile_pad:    padding between tiles to avoid seams
        """
        if scale not in (2, 4):
            raise ValueError(f"scale must be 2 or 4, got {scale}")

        self.scale = scale
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half = half and self.device.type == "cuda"
        self.tile = tile
        self.tile_pad = tile_pad

        # Build model
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, gc=32, scale=scale,
        )

        # Load weights
        state = self._load_weights(weight_path)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)
        if self.half:
            self.model = self.model.half()

    # ------------------------------------------------------------------
    def _load_weights(self, weight_path: Optional[str]) -> dict:
        if weight_path and os.path.exists(weight_path):
            ckpt = torch.load(weight_path, map_location="cpu")
        else:
            url = _WEIGHT_URLS[self.scale]
            ckpt = torch.hub.load_state_dict_from_url(
                url, map_location="cpu", progress=True
            )
        # Official checkpoints store weights under 'params_ema' or 'params'
        if "params_ema" in ckpt:
            return ckpt["params_ema"]
        if "params" in ckpt:
            return ckpt["params"]
        return ckpt

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _infer(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Run the model with optional tiling. img_tensor: (1, 3, H, W) in [0, 1]."""
        if self.tile <= 0:
            return self.model(img_tensor)
        return self._tiled_infer(img_tensor)

    @torch.no_grad()
    def _tiled_infer(self, img: torch.Tensor) -> torch.Tensor:
        """Process the image tile-by-tile to bound VRAM usage."""
        b, c, h, w = img.shape
        out_h, out_w = h * self.scale, w * self.scale
        output = img.new_zeros((b, c, out_h, out_w))

        tiles_x = (w + self.tile - 1) // self.tile
        tiles_y = (h + self.tile - 1) // self.tile

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Input tile bounds
                x0 = tx * self.tile
                y0 = ty * self.tile
                x1 = min(x0 + self.tile, w)
                y1 = min(y0 + self.tile, h)

                # Padded bounds
                px0 = max(x0 - self.tile_pad, 0)
                py0 = max(y0 - self.tile_pad, 0)
                px1 = min(x1 + self.tile_pad, w)
                py1 = min(y1 + self.tile_pad, h)

                tile_in = img[:, :, py0:py1, px0:px1]
                tile_out = self.model(tile_in)

                # Map the unpadded region from the upscaled tile to output
                ox0 = x0 * self.scale
                oy0 = y0 * self.scale
                ox1 = x1 * self.scale
                oy1 = y1 * self.scale

                in_x0 = (x0 - px0) * self.scale
                in_y0 = (y0 - py0) * self.scale
                in_x1 = in_x0 + (x1 - x0) * self.scale
                in_y1 = in_y0 + (y1 - y0) * self.scale

                output[:, :, oy0:oy1, ox0:ox1] = tile_out[:, :, in_y0:in_y1, in_x0:in_x1]

        return output

    # ------------------------------------------------------------------
    def enhance(
        self,
        image: Image.Image,
        outscale: Optional[float] = None,
    ) -> Image.Image:
        """
        Enhance a full image.

        Args:
            image:    input PIL RGB image
            outscale: final output scale relative to input (default = model scale).
                      e.g. with a x4 model, outscale=2 downsamples the 4x result to 2x.

        Returns:
            Enhanced PIL RGB image.
        """
        img = image.convert("RGB")
        in_w, in_h = img.size

        arr = np.asarray(img).astype(np.float32) / 255.0       # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        tensor = tensor.to(self.device)
        if self.half:
            tensor = tensor.half()

        out = self._infer(tensor)
        out = out.clamp(0, 1).float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = (out * 255.0).round().astype(np.uint8)
        result = Image.fromarray(out)

        # Resize to requested output scale
        if outscale is not None and outscale != self.scale:
            target = (int(in_w * outscale), int(in_h * outscale))
            result = result.resize(target, Image.LANCZOS)

        return result

    # ------------------------------------------------------------------
    def enhance_region(
        self,
        image: Image.Image,
        mask: Image.Image,
        outscale: Optional[float] = None,
        feather: int = 8,
    ) -> Image.Image:
        """
        Enhance only the masked region (e.g. the garment), then composite back
        onto the original at the same resolution. This focuses texture/fold
        sharpening on the clothing without altering the face/background.

        Args:
            image:    input PIL RGB image (the try-on result)
            mask:     PIL L-mode mask, white (255) = region to enhance
            outscale: output scale relative to input (default 1.0 = same size)
            feather:  Gaussian feather radius (px) for a seamless composite

        Returns:
            Composited PIL RGB image at input resolution * outscale.
        """
        from PIL import ImageFilter

        outscale = outscale if outscale is not None else 1.0
        in_w, in_h = image.size
        target = (int(in_w * outscale), int(in_h * outscale))

        # Enhance the whole image then downscale to target (keeps detail)
        enhanced = self.enhance(image, outscale=outscale)

        # Base = original resized to target (LANCZOS keeps it clean)
        base = image.convert("RGB").resize(target, Image.LANCZOS)

        # Prepare mask at target resolution, feathered for a soft blend
        m = mask.convert("L").resize(target, Image.NEAREST)
        if feather > 0:
            m = m.filter(ImageFilter.GaussianBlur(feather))

        # Composite: enhanced inside mask, base outside
        result = Image.composite(enhanced, base, m)
        return result


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def load_enhancer(
    scale: int = 4,
    device: str = "cuda",
    weight_path: Optional[str] = None,
    tile: int = 512,
) -> RealESRGANEnhancer:
    """Build a RealESRGANEnhancer with sensible defaults for try-on enhancement."""
    return RealESRGANEnhancer(
        scale=scale,
        device=device,
        weight_path=weight_path,
        half=True,
        tile=tile,
        tile_pad=32,
    )
