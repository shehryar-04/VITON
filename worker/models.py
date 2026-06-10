from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class JobRecord:
    id: str                          # UUID
    user_image_url: str
    cloth_image_url: str
    cloth_type: str                  # "upper" | "lower" | "overall" | "inner" | "outer"
    status: str                      # "pending" | "processing" | "completed" | "failed"
    priority: int                    # 1 (premium) | 10 (standard)
    user_tier: str                   # "premium" | "standard"
    webhook_url: Optional[str]
    result_image_url: Optional[str]
    error: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    enqueued_at: datetime            # used for anti-starvation age calculation


@dataclass
class PreprocessResult:
    densepose_png: bytes    # lossless PNG bytes of DensePose output
    schp_atr_png: bytes     # lossless PNG bytes of SCHP ATR output
    schp_lip_png: bytes     # lossless PNG bytes of SCHP LIP output


@dataclass
class InferenceResult:
    job_id: str
    image: Optional[Any]    # PIL.Image.Image
    error: Optional[str]


@dataclass
class InferenceConfig:
    pipeline_type: str          # "catvton" | "flux"
    device: str                 # "cuda:0", "cuda:1", ...
    batch_size: int             # default 4
    flash_attention: bool       # default True
    vae_tiling: bool            # default True
    vae_slicing: bool           # default True
    vae_tiling_resolution: int  # default 1024 — enable tiling above this px

    # --- CatVTON sampling controls (garment texture fidelity) ---
    num_inference_steps: int = 50          # DDIM steps; higher = finer texture reconstruction
    guidance_scale: float = 2.5            # CFG scale; CatVTON works best around 2.5
    use_clip_cross_attn: bool = False      # legacy CLIP image cross-attention (off = official CatVTON)

    # --- Real-ESRGAN post-processing (texture / fold enhancement) ---
    enhance: bool = False                  # enable Real-ESRGAN enhancement
    enhance_scale: int = 4                 # model upscale factor (2 or 4)
    enhance_outscale: float = 1.0          # final output scale vs. try-on result
    enhance_region_only: bool = True       # enhance only the garment (masked) region
    enhance_weight_path: str = ""          # local .pth path; empty = download official
    enhance_tile: int = 512                # tile size for tiled inference (0 = off)
