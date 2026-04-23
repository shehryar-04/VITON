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
