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

    # --- Flux try-on pipeline (FLUX.1-Fill-dev based) ---
    flux_base_ckpt: str = "black-forest-labs/FLUX.1-Fill-dev"  # gated HF repo (needs token); provides VAE + scheduler
    flux_transformer_ckpt: str = ""        # fine-tuned try-on transformer (e.g. xiaozaa/catvton-flux-alpha); empty = base (generic fill, NOT try-on)
    flux_transformer_subfolder: str = ""   # subfolder of the transformer repo ("" = root, as in xiaozaa; base uses "transformer")
    flux_lora_path: str = ""               # optional try-on LoRA (alternative to a fine-tuned transformer)
    flux_quantize_4bit: bool = True        # NF4 4-bit transformer — required for <12GB VRAM
    flux_vae_fp32: bool = True             # run VAE in fp32 (T4/fp16 safety against black images)
    flux_cpu_offload: str = "model"        # "model" | "sequential" | "none"
    flux_height: int = 1024                # portrait try-on height (divisible by 16)
    flux_width: int = 768                  # portrait try-on width (divisible by 16)
    flux_num_inference_steps: int = 30     # Flux Fill steps
    flux_guidance_scale: float = 30.0      # Flux Fill distilled guidance (~30 for try-on)

    # --- Real-ESRGAN post-processing (texture / fold enhancement) ---
    enhance: bool = False                  # enable Real-ESRGAN enhancement
    enhance_scale: int = 4                 # model upscale factor (2 or 4)
    enhance_outscale: float = 1.0          # final output scale vs. try-on result
    enhance_region_only: bool = True       # enhance only the garment (masked) region
    enhance_weight_path: str = ""          # local .pth path; empty = download official
    enhance_tile: int = 512                # tile size for tiled inference (0 = off)
