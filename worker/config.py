"""
Configuration loader for the GPU worker.

Reads all tunable parameters from environment variables.
Exits with code 1 if any required variable is missing.
"""
from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def _get_int(name: str, default: str) -> int:
    return int(os.environ.get(name, default))


def _get_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Required variables — exit with code 1 if any are absent
# ---------------------------------------------------------------------------

_REQUIRED_VARS = [
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "CLOUDINARY_CLOUD_NAME",
    "CLOUDINARY_API_KEY",
    "CLOUDINARY_API_SECRET",
]


def _validate_required() -> None:
    missing = [v for v in _REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        for var in missing:
            logger.error("Missing required environment variable: %s", var)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Config values (populated at import time)
# ---------------------------------------------------------------------------

_validate_required()

# Redis
REDIS_URL: str = _get_str("REDIS_URL", "redis://localhost:6379/0")

# Batching
BATCH_SIZE: int = _get_int("BATCH_SIZE", "4")
BATCH_TIMEOUT_MS: int = _get_int("BATCH_TIMEOUT_MS", "100")

# Cache TTLs
PREPROCESS_CACHE_TTL: int = _get_int("PREPROCESS_CACHE_TTL", "86400")
RESULT_CACHE_TTL: int = _get_int("RESULT_CACHE_TTL", "604800")

# Priority / anti-starvation
PRIORITY_WAIT_THRESHOLD: int = _get_int("PRIORITY_WAIT_THRESHOLD", "300")

# Inference optimisations
FLASH_ATTENTION: bool = _get_bool("FLASH_ATTENTION", "true")
VAE_TILING: bool = _get_bool("VAE_TILING", "true")
VAE_SLICING: bool = _get_bool("VAE_SLICING", "true")
VAE_TILING_RESOLUTION: int = _get_int("VAE_TILING_RESOLUTION", "1024")

# Similarity cache
SIMILARITY_METHOD: str = _get_str("SIMILARITY_METHOD", "phash")
SIMILARITY_THRESHOLD: str = _get_str("SIMILARITY_THRESHOLD", "8")
CLIP_MODEL_PATH: str = _get_str("CLIP_MODEL_PATH", "")

# Pipeline / device
PIPELINE_TYPE: str = _get_str("PIPELINE_TYPE", "catvton")
GPU_DEVICE: str = _get_str("GPU_DEVICE", "cuda:0")
WORKER_COUNT: int = _get_int("WORKER_COUNT", "1")

# CatVTON sampling controls (garment texture fidelity)
NUM_INFERENCE_STEPS: int = _get_int("NUM_INFERENCE_STEPS", "50")
GUIDANCE_SCALE: float = float(_get_str("GUIDANCE_SCALE", "2.5"))
# Legacy CLIP image cross-attention. Default OFF — the official CatVTON design
# bypasses cross-attention and transfers garment texture via self-attention.
# Enabling this feeds stock SD1.5 text-trained cross-attention an OOD CLIP token
# and tends to flatten/smear garment texture.
USE_CLIP_CROSS_ATTN: bool = _get_bool("USE_CLIP_CROSS_ATTN", "false")

# Supabase (required — already validated above)
SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_ANON_KEY: str = os.environ["SUPABASE_ANON_KEY"]

# Cloudinary (required — already validated above)
CLOUDINARY_CLOUD_NAME: str = os.environ["CLOUDINARY_CLOUD_NAME"]
CLOUDINARY_API_KEY: str = os.environ["CLOUDINARY_API_KEY"]
CLOUDINARY_API_SECRET: str = os.environ["CLOUDINARY_API_SECRET"]

# CatVTON checkpoint paths (optional — empty string if not set)
BASE_CKPT: str = _get_str("BASE_CKPT", "")
ATTN_CKPT: str = _get_str("ATTN_CKPT", "")
ATTN_CKPT_VERSION: str = _get_str("ATTN_CKPT_VERSION", "mix")

# Flux checkpoint path (optional — empty string if not set)
FLUX_CKPT: str = _get_str("FLUX_CKPT", "")

# Flux try-on pipeline (FLUX.1-Fill-dev based). Requires a gated HF token for the
# base model. NF4 4-bit quantization is required to fit a 12B model under 12GB VRAM.
FLUX_BASE_CKPT: str = _get_str("FLUX_BASE_CKPT", "black-forest-labs/FLUX.1-Fill-dev")
# Fine-tuned try-on transformer. The base Fill model is a GENERIC inpainter and
# will not perform try-on (it ignores the garment reference). Set this to a
# CatVTON-Flux transformer, e.g. "xiaozaa/catvton-flux-alpha" (CC-BY-NC, dev only).
FLUX_TRANSFORMER_CKPT: str = _get_str("FLUX_TRANSFORMER_CKPT", "")
FLUX_TRANSFORMER_SUBFOLDER: str = _get_str("FLUX_TRANSFORMER_SUBFOLDER", "")
FLUX_LORA_PATH: str = _get_str("FLUX_LORA_PATH", "")
FLUX_QUANTIZE_4BIT: bool = _get_bool("FLUX_QUANTIZE_4BIT", "true")
# Run the VAE in fp32. The Flux VAE is prone to NaN/black images in fp16 (the
# T4 compute dtype), so default to fp32 — it is small (~0.7GB) and fits easily.
FLUX_VAE_FP32: bool = _get_bool("FLUX_VAE_FP32", "true")
# Offload strategy: "model" (per-module, fits ~12GB), "sequential" (per-submodule,
# fits <8GB but very slow), or "none" (everything on GPU — needs ample VRAM).
FLUX_CPU_OFFLOAD: str = _get_str("FLUX_CPU_OFFLOAD", "model")
FLUX_HEIGHT: int = _get_int("FLUX_HEIGHT", "1024")
FLUX_WIDTH: int = _get_int("FLUX_WIDTH", "768")
FLUX_NUM_INFERENCE_STEPS: int = _get_int("FLUX_NUM_INFERENCE_STEPS", "30")
FLUX_GUIDANCE_SCALE: float = float(_get_str("FLUX_GUIDANCE_SCALE", "30.0"))

# AutoMasker checkpoint paths (optional — empty string if not set)
DENSEPOSE_CKPT: str = _get_str("DENSEPOSE_CKPT", "")
SCHP_CKPT: str = _get_str("SCHP_CKPT", "")

# Real-ESRGAN enhancement (texture / fold sharpening)
ENHANCE: bool = _get_bool("ENHANCE", "false")
ENHANCE_SCALE: int = _get_int("ENHANCE_SCALE", "4")
ENHANCE_OUTSCALE: float = float(_get_str("ENHANCE_OUTSCALE", "1.0"))
ENHANCE_REGION_ONLY: bool = _get_bool("ENHANCE_REGION_ONLY", "true")
ENHANCE_WEIGHT_PATH: str = _get_str("ENHANCE_WEIGHT_PATH", "")
ENHANCE_TILE: int = _get_int("ENHANCE_TILE", "512")
