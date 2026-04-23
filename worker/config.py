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

# AutoMasker checkpoint paths (optional — empty string if not set)
DENSEPOSE_CKPT: str = _get_str("DENSEPOSE_CKPT", "")
SCHP_CKPT: str = _get_str("SCHP_CKPT", "")
