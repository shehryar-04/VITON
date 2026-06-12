"""Preprocessing cache for DensePose and SCHP outputs.

Caches expensive preprocessing results (DensePose, SCHP ATR, SCHP LIP) keyed
by person image content hash, so repeated mask requests for the same person
image skip recomputation.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from PIL import Image

from worker.models import PreprocessResult

logger = logging.getLogger(__name__)


class PreprocessingCache:
    """
    Cache for DensePose and SCHP outputs keyed by person image content.

    Storage format: lossless PNG bytes for each of the three outputs.
    Cache key: SHA-256 of the person image's raw RGB pixel bytes.
    """

    def __init__(self) -> None:
        self._store: dict[str, PreprocessResult] = {}

    @staticmethod
    def compute_key(image: Image.Image) -> str:
        """Compute a deterministic cache key from image pixel content.

        Converts the image to RGB mode and hashes the raw pixel bytes with
        SHA-256. Two images with identical RGB pixel data will always produce
        the same key regardless of creation path, metadata, or mode.

        Args:
            image: Any PIL Image.

        Returns:
            Hex-encoded SHA-256 digest string.
        """
        rgb = image.convert("RGB")
        return hashlib.sha256(rgb.tobytes()).hexdigest()

    def get(self, image_key: str) -> Optional[PreprocessResult]:
        """Retrieve cached preprocessing outputs.

        Returns None on cache miss, partial entry (any field empty/missing),
        or if the entry is corrupt/unreadable.

        Args:
            image_key: SHA-256 hex string produced by compute_key.

        Returns:
            PreprocessResult with all three PNG byte fields populated,
            or None if entry is missing, incomplete, or corrupt.
        """
        try:
            result = self._store.get(image_key)
            if result is None:
                return None

            # Validate that all three fields are non-empty bytes (Requirement 7.4)
            if (
                not isinstance(result.densepose_png, bytes)
                or not result.densepose_png
            ):
                logger.warning(
                    "Cache entry for key %s has missing/empty densepose_png, "
                    "treating as miss.",
                    image_key[:16],
                )
                return None

            if (
                not isinstance(result.schp_atr_png, bytes)
                or not result.schp_atr_png
            ):
                logger.warning(
                    "Cache entry for key %s has missing/empty schp_atr_png, "
                    "treating as miss.",
                    image_key[:16],
                )
                return None

            if (
                not isinstance(result.schp_lip_png, bytes)
                or not result.schp_lip_png
            ):
                logger.warning(
                    "Cache entry for key %s has missing/empty schp_lip_png, "
                    "treating as miss.",
                    image_key[:16],
                )
                return None

            return result

        except Exception:
            # Handle corrupt/unreadable entries gracefully (Requirement 7.5)
            logger.warning(
                "Failed to read cache entry for key %s, treating as miss.",
                image_key[:16],
                exc_info=True,
            )
            return None

    def put(self, image_key: str, result: PreprocessResult) -> None:
        """Store preprocessing outputs in the cache.

        Stores the full PreprocessResult containing all three lossless PNG
        byte fields.

        Args:
            image_key: SHA-256 hex string produced by compute_key.
            result: PreprocessResult with densepose_png, schp_atr_png,
                    and schp_lip_png all populated.
        """
        self._store[image_key] = result
