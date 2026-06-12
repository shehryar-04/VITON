"""Unit tests for worker.preprocessing_cache.PreprocessingCache."""

import pytest
from PIL import Image

from worker.models import PreprocessResult
from worker.preprocessing_cache import PreprocessingCache


class TestComputeKey:
    """Tests for PreprocessingCache.compute_key."""

    def test_deterministic_for_same_content(self):
        """Two images with identical RGB pixel content produce identical keys."""
        img1 = Image.new("RGB", (50, 50), color=(100, 150, 200))
        img2 = Image.new("RGB", (50, 50), color=(100, 150, 200))
        assert PreprocessingCache.compute_key(img1) == PreprocessingCache.compute_key(img2)

    def test_different_content_produces_different_keys(self):
        """Different pixel content produces different keys."""
        img1 = Image.new("RGB", (50, 50), color=(100, 150, 200))
        img2 = Image.new("RGB", (50, 50), color=(200, 150, 100))
        assert PreprocessingCache.compute_key(img1) != PreprocessingCache.compute_key(img2)

    def test_rgba_converted_to_rgb(self):
        """RGBA image produces same key as equivalent RGB image."""
        img_rgb = Image.new("RGB", (30, 30), color=(10, 20, 30))
        img_rgba = img_rgb.convert("RGBA")
        assert PreprocessingCache.compute_key(img_rgba) == PreprocessingCache.compute_key(img_rgb)

    def test_l_mode_converted_to_rgb(self):
        """Grayscale (L) image produces same key as its RGB equivalent."""
        img_l = Image.new("L", (20, 20), color=128)
        img_rgb = img_l.convert("RGB")
        assert PreprocessingCache.compute_key(img_l) == PreprocessingCache.compute_key(img_rgb)

    def test_key_is_hex_string(self):
        """Key is a 64-character hex string (SHA-256)."""
        img = Image.new("RGB", (10, 10), color=(0, 0, 0))
        key = PreprocessingCache.compute_key(img)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


class TestGet:
    """Tests for PreprocessingCache.get."""

    def test_returns_none_for_missing_key(self):
        """Returns None when key does not exist in cache."""
        cache = PreprocessingCache()
        assert cache.get("nonexistent_key") is None

    def test_returns_result_for_complete_entry(self):
        """Returns PreprocessResult when all fields are valid non-empty bytes."""
        cache = PreprocessingCache()
        result = PreprocessResult(
            densepose_png=b"dp_data",
            schp_atr_png=b"atr_data",
            schp_lip_png=b"lip_data",
        )
        cache.put("test_key", result)
        retrieved = cache.get("test_key")
        assert retrieved is not None
        assert retrieved.densepose_png == b"dp_data"
        assert retrieved.schp_atr_png == b"atr_data"
        assert retrieved.schp_lip_png == b"lip_data"

    def test_partial_entry_empty_densepose(self):
        """Returns None when densepose_png is empty (partial entry)."""
        cache = PreprocessingCache()
        result = PreprocessResult(
            densepose_png=b"",
            schp_atr_png=b"atr_data",
            schp_lip_png=b"lip_data",
        )
        cache.put("partial_key", result)
        assert cache.get("partial_key") is None

    def test_partial_entry_empty_schp_atr(self):
        """Returns None when schp_atr_png is empty (partial entry)."""
        cache = PreprocessingCache()
        result = PreprocessResult(
            densepose_png=b"dp_data",
            schp_atr_png=b"",
            schp_lip_png=b"lip_data",
        )
        cache.put("partial_key", result)
        assert cache.get("partial_key") is None

    def test_partial_entry_empty_schp_lip(self):
        """Returns None when schp_lip_png is empty (partial entry)."""
        cache = PreprocessingCache()
        result = PreprocessResult(
            densepose_png=b"dp_data",
            schp_atr_png=b"atr_data",
            schp_lip_png=b"",
        )
        cache.put("partial_key", result)
        assert cache.get("partial_key") is None

    def test_corrupt_entry_returns_none(self):
        """Returns None and does not raise when entry is corrupt."""
        cache = PreprocessingCache()
        # Manually inject a corrupt entry (non-PreprocessResult)
        cache._store["corrupt_key"] = "not_a_result"  # type: ignore
        # Should not raise, returns None
        assert cache.get("corrupt_key") is None


class TestPut:
    """Tests for PreprocessingCache.put."""

    def test_stores_and_retrieves_entry(self):
        """Stored entry can be retrieved with the same key."""
        cache = PreprocessingCache()
        result = PreprocessResult(
            densepose_png=b"a",
            schp_atr_png=b"b",
            schp_lip_png=b"c",
        )
        cache.put("my_key", result)
        assert cache.get("my_key") is result

    def test_overwrites_existing_entry(self):
        """Putting with an existing key overwrites the previous entry."""
        cache = PreprocessingCache()
        result1 = PreprocessResult(
            densepose_png=b"old",
            schp_atr_png=b"old",
            schp_lip_png=b"old",
        )
        result2 = PreprocessResult(
            densepose_png=b"new",
            schp_atr_png=b"new",
            schp_lip_png=b"new",
        )
        cache.put("key", result1)
        cache.put("key", result2)
        retrieved = cache.get("key")
        assert retrieved is not None
        assert retrieved.densepose_png == b"new"


class TestEndToEnd:
    """End-to-end tests combining compute_key, put, and get."""

    def test_full_workflow(self):
        """compute_key -> put -> get round-trip works correctly."""
        cache = PreprocessingCache()
        img = Image.new("RGB", (64, 64), color=(255, 0, 128))
        key = cache.compute_key(img)

        result = PreprocessResult(
            densepose_png=b"densepose_bytes",
            schp_atr_png=b"schp_atr_bytes",
            schp_lip_png=b"schp_lip_bytes",
        )
        cache.put(key, result)

        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved == result

    def test_same_image_content_hits_cache(self):
        """Two images with same pixels share the same cache entry."""
        cache = PreprocessingCache()
        img1 = Image.new("RGB", (32, 32), color=(1, 2, 3))
        img2 = Image.new("RGB", (32, 32), color=(1, 2, 3))

        result = PreprocessResult(
            densepose_png=b"dp",
            schp_atr_png=b"atr",
            schp_lip_png=b"lip",
        )
        cache.put(cache.compute_key(img1), result)

        # img2 with same content should find the entry
        retrieved = cache.get(cache.compute_key(img2))
        assert retrieved is result
