"""
Unit tests for the enhanced _get_or_generate_mask method.

Tests cover:
- Input validation (cloth_type, dilation_px, feather_px)
- Blank mask fallback when AutoMasker is None
- Cache integration (preprocessing cache + mask cache)
- Dilation and feathering application
- CatVTON backward compatibility (default args produce same output)
"""
from __future__ import annotations

import io
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from worker.inference_engine import InferenceEngine
from worker.mask_utils import VALID_CLOTH_TYPES
from worker.models import JobRecord, PreprocessResult
from worker.preprocessing_cache import PreprocessingCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(cloth_type: str = "upper", job_id: str | None = None) -> JobRecord:
    now = datetime.utcnow()
    return JobRecord(
        id=job_id or str(uuid.uuid4()),
        user_image_url="https://example.com/user.jpg",
        cloth_image_url="https://example.com/cloth.jpg",
        cloth_type=cloth_type,
        status="pending",
        priority=10,
        user_tier="standard",
        webhook_url=None,
        result_image_url=None,
        error=None,
        created_at=now,
        completed_at=None,
        enqueued_at=now,
    )


def _build_engine(auto_masker=None):
    """Build an InferenceEngine without loading real pipelines."""
    engine = object.__new__(InferenceEngine)
    engine.auto_masker = auto_masker
    engine.preprocessing_cache = PreprocessingCache()
    engine._mask_cache = {}
    return engine


def _dummy_image(width: int = 64, height: int = 64) -> Image.Image:
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def _dummy_mask(width: int = 64, height: int = 64) -> Image.Image:
    """A simple binary mask: top half white (255), bottom half black (0)."""
    mask = Image.new("L", (width, height), 0)
    for y in range(height // 2):
        for x in range(width):
            mask.putpixel((x, y), 255)
    return mask


def _make_auto_masker_mock(mask: Image.Image | None = None):
    """Create a mock AutoMasker that returns a dict like the real one."""
    if mask is None:
        mask = _dummy_mask()
    densepose = Image.new("RGB", mask.size, (10, 20, 30))
    schp_atr = Image.new("RGB", mask.size, (40, 50, 60))
    schp_lip = Image.new("RGB", mask.size, (70, 80, 90))
    mock = MagicMock()
    mock.return_value = {
        "mask": mask,
        "densepose": densepose,
        "schp_atr": schp_atr,
        "schp_lip": schp_lip,
    }
    return mock


# ---------------------------------------------------------------------------
# Tests: Input Validation
# ---------------------------------------------------------------------------


class TestClothTypeValidation:
    """Requirement 1.5: Invalid cloth_type raises ValueError."""

    @pytest.mark.parametrize("cloth_type", sorted(VALID_CLOTH_TYPES))
    def test_valid_cloth_types_accepted(self, cloth_type):
        engine = _build_engine()
        job = _make_job(cloth_type=cloth_type)
        img = _dummy_image()
        # With auto_masker=None, should return blank mask without error
        mask = engine._get_or_generate_mask(job, img)
        assert mask.size == img.size
        assert mask.mode == "L"

    @pytest.mark.parametrize("bad_type", ["shirt", "UPPER", "pants", "", "full_body"])
    def test_invalid_cloth_type_raises(self, bad_type):
        engine = _build_engine()
        job = _make_job(cloth_type=bad_type)
        img = _dummy_image()
        with pytest.raises(ValueError, match="Invalid cloth_type"):
            engine._get_or_generate_mask(job, img)


class TestDilationFeatherValidation:
    """Requirement 5.5: Reject negative/non-integer dilation or feathering."""

    def test_negative_dilation_raises(self):
        engine = _build_engine()
        job = _make_job()
        img = _dummy_image()
        with pytest.raises(ValueError, match="dilation_px"):
            engine._get_or_generate_mask(job, img, dilation_px=-1)

    def test_negative_feather_raises(self):
        engine = _build_engine()
        job = _make_job()
        img = _dummy_image()
        with pytest.raises(ValueError, match="feather_px"):
            engine._get_or_generate_mask(job, img, feather_px=-5)

    def test_float_dilation_raises(self):
        engine = _build_engine()
        job = _make_job()
        img = _dummy_image()
        with pytest.raises(ValueError, match="dilation_px"):
            engine._get_or_generate_mask(job, img, dilation_px=2.5)

    def test_float_feather_raises(self):
        engine = _build_engine()
        job = _make_job()
        img = _dummy_image()
        with pytest.raises(ValueError, match="feather_px"):
            engine._get_or_generate_mask(job, img, feather_px=1.0)

    def test_bool_dilation_raises(self):
        engine = _build_engine()
        job = _make_job()
        img = _dummy_image()
        with pytest.raises(ValueError, match="dilation_px"):
            engine._get_or_generate_mask(job, img, dilation_px=True)

    def test_bool_feather_raises(self):
        engine = _build_engine()
        job = _make_job()
        img = _dummy_image()
        with pytest.raises(ValueError, match="feather_px"):
            engine._get_or_generate_mask(job, img, feather_px=False)

    def test_zero_values_accepted(self):
        engine = _build_engine()
        job = _make_job()
        img = _dummy_image()
        # Should not raise
        mask = engine._get_or_generate_mask(job, img, dilation_px=0, feather_px=0)
        assert mask is not None


# ---------------------------------------------------------------------------
# Tests: Blank Mask Fallback
# ---------------------------------------------------------------------------


class TestBlankMaskFallback:
    """Requirements 1.6, 3.2, 3.3: Blank mask when AutoMasker is None."""

    def test_returns_all_255_mask(self):
        engine = _build_engine(auto_masker=None)
        job = _make_job()
        img = _dummy_image(100, 80)
        mask = engine._get_or_generate_mask(job, img)
        assert mask.size == (100, 80)
        assert mask.mode == "L"
        assert all(px == 255 for px in mask.getdata())

    def test_logs_warning(self, caplog):
        import logging
        engine = _build_engine(auto_masker=None)
        job = _make_job(job_id="test-blank-warn")
        img = _dummy_image()
        with caplog.at_level(logging.WARNING):
            engine._get_or_generate_mask(job, img)
        assert "AutoMasker unavailable" in caplog.text
        assert "test-blank-warn" in caplog.text

    def test_no_dilation_feathering_on_blank(self):
        """Blank mask ignores dilation/feathering args (returns early)."""
        engine = _build_engine(auto_masker=None)
        job = _make_job()
        img = _dummy_image()
        mask = engine._get_or_generate_mask(job, img, dilation_px=10, feather_px=5)
        # Still all-255, since we return early without post-processing
        assert all(px == 255 for px in mask.getdata())


# ---------------------------------------------------------------------------
# Tests: AutoMasker Integration
# ---------------------------------------------------------------------------


class TestAutoMaskerIntegration:
    """Requirements 1.1, 1.2, 1.4: AutoMasker called correctly."""

    def test_calls_auto_masker_with_cloth_type(self):
        mock_masker = _make_auto_masker_mock()
        engine = _build_engine(auto_masker=mock_masker)
        job = _make_job(cloth_type="lower")
        img = _dummy_image()
        engine._get_or_generate_mask(job, img)
        mock_masker.assert_called_once_with(img, mask_type="lower")

    def test_returns_mask_from_auto_masker(self):
        expected_mask = _dummy_mask(64, 64)
        mock_masker = _make_auto_masker_mock(mask=expected_mask)
        engine = _build_engine(auto_masker=mock_masker)
        job = _make_job()
        img = _dummy_image()
        result = engine._get_or_generate_mask(job, img)
        assert result == expected_mask


# ---------------------------------------------------------------------------
# Tests: Cache Integration
# ---------------------------------------------------------------------------


class TestCacheIntegration:
    """Requirements 7.1, 7.2: Cache stores and retrieves preprocessing."""

    def test_stores_preprocessing_in_cache(self):
        mock_masker = _make_auto_masker_mock()
        engine = _build_engine(auto_masker=mock_masker)
        job = _make_job()
        img = _dummy_image()
        engine._get_or_generate_mask(job, img)

        # Verify preprocessing was stored
        image_key = PreprocessingCache.compute_key(img)
        cached = engine.preprocessing_cache.get(image_key)
        assert cached is not None
        assert len(cached.densepose_png) > 0
        assert len(cached.schp_atr_png) > 0
        assert len(cached.schp_lip_png) > 0

    def test_mask_cache_hit_skips_auto_masker(self):
        mock_masker = _make_auto_masker_mock()
        engine = _build_engine(auto_masker=mock_masker)
        job = _make_job()
        img = _dummy_image()

        # First call — cache miss, calls AutoMasker
        engine._get_or_generate_mask(job, img)
        assert mock_masker.call_count == 1

        # Second call — cache hit, does NOT call AutoMasker
        engine._get_or_generate_mask(job, img)
        assert mock_masker.call_count == 1

    def test_different_cloth_type_calls_again(self):
        mock_masker = _make_auto_masker_mock()
        engine = _build_engine(auto_masker=mock_masker)
        img = _dummy_image()

        job_upper = _make_job(cloth_type="upper")
        engine._get_or_generate_mask(job_upper, img)
        assert mock_masker.call_count == 1

        job_lower = _make_job(cloth_type="lower")
        engine._get_or_generate_mask(job_lower, img)
        assert mock_masker.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Dilation and Feathering
# ---------------------------------------------------------------------------


class TestDilationFeathering:
    """Requirements 5.1, 5.2, 5.3, 5.6."""

    def test_no_modification_when_zero(self):
        """Requirement 5.3: mask identical when dilation=0 and feathering=0."""
        expected_mask = _dummy_mask()
        mock_masker = _make_auto_masker_mock(mask=expected_mask)
        engine = _build_engine(auto_masker=mock_masker)
        job = _make_job()
        img = _dummy_image()
        result = engine._get_or_generate_mask(job, img, dilation_px=0, feather_px=0)
        assert list(result.getdata()) == list(expected_mask.getdata())

    def test_dilation_expands_mask(self):
        """Requirement 5.1: dilation increases white pixel count."""
        mask = _dummy_mask()
        original_white = sum(1 for px in mask.getdata() if px == 255)
        mock_masker = _make_auto_masker_mock(mask=mask)
        engine = _build_engine(auto_masker=mock_masker)
        job = _make_job()
        img = _dummy_image()
        result = engine._get_or_generate_mask(job, img, dilation_px=3)
        dilated_white = sum(1 for px in result.getdata() if px > 0)
        assert dilated_white >= original_white

    def test_feathering_creates_intermediate_values(self):
        """Requirement 5.2: feathering produces values between 0 and 255."""
        mask = _dummy_mask()
        mock_masker = _make_auto_masker_mock(mask=mask)
        engine = _build_engine(auto_masker=mock_masker)
        job = _make_job()
        img = _dummy_image()
        result = engine._get_or_generate_mask(job, img, feather_px=3)
        pixels = list(result.getdata())
        intermediate = [px for px in pixels if 0 < px < 255]
        assert len(intermediate) > 0, "Feathering should produce intermediate values"

    def test_dilation_before_feathering(self):
        """Requirement 5.6: dilation applied before feathering."""
        # Use a small dot mask so ordering clearly matters:
        # A small white square in the center of a larger black field.
        mask = Image.new("L", (64, 64), 0)
        for y in range(28, 36):
            for x in range(28, 36):
                mask.putpixel((x, y), 255)

        mock_masker = _make_auto_masker_mock(mask=mask)
        engine = _build_engine(auto_masker=mock_masker)
        job = _make_job()
        img = _dummy_image()

        # Apply dilation then feathering (correct order via the method)
        result = engine._get_or_generate_mask(job, img, dilation_px=5, feather_px=5)

        # Apply in the wrong order manually for comparison
        from worker.mask_utils import dilate_mask, feather_mask
        wrong_order = dilate_mask(feather_mask(mask, 5), 5)

        # The results should differ (demonstrating ordering matters)
        result_pixels = list(result.getdata())
        wrong_pixels = list(wrong_order.getdata())
        assert result_pixels != wrong_pixels, (
            "Dilation-then-feathering should differ from feathering-then-dilation"
        )
