# Feature: flux-automasker-integration — Property-based tests for _get_or_generate_mask
"""Property-based tests for the enhanced _get_or_generate_mask method."""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from worker.inference_engine import InferenceEngine
from worker.mask_utils import VALID_CLOTH_TYPES
from worker.models import JobRecord
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


# ---------------------------------------------------------------------------
# Property 15: Cache hit skips recomputation
# ---------------------------------------------------------------------------

# Feature: flux-automasker-integration, Property 15: Cache hit skips recomputation

@settings(max_examples=50)
@given(
    cloth_type=st.sampled_from(sorted(VALID_CLOTH_TYPES)),
    width=st.integers(16, 128),
    height=st.integers(16, 128),
)
def test_cache_hit_skips_recomputation(cloth_type, width, height):
    """Property 15: With pre-populated mask cache, AutoMasker is not called.

    **Validates: Requirements 7.1, 7.4**
    """
    person_image = Image.new("RGB", (width, height), (100, 100, 100))
    expected_mask = Image.new("L", (width, height), 200)

    mock_masker = MagicMock()
    engine = _build_engine(auto_masker=mock_masker)

    # Pre-populate the mask cache
    image_key = PreprocessingCache.compute_key(person_image)
    cache_key = f"{image_key}:{cloth_type}"
    engine._mask_cache[cache_key] = expected_mask

    job = _make_job(cloth_type=cloth_type)
    result = engine._get_or_generate_mask(job, person_image)

    # AutoMasker should NOT have been called (cache hit)
    mock_masker.assert_not_called()

    # Result should be the cached mask (possibly with dilation/feathering at 0 = no change)
    assert result.size == (width, height)


# ---------------------------------------------------------------------------
# Property 1: AutoMasker delegation correctness
# ---------------------------------------------------------------------------

# Feature: flux-automasker-integration, Property 1: AutoMasker delegation correctness


@settings(max_examples=100)
@given(
    cloth_type=st.sampled_from(sorted(VALID_CLOTH_TYPES)),
    width=st.integers(16, 256),
    height=st.integers(16, 256),
)
def test_automasker_delegation_correctness(cloth_type, width, height):
    """Property 1: For valid cloth_types, AutoMasker invoked once with correct args, returns correct-sized mask.

    **Validates: Requirements 1.1, 1.2, 1.4, 2.1**
    """
    person_image = Image.new("RGB", (width, height), (128, 128, 128))
    mask_output = Image.new("L", (width, height), 200)

    mock_masker = MagicMock()
    mock_masker.return_value = {
        "mask": mask_output,
        "densepose": Image.new("RGB", (width, height)),
        "schp_atr": Image.new("RGB", (width, height)),
        "schp_lip": Image.new("RGB", (width, height)),
    }

    engine = _build_engine(auto_masker=mock_masker)
    job = _make_job(cloth_type=cloth_type)

    result = engine._get_or_generate_mask(job, person_image)

    # AutoMasker called exactly once with correct arguments
    mock_masker.assert_called_once_with(person_image, mask_type=cloth_type)

    # Returned mask has correct dimensions
    assert result.size == (width, height)
    assert result.mode == "L"


# ---------------------------------------------------------------------------
# Property 2: Invalid cloth_type rejection
# ---------------------------------------------------------------------------

# Feature: flux-automasker-integration, Property 2: Invalid cloth_type rejection
import pytest


@settings(max_examples=100)
@given(
    cloth_type=st.text(min_size=1, max_size=20).filter(lambda s: s not in VALID_CLOTH_TYPES),
    width=st.integers(16, 256),
    height=st.integers(16, 256),
)
def test_invalid_cloth_type_rejection(cloth_type, width, height):
    """Property 2: For strings not in VALID_CLOTH_TYPES, ValueError raised, AutoMasker not invoked.

    **Validates: Requirements 1.5**
    """
    person_image = Image.new("RGB", (width, height), (128, 128, 128))

    mock_masker = MagicMock()
    engine = _build_engine(auto_masker=mock_masker)
    job = _make_job(cloth_type=cloth_type)

    with pytest.raises(ValueError, match="Invalid cloth_type"):
        engine._get_or_generate_mask(job, person_image)

    # AutoMasker must NOT have been called
    mock_masker.assert_not_called()


# ---------------------------------------------------------------------------
# Property 3: Blank mask fallback
# ---------------------------------------------------------------------------

# Feature: flux-automasker-integration, Property 3: Blank mask fallback
import numpy as np


@settings(max_examples=100)
@given(
    cloth_type=st.sampled_from(sorted(VALID_CLOTH_TYPES)),
    width=st.integers(1, 512),
    height=st.integers(1, 512),
)
def test_blank_mask_fallback(cloth_type, width, height):
    """Property 3: When auto_masker is None, returns mode L mask, all 255, matching dimensions.

    **Validates: Requirements 1.6, 3.2, 8.3**
    """
    person_image = Image.new("RGB", (width, height), (64, 64, 64))

    engine = _build_engine(auto_masker=None)
    job = _make_job(cloth_type=cloth_type)

    mask = engine._get_or_generate_mask(job, person_image)

    # Mode "L"
    assert mask.mode == "L"

    # Dimensions match person image
    assert mask.size == (width, height)

    # All pixels are 255
    arr = np.array(mask)
    assert np.all(arr == 255), f"Not all pixels are 255: min={arr.min()}, max={arr.max()}"
