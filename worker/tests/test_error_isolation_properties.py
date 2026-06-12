# Feature: flux-automasker-integration — Property 6: Error isolation in batch processing
"""Property-based tests for error isolation in batch processing (Property 6)."""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from worker.inference_engine import InferenceEngine
from worker.models import InferenceResult, JobRecord
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


def _build_flux_engine(auto_masker=None):
    """Build an InferenceEngine configured for Flux without loading real pipelines."""
    engine = object.__new__(InferenceEngine)
    engine.auto_masker = auto_masker
    engine.preprocessing_cache = PreprocessingCache()
    engine._mask_cache = {}
    engine.enhancer = None
    # Minimal config for Flux
    engine.config = MagicMock()
    engine.config.pipeline_type = "flux"
    engine.config.mask_dilation_px = 0
    engine.config.mask_feather_px = 0
    engine.config.mask_composite_enabled = False
    engine.config.mask_composite_feather_px = 0
    engine.config.flux_width = 768
    engine.config.flux_height = 1024
    engine.config.flux_num_inference_steps = 1
    engine.config.flux_guidance_scale = 30.0
    engine.config.vae_tiling = False
    engine.config.enhance = False
    # Mock pipeline
    engine.pipeline = MagicMock()
    mock_output = MagicMock()
    mock_output.images = [Image.new("RGB", (768, 1024), (200, 200, 200))]
    engine.pipeline.return_value = mock_output
    return engine


# ---------------------------------------------------------------------------
# Property 6: Error isolation in batch processing
# ---------------------------------------------------------------------------

# Feature: flux-automasker-integration, Property 6: Error isolation in batch processing


@settings(max_examples=20, deadline=None)
@given(
    n_jobs=st.integers(2, 6),
    failing_idx=st.data(),
)
def test_error_isolation_in_batch(n_jobs, failing_idx):
    """Property 6: In a batch of N >= 2 jobs where one AutoMasker raises, N-1 produce valid results.

    **Validates: Requirements 3.4**
    """
    fail_idx = failing_idx.draw(st.integers(0, n_jobs - 1))

    jobs = [_make_job(cloth_type="upper", job_id=f"job-{i}") for i in range(n_jobs)]

    # Track which job_id should fail
    fail_job_id = jobs[fail_idx].id

    # Use unique images per job so each triggers a separate AutoMasker call
    # (avoids mask cache hits that would skip subsequent AutoMasker invocations).
    # We achieve this by giving each job a unique URL that _download_image will map
    # to a unique image (different pixel color per job).
    for i, job in enumerate(jobs):
        job.user_image_url = f"https://example.com/user_{i}.jpg"

    # Track the download call index to determine which job is being processed
    download_call_count = [0]

    def mock_download(url: str) -> Image.Image:
        """Return a unique image per job based on the URL index."""
        # Extract index from URL if it matches our pattern
        idx = download_call_count[0] // 2  # 2 downloads per job (user + cloth)
        download_call_count[0] += 1
        # Use slightly different color per job so cache keys differ
        color_val = 50 + (idx * 10) % 200
        return Image.new("RGB", (768, 1024), (color_val, color_val, color_val))

    # Create a mock AutoMasker that raises for the specific failing job
    auto_masker_call_count = [0]

    def mock_auto_masker_call(user_image, mask_type=None):
        idx = auto_masker_call_count[0]
        auto_masker_call_count[0] += 1
        if idx == fail_idx:
            raise RuntimeError(f"Simulated AutoMasker failure for job index {idx}")
        return {
            "mask": Image.new("L", user_image.size, 255),
            "densepose": Image.new("RGB", user_image.size),
            "schp_atr": Image.new("RGB", user_image.size),
            "schp_lip": Image.new("RGB", user_image.size),
        }

    mock_masker = MagicMock(side_effect=mock_auto_masker_call)
    engine = _build_flux_engine(auto_masker=mock_masker)

    # Patch _download_image to return unique dummy images per job
    with patch("worker.inference_engine._download_image", side_effect=mock_download):
        results = engine._run_flux_batch(jobs)

    # Should have results for all jobs
    assert len(results) == n_jobs

    # The failing job should have an error
    assert results[fail_idx].error is not None, (
        f"Expected error for job at index {fail_idx}, got: {results[fail_idx]}"
    )
    assert results[fail_idx].image is None

    # All other jobs should have succeeded
    successful = [r for i, r in enumerate(results) if i != fail_idx]
    assert len(successful) == n_jobs - 1
    for i, r in enumerate(successful):
        assert r.error is None, f"Unexpected error in successful job {i}: {r.error}"
        assert r.image is not None, f"Expected image in successful job {i}"
