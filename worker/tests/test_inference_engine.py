"""
Property-based tests for InferenceEngine.

# Feature: queue-batching-worker-optimization
# Property 1: Batch output order matches input order  (Requirements 3.4)
# Property 2: Worker isolation — one failure does not affect other batch members (Requirements 2.5, 3.5)
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional
from unittest.mock import patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from PIL import Image

from worker.models import InferenceConfig, InferenceResult, JobRecord


# ---------------------------------------------------------------------------
# Strategies / helpers
# ---------------------------------------------------------------------------

def _make_job(job_id: Optional[str] = None) -> JobRecord:
    now = datetime.utcnow()
    return JobRecord(
        id=job_id or str(uuid.uuid4()),
        user_image_url="https://example.com/user.jpg",
        cloth_image_url="https://example.com/cloth.jpg",
        cloth_type="upper",
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


@st.composite
def job_strategy(draw) -> JobRecord:
    return _make_job(draw(st.uuids()).hex)


def _mock_config(pipeline_type: str = "catvton") -> InferenceConfig:
    return InferenceConfig(
        pipeline_type=pipeline_type,
        device="cpu",
        batch_size=4,
        flash_attention=False,
        vae_tiling=False,
        vae_slicing=False,
        vae_tiling_resolution=1024,
    )


def _dummy_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(128, 128, 128))


def _build_engine(pipeline_type: str = "catvton"):
    """Build an InferenceEngine with mocked pipeline and auto_masker."""
    from worker.inference_engine import InferenceEngine
    engine = object.__new__(InferenceEngine)
    engine.config = _mock_config(pipeline_type)
    engine.pipeline = None  # replaced per-test
    engine.auto_masker = None
    engine.enhancer = None
    engine.preprocessing_cache = None
    engine._mask_cache = {}
    return engine


def _patch_download():
    return patch("worker.inference_engine._download_image", return_value=_dummy_image())


# ---------------------------------------------------------------------------
# Property 1: Batch output order matches input order
# Validates: Requirements 3.4
# ---------------------------------------------------------------------------

@given(jobs=st.lists(job_strategy(), min_size=1, max_size=4))
@settings(max_examples=100, deadline=None)
def test_batch_output_order_catvton(jobs):
    """results[i].job_id == jobs[i].id for all i (CatVTON)."""
    engine = _build_engine("catvton")
    dummy = _dummy_image()
    engine._run_catvton_single = lambda job: InferenceResult(job_id=job.id, image=dummy, error=None)

    with _patch_download():
        results = engine.run_batch(jobs)

    assert len(results) == len(jobs)
    for i, (job, result) in enumerate(zip(jobs, results)):
        assert result.job_id == job.id, f"Order mismatch at index {i}"


@given(jobs=st.lists(job_strategy(), min_size=1, max_size=4))
@settings(max_examples=100, deadline=None)
def test_batch_output_order_flux(jobs):
    """results[i].job_id == jobs[i].id for all i (Flux)."""
    engine = _build_engine("flux")
    dummy = _dummy_image()
    engine._run_flux_single = lambda job, u, c, m: InferenceResult(job_id=job.id, image=dummy, error=None)

    with _patch_download():
        results = engine.run_batch(jobs)

    assert len(results) == len(jobs)
    for i, (job, result) in enumerate(zip(jobs, results)):
        assert result.job_id == job.id, f"Order mismatch at index {i}"


# ---------------------------------------------------------------------------
# Property 2: Worker isolation — one failure does not affect other batch members
# Validates: Requirements 2.5, 3.5
# ---------------------------------------------------------------------------

@given(
    jobs=st.lists(job_strategy(), min_size=2, max_size=4),
    fail_index=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=100, deadline=None)
def test_worker_isolation_catvton(jobs, fail_index):
    assume(fail_index < len(jobs))
    engine = _build_engine("catvton")
    failing_id = jobs[fail_index].id
    dummy = _dummy_image()

    def patched_single(job):
        if job.id == failing_id:
            return InferenceResult(job_id=job.id, image=None, error="simulated failure")
        return InferenceResult(job_id=job.id, image=dummy, error=None)

    engine._run_catvton_single = patched_single

    with _patch_download():
        results = engine.run_batch(jobs)

    assert len(results) == len(jobs)
    for i, (job, result) in enumerate(zip(jobs, results)):
        if i == fail_index:
            assert result.error is not None
            assert result.image is None
        else:
            assert result.error is None
            assert result.image is not None


@given(
    jobs=st.lists(job_strategy(), min_size=2, max_size=4),
    fail_index=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=100, deadline=None)
def test_worker_isolation_flux(jobs, fail_index):
    assume(fail_index < len(jobs))
    engine = _build_engine("flux")
    failing_id = jobs[fail_index].id
    dummy = _dummy_image()

    def patched_flux_single(job, u, c, m):
        if job.id == failing_id:
            return InferenceResult(job_id=job.id, image=None, error="simulated failure")
        return InferenceResult(job_id=job.id, image=dummy, error=None)

    engine._run_flux_single = patched_flux_single

    with _patch_download():
        results = engine.run_batch(jobs)

    assert len(results) == len(jobs)
    for i, (job, result) in enumerate(zip(jobs, results)):
        if i == fail_index:
            assert result.error is not None
            assert result.image is None
        else:
            assert result.error is None
            assert result.image is not None
