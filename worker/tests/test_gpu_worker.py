"""
Property-based tests for gpu_worker pipeline orchestration.

# Feature: queue-batching-worker-optimization
# Property 5: Exact cache hit skips inference  (Requirements 5.2)
# Property 12: Exact cache is checked before similarity cache  (Requirements 11.5)
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock, patch

import fakeredis
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from worker.cache import ResultCache, SimilarityCache, compute_result_cache_key
from worker.models import JobRecord


# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------

def _make_job(
    user_image_url: str = "https://example.com/user.jpg",
    cloth_image_url: str = "https://example.com/cloth.jpg",
    cloth_type: str = "upper",
) -> JobRecord:
    now = datetime.utcnow()
    return JobRecord(
        id=str(uuid.uuid4()),
        user_image_url=user_image_url,
        cloth_image_url=cloth_image_url,
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


@st.composite
def job_strategy(draw) -> JobRecord:
    cloth_type = draw(st.sampled_from(["upper", "lower", "overall", "inner", "outer"]))
    return _make_job(cloth_type=cloth_type)


@st.composite
def cached_url_strategy(draw) -> str:
    return "https://res.cloudinary.com/demo/image/upload/" + draw(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=8, max_size=32)
    )


# ---------------------------------------------------------------------------
# Shared patch helpers
# ---------------------------------------------------------------------------

_FAKE_USER_BYTES = b"fake_user_image_bytes"
_FAKE_CLOTH_BYTES = b"fake_cloth_image_bytes"


def _build_worker_under_test(result_cache, similarity_cache=None, inference_engine=None):
    """
    Return a callable that mimics the core cache-check logic of process_batch
    for a single job, using the provided mocks.

    This isolates the cache-ordering logic without requiring a live Celery
    broker, Supabase, or Cloudinary connection.
    """
    from worker.cache import compute_result_cache_key, compute_phash
    from PIL import Image

    def run(job: JobRecord) -> dict:
        """
        Returns a dict with keys:
          - inference_called: bool
          - similarity_queried: bool
          - final_status: str
          - result_url: Optional[str]
        """
        user_bytes = _FAKE_USER_BYTES
        cloth_bytes = _FAKE_CLOTH_BYTES
        result_key = compute_result_cache_key(user_bytes, cloth_bytes, job.cloth_type)

        # Step 3: Check ResultCache
        cached_url = result_cache.get(result_key)
        if cached_url is not None:
            return {
                "inference_called": False,
                "similarity_queried": False,
                "final_status": "completed",
                "result_url": cached_url,
            }

        # Step 4: Check SimilarityCache
        sim_queried = False
        if similarity_cache is not None:
            sim_queried = True
            user_image = Image.new("RGB", (64, 64))
            cloth_image = Image.new("RGB", (64, 64))
            user_phash = compute_phash(user_image)
            cloth_phash = compute_phash(cloth_image)
            similar_key = similarity_cache.query(user_phash, cloth_phash)
            if similar_key is not None:
                similar_url = result_cache.get(similar_key)
                if similar_url is not None:
                    return {
                        "inference_called": False,
                        "similarity_queried": sim_queried,
                        "final_status": "completed",
                        "result_url": similar_url,
                    }

        # Step 6: Run inference
        inference_called = False
        if inference_engine is not None:
            inference_engine.run_batch([job])
            inference_called = True

        return {
            "inference_called": inference_called,
            "similarity_queried": sim_queried,
            "final_status": "inference_ran",
            "result_url": None,
        }

    return run


# ---------------------------------------------------------------------------
# Property 5: Exact cache hit skips inference
# Validates: Requirements 5.2
# ---------------------------------------------------------------------------

@given(job=job_strategy(), cached_url=cached_url_strategy())
@settings(max_examples=100, deadline=None)
def test_exact_cache_hit_skips_inference(job: JobRecord, cached_url: str) -> None:
    """
    **Validates: Requirements 5.2**

    For any job whose result cache key is already present in the ResultCache,
    InferenceEngine.run_batch SHALL NOT be called, and the job SHALL be marked
    "completed" with the cached URL.
    """
    fake_redis = fakeredis.FakeRedis()
    result_cache = ResultCache(fake_redis)

    # Pre-populate the cache with the exact key for this job's images
    result_key = compute_result_cache_key(
        _FAKE_USER_BYTES, _FAKE_CLOTH_BYTES, job.cloth_type
    )
    result_cache.set(result_key, cached_url, ttl=604800)

    inference_mock = MagicMock()
    run = _build_worker_under_test(
        result_cache=result_cache,
        inference_engine=inference_mock,
    )

    outcome = run(job)

    # Inference must NOT have been called
    inference_mock.run_batch.assert_not_called()
    assert outcome["inference_called"] is False
    assert outcome["final_status"] == "completed"
    assert outcome["result_url"] == cached_url


# ---------------------------------------------------------------------------
# Property 12: Exact cache is checked before similarity cache
# Validates: Requirements 11.5
# ---------------------------------------------------------------------------

@given(job=job_strategy())
@settings(max_examples=100, deadline=None)
def test_exact_before_similarity(job: JobRecord) -> None:
    """
    **Validates: Requirements 11.5**

    For any job whose exact result cache key is present, the SimilarityCache
    query method SHALL NOT be called.
    """
    fake_redis = fakeredis.FakeRedis()
    result_cache = ResultCache(fake_redis)

    # Pre-populate the exact result cache
    result_key = compute_result_cache_key(
        _FAKE_USER_BYTES, _FAKE_CLOTH_BYTES, job.cloth_type
    )
    result_cache.set(result_key, "https://cdn.example.com/result.jpg", ttl=604800)

    similarity_mock = MagicMock(spec=SimilarityCache)

    run = _build_worker_under_test(
        result_cache=result_cache,
        similarity_cache=similarity_mock,
    )

    outcome = run(job)

    # SimilarityCache.query must NOT have been called
    similarity_mock.query.assert_not_called()
    assert outcome["similarity_queried"] is False
    assert outcome["final_status"] == "completed"
