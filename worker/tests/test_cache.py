"""
Property-based tests for worker/cache.py.

Properties covered:
  - Property 4: Preprocessing cache round-trip preserves mask data exactly
    Validates: Requirements 4.5
  - Property 3: Result cache key is deterministic
    Validates: Requirements 5.1
  - Property 11: Similarity cache requires BOTH images to match
    Validates: Requirements 11.2
  - Property 13: pHash Hamming distance is symmetric and bounded [0, 64]
    Validates: Requirements 11.7
"""
from __future__ import annotations

import io
from typing import Tuple

import fakeredis
import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st
from PIL import Image

from worker.cache import (
    PreprocessingCache,
    ResultCache,
    SimilarityCache,
    compute_phash,
    compute_result_cache_key,
    hamming_distance,
)
from worker.models import PreprocessResult

# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------


def _make_redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


def _png_bytes(width: int = 4, height: int = 4, color: int = 128) -> bytes:
    """Return minimal PNG bytes for a grayscale image."""
    img = Image.new("L", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@st.composite
def preprocess_result_strategy(draw) -> PreprocessResult:
    """Generate a PreprocessResult with random PNG bytes."""
    color_a = draw(st.integers(min_value=0, max_value=255))
    color_b = draw(st.integers(min_value=0, max_value=255))
    color_c = draw(st.integers(min_value=0, max_value=255))
    return PreprocessResult(
        densepose_png=_png_bytes(color=color_a),
        schp_atr_png=_png_bytes(color=color_b),
        schp_lip_png=_png_bytes(color=color_c),
    )


@st.composite
def job_input_strategy(draw) -> Tuple[bytes, bytes, str]:
    user_bytes = draw(st.binary(min_size=1, max_size=256))
    cloth_bytes = draw(st.binary(min_size=1, max_size=256))
    cloth_type = draw(st.sampled_from(["upper", "lower", "overall", "inner", "outer"]))
    return (user_bytes, cloth_bytes, cloth_type)


@st.composite
def pil_image_strategy(draw) -> Image.Image:
    """Generate a small random PIL image."""
    w = draw(st.integers(min_value=8, max_value=32))
    h = draw(st.integers(min_value=8, max_value=32))
    pixels = draw(st.binary(min_size=w * h, max_size=w * h))
    img = Image.frombytes("L", (w, h), pixels)
    return img


# ---------------------------------------------------------------------------
# Property 4: Preprocessing cache round-trip preserves mask data exactly
# Validates: Requirements 4.5
# ---------------------------------------------------------------------------


# Feature: queue-batching-worker-optimization, Property 4: Preprocessing cache round-trip preserves mask data exactly
@given(result=preprocess_result_strategy())
@settings(max_examples=100)
def test_preprocessing_cache_round_trip(result: PreprocessResult) -> None:
    """Serialize → store → retrieve → deserialize must yield pixel-identical bytes."""
    cache = PreprocessingCache(redis_client=_make_redis())
    key = "test_key"
    cache.set(key, result, ttl=3600)
    retrieved = cache.get(key)

    assert retrieved is not None
    assert retrieved.densepose_png == result.densepose_png
    assert retrieved.schp_atr_png == result.schp_atr_png
    assert retrieved.schp_lip_png == result.schp_lip_png


# ---------------------------------------------------------------------------
# Property 3: Result cache key is deterministic
# Validates: Requirements 5.1
# ---------------------------------------------------------------------------


# Feature: queue-batching-worker-optimization, Property 3: Result cache key is deterministic (same inputs → same key)
@given(
    user_bytes=st.binary(min_size=1, max_size=256),
    cloth_bytes=st.binary(min_size=1, max_size=256),
    cloth_type=st.sampled_from(["upper", "lower", "overall", "inner", "outer"]),
)
@settings(max_examples=100)
def test_result_cache_key_determinism(
    user_bytes: bytes, cloth_bytes: bytes, cloth_type: str
) -> None:
    """Same inputs must always produce the same cache key."""
    key1 = compute_result_cache_key(user_bytes, cloth_bytes, cloth_type)
    key2 = compute_result_cache_key(user_bytes, cloth_bytes, cloth_type)
    assert key1 == key2


# Feature: queue-batching-worker-optimization, Property 3: Result cache key is deterministic (different inputs → different key)
@given(a=job_input_strategy(), b=job_input_strategy())
@settings(max_examples=100)
def test_result_cache_key_uniqueness(
    a: Tuple[bytes, bytes, str], b: Tuple[bytes, bytes, str]
) -> None:
    """Different inputs must produce different cache keys."""
    assume(a != b)
    assert compute_result_cache_key(*a) != compute_result_cache_key(*b)


# ---------------------------------------------------------------------------
# Property 11: Similarity cache requires BOTH images to match
# Validates: Requirements 11.2
# ---------------------------------------------------------------------------


@st.composite
def similarity_entry_strategy(draw):
    user_hash = draw(st.integers(min_value=0, max_value=2**64 - 1))
    cloth_hash = draw(st.integers(min_value=0, max_value=2**64 - 1))
    result_key = draw(st.text(alphabet="abcdef0123456789", min_size=8, max_size=16))
    return (user_hash, cloth_hash, result_key)


# Feature: queue-batching-worker-optimization, Property 11: Similarity cache requires BOTH images to match
@given(
    entry=similarity_entry_strategy(),
    threshold=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100)
def test_similarity_requires_both_images(entry, threshold: int) -> None:
    """A single-image match must be a miss; only both-image match is a hit."""
    user_hash, cloth_hash, result_key = entry
    cache = SimilarityCache(redis_client=_make_redis(), method="phash", threshold=threshold)
    cache.store(user_hash, cloth_hash, result_key, ttl=3600)

    # Flip all 64 bits of each hash → guaranteed Hamming distance 64, always > any threshold ≤ 8
    far_cloth_hash = cloth_hash ^ 0xFFFFFFFFFFFFFFFF
    far_user_hash = user_hash ^ 0xFFFFFFFFFFFFFFFF

    # User matches (same hash), cloth does NOT match (flipped)
    result = cache.query(user_hash, far_cloth_hash, threshold)
    assert result is None, "Should miss when cloth hash is far"

    # Cloth matches (same hash), user does NOT match (flipped)
    result = cache.query(far_user_hash, cloth_hash, threshold)
    assert result is None, "Should miss when user hash is far"

    # Both match (exact same hashes → distance 0, always ≤ threshold)
    result = cache.query(user_hash, cloth_hash, threshold)
    assert result == result_key, "Should hit when both hashes match"


# ---------------------------------------------------------------------------
# Property 13: pHash Hamming distance is symmetric and bounded [0, 64]
# Validates: Requirements 11.7
# ---------------------------------------------------------------------------


# Feature: queue-batching-worker-optimization, Property 13: pHash Hamming distance is symmetric and bounded [0, 64]
@given(img_a=pil_image_strategy(), img_b=pil_image_strategy())
@settings(max_examples=100, deadline=None)
def test_phash_distance_symmetric_and_bounded(
    img_a: Image.Image, img_b: Image.Image
) -> None:
    """hamming_distance(phash(A), phash(B)) must equal hamming_distance(phash(B), phash(A)) and be in [0, 64]."""
    ha = compute_phash(img_a)
    hb = compute_phash(img_b)

    d_ab = hamming_distance(ha, hb)
    d_ba = hamming_distance(hb, ha)

    assert d_ab == d_ba, "Hamming distance must be symmetric"
    assert 0 <= d_ab <= 64, f"Hamming distance {d_ab} out of bounds [0, 64]"
