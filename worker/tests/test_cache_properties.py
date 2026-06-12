# Feature: flux-automasker-integration, Property 14: Cache round-trip fidelity
"""Property-based tests for PreprocessingCache."""

import io

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from PIL import Image

from worker.models import PreprocessResult
from worker.preprocessing_cache import PreprocessingCache


def _to_png_bytes(img: Image.Image) -> bytes:
    """Serialize a PIL image to lossless PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _from_png_bytes(data: bytes) -> Image.Image:
    """Deserialize PNG bytes back to a PIL image."""
    return Image.open(io.BytesIO(data))


@settings(max_examples=50)
@given(
    width=st.integers(10, 100),
    height=st.integers(10, 100),
    data=st.data(),
)
def test_cache_round_trip_fidelity(width, height, data):
    """Property 14: PNG serialize/deserialize round-trip produces byte-identical images.

    **Validates: Requirements 7.3**
    """
    # Generate 3 random images (simulating DensePose, SCHP-ATR, SCHP-LIP outputs)
    dp_arr = data.draw(arrays(np.uint8, shape=(height, width, 3)))
    atr_arr = data.draw(arrays(np.uint8, shape=(height, width, 3)))
    lip_arr = data.draw(arrays(np.uint8, shape=(height, width, 3)))

    dp_img = Image.fromarray(dp_arr, mode="RGB")
    atr_img = Image.fromarray(atr_arr, mode="RGB")
    lip_img = Image.fromarray(lip_arr, mode="RGB")

    # Serialize to PNG bytes (as the cache would store them)
    dp_png = _to_png_bytes(dp_img)
    atr_png = _to_png_bytes(atr_img)
    lip_png = _to_png_bytes(lip_img)

    # Store in cache
    cache = PreprocessingCache()
    result = PreprocessResult(
        densepose_png=dp_png,
        schp_atr_png=atr_png,
        schp_lip_png=lip_png,
    )
    cache.put("test_key", result)

    # Retrieve from cache
    retrieved = cache.get("test_key")
    assert retrieved is not None

    # Deserialize and verify byte-identical to originals
    dp_restored = _from_png_bytes(retrieved.densepose_png)
    atr_restored = _from_png_bytes(retrieved.schp_atr_png)
    lip_restored = _from_png_bytes(retrieved.schp_lip_png)

    assert np.array_equal(np.array(dp_restored), dp_arr), "DensePose PNG round-trip not identical"
    assert np.array_equal(np.array(atr_restored), atr_arr), "SCHP-ATR PNG round-trip not identical"
    assert np.array_equal(np.array(lip_restored), lip_arr), "SCHP-LIP PNG round-trip not identical"


# Feature: flux-automasker-integration, Property 16: Cache key determinism
@settings(max_examples=100)
@given(
    width=st.integers(1, 200),
    height=st.integers(1, 200),
    data=st.data(),
)
def test_cache_key_determinism(width, height, data):
    """Property 16: Identical RGB content produces identical keys regardless of creation path.

    **Validates: Requirements 7.6**
    """
    # Generate random pixel data
    pixel_arr = data.draw(arrays(np.uint8, shape=(height, width, 3)))

    # Create two images from the same data via different paths
    img1 = Image.fromarray(pixel_arr, mode="RGB")
    img2 = Image.fromarray(pixel_arr.copy(), mode="RGB")

    # Also test via RGBA conversion path
    img_rgba = img1.convert("RGBA")

    key1 = PreprocessingCache.compute_key(img1)
    key2 = PreprocessingCache.compute_key(img2)
    key3 = PreprocessingCache.compute_key(img_rgba)

    assert key1 == key2, "Same pixel data from different arrays should produce same key"
    assert key1 == key3, "RGBA image with same RGB content should produce same key"
