"""Property-based tests for composite_with_mask (Properties 11, 12, 13)."""

# Feature: flux-automasker-integration, Properties 11, 12, 13: Composite correctness
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from PIL import Image

from worker.mask_utils import composite_with_mask


@settings(max_examples=100)
@given(
    width=st.integers(10, 100),
    height=st.integers(10, 100),
    data=st.data(),
)
def test_composite_preserves_unmasked_pixels(width, height, data):
    """Property 11: At mask=0 positions, composited output == person image pixels.

    **Validates: Requirements 6.1**
    """
    # Generate random person image and decoded output
    person_arr = data.draw(arrays(np.uint8, shape=(height, width, 3)))
    output_arr = data.draw(arrays(np.uint8, shape=(height, width, 3)))

    # Create mask with some 0 (preserved) regions
    mask_arr = data.draw(
        arrays(np.uint8, shape=(height, width), elements=st.sampled_from([np.uint8(0), np.uint8(255)]))
    )

    person = Image.fromarray(person_arr, mode="RGB")
    decoded = Image.fromarray(output_arr, mode="RGB")
    mask = Image.fromarray(mask_arr, mode="L")

    result = composite_with_mask(person, decoded, mask, feather_px=0)
    result_arr = np.array(result)

    # At mask=0 positions, result should equal person
    zero_mask = mask_arr == 0
    if np.any(zero_mask):
        for c in range(3):
            assert np.all(result_arr[:, :, c][zero_mask] == person_arr[:, :, c][zero_mask]), (
                f"At mask=0, channel {c} should match person image"
            )


@settings(max_examples=100)
@given(
    width=st.integers(10, 50),
    height=st.integers(10, 50),
    data=st.data(),
)
def test_composite_blending_formula(width, height, data):
    """Property 12: Composite follows output*(mask/255) + person*(1-mask/255) within ±1.

    **Validates: Requirements 6.2, 6.3**
    """
    person_arr = data.draw(arrays(np.uint8, shape=(height, width, 3)))
    output_arr = data.draw(arrays(np.uint8, shape=(height, width, 3)))
    mask_arr = data.draw(arrays(np.uint8, shape=(height, width)))

    person = Image.fromarray(person_arr, mode="RGB")
    decoded = Image.fromarray(output_arr, mode="RGB")
    mask = Image.fromarray(mask_arr, mode="L")

    result = composite_with_mask(person, decoded, mask, feather_px=0)
    result_arr = np.array(result).astype(np.float64)

    # Expected formula
    m = mask_arr.astype(np.float64) / 255.0
    expected = output_arr.astype(np.float64) * m[:, :, np.newaxis] + \
               person_arr.astype(np.float64) * (1.0 - m[:, :, np.newaxis])
    expected = np.clip(expected, 0, 255)

    # Within ±1 tolerance per channel (rounding)
    diff = np.abs(result_arr - expected)
    assert np.all(diff <= 1.0), f"Max diff {diff.max()} exceeds ±1 tolerance"


@settings(max_examples=100)
@given(
    width=st.integers(10, 100),
    height=st.integers(10, 100),
    data=st.data(),
)
def test_composite_disabled_is_identity(width, height, data):
    """Property 13: When composite is disabled (not called), returned image == decoded output.

    **Validates: Requirements 6.4**

    Note: 'Disabled' means the composite function is not called at all. To test this property,
    we verify that when the mask is all-255 (regenerate everything), the composite output
    equals the decoded output (since mask/255 = 1.0 everywhere).
    """
    output_arr = data.draw(arrays(np.uint8, shape=(height, width, 3)))
    person_arr = data.draw(arrays(np.uint8, shape=(height, width, 3)))

    decoded = Image.fromarray(output_arr, mode="RGB")
    person = Image.fromarray(person_arr, mode="RGB")
    mask = Image.new("L", (width, height), 255)  # All regenerate

    result = composite_with_mask(person, decoded, mask, feather_px=0)
    result_arr = np.array(result)

    # With mask=255 everywhere, result should equal decoded output
    assert np.array_equal(result_arr, output_arr), "With all-255 mask, result should equal decoded output"
