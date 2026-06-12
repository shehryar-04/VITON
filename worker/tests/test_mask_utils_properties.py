"""Property-based tests for worker/mask_utils.py using Hypothesis."""

# Feature: flux-automasker-integration, Property 8: Feathering produces intermediate values
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
from PIL import Image

from worker.mask_utils import feather_mask


@settings(max_examples=100)
@given(
    width=st.integers(10, 100),
    height=st.integers(10, 100),
    feather_px=st.integers(1, 10),
    data=st.data(),
)
def test_feathering_produces_intermediate_values(width, height, feather_px, data):
    """Property 8: For binary masks with both 0 and 255 regions, feathering produces intermediate values.

    **Validates: Requirements 5.2**
    """
    # Generate random binary mask using numpy strategy
    arr_np = data.draw(
        arrays(
            dtype=np.uint8,
            shape=(height, width),
            elements=st.sampled_from([np.uint8(0), np.uint8(255)]),
        )
    )

    # Must have both 0 and 255 regions
    assume(np.any(arr_np == 0) and np.any(arr_np == 255))

    mask = Image.fromarray(arr_np, mode="L")

    feathered = feather_mask(mask, feather_px)
    feathered_arr = np.array(feathered)

    # At least one pixel should be strictly between 0 and 255
    intermediate = (feathered_arr > 0) & (feathered_arr < 255)
    assert np.any(intermediate), "Feathering should produce at least one intermediate value"


# Feature: flux-automasker-integration, Property 7: Dilation monotonicity
from worker.mask_utils import dilate_mask


@settings(max_examples=100)
@given(
    width=st.integers(3, 100),
    height=st.integers(3, 100),
    dilation_px=st.integers(1, 10),
    data=st.data(),
)
def test_dilation_monotonicity(width, height, dilation_px, data):
    """Property 7: For any binary mask and positive dilation_px, white-pixel count is non-decreasing.

    **Validates: Requirements 5.1**
    """
    # Generate random binary mask using numpy strategy (avoids Hypothesis BUFFER_SIZE limit)
    arr_np = data.draw(
        arrays(
            dtype=np.uint8,
            shape=(height, width),
            elements=st.sampled_from([np.uint8(0), np.uint8(255)]),
        )
    )
    mask = Image.fromarray(arr_np, mode="L")

    original_white = np.sum(np.array(mask) == 255)

    dilated = dilate_mask(mask, dilation_px)
    dilated_white = np.sum(np.array(dilated) == 255)

    assert dilated_white >= original_white, (
        f"Dilation decreased white pixels: {original_white} -> {dilated_white}"
    )


# Feature: flux-automasker-integration, Property 9: Identity when dilation=0 and feathering=0


@settings(max_examples=100, deadline=None)
@given(
    width=st.integers(1, 50),
    height=st.integers(1, 50),
    data=st.data(),
)
def test_identity_when_dilation_and_feathering_zero(width, height, data):
    """Property 9: When dilation=0 and feathering=0, mask is byte-identical to input.

    **Validates: Requirements 5.3, 8.2, 8.4**
    """
    # Generate random mask values (any grayscale values)
    arr_np = data.draw(
        arrays(
            dtype=np.uint8,
            shape=(height, width),
            elements=st.integers(0, 255),
        )
    )
    mask = Image.fromarray(arr_np, mode="L")

    original_bytes = mask.tobytes()

    # Apply dilation with 0
    after_dilate = dilate_mask(mask, 0)
    assert after_dilate.tobytes() == original_bytes, "dilate_mask(mask, 0) should return identical mask"

    # Apply feathering with 0
    after_feather = feather_mask(mask, 0)
    assert after_feather.tobytes() == original_bytes, "feather_mask(mask, 0) should return identical mask"

    # Apply both in sequence
    after_both = feather_mask(dilate_mask(mask, 0), 0)
    assert after_both.tobytes() == original_bytes, "Both at 0 should return identical mask"


# Feature: flux-automasker-integration, Property 10: Dilation-before-feathering ordering


@settings(max_examples=100)
@given(
    width=st.integers(40, 100),
    height=st.integers(40, 100),
    dilation_px=st.integers(2, 5),
    feather_px=st.integers(2, 5),
    data=st.data(),
)
def test_dilation_before_feathering_ordering(width, height, dilation_px, feather_px, data):
    """Property 10: Dilation then feathering differs from feathering then dilation.

    **Validates: Requirements 5.6**
    """
    from hypothesis.extra.numpy import arrays as np_arrays

    # Generate random binary mask with both 0 and 255 regions
    arr_np = data.draw(
        np_arrays(
            dtype=np.uint8,
            shape=(height, width),
            elements=st.sampled_from([np.uint8(0), np.uint8(255)]),
        )
    )
    assume(np.any(arr_np == 0) and np.any(arr_np == 255))

    # Ensure the black region is large enough that dilation won't completely fill it.
    # The black region must have more pixels than what dilation could fill from the boundary.
    # A conservative check: require at least (2*dilation_px+1)^2 black pixels so there's
    # a core that survives dilation.
    n_black = int(np.sum(arr_np == 0))
    min_black_needed = (2 * dilation_px + 1) ** 2
    assume(n_black >= min_black_needed)

    # Also require enough white pixels so feathering has room
    n_white = int(np.sum(arr_np == 255))
    assume(n_white >= min_black_needed)

    mask = Image.fromarray(arr_np, mode="L")

    # Order 1: dilation then feathering (correct order per spec)
    dilated_then_feathered = feather_mask(dilate_mask(mask, dilation_px), feather_px)

    # Order 2: feathering then dilation (wrong order)
    feathered_then_dilated = dilate_mask(feather_mask(mask, feather_px), dilation_px)

    # They should produce different results for non-trivial masks
    arr1 = np.array(dilated_then_feathered)
    arr2 = np.array(feathered_then_dilated)
    assert not np.array_equal(arr1, arr2), (
        "Dilation-then-feathering should differ from feathering-then-dilation for non-trivial masks"
    )
