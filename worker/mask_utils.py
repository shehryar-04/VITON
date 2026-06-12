"""
Mask utility functions for the Flux AutoMasker integration.

Provides:
- VALID_CLOTH_TYPES: allowed garment type identifiers
- align_to_multiple_of_16: dimension rounding for Flux pipeline
- dilate_mask: binary mask expansion via MaxFilter
- feather_mask: Gaussian blur for smooth mask transitions
- composite_with_mask: linear blend of person and decoded output using mask
"""

from PIL import Image, ImageFilter
import numpy as np


VALID_CLOTH_TYPES = frozenset({"upper", "lower", "overall", "inner", "outer"})


def align_to_multiple_of_16(width: int, height: int) -> tuple[int, int]:
    """
    Round width and height up to the nearest multiple of 16.

    Args:
        width: Input width in pixels.
        height: Input height in pixels.

    Returns:
        (aligned_width, aligned_height) each divisible by 16.
    """
    aligned_w = ((width + 15) // 16) * 16
    aligned_h = ((height + 15) // 16) * 16
    return aligned_w, aligned_h


def dilate_mask(mask: Image.Image, dilation_px: int) -> Image.Image:
    """
    Expand white regions of a binary mask by dilation_px using a circular kernel.

    Uses PIL MaxFilter with kernel size = 2 * dilation_px + 1 as an approximation
    of circular dilation.

    Args:
        mask: Mode "L" PIL image (binary: 0 or 255).
        dilation_px: Radius of the dilation kernel in pixels.

    Returns:
        Dilated mask (mode "L"). Returns input unchanged when dilation_px is 0.
    """
    if dilation_px == 0:
        return mask

    kernel_size = 2 * dilation_px + 1
    return mask.filter(ImageFilter.MaxFilter(size=kernel_size))


def feather_mask(mask: Image.Image, feather_px: int) -> Image.Image:
    """
    Apply Gaussian blur to mask edges to create smooth transitions.

    Args:
        mask: Mode "L" PIL image.
        feather_px: Radius of the Gaussian blur in pixels.

    Returns:
        Feathered mask (mode "L") with values transitioning from 255 to 0.
        Returns input unchanged when feather_px is 0.
    """
    if feather_px == 0:
        return mask

    # Ensure we're working with a grayscale image
    if mask.mode != "L":
        mask = mask.convert("L")

    return mask.filter(ImageFilter.GaussianBlur(radius=feather_px))


def composite_with_mask(
    person_image: Image.Image,
    decoded_output: Image.Image,
    mask: Image.Image,
    feather_px: int = 0,
) -> Image.Image:
    """
    Composite the decoded Flux output with the original person image
    using the mask as blend weight.

    Where mask=0: pixel from person_image (preserved).
    Where mask=255: pixel from decoded_output (regenerated).
    Intermediate values: linear blend.

    Formula: output × (mask/255) + person × (1 − mask/255)

    All inputs are resized to decoded_output dimensions before compositing.

    Args:
        person_image: Original person image.
        decoded_output: Flux pipeline decoded result.
        mask: Single-channel mask used for inference.
        feather_px: Gaussian blur radius for feathered compositing (0 = hard edges).

    Returns:
        Composited RGB image at decoded_output dimensions.
    """
    target_size = decoded_output.size  # (width, height)

    # Resize person image and mask to match decoded output dimensions
    person_resized = person_image.convert("RGB").resize(target_size, Image.LANCZOS)
    mask_resized = mask.convert("L").resize(target_size, Image.NEAREST)

    # Apply feathering to the mask if requested
    if feather_px > 0:
        mask_resized = mask_resized.filter(ImageFilter.GaussianBlur(radius=feather_px))

    # Convert to numpy arrays for the blend calculation
    person_arr = np.array(person_resized, dtype=np.float64)
    output_arr = np.array(decoded_output.convert("RGB"), dtype=np.float64)
    mask_arr = np.array(mask_resized, dtype=np.float64) / 255.0

    # Expand mask to 3 channels for broadcasting
    mask_3ch = mask_arr[:, :, np.newaxis]

    # Linear blend: output × (mask/255) + person × (1 − mask/255)
    composited = output_arr * mask_3ch + person_arr * (1.0 - mask_3ch)

    # Clip and convert back to uint8
    composited = np.clip(composited, 0, 255).astype(np.uint8)

    return Image.fromarray(composited, mode="RGB")
