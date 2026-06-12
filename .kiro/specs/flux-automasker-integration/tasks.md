# Implementation Plan: Flux AutoMasker Integration

## Overview

Integrate the existing AutoMasker into the Flux try-on path by enhancing `_get_or_generate_mask` with cloth_type validation, dilation, feathering, dimension alignment, preprocessing cache, mask composite post-decode, and updated smoke test. CatVTON behavior stays byte-identical.

## Tasks

- [x] 1. Add mask utility functions and constants
  - [x] 1.1 Create `worker/mask_utils.py` with VALID_CLOTH_TYPES, align_to_multiple_of_16, dilate_mask, feather_mask, and composite_with_mask
    - Define `VALID_CLOTH_TYPES = frozenset({"upper", "lower", "overall", "inner", "outer"})`
    - Implement `align_to_multiple_of_16(width, height) -> tuple[int, int]` that rounds each dimension up to nearest multiple of 16
    - Implement `dilate_mask(mask: Image.Image, dilation_px: int) -> Image.Image` using PIL `MaxFilter` with circular kernel approximation
    - Implement `feather_mask(mask: Image.Image, feather_px: int) -> Image.Image` using `GaussianBlur` on a mode "L" image
    - Implement `composite_with_mask(person_image, decoded_output, mask, feather_px=0) -> Image.Image` with the linear blend formula: `output × (mask/255) + person × (1 − mask/255)`
    - Ensure `dilate_mask` and `feather_mask` return input unchanged when their px argument is 0
    - _Requirements: 2.3, 2.5, 5.1, 5.2, 5.3, 5.6, 6.1, 6.2, 6.3, 6.4_

  - [x] 1.2 Write property test: dimension alignment (Property 4)
    - **Property 4: Dimension alignment to multiples of 16**
    - Use Hypothesis `@given(st.integers(1, 8192), st.integers(1, 8192))` to verify output ≥ input, divisible by 16, and minimal
    - **Validates: Requirements 2.3, 2.5**

  - [x] 1.3 Write property test: nearest-neighbor resize preserves binary values (Property 5)
    - **Property 5: Nearest-neighbor resize preserves binary values**
    - Generate random binary masks, resize using nearest-neighbor to Flux_Dimensions, assert all pixels ∈ {0, 255}
    - **Validates: Requirements 2.2, 2.4**

  - [x] 1.4 Write property test: dilation monotonicity (Property 7)
    - **Property 7: Dilation monotonicity**
    - For any binary mask and positive dilation_px, dilated mask white-pixel count ≥ original white-pixel count
    - **Validates: Requirements 5.1**

  - [x] 1.5 Write property test: feathering produces intermediate values (Property 8)
    - **Property 8: Feathering produces intermediate values**
    - For binary masks with both 0 and 255 regions and feather_px > 0, verify at least one pixel ∈ (0, 255)
    - **Validates: Requirements 5.2**

  - [x] 1.6 Write property test: identity when dilation=0 and feathering=0 (Property 9)
    - **Property 9: Identity when dilation=0 and feathering=0**
    - Verify mask is byte-identical to input when both are 0
    - **Validates: Requirements 5.3, 8.2, 8.4**

  - [x] 1.7 Write property test: dilation-before-feathering ordering (Property 10)
    - **Property 10: Dilation-before-feathering ordering**
    - For non-trivial binary masks with both dilation > 0 and feathering > 0, verify applying dilation then feathering produces a different result than feathering then dilation
    - **Validates: Requirements 5.6**

  - [x] 1.8 Write property tests: composite correctness (Properties 11, 12, 13)
    - **Property 11: Composite preserves unmasked pixels** — at mask=0 positions, output == person image pixels
    - **Property 12: Composite blending formula** — verify `output × (mask/255) + person × (1 − mask/255)` within ±1 per channel
    - **Property 13: Composite disabled is identity** — when disabled, returned image == decoded output
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**

- [x] 2. Implement PreprocessingCache
  - [x] 2.1 Create `worker/preprocessing_cache.py` with PreprocessingCache class
    - Implement `compute_key(image: Image.Image) -> str` using SHA-256 of `image.convert("RGB").tobytes()`
    - Implement `get(image_key: str) -> Optional[PreprocessResult]` that returns None on miss or corrupt entry
    - Implement `put(image_key: str, result: PreprocessResult) -> None` to store all three PNG outputs
    - Handle partial cache entries (fewer than 3 outputs) as cache misses per Requirement 7.4
    - Handle corrupt/unreadable entries gracefully as misses per Requirement 7.5
    - Use `PreprocessResult` dataclass from `worker/models.py`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 2.2 Write property test: cache key determinism (Property 16)
    - **Property 16: Cache key determinism**
    - Two Image objects with identical RGB pixel content produce identical cache keys regardless of creation path
    - **Validates: Requirements 7.6**

  - [x] 2.3 Write property test: cache round-trip fidelity (Property 14)
    - **Property 14: Cache round-trip fidelity**
    - Serialize DensePose/SCHP outputs to PNG bytes and deserialize; verify masks built from cached outputs are byte-identical to originals
    - **Validates: Requirements 7.3**

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Enhance _get_or_generate_mask with validation, dilation, feathering, and cache integration
  - [x] 4.1 Add mask config fields to InferenceConfig in `worker/models.py`
    - Add `mask_dilation_px: int = 0`
    - Add `mask_feather_px: int = 0`
    - Add `mask_composite_enabled: bool = True`
    - Add `mask_composite_feather_px: int = 0`
    - _Requirements: 5.4, 6.6_

  - [x] 4.2 Enhance `_get_or_generate_mask` in `worker/inference_engine.py`
    - Add `dilation_px: int = 0` and `feather_px: int = 0` keyword arguments
    - Validate `job.cloth_type` against `VALID_CLOTH_TYPES` at the top; raise `ValueError` with descriptive message for invalid types
    - Integrate PreprocessingCache: check cache before calling AutoMasker, store results on miss
    - After obtaining mask from AutoMasker (or cache), apply `dilate_mask` then `feather_mask` (only when px > 0)
    - Preserve blank-mask fallback when `self.auto_masker is None` (log warning, return all-255 mask)
    - Reject negative/non-integer dilation or feathering values with error per Requirement 5.5
    - _Requirements: 1.1, 1.2, 1.4, 1.5, 1.6, 3.1, 3.2, 3.3, 3.4, 3.5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 7.1, 7.2, 8.1_

  - [x] 4.3 Write property test: AutoMasker delegation correctness (Property 1)
    - **Property 1: AutoMasker delegation correctness**
    - For valid cloth_types, verify AutoMasker invoked exactly once with correct mask_type, returns mask matching person image dimensions
    - Use mock AutoMasker to verify call arguments
    - **Validates: Requirements 1.1, 1.2, 1.4, 2.1**

  - [x] 4.4 Write property test: invalid cloth_type rejection (Property 2)
    - **Property 2: Invalid cloth_type rejection**
    - For strings not in VALID_CLOTH_TYPES, verify ValueError raised without invoking AutoMasker
    - **Validates: Requirements 1.5**

  - [x] 4.5 Write property test: blank mask fallback (Property 3)
    - **Property 3: Blank mask fallback**
    - When auto_masker is None, verify returned mask is mode "L", all pixels 255, dimensions match person image
    - **Validates: Requirements 1.6, 3.2, 8.3**

  - [x] 4.6 Write property test: cache hit skips recomputation (Property 15)
    - **Property 15: Cache hit skips recomputation**
    - With a pre-populated cache, verify mask is produced without calling DensePose/SCHP processors
    - **Validates: Requirements 7.1, 7.4**

- [x] 5. Wire Flux path with aligned dimensions and post-decode composite
  - [x] 5.1 Update `_run_flux_batch` and `_run_flux_single` in `worker/inference_engine.py`
    - Call `_get_or_generate_mask` with `dilation_px` and `feather_px` from config
    - Use `align_to_multiple_of_16` on configured flux dimensions before resizing
    - Resize person image, garment image, and mask to the aligned Flux_Dimensions using nearest-neighbor for the mask
    - After pipeline decode, call `composite_with_mask` when `config.mask_composite_enabled is True`
    - When composite is disabled, return decoded output unmodified
    - Handle error isolation: catch per-job mask/inference errors and continue batch
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [x] 5.2 Update CatVTON path to use shared Mask_Helper with no-op defaults
    - Ensure `_run_catvton_single` calls `_get_or_generate_mask` with `dilation_px=0`, `feather_px=0`
    - Do NOT enable composite for CatVTON (composite disabled by default for CatVTON jobs)
    - Verify CatVTON output is byte-identical to pre-feature behavior for default settings
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 5.3 Write property test: error isolation in batch processing (Property 6)
    - **Property 6: Error isolation in batch processing**
    - In a batch of N ≥ 2 jobs where one job's AutoMasker raises, verify remaining N−1 produce valid InferenceResults
    - **Validates: Requirements 3.4**

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Update smoke test to use AutoMasker
  - [x] 7.1 Refactor `scripts/flux_t4_smoketest.py` to use AutoMasker masks
    - Add `--cloth-type` CLI argument constrained to {upper, lower, overall, inner, outer}, defaulting to "upper"
    - Validate `--cloth-type`; exit with code 1 and descriptive error for invalid values
    - Change mask resolution flow: (1) if `--mask` provided → load/convert/resize, (2) else if AutoMasker available → generate with cloth_type, (3) else → blank mask fallback with warning
    - Validate `--mask` path exists and is readable; exit code 1 if not
    - Import and instantiate AutoMasker if checkpoints available; gracefully handle unavailability
    - Remove the `_make_mask` synthetic rectangle function (replaced by AutoMasker or blank fallback)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [x] 7.2 Write unit tests for smoke test CLI argument parsing
    - Test valid cloth_type accepted, invalid cloth_type rejected with exit code 1
    - Test manual mask override loads correctly
    - Test blank mask fallback when AutoMasker unavailable
    - _Requirements: 4.3, 4.4, 4.5, 4.6_

- [x] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The project uses Python with pytest + Hypothesis for property-based testing and PIL/Pillow for image manipulation
- All new code goes in `worker/mask_utils.py` and `worker/preprocessing_cache.py`; existing code in `worker/inference_engine.py` and `scripts/flux_t4_smoketest.py` is enhanced

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "4.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "2.2", "2.3"] },
    { "id": 2, "tasks": ["4.2"] },
    { "id": 3, "tasks": ["4.3", "4.4", "4.5", "4.6"] },
    { "id": 4, "tasks": ["5.1", "5.2"] },
    { "id": 5, "tasks": ["5.3"] },
    { "id": 6, "tasks": ["7.1"] },
    { "id": 7, "tasks": ["7.2"] }
  ]
}
```
