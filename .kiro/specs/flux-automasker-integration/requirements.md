# Requirements Document

## Introduction

This feature integrates the existing garment-agnostic mask generator (AutoMasker, based on DensePose + SCHP, in `model/cloth_masker.py`) into the Flux try-on path so that inpaint masks are constructed per garment type instead of using a crude fixed-rectangle placeholder.

Today, the standalone Flux smoke test (`scripts/flux_t4_smoketest.py`) builds a synthetic center-torso rectangle mask. Because that rectangle overlaps the lower body, changing only the top causes the pants (which should be untouched) to come out altered or "torn". A proper garment-agnostic mask covers only the target garment region (selected by `cloth_type`: `upper`/`lower`/`overall`/`inner`/`outer`) and protects everything else.

The inference engine (`worker/inference_engine.py`) already wires AutoMasker for both the CatVTON and Flux paths through the `_get_or_generate_mask(job, user_image)` helper, which calls `self.auto_masker(user_image, mask_type=job.cloth_type)` and falls back to a blank white mask when AutoMasker is unavailable. The remaining gaps this feature addresses are:

- Ensuring the Flux worker path reliably produces and uses a `cloth_type`-aware AutoMasker mask, correctly aligned to the Flux input dimensions.
- Replacing the smoke test's synthetic rectangle with an AutoMasker-generated mask, while still allowing a manually supplied mask.
- Hardening graceful fallback behavior when AutoMasker or its checkpoints are unavailable.
- Optionally protecting unmasked regions (face/pants/background) from drift via a feathered composite.
- Reusing cached preprocessing outputs for performance.

CatVTON behavior must remain unchanged.

## Glossary

- **AutoMasker**: The mask generator in `model/cloth_masker.py`. Given a PIL person image and a `mask_type`, returns a dict containing `mask` (PIL grayscale), `densepose`, `schp_lip`, and `schp_atr`. Requires DensePose and SCHP checkpoints.
- **Mask_Type**: One of `upper`, `lower`, `overall`, `inner`, `outer`. Selects which garment region AutoMasker covers and which regions it protects.
- **Cloth_Type**: The `cloth_type` field on a `JobRecord`, used as the `mask_type` for AutoMasker. Values match Mask_Type.
- **Flux_Pipeline**: The `FluxTryOnPipeline` in `model/flux/pipeline_flux_tryon.py`. Concatenates the masked person and the garment side-by-side in latent space and regenerates only the masked region. Accepts `image`, `condition_image`, `mask_image`, `height`, `width`, `num_inference_steps`, `guidance_scale`.
- **Inference_Engine**: The `InferenceEngine` in `worker/inference_engine.py` that loads pipelines and runs jobs.
- **Mask_Helper**: The `_get_or_generate_mask(job, user_image)` method on Inference_Engine that produces the mask for a job.
- **Smoke_Test**: The standalone script `scripts/flux_t4_smoketest.py` used to validate Flux load + run on a Tesla T4.
- **Blank_Mask**: A fully white (value 255) grayscale mask the same size as the person image, meaning "regenerate the entire image".
- **Flux_Dimensions**: The target try-on resolution for the Flux path, defaulting to width 768 x height 1024, where width and height are each divisible by 16.
- **Preprocessing_Cache**: The cache (`PreprocessResult` / cache layer) storing DensePose and SCHP outputs (`densepose_png`, `schp_atr_png`, `schp_lip_png`) keyed per person image.
- **Mask_Composite**: An optional post-decode step that pastes the original person image back into the result outside the (feathered) mask region, guaranteeing untouched regions stay pixel-identical.
- **Feathering**: Applying a blur/dilation to the mask edge so the composite transition is smooth rather than a hard seam.

## Requirements

### Requirement 1: Flux path uses cloth_type-aware AutoMasker masks

**User Story:** As a user trying on a top, I want only the top region regenerated, so that my pants and the rest of the image stay untouched instead of coming out altered.

#### Acceptance Criteria

1. WHEN the Inference_Engine processes a Flux job AND AutoMasker is available, THE Mask_Helper SHALL invoke AutoMasker exactly once, passing the job person image and `mask_type` equal to the job Cloth_Type.
2. THE Mask_Helper SHALL return the `mask` entry produced by AutoMasker — a single-channel grayscale image whose width and height equal those of the person image — as the mask used for Flux inference.
3. WHEN a Flux job has Cloth_Type `upper`, THE Mask_Helper SHALL produce a mask in which the upper-garment region has value 255 (to be regenerated) and the lower-body garment region has value 0 (to be preserved).
4. WHERE a Flux job specifies a Cloth_Type in the set `upper`, `lower`, `overall`, `inner`, `outer`, THE Mask_Helper SHALL pass that value unchanged as the AutoMasker `mask_type`.
5. IF a Flux job specifies a Cloth_Type outside the set `upper`, `lower`, `overall`, `inner`, `outer`, THEN THE Mask_Helper SHALL return a descriptive error identifying the invalid Cloth_Type, SHALL NOT invoke AutoMasker or the Flux_Pipeline, and SHALL leave the job inputs unmodified.
6. IF AutoMasker is unavailable WHEN the Mask_Helper is invoked for a Flux job, THEN THE Mask_Helper SHALL return a Blank_Mask (value 255) whose width and height equal those of the person image AND SHALL log a warning identifying the affected job.

### Requirement 2: Mask resolution and alignment with Flux input dimensions

**User Story:** As a developer, I want the mask aligned to the Flux input dimensions, so that the protected and regenerated regions correspond to the correct pixels of the person image.

#### Acceptance Criteria

1. WHEN a mask is requested for a job, THE Mask_Helper SHALL return a single-channel grayscale mask whose width and height in pixels are each equal to the corresponding width and height of the person image supplied for that same job.
2. WHEN the mask width or height differs from the Flux_Dimensions used for inference, THE Flux_Pipeline SHALL resize the mask to the Flux_Dimensions using nearest-neighbor interpolation before constructing the masked image.
3. THE Flux path SHALL use Flux_Dimensions whose width and height are each an integer multiple of 16, defaulting to width 768 and height 1024 when no dimensions are otherwise specified.
4. WHEN the person image and the garment image are supplied to the Flux path, THE Flux path SHALL resize the person image, garment image, and mask to one common set of Flux_Dimensions such that the resulting masked person image and mask have identical width and height and are pixel-aligned at every coordinate.
5. IF the requested or computed Flux_Dimensions are not each an integer multiple of 16, THEN THE Flux path SHALL adjust each dimension up to the nearest integer multiple of 16 before resizing any image, condition image, or mask.
6. IF the Mask_Helper cannot return a mask matching the person image dimensions for a job, THEN THE Flux_Pipeline SHALL abort that job without constructing the masked image and SHALL surface an error indicating the mask dimension mismatch, leaving the input images unmodified.

### Requirement 3: Graceful fallback when AutoMasker or checkpoints are unavailable

**User Story:** As an operator, I want the Flux path to degrade gracefully when mask generation cannot load, so that the worker stays running and the failure is visible rather than crashing.

#### Acceptance Criteria

1. IF AutoMasker cannot be loaded at Inference_Engine startup, THEN THE Inference_Engine SHALL log a warning identifying the load-failure reason, SHALL set its AutoMasker availability state to disabled, and SHALL complete startup without raising an exception.
2. WHILE AutoMasker availability state is disabled, WHEN the Mask_Helper is invoked for a job, THE Mask_Helper SHALL return a Blank_Mask — a single-channel grayscale image with every pixel set to 255 and width and height matching the person image.
3. WHILE AutoMasker availability state is disabled, WHEN the Mask_Helper returns a Blank_Mask for a job, THE Mask_Helper SHALL log a warning identifying the affected job.
4. IF AutoMasker is available but raises an error while generating a mask for a specific job, THEN THE Mask_Helper SHALL record the error against that job's InferenceResult AND THE Inference_Engine SHALL continue processing the remaining jobs in the batch without aborting.
5. THE CatVTON and Flux paths SHALL both obtain masks through one shared Mask_Helper, producing the same fallback outcome for the same AutoMasker availability state.

### Requirement 4: Smoke test uses AutoMasker with an optional manual mask override

**User Story:** As a developer validating Flux on a T4, I want the smoke test to use the real AutoMasker mask by default, so that the smoke test reproduces production masking behavior instead of a crude rectangle.

#### Acceptance Criteria

1. WHEN the Smoke_Test runs with a person image and a Cloth_Type AND no manual mask path is provided, THE Smoke_Test SHALL generate a single-channel grayscale (mode "L") mask using AutoMasker with the provided Cloth_Type, resized to the configured width and height, where pixel value 255 marks the inpaint region and 0 marks the preserved region.
2. WHERE the Smoke_Test is invoked with a manual mask path that points to a readable image file, THE Smoke_Test SHALL load the mask from that path, convert it to single-channel grayscale (mode "L"), resize it to the configured width and height, and use it instead of invoking AutoMasker.
3. THE Smoke_Test SHALL accept a Cloth_Type argument constrained to exactly the set {`upper`, `lower`, `overall`, `inner`, `outer`}, defaulting to `upper`.
4. IF the provided Cloth_Type is not a member of the set {`upper`, `lower`, `overall`, `inner`, `outer`}, THEN THE Smoke_Test SHALL terminate before loading the pipeline with a nonzero exit code and SHALL emit an error message identifying the invalid Cloth_Type and listing the permitted values.
5. IF AutoMasker is unavailable AND no manual mask path is provided, THEN THE Smoke_Test SHALL log a warning identifying the missing mask source and SHALL fall back to a Blank_Mask, defined as a single-channel grayscale (mode "L") image matching the configured width and height with every pixel set to 255.
6. IF a manual mask path is provided but the file does not exist or cannot be opened as an image, THEN THE Smoke_Test SHALL terminate with a nonzero exit code and SHALL emit an error message identifying the unreadable mask path.
7. WHEN the Smoke_Test generates synthetic person and garment images because real inputs were not provided, THE Smoke_Test SHALL still produce a usable mask sized to the configured width and height via AutoMasker, the manual mask path, or the Blank_Mask fallback.

### Requirement 5: Optional mask dilation and feathering

**User Story:** As a user, I want the boundary between the regenerated garment and the rest of the image to blend smoothly, so that the result has no hard seam around the garment.

#### Acceptance Criteria

1. WHERE the mask dilation amount is greater than 0 pixels, THE Mask_Helper SHALL expand the AutoMasker mask region outward by the configured dilation amount in pixels before returning the mask.
2. WHERE the mask feathering radius is greater than 0 pixels, THE Mask_Composite SHALL apply a Gaussian blur of the configured radius in pixels to the mask edge before compositing, so that mask pixel values transition from 255 (regenerated region) to 0 (preserved region) across a band whose width is proportional to the configured radius.
3. WHERE the dilation amount is 0 pixels AND the feathering radius is 0 pixels, THE Mask_Helper SHALL return the AutoMasker mask unmodified, with mask pixel values identical to the AutoMasker output.
4. THE Mask_Helper SHALL expose the dilation amount and the feathering radius as integer configuration values, each measured in pixels, each defaulting to 0.
5. IF the configured dilation amount or feathering radius is negative or non-integer, THEN THE Mask_Helper SHALL reject the value, retain the previously applied valid value, and return an error indication identifying the invalid parameter.
6. WHERE both the dilation amount and the feathering radius are greater than 0 pixels, THE Mask_Helper SHALL apply dilation before feathering.

### Requirement 6: Preserve unmasked regions via optional composite

**User Story:** As a user, I want my face, pants, and background to stay exactly as in my original photo when I change one garment, so that only the targeted garment changes.

#### Acceptance Criteria

1. WHERE Mask_Composite is enabled, WHEN the Flux_Pipeline finishes decoding the result image, THE Flux path SHALL produce a composited image in which every pixel whose corresponding mask value is 0 (preserve) is byte-identical to the original person image, and every pixel whose corresponding mask value is 255 (regenerate) is taken from the decoded Flux_Pipeline output.
2. WHERE Mask_Composite is enabled AND feathering is configured with a feather radius greater than 0 pixels, WHEN compositing, THE Flux path SHALL blend the original person image and the decoded output across the feather transition band using the normalized mask as the blend weight (result = output × mask + person × (1 − mask), with mask scaled to the range 0.0 to 1.0), where the feather radius SHALL be a configurable integer between 0 and 50 pixels with a default of 0 pixels documented in the design.
3. WHERE Mask_Composite is enabled, WHEN the original person image dimensions or the mask dimensions differ from the decoded result image dimensions, THE Flux path SHALL resize both the original person image and the mask to the result image dimensions before compositing so that all three are pixel-aligned to identical width and height.
4. WHERE Mask_Composite is disabled, THE Flux path SHALL return the decoded Flux_Pipeline output unmodified, with no pixel altered.
5. WHERE Mask_Composite is enabled, IF the mask is absent, empty, or cannot be aligned to the result image dimensions, THEN THE Flux path SHALL return the decoded Flux_Pipeline output unmodified and surface an error indication identifying the missing or invalid mask, without partially compositing the result.
6. THE Mask_Composite control SHALL be exposed as a boolean configuration value with a default of enabled documented in the design.

### Requirement 7: Reuse cached preprocessing outputs

**User Story:** As an operator, I want DensePose and SCHP outputs reused across requests for the same person image, so that mask generation does not repeat expensive preprocessing.

#### Acceptance Criteria

1. WHEN a mask is requested for a person image and a complete cache entry containing all three outputs (`densepose`, `schp_atr`, `schp_lip`) exists in the Preprocessing_Cache for that person image, THE Mask_Helper SHALL build the mask using the cached outputs without invoking DensePose or SCHP.
2. WHEN AutoMasker computes DensePose and SCHP outputs for a person image that has no matching cache entry, THE Mask_Helper SHALL store the `densepose`, `schp_atr`, and `schp_lip` outputs as a single entry in the Preprocessing_Cache keyed by the person image identity.
3. THE Mask_Helper SHALL store cached DensePose and SCHP outputs as lossless PNG bytes such that a mask built from reused cached outputs is byte-identical to a mask built from freshly computed outputs for the same person image.
4. IF fewer than all three outputs (`densepose`, `schp_atr`, `schp_lip`) are present in the Preprocessing_Cache for a requested person image, THEN THE Mask_Helper SHALL treat the lookup as a cache miss, recompute the outputs via AutoMasker, and store the complete set of three outputs in the Preprocessing_Cache.
5. IF a cached entry for a requested person image cannot be read or deserialized into the three expected outputs, THEN THE Mask_Helper SHALL treat the lookup as a cache miss and recompute the outputs via AutoMasker, returning a mask without raising an error to the caller.
6. WHEN two mask requests reference person images with identical image content, THE Mask_Helper SHALL resolve both requests to the same Preprocessing_Cache entry.

### Requirement 8: CatVTON behavior remains unchanged

**User Story:** As an operator running CatVTON, I want this change to leave the CatVTON path untouched, so that existing CatVTON results are not regressed.

#### Acceptance Criteria

1. THE Inference_Engine SHALL generate CatVTON masks exclusively through the shared Mask_Helper (`_get_or_generate_mask`) that was used prior to this feature, invoking no alternate mask-generation path.
2. WHEN a CatVTON job is processed with AutoMasker available, THE Mask_Helper SHALL return a mask that is byte-for-byte identical (100% of pixel values matching, zero differing pixels) to the mask produced before this feature for the same person image and the same Cloth_Type.
3. IF AutoMasker is unavailable when a CatVTON job is processed, THEN THE Mask_Helper SHALL return a blank mask of the same pixel dimensions as the person image with all pixel values equal to 255, identical to the pre-existing fallback behavior, and SHALL emit a warning log indicating AutoMasker is unavailable.
4. WHERE the Mask_Composite, dilation, or feathering controls are not explicitly enabled by the job (default state), THE CatVTON path SHALL set dilation to 0 pixels, feathering to 0 pixels, and Mask_Composite to disabled, producing CatVTON output byte-for-byte identical to pre-feature output for equivalent inputs.
5. WHERE the Mask_Composite, dilation, or feathering controls are explicitly enabled by the job, THE CatVTON path SHALL apply each enabled control and SHALL leave all controls that are not explicitly enabled at their default no-op values (dilation 0 pixels, feathering 0 pixels, Mask_Composite disabled).
