"""
Inference engine — unified wrapper over CatVTONPipeline and FluxTryOnPipeline.

Handles:
- Pipeline loading based on config.pipeline_type
- Optional xformers / flash-attention acceleration
- VAE tiling and slicing
- Batched forward passes (Flux) or per-job loops (CatVTON)
- Per-job error isolation: one failure does not abort the batch
- VAE tiling error recovery: disable tiling and retry on RuntimeError
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import requests
from PIL import Image

from worker.models import InferenceConfig, InferenceResult, JobRecord

logger = logging.getLogger(__name__)


def _download_image(url: str) -> Image.Image:
    """Download an image from a URL and return a PIL Image."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


class InferenceEngine:
    """Unified inference interface for CatVTON and Flux try-on pipelines."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.pipeline = self._load_pipeline(config)
        self.auto_masker = self._load_auto_masker(config)
        self.enhancer = self._load_enhancer(config)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_pipeline(self, config: InferenceConfig):
        import worker.config as cfg
        if config.pipeline_type == "catvton":
            return self._load_catvton(config, cfg)
        elif config.pipeline_type == "flux":
            return self._load_flux(config, cfg)
        else:
            raise ValueError(f"Unknown pipeline_type: {config.pipeline_type!r}")

    def _load_catvton(self, config: InferenceConfig, cfg):
        from model.pipeline import CatVTONPipeline
        pipeline = CatVTONPipeline(
            base_ckpt=cfg.BASE_CKPT,
            attn_ckpt=cfg.ATTN_CKPT,
            attn_ckpt_version=cfg.ATTN_CKPT_VERSION,
            device=config.device,
        )
        try:
            pipeline.unet.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory-efficient attention enabled for CatVTON UNet")
        except (ImportError, ModuleNotFoundError):
            logger.info("xformers not installed — using default attention for CatVTON")
        return pipeline

    def _load_flux(self, config: InferenceConfig, cfg):
        from model.flux.pipeline_flux_tryon import FluxTryOnPipeline
        pipeline = FluxTryOnPipeline.from_pretrained(cfg.FLUX_CKPT)
        try:
            import flash_attn  # noqa: F401
            logger.info("flash-attn package found — PyTorch SDPA will use flash attention kernels")
        except (ImportError, ModuleNotFoundError):
            logger.info("flash-attn not installed — continuing with PyTorch SDPA default")
        if config.vae_tiling:
            pipeline.enable_vae_tiling()
            logger.info("VAE tiling enabled (threshold: %d px)", config.vae_tiling_resolution)
        if config.vae_slicing and config.batch_size > 1:
            pipeline.enable_vae_slicing()
            logger.info("VAE slicing enabled (batch_size=%d)", config.batch_size)
        return pipeline

    def _load_auto_masker(self, config: InferenceConfig):
        import worker.config as cfg
        try:
            from model.cloth_masker import AutoMasker
            masker = AutoMasker(
                densepose_ckpt=cfg.DENSEPOSE_CKPT or "./Models/DensePose",
                schp_ckpt=cfg.SCHP_CKPT or "./Models/SCHP",
                device=config.device,
            )
            logger.info("AutoMasker loaded successfully")
            return masker
        except Exception as exc:
            logger.warning("AutoMasker could not be loaded (%s) — blank mask fallback", exc)
            return None

    def _load_enhancer(self, config: InferenceConfig):
        """Load the Real-ESRGAN enhancer if enabled in config."""
        if not config.enhance:
            return None
        try:
            from model.enhancer import RealESRGANEnhancer
            enhancer = RealESRGANEnhancer(
                scale=config.enhance_scale,
                device=config.device,
                weight_path=config.enhance_weight_path or None,
                half=True,
                tile=config.enhance_tile,
                tile_pad=32,
            )
            logger.info(
                "Real-ESRGAN enhancer loaded (scale=x%d, region_only=%s)",
                config.enhance_scale, config.enhance_region_only,
            )
            return enhancer
        except Exception as exc:
            logger.warning("Real-ESRGAN enhancer could not be loaded (%s) — skipping", exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_batch(self, jobs: list[JobRecord]) -> list[InferenceResult]:
        """Run inference for a batch of jobs. Returns results in the same order as input."""
        if self.config.pipeline_type == "catvton":
            return self._run_catvton_batch(jobs)
        return self._run_flux_batch(jobs)

    # ------------------------------------------------------------------
    # CatVTON — per-job loop
    # ------------------------------------------------------------------

    def _run_catvton_batch(self, jobs: list[JobRecord]) -> list[InferenceResult]:
        return [self._run_catvton_single(job) for job in jobs]

    def _run_catvton_single(self, job: JobRecord) -> InferenceResult:
        try:
            user_image = _download_image(job.user_image_url)
            cloth_image = _download_image(job.cloth_image_url)
            mask = self._get_or_generate_mask(job, user_image)
            output_images = self.pipeline(image=user_image, condition_image=cloth_image, mask=mask)
            result_image = self._maybe_enhance(output_images[0], mask)
            return InferenceResult(job_id=job.id, image=result_image, error=None)
        except Exception as exc:
            logger.exception("CatVTON inference failed for job %s", job.id)
            return InferenceResult(job_id=job.id, image=None, error=str(exc))

    # ------------------------------------------------------------------
    # Flux — per-job with VAE tiling recovery
    # ------------------------------------------------------------------

    def _run_flux_batch(self, jobs: list[JobRecord]) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        for job in jobs:
            try:
                user_image = _download_image(job.user_image_url)
                cloth_image = _download_image(job.cloth_image_url)
                mask = self._get_or_generate_mask(job, user_image)
                result = self._run_flux_single(job, user_image, cloth_image, mask)
            except Exception as exc:
                logger.exception("Image download failed for job %s", job.id)
                result = InferenceResult(job_id=job.id, image=None, error=str(exc))
            results.append(result)
        return results

    def _run_flux_single(self, job: JobRecord, user_image, cloth_image, mask) -> InferenceResult:
        try:
            output = self._flux_infer(user_image, cloth_image, mask)
            image = output.images[0] if hasattr(output, "images") else output[0]
            image = self._maybe_enhance(image, mask)
            return InferenceResult(job_id=job.id, image=image, error=None)
        except Exception as exc:
            logger.exception("Flux inference failed for job %s", job.id)
            return InferenceResult(job_id=job.id, image=None, error=str(exc))

    def _flux_infer(self, user_image, cloth_image, mask):
        try:
            return self.pipeline(image=user_image, condition_image=cloth_image, mask_image=mask)
        except RuntimeError as exc:
            if self.config.vae_tiling:
                logger.warning("VAE RuntimeError with tiling — disabling and retrying: %s", exc)
                self.pipeline.disable_vae_tiling()
                try:
                    return self.pipeline(image=user_image, condition_image=cloth_image, mask_image=mask)
                finally:
                    self.pipeline.enable_vae_tiling()
            raise

    # ------------------------------------------------------------------
    # Enhancement helper (Real-ESRGAN)
    # ------------------------------------------------------------------

    def _maybe_enhance(self, image: Image.Image, mask: Optional[Image.Image]) -> Image.Image:
        """
        Apply Real-ESRGAN enhancement to the try-on result if enabled.

        When `enhance_region_only` is True and a mask is available, only the
        garment region is enhanced and composited back — this sharpens cloth
        textures and folds without altering the face or background.
        """
        if self.enhancer is None:
            return image
        try:
            if self.config.enhance_region_only and mask is not None:
                # Mask is at the person's original resolution; resize to the
                # result resolution so the composite aligns correctly.
                m = mask.convert("L")
                if m.size != image.size:
                    m = m.resize(image.size, Image.NEAREST)
                return self.enhancer.enhance_region(
                    image, m, outscale=self.config.enhance_outscale,
                )
            return self.enhancer.enhance(
                image, outscale=self.config.enhance_outscale,
            )
        except Exception as exc:
            logger.warning("Enhancement failed (%s) — returning un-enhanced result", exc)
            return image

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def _get_or_generate_mask(self, job: JobRecord, user_image: Image.Image) -> Image.Image:
        if self.auto_masker is not None:
            result = self.auto_masker(user_image, mask_type=job.cloth_type)
            return result["mask"]
        logger.warning("AutoMasker unavailable for job %s — using blank mask", job.id)
        return Image.new("L", user_image.size, 255)
