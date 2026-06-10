"""
GPU Worker — Celery task that orchestrates the full try-on pipeline per batch.

Task name: tryon.process_batch
Registered on: celery_app from worker.queue

Pipeline per job:
  1. Fetch job from Supabase
  2. Update status → "processing"
  3. Check ResultCache (exact hash) → hit: mark completed, fire webhook, next job
  4. Check SimilarityCache → hit: mark completed, fire webhook, next job
  5. Check PreprocessingCache for user image → use cached masks or run AutoMasker
  6. Run InferenceEngine.run_batch() for all jobs that reached this step
  7. Upload result to Cloudinary, store in caches, update Supabase, fire webhook
  8. On per-job error: mark "failed", fire webhook, continue
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import cloudinary
import cloudinary.uploader
import redis as redis_lib
from PIL import Image
from supabase import create_client

import worker.config as cfg
from worker.cache import (
    PreprocessingCache,
    ResultCache,
    SimilarityCache,
    compute_phash,
    compute_result_cache_key,
)
from worker.inference_engine import InferenceEngine
from worker.models import InferenceConfig, JobRecord, PreprocessResult
from worker.queue import celery_app

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named counters
# ---------------------------------------------------------------------------

CACHE_COUNTERS: dict[str, int] = {
    "result_cache_hit": 0,
    "result_cache_miss": 0,
    "similarity_cache_hit": 0,
    "similarity_cache_miss": 0,
    "preprocess_cache_hit": 0,
    "preprocess_cache_miss": 0,
}

# ---------------------------------------------------------------------------
# Module-level singletons (loaded once when the Celery worker process starts)
# ---------------------------------------------------------------------------

redis_client = redis_lib.from_url(cfg.REDIS_URL)

result_cache = ResultCache(redis_client)
similarity_cache = SimilarityCache(
    redis_client,
    method=cfg.SIMILARITY_METHOD,
    threshold=cfg.SIMILARITY_THRESHOLD,
)
preprocess_cache = PreprocessingCache(redis_client)

inference_engine = InferenceEngine(
    InferenceConfig(
        pipeline_type=cfg.PIPELINE_TYPE,
        device=cfg.GPU_DEVICE,
        batch_size=cfg.BATCH_SIZE,
        flash_attention=cfg.FLASH_ATTENTION,
        vae_tiling=cfg.VAE_TILING,
        vae_slicing=cfg.VAE_SLICING,
        vae_tiling_resolution=cfg.VAE_TILING_RESOLUTION,
        num_inference_steps=cfg.NUM_INFERENCE_STEPS,
        guidance_scale=cfg.GUIDANCE_SCALE,
        use_clip_cross_attn=cfg.USE_CLIP_CROSS_ATTN,
        flux_base_ckpt=cfg.FLUX_BASE_CKPT,
        flux_lora_path=cfg.FLUX_LORA_PATH,
        flux_quantize_4bit=cfg.FLUX_QUANTIZE_4BIT,
        flux_vae_fp32=cfg.FLUX_VAE_FP32,
        flux_cpu_offload=cfg.FLUX_CPU_OFFLOAD,
        flux_height=cfg.FLUX_HEIGHT,
        flux_width=cfg.FLUX_WIDTH,
        flux_num_inference_steps=cfg.FLUX_NUM_INFERENCE_STEPS,
        flux_guidance_scale=cfg.FLUX_GUIDANCE_SCALE,
        enhance=cfg.ENHANCE,
        enhance_scale=cfg.ENHANCE_SCALE,
        enhance_outscale=cfg.ENHANCE_OUTSCALE,
        enhance_region_only=cfg.ENHANCE_REGION_ONLY,
        enhance_weight_path=cfg.ENHANCE_WEIGHT_PATH,
        enhance_tile=cfg.ENHANCE_TILE,
    )
)

supabase_client = create_client(cfg.SUPABASE_URL, cfg.SUPABASE_ANON_KEY)

cloudinary.config(
    cloud_name=cfg.CLOUDINARY_CLOUD_NAME,
    api_key=cfg.CLOUDINARY_API_KEY,
    api_secret=cfg.CLOUDINARY_API_SECRET,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _fetch_job(job_id: str) -> JobRecord:
    """Fetch a job record from Supabase by ID."""
    response = supabase_client.table("jobs").select("*").eq("id", job_id).single().execute()
    data = response.data
    return JobRecord(
        id=data["id"],
        user_image_url=data["user_image_url"],
        cloth_image_url=data["cloth_image_url"],
        cloth_type=data["cloth_type"],
        status=data["status"],
        priority=data.get("priority", 10),
        user_tier=data.get("user_tier", "standard"),
        webhook_url=data.get("webhook_url"),
        result_image_url=data.get("result_image_url"),
        error=data.get("error"),
        created_at=datetime.fromisoformat(data["created_at"]),
        completed_at=(
            datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        ),
        enqueued_at=datetime.fromisoformat(data["enqueued_at"]),
    )


def _update_job_status(
    job_id: str,
    status: str,
    result_url: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Update job status (and optionally result_url / error) in Supabase."""
    payload: dict = {"status": status}
    if result_url is not None:
        payload["result_image_url"] = result_url
    if error is not None:
        payload["error"] = error
    if status in ("completed", "failed"):
        payload["completed_at"] = datetime.now(timezone.utc).isoformat()

    try:
        supabase_client.table("jobs").update(payload).eq("id", job_id).execute()
    except Exception as exc:
        logger.error("Supabase update failed for job %s (status=%s): %s", job_id, status, exc)
        # Retry once
        try:
            supabase_client.table("jobs").update(payload).eq("id", job_id).execute()
        except Exception as exc2:
            logger.error(
                "Supabase update retry also failed for job %s: %s", job_id, exc2
            )


def _upload_to_cloudinary(image: Image.Image) -> str:
    """Upload a PIL Image to Cloudinary and return the secure URL."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    result = cloudinary.uploader.upload(buf, resource_type="image", folder="tryon_results")
    return result["secure_url"]


def _fire_webhook(job: JobRecord) -> None:
    """Call notify() from worker.webhook_notifier if webhook_url is set."""
    if not job.webhook_url:
        return
    try:
        from worker.webhook_notifier import notify  # imported lazily (task 9)
        notify(job)
    except ImportError:
        logger.debug("webhook_notifier not yet available — skipping webhook for job %s", job.id)
    except Exception as exc:
        logger.warning("Webhook fire failed for job %s: %s", job.id, exc)


def _log_status_change(
    job_id: str,
    status: str,
    pipeline_type: str,
    batch_size: int,
    elapsed_ms: float,
) -> None:
    logger.info(
        "job_status_change",
        extra={
            "job_id": job_id,
            "status": status,
            "pipeline_type": pipeline_type,
            "batch_size": batch_size,
            "elapsed_ms": elapsed_ms,
        },
    )


def _log_cache_event(job_id: str, cache_type: str, hit: bool) -> None:
    outcome = "hit" if hit else "miss"
    counter_key = f"{cache_type}_cache_{outcome}"
    CACHE_COUNTERS[counter_key] = CACHE_COUNTERS.get(counter_key, 0) + 1
    logger.info(
        "cache_event",
        extra={
            "job_id": job_id,
            "cache_type": cache_type,
            "outcome": outcome,
            "counter": CACHE_COUNTERS[counter_key],
        },
    )


def _image_bytes_from_url(url: str) -> bytes:
    """Download image bytes from a URL."""
    import requests
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------


@celery_app.task(name="tryon.process_batch", bind=True)
def process_batch(self, job_ids: list[str]) -> None:
    """Orchestrate the full try-on pipeline for a batch of job IDs."""
    batch_size = len(job_ids)
    pipeline_type = cfg.PIPELINE_TYPE
    batch_start = time.monotonic()

    # Jobs that need inference (didn't hit any cache)
    inference_jobs: list[JobRecord] = []
    # Map job_id → (user_bytes, cloth_bytes) for cache key computation
    job_bytes: dict[str, tuple[bytes, bytes]] = {}
    # Map job_id → preprocess result (from cache or freshly computed)
    preprocess_results: dict[str, Optional[PreprocessResult]] = {}
    # Map job_id → start time for elapsed_ms logging
    job_start: dict[str, float] = {}

    # -----------------------------------------------------------------------
    # Phase 1: Fetch, update status, check caches
    # -----------------------------------------------------------------------
    for job_id in job_ids:
        t0 = time.monotonic()
        job_start[job_id] = t0

        # 1. Fetch job
        try:
            job = _fetch_job(job_id)
        except Exception as exc:
            logger.error("Failed to fetch job %s: %s", job_id, exc)
            continue

        # 2. Update status → "processing"
        _update_job_status(job_id, "processing")
        elapsed = (time.monotonic() - t0) * 1000
        _log_status_change(job_id, "processing", pipeline_type, batch_size, elapsed)

        try:
            # Download image bytes for cache key computation
            user_bytes = _image_bytes_from_url(job.user_image_url)
            cloth_bytes = _image_bytes_from_url(job.cloth_image_url)
            job_bytes[job_id] = (user_bytes, cloth_bytes)

            result_key = compute_result_cache_key(user_bytes, cloth_bytes, job.cloth_type)

            # 3. Check ResultCache (exact hash)
            cached_url = result_cache.get(result_key)
            if cached_url is not None:
                _log_cache_event(job_id, "result", hit=True)
                job.result_image_url = cached_url
                job.status = "completed"
                _update_job_status(job_id, "completed", result_url=cached_url)
                elapsed = (time.monotonic() - t0) * 1000
                _log_status_change(job_id, "completed", pipeline_type, batch_size, elapsed)
                _fire_webhook(job)
                continue
            else:
                _log_cache_event(job_id, "result", hit=False)

            # 4. Check SimilarityCache
            user_image = Image.open(io.BytesIO(user_bytes)).convert("RGB")
            cloth_image = Image.open(io.BytesIO(cloth_bytes)).convert("RGB")
            user_phash = compute_phash(user_image)
            cloth_phash = compute_phash(cloth_image)

            similar_key = similarity_cache.query(user_phash, cloth_phash)
            if similar_key is not None:
                similar_url = result_cache.get(similar_key)
                if similar_url is not None:
                    _log_cache_event(job_id, "similarity", hit=True)
                    job.result_image_url = similar_url
                    job.status = "completed"
                    _update_job_status(job_id, "completed", result_url=similar_url)
                    elapsed = (time.monotonic() - t0) * 1000
                    _log_status_change(job_id, "completed", pipeline_type, batch_size, elapsed)
                    _fire_webhook(job)
                    continue
            _log_cache_event(job_id, "similarity", hit=False)

            # 5. Check PreprocessingCache for user image
            user_hash = hashlib.sha256(user_bytes).hexdigest()
            cached_preprocess = preprocess_cache.get(user_hash)
            if cached_preprocess is not None:
                _log_cache_event(job_id, "preprocess", hit=True)
                preprocess_results[job_id] = cached_preprocess
            else:
                _log_cache_event(job_id, "preprocess", hit=False)
                preprocess_results[job_id] = None  # will run AutoMasker via InferenceEngine

            inference_jobs.append(job)

        except Exception as exc:
            logger.exception("Pipeline error for job %s during cache phase: %s", job_id, exc)
            _update_job_status(job_id, "failed", error=str(exc))
            elapsed = (time.monotonic() - t0) * 1000
            _log_status_change(job_id, "failed", pipeline_type, batch_size, elapsed)
            try:
                job.status = "failed"
                job.error = str(exc)
                _fire_webhook(job)
            except Exception:
                pass

    if not inference_jobs:
        return

    # -----------------------------------------------------------------------
    # Phase 2: Run inference for jobs that missed all caches
    # -----------------------------------------------------------------------
    try:
        inference_results = inference_engine.run_batch(inference_jobs)
    except Exception as exc:
        logger.exception("InferenceEngine.run_batch raised unexpectedly: %s", exc)
        for job in inference_jobs:
            t0 = job_start.get(job.id, time.monotonic())
            _update_job_status(job.id, "failed", error=str(exc))
            elapsed = (time.monotonic() - t0) * 1000
            _log_status_change(job.id, "failed", pipeline_type, batch_size, elapsed)
            job.status = "failed"
            job.error = str(exc)
            _fire_webhook(job)
        return

    # -----------------------------------------------------------------------
    # Phase 3: Upload results, update caches, update Supabase, fire webhooks
    # -----------------------------------------------------------------------
    job_map = {job.id: job for job in inference_jobs}

    for inf_result in inference_results:
        job_id = inf_result.job_id
        job = job_map.get(job_id)
        t0 = job_start.get(job_id, time.monotonic())

        if job is None:
            logger.error("InferenceResult for unknown job_id %s", job_id)
            continue

        if inf_result.error is not None or inf_result.image is None:
            err_msg = inf_result.error or "inference returned no image"
            logger.error("Inference failed for job %s: %s", job_id, err_msg)
            _update_job_status(job_id, "failed", error=err_msg)
            elapsed = (time.monotonic() - t0) * 1000
            _log_status_change(job_id, "failed", pipeline_type, batch_size, elapsed)
            job.status = "failed"
            job.error = err_msg
            _fire_webhook(job)
            continue

        try:
            # g. Upload image to Cloudinary
            result_url = _upload_to_cloudinary(inf_result.image)

            # Store in ResultCache
            user_bytes, cloth_bytes = job_bytes.get(job_id, (b"", b""))
            result_key = compute_result_cache_key(user_bytes, cloth_bytes, job.cloth_type)
            result_cache.set(result_key, result_url, ttl=cfg.RESULT_CACHE_TTL)

            # Store in SimilarityCache
            user_image = Image.open(io.BytesIO(user_bytes)).convert("RGB")
            cloth_image = Image.open(io.BytesIO(cloth_bytes)).convert("RGB")
            user_phash = compute_phash(user_image)
            cloth_phash = compute_phash(cloth_image)
            similarity_cache.store(user_phash, cloth_phash, result_key, ttl=cfg.RESULT_CACHE_TTL)

            # Update Supabase → "completed"
            _update_job_status(job_id, "completed", result_url=result_url)
            elapsed = (time.monotonic() - t0) * 1000
            _log_status_change(job_id, "completed", pipeline_type, batch_size, elapsed)

            job.result_image_url = result_url
            job.status = "completed"
            _fire_webhook(job)

        except Exception as exc:
            logger.exception("Post-inference error for job %s: %s", job_id, exc)
            _update_job_status(job_id, "failed", error=str(exc))
            elapsed = (time.monotonic() - t0) * 1000
            _log_status_change(job_id, "failed", pipeline_type, batch_size, elapsed)
            job.status = "failed"
            job.error = str(exc)
            _fire_webhook(job)
