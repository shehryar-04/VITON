"""
API Server — FastAPI app that replaces FrontEnd/fastAPI/main.py.

Routes:
  POST /api/try-on          — validate, upload, enqueue, return job ID
  GET  /api/try-on/{id}     — poll job status from Supabase
  GET  /api/health          — queue depth, active workers, cache hit rates
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import cloudinary
import cloudinary.uploader
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client

import worker.config as cfg
from worker.models import JobRecord
from worker.queue import enqueue_job, get_queue_depth, celery_app

logger = logging.getLogger(__name__)

app = FastAPI(title="Virtual Try-On API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Client = create_client(cfg.SUPABASE_URL, cfg.SUPABASE_ANON_KEY)

cloudinary.config(
    cloud_name=cfg.CLOUDINARY_CLOUD_NAME,
    api_key=cfg.CLOUDINARY_API_KEY,
    api_secret=cfg.CLOUDINARY_API_SECRET,
)

_UPPER_CATEGORIES = {"t-shirts", "full-sleeves", "hoodies", "polo"}


def _cloth_type_from_category(category_slug: str) -> str:
    return "upper" if category_slug in _UPPER_CATEGORIES else "lower"


async def _upload_image(file: UploadFile, prefix: str) -> str:
    content = await file.read()
    public_id = f"{prefix}_{uuid.uuid4()}"
    result = cloudinary.uploader.upload(content, public_id=public_id, folder="virtual-try-on")
    return result["secure_url"]


@app.post("/api/try-on")
async def try_on(
    user_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    category_id: str = Form(...),
    user_tier: str = Form(default="standard"),
    webhook_url: Optional[str] = Form(default=None),
):
    if not (user_image.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="user_image must be an image file")
    if not (cloth_image.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="cloth_image must be an image file")

    cloth_type = _cloth_type_from_category(category_id)
    priority = 1 if user_tier == "premium" else 10

    user_image_url = await _upload_image(user_image, "user")
    cloth_image_url = await _upload_image(cloth_image, "cloth")

    now = datetime.now(timezone.utc).isoformat()
    job_id = str(uuid.uuid4())
    record = {
        "id": job_id,
        "user_image_url": user_image_url,
        "cloth_image_url": cloth_image_url,
        "cloth_type": cloth_type,
        "status": "pending",
        "priority": priority,
        "user_tier": user_tier,
        "webhook_url": webhook_url,
        "created_at": now,
        "enqueued_at": now,
    }
    supabase.table("try_on_history").insert(record).execute()

    job = JobRecord(
        id=job_id,
        user_image_url=user_image_url,
        cloth_image_url=cloth_image_url,
        cloth_type=cloth_type,
        status="pending",
        priority=priority,
        user_tier=user_tier,
        webhook_url=webhook_url,
        result_image_url=None,
        error=None,
        created_at=datetime.now(timezone.utc),
        completed_at=None,
        enqueued_at=datetime.now(timezone.utc),
    )

    try:
        enqueue_job(job, priority)
    except Exception:
        logger.exception("Failed to enqueue job %s", job_id)
        raise HTTPException(status_code=503, detail={"error": "queue_unavailable"})

    return {
        "status": "success",
        "data": {
            "id": job_id,
            "status": "pending",
            "user_image_url": user_image_url,
            "cloth_image_url": cloth_image_url,
            "cloth_type": cloth_type,
        },
    }


@app.get("/api/try-on/{try_on_id}")
async def get_try_on_status(try_on_id: str):
    result = supabase.table("try_on_history").select("*").eq("id", try_on_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Try-on record not found")
    return {"status": "success", "data": result.data[0]}


@app.get("/api/health")
async def health():
    queue_depth = get_queue_depth()

    active_workers = 0
    try:
        inspect = celery_app.control.inspect(timeout=2)
        active = inspect.active()
        if active:
            active_workers = sum(len(tasks) for tasks in active.values())
    except Exception:
        logger.warning("Could not inspect Celery workers", exc_info=True)

    cache_hit_rates: dict = {}
    try:
        from worker.gpu_worker import CACHE_COUNTERS
        for cache_type in ("result", "similarity", "preprocess"):
            hits = CACHE_COUNTERS.get(f"{cache_type}_cache_hit", 0)
            misses = CACHE_COUNTERS.get(f"{cache_type}_cache_miss", 0)
            total = hits + misses
            cache_hit_rates[cache_type] = round(hits / total, 4) if total > 0 else 0.0
    except ImportError:
        pass

    return {
        "queue_depth": queue_depth,
        "active_workers": active_workers,
        "cache_hit_rates": cache_hit_rates,
    }


if __name__ == "__main__":
    uvicorn.run("worker.api_server:app", host="0.0.0.0", port=8000, reload=True)
