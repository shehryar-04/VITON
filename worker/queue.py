"""
Job queue module for the GPU worker pipeline.

Sets up the Celery application with two queues:
  - tryon.priority  (priority 1 — premium tier)
  - tryon.standard  (priority 10 — standard tier)

Usage (start worker):
    celery -A worker.queue worker --queues tryon.priority,tryon.standard
"""
from __future__ import annotations

import logging

import redis as redis_lib
from celery import Celery
from kombu import Queue

from worker.config import REDIS_URL
from worker.models import JobRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------

celery_app = Celery(
    "tryon",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_queues=(
        Queue("tryon.priority"),
        Queue("tryon.standard"),
    ),
    task_default_queue="tryon.standard",
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)

# ---------------------------------------------------------------------------
# Queue names
# ---------------------------------------------------------------------------

QUEUE_PRIORITY = "tryon.priority"
QUEUE_STANDARD = "tryon.standard"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enqueue_job(job: JobRecord, priority: int) -> str:
    """Enqueue a try-on job and return the Celery task ID.

    Routes to ``tryon.priority`` when *priority* is 1, otherwise to
    ``tryon.standard``.
    """
    queue_name = QUEUE_PRIORITY if priority == 1 else QUEUE_STANDARD
    result = celery_app.send_task(
        "tryon.process_batch",
        args=[[job.id]],
        queue=queue_name,
    )
    logger.info(
        "Enqueued job %s to queue %s (task_id=%s)",
        job.id,
        queue_name,
        result.id,
    )
    return result.id


def get_queue_depth() -> dict[str, int]:
    """Return the number of pending tasks in each queue.

    Inspects the Redis list lengths directly.  Returns 0 for any queue on
    Redis error so the health endpoint never hard-fails.
    """
    depths: dict[str, int] = {QUEUE_PRIORITY: 0, QUEUE_STANDARD: 0}
    try:
        client = redis_lib.from_url(REDIS_URL)
        for queue_name in depths:
            depths[queue_name] = client.llen(queue_name)
    except Exception:
        logger.warning("Redis unavailable — returning zero queue depths", exc_info=True)
    return depths
