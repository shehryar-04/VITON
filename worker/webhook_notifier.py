"""
Webhook notifier — sends HTTP POST callbacks when a job reaches a terminal state.

Retries up to max_retries times with exponential backoff on non-2xx or network error.
After exhausting retries, logs the failure and returns without raising.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from worker.models import JobRecord

logger = logging.getLogger(__name__)


def notify(
    job: JobRecord,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> None:
    """POST job result to job.webhook_url with exponential backoff retry.

    Validates: Requirements 9.2, 9.3, 9.4
    """
    if not job.webhook_url:
        return

    payload = {
        "job_id": job.id,
        "status": job.status,
        "result_image_url": job.result_image_url,
        "error": job.error,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(1 + max_retries):  # 1 initial + max_retries retries
        try:
            response = requests.post(job.webhook_url, json=payload, timeout=10)
            if response.status_code < 300:
                logger.info(
                    "Webhook delivered for job %s (attempt %d, status %d)",
                    job.id,
                    attempt + 1,
                    response.status_code,
                )
                return
            last_exc = None
            logger.warning(
                "Webhook non-2xx for job %s (attempt %d/%d, status %d)",
                job.id,
                attempt + 1,
                1 + max_retries,
                response.status_code,
            )
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(
                "Webhook network error for job %s (attempt %d/%d): %s",
                job.id,
                attempt + 1,
                1 + max_retries,
                exc,
            )

        if attempt < max_retries:
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)

    logger.error(
        "Webhook delivery failed for job %s after %d attempts. Last error: %s",
        job.id,
        1 + max_retries,
        last_exc,
    )
