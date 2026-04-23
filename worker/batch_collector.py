"""
Batch collector for the GPU worker.

Accumulates jobs from an in-memory queue up to max_size, prioritizing
premium (priority=1) jobs before standard (priority=10) jobs.

Also provides promote_stale_jobs() for anti-starvation: standard jobs
older than threshold_seconds are promoted to priority 1.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from worker.models import JobRecord


class BatchCollector:
    """Collects a batch of jobs from an in-memory queue with priority ordering."""

    def __init__(self, max_size: int, timeout_ms: int) -> None:
        self.max_size = max_size
        self.timeout_ms = timeout_ms

    def collect(self, queue: List[JobRecord]) -> List[JobRecord]:
        """Drain up to max_size jobs from queue, priority=1 first then priority=10.

        Mutates the input list in-place by removing the selected jobs.
        Returns the selected jobs in priority order.
        """
        # Partition into premium (P1) and standard (P10)
        premium = [j for j in queue if j.priority == 1]
        standard = [j for j in queue if j.priority != 1]

        # Take up to max_size, filling from premium first
        selected: List[JobRecord] = []
        remaining = self.max_size

        take_premium = min(remaining, len(premium))
        selected.extend(premium[:take_premium])
        remaining -= take_premium

        take_standard = min(remaining, len(standard))
        selected.extend(standard[:take_standard])

        # Remove selected jobs from the input list in-place
        selected_set = set(id(j) for j in selected)
        queue[:] = [j for j in queue if id(j) not in selected_set]

        return selected


def promote_stale_jobs(queue: List[JobRecord], threshold_seconds: int) -> None:
    """Promote standard jobs (priority=10) older than threshold_seconds to priority=1.

    Iterates the list in-place and mutates job.priority directly.
    """
    now = datetime.now(timezone.utc)
    for job in queue:
        if job.priority != 10:
            continue
        # Support both aware and naive datetimes
        enqueued = job.enqueued_at
        if enqueued.tzinfo is None:
            enqueued = enqueued.replace(tzinfo=timezone.utc)
        age_seconds = (now - enqueued).total_seconds()
        if age_seconds > threshold_seconds:
            job.priority = 1
