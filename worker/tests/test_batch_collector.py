"""
Property-based tests for worker/batch_collector.py.

Properties covered:
  - Property 7: Batch collector drains premium jobs before standard jobs
    Validates: Requirements 8.2
  - Property 8: Anti-starvation promotes standard jobs past the wait threshold
    Validates: Requirements 8.3
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from worker.batch_collector import BatchCollector, promote_stale_jobs
from worker.models import JobRecord


# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------


def _make_job(priority: int, enqueued_at: datetime | None = None) -> JobRecord:
    now = datetime.now(timezone.utc)
    return JobRecord(
        id=str(uuid.uuid4()),
        user_image_url="https://example.com/user.jpg",
        cloth_image_url="https://example.com/cloth.jpg",
        cloth_type="upper",
        status="pending",
        priority=priority,
        user_tier="premium" if priority == 1 else "standard",
        webhook_url=None,
        result_image_url=None,
        error=None,
        created_at=now,
        completed_at=None,
        enqueued_at=enqueued_at or now,
    )


# ---------------------------------------------------------------------------
# Property 7: Batch collector drains premium jobs before standard jobs
# Validates: Requirements 8.2
# ---------------------------------------------------------------------------


# Feature: queue-batching-worker-optimization, Property 7: Batch collector drains premium jobs before standard jobs
@given(
    premium_count=st.integers(min_value=0, max_value=4),
    standard_count=st.integers(min_value=0, max_value=4),
)
@settings(max_examples=100)
def test_batch_priority_ordering(premium_count: int, standard_count: int) -> None:
    """All P1 jobs in the batch must appear before any P10 job."""
    assume(premium_count + standard_count >= 1)

    # Build a mixed queue (interleaved to stress ordering)
    queue: list[JobRecord] = []
    for _ in range(standard_count):
        queue.append(_make_job(priority=10))
    for _ in range(premium_count):
        queue.append(_make_job(priority=1))

    collector = BatchCollector(max_size=4, timeout_ms=0)
    batch = collector.collect(queue)

    assert len(batch) >= 1
    priorities = [j.priority for j in batch]

    # Find the index of the first standard job in the batch
    first_standard = next(
        (i for i, p in enumerate(priorities) if p == 10), len(priorities)
    )
    # Everything before that index must be premium
    assert all(
        p == 1 for p in priorities[:first_standard]
    ), f"Expected all P1 before any P10, got: {priorities}"


# Feature: queue-batching-worker-optimization, Property 7 (mutation): collect() removes selected jobs from queue
@given(
    premium_count=st.integers(min_value=0, max_value=4),
    standard_count=st.integers(min_value=0, max_value=4),
)
@settings(max_examples=100)
def test_collect_mutates_queue(premium_count: int, standard_count: int) -> None:
    """Jobs returned by collect() must be removed from the input queue."""
    assume(premium_count + standard_count >= 1)

    queue: list[JobRecord] = [_make_job(1) for _ in range(premium_count)] + [
        _make_job(10) for _ in range(standard_count)
    ]
    original_count = len(queue)

    collector = BatchCollector(max_size=4, timeout_ms=0)
    batch = collector.collect(queue)

    assert len(batch) + len(queue) == original_count
    batch_ids = {j.id for j in batch}
    remaining_ids = {j.id for j in queue}
    assert batch_ids.isdisjoint(remaining_ids), "Batch and remaining queue must not overlap"


# ---------------------------------------------------------------------------
# Property 8: Anti-starvation promotes standard jobs past the wait threshold
# Validates: Requirements 8.3
# ---------------------------------------------------------------------------


# Feature: queue-batching-worker-optimization, Property 8: Anti-starvation promotes standard jobs past the wait threshold
@given(
    wait_seconds=st.integers(min_value=301, max_value=3600),
    threshold=st.just(300),
)
@settings(max_examples=100)
def test_anti_starvation_promotion(wait_seconds: int, threshold: int) -> None:
    """Standard jobs older than threshold_seconds must be promoted to priority=1."""
    enqueued_at = datetime.now(timezone.utc) - timedelta(seconds=wait_seconds)
    job = _make_job(priority=10, enqueued_at=enqueued_at)

    promote_stale_jobs(queue=[job], threshold_seconds=threshold)

    assert job.priority == 1, (
        f"Job waited {wait_seconds}s (threshold={threshold}s) but priority={job.priority}"
    )


# Feature: queue-batching-worker-optimization, Property 8 (boundary): jobs below threshold are NOT promoted
@given(
    wait_seconds=st.integers(min_value=0, max_value=299),
    threshold=st.just(300),
)
@settings(max_examples=100)
def test_anti_starvation_no_premature_promotion(wait_seconds: int, threshold: int) -> None:
    """Standard jobs younger than threshold_seconds must NOT be promoted."""
    enqueued_at = datetime.now(timezone.utc) - timedelta(seconds=wait_seconds)
    job = _make_job(priority=10, enqueued_at=enqueued_at)

    promote_stale_jobs(queue=[job], threshold_seconds=threshold)

    assert job.priority == 10, (
        f"Job waited only {wait_seconds}s (threshold={threshold}s) but was promoted"
    )


# Feature: queue-batching-worker-optimization, Property 8 (premium untouched): premium jobs are never demoted
@given(
    wait_seconds=st.integers(min_value=0, max_value=3600),
    threshold=st.integers(min_value=1, max_value=300),
)
@settings(max_examples=100)
def test_anti_starvation_leaves_premium_untouched(
    wait_seconds: int, threshold: int
) -> None:
    """promote_stale_jobs must never change the priority of already-premium jobs."""
    enqueued_at = datetime.now(timezone.utc) - timedelta(seconds=wait_seconds)
    job = _make_job(priority=1, enqueued_at=enqueued_at)

    promote_stale_jobs(queue=[job], threshold_seconds=threshold)

    assert job.priority == 1
