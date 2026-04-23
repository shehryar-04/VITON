"""
Property-based tests for worker/webhook_notifier.py.

# Feature: queue-batching-worker-optimization
# Property 9: Webhook delivery is attempted for every terminal job state  (Requirements 9.2, 9.3)
# Property 10: Webhook retries exactly up to 3 times on failure  (Requirements 9.4)
"""
from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from worker.models import JobRecord
from worker.webhook_notifier import notify


# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------

def _make_job(status: str, webhook_url: str = "https://example.com/webhook") -> JobRecord:
    now = datetime.utcnow()
    return JobRecord(
        id=str(uuid.uuid4()),
        user_image_url="https://example.com/user.jpg",
        cloth_image_url="https://example.com/cloth.jpg",
        cloth_type="upper",
        status=status,
        priority=10,
        user_tier="standard",
        webhook_url=webhook_url,
        result_image_url="https://cdn.example.com/result.jpg" if status == "completed" else None,
        error="something went wrong" if status == "failed" else None,
        created_at=now,
        completed_at=now if status in ("completed", "failed") else None,
        enqueued_at=now,
    )


@st.composite
def terminal_job_strategy(draw) -> JobRecord:
    status = draw(st.sampled_from(["completed", "failed"]))
    return _make_job(status)


# ---------------------------------------------------------------------------
# Property 9: Webhook delivery is attempted for every terminal job state
# Validates: Requirements 9.2, 9.3
# ---------------------------------------------------------------------------

@given(job=terminal_job_strategy())
@settings(max_examples=100, deadline=None)
def test_webhook_delivery_on_terminal_state(job: JobRecord) -> None:
    """
    **Validates: Requirements 9.2, 9.3**
    For any job with a webhook_url in a terminal state, at least one POST
    attempt is made containing job_id and final status.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("worker.webhook_notifier.requests.post", return_value=mock_response) as mock_post:
        notify(job, max_retries=3, base_delay=0.0)

    assert mock_post.call_count >= 1
    call_kwargs = mock_post.call_args
    payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
    assert payload["job_id"] == job.id
    assert payload["status"] == job.status


# ---------------------------------------------------------------------------
# Property 10: Webhook retries exactly up to 3 times on failure
# Validates: Requirements 9.4
# ---------------------------------------------------------------------------

@given(job=terminal_job_strategy())
@settings(max_examples=100, deadline=None)
def test_webhook_retry_count_on_persistent_failure(job: JobRecord) -> None:
    """
    **Validates: Requirements 9.4**
    When every POST attempt returns a non-2xx response, notify() makes exactly
    1 initial attempt + 3 retries = 4 total attempts, then stops.
    """
    mock_response = MagicMock()
    mock_response.status_code = 500  # always fail

    with patch("worker.webhook_notifier.requests.post", return_value=mock_response) as mock_post:
        with patch("worker.webhook_notifier.time.sleep"):  # skip actual delays
            notify(job, max_retries=3, base_delay=0.0)

    assert mock_post.call_count == 4, (
        f"Expected 4 attempts (1 + 3 retries), got {mock_post.call_count}"
    )


@given(job=terminal_job_strategy())
@settings(max_examples=100, deadline=None)
def test_webhook_no_call_without_webhook_url(job: JobRecord) -> None:
    """notify() must make no HTTP calls when webhook_url is None."""
    job.webhook_url = None

    with patch("worker.webhook_notifier.requests.post") as mock_post:
        notify(job, max_retries=3, base_delay=0.0)

    mock_post.assert_not_called()
