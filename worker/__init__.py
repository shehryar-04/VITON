# worker package
# celery_app is re-exported here so workers can be started with:
#   celery -A worker.queue worker --queues tryon.priority,tryon.standard
# Lazy import so tests can run without celery/redis installed.
try:
    from worker.queue import celery_app  # noqa: F401
except ImportError:
    pass
