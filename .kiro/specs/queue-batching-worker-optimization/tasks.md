# Implementation Plan: Queue Batching Worker Optimization

## Overview

Replace the fire-and-forget `BackgroundTasks` pattern in `FrontEnd/fastAPI/main.py` with a
production-grade, queue-backed inference pipeline. Implementation proceeds bottom-up: data models
and config first, then cache, queue, batch collector, inference engine, GPU worker orchestration,
API server, webhook notifier, and finally property-based tests and migration wiring.

## Tasks

- [x] 1. Create data models and configuration
  - Create `worker/models.py` with `JobRecord`, `PreprocessResult`, `InferenceResult`, and `InferenceConfig` dataclasses exactly as specified in the design
  - Create `worker/config.py` that reads all env vars from the design's environment variable table, validates required vars are present, and exits with code 1 if any are missing
  - _Requirements: 1.1, 2.3, 3.1, 8.1, 10.1, 10.5_

- [x] 2. Implement the cache layer
  - [x] 2.1 Implement `PreprocessingCache` in `worker/cache.py`
    - `get(image_hash)` deserializes msgpack → `PreprocessResult` (PNG bytes fields)
    - `set(image_hash, result, ttl)` serializes `PreprocessResult` to msgpack and writes to Redis with TTL
    - Gracefully handles Redis unavailability (log warning, return None)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 2.2 Write property test for preprocessing cache round-trip
    - **Property 4: Preprocessing cache round-trip preserves mask data exactly**
    - **Validates: Requirements 4.5**

  - [x] 2.3 Implement `ResultCache` in `worker/cache.py`
    - `get(result_key)` returns cached Cloudinary URL or None
    - `set(result_key, url, ttl)` stores URL string in Redis with TTL
    - Expose `compute_result_cache_key(user_bytes, cloth_bytes, cloth_type) -> str` as a module-level function using SHA-256
    - Gracefully handles Redis unavailability (log warning, return None)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 2.4 Write property tests for result cache key
    - **Property 3: Result cache key is deterministic**
    - **Validates: Requirements 5.1**

  - [x] 2.5 Implement `SimilarityCache` in `worker/cache.py`
    - `query(user_hash, cloth_hash, threshold)` scans Redis similarity index and returns result_key if both hashes are within threshold, else None
    - `store(user_hash, cloth_hash, result_key, ttl)` writes a `SimilarityEntry` hash to Redis
    - Support both `phash` (Hamming distance) and `clip` (cosine similarity) methods, selected via config
    - Expose `compute_phash(image) -> int` and `hamming_distance(a, b) -> int` as module-level functions
    - Gracefully handles Redis unavailability (log warning, return None)
    - _Requirements: 11.1, 11.2, 11.4, 11.6, 11.7, 11.8, 11.9, 11.10, 11.11_

  - [ ]* 2.6 Write property tests for similarity cache
    - **Property 11: Similarity cache requires BOTH images to match**
    - **Validates: Requirements 11.2**
    - **Property 13: Perceptual hash distance is symmetric and bounded**
    - **Validates: Requirements 11.7**

- [x] 3. Set up the job queue
  - Create `worker/queue.py` with the Celery app configured against `REDIS_URL`
  - Define two queues: `tryon.priority` (priority 1) and `tryon.standard` (priority 10)
  - Implement `enqueue_job(job: JobRecord, priority: int) -> str` that routes to the correct queue
  - Implement `get_queue_depth() -> dict[str, int]` for the health endpoint
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 8.1_

  - [ ]* 3.1 Write property test for priority assignment
    - **Property 6: Priority assignment maps user tier to correct priority value**
    - **Validates: Requirements 8.1**

- [x] 4. Implement the batch collector
  - Create `worker/batch_collector.py` with `BatchCollector(max_size, timeout_ms)`
  - `collect()` blocks until `max_size` jobs are accumulated or `timeout_ms` elapses, then returns 1..max_size jobs
  - Drain priority-1 jobs before priority-10 jobs within each collection window
  - Implement `promote_stale_jobs(queue, threshold_seconds)` for anti-starvation: jobs older than threshold are promoted to priority 1
  - _Requirements: 3.1, 3.2, 3.3, 8.2, 8.3_

  - [ ]* 4.1 Write property tests for batch collector ordering and anti-starvation
    - **Property 7: Batch collector drains premium jobs before standard jobs**
    - **Validates: Requirements 8.2**
    - **Property 8: Anti-starvation promotes standard jobs past the wait threshold**
    - **Validates: Requirements 8.3**

- [x] 5. Implement the inference engine
  - Create `worker/inference_engine.py` with `InferenceEngine(config: InferenceConfig)`
  - `run_batch(jobs: list[JobRecord]) -> list[InferenceResult]` executes a single batched forward pass and returns results in input order
  - On init: enable xformers on CatVTON UNet if available; enable flash-attn on Flux transformer if available; fall back gracefully with info log if not installed
  - Call `enable_vae_tiling()` when resolution exceeds `vae_tiling_resolution`; call `enable_vae_slicing()` when batch_size > 1
  - If VAE tiling causes a runtime error, disable tiling, log the error, and retry without tiling
  - Set Flux `batch_size` parameter to the number of jobs in the current batch
  - _Requirements: 2.2, 3.2, 3.3, 3.4, 3.5, 3.6, 6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 7.4_

  - [ ]* 5.1 Write property tests for inference engine batch ordering and worker isolation
    - **Property 1: Batch output order matches input order**
    - **Validates: Requirements 3.4**
    - **Property 2: Worker isolation — one failure does not affect other batch members**
    - **Validates: Requirements 2.5, 3.5**

- [x] 6. Implement the GPU worker orchestration
  - Create `worker/gpu_worker.py` with the `process_batch` Celery task
  - Orchestrate the full pipeline per batch: check result cache → check similarity cache → check preprocessing cache → run inference → upload to Cloudinary → write result cache + similarity cache → update Supabase → fire webhook
  - Emit structured log entries on every job status change (job ID, status, pipeline type, batch size, elapsed ms)
  - Increment named counters and emit structured logs on every cache hit/miss (preprocessing, result, similarity)
  - Mark individual jobs "failed" on inference error without aborting the rest of the batch
  - _Requirements: 2.1, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3, 4.6, 5.1, 5.2, 5.3, 5.5, 10.2, 10.3, 11.1, 11.2, 11.3, 11.4, 11.5, 11.10, 11.12_

  - [ ]* 6.1 Write property tests for cache lookup ordering
    - **Property 5: Exact cache hit skips inference**
    - **Validates: Requirements 5.2**
    - **Property 12: Exact cache is checked before similarity cache**
    - **Validates: Requirements 11.5**

- [ ] 7. Checkpoint — ensure all unit and property tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement the API server
  - Create `worker/api_server.py` as a new FastAPI app
  - Implement `POST /api/try-on`: validate input, upload images to Cloudinary, insert Supabase job record (status=pending), call `enqueue_job`, return `{job_id, status: "pending"}` within 500ms; return HTTP 503 if queue is unavailable
  - Implement `GET /api/try-on/{try_on_id}`: read job status from Supabase; include `result_image_url` when status is "completed"
  - Implement `GET /api/health`: return queue depth (`get_queue_depth()`), active worker count, and cache hit rates as JSON
  - Accept optional `user_tier` and `webhook_url` fields in the try-on request
  - _Requirements: 1.1, 1.3, 1.4, 8.1, 8.4, 9.1, 9.5, 9.6, 10.4_

- [x] 9. Implement the webhook notifier
  - Create `worker/webhook_notifier.py` with `notify(job: JobRecord, max_retries=3, base_delay=1.0)`
  - POST JSON payload `{job_id, status, result_image_url, error}` to `job.webhook_url`
  - Retry up to 3 times with exponential backoff starting at `base_delay` seconds on non-2xx or network error
  - After exhausting retries, log failure with job ID (do not raise)
  - _Requirements: 9.2, 9.3, 9.4_

  - [ ]* 9.1 Write property tests for webhook delivery and retry count
    - **Property 9: Webhook delivery is attempted for every terminal job state**
    - **Validates: Requirements 9.2, 9.3**
    - **Property 10: Webhook retries exactly up to 3 times on failure**
    - **Validates: Requirements 9.4**

- [x] 10. Wire everything together and migrate from old main.py
  - Create `worker/__init__.py` exporting the Celery app so workers can be started with `celery -A worker.queue worker --queues tryon.priority,tryon.standard`
  - Verify `worker/api_server.py` is a drop-in replacement for `FrontEnd/fastAPI/main.py` (same route paths, same Supabase/Cloudinary integration)
  - Update any import references in the project that point to `FrontEnd/fastAPI/main.py` to use `worker/api_server.py`
  - _Requirements: 1.1, 1.2, 2.1, 2.6, 9.5_

- [ ] 11. Final checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- All property-based tests use [Hypothesis](https://hypothesis.readthedocs.io/) with a minimum of 100 iterations and are tagged with a comment referencing the design property number they validate
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation before proceeding to the next phase
