# Requirements Document

## Introduction

This feature replaces the current fire-and-forget FastAPI background task system with a production-grade
queue-based architecture for the Virtual Try-On AI application. The system currently processes one request
at a time with no concurrency control, no batching, and no caching, causing high latency and poor GPU
utilization. The improvements target: a persistent job queue, batched GPU inference, dedicated worker
processes, preprocessing result caching, flash attention, VAE tiling/slicing, result caching, priority
queuing for premium users, webhook-based status notifications, and smart caching for visually similar
images using perceptual hashing or embedding-based similarity to avoid redundant inference on near-duplicate inputs.

## Glossary

- **API_Server**: The FastAPI web process that receives HTTP requests and enqueues jobs.
- **Job_Queue**: A Redis-backed persistent queue (via Celery or RQ) that holds pending try-on jobs.
- **GPU_Worker**: A dedicated Python process that dequeues jobs, runs preprocessing and inference, and writes results.
- **AutoMasker**: The preprocessing component that runs DensePose + SCHP (ATR + LIP) to produce cloth-agnostic masks.
- **Preprocessing_Cache**: A Redis or disk-backed store that maps a hash of the user image to its DensePose/SCHP outputs.
- **Result_Cache**: A Redis or disk-backed store that maps a hash of (user_image, cloth_image, cloth_type) to the final result image URL.
- **Batch_Collector**: The component inside GPU_Worker that accumulates individual jobs into a batch before inference.
- **CatVTONPipeline**: The Stable Diffusion UNet + DDIM inference pipeline.
- **FluxTryOnPipeline**: The Flux transformer-based inference pipeline.
- **Inference_Engine**: The abstraction over CatVTONPipeline and FluxTryOnPipeline that executes batched inference.
- **Priority_Queue**: A queue variant that orders jobs by user tier (premium before standard).
- **Webhook_Notifier**: The component that sends HTTP callbacks to registered client URLs when a job completes or fails.
- **Job**: A single try-on request represented as a record with status, image URLs, cloth type, priority, and result.
- **User_Tier**: A classification of a user as "standard" or "premium" that determines queue priority.
- **Similarity_Cache**: A Redis-backed index that stores perceptual hash values or image embeddings alongside Result_Cache entries to enable approximate nearest-neighbor lookups for near-duplicate images.
- **Perceptual_Hash**: A compact fingerprint of an image (e.g., pHash or dHash) computed such that visually similar images produce hashes with a small Hamming distance, unlike cryptographic hashes which change entirely with minor pixel differences.
- **Similarity_Threshold**: A configurable numeric bound that determines when two images are considered visually similar — expressed as a maximum Hamming distance for perceptual hashing (default: 8) or a minimum cosine similarity for embedding-based comparison (default: 0.97).

---

## Requirements

### Requirement 1: Persistent Job Queue

**User Story:** As a backend engineer, I want all try-on requests to be enqueued in a persistent queue, so that no jobs are lost if the API server restarts and requests are processed in order.

#### Acceptance Criteria

1. WHEN a try-on request is received by the API_Server, THE API_Server SHALL enqueue a Job in the Job_Queue before returning a response to the client.
2. THE Job_Queue SHALL persist jobs to Redis so that jobs survive an API_Server restart.
3. WHEN the API_Server enqueues a Job, THE API_Server SHALL return the job ID and initial status "pending" to the client within 500ms.
4. IF the Job_Queue is unavailable when a request arrives, THEN THE API_Server SHALL return an HTTP 503 response with a descriptive error message.
5. THE Job_Queue SHALL support at least 1000 concurrently enqueued jobs without data loss.

---

### Requirement 2: Dedicated GPU Worker Processes

**User Story:** As a backend engineer, I want GPU inference to run in dedicated worker processes separate from the web process, so that inference does not block HTTP request handling and workers can be scaled independently.

#### Acceptance Criteria

1. THE GPU_Worker SHALL run as a process separate from the API_Server process.
2. WHEN a GPU_Worker starts, THE GPU_Worker SHALL load the Inference_Engine into GPU memory before polling the Job_Queue.
3. WHILE a GPU_Worker is processing a Job, THE GPU_Worker SHALL update the Job status to "processing" in the Supabase database.
4. WHEN a GPU_Worker completes a Job successfully, THE GPU_Worker SHALL update the Job status to "completed" and store the result image URL in the Supabase database.
5. IF a GPU_Worker encounters an unrecoverable error during inference, THEN THE GPU_Worker SHALL update the Job status to "failed" with an error message and continue processing the next Job.
6. THE system SHALL support running multiple GPU_Worker instances concurrently, each bound to a separate GPU device.

---

### Requirement 3: Batched Inference

**User Story:** As a backend engineer, I want the GPU worker to group multiple pending jobs into a single batched inference call, so that GPU utilization is maximized and per-request latency is reduced under load.

#### Acceptance Criteria

1. THE Batch_Collector SHALL accumulate jobs from the Job_Queue up to a configurable maximum batch size (default: 4) or until a configurable wait timeout (default: 100ms) elapses, whichever comes first.
2. WHEN the Batch_Collector has collected at least one job, THE Inference_Engine SHALL execute a single forward pass with all collected jobs as a batch.
3. THE Inference_Engine SHALL support batch sizes from 1 to the configured maximum without error.
4. WHEN a batch contains N jobs, THE Inference_Engine SHALL produce exactly N result images in the same order as the input jobs.
5. IF a single job in a batch causes an inference error, THEN THE GPU_Worker SHALL mark only that job as "failed" and return results for the remaining jobs in the batch.
6. WHERE the FluxTryOnPipeline is selected, THE Inference_Engine SHALL set the pipeline batch_size parameter to the number of jobs in the current batch.

---

### Requirement 4: Preprocessing Cache (DensePose / SCHP)

**User Story:** As a backend engineer, I want DensePose and SCHP results to be cached per user image, so that repeated try-on requests with the same person image skip expensive preprocessing.

#### Acceptance Criteria

1. WHEN the GPU_Worker is about to run AutoMasker on a user image, THE Preprocessing_Cache SHALL be checked first using a SHA-256 hash of the user image bytes as the cache key.
2. WHEN a Preprocessing_Cache hit occurs, THE GPU_Worker SHALL use the cached DensePose and SCHP outputs and skip calling AutoMasker.
3. WHEN a Preprocessing_Cache miss occurs, THE GPU_Worker SHALL run AutoMasker, store the outputs in the Preprocessing_Cache keyed by the image hash, and then proceed with inference.
4. THE Preprocessing_Cache SHALL store entries with a configurable TTL (default: 24 hours).
5. THE Preprocessing_Cache SHALL serialize and deserialize DensePose and SCHP mask outputs as lossless PNG bytes to preserve mask accuracy.
6. IF the Preprocessing_Cache is unavailable, THEN THE GPU_Worker SHALL run AutoMasker without caching and log a warning.

---

### Requirement 5: Result Cache

**User Story:** As a backend engineer, I want identical try-on requests to return a cached result immediately, so that duplicate requests do not consume GPU resources.

#### Acceptance Criteria

1. WHEN a Job is dequeued, THE GPU_Worker SHALL compute a SHA-256 hash of the concatenation of (user_image_bytes, cloth_image_bytes, cloth_type) as the result cache key.
2. WHEN a Result_Cache hit occurs, THE GPU_Worker SHALL mark the Job as "completed" with the cached result image URL without running inference.
3. WHEN a Result_Cache miss occurs and inference completes successfully, THE GPU_Worker SHALL store the result image URL in the Result_Cache keyed by the computed hash.
4. THE Result_Cache SHALL store entries with a configurable TTL (default: 7 days).
5. IF the Result_Cache is unavailable, THEN THE GPU_Worker SHALL proceed with inference without caching and log a warning.

---

### Requirement 6: Flash Attention

**User Story:** As a backend engineer, I want flash attention enabled in the UNet and Flux transformer, so that attention computation is faster and uses less GPU memory.

#### Acceptance Criteria

1. WHEN the GPU_Worker initializes the CatVTONPipeline, THE Inference_Engine SHALL enable xformers memory-efficient attention on the UNet if xformers is installed.
2. WHEN the GPU_Worker initializes the FluxTryOnPipeline, THE Inference_Engine SHALL enable flash attention on the Flux transformer if the flash-attn package is installed.
3. IF xformers or flash-attn is not installed, THEN THE Inference_Engine SHALL fall back to the default PyTorch attention implementation and log an informational message.
4. THE Inference_Engine SHALL expose a configuration flag to disable flash attention without code changes.

---

### Requirement 7: VAE Tiling and Slicing

**User Story:** As a backend engineer, I want VAE tiling and slicing enabled for large-resolution inputs, so that GPU memory usage during VAE encode/decode is reduced and out-of-memory errors are avoided.

#### Acceptance Criteria

1. WHEN the GPU_Worker initializes the FluxTryOnPipeline, THE Inference_Engine SHALL call enable_vae_tiling() on the pipeline if the configured input resolution exceeds 1024px on either dimension.
2. WHEN the GPU_Worker initializes the FluxTryOnPipeline with batch_size greater than 1, THE Inference_Engine SHALL call enable_vae_slicing() on the pipeline.
3. THE Inference_Engine SHALL expose configuration flags for vae_tiling and vae_slicing that can be set independently.
4. IF enabling VAE tiling causes a runtime error, THEN THE Inference_Engine SHALL disable tiling, log the error, and retry the inference step without tiling.

---

### Requirement 8: Priority Queue

**User Story:** As a product manager, I want premium users' try-on jobs to be processed before standard users' jobs, so that premium subscribers experience lower wait times.

#### Acceptance Criteria

1. WHEN a Job is enqueued, THE API_Server SHALL assign a priority value based on the User_Tier: premium jobs SHALL receive priority 1 and standard jobs SHALL receive priority 10.
2. WHEN the Batch_Collector selects jobs from the Job_Queue, THE Batch_Collector SHALL select all available priority-1 jobs before selecting priority-10 jobs.
3. THE Priority_Queue SHALL guarantee that a standard job is not starved indefinitely; WHEN a standard job has been waiting longer than a configurable maximum wait time (default: 300 seconds), THE Priority_Queue SHALL promote the job to priority 1.
4. THE API_Server SHALL accept an optional user_tier field in the try-on request payload to set the job priority.

---

### Requirement 9: Webhook and Polling Status Notifications

**User Story:** As a frontend developer, I want to receive a webhook callback when a try-on job finishes, so that the frontend does not need to poll repeatedly and can update the UI immediately.

#### Acceptance Criteria

1. WHEN a try-on request is submitted with an optional webhook_url field, THE API_Server SHALL store the webhook_url alongside the Job record.
2. WHEN a Job transitions to "completed" or "failed", THE Webhook_Notifier SHALL send an HTTP POST request to the stored webhook_url within 5 seconds of the status change.
3. THE Webhook_Notifier SHALL include the job ID, final status, result image URL (if completed), and error message (if failed) in the POST request body as JSON.
4. IF the webhook POST request fails with a non-2xx response or a network error, THEN THE Webhook_Notifier SHALL retry the request up to 3 times with exponential backoff starting at 1 second.
5. THE existing GET /api/try-on/{try_on_id} polling endpoint SHALL remain available and return the current job status from the Supabase database.
6. WHEN a client polls GET /api/try-on/{try_on_id} and the job status is "completed", THE API_Server SHALL include the result_image_url in the response body.

---

### Requirement 10: Observability and Configuration

**User Story:** As a backend engineer, I want all queue, worker, and cache components to be configurable via environment variables and to emit structured logs, so that the system can be tuned and debugged in production.

#### Acceptance Criteria

1. THE system SHALL read all tunable parameters (batch size, batch wait timeout, cache TTLs, max queue depth, worker count, priority wait threshold, flash attention flags, VAE tiling flags) from environment variables at startup.
2. WHEN a Job changes status, THE GPU_Worker SHALL emit a structured log entry containing the job ID, status, pipeline type, batch size, and elapsed time in milliseconds.
3. WHEN the Preprocessing_Cache or Result_Cache is hit or missed, THE GPU_Worker SHALL increment a named counter and emit a structured log entry.
4. THE API_Server SHALL expose a GET /api/health endpoint that returns the Job_Queue depth, number of active GPU_Worker instances, and cache hit rates as JSON.
5. IF any required environment variable is missing at startup, THEN THE system SHALL log the missing variable name and exit with a non-zero status code.

---

### Requirement 11: Smart Similarity Cache

**User Story:** As a backend engineer, I want near-duplicate try-on requests (same person with minor lighting changes, compression artifacts, or slight crops) to return a cached result without running inference, so that GPU resources are not wasted on visually identical inputs that differ only at the pixel level.

#### Acceptance Criteria

1. WHEN a Job is dequeued and a Result_Cache miss occurs, THE GPU_Worker SHALL compute a Perceptual_Hash for both the user image and the cloth image and query the Similarity_Cache for entries whose stored hashes fall within the configured Similarity_Threshold.
2. WHEN the Similarity_Cache contains an entry where the user image hash distance is within the Similarity_Threshold AND the cloth image hash distance is within the Similarity_Threshold, THE GPU_Worker SHALL treat the lookup as a similarity cache hit and return the cached result image URL without running inference.
3. WHEN a similarity cache hit occurs, THE GPU_Worker SHALL mark the Job as "completed" with the matched cached result image URL and log the Hamming distances (or cosine similarities) of both the user image and cloth image matches.
4. WHEN a Result_Cache miss and a Similarity_Cache miss both occur and inference completes successfully, THE GPU_Worker SHALL store the Perceptual_Hash of the user image and the cloth image alongside the result entry in the Similarity_Cache.
5. THE GPU_Worker SHALL check the exact Result_Cache (Requirement 5) before querying the Similarity_Cache; the similarity lookup SHALL only be performed after an exact hash miss.
6. THE Similarity_Cache SHALL support two configurable similarity methods: perceptual hashing with Hamming distance comparison, and CLIP embedding vectors with cosine similarity comparison; the active method SHALL be selected via an environment variable.
7. WHERE the perceptual hashing method is selected, THE GPU_Worker SHALL compute pHash values as 64-bit integers and compare them using Hamming distance with a configurable maximum distance (default: 8).
8. WHERE the CLIP embedding method is selected, THE GPU_Worker SHALL compute 512-dimensional CLIP image embeddings and compare them using cosine similarity with a configurable minimum similarity score (default: 0.97).
9. THE Similarity_Cache SHALL store perceptual hash values or embedding vectors in Redis alongside the corresponding result cache key, with the same configurable TTL as the Result_Cache (default: 7 days).
10. IF the Similarity_Cache is unavailable, THEN THE GPU_Worker SHALL skip the similarity lookup, proceed with inference, and log a warning.
11. THE system SHALL read the similarity method, Similarity_Threshold value, and embedding model path from environment variables at startup.
12. WHEN the Similarity_Cache is hit or missed, THE GPU_Worker SHALL increment a named counter and emit a structured log entry containing the job ID, similarity method used, and the computed distance or similarity score for both images.
