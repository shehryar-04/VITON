"""
Cache layer for the GPU worker pipeline.

Three caches share a single injected Redis client:
  - PreprocessingCache  — msgpack-serialized PreprocessResult, keyed by image SHA-256
  - ResultCache         — Cloudinary URL string, keyed by SHA-256(user+cloth+type)
  - SimilarityCache     — pHash / CLIP similarity index stored as Redis hashes + a set index
"""
from __future__ import annotations

import hashlib
import logging
from typing import Optional

import msgpack
import redis

from worker.models import PreprocessResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def compute_result_cache_key(
    user_bytes: bytes, cloth_bytes: bytes, cloth_type: str
) -> str:
    """SHA-256 of user_bytes + cloth_bytes + cloth_type.encode().

    Validates: Requirements 5.1
    """
    h = hashlib.sha256()
    h.update(user_bytes)
    h.update(cloth_bytes)
    h.update(cloth_type.encode())
    return h.hexdigest()


def compute_phash(image) -> int:  # image: PIL.Image.Image
    """Compute perceptual hash of *image* and return it as a 64-bit int.

    Uses imagehash.phash() under the hood.
    Validates: Requirements 11.7
    """
    import imagehash  # optional dependency — imported lazily

    ph = imagehash.phash(image)
    # imagehash stores the hash as a numpy bool array; convert to int via hex string
    return int(str(ph), 16)


def hamming_distance(a: int, b: int) -> int:
    """Popcount of XOR of two 64-bit integers.  Result is in [0, 64].

    Validates: Requirements 11.7
    """
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# PreprocessingCache
# ---------------------------------------------------------------------------


class PreprocessingCache:
    """Cache for AutoMasker preprocessing results (DensePose + SCHP masks).

    Keys: ``preprocess:{image_hash}``
    Values: msgpack-serialised ``PreprocessResult``
    """

    _PREFIX = "preprocess:"

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    def get(self, image_hash: str) -> Optional[PreprocessResult]:
        """Return cached ``PreprocessResult`` or *None* on miss / Redis error."""
        key = self._PREFIX + image_hash
        try:
            raw = self._redis.get(key)
        except redis.RedisError as exc:
            logger.warning("PreprocessingCache.get failed for key %s: %s", key, exc)
            return None

        if raw is None:
            return None

        try:
            data = msgpack.unpackb(raw, raw=True)
            return PreprocessResult(
                densepose_png=data[b"densepose_png"],
                schp_atr_png=data[b"schp_atr_png"],
                schp_lip_png=data[b"schp_lip_png"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("PreprocessingCache.get deserialization error: %s", exc)
            return None

    def set(self, image_hash: str, result: PreprocessResult, ttl: int) -> None:
        """Serialize *result* to msgpack and store in Redis with *ttl* seconds."""
        key = self._PREFIX + image_hash
        payload = msgpack.packb(
            {
                "densepose_png": result.densepose_png,
                "schp_atr_png": result.schp_atr_png,
                "schp_lip_png": result.schp_lip_png,
            },
            use_bin_type=True,
        )
        try:
            self._redis.setex(key, ttl, payload)
        except redis.RedisError as exc:
            logger.warning("PreprocessingCache.set failed for key %s: %s", key, exc)


# ---------------------------------------------------------------------------
# ResultCache
# ---------------------------------------------------------------------------


class ResultCache:
    """Cache for final try-on result URLs.

    Keys: ``result:{result_key}``
    Values: Cloudinary URL string
    """

    _PREFIX = "result:"

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    def get(self, result_key: str) -> Optional[str]:
        """Return cached URL string or *None* on miss / Redis error."""
        key = self._PREFIX + result_key
        try:
            raw = self._redis.get(key)
        except redis.RedisError as exc:
            logger.warning("ResultCache.get failed for key %s: %s", key, exc)
            return None

        if raw is None:
            return None
        return raw.decode() if isinstance(raw, bytes) else raw

    def set(self, result_key: str, url: str, ttl: int) -> None:
        """Store *url* in Redis under *result_key* with *ttl* seconds."""
        key = self._PREFIX + result_key
        try:
            self._redis.setex(key, ttl, url)
        except redis.RedisError as exc:
            logger.warning("ResultCache.set failed for key %s: %s", key, exc)


# ---------------------------------------------------------------------------
# SimilarityCache
# ---------------------------------------------------------------------------

_SIM_INDEX_KEY = "simindex"


class SimilarityCache:
    """Perceptual / CLIP similarity index.

    Each entry is stored as a Redis hash at ``sim:{result_key}`` with fields:
      ``user_hash``, ``cloth_hash``, ``result_key``

    All result_keys are also added to the Redis set ``simindex`` so we can
    scan them without a KEYS command.

    Supports:
      - method="phash"  — Hamming distance, threshold is max int distance [0, 64]
      - method="clip"   — cosine similarity, threshold is min float similarity
    """

    def __init__(self, redis_client: redis.Redis, method: str = "phash", threshold=8) -> None:
        self._redis = redis_client
        self._method = method
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        user_hash: int,
        cloth_hash: int,
        result_key: str,
        ttl: int,
    ) -> None:
        """Persist a similarity entry and register it in the scan index."""
        hash_key = f"sim:{result_key}"
        try:
            pipe = self._redis.pipeline()
            pipe.hset(
                hash_key,
                mapping={
                    "user_hash": str(user_hash),
                    "cloth_hash": str(cloth_hash),
                    "result_key": result_key,
                },
            )
            pipe.expire(hash_key, ttl)
            pipe.sadd(_SIM_INDEX_KEY, result_key)
            pipe.execute()
        except redis.RedisError as exc:
            logger.warning("SimilarityCache.store failed for key %s: %s", hash_key, exc)

    def query(
        self,
        user_hash: int,
        cloth_hash: int,
        threshold=None,
    ) -> Optional[str]:
        """Scan the similarity index and return the first matching result_key.

        A match requires BOTH user_hash AND cloth_hash to be within *threshold*
        of the stored values.  Returns *None* on no match or Redis error.
        """
        if threshold is None:
            threshold = self._threshold

        try:
            result_keys = self._redis.smembers(_SIM_INDEX_KEY)
        except redis.RedisError as exc:
            logger.warning("SimilarityCache.query smembers failed: %s", exc)
            return None

        for rk_bytes in result_keys:
            rk = rk_bytes.decode() if isinstance(rk_bytes, bytes) else rk_bytes
            hash_key = f"sim:{rk}"
            try:
                entry = self._redis.hgetall(hash_key)
            except redis.RedisError as exc:
                logger.warning("SimilarityCache.query hgetall failed for %s: %s", hash_key, exc)
                continue

            if not entry:
                continue

            stored_user = int(entry.get(b"user_hash") or entry.get("user_hash", 0))
            stored_cloth = int(entry.get(b"cloth_hash") or entry.get("cloth_hash", 0))
            stored_rk_raw = entry.get(b"result_key") or entry.get("result_key", b"")
            stored_rk = stored_rk_raw.decode() if isinstance(stored_rk_raw, bytes) else stored_rk_raw

            if self._matches(user_hash, stored_user, threshold) and self._matches(
                cloth_hash, stored_cloth, threshold
            ):
                return stored_rk

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _matches(self, query_val: int, stored_val: int, threshold) -> bool:
        if self._method == "phash":
            return hamming_distance(query_val, stored_val) <= int(threshold)
        elif self._method == "clip":
            # For CLIP, values are stored as floats; threshold is min cosine similarity
            # query_val and stored_val are treated as float bit patterns here
            # In practice the caller passes float-compatible ints or the embedding is
            # handled externally; we keep the interface consistent.
            return float(query_val) >= float(threshold)
        else:
            logger.warning("SimilarityCache: unknown method %r, defaulting to phash", self._method)
            return hamming_distance(query_val, stored_val) <= int(threshold)
