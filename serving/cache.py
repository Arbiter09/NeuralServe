"""Redis KV-cache layer with LFU eviction and Prometheus hit-rate tracking."""

from __future__ import annotations

import hashlib
import json
from typing import Optional


def _make_cache_key(prompt: str, params: dict) -> str:
    """SHA-256 hash of prompt + generation parameters for deterministic cache keys."""
    raw = json.dumps({"prompt": prompt, **params}, sort_keys=True)
    return f"ns:gen:{hashlib.sha256(raw.encode()).hexdigest()}"


class RedisCache:
    """Async Redis cache for generated text responses.

    Uses Redis LFU eviction (maxmemory-policy = allkeys-lfu) configured at
    the Redis server level (see infra/docker-compose.yml).
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self.redis_url = redis_url
        self._client = None
        self._connected = False

    async def connect(self) -> None:
        """Initialize async Redis connection pool."""
        import redis.asyncio as aioredis

        self._client = aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        try:
            await self._client.ping()
            self._connected = True
            print(f"[RedisCache] Connected to {self.redis_url}")
        except Exception as exc:
            self._connected = False
            print(f"[RedisCache] Warning: could not connect to Redis — {exc}")

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def get(self, key: str) -> Optional[str]:
        """Return cached value or None on miss."""
        if not self._connected or not self._client:
            return None
        try:
            value = await self._client.get(key)
            if value is not None:
                try:
                    from observability.metrics import CACHE_HITS

                    CACHE_HITS.inc()
                except Exception:
                    pass
            else:
                try:
                    from observability.metrics import CACHE_MISSES

                    CACHE_MISSES.inc()
                except Exception:
                    pass
            return value
        except Exception:
            return None

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        """Store a value with TTL expiry."""
        if not self._connected or not self._client:
            return
        try:
            await self._client.set(key, value, ex=ttl)
        except Exception:
            pass

    async def cache_hit_rate(self) -> float:
        """Return current cache hit rate based on Redis INFO stats."""
        if not self._connected or not self._client:
            return 0.0
        try:
            info = await self._client.info("stats")
            hits = int(info.get("keyspace_hits", 0))
            misses = int(info.get("keyspace_misses", 0))
            total = hits + misses
            return hits / total if total > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def make_key(prompt: str, params: dict) -> str:
        """Public helper for deterministic cache key generation."""
        return _make_cache_key(prompt, params)
