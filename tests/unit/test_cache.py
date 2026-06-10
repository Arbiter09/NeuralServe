"""Unit tests for RedisCache using fakeredis for isolation."""

from __future__ import annotations

import asyncio

import pytest

from serving.cache import RedisCache


@pytest.fixture
def cache(monkeypatch):
    """Return a RedisCache wired to fakeredis (no real Redis required)."""
    import fakeredis.aioredis as fake_aioredis

    fake_client = fake_aioredis.FakeRedis(decode_responses=True)

    c = RedisCache(redis_url="redis://localhost:6379")
    c._client = fake_client
    c._connected = True
    return c


@pytest.mark.asyncio
async def test_cache_miss_returns_none(cache):
    result = await cache.get("nonexistent-key")
    assert result is None


@pytest.mark.asyncio
async def test_cache_set_and_get_roundtrip(cache):
    await cache.set("key1", "hello world", ttl=60)
    result = await cache.get("key1")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_cache_key_deterministic():
    """Same prompt + params must always produce the same cache key."""
    params = {"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9}
    key1 = RedisCache.make_key("Explain gradient descent.", params)
    key2 = RedisCache.make_key("Explain gradient descent.", params)
    assert key1 == key2
    assert len(key1) == len("ns:gen:") + 64  # SHA-256 hex = 64 chars


@pytest.mark.asyncio
async def test_cache_key_differs_for_different_prompts():
    params = {"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9}
    k1 = RedisCache.make_key("prompt A", params)
    k2 = RedisCache.make_key("prompt B", params)
    assert k1 != k2


@pytest.mark.asyncio
async def test_cache_key_differs_for_different_params():
    prompt = "Same prompt"
    k1 = RedisCache.make_key(prompt, {"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9})
    k2 = RedisCache.make_key(prompt, {"max_new_tokens": 128, "temperature": 0.7, "top_p": 0.9})
    assert k1 != k2


@pytest.mark.asyncio
async def test_cache_ttl_expiry(cache):
    """Value stored with very short TTL should not persist."""
    await cache.set("expires-fast", "temporary", ttl=1)
    result_before = await cache.get("expires-fast")
    assert result_before == "temporary"

    await asyncio.sleep(1.1)
    result_after = await cache.get("expires-fast")
    assert result_after is None


@pytest.mark.asyncio
async def test_cache_disconnected_returns_none():
    """Cache with no connection should silently return None."""
    c = RedisCache(redis_url="redis://nonexistent:9999")
    c._connected = False
    result = await c.get("any-key")
    assert result is None


@pytest.mark.asyncio
async def test_cache_disconnected_set_is_noop():
    """Cache.set on disconnected cache should not raise."""
    c = RedisCache(redis_url="redis://nonexistent:9999")
    c._connected = False
    await c.set("key", "value")  # should not raise
