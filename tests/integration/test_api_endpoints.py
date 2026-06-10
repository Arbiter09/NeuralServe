"""Integration tests for the FastAPI endpoints — uses mocked ModelEngine and RedisCache."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from serving.batch_engine import DynamicBatcher
from serving.cache import RedisCache
from serving.inference import GenerationParams, ModelEngine
from serving.router import router


def make_test_engine() -> ModelEngine:
    engine = MagicMock(spec=ModelEngine)
    engine.is_loaded = True
    engine.generate = MagicMock(return_value=["Gradient descent minimizes loss iteratively."])
    engine.cache_key_for_params = MagicMock(return_value="test-cache-key")
    return engine


def make_test_cache() -> RedisCache:
    cache = MagicMock(spec=RedisCache)
    cache.is_connected = True
    cache.get = AsyncMock(return_value=None)  # cache miss by default
    cache.set = AsyncMock(return_value=None)
    cache.make_key = RedisCache.make_key
    return cache


@pytest.fixture
def app():
    """Build a minimal FastAPI test app with mocked dependencies."""
    engine = make_test_engine()
    cache = make_test_cache()

    batcher = DynamicBatcher(model_engine=engine, max_batch_size=8, max_wait_ms=50)
    batcher.start()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.start_time = time.monotonic()
        app.state.engine = engine
        app.state.cache = cache
        app.state.batcher = batcher
        yield
        await batcher.stop()

    test_app = FastAPI(lifespan=lifespan)
    test_app.include_router(router)
    return test_app


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint_returns_200(client):
    response = await client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "model_loaded" in data
    assert "cache_connected" in data
    assert "uptime_seconds" in data


@pytest.mark.asyncio
async def test_generate_endpoint_returns_response(client):
    payload = {"prompt": "Explain gradient descent in simple terms.", "max_new_tokens": 50}
    response = await client.post("/v1/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "generated_text" in data
    assert len(data["generated_text"]) > 0
    assert "tokens_generated" in data
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_generate_missing_prompt_returns_422(client):
    response = await client.post("/v1/generate", json={"max_new_tokens": 100})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_generate_endpoint_cache_hit_on_repeat_request(app):
    """Second identical request should be served from cache."""
    engine = app.state if hasattr(app, "state") else None

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # First request — cache miss (default mock)
        payload = {"prompt": "What is LoRA?", "max_new_tokens": 50}
        r1 = await ac.post("/v1/generate", json=payload)
        assert r1.status_code == 200
        assert r1.json()["cached"] is False

        # Patch cache.get to return a hit
        cached_text = "LoRA is Low-Rank Adaptation for LLMs."
        app.state.cache.get = AsyncMock(return_value=cached_text)

        r2 = await ac.post("/v1/generate", json=payload)
        assert r2.status_code == 200
        assert r2.json()["cached"] is True
        assert r2.json()["generated_text"] == cached_text


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_format(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/plain" in content_type or "openmetrics" in content_type
    # Prometheus text format should start with a comment or metric name
    body = response.text
    assert len(body) > 0


@pytest.mark.asyncio
async def test_batch_stats_endpoint(client):
    response = await client.get("/v1/metrics/batch")
    assert response.status_code == 200
    data = response.json()
    assert "batch_size" in data
    assert "queue_depth" in data
    assert "avg_latency_ms" in data
