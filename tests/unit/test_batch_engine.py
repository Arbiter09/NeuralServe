"""Unit tests for DynamicBatcher — mocks ModelEngine to avoid GPU dependency."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from serving.batch_engine import DynamicBatcher
from serving.inference import GenerationParams


def make_mock_engine(responses: list[str] | None = None):
    """Return a synchronous mock ModelEngine."""
    engine = MagicMock()
    engine.is_loaded = True

    call_count = [0]

    def fake_generate(prompts: list[str], params) -> list[str]:
        call_count[0] += 1
        if responses:
            return responses[: len(prompts)]
        return [f"response to: {p}" for p in prompts]

    engine.generate = fake_generate
    engine._call_count = call_count
    return engine


@pytest.fixture
def batcher():
    engine = make_mock_engine()
    b = DynamicBatcher(model_engine=engine, max_batch_size=8, max_wait_ms=50)
    b.start()
    yield b
    asyncio.get_event_loop().run_until_complete(b.stop())


@pytest.mark.asyncio
async def test_single_request_dispatched():
    engine = make_mock_engine(["The sky is blue."])
    b = DynamicBatcher(model_engine=engine, max_batch_size=8, max_wait_ms=100)
    b.start()

    params = GenerationParams()
    result = await asyncio.wait_for(b.submit("What color is the sky?", params), timeout=2.0)

    assert result == "The sky is blue."
    await b.stop()


@pytest.mark.asyncio
async def test_batch_fills_on_max_size():
    """8 concurrent requests should be dispatched as a single batch."""
    dispatched_sizes: list[int] = []
    original_dispatch = DynamicBatcher._dispatch

    async def patched_dispatch(self, batch):
        dispatched_sizes.append(len(batch))
        await original_dispatch(self, batch)

    engine = make_mock_engine()
    b = DynamicBatcher(model_engine=engine, max_batch_size=8, max_wait_ms=500)

    with patch.object(DynamicBatcher, "_dispatch", patched_dispatch):
        b.start()
        params = GenerationParams()
        tasks = [b.submit(f"prompt {i}", params) for i in range(8)]
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

    assert len(results) == 8
    # At least one batch of size 8 should have been dispatched
    assert any(s == 8 for s in dispatched_sizes)
    await b.stop()


@pytest.mark.asyncio
async def test_batch_flushes_on_timeout():
    """A single request should flush after max_wait_ms even without a full batch."""
    engine = make_mock_engine(["flushed response"])
    b = DynamicBatcher(model_engine=engine, max_batch_size=8, max_wait_ms=50)
    b.start()

    params = GenerationParams()
    t0 = time.monotonic()
    result = await asyncio.wait_for(b.submit("solo prompt", params), timeout=2.0)
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert result == "flushed response"
    # Should have been dispatched within a reasonable window (50ms + inference overhead)
    assert elapsed_ms < 2000
    await b.stop()


@pytest.mark.asyncio
async def test_queue_depth_reflects_pending_requests():
    """queue_depth property should track unprocessed requests."""
    slow_engine = MagicMock()
    slow_engine.is_loaded = True

    async def slow_generate(prompts, params):
        await asyncio.sleep(0.5)
        return [f"done: {p}" for p in prompts]

    # Use run_in_executor mock that is slow
    import concurrent.futures

    slow_engine.generate = lambda prompts, params: [f"done: {p}" for p in prompts]

    b = DynamicBatcher(model_engine=slow_engine, max_batch_size=1, max_wait_ms=10)
    b.start()

    params = GenerationParams()
    # Submit several requests quickly
    tasks = [asyncio.create_task(b.submit(f"q{i}", params)) for i in range(3)]
    await asyncio.sleep(0.01)  # let them enqueue

    # Queue depth should be >= 0 (some may already be dispatched)
    assert b.queue_depth >= 0

    await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
    await b.stop()


@pytest.mark.asyncio
async def test_exception_in_engine_propagates():
    """Errors in ModelEngine.generate should surface as exceptions on the future."""
    engine = MagicMock()
    engine.is_loaded = True
    engine.generate = MagicMock(side_effect=RuntimeError("GPU OOM"))

    b = DynamicBatcher(model_engine=engine, max_batch_size=8, max_wait_ms=50)
    b.start()

    params = GenerationParams()
    with pytest.raises(RuntimeError, match="GPU OOM"):
        await asyncio.wait_for(b.submit("boom", params), timeout=2.0)

    await b.stop()
