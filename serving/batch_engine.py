"""DynamicBatcher — collects requests into micro-batches for efficient GPU utilization."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PendingRequest:
    request_id: str
    prompts: list[str]
    params: Any  # GenerationParams
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    enqueue_time: float = field(default_factory=time.monotonic)


class DynamicBatcher:
    """Collects inference requests into batches for efficient GPU utilization.

    Waits up to max_wait_ms milliseconds or until max_batch_size requests
    are queued — whichever comes first — before dispatching to ModelEngine.
    """

    def __init__(
        self,
        model_engine,
        max_batch_size: int = 8,
        max_wait_ms: int = 50,
    ) -> None:
        self.engine = model_engine
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self._queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Rolling stats
        self._last_batch_size: int = 0
        self._latency_window: list[float] = []
        self._window_size = 50

    def start(self) -> None:
        self._running = True
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._batch_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, prompt: str, params) -> Any:
        """Enqueue a single generation request and await its result."""
        request_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        req = PendingRequest(
            request_id=request_id,
            prompts=[prompt],
            params=params,
            future=loop.create_future(),
        )

        try:
            from observability.metrics import QUEUE_DEPTH

            QUEUE_DEPTH.inc()
        except Exception:
            pass

        await self._queue.put(req)
        return await req.future

    async def _batch_loop(self) -> None:
        """Background coroutine that drains the queue and dispatches batches."""
        while self._running:
            batch: list[PendingRequest] = []
            deadline = time.monotonic() + self.max_wait_ms / 1000.0

            # Collect the first request (blocking wait)
            try:
                first = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self.max_wait_ms / 1000.0,
                )
                batch.append(first)
            except asyncio.TimeoutError:
                continue

            # Drain additional requests within deadline or until batch full
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            await self._dispatch(batch)

    async def _dispatch(self, batch: list[PendingRequest]) -> None:
        """Run inference on a collected batch and resolve all futures."""
        if not batch:
            return

        prompts = [req.prompts[0] for req in batch]
        params = batch[0].params  # use first request's params
        t0 = time.monotonic()

        try:
            from observability.metrics import BATCH_SIZE, QUEUE_DEPTH, TOKEN_THROUGHPUT

            BATCH_SIZE.observe(len(batch))
            for _ in batch:
                QUEUE_DEPTH.dec()
        except Exception:
            pass

        try:
            loop = asyncio.get_event_loop()
            results: list[str] = await loop.run_in_executor(
                None, self.engine.generate, prompts, params
            )
        except Exception as exc:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)
            return

        latency_ms = (time.monotonic() - t0) * 1000
        self._last_batch_size = len(batch)
        self._latency_window.append(latency_ms)
        if len(self._latency_window) > self._window_size:
            self._latency_window.pop(0)

        try:
            from observability.metrics import TOKEN_THROUGHPUT

            total_tokens = sum(len(r.split()) for r in results)
            TOKEN_THROUGHPUT.inc(total_tokens)
        except Exception:
            pass

        for req, result in zip(batch, results):
            if not req.future.done():
                req.future.set_result(result)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def avg_latency_ms(self) -> float:
        if not self._latency_window:
            return 0.0
        return sum(self._latency_window) / len(self._latency_window)
