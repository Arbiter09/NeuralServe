"""FastAPI application factory for NeuralServe inference server."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator


def create_app():
    """Application factory — creates and configures the FastAPI instance.

    Singletons (ModelEngine, RedisCache, DynamicBatcher) are attached to
    app.state so they are shared across requests without globals.
    """
    import asyncio

    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware

    from serving.batch_engine import DynamicBatcher
    from serving.cache import RedisCache
    from serving.inference import ModelEngine
    from serving.router import router

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # ── Startup ───────────────────────────────────────────────────────────
        app.state.start_time = time.monotonic()

        # Tracing
        otlp_endpoint = os.getenv("OTLP_ENDPOINT", "")
        try:
            from observability.tracing import setup_tracing

            setup_tracing(
                service_name=os.getenv("SERVICE_NAME", "neuralserve"),
                otlp_endpoint=otlp_endpoint or None,
            )
        except Exception as exc:
            print(f"[app] Tracing setup skipped: {exc}")

        # Redis cache
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        cache = RedisCache(redis_url=redis_url)
        await cache.connect()
        app.state.cache = cache

        # Model engine
        model_path = os.getenv("MODEL_PATH", "meta-llama/Meta-Llama-3.1-8B")
        adapter_path = os.getenv("ADAPTER_PATH", None)
        device = os.getenv("DEVICE", "cuda")
        engine = ModelEngine(model_path=model_path, adapter_path=adapter_path, device=device)
        engine.load()
        app.state.engine = engine

        # Batch engine
        max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "8"))
        max_wait_ms = int(os.getenv("MAX_WAIT_MS", "50"))
        batcher = DynamicBatcher(
            model_engine=engine,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
        )
        batcher.start()
        app.state.batcher = batcher

        # GPU monitor background task
        try:
            from observability.metrics import gpu_monitor_loop

            asyncio.create_task(gpu_monitor_loop())
        except Exception:
            pass

        yield

        # ── Shutdown ──────────────────────────────────────────────────────────
        await batcher.stop()
        await cache.disconnect()

    app = FastAPI(
        title="NeuralServe",
        description="Production LLM fine-tuning and inference serving with LoRA and dynamic batching",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timeout middleware (30s)
    from starlette.middleware.base import BaseHTTPMiddleware

    class TimeoutMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            import asyncio

            try:
                return await asyncio.wait_for(call_next(request), timeout=30.0)
            except asyncio.TimeoutError:
                from fastapi.responses import JSONResponse

                return JSONResponse({"detail": "Request timeout"}, status_code=504)

    app.add_middleware(TimeoutMiddleware)

    # ── OpenTelemetry FastAPI instrumentation ─────────────────────────────────
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        pass

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(router)

    return app
