"""FastAPI route definitions for the NeuralServe inference API."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from serving.schemas import BatchStats, GenerateRequest, GenerateResponse, HealthResponse

router = APIRouter()


@router.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
    """Submit a prompt for text generation.

    Checks Redis cache first; on miss, routes through DynamicBatcher to ModelEngine.
    """
    t0 = time.monotonic()
    request_id = str(uuid.uuid4())

    state = request.app.state
    cache = state.cache
    batcher = state.batcher

    try:
        from observability.metrics import REQUEST_COUNTER, REQUEST_LATENCY

        REQUEST_COUNTER.labels(status="received").inc()
    except Exception:
        pass

    # ── Cache lookup ──────────────────────────────────────────────────────────
    params_dict = {
        "max_new_tokens": body.max_new_tokens,
        "temperature": body.temperature,
        "top_p": body.top_p,
    }
    cache_key = cache.make_key(body.prompt, params_dict)
    cached_text = await cache.get(cache_key)

    if cached_text is not None:
        latency_ms = (time.monotonic() - t0) * 1000
        tokens = len(cached_text.split())

        try:
            from observability.metrics import REQUEST_COUNTER, REQUEST_LATENCY, TOKEN_THROUGHPUT

            REQUEST_COUNTER.labels(status="cache_hit").inc()
            REQUEST_LATENCY.observe(latency_ms / 1000)
            TOKEN_THROUGHPUT.inc(tokens)
        except Exception:
            pass

        return GenerateResponse(
            request_id=request_id,
            generated_text=cached_text,
            tokens_generated=tokens,
            latency_ms=round(latency_ms, 2),
            cached=True,
        )

    # ── Inference via DynamicBatcher ──────────────────────────────────────────
    from serving.inference import GenerationParams

    gen_params = GenerationParams(
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
    )

    try:
        generated_text: str = await batcher.submit(body.prompt, gen_params)
    except Exception as exc:
        try:
            from observability.metrics import REQUEST_COUNTER

            REQUEST_COUNTER.labels(status="error").inc()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    latency_ms = (time.monotonic() - t0) * 1000
    tokens = len(generated_text.split())

    # ── Cache store ───────────────────────────────────────────────────────────
    import asyncio

    asyncio.create_task(cache.set(cache_key, generated_text))

    try:
        from observability.metrics import REQUEST_COUNTER, REQUEST_LATENCY

        REQUEST_COUNTER.labels(status="success").inc()
        REQUEST_LATENCY.observe(latency_ms / 1000)
    except Exception:
        pass

    return GenerateResponse(
        request_id=request_id,
        generated_text=generated_text,
        tokens_generated=tokens,
        latency_ms=round(latency_ms, 2),
        cached=False,
    )


@router.get("/v1/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness + readiness check."""
    state = request.app.state
    model_loaded = getattr(state, "engine", None) is not None and state.engine.is_loaded
    cache_connected = getattr(state, "cache", None) is not None and state.cache.is_connected
    uptime = time.monotonic() - getattr(state, "start_time", time.monotonic())
    status = "ok" if (model_loaded and cache_connected) else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        cache_connected=cache_connected,
        uptime_seconds=round(uptime, 2),
    )


@router.get("/v1/metrics/batch", response_model=BatchStats)
async def batch_stats(request: Request) -> BatchStats:
    """Return current batch engine statistics."""
    batcher = request.app.state.batcher
    return BatchStats(
        batch_size=batcher._last_batch_size,
        queue_depth=batcher.queue_depth,
        avg_latency_ms=round(batcher.avg_latency_ms, 2),
    )


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Prometheus metrics exposition endpoint."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
