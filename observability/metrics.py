"""Prometheus metrics definitions and background GPU utilization monitor."""

from __future__ import annotations

import asyncio
import subprocess

from prometheus_client import Counter, Gauge, Histogram

# ── Request metrics ───────────────────────────────────────────────────────────
REQUEST_COUNTER = Counter(
    "neuralserve_requests_total",
    "Total requests received by the inference server",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "neuralserve_request_latency_seconds",
    "End-to-end request latency in seconds",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0],
)

# ── Batch metrics ─────────────────────────────────────────────────────────────
BATCH_SIZE = Histogram(
    "neuralserve_batch_size",
    "Distribution of actual batch sizes dispatched to ModelEngine",
    buckets=[1, 2, 4, 6, 8],
)

QUEUE_DEPTH = Gauge(
    "neuralserve_queue_depth",
    "Current number of requests waiting in the batch queue",
)

# ── Token metrics ─────────────────────────────────────────────────────────────
TOKEN_THROUGHPUT = Counter(
    "neuralserve_tokens_generated_total",
    "Total number of tokens generated across all requests",
)

# ── GPU metrics ───────────────────────────────────────────────────────────────
GPU_UTILIZATION = Gauge(
    "neuralserve_gpu_utilization_percent",
    "GPU utilization percentage reported by nvidia-smi",
)

# ── Cache metrics ─────────────────────────────────────────────────────────────
CACHE_HITS = Counter(
    "neuralserve_cache_hits_total",
    "Number of requests served from the Redis cache",
)

CACHE_MISSES = Counter(
    "neuralserve_cache_misses_total",
    "Number of requests that resulted in a cache miss",
)


async def gpu_monitor_loop(interval_seconds: float = 5.0) -> None:
    """Poll nvidia-smi every interval_seconds and update GPU_UTILIZATION gauge.

    Runs as a background asyncio task started during server lifespan.
    Gracefully handles environments without a GPU (e.g. CI, local dev).
    """
    while True:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().splitlines()
                if lines:
                    util = float(lines[0].strip())
                    GPU_UTILIZATION.set(util)
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            # nvidia-smi not available (CPU-only environment)
            pass
        except Exception:
            pass

        await asyncio.sleep(interval_seconds)
