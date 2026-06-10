"""Async load tester for NeuralServe — targets 180 req/min with p95 < 420ms SLA.

Usage:
    python scripts/load_test.py --url http://localhost:8000 --rps 3 --duration 60

This sends `rps` requests per second for `duration` seconds, then reports:
  - Actual achieved req/min
  - p50 / p95 / p99 latency
  - Error rate
  - PASS/FAIL verdict against p95 < 420ms
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

SAMPLE_PROMPTS = [
    "Explain gradient descent in simple terms.",
    "What are the main differences between supervised and unsupervised learning?",
    "Describe the transformer architecture in two sentences.",
    "What is the vanishing gradient problem?",
    "Summarize the key ideas behind attention mechanisms.",
    "What is LoRA and why is it useful for fine-tuning large language models?",
    "Compare ReLU and GELU activation functions.",
    "What is the role of layer normalization in deep learning?",
    "Explain the intuition behind dropout regularization.",
    "What is the difference between BLEU and ROUGE evaluation metrics?",
]

P95_THRESHOLD_MS = 420.0


@dataclass
class RequestResult:
    status_code: int
    latency_ms: float
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and 200 <= self.status_code < 300


@dataclass
class LoadTestReport:
    total_requests: int
    successful: int
    failed: int
    duration_seconds: float
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def req_per_min(self) -> float:
        return (self.total_requests / self.duration_seconds) * 60

    @property
    def error_rate(self) -> float:
        return self.failed / self.total_requests if self.total_requests > 0 else 0.0

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def print(self, p95_threshold_ms: float = P95_THRESHOLD_MS) -> None:
        p50 = self.percentile(50)
        p95 = self.percentile(95)
        p99 = self.percentile(99)
        mean = statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

        print(f"\n{'═'*60}")
        print(f"  NeuralServe Load Test Results")
        print(f"{'─'*60}")
        print(f"  Duration           : {self.duration_seconds:.1f}s")
        print(f"  Total requests     : {self.total_requests}")
        print(f"  Successful         : {self.successful}")
        print(f"  Failed             : {self.failed}")
        print(f"  Error rate         : {self.error_rate:.2%}")
        print(f"{'─'*60}")
        print(f"  Throughput         : {self.req_per_min:.1f} req/min")
        print(f"{'─'*60}")
        print(f"  Latency (ms)")
        print(f"    mean             : {mean:.1f}")
        print(f"    p50              : {p50:.1f}")
        print(f"    p95              : {p95:.1f}   (SLA: < {p95_threshold_ms:.0f}ms)")
        print(f"    p99              : {p99:.1f}")
        print(f"{'─'*60}")

        verdict = "✅ PASS" if p95 < p95_threshold_ms else "❌ FAIL"
        print(f"  p95 SLA ({p95_threshold_ms:.0f}ms)     : {verdict}")
        print(f"{'═'*60}\n")

        if p95 >= p95_threshold_ms:
            raise SystemExit(
                f"SLA VIOLATED: p95 latency {p95:.1f}ms exceeds threshold {p95_threshold_ms:.0f}ms"
            )


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_new_tokens: int = 64,
) -> RequestResult:
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{url}/v1/generate",
            json={"prompt": prompt, "max_new_tokens": max_new_tokens},
            timeout=30.0,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        return RequestResult(status_code=resp.status_code, latency_ms=latency_ms)
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        return RequestResult(status_code=0, latency_ms=latency_ms, error=str(exc))


async def run_load_test(
    url: str,
    rps: float,
    duration: float,
    max_new_tokens: int = 64,
    concurrency: int = 20,
) -> LoadTestReport:
    """Run sustained load for `duration` seconds at `rps` requests per second."""
    results: list[RequestResult] = []
    interval = 1.0 / rps
    semaphore = asyncio.Semaphore(concurrency)
    start_time = time.monotonic()

    async def bounded_request(prompt: str, client: httpx.AsyncClient) -> None:
        async with semaphore:
            result = await send_request(client, url, prompt, max_new_tokens)
            results.append(result)

    async with httpx.AsyncClient() as client:
        tasks: list[asyncio.Task] = []
        prompt_idx = 0

        while time.monotonic() - start_time < duration:
            prompt = SAMPLE_PROMPTS[prompt_idx % len(SAMPLE_PROMPTS)]
            task = asyncio.create_task(bounded_request(prompt, client))
            tasks.append(task)
            prompt_idx += 1
            await asyncio.sleep(interval)

        # Wait for all in-flight requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)

    actual_duration = time.monotonic() - start_time
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    latencies = [r.latency_ms for r in successful]

    return LoadTestReport(
        total_requests=len(results),
        successful=len(successful),
        failed=len(failed),
        duration_seconds=actual_duration,
        latencies_ms=latencies,
    )


async def warmup(url: str, num_requests: int = 5) -> None:
    """Send warmup requests to prime the model cache and batch engine."""
    print(f"Warming up with {num_requests} requests …")
    async with httpx.AsyncClient() as client:
        tasks = [
            send_request(client, url, SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)])
            for i in range(num_requests)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    print("Warmup complete.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralServe async load tester")
    parser.add_argument("--url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--rps", type=float, default=3.0, help="Requests per second (3 = 180/min)")
    parser.add_argument("--duration", type=float, default=60.0, help="Test duration in seconds")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Tokens to generate per req")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent in-flight reqs")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup phase")
    parser.add_argument(
        "--p95-threshold",
        type=float,
        default=P95_THRESHOLD_MS,
        help=f"p95 latency SLA in ms (default: {P95_THRESHOLD_MS})",
    )
    args = parser.parse_args()

    print(f"NeuralServe Load Test")
    print(f"  Target : {args.url}")
    print(f"  Rate   : {args.rps} rps ({args.rps * 60:.0f} req/min)")
    print(f"  Duration: {args.duration}s")
    print(f"  p95 SLA : {args.p95_threshold}ms\n")

    if not args.no_warmup:
        asyncio.run(warmup(args.url))

    report = asyncio.run(
        run_load_test(
            url=args.url,
            rps=args.rps,
            duration=args.duration,
            max_new_tokens=args.max_new_tokens,
            concurrency=args.concurrency,
        )
    )
    report.print(p95_threshold_ms=args.p95_threshold)


if __name__ == "__main__":
    main()
