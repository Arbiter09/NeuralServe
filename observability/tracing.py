"""OpenTelemetry tracer setup for NeuralServe.

Traces key spans across the inference path:
  batch_engine.dispatch → inference.generate → cache.lookup / cache.set
"""

from __future__ import annotations

from typing import Optional


def setup_tracing(
    service_name: str = "neuralserve",
    otlp_endpoint: Optional[str] = None,
):
    """Configure the global OpenTelemetry TracerProvider.

    Uses OTLP gRPC exporter when otlp_endpoint is provided;
    falls back to console (stdout) exporter for local development.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            print(f"[Tracing] OTLP exporter → {otlp_endpoint}")
        except ImportError:
            print("[Tracing] OTLP exporter not available, falling back to console")
            exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()
        print("[Tracing] Console span exporter enabled (set OTLP_ENDPOINT for production)")

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    return trace.get_tracer(service_name)


def get_tracer(name: str = "neuralserve"):
    from opentelemetry import trace

    return trace.get_tracer(name)


# ── Convenience context managers for common spans ────────────────────────────

def trace_batch_dispatch(tracer, batch_size: int):
    span = tracer.start_span("batch_engine.dispatch")
    span.set_attribute("batch.size", batch_size)
    return span


def trace_inference_generate(tracer, num_prompts: int):
    span = tracer.start_span("inference.generate")
    span.set_attribute("inference.num_prompts", num_prompts)
    return span


def trace_cache_lookup(tracer, cache_key: str):
    span = tracer.start_span("cache.lookup")
    span.set_attribute("cache.key_prefix", cache_key[:8])
    return span


def trace_cache_set(tracer, cache_key: str, ttl: int):
    span = tracer.start_span("cache.set")
    span.set_attribute("cache.key_prefix", cache_key[:8])
    span.set_attribute("cache.ttl_seconds", ttl)
    return span
