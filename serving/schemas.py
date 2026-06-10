"""Pydantic request/response models for the NeuralServe API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt for generation")
    max_new_tokens: int = Field(256, ge=1, le=4096, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    stream: bool = Field(False, description="Enable streaming response (SSE)")

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be blank")
        return v

    model_config = {"json_schema_extra": {"example": {
        "prompt": "Explain gradient descent in simple terms.",
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
    }}}


class GenerateResponse(BaseModel):
    request_id: str = Field(..., description="Unique request identifier (UUID4)")
    generated_text: str = Field(..., description="Model-generated text")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds")
    cached: bool = Field(False, description="True if response was served from cache")


class BatchStats(BaseModel):
    batch_size: int = Field(..., description="Average batch size of last N dispatches")
    queue_depth: int = Field(..., description="Current number of requests queued")
    avg_latency_ms: float = Field(..., description="Rolling average latency in ms")


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' or 'degraded'")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    cache_connected: bool = Field(..., description="Whether Redis is reachable")
    uptime_seconds: float = Field(..., description="Seconds since server start")
