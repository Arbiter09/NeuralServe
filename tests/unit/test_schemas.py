"""Unit tests for Pydantic request/response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from serving.schemas import GenerateRequest, GenerateResponse, HealthResponse


class TestGenerateRequest:
    def test_defaults(self):
        req = GenerateRequest(prompt="Hello world")
        assert req.max_new_tokens == 256
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.stream is False

    def test_custom_values(self):
        req = GenerateRequest(
            prompt="Test prompt",
            max_new_tokens=128,
            temperature=0.5,
            top_p=0.8,
            stream=True,
        )
        assert req.max_new_tokens == 128
        assert req.temperature == 0.5
        assert req.top_p == 0.8
        assert req.stream is True

    def test_missing_prompt_raises(self):
        with pytest.raises(ValidationError):
            GenerateRequest()  # type: ignore

    def test_blank_prompt_raises(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="   ")

    def test_max_new_tokens_bounds(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="p", max_new_tokens=0)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="p", max_new_tokens=5000)

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="p", temperature=-0.1)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="p", temperature=2.1)

    def test_top_p_bounds(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="p", top_p=-0.1)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="p", top_p=1.1)


class TestGenerateResponse:
    def test_serialization(self):
        resp = GenerateResponse(
            request_id="abc-123",
            generated_text="The answer is 42.",
            tokens_generated=5,
            latency_ms=120.5,
            cached=False,
        )
        data = resp.model_dump()
        assert data["request_id"] == "abc-123"
        assert data["generated_text"] == "The answer is 42."
        assert data["tokens_generated"] == 5
        assert data["latency_ms"] == 120.5
        assert data["cached"] is False

    def test_json_roundtrip(self):
        resp = GenerateResponse(
            request_id="xyz",
            generated_text="hello",
            tokens_generated=1,
            latency_ms=50.0,
            cached=True,
        )
        json_str = resp.model_dump_json()
        restored = GenerateResponse.model_validate_json(json_str)
        assert restored.request_id == "xyz"
        assert restored.cached is True


class TestHealthResponse:
    def test_healthy_state(self):
        resp = HealthResponse(
            status="ok",
            model_loaded=True,
            cache_connected=True,
            uptime_seconds=3600.0,
        )
        assert resp.status == "ok"

    def test_degraded_state(self):
        resp = HealthResponse(
            status="degraded",
            model_loaded=False,
            cache_connected=True,
            uptime_seconds=10.0,
        )
        assert resp.status == "degraded"
        assert resp.model_loaded is False
