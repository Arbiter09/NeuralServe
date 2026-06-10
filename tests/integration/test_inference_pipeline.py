"""Integration tests for end-to-end inference pipeline — skipped without CUDA."""

from __future__ import annotations

import os

import pytest

try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

skip_no_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA not available — skipping GPU inference tests",
)

MODEL_PATH = os.getenv("TEST_MODEL_PATH", "meta-llama/Meta-Llama-3.1-8B")
ADAPTER_PATH = os.getenv("TEST_ADAPTER_PATH", None)


@skip_no_cuda
def test_model_loads_with_lora_adapter():
    """Verify the ModelEngine loads without errors when a LoRA adapter is present."""
    from serving.inference import ModelEngine

    engine = ModelEngine(
        model_path=MODEL_PATH,
        adapter_path=ADAPTER_PATH,
        device="cuda",
    )
    engine.load()
    assert engine.is_loaded is True


@skip_no_cuda
def test_generate_returns_nonempty_string():
    """Single-prompt generation must return a non-empty string."""
    from serving.inference import GenerationParams, ModelEngine

    engine = ModelEngine(model_path=MODEL_PATH, adapter_path=ADAPTER_PATH, device="cuda")
    engine.load()

    params = GenerationParams(max_new_tokens=32, temperature=0.0, top_p=1.0)
    results = engine.generate(["What is 2 + 2?"], params)

    assert len(results) == 1
    assert isinstance(results[0], str)
    assert len(results[0].strip()) > 0


@skip_no_cuda
def test_batched_generate_consistent_with_single():
    """Batched generation with identical prompts should produce consistent results."""
    from serving.inference import GenerationParams, ModelEngine

    engine = ModelEngine(model_path=MODEL_PATH, adapter_path=ADAPTER_PATH, device="cuda")
    engine.load()

    prompt = "List three primary colors."
    params = GenerationParams(max_new_tokens=32, temperature=0.0, top_p=1.0)

    single = engine.generate([prompt], params)
    batched = engine.generate([prompt, prompt], params)

    assert len(batched) == 2
    # Greedy decoding (temperature=0) should be deterministic
    assert batched[0] == batched[1]
    assert batched[0] == single[0]


@skip_no_cuda
def test_cache_key_is_deterministic():
    """Same prompt + params always produce the same cache key."""
    from serving.inference import GenerationParams, ModelEngine

    engine = ModelEngine(model_path=MODEL_PATH, device="cuda")
    params = GenerationParams(max_new_tokens=256, temperature=0.7, top_p=0.9)

    k1 = engine.cache_key_for_params("hello", params)
    k2 = engine.cache_key_for_params("hello", params)
    assert k1 == k2
    assert len(k1) == 64  # SHA-256 hex digest
