"""ModelEngine — loads LLaMA 3.1 8B + LoRA adapter and performs batched generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationParams:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


class ModelEngine:
    """Manages model lifecycle and batched token generation.

    Loads LLaMA 3.1 8B in 4-bit NF4 quantization and applies the LoRA adapter
    via PeftModel, allowing inference on a single g4dn.xlarge (16 GB VRAM).
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False

        self._system_prompt: Optional[str] = None
        self._system_kv_cache = None

    def load(self) -> None:
        """Load model and tokenizer. Called once at server startup."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        print(f"[ModelEngine] Loading base model: {self.model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        base_model.config.use_cache = True

        if self.adapter_path:
            from peft import PeftModel

            print(f"[ModelEngine] Applying LoRA adapter: {self.adapter_path}")
            self._model = PeftModel.from_pretrained(base_model, self.adapter_path)
        else:
            self._model = base_model

        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self._loaded = True
        print("[ModelEngine] Model ready.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, prompts: list[str], params: GenerationParams) -> list[str]:
        """Run batched generation for a list of prompts.

        All prompts are padded to the same length (left-padding) and processed
        in a single forward pass for maximum GPU utilization.
        """
        import torch

        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self._model.device)

        do_sample = params.temperature > 0.0

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=params.max_new_tokens,
                do_sample=do_sample,
                temperature=params.temperature if do_sample else 1.0,
                top_p=params.top_p if do_sample else 1.0,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        prompt_lengths = inputs["input_ids"].shape[1]
        results = []
        for ids in output_ids:
            new_ids = ids[prompt_lengths:]
            text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append(text.strip())

        return results

    def cache_key_for_params(self, prompt: str, params: GenerationParams) -> str:
        """Generate a deterministic SHA-256 cache key for a prompt + params combo."""
        raw = f"{prompt}|{params.max_new_tokens}|{params.temperature}|{params.top_p}"
        return hashlib.sha256(raw.encode()).hexdigest()
