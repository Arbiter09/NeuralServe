"""LoRA model setup using PEFT — wraps LLaMA 3.1 8B in 4-bit NF4 quantization."""

from __future__ import annotations

from typing import Any

import torch


def load_base_model(config: dict[str, Any]):
    """Load LLaMA 3.1 8B in 4-bit NF4 quantization via BitsAndBytes.

    Uses double quantization + float16 compute dtype to fit on a single
    g4dn.xlarge (16 GB VRAM) while maintaining training-quality precision.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = config["model_name"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"Loading base model: {model_name} (4-bit NF4) …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def wrap_with_lora(model, lora_config: dict[str, Any]):
    """Apply LoRA adapters to the model via PEFT.

    LoRA rank-16 on all attention + MLP projection layers yields ~13M
    trainable parameters out of ~8B total — a 98.6% reduction.
    """
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_config["lora_r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    reduction = (1 - trainable / total) * 100
    print(f"\n{'─'*60}")
    print(f"  Trainable parameters : {trainable:,}  ({trainable/total:.2%})")
    print(f"  Total parameters     : {total:,}")
    print(f"  Parameter reduction  : {reduction:.1f}%")
    print(f"{'─'*60}\n")

    return model
