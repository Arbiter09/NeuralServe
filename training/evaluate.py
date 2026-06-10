"""ROUGE-L evaluation script — compares frozen baseline vs LoRA-tuned model."""

from __future__ import annotations

import argparse
from typing import Optional


def evaluate_rouge(
    model_path: str,
    adapter_path: Optional[str] = None,
    dataset_path: str = "tatsu-lab/alpaca",
    num_samples: int = 500,
    max_new_tokens: int = 128,
    max_prompt_length: int = 512,
    seed: int = 42,
) -> dict[str, float]:
    """Compute ROUGE-L for a frozen baseline and (optionally) a LoRA-tuned model.

    Returns a dict with keys 'frozen_rougeL' and 'lora_rougeL'.
    Expected: frozen ~0.29, LoRA-tuned ~0.41.
    """
    import random

    import torch
    import evaluate as hf_evaluate
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.dataset import format_example, load_records

    records = load_records(dataset_path)
    rng = random.Random(seed)
    rng.shuffle(records)
    eval_records = records[:num_samples]

    rouge = hf_evaluate.load("rouge")

    def run_inference(model, tokenizer, records) -> list[str]:
        model.eval()
        preds = []
        for rec in records:
            prompt = format_example({**rec, "output": ""}).rstrip()
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_length,
            ).to(model.device)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            preds.append(generated.strip())
        return preds

    references = [r.get("output", "").strip() for r in eval_records]

    # ── Frozen baseline ──────────────────────────────────────────────────────
    print(f"\nLoading frozen baseline: {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    frozen_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    frozen_preds = run_inference(frozen_model, tokenizer, eval_records)
    frozen_score = rouge.compute(predictions=frozen_preds, references=references)["rougeL"]
    del frozen_model
    torch.cuda.empty_cache()

    # ── LoRA-tuned model ─────────────────────────────────────────────────────
    lora_score = None
    if adapter_path:
        print(f"\nLoading LoRA-tuned model: {adapter_path} …")
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        lora_model = PeftModel.from_pretrained(base_model, adapter_path)
        lora_preds = run_inference(lora_model, tokenizer, eval_records)
        lora_score = rouge.compute(predictions=lora_preds, references=references)["rougeL"]
        del lora_model, base_model
        torch.cuda.empty_cache()

    # ── Print comparison table ───────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"  ROUGE-L Evaluation ({num_samples} samples)")
    print(f"{'─'*50}")
    print(f"  {'Model':<30} {'ROUGE-L':>10}")
    print(f"{'─'*50}")
    print(f"  {'Frozen baseline':<30} {frozen_score:>10.4f}")
    if lora_score is not None:
        delta = lora_score - frozen_score
        print(f"  {'LoRA-tuned':<30} {lora_score:>10.4f}  (+{delta:.4f})")
    print(f"{'═'*50}\n")

    result = {"frozen_rougeL": frozen_score}
    if lora_score is not None:
        result["lora_rougeL"] = lora_score
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROUGE-L evaluation for NeuralServe")
    parser.add_argument("--model_path", required=True, help="Base model path or HF hub ID")
    parser.add_argument("--adapter_path", default=None, help="LoRA adapter path")
    parser.add_argument("--dataset_path", default="tatsu-lab/alpaca", help="Dataset path")
    parser.add_argument("--num_samples", type=int, default=500, help="Evaluation samples")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    evaluate_rouge(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
    )
