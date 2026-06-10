"""Training callbacks — ROUGE-L evaluation at end of each epoch."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


class RougeLEvalCallback:
    """Compute ROUGE-L on a held-out sample at the end of each epoch and log to wandb."""

    def __init__(self, eval_records: list[dict], tokenizer, num_samples: int = 200) -> None:
        try:
            from transformers import TrainerCallback

            self._base = TrainerCallback
        except ImportError:
            self._base = object

        self.eval_records = eval_records[:num_samples]
        self.tokenizer = tokenizer
        self.num_samples = num_samples

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        import evaluate as hf_evaluate

        rouge = hf_evaluate.load("rouge")

        model.eval()
        predictions, references = [], []

        import torch
        from training.dataset import format_example

        for rec in self.eval_records:
            prompt = format_example({**rec, "output": ""}).rstrip()
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            predictions.append(generated.strip())
            references.append(rec.get("output", "").strip())

        result = rouge.compute(predictions=predictions, references=references)
        rouge_l = result["rougeL"]

        print(f"\n[Epoch {state.epoch:.0f}] ROUGE-L: {rouge_l:.4f}")

        try:
            import wandb

            if wandb.run is not None:
                wandb.log({"eval/rougeL": rouge_l, "epoch": state.epoch})
        except ImportError:
            pass

        model.train()


def make_rouge_callback(eval_records, tokenizer, num_samples=200):
    """Factory that returns a HuggingFace-compatible TrainerCallback."""
    try:
        from transformers import TrainerCallback
    except ImportError:
        return None

    inner = RougeLEvalCallback(eval_records, tokenizer, num_samples)

    class _Callback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            inner.on_epoch_end(args, state, control, **kwargs)

    return _Callback()
