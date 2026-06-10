"""Main fine-tuning entrypoint — trains LLaMA 3.1 8B + LoRA on Alpaca-format data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralServe LoRA fine-tuning")
    parser.add_argument(
        "--lora_config",
        default="configs/lora_config.yaml",
        help="Path to LoRA config YAML",
    )
    parser.add_argument(
        "--training_config",
        default="configs/training_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Override dataset path (HF hub ID or local JSONL)",
    )
    parser.add_argument(
        "--run_name",
        default="lora-llama3-8b",
        help="WandB run name",
    )
    args = parser.parse_args()

    lora_cfg = load_yaml(args.lora_config)
    train_cfg = load_yaml(args.training_config)

    if args.dataset_path:
        train_cfg["dataset_path"] = args.dataset_path

    dataset_path = train_cfg.get("dataset_path", "tatsu-lab/alpaca")
    dataset_size = train_cfg.get("dataset_size", 12000)
    max_seq_length = train_cfg.get("max_seq_length", 2048)
    train_split_ratio = train_cfg.get("train_split_ratio", 0.9)
    output_dir = train_cfg.get("output_dir", "./outputs/lora-llama3-8b")

    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    from training.callbacks import make_rouge_callback
    from training.dataset import build_datasets, load_records
    from training.lora_setup import load_base_model, wrap_with_lora

    model, tokenizer = load_base_model(lora_cfg)
    model = wrap_with_lora(model, lora_cfg)

    train_ds, eval_ds = build_datasets(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset_size=dataset_size,
        train_split_ratio=train_split_ratio,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        fp16=train_cfg.get("fp16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=train_cfg.get("eval_strategy", "epoch"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        report_to=train_cfg.get("report_to", "wandb"),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        run_name=args.run_name,
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_cfg.get("greater_is_better", False),
        remove_unused_columns=False,
        group_by_length=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    # Load raw eval records for ROUGE-L callback
    from training.dataset import load_records as _load

    eval_records_raw = _load(dataset_path, size=dataset_size)
    split = int(len(eval_records_raw) * train_split_ratio)
    eval_records_raw = eval_records_raw[split:]

    rouge_callback = make_rouge_callback(eval_records_raw, tokenizer, num_samples=200)
    callbacks = [rouge_callback] if rouge_callback else []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("Starting training …")
    trainer.train()

    # Save final adapter weights
    final_adapter_dir = Path(output_dir) / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print(f"Adapter saved to {final_adapter_dir}")


if __name__ == "__main__":
    main()
