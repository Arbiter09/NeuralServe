"""InstructionDataset — loads Alpaca-format data and tokenizes for causal LM training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)
PROMPT_NO_INPUT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{output}"


def format_example(example: dict) -> str:
    """Format a single Alpaca example into the prompt template."""
    if example.get("input", "").strip():
        return PROMPT_TEMPLATE.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    return PROMPT_NO_INPUT_TEMPLATE.format(
        instruction=example["instruction"],
        output=example["output"],
    )


def load_records(dataset_path: str, size: Optional[int] = None) -> list[dict]:
    """Load records from a local JSONL file or HuggingFace hub."""
    path = Path(dataset_path)
    if path.exists() and path.suffix == ".jsonl":
        with path.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        from datasets import load_dataset

        ds = load_dataset(dataset_path, split="train")
        records = list(ds)

    if size is not None:
        records = records[:size]
    return records


class InstructionDataset(Dataset):
    """PyTorch Dataset for instruction-following fine-tuning.

    Tokenizes each example and masks prompt tokens with -100 so the model
    only computes loss on the response portion.
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_seq_length: int = 2048,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self._encodings: list[dict] = []
        self._build()

    def _build(self) -> None:
        for rec in self.records:
            full_text = format_example(rec)

            # Tokenize full prompt+response
            full = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors=None,
            )
            input_ids = full["input_ids"]

            # Determine prompt length (without the response) to mask with -100
            if rec.get("input", "").strip():
                prompt_text = (
                    f"### Instruction:\n{rec['instruction']}\n\n"
                    f"### Input:\n{rec['input']}\n\n### Response:\n"
                )
            else:
                prompt_text = f"### Instruction:\n{rec['instruction']}\n\n### Response:\n"

            prompt_ids = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors=None,
            )["input_ids"]
            prompt_len = len(prompt_ids)

            labels = [-100] * prompt_len + input_ids[prompt_len:]
            labels = labels[: self.max_seq_length]

            self._encodings.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(
                        full["attention_mask"], dtype=torch.long
                    ),
                }
            )

    def __len__(self) -> int:
        return len(self._encodings)

    def __getitem__(self, idx: int) -> dict:
        return self._encodings[idx]


def build_datasets(
    dataset_path: str,
    tokenizer,
    max_seq_length: int = 2048,
    dataset_size: int = 12000,
    train_split_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[InstructionDataset, InstructionDataset]:
    """Load, split, and return (train_dataset, eval_dataset)."""
    import random

    records = load_records(dataset_path, size=dataset_size)
    rng = random.Random(seed)
    rng.shuffle(records)

    split = int(len(records) * train_split_ratio)
    train_records = records[:split]
    eval_records = records[split:]

    print(f"Dataset: {len(train_records):,} train / {len(eval_records):,} eval samples")

    train_ds = InstructionDataset(train_records, tokenizer, max_seq_length)
    eval_ds = InstructionDataset(eval_records, tokenizer, max_seq_length)
    return train_ds, eval_ds
