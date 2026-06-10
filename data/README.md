# Data

## Dataset

NeuralServe fine-tunes on a **12,000-sample** subset of the [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) — a collection of 52K instruction-following demonstrations generated using `text-davinci-003`.

### Format

Each example follows the Alpaca instruction format:

```json
{
  "instruction": "Give three tips for staying healthy.",
  "input": "",
  "output": "1. Eat a balanced diet...\n2. Exercise regularly...\n3. Get enough sleep..."
}
```

Formatted into the model prompt template:

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

When `input` is empty the `### Input:` block is omitted.

## Preparing the Dataset

Run the preparation script to download, shuffle, slice to 12K samples, and save locally:

```bash
python data/prepare_dataset.py
```

Output: `data/alpaca_12k.jsonl`

### Options

```
python data/prepare_dataset.py \
  --output data/alpaca_12k.jsonl \
  --size 12000 \
  --seed 42
```

## Custom Dataset

To use your own data, provide a JSONL file with the same schema (`instruction`, `input`, `output`) and pass it via `--dataset_path` in the training config:

```yaml
# configs/training_config.yaml
dataset_path: "./data/my_dataset.jsonl"
```
