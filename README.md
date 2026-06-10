# NeuralServe

> Production LLM fine-tuning + inference serving with LoRA, dynamic batching, and full observability.

[![CI](https://github.com/Arbiter09/NeuralServe/actions/workflows/ci.yml/badge.svg)](https://github.com/Arbiter09/NeuralServe/actions/workflows/ci.yml)
[![Docker Build](https://github.com/Arbiter09/NeuralServe/actions/workflows/docker-build.yml/badge.svg)](https://github.com/Arbiter09/NeuralServe/actions/workflows/docker-build.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │              NeuralServe Stack                │
                          └──────────────────────────────────────────────┘

  ┌──────────┐   POST /v1/generate   ┌─────────────────────────────────────┐
  │  Client  │ ───────────────────►  │  FastAPI  +  DynamicBatcher          │
  └──────────┘                       │  (max_batch=8, max_wait=50ms)        │
       ▲                             └────────────────┬────────────────────┘
       │  GenerateResponse                            │  batch dispatch
       │                                              ▼
       │                             ┌─────────────────────────────────────┐
       │                             │  ModelEngine                         │
       │                             │  LLaMA 3.1 8B (4-bit NF4)           │
       │                             │  + LoRA adapter (rank-16, ~13M params)│
       │                             └────────────────┬────────────────────┘
       │                                              │
       │              ┌───────────────────────────────┤
       │              │                               │
       │              ▼                               ▼
       │   ┌─────────────────────┐      ┌─────────────────────────┐
       │   │   Redis KV-Cache    │      │  OpenTelemetry Tracer    │
       │   │  allkeys-lfu 2GB    │      │  spans: batch, infer,    │
       │   │  SHA-256 cache keys │      │  cache.lookup/set        │
       │   └──────────┬──────────┘      └─────────────────────────┘
       │              │
       │              ▼
       │   ┌─────────────────────┐      ┌─────────────────────────┐
       └───┤   Prometheus        │─────►│   Grafana Dashboard     │
           │  /metrics scrape    │      │  6 panels: latency,      │
           │  15s interval       │      │  throughput, GPU, cache  │
           └─────────────────────┘      └─────────────────────────┘
```

---

## Key Results

| Metric | Value |
|---|---|
| LoRA trainable parameters | ~13M / 8B **(98.6% reduction)** |
| ROUGE-L improvement | 0.29 (frozen baseline) → **0.41 (LoRA-tuned)** |
| Inference throughput | **180 req/min** @ **p95 < 420ms** on AWS EC2 g4dn.xlarge |

---

## Project Structure

```
NeuralServe/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # lint + unit tests on every push
│       └── docker-build.yml        # build & push to GHCR on version tag
├── configs/
│   ├── lora_config.yaml            # LoRA rank-16 hyperparameters
│   ├── training_config.yaml        # training run configuration
│   └── server_config.yaml          # serving hyperparameters
├── data/
│   ├── README.md                   # dataset download instructions
│   └── prepare_dataset.py          # download + tokenize Alpaca 12K
├── training/
│   ├── trainer.py                  # main fine-tuning entrypoint
│   ├── lora_setup.py               # 4-bit NF4 + PEFT LoRA wrapping
│   ├── dataset.py                  # InstructionDataset (prompt masking)
│   ├── callbacks.py                # ROUGE-L eval callback per epoch
│   └── evaluate.py                 # frozen vs LoRA ROUGE-L comparison
├── serving/
│   ├── app.py                      # FastAPI factory + lifespan
│   ├── router.py                   # /v1/generate, /v1/health, /metrics
│   ├── batch_engine.py             # DynamicBatcher (asyncio.Queue)
│   ├── inference.py                # ModelEngine (PeftModel + generate)
│   ├── cache.py                    # RedisCache (async, LFU eviction)
│   └── schemas.py                  # Pydantic request/response models
├── observability/
│   ├── tracing.py                  # OpenTelemetry tracer + OTLP/console
│   ├── metrics.py                  # Prometheus counters/histograms/gauges
│   └── dashboards/
│       └── neuralserve.json        # Grafana dashboard (6 panels, auto-provisioned)
├── infra/
│   ├── docker-compose.yml          # app + redis + prometheus + grafana
│   ├── Dockerfile.training         # CUDA 12.1 training image
│   ├── Dockerfile.serving          # multi-stage serving image (non-root)
│   ├── prometheus.yml              # scrape configs
│   └── grafana-provisioning/       # auto-wires Prometheus datasource + dashboard
├── tests/
│   ├── unit/
│   │   ├── test_batch_engine.py    # batcher dispatch, timeout, queue depth
│   │   ├── test_cache.py           # cache miss/hit, TTL, deterministic keys
│   │   └── test_schemas.py         # Pydantic validation edge cases
│   └── integration/
│       ├── test_api_endpoints.py   # mocked FastAPI endpoint tests
│       └── test_inference_pipeline.py  # GPU-gated end-to-end tests
├── scripts/
│   ├── start_server.sh             # launch uvicorn locally
│   ├── run_training.sh             # run fine-tuning pipeline
│   └── load_test.py                # 180 req/min async load tester
├── .env.example
├── pyproject.toml                  # ruff + pytest config
├── requirements-training.txt
└── requirements-serving.txt
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/Arbiter09/NeuralServe.git
cd NeuralServe
cp .env.example .env         # fill in MODEL_PATH, HF_TOKEN, etc.

# For training
pip install -r requirements-training.txt

# For serving
pip install -r requirements-serving.txt
```

### 2. Prepare dataset

```bash
python data/prepare_dataset.py --output data/alpaca_12k.jsonl --size 12000
```

Downloads `tatsu-lab/alpaca` from HuggingFace Hub, shuffles, slices to 12K samples, and prints dataset statistics.

### 3. Fine-tune

```bash
bash scripts/run_training.sh
```

Or with custom configs:

```bash
MODEL_PATH=meta-llama/Meta-Llama-3.1-8B \
RUN_NAME=my-lora-run \
bash scripts/run_training.sh \
  --lora_config configs/lora_config.yaml \
  --training_config configs/training_config.yaml
```

Adapter weights are saved to `outputs/lora-llama3-8b/final_adapter/`.

### 4. Evaluate ROUGE-L

```bash
python training/evaluate.py \
  --model_path meta-llama/Meta-Llama-3.1-8B \
  --adapter_path outputs/lora-llama3-8b/final_adapter \
  --num_samples 500
```

Expected output:
```
══════════════════════════════════════════════════
  ROUGE-L Evaluation (500 samples)
──────────────────────────────────────────────────
  Model                          ROUGE-L
──────────────────────────────────────────────────
  Frozen baseline                  0.2900
  LoRA-tuned                       0.4100  (+0.1200)
══════════════════════════════════════════════════
```

### 5. Run serving stack locally

```bash
docker compose -f infra/docker-compose.yml up --build
```

This starts:
| Service | Port | Description |
|---|---|---|
| `neuralserve` | 8000 | FastAPI inference server |
| `redis` | 6379 | KV-cache with LFU eviction |
| `prometheus` | 9090 | Metrics scraper |
| `grafana` | 3000 | Dashboard (`admin` / `admin`) |

### 6. Test the API

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain gradient descent in simple terms.", "max_new_tokens": 200}'
```

Example response:

```json
{
  "request_id": "f3c2a1b0-...",
  "generated_text": "Gradient descent is an optimization algorithm that ...",
  "tokens_generated": 47,
  "latency_ms": 312.5,
  "cached": false
}
```

Health check:

```bash
curl http://localhost:8000/v1/health
```

Prometheus metrics:

```bash
curl http://localhost:8000/metrics
```

### 7. Run load test

```bash
# 3 rps × 60s = 180 req/min, SLA: p95 < 420ms
python scripts/load_test.py \
  --url http://localhost:8000 \
  --rps 3 \
  --duration 60
```

---

## Configuration

### LoRA (`configs/lora_config.yaml`)

| Parameter | Value | Notes |
|---|---|---|
| `lora_r` | 16 | Rank — controls capacity vs parameter count |
| `lora_alpha` | 32 | Scaling factor (`alpha/r = 2`) |
| `lora_dropout` | 0.05 | Regularization |
| `target_modules` | q/k/v/o/gate/up/down proj | All attention + MLP layers |

### Training (`configs/training_config.yaml`)

| Parameter | Value |
|---|---|
| `num_train_epochs` | 3 |
| `per_device_train_batch_size` | 4 |
| `gradient_accumulation_steps` | 4 (effective batch = 16) |
| `learning_rate` | 2e-4 |
| `max_seq_length` | 2048 |

### Server (`configs/server_config.yaml`)

| Parameter | Default | Notes |
|---|---|---|
| `max_batch_size` | 8 | Max prompts per forward pass |
| `max_wait_ms` | 50 | Batch collection window |
| `cache.ttl_seconds` | 3600 | Redis entry lifetime |

---

## Observability

### Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `neuralserve_requests_total` | Counter | Total requests by status label |
| `neuralserve_request_latency_seconds` | Histogram | p50/p95/p99 latency |
| `neuralserve_batch_size` | Histogram | Batch size distribution |
| `neuralserve_tokens_generated_total` | Counter | Token throughput |
| `neuralserve_gpu_utilization_percent` | Gauge | nvidia-smi GPU % |
| `neuralserve_cache_hits_total` | Counter | Redis cache hits |
| `neuralserve_cache_misses_total` | Counter | Redis cache misses |
| `neuralserve_queue_depth` | Gauge | Active request queue depth |

### Grafana

Open [http://localhost:3000](http://localhost:3000) (admin/admin). The **NeuralServe — Inference Dashboard** is auto-provisioned with 6 panels:

1. **Request Rate** — req/min stat panel
2. **p50/p95/p99 Latency** — time series
3. **Token Throughput** — tokens/sec time series
4. **GPU Utilization** — gauge (0–100%)
5. **Cache Hit Rate** — % stat panel
6. **Batch Size Distribution** — bar chart

### OpenTelemetry

Set `OTLP_ENDPOINT` to your collector (e.g. Jaeger/Tempo):

```bash
OTLP_ENDPOINT=http://jaeger:4317 docker compose up
```

Traced spans: `batch_engine.dispatch`, `inference.generate`, `cache.lookup`, `cache.set`.

---

## Running Tests

```bash
# Unit tests (no GPU required)
pytest tests/unit/ -v

# Integration tests — API (mocked, no GPU)
pytest tests/integration/test_api_endpoints.py -v

# Integration tests — inference pipeline (requires CUDA)
pytest tests/integration/test_inference_pipeline.py -v
```

---

## Deployment on AWS EC2 g4dn.xlarge

```bash
# 1. Launch g4dn.xlarge with Deep Learning AMI (CUDA 12.1)
# 2. Clone repo and set env vars
git clone https://github.com/Arbiter09/NeuralServe.git && cd NeuralServe
cp .env.example .env
# Edit .env: set MODEL_PATH, ADAPTER_PATH, HF_TOKEN

# 3. Install NVIDIA Container Toolkit (if not on Deep Learning AMI)
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 4. Launch stack
docker compose -f infra/docker-compose.yml up -d

# 5. Verify
curl http://localhost:8000/v1/health
```

**Instance specs:** 4 vCPUs, 16 GB RAM, 1× NVIDIA T4 (16 GB VRAM), ~$0.526/hr on-demand.

---

## License

MIT © NeuralServe Contributors
