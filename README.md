# Available models

| Short ID              | Base model                             |
| --------------------- | -------------------------------------- |
| `deepseek-coder-1.3b` | `deepseek-ai/deepseek-coder-1.3b-base` |
| `deepseek-coder-6.7b` | `deepseek-ai/deepseek-coder-6.7b-base` |
| `qwen2.5-coder-1.5b`  | `Qwen/Qwen2.5-Coder-1.5B`              |
| `qwen2.5-coder-7b`    | `Qwen/Qwen2.5-Coder-7B`                |

LoRA adapters are baked into the Docker image. Base model weights are downloaded from HuggingFace at first launch and cached in a named Docker volume.

---

## Docker

### Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- An NVIDIA GPU with enough VRAM for the model you want to serve (≥8 GB recommended for 7B models)

### Build

```bash
docker build -t pie-tactic-predictor .
```

The image is around 10–12 GB (CUDA + PyTorch + vLLM + adapters). Build once; base model weights are not included and are downloaded on first run.

### Run

```bash
# Serve a model (add --gpus all to expose the GPU)
docker run --gpus all -p 8000:8000 pie-tactic-predictor --model qwen2.5-coder-1.5b

# Serve a larger model with a persistent HF cache volume
docker run --gpus all -p 8000:8000 \
  -v hf-cache:/root/.cache/huggingface \
  pie-tactic-predictor --model qwen2.5-coder-7b

# Use multiple GPUs (tensor parallel)
docker run --gpus all -p 8000:8000 \
  pie-tactic-predictor --model deepseek-coder-6.7b --tp 2
```

#### All CLI flags

| Flag                       | Default                 | Description                                                        |
| -------------------------- | ----------------------- | ------------------------------------------------------------------ |
| `--model`                  | _(required)_            | Short ID from the table above                                      |
| `--port`                   | `8000`                  | Port to listen on                                                  |
| `--host`                   | `0.0.0.0`               | Bind address                                                       |
| `--tp`                     | `1`                     | Tensor-parallel degree (number of GPUs)                            |
| `--max-model-len`          | `2048`                  | Maximum context length                                             |
| `--gpu-memory-utilization` | `0.9`                   | Fraction of GPU VRAM to allocate                                   |
| `--merged`                 | off                     | Serve a pre-merged model from `output/<model>/merged/` (see below) |
| `--adapter-path`           | `output/<model>/final/` | Override the LoRA adapter path                                     |

### docker compose

`docker-compose.yml` builds and runs the default model (`qwen2.5-coder-1.5b`):

```bash
docker compose up --build
```

To serve a different model without editing the file:

```bash
docker compose run --service-ports predictor --model deepseek-coder-6.7b
```

---

## API

```
POST /predict
Content-Type: application/json

{ "prompt": "<your context + goal string>", "max_tokens": 128, "temperature": 0.0 }
```

Response:

```json
{ "tactic": "intro n" }
```

Health check:

```
GET /health  ->  { "status": "ok" }
```

---

## Pre-merging adapters (optional)

Merging a LoRA adapter into the base weights removes the LoRA overhead at inference time and lets you serve without `--enable-lora`:

```bash
# On the host (requires transformers + peft)
python merge_adapter.py --model qwen2.5-coder-7b

# Then serve the merged model inside Docker
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/output:/app/output \
  pie-tactic-predictor --model qwen2.5-coder-7b --merged
```

The merged weights are written to `output/<model>/merged/` and are excluded from the Docker image by `.dockerignore` — mount them as a volume if you want to use them inside the container.

---

## Notes

- **HuggingFace cache** — base model weights (~1–14 GB depending on model) are downloaded from HuggingFace on the first run. The `hf-cache` Docker volume persists them. Set `HF_TOKEN` if you need a gated model.
- **Checkpoints** — only the `final/` adapter is included in the image. If you want to serve a specific checkpoint, mount the output directory and pass `--adapter-path`.
- **CPU-only** — vLLM requires CUDA; the image will not work on CPU-only machines.
