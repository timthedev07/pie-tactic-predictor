# ---------------------------------------------------------------------------
# Pie Tactic Predictor – GPU inference image
#
# Base: vLLM's official CUDA image (ships vLLM + PyTorch + CUDA runtime)
# Build:
#   docker build -t pie-tactic-predictor .
#
# Run (serve one model at a time):
#   docker run --gpus all -p 8000:8000 \
#     pie-tactic-predictor --model qwen2.5-coder-1.5b
# ---------------------------------------------------------------------------
FROM vllm/vllm-openai:latest

WORKDIR /app

# Install Python deps not already in the vLLM image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code + model config
COPY serve.py models.json merge_adapter.py ./

# Copy all four fine-tuned LoRA adapters
# NOTE: deepseek-coder-6.7b's final adapter lives at output/final/ on the host;
#       it should be moved to output/deepseek-coder-6.7b/final/ before building,
#       or you can add a symlink / copy step.
COPY output/deepseek-coder-1.3b/final/ output/deepseek-coder-1.3b/final/
COPY output/deepseek-coder-6.7b/final/ output/deepseek-coder-6.7b/final/
COPY output/qwen2.5-coder-1.5b/final/  output/qwen2.5-coder-1.5b/final/
COPY output/qwen2.5-coder-7b/final/    output/qwen2.5-coder-7b/final/

EXPOSE 8000

# Default: serve the smallest model. Override --model at runtime.
ENTRYPOINT ["python", "serve.py"]
CMD ["--model", "qwen2.5-coder-1.5b", "--host", "0.0.0.0", "--port", "8000"]
