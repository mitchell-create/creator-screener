FROM python:3.11-slim

# System dependencies (ffmpeg for audio extraction + frame capture, git for torch.hub)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip/setuptools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy project files and install dependencies
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Pre-download ML model weights to avoid cold-start latency (~200MB total)
# Silero VAD (~2MB) + TOPIQ-NR (~173MB)
RUN python -c "\
import torch; \
torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True); \
import pyiqa; \
pyiqa.create_metric('topiq_nr', device=torch.device('cpu')); \
print('Model weights pre-downloaded')"

# Create working directories
RUN mkdir -p /data /tmp/affiliate_pipeline

# Default paths for Railway (override via env vars in dashboard)
ENV AFF_INPUT_CSV_PATH=/data/input.csv \
    AFF_OUTPUT_CSV_PATH=/data/output.csv \
    AFF_TEMP_DIR=/tmp/affiliate_pipeline

ENTRYPOINT ["python", "-m", "src.main"]
