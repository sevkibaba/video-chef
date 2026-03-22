# Build argument: cpu or gpu
ARG RUNTIME=cpu

# GPU build stage
FROM nvidia/cuda:12.2-runtime-ubuntu22.04 as gpu-base
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# CPU build stage  
FROM python:3.11-slim as cpu-base
RUN apt-get update && apt-get install -y ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Select base image
FROM ${RUNTIME}-base as final

WORKDIR /app

# Python dependencies
COPY requirements.txt .

# Install ONNX Runtime with GPU or CPU
RUN if [ "${RUNTIME}" = "gpu" ]; then \
      pip install --no-cache-dir onnxruntime-gpu; \
    else \
      pip install --no-cache-dir onnxruntime; \
    fi

RUN pip install --no-cache-dir \
    rembg opencv-python Pillow numpy tqdm PyYAML click rich

# Copy source code
COPY src/ ./src/
COPY pipelines/ ./pipelines/
COPY config.yaml .

# Create directories
RUN mkdir -p /app/input /app/output /app/temp

# Environment
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

# Default command
ENTRYPOINT ["python", "pipelines/instagram_pipeline.py"]
CMD ["--help"]
