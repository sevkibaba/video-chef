FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-gnu \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir rembg onnxruntime opencv-python Pillow numpy tqdm PyYAML click rich

# Copy source code
COPY src/ ./src/
COPY pipelines/ ./pipelines/
COPY config.yaml .

# Create directories
RUN mkdir -p /app/input /app/output /app/temp

# Default command
ENTRYPOINT ["python", "pipelines/instagram_pipeline.py"]
