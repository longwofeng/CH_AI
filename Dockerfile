FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

# OpenCV and PyTorch runtime libraries required by Ultralytics.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Prefer the smaller CPU-only PyTorch wheels inside the container.
RUN python -m pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY main.py .
# Bake yolov8n.pt into the image so the container does not download it at startup.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]