"""FastAPI + YOLOv8 visual detection microservice for UAV/aerial imagery.

The service accepts an absolute image URL or a Dify-relative path such as
/files/5dcc1d5f... and resolves the latter against DIFY_BASE_URL. This keeps
Dify tool calls small and avoids Base64 request-body validation failures.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from ultralytics import YOLO

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("vision-service")

# Only report objects relevant to low-altitude inspection. COCO false
# positives such as "oven" and "refrigerator" are filtered out.
TARGET_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person']

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")

# Dify may return a relative file path such as /files/5dcc1d5f... instead of
# a fully-qualified URL. Resolve those paths against this base URL.
DIFY_BASE_URL = os.getenv("DIFY_BASE_URL", "http://127.0.0.1:5001")

# Aerial targets are small after whole-image resizing, so use a larger input
# resolution. A 0.25 confidence threshold suppresses low-quality guesses.
IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "1280"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))

# Download limits protect the service from huge files and slow upstream URLs.
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))
DOWNLOAD_TIMEOUT_SECONDS = float(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "30"))
CONNECT_TIMEOUT_SECONDS = float(os.getenv("CONNECT_TIMEOUT_SECONDS", "10"))

# A single model instance is shared by all requests. Inference is serialized
# because concurrent calls on one Ultralytics model are not guaranteed safe.
_model: Optional[YOLO] = None
_http_client: Optional[httpx.AsyncClient] = None
_model_lock = threading.Lock()


class DetectUrlRequest(BaseModel):
    image_url: str = Field(
        ...,
        description="Aerial image URL or relative path",
    )


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Pixel coordinates [x1, y1, x2, y2]",
    )


class DetectResponse(BaseModel):
    status: str
    summary: Dict[str, int]
    details: List[Detection]


class HealthResponse(BaseModel):
    status: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the YOLO model and shared HTTP client once at startup."""
    global _model, _http_client

    logger.info("Loading YOLO model: %s", MODEL_PATH)
    print(f"[MODEL] Loading: {MODEL_PATH}", flush=True)
    try:
        _model = YOLO(MODEL_PATH)
    except Exception as exc:  # pragma: no cover - depends on external model
        logger.exception("Failed to load YOLO model from %s", MODEL_PATH)
        print(f"[MODEL] Failed: {MODEL_PATH} ({exc})", flush=True)
        raise RuntimeError(f"Unable to load YOLO model '{MODEL_PATH}'") from exc

    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            DOWNLOAD_TIMEOUT_SECONDS,
            connect=CONNECT_TIMEOUT_SECONDS,
        ),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        headers={"User-Agent": "low-altitude-vision-service/1.0"},
    )

    print(f"[MODEL] Loaded: {MODEL_PATH}", flush=True)
    logger.info(
        "Model ready (model=%s, confidence_threshold=%s, iou_threshold=%s, image_size=%s, max_image_bytes=%s)",
        MODEL_PATH,
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD,
        IMAGE_SIZE,
        MAX_IMAGE_BYTES,
    )
    try:
        yield
    finally:
        if _http_client is not None:
            await _http_client.aclose()
            _http_client = None
        _model = None
        logger.info("YOLO model and HTTP client released")


app = FastAPI(
    title="Low-Altitude Inspection Vision Service",
    version="1.0.0",
    lifespan=lifespan,
)


def _decode_image_bytes(raw: bytes) -> Image.Image:
    """Decode downloaded bytes into an upright RGB PIL image."""
    if not raw:
        raise ValueError("downloaded image is empty")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
    ) as exc:
        raise ValueError("downloaded content is not a supported image") from exc


async def _download_image(url: str) -> Image.Image:
    """Download an image URL asynchronously and decode it with PIL."""
    client = _http_client
    if client is None:
        raise HTTPException(status_code=503, detail="HTTP client is not ready")

    try:
        response = await client.get(url)
        response.raise_for_status()

        content_type = (
            response.headers.get("content-type", "")
            .split(";", 1)[0]
            .strip()
            .lower()
        )
        if (
            content_type
            and not content_type.startswith("image/")
            and content_type not in {"application/octet-stream", "binary/octet-stream"}
        ):
            raise HTTPException(
                status_code=415,
                detail=f"URL did not return an image (Content-Type: {content_type})",
            )

        content_length = response.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_IMAGE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image exceeds the {MAX_IMAGE_BYTES} byte limit",
                    )
            except ValueError:
                pass

        raw = response.content
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds the {MAX_IMAGE_BYTES} byte limit",
            )
    except HTTPException:
        raise
    except httpx.TimeoutException as exc:
        logger.warning("Timed out downloading image URL")
        raise HTTPException(
            status_code=504, detail="Timed out downloading the image URL"
        ) from exc
    except httpx.HTTPStatusError as exc:
        logger.warning("Image URL returned HTTP %s", exc.response.status_code)
        raise HTTPException(
            status_code=502,
            detail=f"Image URL returned HTTP {exc.response.status_code}",
        ) from exc
    except httpx.RequestError as exc:
        logger.warning("Failed to download image URL: %s", exc)
        raise HTTPException(
            status_code=502, detail="Failed to download the image URL"
        ) from exc

    logger.info("Downloaded image: %d bytes", len(raw))
    try:
        return await run_in_threadpool(_decode_image_bytes, raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _infer(image: Image.Image) -> List[Detection]:
    """Run YOLOv8 inference and convert Ultralytics results to API objects."""
    with _model_lock:
        if _model is None:
            raise RuntimeError("YOLO model is not loaded")
        results = _model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

    width, height = image.size
    detections: List[Detection] = []

    for result in results:
        names = result.names
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls.item())
            if isinstance(names, dict):
                cls_name = str(names.get(class_id, class_id))
            else:
                cls_name = str(names[class_id])

            # Keep only inspection-relevant classes; drop oven/refrigerator etc.
            if cls_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            bbox = [
                max(0, min(width, int(round(x1)))),
                max(0, min(height, int(round(y1)))),
                max(0, min(width, int(round(x2)))),
                max(0, min(height, int(round(y2)))),
            ]

            detections.append(
                Detection(
                    label=cls_name,
                    confidence=round(float(box.conf.item()), 4),
                    bbox=bbox,
                )
            )

    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/detect_url", response_model=DetectResponse, tags=["detection"])
async def detect_url(payload: DetectUrlRequest) -> DetectResponse:
    image_url = payload.image_url.strip()
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url is empty")

    if image_url.lower().startswith(("http://", "https://")):
        final_url = image_url
    elif image_url.startswith("/"):
        final_url = f"{DIFY_BASE_URL.rstrip('/')}{image_url}"
    else:
        raise HTTPException(
            status_code=400,
            detail="image_url must be an absolute HTTP(S) URL or start with '/'",
        )

    # Dify runs inside Docker and may return host.docker.internal. This service
    # runs on the host, where that Docker-only hostname is not resolvable.
    final_url = final_url.replace("host.docker.internal", "127.0.0.1")

    print(f"[DETECT_URL] Final downloading URL: {final_url}", flush=True)
    image = await _download_image(final_url)

    try:
        detections = await run_in_threadpool(_infer, image)
    except Exception as exc:
        logger.exception("YOLO inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    summary: Dict[str, int] = {}
    for item in detections:
        summary[item.label] = summary.get(item.label, 0) + 1
    summary = dict(sorted(summary.items()))

    print(f"[DETECT] summary={summary}", flush=True)

    return DetectResponse(
        status="success",
        summary=summary,
        details=detections,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)