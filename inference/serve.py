import time
import base64
import io
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

model = None
model_load_time = None
request_count = 0
total_inference_time = 0.0

class DetectionRequest(BaseModel):
    image: str
    confidence: float = 0.5

class BoundingBox(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

class DetectionResponse(BaseModel):
    detections: list[BoundingBox]
    inference_time_ms: float
    image_width: int
    image_height: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_load_time
    model_path = Path("/app/model/best.pt")
    if not model_path.exists():
        model_path = Path("yolov8s.pt")
    logger.info(f"Loading model from: {model_path}")
    start = time.time()
    model = YOLO(str(model_path))
    model_load_time = time.time() - start
    device = "GPU" if torch.cuda.is_available() else "CPU"
    logger.info(f"Model loaded in {model_load_time:.2f}s on {device}")
    yield
    logger.info("Shutting down")

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="Real-time object detection using YOLOv8",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "model_load_time_s": round(model_load_time, 2) if model_load_time else None
    }

@app.get("/metrics")
def metrics():
    avg_inference = (total_inference_time / request_count) if request_count > 0 else 0
    return {
        "total_requests": request_count,
        "avg_inference_time_ms": round(avg_inference, 2),
        "model_loaded": model is not None,
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    global request_count, total_inference_time
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_width, img_height = image.size
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    start = time.time()
    results = model(image, conf=request.confidence, verbose=False)
    inference_time = (time.time() - start) * 1000
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(BoundingBox(
                class_id=int(box.cls[0]),
                class_name=model.names[int(box.cls[0])],
                confidence=round(float(box.conf[0]), 3),
                x1=round(x1, 1), y1=round(y1, 1),
                x2=round(x2, 1), y2=round(y2, 1),
            ))
    request_count += 1
    total_inference_time += inference_time
    logger.info(f"Detected {len(detections)} objects in {inference_time:.1f}ms")
    return DetectionResponse(
        detections=detections,
        inference_time_ms=round(inference_time, 2),
        image_width=img_width,
        image_height=img_height,
    )

@app.get("/classes")
def get_classes():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"classes": model.names}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
