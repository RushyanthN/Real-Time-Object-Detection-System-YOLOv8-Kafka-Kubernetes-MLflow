# Real-Time Object Detection System

**Event-driven object detection pipeline using YOLOv8, Kafka,Docker, Kubernetes,Prometheus ,Grafana and MLflow for scalable video inference.**

---

## Overview

This project implements an end-to-end ML pipeline that streams video frames through Kafka, runs real-time object detection with YOLOv8, and deploys the inference service on Kubernetes with auto-scaling.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Video     │────▶│   Kafka    │────▶│   YOLOv8         │────▶│   Detection     │
│   Producer  │     │   (Broker) │     │   Inference API  │     │   Results       │
└─────────────┘     └─────────────┘     └──────────────────┘     └─────────────────┘
      │                     │                      │
      │                     │                      │
   Webcam/File         video-frames           FastAPI :8000
```

- **Producer**: Captures video (webcam or file), encodes frames to base64, publishes to Kafka
- **Kafka**: Message broker for decoupled, scalable frame streaming
- **Inference**: FastAPI service with YOLOv8 model for detection
- **Consumer**: Subscribes to Kafka, sends frames to inference API, logs results

## Tech Stack

| Component   | Technology                    |
|------------|-------------------------------|
| Model      | YOLOv8 (Ultralytics)          |
| ML Ops     | MLflow                        |
| Messaging  | Apache Kafka + Zookeeper      |
| API        | FastAPI                       |
| Container  | Docker, Docker Compose        |
| Orchestration | Kubernetes, HPA           |

## Project Structure

```
├── training/
│   ├── config.yaml          # Training & inference config
│   └── train.py             # YOLOv8 training with MLflow logging
├── inference/
│   ├── serve.py             # FastAPI detection API
│   ├── Dockerfile           # Inference container
│   └── requirements.txt
├── streaming/
│   ├── producer.py          # Kafka frame producer
│   └── consumer.py          # Kafka frame consumer
├── k8s/
│   ├── inference-deployment.yaml
│   ├── inference-service.yaml
│   └── inference-hpa.yaml   # Horizontal Pod Autoscaler (1–3 replicas)
└── docker-compose.yml       # Local dev: Zookeeper + Kafka + Inference
```

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- [Optional] Kubernetes cluster (Minikube, Kind, etc.)
- [Optional] GPU with CUDA for faster inference

## Quick Start

### 1. Start Infrastructure (Kafka + Zookeeper + Inference)

```bash
# Build inference image (place best.pt in inference/model/ first, or use yolov8s.pt)
docker build -t yolov8-inference:v1 ./inference

# Start all services
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

### 2. Run the Pipeline

**Terminal 1 – Producer** (stream video frames to Kafka):
```bash
pip install kafka-python opencv-python
python streaming/producer.py --source test.mp4  # or --source 0 for webcam
```

**Terminal 2 – Consumer** (consume frames, call inference API):
```bash
pip install kafka-python requests
python streaming/consumer.py --inference http://localhost:8000/detect
```

### 3. Train a Custom Model (Optional)

```bash
pip install ultralytics mlflow pyyaml torch
# Ensure MLflow server is running at http://localhost:5000
python training/train.py
```

Training logs mAP50, mAP50-95, precision, recall to MLflow. Best model saved to `logs/runs/yolov8s-detection/weights/best.pt`.

## Configuration

Key settings in `training/config.yaml`:

| Section     | Key                | Default        | Description                    |
|-------------|--------------------|----------------|--------------------------------|
| Model       | architecture       | yolov8s        | YOLOv8 variant                 |
| Model       | input_size         | 640            | Input resolution               |
| Kafka       | bootstrap_servers  | localhost:9092 | Kafka broker                   |
| Kafka       | topic_frames       | video-frames   | Producer topic                 |
| Inference   | confidence_threshold | 0.5         | Detection confidence           |
| MLflow      | tracking_uri       | localhost:5000 | MLflow server                  |

## API Endpoints

| Method | Endpoint   | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/health`  | Service health, model status         |
| GET    | `/metrics` | Request count, avg inference time    |
| POST   | `/detect`  | Object detection (body: `image` base64) |
| GET    | `/classes` | List of COCO class names             |

## Kubernetes Deployment

```bash
# Deploy inference service
kubectl apply -f k8s/inference-deployment.yaml
kubectl apply -f k8s/inference-service.yaml
kubectl apply -f k8s/inference-hpa.yaml

# Scale: 1–3 replicas based on 70% CPU utilization
```

Resource limits: 500m–1000m CPU, 512Mi–1Gi memory per pod.


