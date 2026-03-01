"""
consumer.py
============
Reads frames from Kafka topic 'video-frames',
sends each frame to the inference API,
and prints/stores the detection results.

Think of this as the "brain" that processes what the camera sees.
"""

import json
import time
import logging
import argparse
import requests
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_consumer(bootstrap_servers: str, topic: str, retries: int = 5):
    """Create Kafka consumer with retry logic"""
    for attempt in range(retries):
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest",      # only process new messages
                group_id="detection-consumers",  # consumer group for scaling
                max_partition_fetch_bytes=10485760,  # 10MB
            )
            logger.info(f"Connected to Kafka at {bootstrap_servers}")
            return consumer
        except NoBrokersAvailable:
            logger.warning(f"Kafka not ready, retrying {attempt+1}/{retries}...")
            time.sleep(3)
    raise RuntimeError("Could not connect to Kafka after retries")


def process_frames(
    bootstrap_servers: str = "localhost:9092",
    topic: str = "video-frames",
    inference_url: str = "http://localhost:8000/detect",
    confidence: float = 0.5,
):
    """
    Main processing loop:
    1. Read frame from Kafka
    2. Send to inference API
    3. Print detections
    4. Repeat
    """
    consumer = create_consumer(bootstrap_servers, topic)
    logger.info(f"Listening on topic: {topic}")
    logger.info(f"Sending to inference: {inference_url}")

    processed = 0
    total_latency = 0.0

    try:
        for message in consumer:
            frame_data = message.value
            frame_id = frame_data.get("frame_id", "?")
            captured_at = frame_data.get("timestamp", time.time())

            # Send to inference API
            start = time.time()
            try:
                response = requests.post(
                    inference_url,
                    json={
                        "image": frame_data["image"],
                        "confidence": confidence,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()

            except requests.exceptions.RequestException as e:
                logger.error(f"Inference API error: {e}")
                continue

            # Calculate end-to-end latency
            e2e_latency = (time.time() - captured_at) * 1000  # ms
            inference_time = result["inference_time_ms"]
            total_latency += e2e_latency
            processed += 1

            # Print results
            detections = result["detections"]
            if detections:
                classes = [f"{d['class_name']}({d['confidence']*100:.0f}%)"
                          for d in detections]
                logger.info(
                    f"Frame {frame_id} | "
                    f"Found: {', '.join(classes)} | "
                    f"Inference: {inference_time:.0f}ms | "
                    f"E2E latency: {e2e_latency:.0f}ms"
                )
            else:
                logger.info(f"Frame {frame_id} | No objects detected")

            # Print stats every 50 frames
            if processed % 50 == 0:
                avg_latency = total_latency / processed
                logger.info(f"Stats: {processed} frames | Avg latency: {avg_latency:.0f}ms")

    except KeyboardInterrupt:
        logger.info("Consumer stopped by user")
    finally:
        consumer.close()
        if processed > 0:
            logger.info(f"Processed {processed} frames | Avg latency: {total_latency/processed:.0f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detection consumer from Kafka")
    parser.add_argument("--servers", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic", default="video-frames", help="Kafka topic")
    parser.add_argument("--inference", default="http://localhost:8000/detect", help="Inference API URL")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    args = parser.parse_args()

    process_frames(args.servers, args.topic, args.inference, args.confidence)
