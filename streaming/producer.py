"""
producer.py
============
Reads a video file or webcam, extracts frames,
and sends them to Kafka topic 'video-frames'.

Think of this as the "camera" side of the pipeline.
"""

import base64
import time
import json
import logging
import argparse
import cv2
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_producer(bootstrap_servers: str, retries: int = 5):
    """Create Kafka producer with retry logic"""
    for attempt in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                max_request_size=10485760,  # 10MB max message size
            )
            logger.info(f"Connected to Kafka at {bootstrap_servers}")
            return producer
        except NoBrokersAvailable:
            logger.warning(f"Kafka not ready, retrying {attempt+1}/{retries}...")
            time.sleep(3)
    raise RuntimeError("Could not connect to Kafka after retries")


def stream_video(
    source,                          # video file path or 0 for webcam
    bootstrap_servers: str = "localhost:9092",
    topic: str = "video-frames",
    fps_limit: int = 5,              # max frames per second to send
    resize_width: int = 640,
):
    """
    Main streaming loop:
    1. Open video source
    2. Read frame
    3. Encode to base64
    4. Send to Kafka
    5. Repeat
    """
    producer = create_producer(bootstrap_servers)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    logger.info(f"Streaming from: {source}")
    logger.info(f"Topic: {topic} | FPS limit: {fps_limit}")

    frame_count = 0
    frame_interval = 1.0 / fps_limit  # seconds between frames

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                logger.info("End of video — restarting from beginning")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
                continue

            # Resize frame to reduce message size
            height = int(frame.shape[0] * resize_width / frame.shape[1])
            frame = cv2.resize(frame, (resize_width, height))

            # Encode frame as JPEG then base64
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            # Build message
            message = {
                "frame_id": frame_count,
                "timestamp": time.time(),
                "width": resize_width,
                "height": height,
                "image": frame_b64,
            }

            # Send to Kafka
            producer.send(topic, value=message)
            frame_count += 1

            if frame_count % 10 == 0:
                logger.info(f"Sent {frame_count} frames to Kafka topic '{topic}'")

            # Rate limiting
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Producer stopped by user")
    finally:
        producer.flush()
        producer.close()
        cap.release()
        logger.info(f"Total frames sent: {frame_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video frame producer for Kafka")
    parser.add_argument("--source", default="test.mp4", help="Video file or 0 for webcam")
    parser.add_argument("--servers", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic", default="video-frames", help="Kafka topic name")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second to send")
    args = parser.parse_args()

    # Convert "0" string to integer for webcam
    source = int(args.source) if args.source == "0" else args.source
    stream_video(source, args.servers, args.topic, args.fps)
