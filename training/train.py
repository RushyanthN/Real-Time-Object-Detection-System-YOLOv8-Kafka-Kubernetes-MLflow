import mlflow
import yaml
import torch
import logging
from ultralytics import YOLO
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path="training/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def train(config_path="training/config.yaml"):
    config = load_config(config_path)

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        device = str(config["training"]["device"])
    else:
        logger.warning("No GPU — using CPU")
        device = "cpu"

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="yolov8s-coco128") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        mlflow.log_params({
            "model":      config["model"]["architecture"],
            "epochs":     config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "lr":         config["training"]["learning_rate"],
            "input_size": config["model"]["input_size"],
            "dataset":    config["dataset"]["name"],
            "device":     device,
        })

        model = YOLO(f"{config['model']['architecture']}.pt")
        logger.info("Model loaded. Starting training...")

        results = model.train(
            data=config["dataset"]["name"],
            epochs=config["training"]["epochs"],
            batch=config["training"]["batch_size"],
            imgsz=config["model"]["input_size"],
            lr0=config["training"]["learning_rate"],
            patience=config["training"]["patience"],
            device=device,
            workers=config["training"]["workers"],
            project="logs/runs",
            name="yolov8s-detection",
            exist_ok=True,
            verbose=True,
        )

        metrics = results.results_dict
        mlflow.log_metrics({
            "mAP50":     metrics.get("metrics/mAP50(B)", 0),
            "mAP50_95":  metrics.get("metrics/mAP50-95(B)", 0),
            "precision": metrics.get("metrics/precision(B)", 0),
            "recall":    metrics.get("metrics/recall(B)", 0),
            "box_loss":  metrics.get("train/box_loss", 0),
            "cls_loss":  metrics.get("train/cls_loss", 0),
        })

        best_model = Path("logs/runs/yolov8s-detection/weights/best.pt")
        if best_model.exists():
            mlflow.log_artifact(str(best_model), artifact_path="model")
            logger.info("Best model saved to MLflow ✓")

        logger.info("=" * 40)
        logger.info("TRAINING COMPLETE")
        logger.info(f"mAP50:     {metrics.get('metrics/mAP50(B)', 0):.4f}")
        logger.info(f"mAP50-95:  {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        logger.info(f"Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        logger.info(f"Recall:    {metrics.get('metrics/recall(B)', 0):.4f}")
        logger.info("=" * 40)

if __name__ == "__main__":
    train()
