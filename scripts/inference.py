# scripts/inference.py
from ultralytics import YOLO
from shapely.geometry import box
import os


MODEL_DIR = "/home/smartpark/node/models/"

def run_yolo(frame, lot_name: str, conf: float = 0.5):
    model_path = os.path.join(MODEL_DIR, f"{lot_name}_model.pt")
    model = YOLO(model_path)

    results = model.predict(frame, conf=conf)

    for r in results:
        masks = r.masks
        classes = r.boxes.cls
        orig_shape = r.masks.orig_shape

    return masks, classes, orig_shape
