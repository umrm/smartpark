import time
import json
import cv2
import requests
import paho.mqtt.client as mqtt  
# from scripts.capture import Camera
from scripts.inference import run_yolo
from scripts.match import mask_to_polygons, process_map, match_detections_to_rois
from ultralytics import YOLO

CONFIG_PATH = "/Users/uma/Downloads/smartpark/node/data/config.json"

def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in config file")
    
def load_spaces():
    spaces_path = f"/Users/uma/Downloads/smartpark/node/data/ml01_master.json"
    try:
        with open(spaces_path, "r") as f:
            data = json.load(f)
            return data["spaces"]
    except FileNotFoundError:
        raise FileNotFoundError(f"Spaces file not found at {spaces_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {spaces_path}")

def main():
    config = load_config()
    lot_id = config["lot_id"] 
    camera_id = config["camera_id"]
    resolution = tuple(config["capture_resolution"])
    conf_threshold = config["confidence_threshold"]
    iou_threshold = config["iou_threshold"]
    interval = config["interval_seconds"]
    spaces = load_spaces()

    print(f"Starting SmartPark Node for Lot '{lot_id}' (Camera ID: {camera_id})")

    model = YOLO("/Users/uma/Downloads/yolo/weights/best.pt")
    img_path = "/Users/uma/Downloads/pklot_seg/test/images/image_081_jpg.rf.6605d44afdd29b94a2cd1814b7857321.jpg"
    results = model.predict(img_path, conf=0.5)
    masks = results[0].masks
    classes = results[0].boxes.cls
    orig_shape = results[0].masks.orig_shape

    pred_polygons = mask_to_polygons(masks, classes, orig_shape)
    master_spaces = process_map(spaces)
    status_data = match_detections_to_rois(master_spaces, pred_polygons, iou_threshold) 

    print(f"[INFO] Status: {status_data}")

if __name__ == "__main__":
    main()