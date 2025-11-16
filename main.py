import time
import json
import cv2
import requests
import paho.mqtt.client as mqtt  
from scripts.capture import Camera
from scripts.inference import run_yolo
from scripts.match import mask_to_polygons, process_map, match_detections_to_rois

CONFIG_PATH = "/home/smartpark/config/config.json"

def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in config file")
    
def load_spaces():
    spaces_path = f"/home/smartpark/data/ml01.master.json"
    try:
        with open(spaces_path, "r") as f:
            data = json.load(f)
            return data["spaces"]
    except FileNotFoundError:
        raise FileNotFoundError(f"Spaces file not found at {spaces_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {spaces_path}")

def send_to_backend(camera_id, lot_id, status_data, config):
    """
    Sends parking space status to the backend server via HTTP or MQTT.
    """
    payload = {
        "camera_id": camera_id,
        "lot_name": lot_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spaces": status_data
    }
    payload_json = json.dumps(payload)  # Prepare JSON for both

    try:
        topic = f"smartpark/nodes/{camera_id}/status"
        client = mqtt.Client()
        client.connect(config["mqtt_broker"], config["mqtt_port"], 60)
        client.publish(topic, payload_json, qos=1)
        client.disconnect()
        print(f"[MQTT] Published update → {topic}")
    except Exception as e:
        print(f"[ERROR] MQTT publish failed: {e}")

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

    camera = Camera(resolution) 
    time.sleep(2)

    try:
        while True:
            frame = camera.capture_frame()

            masks, classes, orig_shape = run_yolo(frame, lot_id, conf=conf_threshold)
            pred_polygons = mask_to_polygons(masks, classes, orig_shape)
            master_spaces = process_map(spaces)
            status_data = match_detections_to_rois(master_spaces, pred_polygons, iou_threshold) 

            print(f"[INFO] Status: {status_data}")

            send_to_backend(camera_id, lot_id, status_data, config) 

            time.sleep(interval)

    except KeyboardInterrupt:
        print("Stopping node...")

    finally:
        camera.stop()

if __name__ == "__main__":
    main()