import json
import time
import paho.mqtt.client as mqtt
import os

CONFIG_PATH = "/Users/uma/Downloads/smartpark/node/data/config.json"
JSON_PATH = "/Users/uma/Downloads/smartpark/node/data/ml01_cam1_calibrated.json"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def load_json():
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"JSON file not found at {JSON_PATH}")
    with open(JSON_PATH, "r") as f:
        return json.load(f)

def main():
    config = load_config()
    data = load_json()

    camera_id = config["camera_id"]
    lot_id = config["lot_id"]

    broker = "100.95.3.23"
    port = 1883
    topic = f"smartpark/test"

    client = mqtt.Client()
    client.connect(broker, port, 60)
    print(f"[TEST] Connected to MQTT broker at {broker}:{port}")
    print(f"[TEST] Publishing to topic: {topic}")

    try:
        payload = load_json()
        client.publish(topic, json.dumps(payload), qos=1)
        print(f"[TEST] Published JSON from {os.path.basename(JSON_PATH)}")

    except Exception as e:
        print(f"[ERROR] Failed to publish: {e}")

    finally:
        client.disconnect()
        print("[TEST] Disconnected from MQTT broker.")

if __name__ == "__main__":
    main()
