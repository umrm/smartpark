# scripts/match.py
import numpy as np
import cv2
import json
from shapely.geometry import Polygon


def mask_to_polygons(masks, classes, orig_shape):
    h, w = orig_shape
    all_polygons = []

    mask_array = masks.data.cpu().numpy()
    class_ids = classes.cpu().numpy()

    for mask, cls_id in zip(mask_array, class_ids):

        mask_resized = cv2.resize(mask, (w, h))
        mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            polygon_points = [(pt[0][0], pt[0][1]) for pt in approx]
            all_polygons.append({
                "class_id": int(cls_id),
                "points": polygon_points
            })

    pred_polygons = []

    for p in all_polygons:
        pred_polygons.append({
            "class_id": p["class_id"],
            "polygon": Polygon(p["points"])
        })

    return pred_polygons

def process_map(spaces):
    master_spaces = {}

    for space in spaces:
        space_id = space['id']
        pts = space['points']
        master_spaces[space_id] = Polygon(pts)

    return master_spaces


def match_detections_to_rois(master_spaces, pred_polygons,iou_threshold):
    space_labels = {}

    for space_id, master_poly in master_spaces.items():
        best_iou = 0
        best_class = None
        for pred in pred_polygons:
            pred_poly = pred["polygon"]
            if not master_poly.is_valid or not pred_poly.is_valid:
                continue
            inter_area = master_poly.intersection(pred_poly).area
            union_area = master_poly.union(pred_poly).area
            iou = inter_area / union_area
            if iou > best_iou:
                best_iou = iou
                best_class = pred["class_id"]
        
        if best_iou >= iou_threshold:
            space_labels[space_id] = best_class
        else:
            space_labels[space_id] = None
 
    return space_labels
