"""
Car Plate Detection Utilities
Uses YOLO model for license plate detection and analysis.
"""

import numpy as np
import cv2
import os
from ultralytics import YOLO

class PlateDetector:
    """
    Detects license plates in vehicle images using a trained YOLO model.
    Uses YOLO-format labels when available, falls back to model detection.
    """

    def __init__(self, model_path="Models/best.pt", labels_dir=None):
        self.labels_dir = labels_dir
        self.model_loaded = False
        
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.model_loaded = True
            except Exception as e:
                print(f"⚠️ Error loading YOLO model: {e}")
        else:
            print(f"⚠️ Warning: Model {model_path} not found.")

    def _parse_yolo_label(self, label_path, img_h, img_w):
        """Parse a YOLO-format label file and return bounding boxes."""
        boxes = []
        if not os.path.exists(label_path):
            return boxes
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_c, y_c, box_w, box_h = map(float, parts)
                    x1 = int((x_c - box_w / 2) * img_w)
                    y1 = int((y_c - box_h / 2) * img_h)
                    w  = int(box_w * img_w)
                    h  = int(box_h * img_h)
                    boxes.append({
                        "class_id":   int(class_id),
                        "x":          max(x1, 0),
                        "y":          max(y1, 0),
                        "w":          w,
                        "h":          h,
                        "confidence": 1.0,
                        "source":     "ground_truth",
                    })
        return boxes

    def detect_plates_yolo(self, image):
        """
        Detect plates using the loaded YOLO model.
        Returns a list of dicts with 'x', 'y', 'w', 'h', 'confidence', 'source'.
        """
        if not self.model_loaded:
            return []
            
        results = self.model(image, conf=0.25, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            
            detections.append({
                "class_id": 0,
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "confidence": round(conf, 3),
                "source": "yolo_model"
            })
            
        # Sort by confidence descending
        detections.sort(key=lambda c: c["confidence"], reverse=True)
        return detections

    def detect(self, image, image_name=None):
        """
        Detect plates. Uses ground-truth labels if available,
        otherwise falls back to YOLO detection.

        Args:
            image:      RGB numpy array
            image_name: filename string (used to look up label file)
        """
        if self.labels_dir and image_name:
            label_name = os.path.splitext(image_name)[0] + ".txt"
            label_path = os.path.join(self.labels_dir, label_name)
            h, w       = image.shape[:2]
            gt_boxes   = self._parse_yolo_label(label_path, h, w)
            if gt_boxes:
                return gt_boxes

        return self.detect_plates_yolo(image)

    def annotate_image(self, image, detections):
        """
        Draw detection boxes on the image.
        The highest-confidence detection (index 0) is highlighted in gold
        with a '★ Best Match' label.
        """
        annotated = image.copy()

        for i, det in enumerate(detections):
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            conf        = det["confidence"]
            source      = det.get("source", "")
            is_best     = (i == 0)

            # ── Colours in RGB order (image array is RGB) ─────────────────
            if source == "ground_truth":
                color     = (0, 200, 80)     # green
                label     = "Plate (GT)"
                thickness = 2
            elif is_best:
                color     = (255, 190, 0)    # gold  — "this is the real plate"
                label     = f"\u2605 Best Match {conf:.0%}"
                thickness = 3               # thicker border draws the eye
            else:
                color     = (100, 140, 220)  # dim blue for secondary candidates
                label     = f"Candidate {conf:.0%}"
                thickness = 1

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

            # ── Label background — clamp top edge so text is always visible ─
            font_scale = 0.65 if is_best else 0.55
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            label_top    = max(y - th - 8, 0)
            label_bottom = label_top + th + 8

            cv2.rectangle(
                annotated,
                (x, label_top),
                (x + tw + 6, label_bottom),
                color, -1,
            )
            cv2.putText(
                annotated, label,
                (x + 3, label_bottom - 4),      # text baseline anchored to bottom
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0) if is_best else (255, 255, 255),
                1, cv2.LINE_AA,
            )

        return annotated

    def extract_plate_crops(self, image, detections):
        """Crop and return individual plate regions (RGB)."""
        crops = []
        for det in detections:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            crop = image[y: y + h, x: x + w]
            if crop.size > 0:
                crops.append(crop)
        return crops


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_image_stats(image):
    """Compute basic image statistics for display."""
    h, w   = image.shape[:2]
    gray   = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges  = cv2.Canny(gray, 50, 150)

    return {
        "width":           w,
        "height":          h,
        "channels":        image.shape[2] if len(image.shape) == 3 else 1,
        "mean_brightness": round(float(np.mean(gray)), 1),
        "contrast":        round(float(np.std(gray)), 1),
        "edge_density":    round(float(np.sum(edges > 0) / (h * w)) * 100, 2),
        "sharpness":       round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
    }

def _compute_iou(box1, box2):
    """Compute Intersection over Union between two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa    = max(x1, x2)
    ya    = max(y1, y2)
    xb    = min(x1 + w1, x2 + w2)
    yb    = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0