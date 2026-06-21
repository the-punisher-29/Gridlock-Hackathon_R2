from ultralytics import YOLO
from app.config import SEATBELT_MODEL_PATH, CONF_THRESHOLD


class SeatbeltDetector:
    def __init__(self):
        self.model = YOLO(SEATBELT_MODEL_PATH)
        self.class_names = self.model.names

    def detect(self, cropped_frame):
        results = self.model(cropped_frame, conf=CONF_THRESHOLD, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class_name": self.class_names[cls_id],
                "conf": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })
        return detections

    def evaluate_violation(self, cropped_frame):
        dets = self.detect(cropped_frame)
        has_seatbelt = any(d["class_name"] == "seatbelt" for d in dets)
        return {
            "no_seatbelt": not has_seatbelt and len(dets) == 0 is False,  # only flag if windshield/driver was visible
            "detected": len(dets) > 0,
            "raw_detections": dets
        }