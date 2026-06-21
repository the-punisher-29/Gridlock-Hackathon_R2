from ultralytics import YOLO
from app.config import HELMET_MODEL_PATH, CONF_THRESHOLD


class HelmetDetector:
    def __init__(self):
        self.model = YOLO(HELMET_MODEL_PATH)
        self.class_names = self.model.names  # dict {id: name}

    def detect(self, cropped_frame):
        """
        Run on a cropped region containing a motorcycle + rider(s).
        Returns list of dicts: [{class_name, conf, bbox}, ...]
        """
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

    def evaluate_violations(self, cropped_frame):
        """
        Higher-level helper: returns a dict summarizing rider violations.
        """
        dets = self.detect(cropped_frame)
        rider_count = sum(1 for d in dets if d["class_name"] in ("with_helmet", "without_helmet", "rider"))
        no_helmet = any(d["class_name"] == "without_helmet" for d in dets)
        mobile_use = any(d["class_name"] == "using_mobile" for d in dets)

        return {
            "no_helmet": no_helmet,
            "triple_riding": rider_count > 2,
            "using_mobile": mobile_use,
            "rider_count": rider_count,
            "raw_detections": dets
        }