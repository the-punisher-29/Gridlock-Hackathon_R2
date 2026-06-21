from ultralytics import YOLO
from app.config import PLATE_MODEL_PATH, CONF_THRESHOLD


class PlateDetector:
    def __init__(self):
        self.model = YOLO(PLATE_MODEL_PATH)

    def detect(self, cropped_frame):
        """Returns list of plate bboxes within the given vehicle crop."""
        results = self.model(cropped_frame, conf=CONF_THRESHOLD, verbose=False)[0]
        plates = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            plates.append({
                "bbox": [x1, y1, x2, y2],
                "conf": float(box.conf[0])
            })
        return plates