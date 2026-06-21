from ultralytics import YOLO
import supervision as sv
from app.config import GENERAL_MODEL_PATH, VEHICLE_CLASS_IDS, PERSON_CLASS_ID, CONF_THRESHOLD


class VehicleDetector:
    def __init__(self):
        self.model = YOLO(GENERAL_MODEL_PATH)
        self.tracker = sv.ByteTrack()

    def detect_and_track(self, frame):
        """
        Returns a supervision Detections object with:
        - xyxy boxes
        - class_id (COCO ids)
        - tracker_id (persistent ID across frames)
        """
        results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # keep only vehicles + persons
        keep_ids = set(VEHICLE_CLASS_IDS.keys()) | {PERSON_CLASS_ID}
        mask = [cls_id in keep_ids for cls_id in detections.class_id]
        detections = detections[mask]

        detections = self.tracker.update_with_detections(detections)
        return detections

    def class_name(self, class_id):
        if class_id == PERSON_CLASS_ID:
            return "person"
        return VEHICLE_CLASS_IDS.get(class_id, "unknown")