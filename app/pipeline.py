import os
import cv2
import datetime
from app.detectors.vehicle_detector import VehicleDetector
from app.detectors.helmet_detector import HelmetDetector
from app.detectors.seatbelt_detector import SeatbeltDetector
from app.detectors.plate_detector import PlateDetector
from app.ocr.plate_ocr import read_plate
from app.rules.rule_engine import RuleEngine
from app.rules.zone_config import ZoneConfig
from app.db import log_violation

EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)


class TrafficViolationPipeline:
    def __init__(self, camera_id="default"):
        self.camera_id = camera_id
        self.vehicle_detector = VehicleDetector()
        self.helmet_detector = HelmetDetector()
        self.seatbelt_detector = SeatbeltDetector()
        self.plate_detector = PlateDetector()
        self.zone_config = ZoneConfig()
        self.rule_engine = RuleEngine(self.zone_config)
        self.already_flagged = set()

    def process_frame(self, frame, frame_idx=0):
        detections = self.vehicle_detector.detect_and_track(frame)
        annotated = frame.copy()

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            class_id = int(detections.class_id[i])
            class_name = self.vehicle_detector.class_name(class_id)

            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            self.rule_engine.update_track(tracker_id, centroid)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            violations_found = []

            # --- Appearance violations ---
            if class_name == "motorcycle":
                helmet_result = self.helmet_detector.evaluate_violations(crop)
                if helmet_result["no_helmet"]:
                    violations_found.append(("no_helmet", 0.8))
                if helmet_result["triple_riding"]:
                    violations_found.append(("triple_riding", 0.8))
                if helmet_result["using_mobile"]:
                    violations_found.append(("using_mobile", 0.7))

            if class_name in ("car", "bus", "truck"):
                seatbelt_result = self.seatbelt_detector.evaluate_violation(crop)
                if seatbelt_result["no_seatbelt"]:
                    violations_found.append(("no_seatbelt", 0.7))

            # --- Rule-based violations (wrong-side removed) ---
            if self.rule_engine.check_red_light_violation(tracker_id):
                violations_found.append(("red_light_violation", 0.75))

            # --- Plate + OCR ---
            for v_type, v_conf in violations_found:
                key = (tracker_id, v_type)
                if key in self.already_flagged:
                    continue
                self.already_flagged.add(key)

                plate_text, plate_conf = None, 0.0
                plates = self.plate_detector.detect(crop)
                if plates:
                    px1, py1, px2, py2 = [int(v) for v in plates[0]["bbox"]]
                    plate_crop = crop[py1:py2, px1:px2]
                    plate_text, plate_conf = read_plate(plate_crop)

                evidence_frame = frame.copy()
                cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{v_type} ({plate_text or 'unreadable'})"
                cv2.putText(evidence_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_path = os.path.join(EVIDENCE_DIR, f"{v_type}_{tracker_id}_{ts}.jpg")
                cv2.imwrite(img_path, evidence_frame)

                log_violation(
                    camera_id=self.camera_id,
                    tracker_id=tracker_id,
                    violation_type=v_type,
                    confidence=v_conf,
                    plate_number=plate_text,
                    plate_confidence=plate_conf,
                    evidence_image_path=img_path
                )

            color = (0, 0, 255) if violations_found else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID{tracker_id} {class_name}"
            if violations_found:
                label += " | " + ",".join(v[0] for v in violations_found)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return annotated

    def process_video(self, video_path, output_path="output_annotated.mp4", sample_every_n=1):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every_n == 0:
                annotated = self.process_frame(frame, frame_idx)
                out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"Done. Output saved to {output_path}")