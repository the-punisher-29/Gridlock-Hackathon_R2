# scripts/evaluate_models.py
from ultralytics import YOLO

for name, path in [
    ("helmet", "models/helmet_triple_mobile/run1/weights/best.pt"),
    ("seatbelt", "models/seatbelt/run1/weights/best.pt"),
    ("license_plate", "models/license_plate/run1/weights/best.pt")
]:
    model = YOLO(path)
    metrics = model.val()
    print(f"\n{name} -> mAP50: {metrics.box.map50:.3f}, mAP50-95: {metrics.box.map:.3f}")