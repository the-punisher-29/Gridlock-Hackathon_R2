# scripts/train_plate_model.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # plate is a single simple class, nano is enough

model.train(
    data="data/license_plate/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="models/license_plate",
    name="run1",
    device="cpu"
)