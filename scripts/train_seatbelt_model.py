# scripts/train_seatbelt_model.py
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="data/seatbelt/data.yaml",
    epochs=80,
    imgsz=640,
    batch=16,
    patience=20,
    project="models/seatbelt",
    name="run1",
    device="cpu",
    augment=True,
    val=True
)