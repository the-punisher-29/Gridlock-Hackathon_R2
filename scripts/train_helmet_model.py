# scripts/train_helmet_model.py
from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # small model, good accuracy/speed tradeoff

model.train(
    data="data/helmet_triple_mobile/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,           # early stopping
    project="models/helmet_triple_mobile",
    name="run1",
    device="cpu",              # set to 'cpu' if no GPU
    augment=True,
    mosaic=1.0,
    val=True
)