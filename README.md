# Automated Traffic Violation Detection System

An AI-powered computer vision system that automatically detects, classifies, and documents traffic violations from CCTV/traffic-pole footage — including helmet non-compliance, seatbelt non-compliance, triple riding, mobile phone usage while riding, and red-light/stop-line violations. Detected vehicles' license plates are automatically read using OCR, and every violation is logged with annotated evidence images for review.

A Streamlit web application is included for end-to-end testing: upload a video, calibrate the stop line, run detection, and review/export violation reports.

---

## Project Overview

Manual review of traffic camera footage for violations is slow, inconsistent, and doesn't scale. This project automates that process using a pipeline of object detection models, object tracking, and rule-based geometric logic, producing structured, searchable violation records with photographic evidence — ready for downstream enforcement review.

**Violations currently detected:**

| Violation | Detection Method |
|---|---|
| Helmet non-compliance | Trained YOLOv8 detector |
| Triple riding | Trained YOLOv8 detector (rider count per motorcycle) |
| Mobile phone usage (riding) | Trained YOLOv8 detector |
| Seatbelt non-compliance | Trained YOLOv8 detector |
| Red-light / stop-line violation | Rule-based geometry (stop line crossing + light state) |
| License plate recognition | Trained YOLOv8 detector + OCR (EasyOCR) |

---

## Methodology

The system splits violations into two categories, since not every violation needs a trained model:

1. **Appearance-based violations** (helmet, seatbelt, triple riding, mobile use) — these require fine-tuned object detection models trained on labeled image datasets, since they depend on visual appearance.
2. **Rule-based violations** (red-light) — these are derived from object tracking + simple geometry, requiring no additional training data.

### Pipeline Flow

```
Video Input
   │
   ▼
Frame Extraction & Preprocessing (OpenCV)
   │
   ▼
General Vehicle/Person Detection + Multi-Object Tracking
   (YOLOv8n pretrained on COCO + ByteTrack via `supervision`)
   │
   ├──► Crop: Motorcycle + Rider ──► Helmet/Triple-Riding/Mobile-Use Detector (custom YOLOv8)
   ├──► Crop: Car/Bus/Truck ──────► Seatbelt Detector (custom YOLOv8)
   ├──► Tracked Centroid ──────────► Rule Engine (stop-line crossing + light state)
   │
   ▼
On Violation Detected:
   ├──► Crop & Detect License Plate (custom YOLOv8)
   ├──► Read Plate Text (EasyOCR)
   ├──► Save Annotated Evidence Image
   └──► Log Record to SQLite Database (timestamp, violation type, confidence, plate, camera ID)
   │
   ▼
Streamlit App: Review, Filter, Visualize, Export Report (CSV)
```

---

## Models Used

| # | Model | Purpose | Base Architecture | Training Data |
|---|-------|---------|--------------------|----------------|
| 1 | General Detector | Vehicle & person detection + tracking | YOLOv8n (pretrained on COCO, used as-is) | COCO (no fine-tuning) |
| 2 | Helmet/Triple/Mobile Detector | Two-wheeler appearance violations | YOLOv8s (fine-tuned) | Roboflow Universe datasets |
| 3 | Seatbelt Detector | Car/bus/truck seatbelt compliance | YOLOv8s (fine-tuned) | Roboflow Universe datasets |
| 4 | License Plate Detector | Plate localization | YOLOv8n/s (fine-tuned) | Roboflow Universe / Kaggle |
| — | OCR Engine | Plate text recognition | EasyOCR (pretrained, not fine-tuned) | N/A |
| — | Object Tracker | Persistent vehicle IDs across frames | ByteTrack (via `supervision`) | N/A |

---

## Project Structure

```
project_root/
├── app/
│   ├── detectors/          # vehicle, helmet, seatbelt, plate detector wrappers
│   ├── ocr/                # license plate OCR
│   ├── rules/              # zone configuration + rule engine (stop-line/red-light)
│   ├── db.py               # SQLite violation records
│   ├── pipeline.py         # orchestrates the full detection pipeline
│   └── config.py           # model paths, thresholds
├── models/                 # trained model weights (.pt) — see Download section below
├── data/                   # training datasets (merged, YOLO format) — see Download section below
├── scripts/                # dataset download/merge, training, zone setup, testing utilities
├── evidence/               # auto-generated annotated violation images
├── config/                 # saved zone_config.json (stop line, light state)
├── streamlit_app.py        # main Streamlit application
├── requirements.txt
└── README.md
```

---

## Trained Models & Datasets

Trained model weights and the merged training datasets are **not included in this GitHub repository** due to size. They are hosted on Google Drive:

**Download link:** `https://drive.google.com/drive/folders/1nVTaJNPKV65fgyGePNtAi6jfZSjgCyg3?usp=sharing`

The Drive folder contains:
- `models/general/yolov8n.pt` — pretrained COCO weights
- `models/helmet_triple_mobile/best.pt` — fine-tuned helmet/triple-riding/mobile-use detector
- `models/seatbelt/best.pt` — fine-tuned seatbelt detector
- `models/license_plate/best.pt` — fine-tuned license plate detector
- `data/` — merged YOLO-format training datasets used for all fine-tuned models

**After downloading**, place the folders so they match the structure below exactly:
```
project_root/
├── models/
│   ├── general/yolov8n.pt
│   ├── helmet_triple_mobile/best.pt
│   ├── seatbelt/best.pt
│   └── license_plate/best.pt
└── data/
    ├── helmet_triple_mobile/merged/
    ├── seatbelt/merged/
    └── license_plate/
```

---

## Demo Video

A walkthrough of the Streamlit application — uploading footage, calibrating the stop line, running detection, and reviewing the violation report — is available here:

**Demo video:** `https://drive.google.com/file/d/1O7X3Mcw_fyBkw3tlDTuPeQj2fIfSHHg-/view?usp=drive_link`

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/deepak5512/Gridlock-Hackathon.git
cd traffic-violation-detection
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download model weights & datasets
Download from the Google Drive link above and place files exactly as shown in the **Trained Models & Datasets** section.

---

## How to Run

### Run the Streamlit application
```bash
streamlit run streamlit_app.py
```
Open the app at `http://localhost:8501` and:
1. **Tab 1 — Upload & Calibrate:** Upload a traffic video, set the stop line coordinates on the reference frame, choose the simulated signal state.
2. **Tab 2 — Run Detection:** Set frame sampling rate, click "Run Detection Pipeline," preview the annotated output video.
3. **Tab 3 — Results & Reports:** View the violations table, filter by violation type or plate number, browse evidence images, download the CSV report.

### Run the pipeline directly via script (no UI)
```bash
python scripts/run_test_pipeline.py
```
Processes `test/traffic.mp4` and saves annotated output + logs violations to `violations.db`.

---

## Evaluation

Each fine-tuned model is evaluated using standard object detection metrics:
```bash
python scripts/evaluate_models.py
```
Reports Precision, Recall, mAP@50, and mAP@50-95 per class. End-to-end pipeline accuracy (false positive/negative rate on flagged violations) should additionally be manually spot-checked against the evidence gallery, since per-model mAP doesn't fully capture pipeline-level correctness.

---

## Tech Stack

- **Detection/Tracking:** Ultralytics YOLOv8, `supervision` (ByteTrack)
- **OCR:** EasyOCR
- **Backend Pipeline:** Python, OpenCV
- **Frontend:** Streamlit
- **Training Data Source:** Roboflow Universe, Kaggle
