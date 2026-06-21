GENERAL_MODEL_PATH = "models/general/yolov8n.pt"
HELMET_MODEL_PATH = "models/helmet_triple_mobile/best.pt"
SEATBELT_MODEL_PATH = "models/seatbelt/best.pt"
PLATE_MODEL_PATH = "models/license_plate/best.pt"

VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_CLASS_ID = 0

CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5