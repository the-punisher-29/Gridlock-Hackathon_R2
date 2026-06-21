import easyocr
import cv2
import re

reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if available

PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$')


def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(gray)
    return gray


def read_plate(plate_img):
    """
    plate_img: cropped numpy array of the license plate region
    Returns: (text, confidence) or (None, 0) if nothing readable
    """
    if plate_img.size == 0:
        return None, 0.0

    processed = preprocess_plate(plate_img)
    results = reader.readtext(processed)

    if not results:
        return None, 0.0

    # pick the highest-confidence text segment, concatenate if multiple lines
    full_text = "".join([r[1] for r in results]).upper().replace(" ", "")
    avg_conf = sum(r[2] for r in results) / len(results)

    is_valid_format = bool(PLATE_REGEX.match(full_text))

    return full_text, avg_conf if is_valid_format else avg_conf * 0.5