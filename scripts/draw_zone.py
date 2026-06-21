import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from app.rules.zone_config import ZoneConfig


def draw_zone_overlay(frame_path, output_path="test/zone_overlay.jpg"):
    zc = ZoneConfig()
    frame = cv2.imread(frame_path)

    if zc.data["stop_line"]:
        p1, p2 = zc.data["stop_line"]
        p1 = tuple(map(int, p1))
        p2 = tuple(map(int, p2))
        cv2.line(frame, p1, p2, (0, 0, 255), 4)
        cv2.putText(frame, f"STOP LINE (light: {zc.data['traffic_light_state']})",
                    (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(output_path, frame)
    print(f"Saved zone overlay to {output_path}")


if __name__ == "__main__":
    draw_zone_overlay("test/sample_frame.jpg")