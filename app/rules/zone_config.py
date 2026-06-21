"""
Holds per-camera calibration data: stop line + traffic light state.
(Lane/wrong-side detection removed for now.)
"""
import json
import os


class ZoneConfig:
    def __init__(self, config_path="config/zone_config.json"):
        self.config_path = config_path
        self.data = {
            "stop_line": None,                 # [(x1, y1), (x2, y2)]
            "traffic_light_state": "unknown"   # "red" / "yellow" / "green" / "unknown"
        }
        if os.path.exists(config_path):
            self.load()

    def set_stop_line(self, p1, p2):
        self.data["stop_line"] = [list(p1), list(p2)]

    def set_light_state(self, state):
        assert state in ("red", "yellow", "green", "unknown")
        self.data["traffic_light_state"] = state

    def save(self):
        dirname = os.path.dirname(self.config_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def load(self):
        with open(self.config_path) as f:
            loaded = json.load(f)
        self.data.update(loaded)