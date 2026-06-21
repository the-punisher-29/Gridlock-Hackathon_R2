import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rules.zone_config import ZoneConfig

zc = ZoneConfig()

# Stop line — adjust to your actual frame coordinates (signal-pole front view)
zc.set_stop_line((0, 400), (1280, 420))
zc.set_light_state("red")

zc.save()
print("Zone config saved to config/zone_config.json")