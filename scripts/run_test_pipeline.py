# scripts/run_test_pipeline.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pipeline import TrafficViolationPipeline

pipeline = TrafficViolationPipeline(camera_id="cam1")
pipeline.process_video(
    video_path="test/traffic.mp4",
    output_path="test/output_annotated.mp4",
    sample_every_n=2   # process every 2nd frame for speed; set to 1 for full accuracy
)