# scripts/check_video_info.py
import cv2

cap = cv2.VideoCapture("test/traffic.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Width: {w}, Height: {h}, FPS: {fps}, Total frames: {total_frames}")
cap.release()