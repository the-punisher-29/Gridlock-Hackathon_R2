# scripts/extract_sample_frame.py
import cv2

cap = cv2.VideoCapture("test/traffic.mp4")
ret, frame = cap.read()
if ret:
    cv2.imwrite("test/sample_frame.jpg", frame)
    print("Saved test/sample_frame.jpg — open it to pick stop-line coordinates")
cap.release()