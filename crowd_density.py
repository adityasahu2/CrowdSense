import cv2
import torch
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

video_paths = [
    r"C:\Users\RAMESWAR BISOYI\Downloads\istockphoto-1474154311-640_adpp_is.mp4",
    r"C:\Users\RAMESWAR BISOYI\Downloads\crowd sample 1.mp4"
]

caps = [cv2.VideoCapture(video) for video in video_paths]

CONGESTION_THRESHOLD = 50  

while True:
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            continue

        height, width, _ = frame.shape
        total_area = width * height

        # Run YOLOv8 Inference
        results = model(frame)

        # Extract People Bounding Boxes
        person_boxes = []
        occupied_area = 0

        for r in results:
            for box in r.boxes:
                if int(box.cls) == 0:  # Class 0 = Person
                    person_boxes.append(box.xyxy[0].cpu().numpy())
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    occupied_area += (x2 - x1) * (y2 - y1)  # Area occupied by people

        # Compute Crowd Density Percentage
        density = (occupied_area / total_area) * 100

        # Generate Heatmap
        heatmap = np.zeros((height, width), dtype=np.float32)
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            heatmap[y1:y2, x1:x2] += 1  

        heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
        heatmap = (heatmap / heatmap.max()) * 255
        heatmap = np.uint8(heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Blend Heatmap with Frame
        overlay = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.5, 0)

        # Display Density Percentage
        text = f"Crowd Density: {density:.2f}%"
        color = (0, 255, 0) if density < CONGESTION_THRESHOLD else (0, 0, 255)
        cv2.putText(overlay, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Congestion Alert
        if density >= CONGESTION_THRESHOLD:
            cv2.putText(overlay, " HIGH CONGESTION ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow(f"Camera {i+1}", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
