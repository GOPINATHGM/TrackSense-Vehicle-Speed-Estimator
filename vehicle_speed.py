import cv2
import os
import numpy as np
from collections import deque

# Define paths
weights_path = r"D:\python\project_major\yolov3.weight"
config_path = r"D:\python\project_major\yolov3.cfg"
names_path = r"D:\python\project_major\coco.names"

# Verify that the files exist
if not os.path.isfile(weights_path):
    raise FileNotFoundError(f"YOLO weights file not found: {weights_path}")
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"YOLO config file not found: {config_path}")
if not os.path.isfile(names_path):
    raise FileNotFoundError(f"COCO names file not found: {names_path}")

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()
if isinstance(unconnected_out_layers, list) or isinstance(unconnected_out_layers, np.ndarray):
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
else:
    output_layers = [layer_names[unconnected_out_layers - 1]]

# Load COCO labels
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture
cap = cv2.VideoCapture("video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

# Tracking vehicles
vehicle_tracks = {}
track_length = 20
min_track_length = 5
pixel_to_meter = 0.05  # Approximate conversion, should be calibrated based on the actual scene

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process YOLO outputs
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            # Track vehicle
            center = (x + w // 2, y + h // 2)
            if label not in vehicle_tracks:
                vehicle_tracks[label] = deque(maxlen=track_length)
            vehicle_tracks[label].append(center)

            # Calculate speed if enough tracking points are available
            if len(vehicle_tracks[label]) >= min_track_length:
                dx = vehicle_tracks[label][-1][0] - vehicle_tracks[label][0][0]
                dy = vehicle_tracks[label][-1][1] - vehicle_tracks[label][0][1]
                distance = np.sqrt(dx ** 2 + dy ** 2) * pixel_to_meter  # Convert pixels to meters
                time_elapsed = len(vehicle_tracks[label]) / fps
                speed = distance / time_elapsed * 3.6  # Convert m/s to km/h

                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f'{label} {speed:.2f} km/h', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
