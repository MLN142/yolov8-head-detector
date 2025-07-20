from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model (make sure best.pt is in the same directory)
model = YOLO("best.pt")

# Load image (change filename if needed)
img_path = "your_image"
img = cv2.imread(img_path)

# Run detection
results = model(img_path)

# Draw bounding boxes
boxes = results[0].boxes.xyxy
for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

# Save the result image
cv2.imwrite("clean_prediction.jpg", img)
