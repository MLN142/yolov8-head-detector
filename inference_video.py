# Import required libraries
from ultralytics import YOLO  
import cv2  

# Load trained YOLOv8 model (make sure best.pt is in the same directory)
model = YOLO("best.pt") 

# Set path to input video
video_path = "your_video"  
cap = cv2.VideoCapture(video_path)  

# Get video properties: width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize video writer to save output with detections
out = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()  
    if not ret:
        break  

    # Resize the frame to 1280x720 resolution for consistent processing
    frame = cv2.resize(frame, (1280, 720))

    # Run object detection on the current frame using the YOLO model
    results = model(frame, verbose=False)

    # Get the detected bounding boxes (format: [x1, y1, x2, y2])
    boxes = results[0].boxes.xyxy

    # Loop through each detected box and draw it on the frame
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)  

    # Display the processed frame in a window
    cv2.imshow("YOLOv8 Head Detection", frame)

    # Write the processed frame to the output video file
    out.write(frame)

    # Break the loop if 'q' is pressed on the keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()
