import cv2
from ultralytics import YOLO

# Initialize YOLOv5 model
model = YOLO('yolov5n.pt')  # Ensure the model weights path is correct

# Define the video capture object with higher resolution
cap = cv2.VideoCapture(0)  # 0 is the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

# Define the class ID for "book" (73 in the COCO dataset for YOLOv5)
BOOK_CLASS_ID = 73

# Detection parameters
CONF_THRESHOLD = 0.3  # Minimum confidence to consider a detection
IOU_THRESHOLD = 0.4  # Intersection-over-Union (IoU) threshold for NMS

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Perform inference
    results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)

    # Debug: print the number of detections
    print(f'Detections: {len(results[0].boxes)}')

    # Draw detections on the frame
    for box in results[0].boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf.cpu().numpy())  # Convert confidence to float
        cls = int(box.cls.cpu().numpy())      # Convert class ID to int

        # Debug: print each detection
        print(f'Class: {cls}, Confidence: {conf:.2f}, Box: {x1, y1, x2, y2}')

        if cls == BOOK_CLASS_ID:
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Book {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame with detections
    cv2.imshow('Book Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
    
