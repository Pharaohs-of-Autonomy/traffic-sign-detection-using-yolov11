# import libraries
from ultralytics import YOLO
import cv2
import argparse

# -------------------------
# Argument parser
# -------------------------
parser = argparse.ArgumentParser(description="Traffic sign detection")
parser.add_argument(
    "--source",
    choices=["webcam", "video"],
    required=True,
    help="Choose input source: webcam or video"
)
parser.add_argument(
    "--video_path",
    type=str,
    default=None,
    help="Path to video file (required if source=video)"
)

args = parser.parse_args()

# -------------------------
# Initialize model
# -------------------------
detector = YOLO("./model/traffic_sign_detector.pt", task="detect")

# -------------------------
# Select input source
# -------------------------
if args.source == "webcam":
    cap = cv2.VideoCapture(0)
elif args.source == "video":
    if args.video_path is None:
        print("Error: --video_path must be provided when source=video")
        exit()
    cap = cv2.VideoCapture(args.video_path)

# -------------------------
# Check for errors
# -------------------------
if not cap.isOpened():
    print("Unable to open the input source")
    exit()

# -------------------------
# Processing loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector(frame)

    for detection in detections:
        for bbox in detection.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Get class name using the class ID
            class_id = int(bbox.cls)
            class_name = detection.names[class_id]  # detection.names holds the class names
            
            # Display the class name on the image
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        

    cv2.imshow("Traffic sign detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
