# import libraries
from ultralytics import YOLO
import cv2

# initialize the detector model
detector = YOLO("./model/traffic_sign_detector.pt", task="detect")

# open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# check for errors
if not cap.isOpened():
    print("Unable to open the webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector(frame)

    for detection in detections:
        for bbox in detection.boxes:
            x1, y1, x2, y2 = bbox.xyxy[0]

            # convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
  
        # Display the class name on the image
            class_name = detection.names[int(bbox.cls)]
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
            # draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Traffic sign detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
