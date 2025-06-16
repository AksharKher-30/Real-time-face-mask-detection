# realtime_mask_detection.py
import cv2
from ultralytics import YOLO

# Load model
model = YOLO('/Users/akshar/Documents/DL/face mask detection/runs/s_runs/s_train/face_mask_detector_s/weights/best.pt') # update with your model path
model.conf = 0.5  # confidence threshold

class_names = ['Mask', 'No Mask']

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Contrast Enhancement with CLAHE ===
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    # === End of Enhancement ===

    # Run detection
    results = model.predict(source=frame, save=False, imgsz=640, conf=0.5, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{class_names[cls_id]}: {conf*100:.1f}%"

            # Draw bounding box and label
            color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Mask Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Properly release everything
cap.release()
cv2.destroyAllWindows()