import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("C:/Users/Hp/OneDrive/Masaüstü/best (1)/best (1).pt")

# Load an image instead of a video
image = cv2.imread("photo.jpg")  # Replace with your image path

# Run detection
results = model(image)

# Draw results
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show image
cv2.imshow("Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
