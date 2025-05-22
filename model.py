import cv2
from ultralytics import YOLO
from collections import defaultdict

model = YOLO("C:/Users/Hp/OneDrive/Masaüstü/project/best (1).pt")  

def detect_objects_on_image(image_path, output_path):
    image = cv2.imread(image_path)
    results = model.predict(image, conf=0.3)
    
    class_counts = defaultdict(int)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            label = model.names[cls_id]
            conf = float(box.conf)
            class_counts[label] += 1
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    ext = output_path.split('.')[-1].lower()
    if ext in ['jpg', 'jpeg']:
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(output_path, image)

    return class_counts

def detect_objects_on_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    skip_frame = 3
    last_detections = []
    class_counts = defaultdict(int)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_frame == 0:
            results = model.predict(frame, conf=0.3)
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls)
                    label = model.names[cls_id]
                    conf = float(box.conf)
                    detections.append((x1, y1, x2, y2, label, conf))
                    class_counts[label] += 1
            last_detections = detections

        for (x1, y1, x2, y2, label, conf) in last_detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return class_counts