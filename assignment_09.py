# Install: pip install ultralytics opencv-python
# Sample CCTV / traffic videos (download any .mp4 and name it cctv.mp4):
#   UA-DETRAC : https://detrac-db.rit.albany.edu/
#   MOT17     : https://motchallenge.net/data/MOT17/
#   Free clips: https://www.pexels.com/search/videos/cctv/
# The pretrained model yolov8n.pt downloads automatically on first run.
import cv2
from ultralytics import YOLO

# 1. Load the pretrained YOLOv8 model (trained on COCO dataset)
model = YOLO("yolov8n.pt")
targets = ["person", "car", "bus", "truck", "motorcycle", "bicycle"]

# 2. Open the CCTV video and prepare the output writer
cap = cv2.VideoCapture("cctv.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
out = cv2.VideoWriter("cctv_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (width, height))

frame_no = 0
total_counts = {}

# 3. Process the video frame by frame
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1

    results = model(frame, conf=0.4, verbose=False)[0]
    frame_counts = {}

    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        if label not in targets:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame_counts[label] = frame_counts.get(label, 0) + 1

    # keep the maximum count seen per class and show counts on the frame
    for k, v in frame_counts.items():
        total_counts[k] = max(total_counts.get(k, 0), v)
    text = ", ".join(f"{k}: {v}" for k, v in frame_counts.items())
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("CCTV Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 4. Release resources and print summary
cap.release()
out.release()
cv2.destroyAllWindows()
print("Frames processed:", frame_no)
print("Maximum objects seen in one frame:", total_counts)

# 5. Detection on a single image (optional)
res = model("cctv_frame.jpg", conf=0.4)
res[0].save("cctv_frame_detected.jpg")
