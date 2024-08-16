# ultralytics version is 8.26.0, not latest version.
from ultralytics import YOLO
import cv2
from PIL import Image

model = YOLO("yolov8x.pt")
model.info()
im1 = Image.open("images/test.jpg")
results = model.predict(source=im1, save=True, save_txt=True)

cv2.imshow("YOLOv5 Detection", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
# 如果需要获取预测的详细信息，可以查看results列表
# 例如，打印每个检测到的物体及其类别和置信度
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = int(box.cls[0])
        print(f"Class: {cls}, Confidence: {conf:.2f}, Box: ({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f})")
