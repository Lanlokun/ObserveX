from ultralytics import YOLO

def load_yolo_model():
    model = YOLO("yolov5s.pt")  # or yolov8n.pt if you're using YOLOv8
    return model
