from ultralytics import YOLO
    
class PlayerDetector:
    def __init__(self, model_path="yolov8s.pt", conf=0.4, iou=0.3, imgsz=1280):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def detect(self, video_path):
        return self.model(
            video_path,
            stream=True,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz
        )