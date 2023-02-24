from ultralytics import YOLO 
import cv2 
import numpy as np 
from detectron2.engine import DefaultPredictor 
from detectron2.evaluation import COCOEvaluator, inference_on_dataset 
from detectron2.data import build_detection_test_loader 

yolo_model = YOLO("yolov8n.pt")
yolo_model.train()




# FROM CHAT GPT 
image = cv2.imread("input_image.jpg")
boxes = yolo_model.detect_bridges(image)




rois = [] 
for box in boxes: 
    x1, y1, x2, y2 = box 
    roi = image[y1:y2, x1:x2]
    rois.append(roi)

mask_rcnn_model = DefaultPredictor()
 