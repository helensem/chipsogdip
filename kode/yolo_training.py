from ultralytics import YOLO 
import cv2

#yolo_model = YOLO("yolov8n-seg.pt")
#yolo_model.train(data = "/cluster/home/helensem/Master/sky_data/data.yaml", epochs = 40, patience = 35)


model_pred = YOLO("/cluster/home/helensem/Master/runs/segment/train9/weights/best.pt")

im2 = cv2.imread("/cluster/home/helensem/Master/Labeled_pictures/val/img6/img6.jpg")
results = model_pred.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
