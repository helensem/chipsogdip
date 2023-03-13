from ultralytics import YOLO 


yolo_model = YOLO("yolov8n-seg.pt")
yolo_model.train(data = "/cluster/home/helensem/Master/sky_data/coco8-seg/data.yaml", epochs = 30, patience = 25)

