from ultralytics import YOLO 
from ultralytics.yolo.utils.ops import scale_image
import cv2
import numpy as np

#yolo_model = YOLO("yolov8n-seg.pt")
#yolo_model.train(data = "/cluster/home/helensem/Master/sky_data/data.yaml", epochs = 40, patience = 35)


model_pred = YOLO("/cluster/home/helensem/Master/runs/segment/train9/weights/best.pt")

image = cv2.imread("/cluster/home/helensem/Master/Labeled_pictures/val/img6/img6.jpg")
#height, widht = image.shape[:2]
results = model_pred.predict(source=image, save=True, save_txt=True)  # save predictions as labels

#print(results)
for result in results: 
    masks = result.masks.masks.cpu().numpy()     # masks, (N, H, W)
    masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
    # rescale masks to original image
    masks = scale_image(masks.shape[:2], masks, result.masks.orig_shape)
    masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)


    for mask in masks:
        binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]
        # Convert the binary mask to the same datatype as the image
        binary_mask = binary_mask.astype(np.uint8)
        mask = cv2.bitwise_not(binary_mask)
        image = cv2.bitwise_and(image, image, mask=mask)

# load the original input image and display it to our screen
# a mask is the same size as our image, but has only two pixel
# values, 0 and 255 -- pixels with a value of 0 (background) are
# ignored in the original image while mask pixels with a value of
# 255 (foreground) are allowed to be kept
#mask = np.zeros(image.shape[:2], dtype="uint8")
#cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
#cv2.imshow("Rectangular Mask", masks)
# apply our mask -- notice how only the person in the image is
# cropped out

    cv2.imwrite(r"/cluster/home/helensem/output/sky/test.jpg", image)
