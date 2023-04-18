from ultralytics import YOLO 
from ultralytics.yolo.utils.ops import scale_image
import cv2
import numpy as np
import os



def remove_sky(image): 
    model = YOLO("/cluster/home/helensem/Master/runs/segment/train10/weights/best.pt")
    results = model.predict(source=image, save=False, save_txt=False, conf=0.6)  # save predictions as labels

        #print(results)
    for result in results: 
        if result.masks is None: 
            print("no detections in results")
            continue
        masks = result.masks.masks.cpu().numpy()     # masks, (N, H, W)
        masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
        # rescale masks to original image
        masks = scale_image(masks.shape[:2], masks, result.masks.orig_shape)
        masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)
        #cv2.imwrite(r"/cluster/home/helensem/Master/output/sky/test.jpg", (result.masks.masks[0].cpu().numpy()*255).astype("uint8"))
        #print(masks)

        for mask in masks:
            mask = (mask*255).astype("uint8")
            print(mask.shape)
            print(image.shape)
            #print(mask)
            #binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]
            #print(binary_mask)
            # Convert the binary mask to the same datatype as the image
            #mask = mask.astype(np.uint8)
            mask = cv2.bitwise_not(mask)
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


            #destination = os.path.join(r"/cluster/home/helensem/Master/output/sky", id)
            #cv2.imwrite(destination, image)
    return image    



if __name__ == "__main__": 
    mode = "predict"

    if mode == "train":

        yolo_model = YOLO("yolov8n-seg.pt")
        yolo_model.train(data = "/cluster/home/helensem/Master/sky_data/data.yaml")

    elif mode == "predict":
        model_pred = YOLO("/cluster/home/helensem/Master/runs/segment/train10/weights/best.pt")


    elif mode == "predict_seg":
        model_pred = YOLO("/cluster/home/helensem/Master/runs/segment/train10/weights/best.pt")
        #model_pred.overrides['conf'] = 0.6  # NMS confidence threshold

        paths = []
        path = r"/cluster/home/helensem/Master/damage_data"
        image_ids = next(os.walk(path))[2]

        for id in image_ids: 

            image_path = os.path.join(path, id)
            paths.append(image_path)

        #image = cv2.imread(paths)
        #height, widht = image.shape[:2]
        results = model_pred.predict(source=paths, save=True, save_txt=True)  # save predictions as labels

        #print(results)
        for result in results: 
            #print(result)
            if result.masks is None: 
                print("no detections for: ", id)
                continue
            masks = result.masks.masks.cpu().numpy()     # masks, (N, H, W)
            masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
            # rescale masks to original image
            masks = scale_image(masks.shape[:2], masks, result.masks.orig_shape)
            masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)
            #cv2.imwrite(r"/cluster/home/helensem/Master/output/sky/test.jpg", (result.masks.masks[0].cpu().numpy()*255).astype("uint8"))
            #print(masks)
            image = result.orig_img
            for idx, mask in enumerate(masks):
                if result.boxes.conf[idx] > 0.6: #Only take predictions with high confidence scores 
                    mask = (mask*255).astype("uint8")
                    print(mask.shape)
                    print(image.shape)
                    #print(mask)
                    #binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]
                    #print(binary_mask)
                    # Convert the binary mask to the same datatype as the image
                    #mask = mask.astype(np.uint8)
                    mask = cv2.bitwise_not(mask)
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
            destination = os.path.join(r"/cluster/home/helensem/Master/output/sky", id)
            cv2.imwrite(destination, image)
