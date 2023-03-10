import numpy as np 
import cv2
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
import os 
import sys 

sys.path.append("cluster/home/helensem/Master/chipsogdip/kode")
from dataset import load_mask


local_class_colors = [(0, 0, 0), (0, 0, 255)]
mask_rcnn_colors = local_class_colors
def apply_inference(predictor, metadata, output_path, data, image_path=None): #*Saves all val images and compares to original image 
    # Load image
    image = cv2.imread(image_path)
    # Run detection
    results = predictor(image)
    # Visualize results
    v = Visualizer(image[:, :, ::-1],
                    metadata,
                    scale = 0.5,
                    instance_mode = ColorMode.IMAGE)
    out = v.draw_instance_predictions(results["instances"].to("cpu"))

    # Dont know how to reset the image without predictions, so create new visualizer 
    v2 = Visualizer(image[:, :, ::-1],
                    metadata,
                    scale = 0.5,
                    instance_mode = ColorMode.IMAGE)

    image_id = next(os.walk(os.path.dirname(image_path)))[2][0]
    output = os.path.join(output_path, image_id)


    out_truth = v2.draw_dataset_dict(data)
    #cv2.imshow("imageout", out_truth.get_image()[:,:,::-1])

    vis = np.concatenate((out.get_image()[:,:,::-1], out_truth.get_image()[:,:,::-1]))
    cv2.imwrite(output, vis)

   
def evaluate_model(cfg, val_dict, write_to_file = False):
    predictor = DefaultPredictor(cfg)
    #image_ids = dataset_val.image_ids
    iou_corr_list = []
    iou_bg_list = []
    iou_string = ""
    for d in val_dict: 
        image = cv2.imread(d["file_name"])
        image_dir = os.path.dirname(d["file_name"])

        mask_gt = load_mask(os.path.join(image_dir, "masks"))
        mask_gt = combine_masks_to_one(mask_gt)

        outputs = predictor(image)

        predicted_masks = outputs['instances'].to("cpu").pred_masks.numpy()
        predicted_masks = np.transpose(predicted_masks, (1,2,0)) #* (N x H x W) to (H x W x N)
        if predicted_masks.shape[-1] == 0:
            continue
        mask_pred = combine_masks_to_one(predicted_masks)

        iou_corr = compute_overlaps_masks(mask_gt, mask_pred)[0][0]
        iou_bg = compute_overlaps_masks(mask_gt, mask_pred, BG=True)[0][0]
        print(d["image_id"], "IoU =", (iou_corr, iou_bg))
        string = d["image_id"] + " IoU = " + str((iou_corr, iou_bg)) +"\n"
        iou_string+= string
        iou_corr_list.append(iou_corr)
        iou_bg_list.append(iou_bg)
    mean_corr_iou = sum(iou_corr_list) / len(iou_corr_list)
    mean_bg_iou = sum(iou_bg_list) / len(iou_bg_list)
    print("Total mean values ")
    print(" Corrosion IoU =", mean_corr_iou)
    print("BG IoU=", mean_bg_iou)
    print("Mean IoU =", (mean_corr_iou + mean_bg_iou) / 2)
    if write_to_file:
        iou_string += "Total mean values: \n" + " Corrosion IoU: " + str(mean_corr_iou) + "\n" + "BG IoU=" + str(mean_bg_iou) + "\n" + "Mean IoU =" + str((mean_corr_iou + mean_bg_iou) / 2)
        with open(os.path.join(cfg.OUTPUT_DIR,"output.txt"), "w") as f: 
            f.write(iou_string)
    return mean_corr_iou, mean_bg_iou, (mean_corr_iou + mean_bg_iou) / 2


def combine_masks_to_one(masks):
    combined_mask = masks[:, :, 0]
    for i in range(masks.shape[-1]):
        combined_mask += masks[:, :, i]
    return np.expand_dims(combined_mask, 2)



def iou_numpy(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6

    outputs = outputs.squeeze(axis=-1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou


def compute_overlaps_masks(masks1, masks2, BG=False):
    """Computes IoU overlaps between two sets of masks .
    masks1, masks2: [Height , Width , instances ]
    """
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # f l a t t e n masks and compute their areas
    if BG:
        masks1 = np.reshape(masks1 < .5,
                            (-1, masks1.shape[-1])).astype(np.float32)
        masks2 = np.reshape(masks2 < .5,
                            (-1, masks2.shape[-1])).astype(np.float32)
    else:
        masks1 = np.reshape(masks1 > .5,
                            (-1, masks1.shape[-1])).astype(np.float32)
        masks2 = np.reshape(masks2 > .5,
                            (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

# def overlay_prediction_single_maskrcnn(pr=None, inp=None,
#                                        out_dir=None, overlay_img=True, colors=mask_rcnn_colors):
#     if isinstance(pr, six.string_types):
#         pr = cv2.imread(pr, 0)
#     if isinstance(inp, six.string_types):
#         out_fname = os.path.join(out_dir, os.path.basename(inp))
#         inp = cv2.imread(inp[:-4] + ".jpg")

#     assert len(inp.shape) == 3, "Image should be h,w,3 "
#     seg_img = visualize_segmentation(pr, inp, colors=colors,
#                                      overlay_img=overlay_img, )
#     if out_fname is not None:
#         cv2.imwrite(out_fname, seg_img)
#     return pr

# def overlay_predictions_all_maskrcnn(pr_dir=None, inp_dir=None,
#                                      out_dir=None,
#                                      overlay_img=True, colors=mask_rcnn_colors):
#     for f in next(os.walk(pr_dir))[2]:
#         print("File name =", f)
#         if " DS_Store " in f:
#             print(" Skipping DS_Store file")
#         continue
#         if f.endswith(".png"):
#             pr = os.path.join(pr_dir, f)
#             inp = os.path.join(inp_dir, f)
#             overlay_prediction_single_maskrcnn(pr, inp,
#                                                out_dir, overlay_img, colors)
