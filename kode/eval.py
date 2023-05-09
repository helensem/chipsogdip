import numpy as np 
import cv2
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
import os 
import sys 
from matplotlib import pyplot as plt 

sys.path.append("cluster/home/helensem/Master/chipsogdip/kode")
from dataset import load_mask
from yolo_training import remove_sky


local_class_colors = [(0, 0, 0), (0, 0, 255)]
mask_rcnn_colors = local_class_colors
def apply_inference(predictor, metadata, output_path, data, segment_sky = False): #*Saves all val images and compares to original image 
    # Load image
    image = cv2.imread(data["file_name"])
    if segment_sky: 
        image = remove_sky(image)
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

    image_id = data["image_id"] + ".png"
    output = os.path.join(output_path, image_id)


    out_truth = v2.draw_dataset_dict(data)
    #cv2.imshow("imageout", out_truth.get_image()[:,:,::-1])

    vis = np.concatenate((out.get_image()[:,:,::-1], out_truth.get_image()[:,:,::-1]))
    cv2.imwrite(output, vis)

   
def evaluate_model(cfg, val_dict, write_to_file = False, plot=False, segment_sky=False):
    predictor = DefaultPredictor(cfg)
    #image_ids = dataset_val.image_ids
    iou_corr_list = []
    iou_bg_list = []
    iou_string = ""
    for d in val_dict: 
        image = cv2.imread(d["file_name"])
        image_dir = os.path.dirname(d["file_name"])
        #print(image_dir)
        if segment_sky:
            image = remove_sky(image)
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
    if len(iou_corr_list) == 0: 
        return 0,0,0 
    mean_corr_iou = np.mean(iou_corr_list)
    mean_bg_iou = np.mean(iou_bg_list)
    print("Total mean values ")
    print("Corrosion IoU =", mean_corr_iou)
    print("BG IoU=", mean_bg_iou)
    print("Mean IoU =", (mean_corr_iou + mean_bg_iou) / 2)
    if write_to_file:
        iou_string += "Total mean values: \n" + "Corrosion IoU: " + str(mean_corr_iou) + "\n" + "BG IoU=" + str(mean_bg_iou) + "\n" + "Mean IoU =" + str((mean_corr_iou + mean_bg_iou) / 2)
        if segment_sky: 
            file_name = "output_seg.txt" 
        else: 
            file_name = "output.txt"
        with open(os.path.join(cfg.OUTPUT_DIR, file_name), "w") as f: 
            f.write(iou_string)
    if plot: 
        x = np.arange(1, len(iou_corr_list)+1)
        corr_iou = np.sort(iou_corr_list)
        plt.bar(x, corr_iou)
        plt.axhline(y=np.mean(iou_corr_list), color="g")
        plt.savefig(os.path.join(cfg.OUTPUT_DIR, "corrosion_iou.svg"), format="svg") 
    return mean_corr_iou, mean_bg_iou, (mean_corr_iou + mean_bg_iou) / 2


def evaluate_over_iterations(cfg, val_dict, output_dir, plot=False, segment_sky=False):
    models = next(os.walk(output_dir))[2]
    mean_ious = []
    corr_ious = []
    bg_ious = []
    model_names = []
    for model in models:
        if model.endswith(".pth"):
            cfg.MODEL.WEIGHTS = os.path.join(output_dir, model)
            model_name = os.path.splitext(model)[0]
            model_number = model_name[6:]
            if model_number == "final": 
                model_number = 39600
            else: 
                model_number = int(model_number)+1
            mean_corr, mean_bg, mean_iou = evaluate_model(cfg, val_dict, False, segment_sky)
            corr_ious.append(mean_corr)
            mean_ious.append(mean_iou)
            bg_ious.append(mean_bg)
            model_names.append(model_number)
    
    model_names, bg_ious, corr_ious, mean_ious = (list(t) for t in zip(*sorted(zip(model_names, bg_ious, corr_ious, mean_ious))))

    if plot: 
        plt.plot(model_names, mean_ious, color = 'r'
                ,label = "Mean", marker="o")
        plt.plot(model_names, corr_ious, label="Corrosion", color = "c", marker="o")
        plt.plot(model_names, bg_ious, label="Background", color="m", marker="o")
        #plt.plot(data["epoch"], data["val/cls_loss"] , color = 'b',label = "val")
        
        plt.xticks(rotation = 25)
        plt.xlabel('Step')
        plt.ylabel('IoU')
        #plt.title('Total loss', fontsize = 20)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_dir, "iou_per_model.svg"), format="svg")


def ious_from_file(path): 
    with open(path, "r") as f: 
        data = f.read()

    data = data.split("\n")
    del data [-4:]
    results = {}
    for d in data: 
        d = d.replace(" IoU = ", ":")
        d = d.split(":")
        tup = d[1].replace("(", "")
        tup = tup.replace(")", "")
        res = tuple(map(float, tup.split(', ')))
        results[d[0]] = res
    return results
 
def compare_ious(dict1, dict2): 
    iou_1 = []
    iou_2 = []
    for key in dict2.keys(): 

        if dict1[key][0] > dict2[key][0]: 
            iou_1.append(key)
        else: 
            iou_2.append(key)
    return iou_1, iou_2


def combine_masks_to_one(masks):
    combined_mask = masks[:, :, 0]
    for i in range(masks.shape[-1]):
        combined_mask += masks[:, :, i]
    return np.expand_dims(combined_mask, 2)



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


if __name__ == "__main__":
    path_1 = r"/cluster/work/helensem/Master/output/run_aug/resnet101/output.txt"
    path_2 = r"/cluster/work/helensem/Master/output/reduced_data/resnet101/output.txt"

    results_normal = ious_from_file(path_1)
    results_reduced = ious_from_file(path_2)

    iou_normal, iou_reduced =compare_ious(results_normal, results_reduced)
    print("images where the normal is better: ", iou_normal, "\n", "Images where reduced data is better: ", iou_reduced)


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
