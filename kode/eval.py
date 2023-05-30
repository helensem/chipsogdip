import numpy as np 
import cv2
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
import sys, os, json 
from matplotlib import pyplot as plt 
import pandas as pd

sys.path.append("cluster/home/helensem/Master/chipsogdip/kode")
from dataset import load_mask, combine_masks_to_one
from skyseg import remove_sky


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
    image_truth_id = data["image_id"] + "_truth.png"
    output = os.path.join(output_path, image_id)
    output_truth = os.path.join(output_path, image_truth_id)

    out_truth = v2.draw_dataset_dict(data)
    #cv2.imshow("imageout", out_truth.get_image()[:,:,::-1])

    #vis = np.concatenate((out.get_image()[:,:,::-1], out_truth.get_image()[:,:,::-1]))
    cv2.imwrite(output, out.get_image()[:,:,::-1])
    cv2.imwrite(output_truth, out_truth.get_image()[:,:,::-1])


   
def evaluate_model(cfg, val_dict, write_to_file = False, segment_sky=False):
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
            file_name = "output_train.txt"
        with open(os.path.join(cfg.OUTPUT_DIR, file_name), "w") as f: 
            f.write(iou_string)
    return mean_corr_iou, mean_bg_iou, (mean_corr_iou + mean_bg_iou) / 2


def evaluate_thresholds(cfg, val_dict):
    thresholds = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9] 
    mean_ious = []
    corr_ious = []
    bg_ious = []
    for threshold in thresholds: 
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

        mean_corr, mean_bg, mean_iou = evaluate_model(cfg, val_dict)
        corr_ious.append(mean_corr)
        mean_ious.append(mean_iou)
        bg_ious.append(mean_bg)
    #, bg_ious, corr_ious, mean_ious = (list(t) for t in zip(*sorted(zip(model_names, bg_ious, corr_ious, mean_ious))))
    print(corr_ious)
    plt.clf()
    plt.plot(thresholds, mean_ious, color = 'r'
            ,label = "Mean", marker="o")
    plt.plot(thresholds, corr_ious, label="Corrosion", color = "c", marker="o")
    plt.plot(thresholds, bg_ious, label="Background", color="m", marker="o")
    #plt.plot(data["epoch"], data["val/cls_loss"] , color = 'b',label = "val")
    
    plt.xticks(rotation = 25)
    plt.xlabel('Confidence threshold')
    plt.ylabel('IoU')
    #plt.title('Total loss', fontsize = 20)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "iou_per_threshold.svg"), format="svg")




def evaluate_model_better(cfg, val_dict, write_to_file=False, file_format='json', segment_sky=False):
    predictor = DefaultPredictor(cfg)
    iou_corr_list = []
    iou_bg_list = []
    iou_results = []

    for d in val_dict:
        image = cv2.imread(d["file_name"])
        image_dir = os.path.dirname(d["file_name"])

        if segment_sky:
            image = remove_sky(image)

        mask_gt = load_mask(os.path.join(image_dir, "masks"))
        mask_gt = combine_masks_to_one(mask_gt)

        outputs = predictor(image)
        predicted_masks = outputs['instances'].to("cpu").pred_masks.numpy()
        predicted_masks = np.transpose(predicted_masks, (1, 2, 0))
        
        if predicted_masks.shape[-1] == 0:
            continue
        
        mask_pred = combine_masks_to_one(predicted_masks)
        
        iou_corr = compute_overlaps_masks(mask_gt, mask_pred)[0][0]
        iou_bg = compute_overlaps_masks(mask_gt, mask_pred, BG=True)[0][0]
        
        print(d["image_id"], "IoU =", (iou_corr, iou_bg))
        
        iou_corr_list.append(iou_corr)
        iou_bg_list.append(iou_bg)
        iou_results.append((d["image_id"], iou_corr, iou_bg))

    if len(iou_corr_list) == 0:
        return 0, 0, 0
    
    mean_corr_iou = np.mean(iou_corr_list)
    mean_bg_iou = np.mean(iou_bg_list)
    
    print("Total mean values")
    print("Corrosion IoU =", mean_corr_iou)
    print("BG IoU =", mean_bg_iou)
    print("Mean IoU =", (mean_corr_iou + mean_bg_iou) / 2)

    if write_to_file:
        if segment_sky:
            file_name = "output_seg"
        else:
            file_name = "output"

        if file_format == 'json':
            result_dict = {
                "mean_corr_iou": mean_corr_iou,
                "mean_bg_iou": mean_bg_iou,
                "mean_iou": (mean_corr_iou + mean_bg_iou) / 2,
                "results": iou_results
            }
            with open(os.path.join(cfg.OUTPUT_DIR, file_name + ".json"), "w") as f:
                json.dump(result_dict, f)
        elif file_format == 'csv':
            df = pd.DataFrame(iou_results, columns=["image_id", "iou_corr", "iou_bg"])
            df.to_csv(os.path.join(cfg.OUTPUT_DIR, file_name + ".csv"), index=False)

    return mean_corr_iou, mean_bg_iou, (mean_corr_iou + mean_bg_iou) / 2

def evaluate_over_iterations(cfg, val_dict, output_dir, plot=False, segment_sky=False):
    models = next(os.walk(output_dir))[2]
    mean_ious = []
    corr_ious = []
    bg_ious = []
    model_names = []
    write_to_file = False
    path_to_metrics = os.path.join(output_dir, "metrics.json")
    iterations = get_iteration(path_to_metrics)
    for model in models:
        if model.endswith(".pth"):
            cfg.MODEL.WEIGHTS = os.path.join(output_dir, model)
            model_name = os.path.splitext(model)[0]
            model_number = model_name[6:]
            if model_number == "final": 
                model_number = iterations
                write_to_file = True

            else: 
                model_number = int(model_number)+1
            mean_corr, mean_bg, mean_iou = evaluate_model(cfg, val_dict, write_to_file, segment_sky)
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




def plot_metrics(path_to_metrics, output, metric): 
  with open(path_to_metrics, 'r') as handle:
      json_data = [json.loads(line) for line in handle]
  x = [i['iteration'] for i in json_data]
  plt.clf()
  iterations = json_data[-1]['iteration']
  epochs = int(2*iterations/1500)
  x_mean = np.arange(1,epochs+1)
  print(x_mean)
  steps = 750*x_mean
  steps=steps[0:epochs]
  if metric == 'fp_fn': 
    total_fp_per_epoch = []
    total_fn_per_epoch = []
    y = [[i['mask_rcnn/false_negative'] for i in json_data],[i['mask_rcnn/false_positive'] for i in json_data]]
    fn_parts = np.array_split(y[0], epochs)
    fp_parts = np.array_split(y[1], epochs)

    # Calculate the mean of each part and store in a new array
    total_fp_per_epoch = np.array([part.mean() for part in fp_parts])
    total_fn_per_epoch = np.array([part.mean() for part in fn_parts])
    plt.plot(steps,total_fn_per_epoch, label = "False negative", marker="o", color="g")
    plt.plot(steps,total_fp_per_epoch, label = "False positive", marker="o", color="b")
    plt.xlabel('Step')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output, metric + ".svg"), format = "svg")
    return
  y =  [i[metric] for i in json_data]
  parts = np.array_split(y, epochs)
  y_mean = np.array([part.mean() for part in parts])
  plt.plot(x,y, color="g")
  plt.plot(steps, y_mean, color="b", marker="o")
  plt.grid()
  plt.xlabel('Step')
  plt.ylabel(metric)
  plt.savefig(os.path.join(output, metric + ".svg"), format = "svg")


def ious_from_file(path): 
    with open(path, "r") as f: 
        data = f.read()

    data = data.split("\n")
    del data [-4:]
    results = {}
    for d in data: 
        print(d)
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
    for key in dict1.keys(): 
        if key not in dict2:
            continue
        if dict1[key][0] > dict2[key][0]: 
            iou_1.append(key)
        else: 
            iou_2.append(key)
    return iou_1, iou_2


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

def get_iteration(path_to_metrics):
    with open(path_to_metrics, 'r') as handle:
      json_data = [json.loads(line) for line in handle]
    return json_data[-1]['iteration']  


if __name__ == "__main__":
    path_1 = r"/cluster/work/helensem/Master/output/reduced_data/resnet101/output.txt"
    path_2 = r"/cluster/work/helensem/Master/output/reduced_data/resnet101-DC5/output.txt"

    results_normal = ious_from_file(path_1)
    results_reduced = ious_from_file(path_2)

    iou_normal, iou_reduced =compare_ious(results_normal, results_reduced)
    print(len(iou_normal), len(iou_reduced))
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
