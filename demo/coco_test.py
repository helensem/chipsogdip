# For testing detectron2 mask r-cnn with COCO 

import detectron2 

#Setup logger
from detectron2.utils.logger import setup_logger
setup_logger("/Volumes/helensem/Master/chipsogdip/demo/out.log")

#Some common libraries 
import numpy as np 
import os,json,cv2,random 


#Detectron2 utilities 

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg 
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset 
from detectron2.data import build_detection_test_loader 

from detectron2.structures import BoxMode 

def get_balloon_dicts(img_dir): 
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f: 
        imgs_anns = json.load(f)
    
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx 
        record["height"] = height 
        record["width"] = width 

        annos = v["regions"]
        objs = []

        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px,py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS, 
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record) 
    return dataset_dicts

def config(cfg_file):
    cfg = get_cfg() 
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(cfg_file)
    #cfg.DATASETS.TRAIN = ("balloon_train")
    #cfg.DATASETS.TEST = ()
    #cfg.MODEL.DEVICE = "cpu"
    #cfg.DATALOADER.NUM_WORKERS = 2 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    #cfg.SOLVER.IMS_PER_BATCH = 1 
    #cfg.SOLVER.BASE_LR = 0.00025 
    #cfg.SOLVER.MAX_ITER = 300 
    #cfg.SOLVER.STEPS = [] 
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg 


if __name__ == "__main__": 
    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts(r"/Users/HeleneSemb/Documents/Master/Kode/balloon/" + d))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])

    balloon_metadata = MetadataCatalog.get("ballon_train")

    dataset_dicts = get_balloon_dicts(r"/Volumes/helensem/Master/chipsogdip/demo/balloon/train")

    mode = "train"

    cfg = config(r"/cluster/home/helensem/Master/chipsogdip/config/base_config.yaml")

    if mode == "train":
        #Set pretrained weights 
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        #TRAIN
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train() 
    
    elif mode == "evaluate": 
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

        predictor = DefaultPredictor(cfg) 

        evaluator = COCOEvaluator("damage_val", output_dir = "/cluster/home/helensem/Master/chipsogdip/kode/output")
        val_loader = build_detection_test_loader(cfg, "damage_val")
        print(inference_on_dataset(predictor.model, val_loader, evaluator))
