import detectron2 

from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np 
import os,json,cv2,random 
import skimage
from kode.dataset import * 

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg 
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode 
from detectron2.utils.visualizer import ColorMode 
from detectron2.evaluation import COCOEvaluator, inference_on_dataset 
from detectron2.data import build_detection_test_loader 

from detectron2.engine import DefaultTrainer


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



def train(cfg_file): 
    cfg = get_cfg() 
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(cfg_file)


    #Set pretrained weights 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    #TRAIN
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

#def predict_and(cfg_file, output_dir):
 #   cfg = get_cfg() 
 #   cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  #  cfg.merge_from_file(cfg_file)
  #  cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_final.pth")
  #  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

  #  predictor = DefaultPredictor(cfg) 

#def evaluate():
 #   evaluator = COCOEvaluator("balloon_val", output_dir = "/Users/HeleneSemb/Documents/Master/Kode/output")
  #  val_loader = build_detection_test_loader(cfg, "balloon_val")
   # print(inference_on_dataset(predictor.model, val_loader, evaluator))

   # return


if __name__ == "__main__":
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
    
    elif mode == "predict":

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

        predictor = DefaultPredictor(cfg) 

        dataset_dicts = load_damage_dicts(r"/cluster/home/helensem/Master/chipsogdip/Labeled_pictures", "val")
        for d in random.sample(dataset_dicts, 1): 
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                            metadata = damage_metadata,
                            scale = 0.5,
                            instance_mode = ColorMode.IMAGE_BW)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow("image",out.get_image()[:,:,::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif mode == "evaluate":

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

        predictor = DefaultPredictor(cfg) 

        evaluator = COCOEvaluator("damage_val", output_dir = "/cluster/home/helensem/Master/chipsogdip/kode/output")
        val_loader = build_detection_test_loader(cfg, "damage_val")
        print(inference_on_dataset(predictor.model, val_loader, evaluator))
    
    else: 
        print("No mode chosen")








