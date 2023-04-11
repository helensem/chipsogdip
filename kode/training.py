import detectron2 

from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np 
import os,json,cv2,random, sys 
#import skimage
sys.path.append(r"/cluster/home/helensem/Master/chipsogdip/kode")
from dataset import * 
from eval import * 
from LossEvalHook import LossEvalHook 

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg 
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode 
from detectron2.evaluation import COCOEvaluator, inference_on_dataset 
from detectron2.data import build_detection_test_loader 

from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.engine import DefaultTrainer


class CustomTrainer(DefaultTrainer):
    """
    Custom Trainer deriving from the "DefaultTrainer"

    Overloads build_hooks to add a hook to calculate loss on the test set during training.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            100, # Frequency of calculation - every 100 iterations here
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))

        return hooks



def config():
    """
    Standard config """
    cfg = get_cfg() 
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_regnetx_4gf_dds_fpn_1x.py"))#mask_rcnn_X_101_32x8d_FPN_3x.yaml")) #mask_rcnn_R_101_FPN_3x.yaml"))#  #! MUST MATCH WITH TRAINING WEIGHTS
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATASETS.TRAIN = ("damage_train",)
    cfg.DATASETS.TEST = ()
    #cfg.TEST.EVAL_PERIOD = 1630
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.GAMMA = 0.5
    #cfg.SOLVER.MAX_ITER = 48930 #1631 img* 30 epochs
    cfg.SOLVER.STEPS = [16310, 32620] #Reduce lr by half per 10th epoch  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    cfg.OUTPUT_DIR = "/cluster/work/helensem/Master/output/run3/regnet" #! MUST MATCH WITH CURRENT MODEL 

    return cfg 
 




if __name__ == "__main__":
    mode = "train"
    for d in ["train", "val"]:
        DatasetCatalog.register("damage_" + d, lambda d=d: load_damage_dicts(r"/cluster/home/helensem/Master/Labeled_pictures",d))
        MetadataCatalog.get("damage_" + d).set(thing_classes=["damage"])

    damage_metadata = MetadataCatalog.get("damage_train")

    cfg = config() 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    

    if mode == "train":
        #Set pretrained weights 
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_regnetx_4gf_dds_fpn_1x.py")#mask_rcnn_X_101_32x8d_FPN_3x.yaml") #! MUST MATCH WITH CURRENT MODEL 

        
        #TRAIN
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train() 
    
    elif mode == "inference": 
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

        predictor = DefaultPredictor(cfg)
        output_dir = os.path.join(cfg.OUTPUT_DIR, "images")
        os.makedirs(output_dir, exist_ok=True)

        val_dict = load_damage_dicts(r"/cluster/home/helensem/Master/Labeled_pictures", "val")
        for d in val_dict:
            apply_inference(predictor, damage_metadata, output_dir, d, d["file_name"])

    elif mode == "predict":

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

        predictor = DefaultPredictor(cfg) 
        path = r"/cluster/home/helensem/Master/output/sky"
        image_ids = next(os.walk(path))[2]

        #dataset_dicts = load_damage_dicts(r"/cluster/home/helensem/Master/Labeled_pictures", "val")
        os.makedirs(os.path.join(path, "predictions_sky_resnext"), exist_ok = True)
        
        for d in image_ids:
            image_path = os.path.join(path, d) 
            im = cv2.imread(image_path)
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                            metadata = damage_metadata,
                            scale = 0.5)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            output_path = os.path.join(path, "predictions_sky_resnext", d)

            cv2.imwrite(output_path,out.get_image()[:,:,::-1])

    
    elif mode == "evaluate":
        val_dict = load_damage_dicts(r"/cluster/home/helensem/Master/Labeled_pictures", "val")
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

        evaluate_model(cfg, val_dict, True) 

    else: 
        print("No mode chosen")








