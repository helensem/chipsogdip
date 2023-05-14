import argparse
import os
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
from augmentation import custom_mapper

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset 
from detectron2.data import build_detection_test_loader, build_detection_train_loader

from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import DefaultTrainer


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[T.Resize((800, 800)), #T.RandomBrightness(0.8, 1.8), # T.RandomSaturation(0.8, 1.4),#T.RandomContrast(0.6, 1.3),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),])
        return build_detection_train_loader(cfg, mapper=mapper)
    
    # def build_train_loader(cls, cfg):
    #     return build_detection_train_loader(cfg, mapper=custom_mapper)



def config(backbone_model, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(backbone_model))
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATASETS.TRAIN = ("damage_train",)
    cfg.DATASETS.TEST = ()
    #cfg.TEST.EVAL_PERIOD = 1
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.0005#9062383073017816
    cfg.SOLVER.GAMMA = 0.5
    #cfg.SOLVER.MAX_ITER = 48930 #1631 img* 30 epochs
    cfg.SOLVER.STEPS = [15000, 30000] #Reduce lr by half per 10th epoch  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE =  256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

    ### FROM TUNING
    cfg.SOLVER.MAX_ITER = 1500*22 #30*200 #1631 img* 30 epochs
    # cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2757
    # cfg.MODEL.RPN.NMS_THRESH =  0.800
    # #cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1533
    # #cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1370
    
    # cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.33
    # cfg.SOLVER.MOMENTUM = 0.925
    # #cfg.SOLVER.WEIGHT_DECAY = 9.990238960067115e-05
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8#012575271123081#0.6155672540933761
    # #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8012575271123081
    # #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

    # cfg.INPUT.MIN_SIZE_TRAIN = (836,)
    # cfg.INPUT.MAX_SIZE_TRAIN = 1077
    # #cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.3133287563236277]
    cfg.OUTPUT_DIR = output_dir
    return cfg


def train_model(cfg, backbone):
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(backbone)
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def predict(cfg, damage_metadata):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    output_dir = os.path.join(cfg.OUTPUT_DIR, "images")
    os.makedirs(output_dir, exist_ok=True)
    val_dict = load_damage_dicts(r"/cluster/home/helensem/Master/damage_data", "val")
    for d in val_dict:
        apply_inference(predictor, damage_metadata, output_dir, d, segment_sky=False)


def evaluate(cfg):
    val_dict = load_damage_dicts(r"/cluster/home/helensem/Master/damage_data", "val")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    evaluate_model(cfg, val_dict, write_to_file=True, segment_sky=False)


def inference(cfg):
    val_dict = load_damage_dicts(r"/cluster/home/helensem/Master/damage_data", "val")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    evaluate_over_iterations(cfg, val_dict, cfg.OUTPUT_DIR, plot=True)


if __name__ == "__main__":
    for d in ["train", "val"]:
        DatasetCatalog.register("damage_" + d, lambda d=d: load_damage_dicts(r"/cluster/home/helensem/Master/damage_data",d, segment_sky=False))
        MetadataCatalog.get("damage_" + d).set(thing_classes=["red corrosion"])

    damage_metadata = MetadataCatalog.get("damage_train")

    parser = argparse.ArgumentParser(description="Custom Trainer Script")
    parser.add_argument("--backbone", type=str, default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                        help="Backbone model to use")
    parser.add_argument("--output_dir", type=str, default="/cluster/work/helensem/Master/output/reduced_data/resnext",
                        help="Output directory")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict", "evaluate", "inference"],
                        help="Execution mode")

    args = parser.parse_args()

    mode = args.mode
    backbone_model = args.backbone
    output_dir = args.output_dir

    cfg = config(backbone_model, output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if mode == "train":
        train_model(cfg, backbone_model)
    elif mode == "predict":
        predict(cfg, damage_metadata)
    elif mode == "evaluate":
        evaluate(cfg)
    elif mode == "inference":
        inference(cfg)
    else:
        print("Invalid mode chosen")
