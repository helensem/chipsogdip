import argparse
import numpy as np 
import os, sys 
sys.path.append(r"/cluster/home/helensem/Master/chipsogdip/kode")
from dataset import * 
from eval import * 
from LossEvalHook import LossEvalHook 

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset 
from detectron2.data import build_detection_train_loader

from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
setup_logger()


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[T.RandomFlip(prob=0.5, horizontal=True, vertical=False),T.RandomFlip(prob=0.5, horizontal=False, vertical=True),])
        return build_detection_train_loader(cfg, mapper=mapper)
    
    # def build_train_loader(cls, cfg):
    #     return build_detection_train_loader(cfg, mapper=custom_mapper)

#T.Resize((800, 800)), 
#T.RandomBrightness(0.8, 1.8), 
# #T.RandomSaturation(0.8, 1.4),
#T.RandomContrast(0.6, 1.3),


def config(backbone_model, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(backbone_model))
    # cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATASETS.TRAIN = ("damage_train",)
    cfg.DATASETS.TEST = ()
    #cfg.TEST.EVAL_PERIOD = 1
    cfg.SOLVER.IMS_PER_BATCH = 2
    # cfg.SOLVER.BASE_LR = 0.00070612#0.0005
    #cfg.SOLVER.GAMMA = 0.5
    # cfg.SOLVER.STEPS = []#[6250, 12500] #Reduce lr by half per 10th epoch  15000, 30000
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE =  256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

    ### FROM TUNING
    cfg.SOLVER.MAX_ITER = int(0.5*1500*29) #30*200 #1631 img* 30 epochs
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 1024
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 6857
    cfg.MODEL.RPN.NMS_THRESH =  0.6428571428571428
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2224
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 885
    
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.4938
    cfg.SOLVER.MOMENTUM = 0.95
    cfg.SOLVER.WEIGHT_DECAY = 0.00012163
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7653
    # {'rpn_nms_threshold': 0.6428571428571428, 'rpn_batch_size': 1024, 'pre_nms_limit': 6857, 'post_nms_rois_training': 2224, 'post_nms_rois_inference': 885, 'roi_batch_size': 128, 'roi_positive_ratio': 0.49387755102040815, 'detection_min_confidence': 0.7653061224489797, 'learning_momentum': 0.95, 'weight_decay': 0.0001216326530612245, 'epochs': 29, 'learning_rate': 0.0007061224489795919, 'img_min_size': 989, 'img_max_size': 1148, 'roi_iou_threshold': 0.3571428571428571}
    cfg.INPUT.MIN_SIZE_TRAIN = (989,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1148
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.357]
    cfg.OUTPUT_DIR = output_dir
    return cfg


def train_model(cfg, backbone):
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(backbone)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    inference(cfg)


def predict(cfg, damage_metadata, segment_sky = False):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    output_dir = os.path.join(cfg.OUTPUT_DIR, "images")
    os.makedirs(output_dir, exist_ok=True)
    val_dict = load_damage_dicts(r"/cluster/home/helensem/Master/damage_data", "val")
    for d in val_dict:
        apply_inference(predictor, damage_metadata, output_dir, d, segment_sky)


def evaluate(cfg, segment_sky = False):
    val_dict = DatasetCatalog.get("damage_val")#load_damage_dicts(r"/cluster/home/helensem/Master/damage_data", "val")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    evaluate_model(cfg, val_dict, True, segment_sky)
    evalaute_thresholds(cfg, val_dict)


def inference(cfg):
    val_dict = DatasetCatalog.get("damage_val")#load_damage_dicts(r"/cluster/home/helensem/Master/damage_data", "val")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    evaluate_over_iterations(cfg, val_dict, cfg.OUTPUT_DIR, plot=True, segment_sky=False)

def plot(cfg): 
    print("plotting for: ", cfg.OUTPUT_DIR)
    metrics = ['loss_box_reg', 'loss_cls', 'loss_mask', "fp_fn", 'total_loss']
    path_to_metrics = os.path.join(cfg.OUTPUT_DIR, "metrics.json")
    output_dir = os.path.join(cfg.OUTPUT_DIR, "plots")
    os.makedirs(output_dir, exist_ok=True)
    for m in metrics: 
        plot_metrics(path_to_metrics, output_dir, m)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Trainer Script")
    parser.add_argument("--backbone", type=str, default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                        help="Backbone model to use")
    parser.add_argument("--output_dir", type=str, default="/cluster/work/helensem/Master/output/reduced_data/resnext",
                        help="Output directory")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict", "evaluate", "inference", "plot"],
                        help="Execution mode")
    parser.add_argument("--segment_sky", action="store_true", help="Segment sky")

    args = parser.parse_args()

    mode = args.mode
    backbone_model = args.backbone
    output_dir = args.output_dir
    segment_sky = args.segment_sky
    for d in ["train", "val"]:
        DatasetCatalog.register("damage_" + d, lambda d=d: load_damage_dicts(r"/cluster/home/helensem/Master/damage_data",d))
        MetadataCatalog.get("damage_" + d).set(thing_classes=["red corrosion"])

    damage_metadata = MetadataCatalog.get("damage_train")

    cfg = config(backbone_model, output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if mode == "train":
        train_model(cfg, backbone_model)
    elif mode == "predict":
        predict(cfg, damage_metadata, segment_sky)
    elif mode == "evaluate":
        evaluate(cfg, segment_sky)
    elif mode == "inference":
        inference(cfg)
    elif mode == "plot": 
        plot(cfg)
    else:
        print("Invalid mode chosen")
