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



from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

# experiment = Experiment(
#   api_key = "52npHX7BlppxxDp6baH9WbQ7M",
#   project_name = "corrosion",
#   workspace="helensem"
# )

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[T.Resize((800, 800)), #T.RandomBrightness(0.8, 1.8), # T.RandomSaturation(0.8, 1.4),#T.RandomContrast(0.6, 1.3),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),])
        return build_detection_train_loader(cfg, mapper=mapper)
    
    # def build_train_loader(cls, cfg):
    #     return build_detection_train_loader(cfg, mapper=custom_mapper)

def config():
    """
    Standard config """
    cfg = get_cfg() 
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))#mask_rcnn_X_101_32x8d_FPN_3x.yaml")) #("LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))# #! MUST MATCH WITH TRAINING WEIGHTS
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATASETS.TRAIN = ("damage_train",)
    cfg.DATASETS.TEST = ()
    #cfg.TEST.EVAL_PERIOD = 1
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.0005#9062383073017816
    cfg.SOLVER.GAMMA = 0.5
    #cfg.SOLVER.MAX_ITER = 48930 #1631 img* 30 epochs
    cfg.SOLVER.STEPS = [15000, 30000] #Reduce lr by half per 10th epoch  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8012575271123081#0.6155672540933761
    # #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8012575271123081
    # #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

    # cfg.INPUT.MIN_SIZE_TRAIN = (836,)
    # cfg.INPUT.MAX_SIZE_TRAIN = 1077
    # #cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.3133287563236277]
#{'rpn_nms_threshold': 0.8001330702641066, 'rpn_batch_size': 256, 'pre_nms_limit': 2757, 'post_nms_rois_training': 1533, 'post_nms_rois_inference': 1370, 'mean_pixel': array([      115.9,      117.93,      97.548]), 'roi_batch_size': 128, 'roi_positive_ratio': 0.33132132259563285, 'detection_min_confidence': 0.8012575271123081, 'learning_momentum': 0.9254784359878887, 'weight_decay': 9.990238960067115e-05, 'rpn_bbox_loss': 3.90191755967507, 'roi_bbox_loss': 6.078694180418836, 'epochs': 22, 'learning_rate': 0.0009062383073017816, 'img_min_size': 836, 'img_max_size': 1077, 'roi_iou_threshold': 0.3133287563236277}

    cfg.OUTPUT_DIR = "/cluster/work/helensem/Master/output/reduced_data/resnet50" #! MUST MATCH WITH CURRENT MODEL 

    return cfg 
 

#hyper_params = {'rpn_nms_threshold': 0.5719887128477639, 'rpn_batch_size': 64, 'pre_nms_limit': 1518, 'post_nms_rois_training': 2682, 'post_nms_rois_inference': 709, 'roi_batch_size': 1024, 'roi_positive_ratio': 0.2636980669816439, 'detection_min_confidence': 0.6155672540933761, 'learning_momentum': 0.792350687427757, 'weight_decay': 8.446622163969797e-05, 'rpn_bbox_loss': 8.437500394677944, 'roi_bbox_loss': 7.371256173459417, 'epochs': 21, 'learning_rate': 0.00010860511441900859, 'img_min_size': 591, 'img_max_size': 1124}

#experiment.log_parameters(hyper_params)

if __name__ == "__main__":
    mode = "evaluate"
    for d in ["train", "val"]:
        DatasetCatalog.register("damage_" + d, lambda d=d: load_damage_dicts(r"/cluster/home/helensem/Master/damage_data",d, segment_sky=False))
        MetadataCatalog.get("damage_" + d).set(thing_classes=["red corrosion"])

    damage_metadata = MetadataCatalog.get("damage_train")

    cfg = config() 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    

    if mode == "train":
        #Set pretrained weights 
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")#("LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")# ##! MUST MATCH WITH CURRENT MODEL 

        
        #TRAIN
        trainer = CustomTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train() 
        #log_model(experiment, trainer, model_name="resnet-101")
    
    elif mode == "predict": 
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8012575271123081

        predictor = DefaultPredictor(cfg)
        output_dir = os.path.join(cfg.OUTPUT_DIR, "images")
        os.makedirs(output_dir, exist_ok=True)

        val_dict = load_damage_dicts(r"/cluster/home/helensem/Master/damage_data", "val")
        for d in val_dict:
            apply_inference(predictor, damage_metadata, output_dir, d, segment_sky = False)
    
    elif mode == "evaluate":
        #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        val_dict = load_damage_dicts(r"/cluster/home/helensem/Master/damage_data", "val")
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

        evaluate_model(cfg, val_dict, write_to_file = True, plot=True, segment_sky=False) 

    elif mode == "inference": 
        val_dict = load_damage_dicts(r"/cluster/home/helensem/Master/damage_data", "val")
        #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        evaluate_over_iterations(cfg,val_dict,cfg.OUTPUT_DIR, plot=True)

    # elif mode == "predict":

    #     cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    #     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

    #     predictor = DefaultPredictor(cfg) 
    #     path = r"/cluster/home/helensem/Master/output/sky"
    #     image_ids = next(os.walk(path))[2]

    #     #dataset_dicts = load_damage_dicts(r"/cluster/home/helensem/Master/Labeled_pictures", "val")
    #     os.makedirs(os.path.join(path, "predictions_sky_resnext"), exist_ok = True)
        
    #     for d in image_ids:
    #         image_path = os.path.join(path, d) 
    #         im = cv2.imread(image_path)
    #         outputs = predictor(im)
    #         v = Visualizer(im[:, :, ::-1],
    #                         metadata = damage_metadata,
    #                         scale = 0.5)
    #         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #         output_path = os.path.join(path, "predictions_sky_resnext", d)

    #         cv2.imwrite(output_path,out.get_image()[:,:,::-1])

    
    else: 
        print("No mode chosen")








