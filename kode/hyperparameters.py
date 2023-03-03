import yaml 
import random 
import numpy as np
import sys 
sys.path.append(r"/cluster/home/helensem/Master/chipsogdip/kode")
from training import *
from dataset import * 

def create_init_values(num_indvs): 
    init_values = {} 
    init_values["RPN_ANCHOR_STRIDE"] = np.random.randint(1,4,size=num_indvs)
    init_values["RPN_NMS_THRESHOLD"] = np.linspace(0.5, 1,num=num_indvs)
    train_anchors = np.array([64, 128, 256, 512, 1024])
    init_values["RPN_TRAIN_ANCHORS_PER_IMAGE"] = np.random.choice(train_anchors, size = num_indvs)
    init_values["PRE_NMS_LIMIT"] = np.linspace(4000,8000,num=num_indvs,dtype=int)
    init_values["POST_NMS_ROIS_TRAINING"] = np.linspace(1000,3000,num=num_indvs,dtype=int)
    init_values["POST_NMS_ROIS_INFERENCE"] = np.linspace(600,2000,num=num_indvs,dtype=int)
    init_values["MEAN_PIXEL"] = np.array([np.linspace(115.0,130.0,num=num_indvs,), 
                                            np.linspace(110.0,125.0,num=num_indvs),
                                            np.linspace(95.0,115.0,num=num_indvs)])
    print(init_values["MEAN_PIXEL"]) 
    init_values["TRAIN_ROIS_PER_IMAGE"] = np.linspace(150,500,num=num_indvs,dtype=int)
    init_values["ROI_POSITIVE_RATIO"] = np.linspace(0.2, 0.5, num = num_indvs)
    init_values["MAX_GT_INSTANCES"] = np.linspace(70,400,num=num_indvs,dtype=int)
    init_values["DETECTION_MAX_INSTANCES"] = np.linspace(70,400,num=num_indvs,dtype=int)
    init_values["DETECTION_MIN_CONFIDENCE"] = np.linspace(0.3,0.9, num=num_indvs)
    init_values["DETECTION_NMS_THRESHOLD"] = np.linspace(0.2,0.7,num=num_indvs)
    init_values["LEARNING_MOMENTUM"] = np.linspace(0.75,0.95, num=num_indvs)
    init_values["WEIGHT_DECAY"] = np.linspace(0.00007, 0.000125, num=num_indvs)
    loss_weights = {}
    loss_weights["rpn_class_loss"] = np.linspace(1,10,num=num_indvs)
    loss_weights["rpn_bbox_loss"] = np.linspace(1,10,num=num_indvs)
    loss_weights["mrcnn_class_loss"] = np.linspace(1,10,num=num_indvs)
    loss_weights["mrcnn_bbox_loss"] = np.linspace(1,10,num=num_indvs)
    loss_weights["mrcnn_mask_loss"] = np.linspace(1,10,num=num_indvs)
    init_values["LOSS_WEIGHTS"] = loss_weights
    return init_values



def initialize(num_indvs):
    init_values = create_init_values(num_indvs=num_indvs)
    for i in range(num_indvs): 
        indv = {}
        indv["RPN_ANCHOR_STRIDE"] = init_values["RPN_ANCHOR_STRIDE"][i] 
        indv["RPN_NMS_THRESHOLD"] = init_values["RPN_NMS_THRESHOLD"][i]
        indv["RPN_TRAIN_ANCHORS_PER_IMAGE"] = init_values["RPN_TRAIN_ANCHORS_PER_IMAGE"][i] 
        indv["PRE_NMS_LIMIT"] = init_values["PRE_NMS_LIMIT"][i] 
        indv["POST_NMS_ROIS_TRAINING"] = init_values["POST_NMS_ROIS_TRAINING"][i]
        indv["POST_NMS_ROIS_INFERENCE"]=init_values["POST_NMS_ROIS_INFERENCE"][i]
        indv["MEAN_PIXEL"]=[j[i] for j in init_values["MEAN_PIXEL"]]                                      
        indv["TRAIN_ROIS_PER_IMAGE"]=init_values["TRAIN_ROIS_PER_IMAGE"][i]
        indv["ROI_POSITIVE_RATIO"]=init_values["ROI_POSITIVE_RATIO"][i] 
        indv["MAX_GT_INSTANCES"]=init_values["MAX_GT_INSTANCES"][i]
        indv["DETECTION_MAX_INSTANCES"]= init_values["DETECTION_MAX_INSTANCES"][i]
        indv["DETECTION_MIN_CONFIDENCE"]=init_values["DETECTION_MIN_CONFIDENCE"][i]
        indv["DETECTION_NMS_THRESHOLD"]=init_values["DETECTION_NMS_THRESHOLD"][i]
        indv["LEARNING_MOMENTUM"] = init_values["LEARNING_MOMENTUM"][i]
        indv["WEIGHT_DECAY"]= init_values["WEIGHT_DECAY"][i] 
        loss_weights_indv ={}
        loss_weights = init_values["LOSS_WEIGHTS"]
        loss_weights_indv["rpn_class_loss"]= loss_weights["rpn_class_loss"][i]
        loss_weights_indv["rpn_bbox_loss"] = loss_weights["rpn_bbox_loss"][i]
        loss_weights_indv["mrcnn_class_loss"] = loss_weights["mrcnn_class_loss"][i]
        loss_weights_indv["mrcnn_bbox_loss"] = loss_weights["mrcnn_bbox_loss"][i]
        loss_weights_indv["mrcnn_mask_loss"] = loss_weights["mrcnn_mask_loss"][i]
        indv["LOSS_WEIGHTS"] = loss_weights_indv
        print(indv)
        with open(f"indivdual_{i}.yaml", "w") as f:
            yaml.dump(indv, f) 


def ga_drive(cfg_files_path, image_path):
    cfg_files = next(os.walk(cfg_files_path))[2]
    for cfg in cfg_files:
        cfg_path = os.path.join(cfg_files_path, cfg)

        indv_cfg = config(cfg_path)
        train(indv_cfg)
        
        







if __name__ == "__main__":
    indvs = 2
    print(create_init_values(indvs)) 