import yaml 
import random 
import numpy as np
import sys 
sys.path.append(r"/cluster/home/helensem/Master/chipsogdip/kode")
from training import *
from dataset import * 

def create_hyperparameters(): 
    init_values = {} 
    init_values["RPN_ANCHOR_STRIDE"] = np.array([1,2,3,4])
    init_values["RPN_NMS_THRESHOLD"] = np.linspace(0.5, 1)
    init_values["RPN_TRAIN_ANCHORS_PER_IMAGE"] = np.array([64, 128, 256, 512, 1024])
    init_values["PRE_NMS_LIMIT"] = np.linspace(4000,8000,dtype=int)
    init_values["POST_NMS_ROIS_TRAINING"] = np.linspace(1000,3000,dtype=int)
    init_values["POST_NMS_ROIS_INFERENCE"] = np.linspace(600,2000,dtype=int)
    init_values["MEAN_PIXEL"] = np.array([np.linspace(115.0,130.0,), 
                                            np.linspace(110.0,125.0),
                                            np.linspace(95.0,115.0)])
    print(init_values["MEAN_PIXEL"]) 
    init_values["TRAIN_ROIS_PER_IMAGE"] = np.linspace(150,500,dtype=int)
    init_values["ROI_POSITIVE_RATIO"] = np.linspace(0.2, 0.5)
    init_values["MAX_GT_INSTANCES"] = np.linspace(70,400,dtype=int)
    init_values["DETECTION_MAX_INSTANCES"] = np.linspace(70,400,dtype=int)
    init_values["DETECTION_MIN_CONFIDENCE"] = np.linspace(0.3,0.9)
    init_values["DETECTION_NMS_THRESHOLD"] = np.linspace(0.2,0.7)
    init_values["LEARNING_MOMENTUM"] = np.linspace(0.75,0.95)
    init_values["WEIGHT_DECAY"] = np.linspace(0.00007, 0.000125)
    init_values["rpn_class_loss"] = np.linspace(1,10)
    init_values["rpn_bbox_loss"] = np.linspace(1,10)
    init_values["mrcnn_class_loss"] = np.linspace(1,10)
    init_values["mrcnn_bbox_loss"] = np.linspace(1,10)
    init_values["mrcnn_mask_loss"] = np.linspace(1,10)
    return init_values



# def initia):
#     init_values = create_init_va)
#     for i in r): 
#         indv = {}
#         indv["RPN_ANCHOR_STRIDE"] = init_values["RPN_ANCHOR_STRIDE"][i] 
#         indv["RPN_NMS_THRESHOLD"] = init_values["RPN_NMS_THRESHOLD"][i]
#         indv["RPN_TRAIN_ANCHORS_PER_IMAGE"] = init_values["RPN_TRAIN_ANCHORS_PER_IMAGE"][i] 
#         indv["PRE_NMS_LIMIT"] = init_values["PRE_NMS_LIMIT"][i] 
#         indv["POST_NMS_ROIS_TRAINING"] = init_values["POST_NMS_ROIS_TRAINING"][i]
#         indv["POST_NMS_ROIS_INFERENCE"]=init_values["POST_NMS_ROIS_INFERENCE"][i]
#         indv["MEAN_PIXEL"]=[j[i] for j in init_values["MEAN_PIXEL"]]                                      
#         indv["TRAIN_ROIS_PER_IMAGE"]=init_values["TRAIN_ROIS_PER_IMAGE"][i]
#         indv["ROI_POSITIVE_RATIO"]=init_values["ROI_POSITIVE_RATIO"][i] 
#         indv["MAX_GT_INSTANCES"]=init_values["MAX_GT_INSTANCES"][i]
#         indv["DETECTION_MAX_INSTANCES"]= init_values["DETECTION_MAX_INSTANCES"][i]
#         indv["DETECTION_MIN_CONFIDENCE"]=init_values["DETECTION_MIN_CONFIDENCE"][i]
#         indv["DETECTION_NMS_THRESHOLD"]=init_values["DETECTION_NMS_THRESHOLD"][i]
#         indv["LEARNING_MOMENTUM"] = init_values["LEARNING_MOMENTUM"][i]
#         indv["WEIGHT_DECAY"]= init_values["WEIGHT_DECAY"][i] 
#         init_values_indv ={}
#         init_values = init_values["init_values"]
#         init_values_indv["rpn_class_loss"]= init_values["rpn_class_loss"][i]
#         init_values_indv["rpn_bbox_loss"] = init_values["rpn_bbox_loss"][i]
#         init_values_indv["mrcnn_class_loss"] = init_values["mrcnn_class_loss"][i]
#         init_values_indv["mrcnn_bbox_loss"] = init_values["mrcnn_bbox_loss"][i]
#         init_values_indv["mrcnn_mask_loss"] = init_values["mrcnn_mask_loss"][i]
#         indv["init_values"] = init_values_indv
#         print(indv)
#         with open(f"indivdual_{i}.yaml", "w") as f:
#             yaml.dump(indv, f) 
        







if __name__ == "__main__":
    indvs = 2
    print(create_init_values(indvs)) 