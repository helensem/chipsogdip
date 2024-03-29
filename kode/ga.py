# import deap 
# from deap import base
# from deap import creator
# from deap import tools
# import deap.algorithms


# import dask.dataframe as dd
import pandas as pd

import numpy as np
import random
import json

import sys 
sys.path.append("cluster/home/helensem/Master/chipsogdip/kode")
from eval import evaluate_model
from training import config
from dataset import load_damage_dicts, get_json_dict
import os
from matplotlib import pyplot as plt

from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg 
from detectron2 import model_zoo


# def generate_random_hyperparameters_full():
#     rpn_anchor_stride = random.randint(1,4)
#     rpn_nms_threshold = random.uniform(0.5,1)
#     rpn_train_anchors_per_image = np.random.choice([64, 128, 256, 512, 1024])
#     pre_nms_limit = random.uniform(1000,3000, dtype=int)
#     post_nms_rois_training = random.uniform(1000,3000, dtype=int)
#     post_nms_rois_inference = random.uniform(600,2000, dtype=int)
#     mean_pixel = np.array([random.uniform(115.0,130.0), 
#                                             random.uniform(110.0,125.0),
#                                             random.uniform(95.0,115.0)])
#     train_rois_per_image = random.uniform(150,500, dtype=int)
#     rois_positive_ratio = random.uniform(0.2,0.5)
#     max_gt_instances = random.uniform(70,400, dtype=int)
#     detection_max_instances = random.uniform(70,400, dtype=int)
#     detection_min_confidence = random.uniform(0.3,0.9)
#     detection_nms_threshold = random.uniform(0.2,0.7)
#     learning_momentum = random.uniform(0.75,0.95)
#     weight_decay = random.uniform(0.00007, 0.000125)
#     rpn_class_loss = random.uniform(1,10)
#     rpn_bbox_loss = random.uniform(1,10)
#     mrcnn_class_loss = random.uniform(1,10)
#     mrcnn_bbox_loss = random.uniform(1,10)
#     mrcnn_mask_loss = random.uniform(1,10)
#     return  rpn_anchor_stride, rpn_nms_threshold, rpn_anchor_stride, rpn_train_anchors_per_image, pre_nms_limit, post_nms_rois_training, post_nms_rois_inference, mean_pixel, train_rois_per_image, rois_positive_ratio, max_gt_instances, detection_max_instances, detection_min_confidence, detection_nms_threshold, learning_momentum, rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss 
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def mutate_hyperparameters(hyperparameters):
  mutation_probability = 0.05
  if key == "rpn_anchor_stride":
    hyperparameters['learning_rate'] = random.uniform(0.0001, 0.001)

def mutate(key):
    mut_value = 0
    if key == "rpn_anchor_stride":
        mut_value = random.randint(1,4)
    elif key == "learning_rate": 
        mut_value = random.uniform(0.0001,0.001)
    elif key == "rpn_nms_threshold":
        mut_value = random.uniform(0.5,1)
    elif key == "rpn_batch_size":
         mut_value = np.random.choice([64, 128, 256, 512, 1024])
    elif key == "pre_nms_limit":
         mut_value = random.randint(1000,3000)
    elif key == "post_nms_rois_training":
         mut_value = random.randint(1000,3000)
    elif key == "post_nms_rois_inference":
        mut_value = random.randint(600,2000)
    elif key == "mean_pixel":
         mut_value = np.array([random.uniform(115.0,130.0), 
                                            random.uniform(110.0,125.0),
                                            random.uniform(95.0,115.0)])             
    elif key == "roi_batch_size":
        mut_value = np.random.choice([64, 128, 256, 512, 1024])
    elif key == "roi_positive_ratio":
        mut_value = random.uniform(0.2,0.5)
    elif key == "max_gt_instances":
        mut_value = random.randint(70,400)
    elif key == "detection_max_instances":
        mut_value = random.randint(70,400)
    elif key == "detection_min_confidence":
        mut_value = random.uniform(0.3,0.9)
    elif key == "detection_nms_threshold":
        mut_value = random.uniform(0.2,0.7)
    elif key == "learning_momentum":
        mut_value = random.uniform(0.75,0.95)
    elif key == "weight_decay":
        mut_value = random.uniform(0.00007, 0.000125)
    elif key == "rpn_class_loss":
        mut_value = random.uniform(1,10)
    elif key == "rpn_bbox_loss":
        mut_value = random.uniform(1,10)
    elif key == "mrcnn_class_loss":
        mut_value = random.uniform(1,10)
    elif key == "roi_bbox_loss":
        mut_value = random.uniform(1,10)
    elif key == "mrcnn_mask_loss":
        mut_value = random.uniform(1,10)
    elif key == "epochs": 
        mut_value = random.randint(20,40)
    elif key == "img_min_size": 
        mut_value = random.randint(500,1000)
    elif key == "img_max_size": 
        mut_value = random.randint(900,1600)
    elif key == "roi_iou_threshold":
        mut_value = random.uniform(0.3,0.8)
    else: 
        print("no key available")
        mut_value = 0
    return mut_value

# def build_model(hidden_layer_size, learning_rate, dropout_rate):
#     ##### CHANGE HERE ##########
#     print (hidden_layer_size, learning_rate, dropout_rate)
#     if dropout_rate > 1 :
#         dropout_rate = 0
#     if hidden_layer_size < 1 :
#         hidden_layer_size = 1
#     if learning_rate > 0.5:
#         learning_rate = 0.01
#     #it seems that sometimes, the values are sent in the wrong order so I used int an float to respect the type
#     model = tf.keras.Sequential()
#     #input_shape = (batch_size, time_steps, features)
#     model.add(LSTM(units=int(hidden_layer_size), input_shape=(5, 1)))
#     model.add(Dropout(float(dropout_rate)))
#     model.add(Dense(1))
#     optimizer = Adam(learning_rate=float(learning_rate))
#     model.compile(loss='mean_squared_error', optimizer=optimizer)
#     return model


####* CODE FROM KNAPSACK PROBLEM

def generate_hyperparameters(): 
    init_values = {} 
    init_values["rpn_nms_threshold"] = np.linspace(0.5, 1)
    init_values["rpn_batch_size"] = np.array([64, 128, 256, 512, 1024])
    init_values["pre_nms_limit"] = np.linspace(4000,8000,dtype=int)
    init_values["post_nms_rois_training"] = np.linspace(1000,3000,dtype=int)
    init_values["post_nms_rois_inference"] = np.linspace(600,2000,dtype=int)
    init_values["roi_batch_size"] = np.array([64, 128, 256, 512, 1024])#np.linspace(150,500,dtype=int)
    init_values["roi_positive_ratio"] = np.linspace(0.2, 0.5)
    init_values["detection_min_confidence"] = np.linspace(0.3,0.9)
    init_values["learning_momentum"] = np.linspace(0.75,0.95)
    init_values["weight_decay"] = np.linspace(0.00007, 0.000125)
    init_values["epochs"] = np.linspace(20,40, dtype=int)
    init_values["learning_rate"] = np.linspace(0.0001, 0.001)
    init_values["img_min_size"] = np.linspace(500,1000, dtype=int)
    init_values["img_max_size"] = init_values["img_min_size"] + 200
    init_values["roi_iou_threshold"] = np.linspace(0.3,0.7)
    return init_values

def ga_train(indv, generation, epochs, rpn_batch_size, roi_batch_size, rpn_nms_thresh, learning_rate, pre_nms_limit, post_nms_train, post_nms_val, roi_pos_ratio, momentum, weight_decay, det_thresh, roi_iou_thresh, img_min_size, img_max_size):
    """ For training with the genetic algorithm, changing the hyperparameters
    """
    
    
    cfg = get_cfg() 
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  #! MUST MATCH WITH TRAINING WEIGHTS
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATASETS.TRAIN = ("ga_damage_train",)
    cfg.DATASETS.TEST = ()
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.STEPS = []
    #cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.MAX_ITER = 100*epochs #30*200 #1631 img* 30 epochs
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    cfg.OUTPUT_DIR = f"/cluster/work/helensem/Master/output/run_ga_dc/gen_{generation}/{indv}" #! MUST MATCH WITH CURRENT MODEL 
    
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = rpn_batch_size
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_batch_size

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = pre_nms_limit
    cfg.MODEL.RPN.NMS_THRESH = rpn_nms_thresh
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = post_nms_train 
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = post_nms_val
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [roi_iou_thresh]
    
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = roi_pos_ratio 
    cfg.SOLVER.MOMENTUM = momentum
    cfg.SOLVER.WEIGHT_DECAY = weight_decay
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = det_thresh
    #cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = roi_bbox_loss
    
    #cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = rpn_bbox_loss

    cfg.INPUT.MIN_SIZE_TRAIN = (img_min_size,) 
    if img_max_size < img_min_size: 
        img_max_size = img_min_size + 100
    # maximum image size for the train set
    cfg.INPUT.MAX_SIZE_TRAIN = img_max_size

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    #TRAIN
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml") #! MUST MATCH WITH CURRENT MODEL 

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg


def calculate_fitness(indv_num, hyperparameters, generation):
    
    print(" generation: ", generation, "indv: ", indv_num, "\n", hyperparameters)

    epochs = int(hyperparameters["epochs"])
    rpn_batch_size = int(hyperparameters["rpn_batch_size"])
    roi_batch_size = int(hyperparameters["roi_batch_size"])
    rpn_nms_thresh = float(hyperparameters["rpn_nms_threshold"])
    learning_rate = float(hyperparameters["learning_rate"])
    pre_nms_limit = int(hyperparameters["pre_nms_limit"])
    post_nms_train = int(hyperparameters["post_nms_rois_training"])
    post_nms_val = int(hyperparameters["post_nms_rois_inference"])
    roi_pos_ratio = float(hyperparameters["roi_positive_ratio"])
    momentum = float(hyperparameters["learning_momentum"])
    weight_decay = float(hyperparameters["weight_decay"])
    det_thresh = float(hyperparameters["detection_min_confidence"])
    img_min_size = int(hyperparameters["img_min_size"])
    img_max_size = int(hyperparameters["img_max_size"])
    roi_iou_thresh = float(hyperparameters["roi_iou_threshold"])

    cfg = ga_train(indv_num, generation, epochs, rpn_batch_size, roi_batch_size, rpn_nms_thresh, learning_rate, pre_nms_limit, post_nms_train, post_nms_val, roi_pos_ratio, momentum, weight_decay, det_thresh, roi_iou_thresh, img_min_size, img_max_size)

    #TRAIN

    #EVALUATE 
    val_dict = DatasetCatalog.get("ga_damage_val")#load_damage_dicts(r"/cluster/home/helensem/Master/data/set1", "val")#get_json_dict(r"/cluster/home/helensem/Master/data/set1","val")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    string = json.dumps(hyperparameters, cls=NpEncoder)
    with open(f"/cluster/work/helensem/Master/output/run_ga_dc/gen_{generation}/{indv_num}/hyperparameters.txt", "w") as f: 
       f.write(string)
    corr_iou, bg_iou, mean_iou = evaluate_model(cfg, val_dict, True) 

    return corr_iou#score



def evaluate_indvs(num_gen, num_indv):
    best_performing = []
    ious = []
    for gen in range(num_gen):
        best_iou = 0.0
        best_per_gen = 0 
        for indv in range(num_indv):
            path = f"/cluster/work/helensem/Master/output/run_ga/gen_{gen}/{indv}"
            with open(os.path.join(path,"hyperparameters.txt"), "r") as f: 
                data = f.read()
            # data = data.split(",")
            # if len(data) == 20:
            #     del data[5:8]
            # else:
            #     del data[5]
            # data = ",".join(data)
            # data = data.replace("\'", "\"")
            # print(data)
            hyperparameters = json.loads(data)
            mean_iou = calculate_fitness(indv, hyperparameters, gen)#evaluate_model(cfg, val_dict)
            if mean_iou > best_iou: 
                best_per_gen = indv
                best_iou = mean_iou
        best_performing.append(best_per_gen)
        ious.append(best_iou)
    return best_performing, ious

def plot_hyperparameters(list_of_indvs, key):
    x = np.arange(1,len(list_of_indvs)+1)
    values = []
    for indv, iou in list_of_indvs:
        #path = f"/cluster/work/helensem/Master/output/run_ga/gen_{gen}/{indv}/hyperparameters.txt"
        # with open(path, "r") as f: 
        #     data = f.read()
        # data = data.split(",")
        # if len(data) == 20:
        #     del data[5:8]
        # else:
        #     del data[5]
        # data = ",".join(data)
        # data = data.replace("\'", "\"")
        #hyperparameters = json.loads(data)
        point = indv[key]
        values.append(point) 
    plt.clf()
    plt.plot(x, values, color = "b", marker = "o")
    plt.xlabel("Generations")
    plt.ylabel(key)
    plt.savefig(f"/cluster/work/helensem/Master/output/run_ga_dc/{key}.svg", format = "svg")




if __name__ == "__main__":
    mutation_rate = 0.2
    generations = 15
    #hyperparameters = {}
    hyperparameters = generate_hyperparameters()
    population_size = 10

    population = [dict(zip(hyperparameters.keys(), [random.choice(values) for values in hyperparameters.values()])) for _ in range(population_size)]

    path = f"/cluster/home/helensem/Master/data/set1"

    for d in ["train", "val"]:
        DatasetCatalog.register("ga_damage_" + d, lambda d=d: get_json_dict(path, d))
        MetadataCatalog.get("ga_damage_" + d).set(thing_classes=["damage"])
    
    # best_indvs, ious = evaluate_indvs(generations, population_size)
    # print(best_indvs)
    # print(ious)
    # x = np.arange(1,generations+1)
    # plt.plot(x, ious, color = "b", marker = "o")
    # plt.xlabel("Generations")
    # plt.ylabel("IoU")
    # plt.savefig(f"/cluster/work/helensem/Master/output/run_ga/ious")
    best_indvs = [2, 9, 9, 2, 8, 0, 7, 9, 2, 4, 0, 2, 7, 3, 3]
    # best_indvs = [15, 9, 0, 2, 10, 9, 1, 7, 1, 10, 11, 6, 18, 1, 12]

    for key in hyperparameters.keys(): 
        plot_hyperparameters(best_indvs, key)

    #[2, 9, 9, 2, 8, 0, 7, 9, 2, 4, 0, 2, 7, 3, 3]
#[0.57311934, 0.584835, 0.56750065, 0.5816272, 0.5715605, 0.57222766, 0.57425714, 0.5807295, 0.5800988, 0.5975712, 0.59818673, 0.5800525, 0.5750274, 0.5944383, 0.58665615]
    #
    # fittest_per_gen = []
    # for generation in range(generations):
        
    #     # evaluate the fitness of each individual in the population
    #     fitness_scores = [calculate_fitness(idx, individual, generation) for idx, individual in enumerate(population)]
        
    #     # select the fittest individuals to breed the next generation
    #     sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    #     fittest_individuals = sorted_population[:int(population_size/2)]
    #     fittest_per_gen.append(sorted_population[0])
    #     print("fittest per generation: ", fittest_per_gen)
    #     # create the next generation by breeding the fittest individuals
    #     new_population = []
    #     while len(new_population) < population_size:
    #         # randomly select two parents from the fittest individuals
    #         parent1, parent2 = random.sample(fittest_individuals, k=2)
            
    #         # create a new individual by randomly selecting hyperparameters from the parents
    #         new_individual = {}
    #         for key in hyperparameters.keys():
    #             if random.random() < mutation_rate:
    #                 # randomly mutate the hyperparameter with a small random value
    #                 new_individual[key] = mutate(key)
    #             else:
    #                 # randomly select the hyperparameter from one of the parents
    #                 new_individual[key] = random.choice([parent1[key], parent2[key]])
    #         new_population.append(new_individual)
        
    #     # update the population with the new generation
    #     population = new_population

    # # select the fittest individual from the final population
    # fitness_scores = [calculate_fitness(idx, individual, generations) for idx, individual in enumerate(population)]
    # sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    # fittest_individual = sorted_population[0]
    # txt_file = r"/cluster/work/helensem/Master/output/run_ga/fittest_ind.txt"
    # with open(txt_file, "w") as f:
    #     f.write(str(fittest_individual))
    # print("Best individual is: ", fittest_individual)
    # print(fittest_per_gen)

        
