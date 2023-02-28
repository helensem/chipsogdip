import deap 
from deap import base
from deap import creator
from deap import tools
import deap.algorithms

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import dask.dataframe as dd
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import random
from tensorflow.keras.optimizers import Adam
import sys 
sys.path.append("cluster/home/helensem/Master/chipsogdip/kode")
from eval import evaluate_model
from training import config, train
from dataset import load_damage_dicts
import os
from detectron2.engine import DefaultPredictor 



def windowed_dataset(series, window_size=G.WINDOW_SIZE, batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE):
   """
   We create time windows to create X and y features.
   For example, if we choose a window of 30, we will create a dataset formed by 30 points as X
   """
   dataset = tf.data.Dataset.from_tensor_slices(series)
   dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
   dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
   dataset = dataset.shuffle(shuffle_buffer)
   dataset = dataset.map(lambda window: (window[:-1], window[-1]))
   dataset = dataset.batch(batch_size).prefetch(1)
   return dataset

def evaluate(hyperparameters):
    #####
    ##### CHANGE HERE #####
    ######
    #####

    learning_rate = hyperparameters
    cfg = config(learning_rate)

    #TRAIN
    train(cfg)

    #EVALUATE 
    val_dict = load_damage_dicts(r"/cluster/home/helensem/Master/Labeled_pictures", "val")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    predictor = DefaultPredictor(cfg)

    corr_iou, bg_iou, mean_iou = evaluate_model(predictor, val_dict) 
    score = 1 - mean_iou 

    return (score,)

def generate_random_hyperparameters_full():
    rpn_anchor_stride = random.randint(1,4)
    rpn_nms_threshold = random.uniform(0.5,1)
    rpn_train_anchors_per_image = np.random.choice([64, 128, 256, 512, 1024])
    pre_nms_limit = random.uniform(1000,3000, dtype=int)
    post_nms_rois_training = random.uniform(1000,3000, dtype=int)
    post_nms_rois_inference = random.uniform(600,2000, dtype=int)
    mean_pixel = np.array([random.uniform(115.0,130.0), 
                                            random.uniform(110.0,125.0),
                                            random.uniform(95.0,115.0)])
    train_rois_per_image = random.uniform(150,500, dtype=int)
    rois_positive_ratio = random.uniform(0.2,0.5)
    max_gt_instances = random.uniform(70,400, dtype=int)
    detection_max_instances = random.uniform(70,400, dtype=int)
    detection_min_confidence = random.uniform(0.3,0.9)
    detection_nms_threshold = random.uniform(0.2,0.7)
    learning_momentum = random.uniform(0.75,0.95)
    weight_decay = random.uniform(0.00007, 0.000125)
    rpn_class_loss = random.uniform(1,10)
    rpn_bbox_loss = random.uniform(1,10)
    mrcnn_class_loss = random.uniform(1,10)
    mrcnn_bbox_loss = random.uniform(1,10)
    mrcnn_mask_loss = random.uniform(1,10)
    return  rpn_anchor_stride, rpn_nms_threshold, rpn_anchor_stride, rpn_train_anchors_per_image, pre_nms_limit, post_nms_rois_training, post_nms_rois_inference, mean_pixel, train_rois_per_image, rois_positive_ratio, max_gt_instances, detection_max_instances, detection_min_confidence, detection_nms_threshold, learning_momentum, rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss 


def generate_random_hyperparameters(): #* For testing
    learning_rate = random.uniform(0.001, 0.1)
    return learning_rate


def mutate_hyperparameters(hyperparameters):
  mutation_probability = 0.05
  if random.uniform(0, 1) < mutation_probability:
    hyperparameters['learning_rate'] = random.uniform(0.001, 0.1)

def mutate_random_hyperparameters_full(hyperparameters):
    mutation_probability = 0.05
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["rpn_anchor_stride"] = random.randint(1,4)
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["rpn_nms_threshold"] = random.uniform(0.5,1)
    if random.uniform(0, 1) < mutation_probability:
         hyperparameters["rpn_train_anchors_per_image"] = np.random.choice([64, 128, 256, 512, 1024])
    if random.uniform(0, 1) < mutation_probability:
         hyperparameters["pre_nms_limit"] = random.uniform(1000,3000, dtype=int)
    if random.uniform(0, 1) < mutation_probability:
         hyperparameters["post_nms_rois_training"] = random.uniform(1000,3000, dtype=int)
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["post_nms_rois_inference"] = random.uniform(600,2000, dtype=int)
    if random.uniform(0, 1) < mutation_probability:
         hyperparameters["mean_pixel"] = np.array([random.uniform(115.0,130.0), 
                                            random.uniform(110.0,125.0),
                                            random.uniform(95.0,115.0)])             
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["train_rois_per_image = random.uniform(150,500, dtype=int)"]
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["rois_positive_ratio"] = random.uniform(0.2,0.5)
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["max_gt_instances"] = random.uniform(70,400, dtype=int)
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["detection_max_instances"] = random.uniform(70,400, dtype=int)
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["detection_min_confidence"] = random.uniform(0.3,0.9)
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["detection_nms_threshold"] = random.uniform(0.2,0.7)
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["learning_momentum"] = random.uniform(0.75,0.95)
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["weight_decay"] = random.uniform(0.00007, 0.000125)
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["rpn_class_loss"] = random.uniform(1,10)
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["rpn_bbox_loss"] = random.uniform(1,10)
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["mrcnn_class_loss"] = random.uniform(1,10)
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["mrcnn_bbox_loss"] = random.uniform(1,10)
    if random.uniform(0, 1) < mutation_probability:
        hyperparameters["mrcnn_mask_loss"] = random.uniform(1,10)

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


###### CHANGE ########
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("hyperparameters", generate_random_hyperparameters)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.hyperparameters)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
"""
NGEN = 50
MU = 10
LAMBDA = 50
CXPB = 0.5
MUTPB = 0.1"""


NGEN = 2
MU = 2
LAMBDA = 5
CXPB = 0.5
MUTPB = 0.1

pop = toolbox.population(n=MU)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop, log = deap.algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats=stats, halloffame=hof, verbose=True)

best_hyperparameters = hof[0]

best_inds = [ind for ind in hof.items]
best_inds_df = pd.DataFrame(best_inds)
best_inds_df.to_csv(r"", index=False)
print("Best hyperparameters: ", best_hyperparameters)