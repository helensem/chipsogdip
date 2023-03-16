# import deap 
# from deap import base
# from deap import creator
# from deap import tools
# import deap.algorithms


# import dask.dataframe as dd
import pandas as pd

import numpy as np
import random

import sys 
sys.path.append("cluster/home/helensem/Master/chipsogdip/kode")
from eval import evaluate_model
from training import config
from dataset import load_damage_dicts, get_json_dict
import os

from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog



# def windowed_dataset(series, window_size=G.WINDOW_SIZE, batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE):
#    """
#    We create time windows to create X and y features.
#    For example, if we choose a window of 30, we will create a dataset formed by 30 points as X
#    """
#    dataset = tf.data.Dataset.from_tensor_slices(series)
#    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
#    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
#    dataset = dataset.shuffle(shuffle_buffer)
#    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
#    dataset = dataset.batch(batch_size).prefetch(1)
#    return dataset



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


####* CODE FROM KNAPSACK PROBLEM

def generate_hyperparameters(): 
    init_values = {} 
    init_values["RPN_ANCHOR_STRIDE"] = np.array([1,2,3,4])
    init_values["RPN_NMS_THRESHOLD"] = np.linspace(0.5, 1)
    init_values["RPN_TRAIN_ANCHORS_PER_IMAGE"] = np.array([64, 128, 256, 512, 1024])
    init_values["PRE_NMS_LIMIT"] = np.linspace(4000,8000,dtype=int)
    init_values["POST_NMS_ROIS_TRAINING"] = np.linspace(1000,3000,dtype=int)
    init_values["POST_NMS_ROIS_INFERENCE"] = np.linspace(600,2000,dtype=int)
    #init_values["MEAN_PIXEL"] = np.array([np.linspace(115.0,130.0,), 
                                          #  np.linspace(110.0,125.0),
                                           # np.linspace(95.0,115.0)])
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

def ga_train(indv, generation, learning_rate = 0.00025):
    """ For training with the genetic algorithm, changing the hyperparameters
    """
    print(learning_rate)
    
    cfg = config() 
    cfg.DATASETS.TRAIN = ("ga_damage_train")
    cfg.SOLVER.BASE_LR = learning_rate #0.00025 
    cfg.SOLVER.MAX_ITER = 200*30 
    cfg.SOLVER.STEPS = []
    cfg.OUTPUT_DIR = f"/cluster/work/helensem/Master/output/run_ga/gen_{generation}/{indv}" #! MUST MATCH WITH CURRENT MODEL 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    #TRAIN
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg


def calculate_fitness(indv, hyperparameters, generation):
    
    #dataset = r"/cluster/home/helensem/Master/data/set1"
    learning_rate = float(hyperparameters["learning_rate"])
    cfg = ga_train(indv, generation, learning_rate)

    #TRAIN

    #EVALUATE 
    val_dict = get_jason_dict("val")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7


    corr_iou, bg_iou, mean_iou = evaluate_model(cfg, val_dict) 
    score = 1 - mean_iou 

    return score

def select_chromosomes(population):
	fitness_values = []
	for chromosome in population:
		fitness_values.append(calculate_fitness(chromosome))
	
	fitness_values = [float(i)/sum(fitness_values) for i in fitness_values]
	
	parent1 = random.choices(population, hyperparameters=fitness_values, k=1)[0]
	parent2 = random.choices(population, hyperparameters=fitness_values, k=1)[0]
	
	print("Selected two chromosomes for crossover")
	return parent1, parent2

def crossover(parent1, parent2):
	crossover_point = random.randint(0, len(parent1)-1)
	child1 = parent1[0:crossover_point] + parent2[crossover_point:]
	child2 = parent2[0:crossover_point] + parent1[crossover_point:]
	
	print("Performed crossover between two chromosomes")
	return child1, child2

def mutate(chromosome):
	mutation_point = random.randint(0, len(chromosome)-1)
	if chromosome[mutation_point] == 0:
		chromosome[mutation_point] = 1
	else:
		chromosome[mutation_point] = 0
	print("Performed mutation on a chromosome")
	return chromosome


def get_best(population):
	fitness_values = []
	for chromosome in population:
		fitness_values.append(calculate_fitness(chromosome))

	max_value = max(fitness_values)
	max_index = fitness_values.index(max_value)
	return population[max_index]


mutation_rate = 0.2
generations = 3 
hyperparameters = {}
hyperparameters["learning_rate"]= np.linspace(0.0001, 0.001) #generate_hyperparameters()
population_size = 2

population = [dict(zip(hyperparameters.keys(), [random.choice(values) for values in hyperparameters.values()])) for _ in range(population_size)]

# for _ in range(generations):
# 	# select two chromosomes for crossover
# 	parent1, parent2 = select_chromosomes(population)

# 	# perform crossover to generate two new chromosomes
# 	child1, child2 = crossover(parent1, parent2)

# 	# perform mutation on the two new chromosomes
# 	if random.uniform(0, 1) < mutation_probability:
# 		child1 = mutate(child1)
# 	if random.uniform(0, 1) < mutation_probability:
# 		child2 = mutate(child2)

# 	# replace the old population with the new population
# 	population = [child1, child2] + population[2:]


# run the genetic algorithm


if __name__ == "__main__":
    path = r"/cluster/home/helensem/Master/data/set1"

    for d in ["train", "val"]:
        DatasetCatalog.register("ga_damage_" + d, lambda d=d: get_json_dict(path, d))
        MetadataCatalog.get("ga_damage_" + d).set(thing_classes=["damage"])

    for generation in range(generations):
        # evaluate the fitness of each individual in the population
        fitness_scores = [calculate_fitness(idx, individual, generation) for idx, individual in enumerate(population)]
        
        # select the fittest individuals to breed the next generation
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
        fittest_individuals = sorted_population[:int(population_size/2)]
        
        # create the next generation by breeding the fittest individuals
        new_population = []
        while len(new_population) < population_size:
            # randomly select two parents from the fittest individuals
            parent1, parent2 = random.sample(fittest_individuals, k=2)
            
            # create a new individual by randomly selecting hyperparameters from the parents
            new_individual = {}
            for key in hyperparameters.keys():
                if random.random() < mutation_rate:
                    # randomly mutate the hyperparameter with a small random value
                    new_individual[key] = parent1[key] + random.gauss(0, 0.1)
                else:
                    # randomly select the hyperparameter from one of the parents
                    new_individual[key] = random.choice([parent1[key], parent2[key]])
            new_population.append(new_individual)
        
        # update the population with the new generation
        population = new_population

    # select the fittest individual from the final population
    fitness_scores = [calculate_fitness(individual) for individual in population]
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    fittest_individual = sorted_population[0]
    print("Best individual is: ", fittest_individual)
        





# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)

# toolbox = base.Toolbox()
# toolbox.register("hyperparameters", generate_random_hyperparameters)
# toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.hyperparameters)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("evaluate", evaluate)

# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
# toolbox.register("select", tools.selTournament, tournsize=3)
# """
# NGEN = 50
# MU = 10
# LAMBDA = 50
# CXPB = 0.5
# MUTPB = 0.1"""


# NGEN = 2
# MU = 2
# LAMBDA = 5
# CXPB = 0.5
# MUTPB = 0.1

# print(toolbox.population)

# pop = toolbox.population(n=MU)
# hof = tools.HallOfFame(1)
# stats = tools.Statistics(lambda ind: ind.fitness.values)
# stats.register("avg", np.mean)
# stats.register("std", np.std)
# stats.register("min", np.min)
# stats.register("max", np.max)

# pop, log = deap.algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats=stats, halloffame=hof, verbose=True)

# best_hyperparameters = hof[0]

# best_inds = [ind for ind in hof.items]
# best_inds_df = pd.DataFrame(best_inds)
# best_inds_df.to_csv(r"", index=False)
# print("Best hyperparameters: ", best_hyperparameters)