import random
import numpy as np
import sys 
sys.path.append("cluster/home/helensem/Master/chipsogdip/kode")
from ga import generate_hyperparameters, calculate_fitness, plot_hyperparameters
from dataset import get_json_dict
import os
from matplotlib import pyplot as plt
from detectron2.data import MetadataCatalog, DatasetCatalog

# def generate_hyperparameters():
#     init_values = {}
#     init_values["param1"] = np.linspace(0, 1)
#     init_values["param2"] = np.linspace(0, 10)
#     init_values["param3"] = np.linspace(1, 100, dtype=int)
#     return init_values

# def evaluate_fitness(hyperparameters):
#     # Placeholder function for evaluating fitness
#     # You should replace this with your own evaluation function
#     fitness = 0
#     for param in hyperparameters.values():
#         fitness += param
#     return fitness

def adaptive_mutation(hyperparameters, init_values, generation):
    mutated_hyperparameters = hyperparameters.copy()
    for param, value in mutated_hyperparameters.items():
        # Calculate the mutation range based on the current generation
        mutation_range = 1.0 / (generation + 1)
        if param == "roi_batch_size" or param == "rpn_batch_size": 
            mutated_value = random.choice(init_values[param])
        # Mutate the parameter by a random value within the mutation range
        else: 
            mutated_value = value + random.uniform(-mutation_range, mutation_range)
            # Clip the mutated value within the parameter's defined range
            mutated_value = np.clip(mutated_value, init_values[param].min(), init_values[param].max())
            # Update the mutated parameter value
            mutated_hyperparameters[param] = mutated_value
    return mutated_hyperparameters

def genetic_algorithm(population_size, num_generations, mutation_probability, stop_fitness_score):
    init_values = generate_hyperparameters()

    best_individual = None
    best_fitness = None
    best_per_gen = []

    # Initialize the population
    population = [dict(zip(init_values.keys(), [random.choice(values) 
                                                for values in init_values.values()])) 
                                                for _ in range(population_size)]

    for generation in range(num_generations):
        # Evaluate the fitness of each individual in the population
        fitness_scores = []
        for idx, individual in enumerate(population):
            fitness = calculate_fitness(idx, individual, generation+1)
            fitness_scores.append((individual, fitness))

        # Sort the population based on fitness scores in descending order
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        current_best_individual, current_best_fitness = fitness_scores[0]
        best_per_gen.append((current_best_individual, current_best_fitness))
        if current_best_fitness >= stop_fitness_score:
            best_fitness = current_best_fitness
            best_individual = current_best_individual
            break
        # Select the top individuals for reproduction (elitism)
        elite_population = [individual for individual, _ in 
                            fitness_scores[:int(0.4 * population_size)]]

        # Create the next generation through crossover and mutation
        next_generation = elite_population.copy()

        while len(next_generation) < population_size:
            # Perform crossover by randomly selecting two parents
            parent1, parent2 = random.choices(elite_population, k=2)

            # Create a new child by combining the hyperparameters of the parents
            child = {}
            for param in init_values.keys():
                # Perform uniform crossover by randomly selecting a parent's value
                if random.random() < 0.5:
                    child[param] = parent1[param]
                else:
                    child[param] = parent2[param]

            # Perform mutation on the child
            if random.random() < mutation_probability:
                child = adaptive_mutation(child, init_values, generation)

            # Add the child to the next generation
            next_generation.append(child)

        # Replace the current population with the next generation

        population = next_generation

    # Evaluate the fitness of the final population
    fitness_scores = []
    for individual in population:
        fitness = calculate_fitness(population_size, individual, num_generations)
        fitness_scores.append((individual, fitness))

    # Sort the final population based on fitness scores in descending order
    fitness_scores.sort(key=lambda x: x[1], reverse=True)

    # Return the best individual (hyperparameters) and its fitness score
    best_individual, best_fitness = fitness_scores[0]
    best_per_gen.append([best_individual, best_fitness])
    return best_individual, best_fitness, best_per_gen


if __name__ == "__main__": 
    path = f"/cluster/home/helensem/Master/data/set1"

    for d in ["train", "val"]:
        DatasetCatalog.register("ga_damage_" + d, lambda d=d: get_json_dict(path, d))
        MetadataCatalog.get("ga_damage_" + d).set(thing_classes=["damage"])
    
    population_size = 20 
    generations = 15 
    mutation_rate = 0.1
    stop_fitness_score = 0.64
    hyper = generate_hyperparameters()
    #best_indv, best_fitness, best_per_gen = genetic_algorithm(population_size, generations, mutation_rate, stop_fitness_score)

    #print("Best hyperparameters are: ", best_indv)
    #print("Best IoU reached: ", best_fitness)
    best_per_gen = [({'rpn_nms_threshold': 1.0, 'rpn_batch_size': 512, 'pre_nms_limit': 4734, 'post_nms_rois_training': 1244, 'post_nms_rois_inference': 1914, 'roi_batch_size': 1024, 'roi_positive_ratio': 0.37755102040816324, 'detection_min_confidence': 0.5326530612244897, 'learning_momentum': 0.9173469387755102, 'weight_decay': 0.00010591836734693877, 'epochs': 28, 'learning_rate': 0.0008714285714285715, 'img_min_size': 928, 'img_max_size': 1138, 'roi_iou_threshold': 0.3326530612244898}, 0.591462), ({'rpn_nms_threshold': 0.6428571428571428, 'rpn_batch_size': 256, 'pre_nms_limit': 4244, 'post_nms_rois_training': 2918, 'post_nms_rois_inference': 657, 'roi_batch_size': 128, 'roi_positive_ratio': 0.41428571428571426, 'detection_min_confidence': 0.7040816326530612, 'learning_momentum': 0.95, 'weight_decay': 0.00011714285714285714, 'epochs': 28, 'learning_rate': 0.0002653061224489796, 'img_min_size': 989, 'img_max_size': 975, 'roi_iou_threshold': 0.3571428571428571},0.60091615),
    ({'rpn_nms_threshold': 0.6428571428571428, 'rpn_batch_size': 1024, 'pre_nms_limit': 6857, 'post_nms_rois_training': 2918, 'post_nms_rois_inference': 657, 'roi_batch_size': 1024, 'roi_positive_ratio': 0.41428571428571426, 'detection_min_confidence': 0.7040816326530612, 'learning_momentum': 0.8642857142857143, 'weight_decay': 0.0001216326530612245, 'epochs': 37, 'learning_rate': 0.0007061224489795919, 'img_min_size': 989, 'img_max_size': 975, 'roi_iou_threshold': 0.3571428571428571},
    0.59844404),
    ({'rpn_nms_threshold': 0.6428571428571428, 'rpn_batch_size': 1024, 'pre_nms_limit': 4244, 'post_nms_rois_training': 2918, 'post_nms_rois_inference': 657, 'roi_batch_size': 1024, 'roi_positive_ratio': 0.41428571428571426, 'detection_min_confidence': 0.7040816326530612, 'learning_momentum': 0.95, 'weight_decay': 0.00011714285714285714, 'epochs': 37, 'learning_rate': 0.0002653061224489796, 'img_min_size': 989, 'img_max_size': 975, 'roi_iou_threshold': 0.3571428571428571},
    0.6010027),
    ({'rpn_nms_threshold': 0.6428571428571428, 'rpn_batch_size': 1024, 'pre_nms_limit': 4979, 'post_nms_rois_training': 2918, 'post_nms_rois_inference': 885, 'roi_batch_size': 1024, 'roi_positive_ratio': 0.49387755102040815, 'detection_min_confidence': 0.42244897959183675, 'learning_momentum': 0.9132653061224489, 'weight_decay': 0.0001216326530612245, 'epochs': 37, 'learning_rate': 0.0002653061224489796, 'img_min_size': 989, 'img_max_size': 975, 'roi_iou_threshold': 0.34897959183673466},
    0.60297),
    ({'rpn_nms_threshold': 0.6428571428571428, 'rpn_batch_size': 1024, 'pre_nms_limit': 6857, 'post_nms_rois_training': 2918, 'post_nms_rois_inference': 657, 'roi_batch_size': 128, 'roi_positive_ratio': 0.41428571428571426, 'detection_min_confidence': 0.7653061224489797, 'learning_momentum': 0.8642857142857143, 'weight_decay': 0.0001216326530612245, 'epochs': 29, 'learning_rate': 0.0007061224489795919, 'img_min_size': 989, 'img_max_size': 1148, 'roi_iou_threshold': 0.3571428571428571},
    0.6030673),
    ({'rpn_nms_threshold': 0.6428571428571428, 'rpn_batch_size': 1024, 'pre_nms_limit': 6857, 'post_nms_rois_training': 2224, 'post_nms_rois_inference': 885, 'roi_batch_size': 128, 'roi_positive_ratio': 0.49387755102040815, 'detection_min_confidence': 0.7653061224489797, 'learning_momentum': 0.95, 'weight_decay': 0.0001216326530612245, 'epochs': 29, 'learning_rate': 0.0007061224489795919, 'img_min_size': 989, 'img_max_size': 1148, 'roi_iou_threshold': 0.3571428571428571},
    0.6128894),
    ({'rpn_nms_threshold': 0.6428571428571428, 'rpn_batch_size': 1024, 'pre_nms_limit': 6857, 'post_nms_rois_training': 2224, 'post_nms_rois_inference': 885, 'roi_batch_size': 128, 'roi_positive_ratio': 0.49387755102040815, 'detection_min_confidence': 0.7653061224489797, 'learning_momentum': 0.95, 'weight_decay': 0.0001216326530612245, 'epochs': 29, 'learning_rate': 0.0007061224489795919, 'img_min_size': 989, 'img_max_size': 1148, 'roi_iou_threshold': 0.3571428571428571},
    0.6315419),
    ({'rpn_nms_threshold': 0.6428571428571428, 'rpn_batch_size': 1024, 'pre_nms_limit': 6857, 'post_nms_rois_training': 2918, 'post_nms_rois_inference': 657, 'roi_batch_size': 128, 'roi_positive_ratio': 0.41428571428571426, 'detection_min_confidence': 0.7653061224489797, 'learning_momentum': 0.8642857142857143, 'weight_decay': 0.0001216326530612245, 'epochs': 29, 'learning_rate': 0.0007061224489795919, 'img_min_size': 989, 'img_max_size': 1148, 'roi_iou_threshold': 0.3571428571428571},
    0.5997933),
    ({'rpn_nms_threshold': 0.6428571428571428, 'rpn_batch_size': 1024, 'pre_nms_limit': 6857, 'post_nms_rois_training': 2918, 'post_nms_rois_inference': 657, 'roi_batch_size': 128, 'roi_positive_ratio': 0.41428571428571426, 'detection_min_confidence': 0.7653061224489797, 'learning_momentum': 0.8642857142857143, 'weight_decay': 0.0001216326530612245, 'epochs': 29, 'learning_rate': 0.0007061224489795919, 'img_min_size': 989, 'img_max_size': 1148, 'roi_iou_threshold': 0.3571428571428571},
    0.6059205),
    ({'rpn_nms_threshold': 0.6428571428571428, 'rpn_batch_size': 1024, 'pre_nms_limit': 4326, 'post_nms_rois_training': 2224, 'post_nms_rois_inference': 657, 'roi_batch_size': 128, 'roi_positive_ratio': 0.40816326530612246, 'detection_min_confidence': 0.42244897959183675, 'learning_momentum': 0.9132653061224489, 'weight_decay': 0.0001216326530612245, 'epochs': 37, 'learning_rate': 0.0007061224489795919, 'img_min_size': 989, 'img_max_size': 1148, 'roi_iou_threshold': 0.34897959183673466},
    0.6129323),
    ({'rpn_nms_threshold': 0.6530612244897959, 'rpn_batch_size': 1024, 'pre_nms_limit': 6857, 'post_nms_rois_training': 2224, 'post_nms_rois_inference': 657, 'roi_batch_size': 128, 'roi_positive_ratio': 0.40816326530612246, 'detection_min_confidence': 0.7653061224489797, 'learning_momentum': 0.8642857142857143, 'weight_decay': 0.0001216326530612245, 'epochs': 29, 'learning_rate': 0.0007061224489795919, 'img_min_size': 989, 'img_max_size': 1148, 'roi_iou_threshold': 0.3571428571428571},
    0.6022657),
    ({'rpn_nms_threshold': 0.6530612244897959, 'rpn_batch_size': 1024, 'pre_nms_limit': 4326, 'post_nms_rois_training': 2224, 'post_nms_rois_inference': 657, 'roi_batch_size': 128, 'roi_positive_ratio': 0.49387755102040815, 'detection_min_confidence': 0.42244897959183675, 'learning_momentum': 0.8642857142857143, 'weight_decay': 0.0001216326530612245, 'epochs': 29, 'learning_rate': 0.0007061224489795919, 'img_min_size': 989, 'img_max_size': 1148, 'roi_iou_threshold': 0.3571428571428571},
    0.60449046), ({'rpn_nms_threshold': 0.6530612244897959, 'rpn_batch_size': 1024, 'pre_nms_limit': 6857, 'post_nms_rois_training': 2918, 'post_nms_rois_inference': 657, 'roi_batch_size': 128, 'roi_positive_ratio': 0.49387755102040815, 'detection_min_confidence': 0.7653061224489797, 'learning_momentum': 0.9132653061224489, 'weight_decay': 0.0001216326530612245, 'epochs': 29, 'learning_rate': 0.0007061224489795919, 'img_min_size': 989, 'img_max_size': 1148, 'roi_iou_threshold': 0.3571428571428571},
    0.6054971), ({'rpn_nms_threshold': 0.6530612244897959, 'rpn_batch_size': 1024, 'pre_nms_limit': 6857, 'post_nms_rois_training': 2224, 'post_nms_rois_inference': 657, 'roi_batch_size': 128, 'roi_positive_ratio': 0.49387755102040815, 'detection_min_confidence': 0.42244897959183675, 'learning_momentum': 0.8642857142857143, 'weight_decay': 0.0001216326530612245, 'epochs': 29, 'learning_rate': 0.0007061224489795919, 'img_min_size': 989, 'img_max_size': 1148, 'roi_iou_threshold': 0.3571428571428571},
    0.6123835)]

    ious = []
    gens = np.arange(0,generations)
    for gen, indv in enumerate(best_per_gen): 
      #print(indv[0])
      #print(indv[1])
      ious.append(indv[1])

    plt.plot(gens, ious, color = "b", marker = "o")
    plt.xlabel("Generations")
    plt.ylabel("IoU")
    plt.savefig(f"/cluster/work/helensem/Master/output/run_ga_adv/ious")
    for key in hyper.keys():
        plot_hyperparameters(best_per_gen, key)

