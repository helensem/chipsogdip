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
    #population_size = 10
    #num_generations = 20
    #mutation_probability = 0.1
    best_individual = None
    best_fitness = None
    best_per_gen = []

    # Initialize the population
    population = [dict(zip(init_values.keys(), [random.choice(values) for values in init_values.values()])) for _ in range(population_size)]

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
        elite_population = [individual for individual, _ in fitness_scores[:int(0.4 * population_size)]]

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
    best_indv, best_fitness, best_per_gen = genetic_algorithm(population_size, generations, mutation_rate, stop_fitness_score)
    print("Best hyperparameters are: ", best_indv)
    print("Best IoU reached: ", best_fitness)
 
    ious = []
    gens = np.arange(0,generations+1)
    for gen, indv in enumerate(best_per_gen): 
      print(indv[0])
      print(indv[1])
      ious.append(indv[1])

    plt.plot(gens, ious[:-1], color = "b", marker = "o")
    plt.xlabel("Generations")
    plt.ylabel("IoU")
    plt.savefig(f"/cluster/work/helensem/Master/output/run_ga_adv/ious")
    for key in hyper.keys():
        plot_hyperparameters(best_per_gen, key)

