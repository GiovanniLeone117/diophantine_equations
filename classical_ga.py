#AUTHOR: GINO LEONE

"""
This file implements a classical genetic algorithm for solving linear diophantine equations
of the form a_1x_1 + a_2x_2 + ... a_nx_n = b where a_i and b are integers and x_i are
non-negative integers. The algorithm will use a population of potential candidate solutions
and will utilize selection, crossover and mutation to optimize the population towards the
set of integer solutions that satisfy the equation.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# Define the hyperparameters
POPULATION_SIZE = 100
NUM_GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

# Define the diophantine equation parameters
CHROMOSOME_LENGTH = 5 # Number of variables in the equation
COEFFICIENTS = np.array([1, 2, 3, 4, 5]) # Coefficients for each variable in the equation
b = 35 # Value for the right hand side of the equation


def initialize_population(population_size, chromosome_length, upper_limit) -> np.ndarray:
    """
    Initializes a population of individuals for the genetic algorithm.
    Each individual is represented as a single chromosome, which is a vector of
    non-negative integers. The population is randomly initialized with values between
    0 and a specified upper limit for each variable in the equation.
    :param population_size:
    :param chromosome_length: The number of variables in the equation
    :param upper_limit:
    :return: np.ndarray: A 2D array representing the initialized population of individuals.
    """
    return np.random.randint(0, upper_limit + 1, size=(population_size, chromosome_length))


def get_fitness(individuals, coefficients, b) -> np.ndarray:
    """
    Calculates the fitness of each individual in the population. The fitness
    is defined as the absolute difference between the left hand and right
    hand side of the equation.
    :param individuals:
    :param coefficients:
    :param b: The real solution to the equation
    :return: np.ndarray: A 1D array representing the fitness of each individual in the population.
    """
    # Calculate the candidate solution of the equation for each individual
    candidate_solutions = np.dot(individuals, coefficients)

    # Get the fitness of each individual
    # (absolute difference between candidate solution and the real solution)
    fitness = np.abs(candidate_solutions - b)
    return fitness


def roulette_wheel_selection(population, fitness) -> np.ndarray:
    """
    Selects individuals from the population as parents for the next generations
    using roulette wheel selection. The probability of selecting an individual
    is directly proportional to its fitness. Individuals with lower fitness
    values (closer to the real solution) have a higher chance of being selected.
    :param population:
    :param fitness:
    :return:
    """
    # Calculate the total fitness value of the entire population
    total_fitness = np.sum(fitness)

    # Get the probability of selecting each individual
    probabilities = fitness / total_fitness

    # Select individuals based on their probabilities
    selected_indices = np.random.choice(np.arange(len(population)), size=len(population), p=probabilities)
    return population[selected_indices]


def one_point_crossover(parent1, parent2) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly selects a single crossover point in two parents and creates
    an offspring by combining the genes from both parents.
    :param parent1:
    :param parent2:
    :return: a tuple of 1D arrays representing the 2 new offsprings by combining the genes from both parents
    """
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring_1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    offspring_2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return offspring_1, offspring_2


def scramble_mutation(individuals) -> list:
    """
    Randomly selects a subset of genes (in this case, a subset of the
    variables in the equation) and shuffles their values to create a new individual.
    :param individuals:
    :param mutation_rate:
    :return: a list of potentially mutated individuals
    """
    for individual in individuals:
        if random.random() < MUTATION_RATE:
            # Randomly select a subset of genes to shuffle
            num_genes_to_shuffle = random.randint(1, len(individual))
            genes_to_shuffle = random.sample(range(len(individual)), num_genes_to_shuffle)

            # Shuffle the selected genes
            shuffled_genes = np.random.permutation(individual[genes_to_shuffle])
            individual[genes_to_shuffle] = shuffled_genes

    return individuals


def run_genetic_algorithm():
    """
    Runs the genetic algorithm to find integer solution to the specified linear
    diophantine equation
    """
    # Track the best fitness value in each generation
    best_fitness_history = []
    best_chromosomes = []

    # Initialize the population
    population = initialize_population(POPULATION_SIZE, CHROMOSOME_LENGTH, 10)
    for _ in range(NUM_GENERATIONS):

        # Calculate fitness of each individual in the population
        fitness = get_fitness(population, COEFFICIENTS, b)
        best_fitness_history.append(np.min(fitness))
        best_chromosomes.append(population[np.argmin(fitness)])

        parents = roulette_wheel_selection(population, fitness)
        offspring = []

        # Shuffle the parents to create random pairs for crossover
        np.random.shuffle(parents)
        for i in range(0, len(parents), 2):

            # Apply crossover probability
            if np.random.rand() < CROSSOVER_RATE:
                candidate_1, candidate_2 = one_point_crossover(parents[i], parents[i+1])
            else:
                candidate_1, candidate_2 = parents[i], parents[i+1]

            offspring.extend([candidate_1, candidate_2])

            # Apply potential mutation to the offspring
            mutated_offspring = scramble_mutation(offspring)

            population = np.array(mutated_offspring)

    clean_history = [int(f) for f in best_fitness_history]
    print(clean_history)
    for chromosome in best_chromosomes:
        print(chromosome)

def main():

    run_genetic_algorithm()


if __name__ == "__main__":
    main()
