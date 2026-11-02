# AUTHOR: Gino Leone

"""
This file implements a classical genetic algorithm that optimizes variable solutions for
finding the roots of continuous 5th degree polynomials to 2 decimal precision in the form of
a_1x_1^5 + | - a_2x_2^4.... + | - ... a_nx_n^1 + | - c = 0 where a_n, x_n and c are all real numbers.
The algorithm will use a population of potential candidate solutions and will utilize selection,
and mutation to optimize the population towards the set of integer solutions that
satisfy the equation. Traditional crossover will not be used in this implementation

"""

import numpy as np
import matplotlib.pyplot as plt
import random

POPULATION_SIZE = 100
NUM_GENERATIONS = 1000
#CROSSOVER_RATE = 0.8 traditional gene crossover will not be used in this implementation
MUTATION_RATE = 0.1

# Define the polynomial parameters
COEFFICIENTS = np.array([1, 0, -5, 1, 4, 100]) # Coefficients for x in the equation
b = 0 # Value for the right hand side of the equation.


def initialize_population(population_size, lower_limit, upper_limit, precision):
    """
    Initialize the population where a single gene represents a candidate variable solution to 4 decimal places
    and a chromosome represents the set of variables in the equation.
    :param population_size: the number of candidate solutions for x
    :param lower_limit: the lower bound of the search space for x
    :param upper_limit: the upper bound of the search space for x
    :param precision: the decimal precision of x
    :return: an array of candidate solutions for x as float values within 2 decimal precision
    """
    rng = np.random.default_rng()
    random_floats = rng.uniform(lower_limit, upper_limit, size=population_size)
    candidate_solutions = np.round(random_floats, precision)
    return candidate_solutions


def get_fitness(individuals, coefficients) -> np.ndarray:
    """
    Get all the fitness values for each individual x. The fitness function will be defined
    as the absolute difference between the left and right hand side of the equation
    :param canidate_solutions:
    :param coefficients:
    :return: a 1D array representing the fitness values for each individual candidate solution x
    """
    candidate_solutions = np.polyval(coefficients, individuals)
    fitness_values = abs(candidate_solutions - b)
    return fitness_values


def roulette_wheel_selection(population, fitness_values) -> np.ndarray:
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
    total_fitness = np.sum(fitness_values)
    pop_size = len(population)

    # Get the probability of selecting each individual
    # probabilities = fitness / total_fitness
    probabilities = (1 - (fitness_values / total_fitness)) / (pop_size - 1)

    # Select individuals based on their probabilities
    selected_indices = np.random.choice(np.arange(len(population)), size=len(population), p=probabilities)
    return population[selected_indices]

def creep_mutation(individuals):
    """
    Adds or subtracts a small two decimal value to the individual x candidate
    """
    for i in range(len(individuals)):
        if random.random() < MUTATION_RATE:
            #generate a random two decimal value between 0 and .20
            random_val = round(random.random() * .20, 2)
            if random.random() < 0.5:
                individuals[i] += random_val
            else:
                individuals[i] -= random_val

    return individuals

def run_genetic_algorithm():
    """
    Run the genetic algorithm for a specified number of generations.
    """

    # Track the best fitness value in each generation
    best_fitness_history = []
    best_chromosomes = []

    # Initialize the population
    population = initialize_population(POPULATION_SIZE, -5.00, 5.00, 2)
    for _ in range(NUM_GENERATIONS):
        # Calculate fitness of each individual in the population
        fitness = get_fitness(population, COEFFICIENTS)
        best_fitness_history.append(np.min(fitness))
        best_chromosomes.append(population[np.argmin(fitness)])
        selected_individuals = roulette_wheel_selection(population, fitness)

        population = creep_mutation(selected_individuals)

    print("Fitness History:", [round(float(x), 2) for x in best_fitness_history])
    print("Chromosome", [round(float(x), 2) for x in best_chromosomes])




def main():
    """
    individuals = initialize_population(100, -7.50, 7.50, 2)
    print(individuals[0])
    fitness_values = get_fitness(individuals, COEFFICIENTS)
    pop = roulette_wheel_selection(individuals, fitness_values)
    print(pop)
    print("pop after mutation")
    pop_after_mutation = creep_mutation(pop)
    print(pop_after_mutation)
    """
    run_genetic_algorithm()


if __name__ == "__main__":
    main()



