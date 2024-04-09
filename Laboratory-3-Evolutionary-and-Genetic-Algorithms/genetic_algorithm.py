import random
from numpy import random as numpy_random, nan
from pandas import isna
from typing import Tuple
from functions import init_ranges
from functions import rosenbrock_2d

def set_seed(seed: int) -> None:
    # Set fixed random seed to make the results reproducible
    random.seed(seed)
    numpy_random.seed(seed)


ValueRange = Tuple[int, int]
BinaryIndividual = list[int]


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        mutation_strength: float,
        crossover_rate: float,
        num_generations: int,
        tournament_size: int = 5
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.tournament_size = tournament_size

    def initialize_population(self):
        return [(random.uniform(*init_ranges[rosenbrock_2d][0]), random.uniform(*init_ranges[rosenbrock_2d][1])) for _ in range(self.population_size)]

    def evaluate_population(self, population):
        return [(rosenbrock_2d(*individual)) for individual in population]

    def selection(self, population, fitness_values):
        data = list(zip(population, fitness_values))
        return [min(random.sample(data, self.tournament_size), key=lambda sample: sample[1])[0] for _ in range(self.population_size)]

    def crossover(self, parents):
        offspring = []
        for i in range(0, len(parents), 2):
            first_parent = parents[i]
            second_parent = parents[i+1]
            alpha = self.crossover_rate
            first_child = (alpha * first_parent[0] + (1-alpha) * second_parent[0], alpha * first_parent[1] + (1-alpha) * second_parent[1])
            second_child = (alpha * second_parent[0] + (1-alpha) * first_parent[0], alpha * second_parent[1] + (1-alpha) * first_parent[1])
            offspring.append(first_child)
            offspring.append(second_child)
        return offspring

    def __mutation_value(self):
        return random.choices([0, 1], [1-self.mutation_rate, self.mutation_rate])[0] * random.gauss(0, self.mutation_strength)

    def mutate(self, individuals):
        return [(individual[0] + self.__mutation_value(), individual[1] + self.__mutation_value()) for individual in individuals]

    def evolve(self, seed: int):
        # Run the genetic algorithm and return the lists that contain the best solution for each generation,
        #   the best fitness for each generation and average fitness for each generation
        set_seed(seed)

        population = self.initialize_population()

        best_solutions = []
        best_fitness_values = []
        average_fitness_values = []

        for generation in range(self.num_generations):
            fitness_values = self.evaluate_population(population)

            best_fitness = min(fitness_values)
            best_solution = population[fitness_values.index(best_fitness)]
            average_fitness = sum(fitness_values) / len(fitness_values)
            best_solutions.append(best_solution)
            best_fitness_values.append(best_fitness)
            average_fitness_values.append(average_fitness)

            parents_for_reproduction = self.selection(population, fitness_values)
            offspring = self.crossover(parents_for_reproduction)
            population = self.mutate(offspring)

        return best_solutions, best_fitness_values, average_fitness_values
