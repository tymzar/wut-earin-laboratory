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
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.tournament_size = 5

    def initialize_population(self):
        return [(random.uniform(*init_ranges[rosenbrock_2d][0]), random.uniform(*init_ranges[rosenbrock_2d][1])) for _ in range(self.population_size)]

    def evaluate_population(self, population) -> ...:
        return [(rosenbrock_2d(*individual)) for individual in population]

    def selection(self, population, fitness_values) -> ...:
        data = list(zip(population, fitness_values))
        return [min(random.sample(data, self.tournament_size), key=lambda sample: sample[1])[0] for _ in range(self.population_size)]

    def crossover(self, parents) -> ...:
        # TODO Implement the crossover mechanism over the parents and return the offspring
        return parents

    def mutate(self, individuals) -> ...:
        # TODO Implement mutation mechanism over the given individuals and return the results
        return individuals

    def evolve(self, seed: int) -> ...:
        # Run the genetic algorithm and return the lists that contain the best solution for each generation,
        #   the best fitness for each generation and average fitness for each generation
        set_seed(seed)

        population = self.initialize_population()

        best_solutions = []
        best_fitness_values = []
        average_fitness_values = []

        for generation in range(self.num_generations):
            fitness_values = self.evaluate_population(population)
            parents_for_reproduction = self.selection(population, fitness_values)
            offspring = self.crossover(parents_for_reproduction)
            population = self.mutate(offspring)

            # TODO compute fitness of the new generation and save the best solution, best fitness and average fitness

        return best_solutions, best_fitness_values, average_fitness_values
