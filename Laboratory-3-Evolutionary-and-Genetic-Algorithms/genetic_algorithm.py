import random
from numpy import random as numpy_random, nan
from pandas import isna
from typing import Tuple


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

    def decode_binary_to_decimal(
        self, value_range: ValueRange, number_of_bits: int, individual: BinaryIndividual
    ) -> int:

        if len(individual) != number_of_bits:
            raise ValueError(
                f"Individual length ({len(individual)}) does not match the number of bits ({number_of_bits})"
            )
        if len(value_range) != 2:
            raise ValueError("Value range must contain two values")

        lower_bound, upper_bound = value_range

        if lower_bound >= upper_bound:
            raise ValueError("Lower bound must be smaller than upper bound")

        decoded_value = nan
        largest_value = 2**number_of_bits - 1

        binary_characters = "".join(str(bit) for bit in individual)
        integer_value = int(binary_characters, 2)

        decoded_value = lower_bound + (integer_value / largest_value) * (
            upper_bound - lower_bound
        )

        if decoded_value < lower_bound or decoded_value > upper_bound:
            raise ValueError(
                f"Decoded value ({decoded_value}) is out of the value range ({value_range})"
            )

        if isna(decoded_value):
            raise ValueError("Decoded value is not a number")

        return decoded_value

    def initialize_population(self) -> ...:
        # TODO Initialize the population and return the result
        ...

    def evaluate_population(self, population) -> ...:
        # TODO Evaluate the fitness of the population and return the values for each individual in the population
        ...

    def selection(self, population, fitness_values) -> ...:
        # TODO Implement selection mechanism and return the selected individuals
        pass

    def crossover(self, parents) -> ...:
        # TODO Implement the crossover mechanism over the parents and return the offspring
        ...

    def mutate(self, individuals) -> ...:
        # TODO Implement mutation mechanism over the given individuals and return the results
        ...

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
