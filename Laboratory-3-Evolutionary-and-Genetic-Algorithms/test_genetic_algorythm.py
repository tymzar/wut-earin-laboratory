import unittest
from genetic_algorithm import GeneticAlgorithm


class TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        self.ga = GeneticAlgorithm(
            population_size=10,
            mutation_rate=0.1,
            mutation_strength=0.5,
            crossover_rate=0.8,
            num_generations=100,
            tournament_size=3,
        )

    def test_initialize_population(self):
        population = self.ga.initialize_population()
        self.assertEqual(len(population), self.ga.population_size)
        for individual in population:
            self.assertIsInstance(individual, tuple)
            self.assertEqual(len(individual), 2)

    def test_evaluate_population(self):
        population = self.ga.initialize_population()
        fitness_values = self.ga.evaluate_population(population)
        self.assertEqual(len(fitness_values), len(population))
        for fitness in fitness_values:
            self.assertIsInstance(fitness, float)

    def test_selection(self):
        population = self.ga.initialize_population()
        fitness_values = self.ga.evaluate_population(population)
        selected_population = self.ga.selection(population, fitness_values)
        self.assertEqual(len(selected_population), len(population))

    def test_crossover(self):
        population = self.ga.initialize_population()
        fitness_values = self.ga.evaluate_population(population)
        selected_population = self.ga.selection(population, fitness_values)
        offspring = self.ga.crossover(selected_population)
        self.assertEqual(len(offspring), len(population))

    def test_mutate(self):
        population = self.ga.initialize_population()
        mutated_population = self.ga.mutate(population)
        self.assertEqual(len(mutated_population), len(population))


if __name__ == "__main__":
    unittest.main()
