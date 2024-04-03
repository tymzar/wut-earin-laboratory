from genetic_algorithm import GeneticAlgorithm

if __name__ == "__main__":
    # TODO Experiment 1...
    ga = GeneticAlgorithm(
        population_size=1000,
        mutation_rate=0,
        mutation_strength=0,
        crossover_rate=0,
        num_generations=1,
    )

    population = ga.initialize_population()
    fitness = ga.evaluate_population(population=population)

    # print(str(population))
    print(str(ga.selection(population, fitness)))
    # best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed=...)
