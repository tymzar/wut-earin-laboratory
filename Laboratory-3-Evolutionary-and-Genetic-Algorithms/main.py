from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # TODO Experiment 1...
    ga = GeneticAlgorithm(
        population_size=1000,
        mutation_rate=0.2,
        mutation_strength=10,
        crossover_rate=0.3,
        num_generations=30,
        tournament_size=30
    )

    population = ga.initialize_population()
    fitness = ga.evaluate_population(population=population)

    # print(str(population))
    print(str(ga.selection(population, fitness)))
    best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed=1)

    figure, axis = plt.subplots(2, 2)

    axis[0, 0].plot(range(0, ga.num_generations), list(map(lambda sol: sol[0], best_solutions)))
    axis[0, 0].set_title("Best solutions X")
    axis[0, 0].set_xlabel("Generation")

    axis[0, 1].plot(range(0, ga.num_generations), list(map(lambda sol: sol[1], best_solutions)))
    axis[0, 1].set_title("Best solutions Y")
    axis[0, 1].set_xlabel("Generation")

    axis[1, 0].plot(range(0, ga.num_generations), best_fitness_values)
    axis[1, 0].set_title("Best fitness value")
    axis[1, 0].set_yscale("log")
    axis[1, 0].set_xlabel("Generation")

    axis[1, 1].plot(range(0, ga.num_generations), average_fitness_values)
    axis[1, 1].set_title("Average fitness value")
    axis[1, 1].set_yscale("log")
    axis[1, 1].set_xlabel("Generation")
    plt.show()

    # print(f'Best solutions: ${best_solutions}. Best fitness: ${best_fitness_values} average fitness_values: ${average_fitness_values}')