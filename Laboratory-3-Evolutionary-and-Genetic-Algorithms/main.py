from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import pandas as pd

def compute_and_prepeare_plots(ga: GeneticAlgorithm, index: int, seed=1, legend=""):
    best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed=seed)

    figure = plt.figure(index, figsize=(12, 10))

    if len(figure.get_axes()) == 0:
        figure.subplots(2, 2)
    axis = figure.get_axes()
    figure.canvas.manager.set_window_title(f"Task {index}")

    axis[0].plot(range(0, ga.num_generations), list(map(lambda sol: sol[0], best_solutions)), label=legend)
    axis[0].set_title("Best solutions X")

    axis[1].plot(range(0, ga.num_generations), list(map(lambda sol: sol[1], best_solutions)), label=legend)
    axis[1].set_title("Best solutions Y")

    axis[2].plot(range(0, ga.num_generations), best_fitness_values, label=legend)
    axis[2].set_title("Best fitness value")
    axis[2].set_yscale("log")

    axis[3].plot(range(0, ga.num_generations), average_fitness_values, label=legend)
    axis[3].set_title("Average fitness value")
    axis[3].set_yscale("log")

    for i in range(0, 4):
        axis[i].set_xlabel("Generation")
        if len(legend) > 0:
            axis[i].legend()

    return [ga.population_size, ga.mutation_rate, ga.mutation_strength,
            ga.crossover_rate, ga.num_generations, ga.tournament_size, seed, best_fitness_values[-1], average_fitness_values[-1]]


if __name__ == "__main__":

    #Task 1 - finding best parameters
    task1_experiments = [
        # Experiment 0
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.2,
            mutation_strength=10,
            crossover_rate=0.3,
            num_generations=30,
            tournament_size=30
        ),
        # Experiment 1
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=10,
            crossover_rate=0.3,
            num_generations=30,
            tournament_size=30
        ),
        # Experiment 2
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.05,
            mutation_strength=10,
            crossover_rate=0.3,
            num_generations=30,
            tournament_size=30
        ),
        # Experiment 3
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=30,
            tournament_size=30
        ),
        # Experiment 4
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=2,
            crossover_rate=0.3,
            num_generations=30,
            tournament_size=30
        ),
        # Experiment 5
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.4,
            num_generations=30,
            tournament_size=30
        ),
        # Experiment 6
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.2,
            num_generations=30,
            tournament_size=30
        ),
        # Experiment 7
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=30,
            tournament_size=20
        ),
        # Experiment 8
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=30,
            tournament_size=40
        ),
        # Experiment 9
        GeneticAlgorithm(
            population_size=2000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=30,
            tournament_size=30
        ),
        # Experiment 10
        GeneticAlgorithm(
            population_size=500,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=30,
            tournament_size=30
        ),
        # Experiment 11
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=21,
            tournament_size=30
        ),
    ]
    
    
    task1_results = [compute_and_prepeare_plots(task1_experiments[i], index=1, legend=f"Experiment {i}") for i in range (0, len(task1_experiments))]
    task1_table = pd.DataFrame(task1_results, index=range(0, len(task1_results)),
                                columns=["Population size", "Mutation rate", "Mutation strength", "Crossover rate",
                                         "Number of generations", "Tournament size", "Seed", "Best fitness value", "Average fitness value"])
    print("Task 1")
    print(task1_table)


    #Task 2.1 - influence of random seed
    task2_experiments = [
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=21,
            tournament_size=30
        ),
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=21,
            tournament_size=30
        ),
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=21,
            tournament_size=30
        ),
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=21,
            tournament_size=30
        ),
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=21,
            tournament_size=30
        ),
    ]

    task2_results = [compute_and_prepeare_plots(task2_experiments[i], index=21, seed=pow(17*i, 2) + i + 1, legend=f"Random seed = {pow(17*i, 2) + i + 1}") for i in range (0, len(task2_experiments))]
    task2_table = pd.DataFrame(task2_results, index=range(0, len(task2_results)),
                               columns=["Population size", "Mutation rate", "Mutation strength", "Crossover rate",
                                        "Number of generations", "Tournament size", "Seed", "Best fitness value", "Average fitness value"])
    print("Task 2.1")
    print(task2_table)

    #TODO: Find best solution across all seeds and its fitness value
    #TODO: Make an average across best fitness values of all seeds and compute standard deviation

    #Task 2.2 - influence of decreasing population size
    task2_2_experiments = [
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=21,
            tournament_size=30
        ),
        GeneticAlgorithm(
            population_size=500,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=21,
            tournament_size=30
        ),
        GeneticAlgorithm(
            population_size=250,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=21,
            tournament_size=30
        ),
        GeneticAlgorithm(
            population_size=100,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=21,
            tournament_size=30
        )
    ]

    task2_2_results = [compute_and_prepeare_plots(task2_2_experiments[i], index=22, legend=f"Experiment {i}") for i in range (0, len(task2_2_experiments))]
    task2_2_table = pd.DataFrame(task2_2_results, index=range(0, len(task2_2_results)),
                               columns=["Population size", "Mutation rate", "Mutation strength", "Crossover rate",
                                        "Number of generations", "Tournament size", "Seed", "Best fitness value", "Average fitness value"])
    print("Task 2.2")
    print(task2_2_table)

    #Task 3 - impact of crossover rate

    task3_experiments = [
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=21,
            tournament_size=30
        ),
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.4,
            num_generations=21,
            tournament_size=30
        ),
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.2,
            num_generations=21,
            tournament_size=30
        )
    ]

    task3_results = [compute_and_prepeare_plots(task3_experiments[i], index=3, legend=f"Experiment {i}") for i in range (0, len(task3_experiments))]
    task3_table = pd.DataFrame(task3_results, index=range(0, len(task3_results)),
                               columns=["Population size", "Mutation rate", "Mutation strength", "Crossover rate",
                                        "Number of generations", "Tournament size", "Seed", "Best fitness value", "Average fitness value"])

    print("Task 3")
    print(task3_table)

    #TODO: Make an average across best fitness values of more than one seed
    plt.show()


