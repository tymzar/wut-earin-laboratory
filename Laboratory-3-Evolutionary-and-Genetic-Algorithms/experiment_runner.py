from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import pandas as pd
from expermient_data import experiment_parameters1


def compute_and_prepeare_plots(
    ga: GeneticAlgorithm, index: str, seed=1, legend="", should_plot=True
):

    best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed=seed)

    if not should_plot:
        return [
            ga.population_size,
            ga.mutation_rate,
            ga.mutation_strength,
            ga.crossover_rate,
            ga.num_generations,
            ga.tournament_size,
            seed,
            best_fitness_values[-1],
            average_fitness_values[-1],
            best_solutions[-1][0],
            best_solutions[-1][1],
        ]

    figure = plt.figure(index, figsize=(12, 10))

    if len(figure.get_axes()) == 0:
        figure.subplots(2, 2)
    axis = figure.get_axes()
    figure.canvas.manager.set_window_title(f"Task {index}")

    axis[0].plot(
        range(0, ga.num_generations),
        list(map(lambda sol: sol[0], best_solutions)),
        label=legend,
    )
    axis[0].set_title("Best solutions X")

    axis[1].plot(
        range(0, ga.num_generations),
        list(map(lambda sol: sol[1], best_solutions)),
        label=legend,
    )
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

    return [
        ga.population_size,
        ga.mutation_rate,
        ga.mutation_strength,
        ga.crossover_rate,
        ga.num_generations,
        ga.tournament_size,
        seed,
        best_fitness_values[-1],
        average_fitness_values[-1],
        best_solutions[-1][0],
        best_solutions[-1][1],
    ]


def run_experiment(
    experiment_id,
    population_size,
    mutation_rate,
    mutation_strength,
    crossover_rate,
    num_generations,
    tournament_size,
):
    ga = GeneticAlgorithm(
        population_size=population_size,
        mutation_rate=mutation_rate,
        mutation_strength=mutation_strength,
        crossover_rate=crossover_rate,
        num_generations=num_generations,
        tournament_size=tournament_size,
    )
    return compute_and_prepeare_plots(ga, index=experiment_id)


def experiment_1():

    # Run each experiment and collect the results
    results = [
        run_experiment(experiment_id="1", **params) for params in experiment_parameters1
    ]

    # Create a DataFrame from the results
    table = pd.DataFrame(
        results,
        columns=[
            "Population size",
            "Mutation rate",
            "Mutation strength",
            "Crossover rate",
            "Number of generations",
            "Tournament size",
            "Seed",
            "Best fitness value",
            "Average fitness value",
            "Best solution - X",
            "Best solution - Y",
        ],
    )

    print("Task 1")
    print(table)


def experiment_2_1():
    experiments = [
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=21,
            tournament_size=30,
        )
        for _ in range(5)
    ]

    results = [
        compute_and_prepeare_plots(
            experiments[i],
            index="2.1",
            seed=pow(17 * i, 2) + i + 1,
            legend=f"Random seed = {pow(17*i, 2) + i + 1}",
        )
        for i in range(len(experiments))
    ]

    table = pd.DataFrame(
        results,
        index=range(len(results)),
        columns=[
            "Population size",
            "Mutation rate",
            "Mutation strength",
            "Crossover rate",
            "Number of generations",
            "Tournament size",
            "Seed",
            "Best fitness value",
            "Average fitness value",
            "Best solution - X",
            "Best solution - Y",
        ],
    )

    def get_best_solution(table_data):
        best_solution = table_data.loc[table_data["Best fitness value"].idxmin()]
        return best_solution

    def get_average_of_best_fitness(table_data):
        return table_data["Best fitness value"].mean()

    best_solution = get_best_solution(table)
    best_average_fitness = get_average_of_best_fitness(table)

    def compute_standard_deviation(table_data):
        return table_data["Best fitness value"].std()

    print("Task 2.1")
    print(table)
    print(f"Best solution:\n{best_solution}")
    print(f"Best average fitness:\n{best_average_fitness}")
    print(
        f"Standard deviation of best fitness values: {compute_standard_deviation(table)}"
    )


def experiment_2_2():
    experiments = [
        GeneticAlgorithm(
            population_size=size,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=0.3,
            num_generations=21,
            tournament_size=30,
        )
        for size in [1000, 500, 250, 100]
    ]

    results = [
        compute_and_prepeare_plots(
            experiments[i],
            index="2.2",
            legend=f"Experiment {i}",
        )
        for i in range(len(experiments))
    ]

    table = pd.DataFrame(
        results,
        index=range(len(results)),
        columns=[
            "Population size",
            "Mutation rate",
            "Mutation strength",
            "Crossover rate",
            "Number of generations",
            "Tournament size",
            "Seed",
            "Best fitness value",
            "Average fitness value",
            "Best solution - X",
            "Best solution - Y",
        ],
    )

    print("Task 2.2")
    print(table)


def experiment_3():
    experiments = [
        GeneticAlgorithm(
            population_size=1000,
            mutation_rate=0.1,
            mutation_strength=5,
            crossover_rate=crossover_rate,
            num_generations=21,
            tournament_size=30,
        )
        for crossover_rate in [0.2, 0.3, 0.4, 0.6, 0.9]
    ]

    amount_of_seeds = 5

    seed_results = [
        [
            compute_and_prepeare_plots(
                experiments[i],
                index="3." + str(index),
                seed=pow(17 * index, 2) + index + 1,
                legend=f"Experiment {i}",
            )
            for i in range(len(experiments))
        ]
        for index in range(amount_of_seeds)
    ]

    print("Length of seed results: ", len(seed_results))

    tables = [
        pd.DataFrame(
            seed_results[index],
            index=range(len(seed_results[index])),
            columns=[
                "Population size",
                "Mutation rate",
                "Mutation strength",
                "Crossover rate",
                "Number of generations",
                "Tournament size",
                "Seed",
                "Best fitness value",
                "Average fitness value",
                "Best solution - X",
                "Best solution - Y",
            ],
        )
        for index in range(len(seed_results))
    ]

    average_table = pd.concat(tables).groupby(level=0).mean()

    print("Task 3")
    print(average_table)


def experiment_4_1():
    mutation_rate_increase = [0.5, 0.75, 1, 1.50, 1, 2, 2.5, 3, 5]

    experiments = [
        GeneticAlgorithm(
            population_size=1000,
            num_generations=30,
            crossover_rate=0.3,
            tournament_size=30,
            mutation_rate=0.3 * mutation_rate_increase[index],
            mutation_strength=2,
        )
        for index in range(len(mutation_rate_increase))
    ]

    amount_of_seeds = 5

    seeds = [pow(17 * i, 2) + i + 1 for i in range(amount_of_seeds)]

    seeds_results = [
        [
            compute_and_prepeare_plots(
                experiments[i],
                seed=seeds[index],
                index="4.1." + str(index),
                legend=f"Experiment {i}",
            )
            for i in range(len(experiments))
        ]
        for index in range(amount_of_seeds)
    ]

    tables = [
        pd.DataFrame(
            seeds_results[index],
            index=range(len(seeds_results[index])),
            columns=[
                "Population size",
                "Mutation rate",
                "Mutation strength",
                "Crossover rate",
                "Number of generations",
                "Tournament size",
                "Seed",
                "Best fitness value",
                "Average fitness value",
                "Best solution - X",
                "Best solution - Y",
            ],
        )
        for index in range(len(seeds_results))
    ]

    average_table = pd.concat(tables).groupby(level=0).mean()

    print("Task 4.1")
    print(average_table)


def experiment_4_2():
    mutation_strength_increase = [0.5, 0.75, 1, 1.50, 2, 5, 8, 13]

    experiments = [
        GeneticAlgorithm(
            tournament_size=30,
            population_size=1000,
            num_generations=30,
            crossover_rate=0.3,
            mutation_rate=0.3,
            mutation_strength=2 * mutation_strength_increase[index],
        )
        for index in range(len(mutation_strength_increase))
    ]

    number_of_seeds = 5

    seeds_results = [
        [
            compute_and_prepeare_plots(
                experiments[i],
                seed=pow(17 * index, 2) + index + 1,
                index="4.2." + str(index),
                legend=f"Experiment {i}",
            )
            for i in range(len(experiments))
        ]
        for index in range(number_of_seeds)
    ]

    tables = [
        pd.DataFrame(
            seeds_results[index],
            index=range(len(seeds_results[index])),
            columns=[
                "Population size",
                "Mutation rate",
                "Mutation strength",
                "Crossover rate",
                "Number of generations",
                "Tournament size",
                "Seed",
                "Best fitness value",
                "Average fitness value",
                "Best solution - X",
                "Best solution - Y",
            ],
        )
        for index in range(len(seeds_results))
    ]

    average_table = pd.concat(tables).groupby(level=0).mean()

    print("Task 4.2")
    print(average_table)


def runner():

    experiment_1()

    experiment_2_1()
    experiment_2_2()

    experiment_3()

    experiment_4_1()
    experiment_4_2()

    plt.show()


if __name__ == "__main__":

    runner()
