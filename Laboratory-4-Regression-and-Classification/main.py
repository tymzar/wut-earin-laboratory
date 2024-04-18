import random

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from datasets import load_dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def preprocess_dataset(X: pd.DataFrame):
    X = X.drop(columns=["AveOccup", "Population"])
    return X


def perform_pca_analysis(dataset_to_analyze):
    var_ratio = []

    std_scaler = StandardScaler()

    dataset_to_analyze = std_scaler.fit_transform(dataset_to_analyze)

    number_of_features = dataset_to_analyze.shape[1]
    components = range(1, number_of_features + 1)

    for num in components:
        pca = PCA(n_components=num)
        pca.fit(dataset_to_analyze)
        var_ratio.append(np.sum(pca.explained_variance_ratio_))

    plt.figure(figsize=(4, 2), dpi=320)
    plt.grid()
    plt.plot(components, var_ratio, marker="o")
    plt.xlabel("n_components")
    plt.ylabel("Explained variance ratio")
    plt.title("n_components vs. Explained Variance Ratio")
    plt.show()


def pca_transform(train_dataset, test_dataset, number_of_components):
    pca = PCA(n_components=number_of_components)

    return pca.fit_transform(train_dataset), pca.transform(test_dataset)


def process_model(
    model: LinearRegression | RandomForestRegressor, X_train, y_train, X_test, y_test
):
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=4, scoring="neg_mean_squared_error"
    )

    print(
        f"Model {model.__class__.__name__} CV scores: {cv_scores}, mean: {np.mean(cv_scores)}, std: {np.std(cv_scores)}"
    )

    final_model = model.fit(X_train, y_train)

    predictions = final_model.predict(X_test)

    print(
        f"Model {model.__class__.__name__} MSE: {np.mean((predictions - y_test) ** 2)}, R2: {r2_score(y_test, predictions)}\n"
    )

    return cv_scores, predictions


def main(
    should_test_parameters: bool = False,
    pca_analysis: bool = False,
    train_final_model: bool = False,
):
    # Set seed for reproducibility
    seed = 0
    set_seed(seed)

    X, y = load_dataset("california_housing")

    print("info x: ", X.info())
    print("desc x: ", X.describe())

    std_scaler = StandardScaler()

    preprocessed_X = preprocess_dataset(X)
    X_preprocessed_train, X_preprocessed_test, y_train, y_test = train_test_split(
        preprocessed_X, y, test_size=0.2, random_state=seed
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # START - Dataset with 2 removed features
    X_preprocessed_train = std_scaler.fit_transform(X_preprocessed_train)
    X_preprocessed_test = std_scaler.transform(X_preprocessed_test)
    # END - Dataset with 2 removed features

    # START - PCA analysis
    if pca_analysis:
        perform_pca_analysis(X_train)
    X_pca_train, X_pca_test = pca_transform(X_train, X_test, 7)
    # END - PCA analysis

    # START - Dataset with all features
    X_full_train = std_scaler.fit_transform(X_train)
    X_full_test = std_scaler.transform(X_test)
    # END - Dataset with all features

    if should_test_parameters:
        models = [
            {
                "model": LinearRegression(copy_X=True),
                "parameters": [
                    {"fit_intercept": [True]},
                ],
            },
            {
                "model": RandomForestRegressor(),
                "parameters": [
                    {"n_estimators": [500, 100, 10]},
                    {"max_depth": [1000, 100, 10]},
                    {"min_samples_split": [100, 10, 2]},
                ],
            },
        ]

        datasets = [
            (X_full_train, y_train, X_full_test, y_test, "Full dataset"),
            (X_pca_train, y_train, X_pca_test, y_test, "PCA dataset"),
            (
                X_preprocessed_train,
                y_train,
                X_preprocessed_test,
                y_test,
                "Preprocessed dataset",
            ),
        ]

        for dataset in datasets:
            X_train, y_train, X_test, y_test, name = dataset
            print(f"Dataset: {name}")

            scores = []
            predictions = []

            for model_entry in models:

                model = model_entry["model"]

                for parameter_set in model_entry["parameters"]:

                    for parameter, values in parameter_set.items():

                        for value in values:

                            print(f"Parameter: {parameter}, value: {value}")
                            model.set_params(**{parameter: value})

                            scores, prediction = process_model(
                                model, X_train, y_train, X_test, y_test
                            )

                            scores.append(scores)
                            predictions.append(prediction)

                print(
                    f"Finished model: {model.__class__.__name__} with parameter: {parameter}"
                )

    if train_final_model:
        final_liner_regression = LinearRegression()
        final_random_forest = RandomForestRegressor(
            criterion="absolute_error",
            max_depth=100,
            min_samples_split=2,
            n_estimators=500,
            # Empirical good default values are max_features=n_features for regression problems
            max_features=X_preprocessed_train.shape[1],
        )

        linear_regression_scores, linear_regression_predictions = process_model(
            final_liner_regression,
            X_preprocessed_train,
            y_train,
            X_preprocessed_test,
            y_test,
        )

        random_forest_scores, random_forest_predictions = process_model(
            final_random_forest,
            X_preprocessed_train,
            y_train,
            X_preprocessed_test,
            y_test,
        )

        plt.figure(figsize=(4, 2), dpi=320)
        plt.grid()
        plt.scatter(range(0, len(y_test)), y_test, label="True")
        plt.scatter(
            range(0, len(linear_regression_predictions)),
            linear_regression_predictions,
            label="LinearRegression",
        )
        plt.scatter(
            range(0, len(random_forest_predictions)),
            random_forest_predictions,
            label="RandomForest",
        )
        plt.xlabel("Sample")
        plt.ylabel("Target")
        plt.title("Predictions vs. True values")
        plt.legend()
        plt.show()

        plt.figure(figsize=(4, 2), dpi=320)
        plt.grid()
        plt.plot(np.abs(linear_regression_scores), marker="o", label="LinearRegression")
        plt.plot(np.abs(random_forest_scores), marker="o", label="RandomForest")
        plt.xlabel("Fold")
        plt.ylabel("MSE")
        plt.title("Cross-validation results")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-parameters", action=argparse.BooleanOptionalAction)
    parser.add_argument("--pca-analysis", action=argparse.BooleanOptionalAction)
    parser.add_argument("--train_final_model", action=argparse.BooleanOptionalAction)

    arguments = parser.parse_args()

    main(arguments.test_parameters, arguments.pca_analysis, arguments.train_final_model)
