import random

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def preprocess_dataset(X: pd.DataFrame):
    X = X.drop(columns=["AveOccup","Population"])
    return X


if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 0
    set_seed(seed)

    # TODO Preprocess dataset
    X, y = load_dataset("california_housing")
    X = preprocess_dataset(X)

    ...

    # Split data into train and test partitions with 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # TODO Define the models
    model1 = linear_model.LinearRegression()
    # model2 = SVR(degree=6, kernel="poly")
    model2 = RandomForestRegressor()

    # TODO evaluate model using cross-validation
    scores1 = cross_val_score(model1, X_train, y_train, cv=4, scoring='neg_mean_squared_error')
    scores2 = cross_val_score(model2, X_train, y_train, cv=4, scoring='neg_mean_squared_error')

    # Fit the best model on the entire training set and get the predictions
    final_model1 = model1.fit(X_train, y_train)
    final_model2 = model2.fit(X_train, y_train)

    predictions1 = final_model1.predict(X_test)
    predictions2 = final_model2.predict(X_test)

    print(scores1)
    print(scores2)

    plt.scatter(range(0, len(y_test)), y_test)
    plt.scatter(range(0, len(predictions1)), predictions1)
    plt.scatter(range(0, len(predictions2)), predictions2)
    plt.show()


    # TODO Evaluate the final predictions with the metric of your choice
    ...
