import json
import os
from collections import OrderedDict
from time import sleep

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression


def visualize_decision(X, y, pred, feature=0):
    plt.scatter(X, y)
    plt.scatter(X, pred)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()


def create_custom_datasets(dataset_name="cos"):
    """Cos, poly, sin"""
    X = np.linspace(0, 1, 100)

    if dataset_name == "cos":
        y = np.cos(2 * np.pi * X)
    elif dataset_name == "sin":
        y = np.sin(2 * np.pi * X)
    elif dataset_name == "poly":
        y = 5 * X ** 3 + X ** 2 + 5
    else:
        raise ValueError()
    return X, y


def create_polynomial_features(X, degree=2):
    features = []
    for d in range(1, degree + 1):
        features.append(np.power(X, d))
    return np.stack(features, axis=-1)


class PolyNetwork():

    def __init__(self, n_features):
        self.weights = np.random.uniform(-1, 1, n_features + 1)
        self.n_features = n_features

    def forward(self, X):
        return X @ self.weights

    def calc_weights(self, X, y):
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def optimize(self, X, steps, lr=0.01):
        pass


if __name__ == '__main__':

    # n_elements x 1
    X_origin, y = create_custom_datasets()
    # n_elements x n_features
    X = create_polynomial_features(X_origin, degree=5)
    n_features = X.shape[-1]
    n_elements = X.shape[0]

    # add bias ones term
    X = np.concatenate([X, np.ones(n_elements)[..., None]], axis=-1)

    model = PolyNetwork(n_features=n_features)

    # init
    pred = model.forward(X)
    visualize_decision(X_origin, y, pred)

    # calc weights
    model.calc_weights(X, y)
    pred = model.forward(X)
    visualize_decision(X_origin, y, pred)
