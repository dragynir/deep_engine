import json
import os
from collections import OrderedDict
from time import sleep

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from tqdm import tqdm


def visualize_decision(X, y, pred, name):
    plt.scatter(X, y)
    plt.scatter(X, pred)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(name)
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

    def calc_weights(self, X, y, alpha=0.0):
        x_t = X.T @ X
        x_t += alpha * np.ones(x_t.shape)
        self.weights = np.linalg.inv(x_t) @ X.T @ y

    def optimize(self, X_origin, X, y, steps=4, lr=0.01, alpha=0.0):
        plot_every = steps // 5 if steps > 5 else 1

        for step in tqdm(range(steps)):
            w_grad = 2 * X.T @ (X @ self.weights - y) + 2 * alpha * self.weights
            self.weights -= lr * w_grad

            if step % plot_every == 0:
                pred = self.forward(X)
                visualize_decision(X_origin, y, pred, name='optimize')
        print('Finish')


if __name__ == '__main__':

    # n_elements x 1
    X_origin, y = create_custom_datasets(dataset_name='sin')
    # n_elements x n_features
    X = create_polynomial_features(X_origin, degree=3)
    n_features = X.shape[-1]
    n_elements = X.shape[0]

    # add bias ones term
    X = np.concatenate([X, np.ones(n_elements)[..., None]], axis=-1)

    model = PolyNetwork(n_features=n_features)

    calc_mode = False

    if calc_mode:
        # init
        pred = model.forward(X)
        visualize_decision(X_origin, y, pred, name='init')

        # calc weights
        model.calc_weights(X, y)
        pred = model.forward(X)
        visualize_decision(X_origin, y, pred, name='calc_weights')

        # calc weights
        model.calc_weights(X, y, alpha=0.5)
        pred = model.forward(X)
        visualize_decision(X_origin, y, pred, name='calc_weights_reg')
    else:
        model.optimize(X_origin, X, y, steps=20000, lr=0.005, alpha=0.0)
