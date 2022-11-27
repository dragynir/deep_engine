import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.engine.model_selection import kfold

import warnings
warnings.filterwarnings("ignore")


VIS_LINE = True


def visualize_decision(X, y, pred, name, params=''):
    if VIS_LINE:
        plt.plot(X, y)
        plt.plot(X, pred)
    else:
        plt.scatter(X, y)
        plt.scatter(X, pred)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(name + params)
    plt.show()


def create_custom_datasets(dataset_name="cos", dots=100, noise_level=0.0, noise_type='uniform'):
    """Cos, poly, sin"""
    X = np.linspace(0, 1, dots)

    if dataset_name == "cos":
        y = np.cos(2 * np.pi * X)
    elif dataset_name == "sin":
        y = np.sin(2 * np.pi * X)
    elif dataset_name == "poly":
        y = 5 * X ** 3 + X ** 2 + 5
    else:
        raise ValueError()

    if noise_type == 'uniform':
        noise = np.random.uniform(-noise_level, noise_level, y.shape)
    elif noise_type == 'normal':
        noise = np.random.normal(0, noise_level, y.shape)
    else:
        raise ValueError()

    y+=noise

    return X, y


def create_polynomial_features(X, degree=2):
    features = []
    for d in range(1, degree + 1):
        features.append(np.power(X, d))
    return np.stack(features, axis=-1)


class PolyNetwork:

    def __init__(self, n_features):
        self.weights = np.random.uniform(-1, 1, n_features + 1)
        self.n_features = n_features

    def forward(self, X):
        return X @ self.weights

    def calc_weights(self, X, y, alpha=0.0):
        x_t = X.T @ X
        x_t += alpha * np.ones(x_t.shape)
        self.weights = np.linalg.inv(x_t) @ X.T @ y

    def optimize(self, X_origin, X, y, steps=4, lr=0.01, alpha=0.0, vis=True):
        plot_every = steps // 5 if steps > 5 else 1

        for step in tqdm(range(steps)):
            w_grad = 2 * X.T @ (X @ self.weights - y) + 2 * alpha * self.weights
            self.weights -= lr * w_grad

            if vis and step % plot_every == 0:
                pred = self.forward(X)
                visualize_decision(X_origin, y, pred, name='optimize')
        print('Finish')


if __name__ == '__main__':

    # n_elements x 1
    # X_origin, y = create_custom_datasets(dataset_name='poly', dots=100)
    # X_origin, y = create_custom_datasets(dataset_name='sin', dots=100)
    # X_origin, y = create_custom_datasets(dataset_name='sin', dots=100, noise_level=0.2, noise_type='uniform')
    X_full_origin, y_full = create_custom_datasets(dataset_name='sin', dots=100, noise_level=0.2, noise_type='normal')

    dots_to_fit = 40
    # sample some dots
    indices = np.random.permutation(X_full_origin.shape[0])[:dots_to_fit]
    indices = sorted(indices)
    X_origin, y = X_full_origin[indices], y_full[indices]

    calc_mode = True
    cross_val_mode = False
    cv_splits = 4
    visualize_steps = False
    # poly_grid = [1, 2, 4]
    # alpha_grid = [0, 0.01, 0.5]

    poly_grid = [4]
    alpha_grid = [0.0]

    for poly_degree in poly_grid:
        for alpha in alpha_grid:

            params = f'Use poly {poly_degree}, alpha: {alpha}'

            # n_elements x n_features
            X = create_polynomial_features(X_origin, degree=poly_degree)
            X_full = create_polynomial_features(X_full_origin, degree=poly_degree)
            n_features = X.shape[-1]

            # add bias ones term
            X = np.concatenate([X, np.ones(X.shape[0])[..., None]], axis=-1)
            X_full = np.concatenate([X_full, np.ones(X_full.shape[0])[..., None]], axis=-1)
            model = PolyNetwork(n_features=n_features)

            if calc_mode:
                # init
                pred = model.forward(X_full)
                visualize_decision(X_full_origin, y_full, pred, name='init')

                # calc weights
                model.calc_weights(X, y)
                pred = model.forward(X_full)
                visualize_decision(X_full_origin, y_full, pred, name='calc_weights')

                # calc weights
                model.calc_weights(X, y, alpha=alpha)
                pred = model.forward(X_full)
                visualize_decision(X_full_origin, y_full, pred, name='calc_weights_reg')
            else:

                if cross_val_mode:
                    for fold_i, (val_fold, train_fold) in enumerate(
                            kfold((X, y), n_splits=cv_splits)
                    ):
                        model = PolyNetwork(n_features=n_features)
                        print('Fold index:', fold_i)
                        X_fold, y_fold = train_fold
                        visualize_decision(X_fold[:, 0], y_fold, y_fold, name='fold_data')
                        model.optimize(X_origin, X_fold, y_fold, steps=20000, lr=0.005, alpha=alpha, vis=False)

                        pred = model.forward(X_full)
                        visualize_decision(X_full_origin, y_full, pred, name='optimize')
                else:
                    model.optimize(X_origin, X, y, steps=20000, lr=0.005, alpha=alpha, vis=visualize_steps)
                    pred = model.forward(X_full)
                    visualize_decision(X_full_origin, y_full, pred, name='optimize', params=params)
