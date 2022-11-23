from time import sleep

import numpy as np
import matplotlib.pyplot as plt


from src.engine.core import Value
from src.engine.nn import Neuron


def create_dataset(visualize=True):
    X = np.arange(0, 30)
    y = [
        3,
        4,
        5,
        7,
        10,
        8,
        9,
        10,
        10,
        23,
        27,
        44,
        50,
        63,
        67,
        60,
        62,
        70,
        75,
        88,
        81,
        87,
        95,
        100,
        108,
        135,
        151,
        160,
        169,
        179,
    ]
    if visualize:
        plt.scatter(X, y)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.savefig('dataset.png')

    return np.array(X), np.array(y)


def visualize_decision(model, X_origin, X, y):
    inputs = [list(map(Value, xrow)) for xrow in X]
    pred = list(map(model, inputs))
    pred = [s.data for s in pred]

    plt.scatter(X_origin, y, label='target')
    plt.plot(X_origin, pred, label='pred', color='r')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('decision.png')
    plt.show()


def create_model(in_features):
    model = Neuron(in_features, nonlin=False)
    print(model)
    print('number of parameters:', len(model.parameters()))
    return model


def create_polynomial_features(X, degree=2):
    features = []
    for d in range(1, degree + 1):
        features.append(np.power(X, d))
    return np.stack(features, axis=-1)


def min_max_norm(feature):
    min_v, max_v = feature.min(), feature.max()
    return (feature - min_v) / (max_v - min_v + 1e-16)


def normalize(X, y):
    features = []
    for i in range(X.shape[1]):
        features.append(min_max_norm(X[:, i]))
    X = np.stack(features, axis=-1)
    y = min_max_norm(np.array(y))
    return X, y


def mse_loss(target, pred):
    losses = [(yi - predi) ** 2 for yi, predi in zip(target, pred)]
    return sum(losses) * (1.0 / len(losses))


def train(model, dataset, n_features, steps=100):
    X_origin, y_origin = dataset
    X = create_polynomial_features(X_origin, degree=n_features)
    X, y = normalize(X, y_origin)

    for k in range(steps):

        Xb, yb = X, y  # TODO create dataloader

        # forward
        inputs = [list(map(Value, xrow)) for xrow in Xb]
        pred = list(map(model, inputs))
        loss = mse_loss(yb, pred)

        # l2 regularization # TODO move to package
        alpha = 1e-4
        reg_loss = alpha * sum((p * p for p in model.parameters()))
        total_loss = loss + reg_loss

        # backward
        model.zero_grad()
        total_loss.backward()

        # update weights (sgd) # TODO move
        start_lr = 0.05
        momentum = 0.01
        learning_rate = start_lr - momentum * k / steps
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 100 == 0:
            print(f"step {k} loss {total_loss.data}")
            visualize_decision(model, X_origin, X, y)
            model.zero_grad()
            sleep(1)

    print('Finish training...')


if __name__ == "__main__":
    X, y = create_dataset(visualize=False)
    poly_degree = 4
    model = create_model(in_features=poly_degree)
    train(model, dataset=(X, y), n_features=poly_degree, steps=2000)
