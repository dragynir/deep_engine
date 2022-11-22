import numpy as np
import matplotlib.pyplot as plt


from src.engine.core import Value
from src.engine.nn import Neuron


def create_dataset():
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
    plt.scatter(X, y)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.savefig('dataset.png')

    return np.array(X), np.array(y)


def visualize_decision(model, X_origin, X, y):
    inputs = [list(map(Value, xrow)) for xrow in X]
    pred = list(map(model, inputs))
    pred = [s.data > 0 for s in pred]

    plt.scatter(X_origin, y, label='target')
    plt.scatter(X_origin, pred, label='pred')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('decision.png')


def create_model():
    model = Neuron(2, nonlin=False)
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


def normalize(X):
    features = []
    for i in range(X.shape[1]):
        features.append(min_max_norm(X[:, i]))
    return np.stack(features, axis=-1)


def mse_loss(target, pred):
    losses = [(yi - predi) ** 2 for yi, predi in zip(target, pred)]
    return sum(losses) * (1.0 / len(losses))


def train(model, dataset, steps=100):
    X_origin, y = dataset
    X = create_polynomial_features(X_origin)
    X = normalize(X)

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
        learning_rate = 0.0001  #  1.0 - 0.9 * k / 100
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 5 == 0:
            print(f"step {k} loss {total_loss.data}")

    visualize_decision(model, X_origin, X, y)
    print('Finish training...')


if __name__ == "__main__":
    X, y = create_dataset()
    model = create_model()
    train(model, dataset=(X, y), steps=400)

