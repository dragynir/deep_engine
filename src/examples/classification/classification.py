import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs, make_classification

from src.engine.core import Value
from src.engine.nn import MLP, Neuron
from src.engine.utils import seed_everything


def visualize_decision(model):
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(Value, xrow)) for xrow in Xmesh]
    scores = list(map(model, inputs))
    Z = np.array([s.data > 0 for s in scores])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.savefig("decision.png")
    plt.show()


def plot_dataset(X, y):
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap="jet")
    plt.savefig("data.png")


def create_moons_dataset():
    X, y = make_moons(n_samples=100, noise=0.1)
    y = y * 2 - 1
    plot_dataset(X, y)
    return X, y


def create_blobs_dataset(centers=2):
    X, y = make_blobs(n_features=2, centers=centers)
    y = y * 2 - 1
    plot_dataset(X, y)
    return X, y


def create_classification_dataset(n_informative=2):
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2)
    y = y * 2 - 1
    plot_dataset(X, y)
    return X, y


def max_margin_loss(target, pred):
    """Hinge loss"""
    losses = [(1 + -yi * predi).relu() for yi, predi in zip(target, pred)]
    loss = sum(losses) * (1.0 / len(losses))
    return loss


def binary_cross_entropy(target, pred):
    losses = [yi * predi.log() + (1 - yi) * (1 - predi).log() for yi, predi in zip(target, pred)]
    loss = sum(losses) * (1.0 / len(losses))
    return loss


def accuracy(target, pred):
    acc = [(yi > 0) == (predi.data > 0) for yi, predi in zip(target, pred)]
    return sum(acc) / len(acc)


def train(model, dataset, steps=100):

    X, y = dataset

    for k in range(steps):

        Xb, yb = X, y

        # forward
        inputs = [list(map(Value, xrow)) for xrow in Xb]
        pred = list(map(model, inputs))
        loss = max_margin_loss(yb, pred)
        # loss = binary_cross_entropy(yb, pred)
        acc = accuracy(yb, pred)

        alpha = 1e-4
        reg_loss = alpha * sum((p * p for p in model.parameters()))
        total_loss = loss + reg_loss

        # backward
        model.zero_grad()
        total_loss.backward()

        learning_rate = 1.0 - 0.9 * k / steps
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 5 == 0:
            print(f"step {k} loss {total_loss.data}, accuracy {acc * 100}%")

    print("Finish training...")


if __name__ == "__main__":

    seed_everything(42)
    X, y = create_moons_dataset()
    # X, y = create_blobs_dataset()
    # X, y = create_classification_dataset()

    model = MLP(
        2,
        nouts=[8, 8, 2, 1],
        activations=['relu', 'relu', 'relu', 'tanh'],
        initializer='he',  # xavier, he
    )

    # model = Neuron(2)
    # model = Neuron(2, activation='sigmoid')

    print(model)
    print("number of parameters", len(model.parameters()))

    train(model, dataset=(X, y), steps=50)
    visualize_decision(model)
