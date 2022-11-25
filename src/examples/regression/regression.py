import os
from time import sleep

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression

from src.engine.core import Value
from src.engine.nn import Neuron, MLP
from src.engine.model_selection import kfold
from src.engine.optim import l2_regularization, l1_regularization, SGD
from src.engine.utils import seed_everything


def create_poly_dataset():
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
    return np.array(X), np.array(y)


def visualize_dataset(dataset, out_path, feature=0):
    X, y = dataset
    plt.scatter(X[:, feature], y)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.savefig(os.path.join(out_path, "dataset.png"))


def visualize_decision(
    X,
    y,
    pred,
    feature_to_viz=0,
    out_path="",
    png_name="decision_train",
    decision_as_line=True,
):
    X = X[:, feature_to_viz]
    plt.scatter(X, y, label="target")
    if decision_as_line:
        plt.plot(X, pred, label="pred", color="r")
    else:
        plt.scatter(X, pred, label="pred", color="r")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(png_name)
    plt.legend()
    plt.savefig(os.path.join(out_path, f"{png_name}.png"))
    plt.show()


def create_polynomial_features(X, degree=2):
    features = []
    for d in range(1, degree + 1):
        features.append(np.power(X, d))
    return np.stack(features, axis=-1)


def mse_loss(target, pred):
    losses = [(yi - predi) ** 2 for yi, predi in zip(target, pred)]
    return sum(losses) * (1.0 / len(losses))


def min_max_norm(feature):
    min_v, max_v = feature.min(), feature.max()
    return (feature - min_v) / (max_v - min_v + 1e-16)


def std_norm(feature):
    std, mean = feature.std(), feature.mean()
    return (feature - mean) / (std + 1e-16)


def normalize(X, y, x_norm_func=min_max_norm, y_norm_func=min_max_norm):
    features = []
    for i in range(X.shape[1]):
        features.append(x_norm_func(X[:, i]))
    X = np.stack(features, axis=-1)
    y = y_norm_func(np.array(y))
    return X, y


def preprocess(dataset, n_features):
    X_origin, y_origin = dataset
    X = create_polynomial_features(X_origin, degree=n_features)
    X, y = normalize(X, y_origin)
    return X, y


def predict(model, X):
    inputs = [list(map(Value, xrow)) for xrow in X]
    pred = list(map(model, inputs))
    model.zero_grad()  # TODO add eval mode
    return [s.data for s in pred]


def train(
    model,
    dataset,
    exp_path,
    lr=0.05,
    steps=100,
    batch_size=None,
    regularizatioin_func=l2_regularization,
    decision_as_line=False,
):

    X, y = dataset

    optimizer = SGD(model.parameters(), momentum=0.9)

    for k in range(steps):

        if batch_size is None:
            Xb, yb = X, y
        else:
            indices = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = X[indices], y[indices]

        # forward
        inputs = [list(map(Value, xrow)) for xrow in Xb]
        pred = list(map(model, inputs))
        loss = mse_loss(yb, pred)
        total_loss = loss + regularizatioin_func(model.parameters(), alpha=1e-4)

        # backward
        model.zero_grad()
        total_loss.backward()
        optimizer.step(lr)

        if k % 100 == 0:
            print(f"step {k} loss {total_loss.data}")
            pred = predict(model, X)
            visualize_decision(
                X,
                y,
                pred,
                out_path=exp_path,
                decision_as_line=decision_as_line,
            )
            # sleep(1)

    print("Finish training...")


def validation(model, dataset, exp_path, decision_as_line):
    X, y = dataset
    pred = predict(model, X)
    loss = mse_loss(y, pred)
    print("Validation mse:", loss)
    visualize_decision(
        X,
        y,
        pred,
        out_path=exp_path,
        png_name="decision_val",
        decision_as_line=decision_as_line,
    )
    return {"mse": loss}


if __name__ == "__main__":

    seed_everything(42)

    reg_type = "mlp"  # mlp, poly
    decision_as_line = False
    cv_splits = 2
    batch_size = 16
    cv = True
    steps = 2000
    lr = 0.01
    regularizatioin_func = l2_regularization
    experiment_path = f"{reg_type}_experiment" + ("_cv" if cv else "")
    os.makedirs(experiment_path, exist_ok=True)

    dataset = None
    model = None

    if reg_type == "poly":
        poly_degree = 4
        dataset = create_poly_dataset()
        # visualize_dataset(dataset, experiment_path)
        model = Neuron(poly_degree)
        dataset = preprocess(dataset, n_features=poly_degree)
    elif reg_type == "mlp":
        n_features = 1
        model = MLP(
            n_features,
            nouts=[2, 2, 1],  # [2, 2, 1]
            # activations=['relu', 'relu', None],  # relu, sigmoid, tanh
            # activations=['sigmoid', 'sigmoid', None],  # relu, sigmoid, tanh
            activations=['tanh', 'tanh', None],
            initializer='xavier',  # xavier, he
        )

        # [-1, 1] - веса, то с relu градиенты затухнут для -
        # model = Neuron(1)
        dataset = make_regression(
            n_samples=50,
            n_features=n_features,
            noise=5,
            shuffle=True,
        )
        # visualize_dataset(dataset, experiment_path)
        X, y = dataset
        dataset = normalize(X, y, x_norm_func=std_norm)
    else:
        raise ValueError()

    if cv:
        print("Start cross-validation...")
        cv_metrics = []
        for fold_i, (val_fold, train_fold) in enumerate(
            kfold(dataset, n_splits=cv_splits)
        ):
            fold_exp_path = os.path.join(experiment_path, f"fold_{fold_i}")
            os.makedirs(fold_exp_path, exist_ok=True)

            train(
                model,
                dataset=train_fold,
                exp_path=fold_exp_path,
                lr=lr,
                batch_size=batch_size,
                steps=steps,
                decision_as_line=decision_as_line,
                regularizatioin_func=regularizatioin_func,
            )
            metrics = validation(
                model,
                dataset,
                exp_path=fold_exp_path,
                decision_as_line=decision_as_line,
            )
            print(f"Validation on fold {fold_i}:", metrics)
            cv_metrics.append(metrics["mse"])

        print("Cv result:", sum(cv_metrics) / len(cv_metrics))
    else:
        train(
            model,
            dataset=dataset,
            exp_path=experiment_path,
            lr=lr,
            batch_size=batch_size,
            steps=steps,
            decision_as_line=decision_as_line,
            regularizatioin_func=regularizatioin_func,
        )
        metrics = validation(
            model,
            dataset,
            experiment_path,
            decision_as_line=decision_as_line,
        )
        print("Result metrics: ", metrics)
