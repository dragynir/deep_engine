import os
from time import sleep

import numpy as np
import matplotlib.pyplot as plt


from src.engine.core import Value
from src.engine.nn import Neuron
from src.engine.model_selection import kfold


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
        plt.xlabel("X")
        plt.ylabel("y")
        plt.savefig("dataset.png")

    return np.array(X), np.array(y)


def visualize_decision(
    X,
    y,
    pred,
    feature_to_viz=0,
    out_path="",
    png_name="decision_train",
):
    X = X[:, feature_to_viz]
    plt.scatter(X, y, label="target")
    plt.plot(X, pred, label="pred", color="r")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(png_name)
    plt.legend()
    plt.savefig(os.path.join(out_path, f"{png_name}.png"))
    plt.show()


def create_model(in_features):
    model = Neuron(in_features, nonlin=False)
    print(model)
    print("number of parameters:", len(model.parameters()))
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


def train(model, dataset, exp_path, steps=100):

    X, y = dataset

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
        decay = 0.01
        learning_rate = start_lr - decay * k / steps
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 100 == 0:
            print(f"step {k} loss {total_loss.data}")
            pred = predict(model, X)
            visualize_decision(X, y, pred, out_path=exp_path)
            sleep(1)

    print("Finish training...")


def validation(model, dataset, exp_path):
    X, y = dataset
    pred = predict(model, X)
    loss = mse_loss(y, pred)
    print("Validation mse:", loss)
    visualize_decision(X, y, pred, out_path=exp_path, png_name="decision_val")
    return {"mse": loss}


if __name__ == "__main__":
    dataset = create_dataset(visualize=False)
    poly_degree = 4
    cv_splits = 2
    cv = True
    steps = 2000
    model = create_model(in_features=poly_degree)
    dataset = preprocess(dataset, n_features=poly_degree)
    experiment_path = "experiment" + ("_cv" if cv else "")
    os.makedirs(experiment_path)

    if cv:
        print('Start cross-validation...')
        cv_metrics = []
        for fold_i, (val_fold, train_fold) in enumerate(
            kfold(dataset, n_splits=cv_splits)
        ):

            fold_exp_path = os.path.join(experiment_path, f'fold_{fold_i}')
            os.makedirs(fold_exp_path, exist_ok=True)

            train(model, dataset=train_fold, exp_path=fold_exp_path, steps=steps)
            metrics = validation(model, dataset, exp_path=fold_exp_path)
            print(f"Validation on fold {fold_i}:", metrics)
            cv_metrics.append(metrics["mse"])

        print("Cv result:", sum(cv_metrics) / len(cv_metrics))
    else:
        train(model, dataset=dataset, exp_path=experiment_path, steps=steps)
        metrics = validation(model, dataset, experiment_path)
        print("Result metrics: ", metrics)
