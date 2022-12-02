import math
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import ImageGrid


warnings.filterwarnings("ignore")


def l1_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))


def l2_distance(v1, v2):
    return np.sqrt(np.sum(np.power(v1 - v2, 2)))


class SimpleKNN:

    def __init__(self, acc_size):
        self.acc = np.zeros(acc_size)

    def predict(self, inputs, labels, datapoint, metric=l2_distance, k=4):
        """
        :param inputs: обучающие примеры
        :param labels: лейблы для inputs
        :param datapoint: пример для предикта
        :pram k: кол-во соседей
        """
        indices = np.array(range(len(inputs)))
        distances = np.array(tuple(metric(datapoint, d) for d in inputs))
        sort_ind = np.argsort(distances)

        labels_ind = indices[sort_ind]
        labels = labels[sort_ind]
        top_k = labels[:k]
        labels_ind = labels_ind[:k]

        self.acc[labels_ind] += 1
        majority_class = Counter(top_k)
        return majority_class.most_common(1).pop()[0]

    def get_outliers(self, k=4):
        plt.hist(self.acc)
        plt.title('Сколько раз сэмпл был соседом других сэмплов.')
        plt.show()
        return np.argwhere(self.acc < k)


def add_noise(x, noise_type, noise_level=1):
    if noise_type == "uniform":
        noise = np.random.uniform(-noise_level, noise_level, x.shape)
    elif noise_type == "normal":
        noise = np.random.normal(0, noise_level, x.shape)
    else:
        raise ValueError()
    return x + noise


if __name__ == "__main__":
    digits = load_digits()
    images = digits.images
    labels = digits.target
    images = images.reshape(-1, 64)

    print(labels.shape)
    print(images.shape)

    # images = images[:20]
    # labels = labels[:20]

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)
    print(Counter(y_test))

    metric = l2_distance
    K = 4
    outliers_k = 3
    find_outliers = True

    plt.figure()
    plt.imshow(X_test[0].reshape((8, 8)))
    # add noise
    # X_test = add_noise(X_test, noise_type="normal", noise_level=3)
    plt.figure()
    plt.imshow(X_test[0].reshape((8, 8)))
    plt.show()

    fig = plt.figure(figsize=(4.0, 4.0))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(2, 2),
        axes_pad=0.1,
    )

    for ax, im in zip(grid, X_test[56:60]):
        ax.imshow(im.reshape((8, 8)))
    plt.show()

    if not find_outliers:
        knn = SimpleKNN(len(X_train))
        correct_count = 0
        for x, y in tqdm(zip(X_test, y_test)):
            y_pred = knn.predict(X_train, y_train, x, metric=metric, k=K)
            correct_count += y_pred == y
        print("Accuracy: ", correct_count / len(y_test))
    else:
        knn = SimpleKNN(len(X_train))
        outliers = []
        for ind, (x, y) in tqdm(enumerate(zip(X_train, y_train))):
            y_pred = knn.predict(X_train, y_train, x, metric=metric, k=K)
            if y_pred != y:
                outliers.append(ind)

        print(f'Found simple {len(outliers)} outliers.')

        odin_outliers = knn.get_outliers(k=outliers_k)
        print(f'Found odin {len(odin_outliers)} outliers')

        # for i in outliers:
        #     plt.figure()
        #     plt.title(y_train[i])
        #     plt.imshow(X_train[i].reshape((8, 8)))
        # plt.show()

        # for i in odin_outliers:
        #     plt.figure()
        #     plt.title(y_train[i])
        #     plt.imshow(X_train[i].reshape((8, 8)))
        # plt.show()
