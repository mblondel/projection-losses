# Author: Mathieu Blondel, 2019
# License: BSD

import numpy as np


def load_label_ranking_data(fn):
    f = open(fn)

    next(f)  # skip first line

    X = []
    Y = []

    for line in f:
        arr = line.strip().split(",")
        features = np.array(arr[:-1], dtype=float)
        X.append(features)

        # Labels have the form b > c > a.
        # We encoded it as y = [1 2 0].
        # Therefore, y[rank] = label.
        ranking = arr[-1].split(">")
        y = np.zeros(len(ranking))

        for i, letter in enumerate(ranking):
            label = ord(letter) - ord("a")
            y[i] = label

        Y.append(y)

    return np.array(X), np.array(Y, dtype=int)


if __name__ == '__main__':
    X, Y = load_label_ranking_data("data/ranking/iris.txt")
    print(Y)
