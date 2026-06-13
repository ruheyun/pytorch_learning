import math
import numpy as np


def calculate_entropy(y):
    """
    Calculate the entropy of label array y
    """

    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y==label])
        p = count / len(y)
        entropy += -p * log2(p)

    return entropy


def calculate_entropy_(y):
    """
    Calculate the entropy of label array y
    """

    entropy = 0
    for label in np.unique(y):
        p = np.sum(y == label) / len(y)
        entropy -= p * math.log2(p)

    return entropy


def mean_squared_error(y_true, y_pred):
    """
    Returns the mean squared error between y_true and y_pred
    """

    mse = np.mean(np.power(y_true - y_pred, 2))

    return mse


def calculate_variance(X):
    """
    Return the variance of the features in dataset X
    """

    mean = np.ones(np.shape(X)) * X.mean(axis=0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))

    return variance


def calculate_std_dev(X):
    """
    Calculate the standard deviations of the features in dataset X
    """

    std_dev = np.sqrt(calculate_variance(X))

    return std_dev


def euclidean_distance(x_1, x_2):
    """
    Calculates the l2 distance between two vectors
    """

    assert len(x_1) == len(x_2), 'Length of list or numpy is different.'

    distance = 0
    for i, j in zip(x_1, x_2):
        distance += pow((i - j), 2)

    return math.sqrt(distance)


def accuracy_score(y_true, y_pred):
    """
    Compare y_true to y_pred and return the accuracy
    """

    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)

    return accuracy


def calculate_covariance_matrix(X, Y=None):
    """
    Calculate the covariance matrix for the dataset X
    """

    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)


def calculate_correlation_matrix(X, Y=None):
    """
    Calculate the correlation matrix for the dataset X
    """

    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance = (1 / n_samples) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype=float)


if __name__ == '__main__':
    euclidean_distance([1, 2, 3], [4, 5, 6])
