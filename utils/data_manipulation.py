import numpy as np
from itertools import combinations_with_replacement


def shuffle_data(X, y, seed=None):
    """
    Random shuffle of the samples in X and y
    """

    if seed:
        np.random.seed(seed)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    return X[idx], y[idx]


def batch_iterator(X, y=None, batch_size=64):
    """
    Simple batch generator
    """

    n_samples = X.shape[0]
    for i in range(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]


def divide_on_feature(X, feature_i, threshold):
    """
    Divide dataset based on if sample value on feature index is larger than
    the given threshold
    """

    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])


def fast_divide_on_feature(X, feature_i, threshold):
    """
    Fast divide dataset based on vector
    """

    if isinstance(threshold, (int, float)):
        mask = X[:, feature_i] >= threshold
    else:
        mask = X[:, feature_i] == threshold

    return X[mask], X[~mask]


def polynomial_features(X, degree):
    """
    Automatically expand the original features into all polynomial combination features
    """

    n_samples, n_features = np.shape(X)

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))

    for i, index_combs in enumerate(combinations):
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new


def get_random_subsets(X, y, n_subsets, replacements=True):
    """
    Return random subsets (with replacements) of the data
    """

    n_samples = np.shape(X)[0]
    X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(X_y)
    subsets = []

    subsample_size = int(n_samples // 2)
    if replacements:
        subsample_size = n_samples

    for _ in range(n_subsets):
        idx = np.random.choice(
            n_samples,
            size=subsample_size,
            replace=replacements
        )
        X = X_y[idx][:, :-1]
        y = X_y[idx][:, -1]
        subsets.append([X, y])

    return subsets


def normalize(X, axis=-1, order=2):
    """
    Normalize the dataset X
    """

    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2==0] = 1
    return X / np.expand_dims(l2, axis)


def standardize(X):
    """
    Standardize the dataset X
    """

    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]

    return X_std


def fast_standardize(X):
    """
    Fast standardize the dataset X
    """

    X_std = X.copy()
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X_std = (X_std - mean) / std

    return X_std


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """
    Split the data into train and test sets
    """

    if shuffle:
        X, y = shuffle_data(X, y, seed)

    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


def k_fold_cross_validation_sets(X, y, k, shuffle=True):
    """
    Split the data into k sets of training / test data
    """

    if shuffle:
        X, y = shuffle_data(X, y)

    n_samples = len(y)
    left_overs = {}
    n_left_overs = (n_samples % k)
    if n_left_overs != 0:
        left_overs['X'] = X[-n_left_overs:]
        left_overs['y'] = y[-n_left_overs:]
        X = X[:-n_left_overs]
        y = y[:-n_left_overs]

    X_split = np.split(X, k)
    y_split = np.split(y, k)
    sets = []
    for i in range(k):
        X_test, y_test = X_split[i], y_split[i]
        X_train = np.concatenate(X_split[:i] + X_split[i+1:], axis=0)
        y_train = np.concatenate(y_split[:i] + y_split[i+1:], axis=0)
        sets.append([X_train, X_test, y_train, y_test])

    # if n_left_overs != 0:
    #     np.append(sets[-1][0], left_overs['X'], axis=0)
    #     np.append(sets[-1][2], left_overs['y'], axis=0)

    return np.array(sets)


def to_categorical(x, n_col=None):
    """
    One-hot encoding of nominal values
    """

    if not n_col:
        n_col = np.amax(x) + 1
    
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1

    return one_hot


def to_nominal(x):
    """
    Conversion from one-hot encoding to nominal
    """

    return np.argmax(x, axis=1)


def make_diagonal(x):
    """
    Converts a vector into an diagonal matrix
    """

    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])):
        m[i, i] = x[i]
    
    return m


if __name__ == '__main__':
    polynomial_features(np.array([[2]]), 2)
