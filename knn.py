from math import exp, pi, sqrt
from sklearn.neighbors import NearestNeighbors

FIXED_WINDOW = 1
VARIABLE_WINDOW = 2

MANHATTAN_METRIC = 1
EUCLIDEAN_METRIC = 2
COSINE_METRIC = 3

def uniform_kernel(x):
    return 0.5 if -1 < x < 1 else 0

def triangular_kernel(x):
    return max(0, 1 - abs(x))

def epanechnikov_kernel(x):
    return max(0, 0.75 * (1 - x ** 2))

def gaussian_kernel(x):
    return 1 / sqrt(2 * pi) * exp(- (x ** 2 / 2))

def get_kernel_function(kernel_type):
    if kernel_type == "UNIFORM":
        return uniform_kernel
    elif kernel_type == "TRIANGULAR":
        return triangular_kernel
    elif kernel_type == "EPANECHNIKOV":
        return epanechnikov_kernel
    elif kernel_type == "GAUSSIAN":
        return gaussian_kernel

def fit_knn(y_train, window_param, window_type):
    h = None
    k = None
    if window_type == FIXED_WINDOW:
        h = window_param
    else:
        k = window_param

    classes = len(y_train.value_counts())

    return window_type, h, k, classes

def predict_knn(X_test, X_train, y_train, w, window_type, h, k, kernel_type, metric_type, classes):
    kernel_func = get_kernel_function(kernel_type)
    predictions = []
    all_distances, all_results, all_weights = find_neighbors(X_test, X_train, y_train, w, window_type, k, metric_type)

    for x in range(len(X_test)):
        distances, classes, weights = all_distances[x], all_results[x], all_weights[x]
        scores = [0 for _ in range(classes)]

        for i in range(len(distances) - 1):
            kernel_arg = distances[i] / (h if window_type == FIXED_WINDOW else distances[-1])
            scores[classes[i]] += kernel_func(kernel_arg) * weights[i]

        predictions.append(scores.index(max(scores)))

    return predictions

def find_neighbors(X_test, X_train, y_train, w, window_type, k, metric_type):
    neighbors_count = k + 1 if window_type == VARIABLE_WINDOW else min(int(sqrt(len(X_train))), len(X_train) - 1)
    metric_name = {1: 'manhattan', 2: 'euclidean', 3: 'cosine'}[metric_type]
    nn = NearestNeighbors(n_neighbors=neighbors_count, metric=metric_name)
    nn.fit(X_train)
    distances, ids = nn.kneighbors(X_test, n_neighbors=neighbors_count)

    results, weights = [], []

    for i in ids:
        results.append(y_train.iloc[i].to_list())
        weights.append([w[j] for j in i])

    return distances, results, weights
