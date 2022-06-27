import numpy as np


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))/2


def mse_derivative(y_true, y_pred):
    return (y_pred - y_true) / y_true.size
