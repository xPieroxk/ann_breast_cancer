import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    if x > 0:
        return x
    return 0


def relu_derivative(x):
    if relu(x) > 0:
        return 1
    return 0
