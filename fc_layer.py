from layer import Layer
import numpy as np


class FCLayer(Layer):

    def __init__(self, input_size, output_size):
        super().__init__()
        # initialize seed for random numbers
        np.random.seed(1)
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(1, output_size)

    def forward_propagate(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagate(self, delta_l, learning_rate):
        # dC/da^(l-1)
        input_error = np.dot(delta_l, self.weights.T)
        # update weights and bias
        # dC/dw^l
        weights_error = np.dot(self.input.T, delta_l)
        # dC/db^l
        bias_error = delta_l
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error
