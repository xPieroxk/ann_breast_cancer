from layer import Layer
import numpy as np


class FCLayer(Layer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(input_size)

    def forward_propagate(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagate(self, output_error, learning_rate):
        pass
