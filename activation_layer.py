from layer import Layer
import numpy as np


class ActivationLayer(Layer):

    def __init__(self, activation, activation_first_derivative):
        super().__init__()
        self.activation = activation
        self.activation_first_derivative = activation_first_derivative

    def forward_propagate(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagate(self, output_error, learning_rate):
        # Hadamard product
        # returns dC/dz^l
        return output_error * self.activation_first_derivative(self.input)
