from layer import Layer
import numpy as np


class ActivationLayer(Layer):

    def __init__(self, activation, activation_first_derivative):
        super().__init__()
        self.activation = activation
        self.activation_first_derivative = activation_first_derivative

    def forward_propagate(self, input_data):
        return self.activation(input_data)

    def backward_propagate(self, output_error, learning_rate) -> np.array:
        # Hadamard product
        # returns dC/dz^l
        return np.multiply(output_error, self.activation_first_derivative(self.input))
