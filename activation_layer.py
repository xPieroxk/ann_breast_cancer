from layer import Layer


class ActivationLayer(Layer):

    def __init__(self, activation, activation_first_derivative):
        super().__init__()
        self.activation = activation
        self.activation_first_derivative = activation_first_derivative

    def forward_propagate(self, input_data):
        return self.activation(input_data)

    def backward_propagate(self, output_error, learning_rate):
        return self.activation_first_derivative(self.input) * output_error
