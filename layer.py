# base class layer for an Artificial Neural Network
class Layer:

    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagate(self, input_data):
        raise NotImplementedError

    def backward_propagate(self, output_error, learning_rate):
        raise NotImplementedError
