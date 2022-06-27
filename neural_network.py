from typing import List

from sklearn import datasets


class NeuralNetwork:

    def __init__(self):
        self.__layers = []
        self.__loss = None
        self.__loss_derivative = None

    def add_layer(self, layer):
        self.__layers.append(layer)

    def set_loss(self, loss):
        self.__loss = loss

    def set_loss_derivative(self, loss_derivative):
        self.__loss_derivative = loss_derivative

    # predict output for given input
    def predict(self, input_data) -> List[float]:
        """predict the output for given input"""
        # sample dimension first
        samples = len(input_data)
        result = []
        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.__layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    def validate(self, target, predicted) -> float:
        """validate the network"""
        n_correct = 0
        for i in range(len(target) - 1):
            if target[i] == predicted[i]:
                n_correct += 1
        return n_correct / len(target)

    def fit(self, x_train, y_train, epochs, learning_rate) -> List[float]:
        """train the network"""
        samples = len(x_train)
        # stores losses (for display purpose only)
        losses_values = []
        for epoch in range(epochs):
            loss = 0
            for i in range(samples):
                # forward propagation
                output = self.__forward_propagation(x_train[i])
                loss += self.__loss(y_train[i], output)
                # backward propagation
                # error=dC/da^l
                error = self.__loss_derivative(y_train[i], output)
                self.__backward_propagation(error, learning_rate)
            losses_values.append(loss)
        return losses_values

    def __forward_propagation(self, input_data):
        output = input_data
        for layer in self.__layers:
            output = layer.forward_propagate(output)
        return output

    def __backward_propagation(self, error, learning_rate):
        for layer in reversed(self.__layers):
            error = layer.backward_propagate(error, learning_rate)
