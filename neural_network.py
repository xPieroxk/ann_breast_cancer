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
                output = layer.forward_propagate(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        """train the network"""
        samples = len(x_train)
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
            print('epoch= %d/%d' % (epoch, epochs), ' loss=', loss)

    def __forward_propagation(self, input_data):
        output = input_data
        for layer in self.__layers:
            output = layer.forward_propagate(output)
        return output

    def __backward_propagation(self, error, learning_rate):
        for layer in reversed(self.__layers):
            error = layer.backward_propagate(error, learning_rate)
