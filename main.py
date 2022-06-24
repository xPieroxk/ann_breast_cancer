import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets



class NeuralNetwork:

    def __init__(self, n_input, hidden_layers, n_output):
        self.n_input = n_input
        self.hidden_layers = hidden_layers
        self.n_output = n_output
        self.weights = []
        self.bias = []
        self.activations = []
        # fancy rappresentation of a neural network
        layers = [self.n_input] + self.hidden_layers + [self.n_output]
        # generate random weights and bias
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            b = np.random.rand(layers[i])
            self.weights.append(w)
            self.bias.append(b)

    def forward_propagate(self, inputs):
        # the input layer activation is just the input itself
        activations = inputs
        # save the activations for backpropogation
        self.activations.append(activations)
        for i, w in enumerate(self.weights):
            # calculate the dot product between the previous activation and weight matrix
            net_inputs = np.dot(activations, w.T) + self.bias[i]
            # apply sigmoid activation function
            activations = self.__sigmoid(net_inputs)
            # save the activations values for backprogation
            self.activations.append(activations)
        # return output layer
        return activations

    def backward_propagate(self, y_true, y_pred):
        error = self.__mse(y_true, y_pred)

        pass





    # mse loss function, The 1/2 is included so that exponent is cancelled when we differentiate
    def __mse(self, y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2))/2

    def __mse_prime(self, y_true, y_pred):
        return (y_pred - y_true)/y_true.size

    def __gradient_descent(self):
        pass

