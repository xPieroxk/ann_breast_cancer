import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

class NeuralNetwork:
    
    def __init__(self,n_input,hidden_layers,n_output):
        self.n_input = n_input
        self.hidden_layers = hidden_layers
        self.n_output = n_output
        # initialize weights and bias
        self.weights = []
        self.bias = []
        # fancy rappresentation of a neural network
        layers = [self.n_input]+self.hidden_layers+[self.n_output]
        # i am using three for cycles only to increase readability
        # generate random weights and bias
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i],layers[i+1])
            bias = np.random.rand(layers[i])
            self.weights.append(w)
            self.bias.append(bias)
        # save derivatives
        self.derivatives = []
        for i in range (len(layers)-1):
            d = np.zeros((layers[i],layers[i+1]))
            self.derivatives.append(d)
        # save activation per layer
        self.activations = []
        for i in range (len(layers)):
            a = np.zeros(layers[i])
            self.activations.append(a)

    def forward_propagate(self, inputs):
        # the input layer activation is just the input itself
        activations = inputs
        # save the activations for backpropogation
        self.activations[0] = activations
        for i,w in enumerate(self.weights):
            # calculate the dot product between the previous activation and weight matrix
            net_inputs = np.dot(activations,w)
            # apply sigmoid activation function
            activations = self.__sigmoid(net_inputs)
            # save the activations values for backprogation
            self.activations[i+1] = activations
        # return output layer
        return activations


    def backward_propagate():
        pass
    
    def train(epoch,batch_size=1):
        pass

    def validate():
        pass

    def __sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def __sigmoid_derivative():
        pass

    def __gradient_descent():
        pass



    def load_data_boston():
        return datasets.load_boston()

    def load_data_breast_cancer():
        return datasets.load_breast_cancer()

    def load_data_diabetes():
        return datasets.load_diabetes()
