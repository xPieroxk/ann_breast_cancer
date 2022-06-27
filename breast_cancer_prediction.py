import numpy as np
import dataset as ds
from neural_network import NeuralNetwork
from activation_functions import sigmoid, sigmoid_derivative
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from loss_functions import mse, mse_derivative

x_train, x_test, y_train, y_test = ds.load_data_breast_cancer()

net = NeuralNetwork()
net.add_layer(FCLayer(30, 10))
net.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))
net.add_layer(FCLayer(10, 1))
net.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))

net.set_loss(mse)
net.set_loss_derivative(mse_derivative)
x_train.reshape(x_train.shape[0], 1, 30)
net.fit(x_train, y_train, epochs=100, learning_rate=0.01)
