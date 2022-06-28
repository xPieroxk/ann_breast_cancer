import numpy as np
import dataset as ds
from neural_network import NeuralNetwork
from activation_functions import sigmoid, sigmoid_derivative, tanh, tanh_derivative
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from loss_functions import mse, mse_derivative


def validate(target, predicted) -> float:
    """validate the network"""
    p = [[0] if x < 0.5 else [1] for x in predicted]
    n_correct = 0
    for i in range(len(target) - 1):
        if target[i] == p[i]:
            n_correct += 1
    return n_correct / len(target)


# load data
x_train, x_test, y_train, y_test = ds.load_data_breast_cancer()
# normalize data
x_train = np.reshape(x_train, (512, 1, 30))
x_test = np.reshape(x_test, (57, 1, 30))
y_train = np.reshape(y_train, (512, 1, 1))
y_test = np.reshape(y_test, (57, 1, 1))

# create network
net = NeuralNetwork()
net.add_layer(FCLayer(30, 4))
net.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))
net.add_layer(FCLayer(4, 2))
net.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))
net.add_layer(FCLayer(2, 1))
net.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))
net.set_loss(mse)
net.set_loss_derivative(mse_derivative)
# train network
net.fit(x_train, y_train, epochs=500, learning_rate=0.1)
# predict
print('accuracy of=', validate(y_test, net.predict(x_test)))
