from sklearn import datasets


class NeuralNetwork():

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, x_train, y_train, epochs, learnin_rate):
        pass

    def predict(self, input_data):
        pass

    def validate(self, target, predicted):
        n_correct = 0
        for i in range(len(target) - 1):
            if target[i] == predicted[i]:
                n_correct += 1
        return n_correct

    def load_data_boston(self):
        return datasets.load_boston()

    def load_data_breast_cancer(self):
        return datasets.load_breast_cancer()

    def load_data_diabetes(self):
        return datasets.load_diabetes()
