from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data_boston():
    sc = StandardScaler()
    x, y = datasets.load_boston(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    return x_train, x_test, y_train, y_test


def load_data_breast_cancer():
    sc = StandardScaler()
    x, y = datasets.load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5)
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    return x_train, x_test, y_train, y_test


def load_data_diabetes():
    sc = StandardScaler()
    x, y = datasets.load_diabetes(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5)
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    return x_train, x_test, y_train, y_test
