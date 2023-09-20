import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions
from perceptron import Perceptron 

class PerceptronClassifier:
    def __init__(self, ppn1, ppn2):
        self.ppn1 = ppn1
        self.ppn2 = ppn2

    def predict(self, x):
        return np.where(self.ppn1.predict(x) == 1, 0, np.where(self.ppn2.predict(x) == 1, 2, 1))

def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    y_train_01_subset = y_train.copy()
    y_train_03_subset = y_train.copy()
    X_train_01_subset = X_train.copy()

    y_train_01_subset[(y_train == 1) | (y_train == 2)] = -1
    y_train_01_subset[(y_train_01_subset == 0)] = 1

    y_train_03_subset[(y_train == 1) | (y_train == 0)] = -1
    y_train_03_subset[(y_train_03_subset == 2)] = 1

    print('y_train_01_subset: ', y_train_01_subset)
    print('y_train_03_subset: ', y_train_03_subset)

    ppn1 = Perceptron(eta=0.1, n_iter=300)
    ppn1.fit(X_train_01_subset, y_train_01_subset)

    ppn2 = Perceptron(eta=0.1, n_iter=300)
    ppn2.fit(X_train_01_subset, y_train_03_subset)

    y1_predict = ppn1.predict(X_train)
    y3_predict = ppn2.predict(X_train)

    accuracy_1 = accuracy(ppn1.predict(X_train), y_train_01_subset)
    accuracy_3 = accuracy(ppn2.predict(X_train), y_train_03_subset)
    print("Perceptron #1 accuracy: ", accuracy_1)
    print("Perceptron #2 accuracy: ", accuracy_3)

    if accuracy_1 > accuracy_3:
        y_results = np.where(y1_predict == 0, 0, np.where(y3_predict == 1, 2, 1))
    else:
        y_results = np.where(y3_predict == 0, 2, np.where(y1_predict == 1, 0, 1))

    print("Total accuracy: ", accuracy(y_results, y_train))

    _classifier = PerceptronClassifier(ppn1, ppn2)

    plot_decision_regions(X = X_train, y = y_train, classifier=_classifier)
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    plt.legend(loc='upper left')
    plt.show()

def accuracy(y_results, y_train):
    return (1 - np.mean(y_results != y_train)) * 100


if __name__ == '__main__':
    main()