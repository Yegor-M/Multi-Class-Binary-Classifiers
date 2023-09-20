from reglog import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions

class MultiClassLogisticRegression:
    def __init__(self, classes):
        self.classes = classes
        self.classifiers = {}

    def fit(self, X, y):
        for cls in self.classes:
            y_binary = np.where(y == cls, 1, 0)
            classifier = LogisticRegression()
            classifier.fit(X, y_binary)
            self.classifiers[cls] = classifier

    def predict(self, X):
        predictions = []
        for cls in self.classes:
            classifier = self.classifiers[cls]
            predictions.append(classifier.predict(X))
        return np.argmax(predictions, axis=0)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def calculate_probabilities(self, X):
        probabilities = np.zeros((X.shape[0], len(self.classes)))
        for i, cls in enumerate(self.classes):
            classifier = self.classifiers[cls]
            probabilities[:, i] = classifier.activation(classifier.net_input(X))
        return probabilities
        
def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true) * 100
    
def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    classes = np.unique(y_train)
    classifier = MultiClassLogisticRegression(classes)
    classifier.fit(X_train, y_train)

    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)

    accuracy_train = accuracy(y_pred_train, y_train)
    accuracy_test = accuracy(y_pred_test, y_test)
    print("\nTraining accuracy:", accuracy_train)
    print("\nTesting accuracy: ", accuracy_test)

    probabilities_train = classifier.calculate_probabilities(X_train)
    probabilities_test = classifier.calculate_probabilities(X_test)
    print("\nProbabilities for Training Set: ", probabilities_train)
    print("\nProbabilities for Testing Set: ", probabilities_test)

    plot_decision_regions(X=X_train, y=y_train, classifier=classifier)
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()