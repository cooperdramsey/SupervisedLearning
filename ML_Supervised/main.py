import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


class Algorithms:
    def __init__(self, data):
        self.data = data

    def decision_tree(self):
        classifier = DecisionTreeClassifier()

    def neural_network(self):
        classifier = MLPClassifier()

    def tree_boosting(self):
        # uses decision tree as the base algorithm by default
        classifier = AdaBoostClassifier()

    def support_vector_machine(self):
        classifier = SVC()

    def k_nearest_neighbors(self):
        classifier = KNeighborsClassifier()
