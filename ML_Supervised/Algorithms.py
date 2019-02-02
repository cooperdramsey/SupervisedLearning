import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Algorithms:
    def __init__(self, data_path):
        self.data = self.load_data(data_path)

    def decision_tree(self, parameters):
        algorithm = DecisionTreeClassifier()
        clf = GridSearchCV(algorithm, parameters)

    def neural_network(self, parameters):
        algorithm = MLPClassifier()
        clf = GridSearchCV(algorithm, parameters)

    def tree_boosting(self, parameters):
        # uses decision tree as the base algorithm by default
        algorithm = AdaBoostClassifier()
        clf = GridSearchCV(algorithm, parameters)

    def support_vector_machine(self, parameters):
        algorithm = SVC()
        clf = GridSearchCV(algorithm, parameters)

    def k_nearest_neighbors(self, parameters):
        algorithm = KNeighborsClassifier()
        clf = GridSearchCV(algorithm, parameters)

    def load_data(self, data_path):
        return pd.read_csv(data_path)

    def show_data(self):
        print(self.data.head())

    def get_accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
