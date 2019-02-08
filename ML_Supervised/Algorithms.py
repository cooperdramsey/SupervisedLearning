import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Algorithms:
    def __init__(self, data_path, test_split):
        self.data = self.load_data(data_path)
        self.test_split = test_split

    def decision_tree(self, parameters):
        algorithm = DecisionTreeClassifier()
        clf = GridSearchCV(algorithm, parameters, cv=5, iid=False)
        X_train, X_test, y_train, y_test = self.split_data(self.test_split)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores = self.get_scores(y_test, y_pred)
        print("Decision Tree: ",
              "Acc: " + str(scores['acc'] * 100) + "%")
        print(clf.best_params_.keys())
        print(clf.best_params_.items())

    def neural_network(self, parameters):
        algorithm = MLPClassifier()
        clf = GridSearchCV(algorithm, parameters, cv=5, iid=False)
        X_train, X_test, y_train, y_test = self.split_data(self.test_split)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores = self.get_scores(y_test, y_pred)
        print("Neural Net: ",
              "Acc: " + str(scores['acc'] * 100) + "%")
        print(clf.best_params_.keys())
        print(clf.best_params_.items())

    def tree_boosting(self, parameters):
        # uses decision tree as the base algorithm by default
        algorithm = AdaBoostClassifier()
        clf = GridSearchCV(algorithm, parameters, cv=5, iid=False)
        X_train, X_test, y_train, y_test = self.split_data(self.test_split)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores = self.get_scores(y_test, y_pred)
        print("AdaBoost:",
              "Acc: " + str(scores['acc'] * 100) + "%")
        print(clf.best_params_.keys())
        print(clf.best_params_.items())

    def support_vector_machine(self, parameters):
        algorithm = SVC()
        clf = GridSearchCV(algorithm, parameters, cv=5, iid=False)
        X_train, X_test, y_train, y_test = self.split_data(self.test_split)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores = self.get_scores(y_test, y_pred)
        print("SVM:",
              "Acc: " + str(scores['acc'] * 100) + "%")
        print(clf.best_params_.keys())
        print(clf.best_params_.items())

    def linear_support_vector_machine(self, parameters):
        algorithm = LinearSVC()
        clf = GridSearchCV(algorithm, parameters, cv=5, iid=False)
        X_train, X_test, y_train, y_test = self.split_data(self.test_split)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores = self.get_scores(y_test, y_pred)
        print("Linear SVM:",
              "Acc: " + str(scores['acc'] * 100) + "%")
        print(clf.best_params_.keys())
        print(clf.best_params_.items())

    def k_nearest_neighbors(self, parameters):
        algorithm = KNeighborsClassifier()
        clf = GridSearchCV(algorithm, parameters, cv=5, iid=False)
        X_train, X_test, y_train, y_test = self.split_data(self.test_split)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores = self.get_scores(y_test, y_pred)
        print("KNN:",
              "Acc: " + str(scores['acc'] * 100) + "%")
        print(clf.best_params_.keys())
        print(clf.best_params_.items())

    def load_data(self, data_path):
        return pd.read_csv(data_path)

    def show_data(self):
        print(self.data.head())

    def get_scores(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        #prec = precision_score(y_true, y_pred)
        #recall = recall_score(y_true, y_pred)
        #f1 = f1_score(y_true, y_pred)
        return {'acc': acc
                #'prec': prec,
                #'recall': recall,
                #'f1': f1
                }

    def split_data(self, test_size):
        y = self.data.iloc[:, -1]
        X = self.data.iloc[:, :-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test
