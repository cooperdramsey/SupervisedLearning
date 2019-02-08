import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
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
        algorithm = SVC()
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

    # function from Scikit learn url:
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like, shape (n_ticks,), dtype float or int
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the dtype is float, it is regarded as a
            fraction of the maximum size of the training set (that is determined
            by the selected validation method), i.e. it has to be within (0, 1].
            Otherwise it is interpreted as absolute sizes of the training sets.
            Note that for classification the number of samples usually have to
            be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt

    X, y = digits.data, digits.target

    title = "Learning Curves (Naive Bayes)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = GaussianNB()
    plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = SVC(gamma=0.001)
    plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

    plt.show()