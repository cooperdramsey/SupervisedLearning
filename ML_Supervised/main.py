from Algorithms import Algorithms

# Data set 1
data_path_1 = 'winequality-red.csv'
tests_1 = Algorithms(data_path_1)
param_grids = {'tree':
                   [{'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'min_samples_leaf':[2, 4, 5, 6]}],
               'neural': [],
               'boost': [],
               'svm': [],
               'knn': []
               }

# Decision tree
tests_1.decision_tree(param_grids['tree'])

# neural network
tests_1.neural_network(param_grids['neural'])

# Boosting
tests_1.tree_boosting(param_grids['boost'])

# SVM
tests_1.support_vector_machine(param_grids['svm'])

# KNN
tests_1.k_nearest_neighbors(param_grids['knn'])

####################################################################################
# Data set 2
data_path_2 = 'UCI_Credit_Card.csv'
tests_2 = Algorithms(data_path_2)

# Decision Tree
tests_2.decision_tree(param_grids['tree'])

# neural network
tests_2.neural_network(param_grids['neural'])

# Boosting
tests_2.tree_boosting(param_grids['boost'])

# SVM
tests_2.support_vector_machine(param_grids['svm'])

# KNN
tests_2.k_nearest_neighbors(param_grids['knn'])
