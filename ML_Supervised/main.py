from Algorithms import Algorithms
from timeit import default_timer as timer

# init
param_grids = {'tree':
                   [{'criterion': ['gini', 'entropy'],
                     'splitter': ['best', 'random'],
                     'min_samples_leaf':[1, 2, 4, 5, 6],
                     'min_samples_split': [2]
                     }],
               'neural': [{'hidden_layer_sizes': [(100,)],
                           'activation': ['relu'], #'identity', 'logistic', 'tanh',
                           'alpha': [0.0001],
                           'learning_rate': ['constant'], # 'constant', 'invscaling', 'adaptive'
                           'learning_rate_init': [0.001],
                           'max_iter': [2000],
                           }],
               'boost': [{'n_estimators': [50, 25],
                          'learning_rate': [1, 0.5]
                          }],
               'svm': [{'C': [1],  # just add zeros
                        'kernel': ['linear'], # rbf
                        'gamma': ['auto'] # 0.0001
                        }],
               'knn': [{'n_neighbors': [5, 8, 10]}],
               'linearSVM': [{'C': [1],
                              'gamma': ['scale']}]
               }
test_split = 0.4

# Data set 1
data_path_1 = 'winequality-red.csv'
tests_1 = Algorithms(data_path_1, test_split)

print("DATA SET 1")
# Decision tree
start = timer()
tests_1.decision_tree(param_grids['tree'])
end = timer()
print(end - start)

# neural network
start = timer()
tests_1.neural_network(param_grids['neural'])
end = timer()
print(end - start)

# Boosting
start = timer()
tests_1.tree_boosting(param_grids['boost'])
end = timer()
print(end - start)

# KNN
start = timer()
tests_1.k_nearest_neighbors(param_grids['knn'])
end = timer()
print(end - start)

# Linear SVM
start = timer()
tests_1.linear_support_vector_machine(param_grids['linearSVM'])
end = timer()
print(end - start)

# SVM
start = timer()
tests_1.support_vector_machine(param_grids['svm'])
end = timer()
print(end - start)

####################################################################################
print()
print('DATA SET 2')
# Data set 2
data_path_2 = 'UCI_Credit_Card.csv'
tests_2 = Algorithms(data_path_2, test_split)

# Decision tree
start = timer()
tests_2.decision_tree(param_grids['tree'])
end = timer()
print(end - start)

# neural network
start = timer()
tests_2.neural_network(param_grids['neural'])
end = timer()
print(end - start)

# Boosting
start = timer()
tests_2.tree_boosting(param_grids['boost'])
end = timer()
print(end - start)

# KNN
start = timer()
tests_2.k_nearest_neighbors(param_grids['knn'])
end = timer()
print(end - start)

# # Linear SVM
# start = timer()
# tests_2.linear_support_vector_machine(param_grids['linearSVM'])
# end = timer()
# print(end - start)
#
# # SVM
# start = timer()
# tests_2.support_vector_machine(param_grids['svm'])
# end = timer()
# print(end - start)
