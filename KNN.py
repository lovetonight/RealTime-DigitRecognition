import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority voye
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
# import numpy as np
# import operator
# from operator import itemgetter
# from sklearn.datasets import load_digits
# from keras.datasets import mnist
#
# (X_train, y_train), (m_test, n_test) = mnist.load_data()
# def euc_dist(x1, x2):
#     return np.sqrt(np.sum((x1-x2)**2))
#
#
# class KNN:
#     def __init__(self, K=3):
#         self.K = K
#         self.X_train = m_test
#         self.Y_train = n_test
#
#     def predict(self, X_test):
#         predictions = []
#         for i in range(len(X_test)):
#             dist = np.array([euc_dist(X_test[i], x_t) for x_t in
#                              self.X_train])
#             dist_sorted = dist.argsort()[:self.K]
#             neigh_count = {}
#             for idx in dist_sorted:
#                 if self.Y_train[idx] in neigh_count:
#                     neigh_count[self.Y_train[idx]] += 1
#                 else:
#                     neigh_count[self.Y_train[idx]] = 1
#             sorted_neigh_count = sorted(neigh_count.items(),
#                                         key=operator.itemgetter(1), reverse=True)
#             predictions.append(sorted_neigh_count[0][0])
#         return predictions
