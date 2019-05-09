from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        self.features = features
        self.labels = labels
        #raise NotImplementedError

    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        #import operator
        from collections import Counter
        predicted_labels = []
        for instance in features:
            k_nearest = self.get_k_neighbors(instance)
            majority = {}
            for neigh in k_nearest:
                if neigh[1] not in majority.keys():
                    majority[neigh[1]] = 1
                else:
                    majority[neigh[1]] += 1
            #majority_label = max(majority.items(), key = operator.itemgetter(1))[0]
            majority_label = Counter(majority).most_common()[0][0]
            predicted_labels.append(majority_label)
        return predicted_labels
        #raise NotImplementedError

    #TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        #import operator
        distances = []
        for train_feature, train_label in zip(self.features, self.labels):
            dist = self.distance_function(point, train_feature)
            distances.append([dist, train_label])
        distances = sorted(distances)
        #distances.sort(key = operator.itemgetter(0))
        return distances[:self.k]
        #raise NotImplementedError


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
