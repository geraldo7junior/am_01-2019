from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np
from sklearn import metrics


def find_best_k(data_in, data_class):
    """Calculates the accuracy for the classifiers, to find the best value of K"""
    X_train, X_val, y_train, y_val = train_test_split(data_in, data_class, test_size=0.2)
    k_options = [x for x in range(50) if x % 2 != 0]
    k = k_options[0]
    best_acc = 0
    for neighbors in k_options:
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        knn.fit(X_train, y_train)
        y_predicted = knn.predict(X_val)
        acc = metrics.accuracy_score(y_val, y_predicted)
        if acc > best_acc:
            best_acc = acc
            k = neighbors
        # print("Accuracy: {} using K={} neighbours in view:".format(acc, neighbors))
    # print("Best accuracy: {}, best K: {}".format(best_acc, k))
    return k


class BayesianKNNClassifier:
    """Combined classifier based on the combination of Bayesian theorem and k-neighbours algorithm"""

    def __init__(self, X1, X2, X3, Y):
        """"The values used on the build classifier were defined after some experiments with the data and the KNN classifier
        which ended up with these values as the best"""

        self.k1 = find_best_k(X1, Y)
        self.k2 = find_best_k(X2, Y)
        self.k3 = find_best_k(X3, Y)
        self.mfeat_fac_classifier = self.build_classifier(self.k1, X1, Y)
        self.mfeat_fou_classifier = self.build_classifier(self.k2, X2, Y)
        self.mfeat_kar_classifier = self.build_classifier(self.k3, X3, Y)

        unique, counts = np.unique(Y, return_counts=True)
        self.frequency_dict = dict(zip(unique, counts))
        self.y = Y

    @staticmethod
    def build_classifier(k, data_in, data_class):
        """Builds Knn classifier"""
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(data_in, data_class)
        return knn

    def check_probability(self, x, expected_class, classifier):
        """Calculates the probability of an example to belong to an expected class, it will use the
        k-nearest neighbours to get the closest items from it and check if they belong to the class expected
        the final output will be a probability of the number of neighbours in the class expected divided by
        the number of neighbours checked"""
        match_number = 0
        indexes = classifier.kneighbors(x.reshape(1, -1), return_distance=False)
        for idx in indexes:
            for element in idx:
                if self.y[element] == expected_class:
                    match_number += 1
        return float(match_number)/len(indexes)

    def check_overall_probability(self, x1, x2, x3, expected_class):
        """Calculates the probability of an example to belong to an expected class for all the views"""
        class_probability = self.frequency_dict.get(expected_class, 0)/len(self.y)

        view1_probability = self.check_probability(x1, expected_class, self.mfeat_fac_classifier)
        view2_probability = self.check_probability(x2, expected_class, self.mfeat_fou_classifier)
        view3_probability = self.check_probability(x3, expected_class, self.mfeat_kar_classifier)

        return ((1 - 3) * class_probability) + view1_probability + view2_probability + view3_probability

    def check_max_probability(self, x1, x2, x3):
        """Calculates the probability for an example to belong to a class Wr (0-9), returning the class with the
        maximum probability value found"""
        probabilities = dict()
        classes = list(set(self.y))
        classes.sort()
        for r in range(len(classes)):
            probabilities[classes[r]] = self.check_overall_probability(x1, x2, x3, r)
        return max(probabilities, key=probabilities.get)

    def predict(self, X1, X2, X3):
        predict_y = []
        for i in range(len(X1)):
            predict_y.append(self.check_max_probability(X1[i], X2[i], X3[i]))
        return predict_y
