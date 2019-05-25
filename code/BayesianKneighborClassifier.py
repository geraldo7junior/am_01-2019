from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np
from sklearn import metrics


class BayesianKneighborClassifier:
    """Combined classifier based on the combination of Bayesian theorem and k-neighbours algorithm"""

    def __init__(self, classification_path):
        self.data = []
        self.class_data = np.loadtxt(classification_path, dtype=int)
        self.classifiers = self.build_classifiers()

    views = ['mfeat_fac', 'mfeat_fou', 'mfeat_kar']
    best_neighbours = [15, 13, 13]

    def update_current_data(self, index):
        self.data = BayesianKneighborClassifier.normalize_data(pd.read_csv('../data_bases/' + BayesianKneighborClassifier.views[index] + '.csv', header=None))

    @staticmethod
    def normalize_data(data, norm='l2'):
        """ normalizes input data
        https://scikit-learn.org/stable/modules/preprocessing.html#normalization"""
        normalized_data = preprocessing.normalize(data, norm)
        return normalized_data

    @staticmethod
    def euclidean_distance(instance1, instance2, length):
        dist = 0
        for x in range(length):
            dist += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(dist)

    @staticmethod
    def split_test_and_train_data(self, test_size=0.3):
        """Separate the data into train and test according to the proportion"""
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.class_data, test_size=test_size)
        return X_train, X_test, y_train, y_test

    def split_train_test_application_data(self, test_size, test_split_proportion):
        """Separate the data into train, test and application according to the proportions"""
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.class_data, test_size=test_size)
        X_test, X_application, y_test, y_application = train_test_split(X_test, y_test, test_size=test_split_proportion)
        return X_train, X_test, X_application, y_train, y_test, y_application

    def calculate_class_apriori_probability(self, class_name):
        """Calculate apriori probability based on crisp partition provided to the class"""
        unique, counts = np.unique(self.class_data, return_counts=True)
        frequency_dict = dict(zip(unique, counts))
        return float(frequency_dict[class_name])/len(self.class_data)

    def build_classifiers(self):
        """Builds 3 Knn classifiers, one for each view with fixed number of K based on previous experiments"""
        classifiers = []
        for index, view in enumerate(BayesianKneighborClassifier.views):
            knn = KNeighborsClassifier(n_neighbors=BayesianKneighborClassifier.best_neighbours[index])
            BayesianKneighborClassifier.update_current_data(self, index)
            X_train, X_test, y_train, y_test = BayesianKneighborClassifier.split_test_and_train_data\
                (self, test_size=0.3)
            knn.fit(X_train, y_train)
            classifiers.append(knn)
        return classifiers

    def check_knn_accuracy(self, neighbors=5, view_index=0):
        """Calculates the accuracy for the classifiers, in order to find the best values of K"""
        BayesianKneighborClassifier.update_current_data(self, view_index)
        X_train, X_test, X_application, y_train, y_test, y_application = \
            BayesianKneighborClassifier.split_train_test_application_data(self, test_size=0.3, test_split_proportion=0.5)
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_application)
        print("Accuracy: {} using K={} neighbours in view: {}".format(
            metrics.accuracy_score(y_application, y_pred), neighbors,
            BayesianKneighborClassifier.views[view_index]))
        return metrics.accuracy_score(y_application, y_pred)



