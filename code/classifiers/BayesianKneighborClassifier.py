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
        self.data = list()
        self.class_data = np.loadtxt(classification_path, dtype=int)
        self.mfeat_fac_classifier = self.build_classifier(15, 0)
        self.mfeat_fou_classifier = self.build_classifier(13, 1)
        self.mfeat_kar_classifier = self.build_classifier(13, 2)

    views = ['mfeat_fac', 'mfeat_fou', 'mfeat_kar']
    best_neighbours = [15, 13, 13]

    def update_current_data(self, index):
        self.data.append(self.normalize_data(pd.read_csv('../data_bases/' + self.views[index] + '.csv', header=None)))

    @staticmethod
    def return_view_sample(view_index, example_index):
        return BayesianKneighborClassifier.normalize_data(
            pd.read_csv('../data_bases/' +
                        BayesianKneighborClassifier.views[view_index] + '.csv', header=None))[example_index]

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
    def split_test_and_train_data(self, test_size=0.3, view=0):
        """Separate the data into train and test according to the proportion"""
        X_train, X_test, y_train, y_test = train_test_split(self.data[view], self.class_data, test_size=test_size)
        return X_train, X_test, y_train, y_test

    def split_train_test_application_data(self, test_size, test_split_proportion, view=0):
        """Separate the data into train, test and application according to the proportions"""
        X_train, X_test, y_train, y_test = train_test_split(self.data[view], self.class_data, test_size=test_size)
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

    def build_classifier(self, n_neighbours, data_index):
        """Builds 3 Knn classifiers, one for each view with fixed number of K based on previous experiments"""
        knn = KNeighborsClassifier(n_neighbors=n_neighbours)
        BayesianKneighborClassifier.update_current_data(self, data_index)
        X_train, X_test, y_train, y_test = BayesianKneighborClassifier.split_test_and_train_data\
            (self, 0.3, data_index)
        knn.fit(X_train, y_train)
        y_predicted = knn.predict(X_test)
        print("KNN classifier built. Accuracy score: {} using K={} neighbours in view: {}".format(
            metrics.accuracy_score(y_test, y_predicted), n_neighbours,
            BayesianKneighborClassifier.views[data_index]))
        return knn

    def check_knn_accuracy(self, neighbors=5, view_index=0):
        """Calculates the accuracy for the classifiers, in order to find the best values of K"""
        BayesianKneighborClassifier.update_current_data(self, view_index)
        X_train, X_test, X_application, y_train, y_test, y_application = \
            BayesianKneighborClassifier.split_train_test_application_data(self, test_size=0.3,
                                                                          test_split_proportion=0.5, view=view_index)
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        knn.fit(X_train, y_train)
        y_predicted = knn.predict(X_application)
        print("Accuracy: {} using K={} neighbours in view: {}".format(
            metrics.accuracy_score(y_application, y_predicted), neighbors,
            BayesianKneighborClassifier.views[view_index]))
        return metrics.accuracy_score(y_application, y_predicted)

    def check_probability(self, x, k_neighbours, expected_class, classifier, view = 0):
        """Calculates the probability of an example to belong to an expected class, it will use the
        k-nearest neighbours to get the closest items from it and check if they belong to the class expected
        the final output will be a probability of the number of neighbours in the class expected divided by
        the number of neighbours checked"""
        match_number = 0
        distances, indexes = classifier.kneighbors(x.reshape(1, -1), k_neighbours)
        for idx in indexes:
            for element in idx:
                predict = classifier.predict(self.data[view][element].reshape(1, -1))
                if predict[0] and predict[0] == expected_class:
                    match_number += 1
        return float(match_number)/k_neighbours

    def check_overall_probability(self, x, expected_class):
        """Calculates the probability of an example to belong to an expected class for all the views
         available on the classifier"""
        class_probability = self.calculate_class_apriori_probability(expected_class)
        view1_probability = self.check_probability(x[0], self.best_neighbours[0], expected_class, self.mfeat_fac_classifier, 0)
        view2_probability = self.check_probability(x[1], self.best_neighbours[1], expected_class, self.mfeat_fou_classifier, 1)
        view3_probability = self.check_probability(x[2], self.best_neighbours[2], expected_class, self.mfeat_kar_classifier, 2)
        return ((1 - 3) * class_probability) + view1_probability + view2_probability + view3_probability

    def check_max_probability(self, x):
        """Calculates the probability for an example to belong to a class Wr (0-9), returning the class with the
        maximum probability value found"""
        probabilities = dict()
        for r in range(0, 10):
            self.check_overall_probability(x, r)
            probabilities[r] = self.check_overall_probability(x, r)
        return max(probabilities, key=probabilities.get)










