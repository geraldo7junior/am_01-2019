from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np


class GaussianBayesClassifier:

    def __init__(self, classification_path):
        # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2], [10, 10]])
        self.data_mfeat_fac = self.normalize_data(pd.read_csv('../data_bases/mfeat_fac.csv', header=None))
        self.data_mfeat_fou = self.normalize_data(pd.read_csv('../data_bases/mfeat_fou.csv', header=None))
        self.data_mfeat_kar = self.normalize_data(pd.read_csv('../data_bases/mfeat_kar.csv', header=None))

        # Y = np.array([1, 1, 1, 2, 2, 2, 5])
        self.class_data = np.loadtxt(classification_path, dtype=int)
        self.mfeat_fac_classifier = None
        self.mfeat_fou_classifier = None
        self.mfeat_kar_classifier = None

    def normalize_data(self, data):
        """ normalizes input data
        https://scikit-learn.org/stable/modules/preprocessing.html#normalization"""
        norm = 'l2'
        normalized_data = preprocessing.normalize(data, norm)
        return normalized_data

    def build_classifiers(self):
        self.mfeat_fac_classifier = GaussianNB()
        self.mfeat_fac_classifier.fit(self.data_mfeat_fac, self.class_data)

        self.mfeat_fou_classifier = GaussianNB()
        self.mfeat_fou_classifier.fit(self.data_mfeat_fou, self.class_data)

        self.mfeat_kar_classifier = GaussianNB()
        self.mfeat_kar_classifier.fit(self.data_mfeat_kar, self.class_data)

    def calculate_class_apriori_probability(self, class_name):
        """Calculate apriori probability based on crisp partition provided to the class"""
        unique, counts = np.unique(self.class_data, return_counts=True)
        frequency_dict = dict(zip(unique, counts))
        return float(frequency_dict[class_name])/len(self.class_data)

    def check_overall_probability(self, x, expected_class):
        """Calculates the probability of an example to belong to an expected class for all the views
         available on the classifier"""
        class_probability = self.calculate_class_apriori_probability(expected_class)

        view1_probability = self.mfeat_fac_classifier.predict_proba([self.data_mfeat_fac[x]])[0][expected_class]  # view_probability[numero do exemplo][numero da classe]
        view2_probability = self.mfeat_fou_classifier.predict_proba([self.data_mfeat_fou[x]])[0][expected_class]
        view3_probability = self.mfeat_kar_classifier.predict_proba([self.data_mfeat_kar[x]])[0][expected_class]

        return ((1 - 3) * class_probability) + view1_probability + view2_probability + view3_probability

    def check_max_probability(self, x):
        """Calculates the probability for an example to belong to a class Wr (0-9), returning the class with the
        maximum probability value found"""
        probabilities = dict()
        for r in range(0, 10):
            self.check_overall_probability(x, r)
            probabilities[r] = self.check_overall_probability(x, r)
        return max(probabilities, key=probabilities.get)
