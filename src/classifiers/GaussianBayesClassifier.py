from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np


class GaussianBayesClassifier:

    def __init__(self, X1, X2, X3, Y):
        # Y = np.array([1, 1, 1, 2, 2, 2, 5])
        self.y = Y
        self.mfeat_fac_classifier = self.build_classifier(X1, Y)
        self.mfeat_fou_classifier = self.build_classifier(X2, Y)
        self.mfeat_kar_classifier = self.build_classifier(X3, Y)

        unique, counts = np.unique(Y, return_counts=True)
        self.frequency_dict = dict(zip(unique, counts))
        self.y = Y

    @staticmethod
    def build_classifier(data_in, data_class):
        gbc = GaussianNB()
        gbc.fit(data_in, data_class)
        return gbc

    def check_overall_probability(self, x1, x2, x3, expected_class):
        """Calculates the probability of an example to belong to an expected class for all the views
         available on the classifier"""
        class_probability = self.frequency_dict.get(expected_class, 0)/len(self.y)

        # view_probability[numero do exemplo][numero da classe]
        view1_probability = self.mfeat_fac_classifier.predict_proba([x1])[0][expected_class]
        view2_probability = self.mfeat_fou_classifier.predict_proba([x2])[0][expected_class]
        view3_probability = self.mfeat_kar_classifier.predict_proba([x3])[0][expected_class]

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
