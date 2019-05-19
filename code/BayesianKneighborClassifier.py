from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import math
import pandas as pd


class BayesianKneighborClassifier:
    """Combined classifier based on the sum of Bayesian k-neighbour based classifier"""

    def __init__(self, ksize, views, data_path):
        self.neighbors = ksize
        self.views = views
        self.data_path = data_path
        self.data = self.read_file()

    @staticmethod
    def normalize_data(data, norm='l2'):
        """ normalizes input data
        https://scikit-learn.org/stable/modules/preprocessing.html#normalization"""
        normalized_data = preprocessing.normalize(data, norm)
        return normalized_data

    def read_file(self):
        """Extract the content from a file into a dataframe"""
        df = pd.read_fwf(self.data_path)
        return df

    @staticmethod
    def euclidean_distance(instance1, instance2, length):
        dist = 0
        for x in range(length):
            dist += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(dist)

    @staticmethod
    def split_test_and_train_data(input_data, test_size=0.7):
        """Separate the data into train and test according to the proportion"""
        train, test = train_test_split(input_data, test_size=test_size)
        return train, test

    def run_knn_pipeline(self, train_data, test_data):
        knn = KNeighborsClassifier(n_neighbors=self.neighbors)
        knn_pipe = Pipeline([('knn', knn)])
        knn_pipe.fit(train_data)
        print(knn_pipe.score(test_data))



