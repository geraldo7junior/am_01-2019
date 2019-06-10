import numpy as np
import pandas as pd
from classifiers.BayesianKNNClassifier import BayesianKNNClassifier
from classifiers.GaussianBayesClassifier import GaussianBayesClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import metrics
import statistics


def normalize_data(data, norm='l2'):
    """ normalizes input data
    https://scikit-learn.org/stable/modules/preprocessing.html#normalization"""
    normalized_data = preprocessing.normalize(data, norm)
    return normalized_data


def main():

    # list_1 = [30, 19, 19, 23, 29, 178, 42, 20, 12, 39, 14, 81, 17, 31, 52]
    # list_2 = [30, 6, 14, 8, 14, 52, 14, 22, 17, 8, 11, 30, 14, 17, 15]
    # ExperimentUtil.perform_wilcoxon_validation(list_1, list_2)
    # ExperimentUtil.calculate_error_margin(list(), 0.95)

    data_mfeat_fac = normalize_data(pd.read_csv('../data_bases/mfeat_fac.csv', header=None))
    data_mfeat_fou = normalize_data(pd.read_csv('../data_bases/mfeat_fou.csv', header=None))
    data_mfeat_kar = normalize_data(pd.read_csv('../data_bases/mfeat_kar.csv', header=None))
    data_class = np.loadtxt('../data_bases/daniel/crisp-partition.txt', dtype=int)

    kf = KFold(n_splits=10, shuffle=True)
    medias_bkc = []
    medias_gbc = []
    for x in range(30):
        acc_bkc = []
        acc_gbc = []
        for train_index, test_index in kf.split(data_mfeat_fac):
            X1_train, X1_test = data_mfeat_fac[train_index], data_mfeat_fac[test_index]
            X2_train, X2_test = data_mfeat_fou[train_index], data_mfeat_fou[test_index]
            X3_train, X3_test = data_mfeat_kar[train_index], data_mfeat_kar[test_index]

            y_train, y_test = data_class[train_index], data_class[test_index]
            # Treinamento
            bkc = BayesianKNNClassifier(X1_train, X2_train, X3_train, y_train)
            gbc = GaussianBayesClassifier(X1_train, X2_train, X3_train, y_train)

            # Accuracy
            bkc_predict_y = bkc.predict(X1_test, X2_test, X3_test)
            gbc_predict_y = gbc.predict(X1_test, X2_test, X3_test)

            acc_bkc.append(metrics.accuracy_score(y_test, bkc_predict_y))
            acc_gbc.append(metrics.accuracy_score(y_test, gbc_predict_y))
        medias_bkc.append(statistics.mean(acc_bkc))
        medias_gbc.append(statistics.mean(acc_gbc))
    print("BayesianKNNClassifier: ", medias_bkc)
    print("GaussianBayesClassifier: ", medias_gbc)


if __name__ == '__main__':
    main()
