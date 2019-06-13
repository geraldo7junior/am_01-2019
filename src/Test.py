import numpy as np
import pandas as pd
from classifiers.BayesianKNNClassifier import BayesianKNNClassifier
from classifiers.GaussianBayesClassifier import GaussianBayesClassifier
from util.ExperimentUtil import ExperimentUtil
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn import metrics
import statistics


def normalize_data(data, norm='l2'):
    """ normalizes input data
    https://scikit-learn.org/stable/modules/preprocessing.html#normalization"""
    normalized_data = preprocessing.normalize(data, norm)
    return normalized_data


def main():

    gaussian_scores = [0.7145, 0.7015, 0.7025, 0.715, 0.716, 0.6985, 0.6915, 0.6955, 0.7035, 0.7095, 0.6955, 0.7225, 0.6985, 0.7015, 0.7085, 0.716, 0.7035, 0.708, 0.6975, 0.716, 0.709, 0.7005, 0.708, 0.699, 0.6930000000000001, 0.701, 0.7095, 0.704, 0.7245, 0.709]
    knn_scores = [0.786, 0.7775000000000001, 0.781, 0.7845, 0.7725, 0.772, 0.771, 0.77, 0.775, 0.763, 0.766, 0.792, 0.7735, 0.763, 0.77, 0.78, 0.768, 0.7655, 0.7735000000000001, 0.7705, 0.778, 0.776, 0.775, 0.756, 0.7785, 0.771, 0.7765, 0.7705, 0.781, 0.776]
    #ExperimentUtil.calculate_confidence_interval(gaussian_scores, 0.95)
    #ExperimentUtil.calculate_confidence_interval(knn_scores, 0.95)
    #ExperimentUtil.perform_wilcoxon_validation(knn_scores, gaussian_scores)

    ExperimentUtil.wilcoxon(knn_scores, gaussian_scores)


    data_mfeat_fac = normalize_data(pd.read_csv('../data_bases/mfeat_fac.csv', header=None))
    data_mfeat_fou = normalize_data(pd.read_csv('../data_bases/mfeat_fou.csv', header=None))
    data_mfeat_kar = normalize_data(pd.read_csv('../data_bases/mfeat_kar.csv', header=None))
    data_class = np.loadtxt('../data_bases/daniel/crisp-partition.txt', dtype=int)

    kf = KFold(n_splits=10, shuffle=True)
    sss = StratifiedShuffleSplit(n_splits=10)
    medias_bkc = []
    medias_gbc = []
    for x in range(30):
        acc_bkc = []
        acc_gbc = []
        for train_index, test_index in sss.split(data_mfeat_fac, data_class):
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


def best_J():
    count = 0
    indices = []
    for i in range(100):
        teste = np.loadtxt('../saidas/Execucao-{}-crisp-partition.txt'.format(i+1), dtype=int)
        if len(set(teste)) == 10:
            count += 1
            indices.append(i)
    print(count)
    print(indices)
    jotas = []
    for i in indices:
        file = open('../saidas/Execucao-{}-output.txt'.format(i+1), "r")
        jotas.append(float(file.readline()[4:-1]))
        file.close()
    print(min(jotas))
    # best J = 1240.8175952498866


if __name__ == '__main__':
    main()
