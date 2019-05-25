from BayesianKneighborClassifier import BayesianKneighborClassifier
import numpy as np


def main():
    bkc = BayesianKneighborClassifier('../data_bases/daniel/crisp-partition.txt')


    #rain, test = bkc.split_test_and_train_data(normalized_l2)
    #bkc.run_knn_pipeline(train, test)


if __name__ == '__main__':
    main()