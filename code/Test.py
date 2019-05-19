from BayesianKneighborClassifier import BayesianKneighborClassifier
import numpy as np


def main():
    bkc = BayesianKneighborClassifier(3, 3, '../data_bases/mfeat-fac')
    df = bkc.read_file()
    #TODO: Check why np.array is removing the first line of the dataframe
    np_array = np.array(df)
    normalized_l2 = bkc.normalize_data(np_array)
    train, test = bkc.split_test_and_train_data(normalized_l2)
    bkc.run_knn_pipeline(train, test)


if __name__ == '__main__':
    main()