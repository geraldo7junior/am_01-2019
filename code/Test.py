from BayesianKneighborClassifier import BayesianKneighborClassifier


def main():
    bkc = BayesianKneighborClassifier('../data_bases/daniel/crisp-partition.txt')
    for data in bkc.data:
            probability = bkc.check_probability(data, bkc.mfeat_fac_classifier.n_neighbors, i, bkc.mfeat_fac_classifier)
            print (probability)


if __name__ == '__main__':
    main()