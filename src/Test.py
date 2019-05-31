from classifiers.GaussianBayesClassifier import GaussianBayesClassifier
from classifiers.BayesianKneighborClassifier import BayesianKneighborClassifier
from random import randint
from util.ExperimentUtil import ExperimentUtil


def main():
    # bkc = BayesianKneighborClassifier('../data_bases/daniel/crisp-partition.txt')
    # random_data = list()
    # random_index = randint(0, 2000)
    # random_data.append(bkc.return_view_sample(0, random_index))
    # random_data.append(bkc.return_view_sample(1, random_index))
    # random_data.append(bkc.return_view_sample(2, random_index))
    # bkc.check_max_probability(random_data)
    ExperimentUtil.calculate_error_margin(list(), 0.95)
    bkc = BayesianKneighborClassifier('../data_bases/daniel/crisp-partition.txt')
    gbc = GaussianBayesClassifier('../data_bases/daniel/crisp-partition.txt')
    random_index = randint(0, 2000)

    bkc.check_max_probability(random_index)


if __name__ == '__main__':
    main()
