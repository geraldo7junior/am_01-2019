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
    list_1 = [30, 19, 19, 23, 29, 178, 42 , 20, 12, 39, 14, 81, 17, 31, 52]
    list_2 = [30, 6, 14, 8, 14, 52, 14, 22, 17, 8, 11, 30, 14, 17, 15]
    ExperimentUtil.perform_wilcoxon_validation(list_1, list_2)
    # ExperimentUtil.calculate_error_margin(list(), 0.95)
    # bkc = BayesianKneighborClassifier('../data_bases/daniel/crisp-partition.txt')
    # gbc = GaussianBayesClassifier('../data_bases/daniel/crisp-partition.txt')
    # random_index = randint(0, 2000)

    #bkc.check_max_probability(random_index)


if __name__ == '__main__':
    main()
