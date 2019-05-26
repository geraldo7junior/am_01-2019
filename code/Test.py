from BayesianKneighborClassifier import BayesianKneighborClassifier
from random import randint


def main():
    bkc = BayesianKneighborClassifier('../data_bases/daniel/crisp-partition.txt')
    random_data = list()
    random_index = randint(0, 2000)
    random_data.append(bkc.return_view_sample(0, random_index))
    random_data.append(bkc.return_view_sample(1, random_index))
    random_data.append(bkc.return_view_sample(2, random_index))
    bkc.check_max_probability(random_data)


if __name__ == '__main__':
    main()