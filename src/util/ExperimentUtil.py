import statistics
import math
import scipy.stats as st


class ExperimentUtil:

    def __init__(self):
        pass

    # extracted from: http://www2.fm.usp.br/dim/info/wilcoxon/index.php
    wilcox_table = [0, 0, 0, 0, 0, 0, 0, 2, 3, 5, 8, 10, 13, 17, 21, 25]

    @staticmethod
    def _calculate_error_margin(input_data, confidence_coefficient):
        critical_value = st.norm.ppf(1 - (1 - confidence_coefficient) / 2)
        std_error = statistics.stdev(input_data) / math.sqrt(len(input_data))
        return critical_value * std_error
        return error_margin

    @staticmethod
    def calculate_confidence_interval(input_data, confidence_coeficient=0.95):
        """"The method calculates de confidence interval superior and inferior limits
        The code implementation was based on this link: https://pt.wikihow.com/Calcular-o-Intervalo-de-Confian%C3%A7a
        """
        error_margin = ExperimentUtil._calculate_error_margin(input_data, confidence_coeficient)
        superior_limit = statistics.mean(input_data) + error_margin
        inferior_limit = statistics.mean(input_data) - error_margin
        return superior_limit, inferior_limit

    @staticmethod
    def _calculate_differences(classifier1, classifier2):
        differences = []
        sorted_diffs = []
        for index, score in enumerate(classifier1):
            difference = score - classifier2[index]
            if difference:
                differences.append(score - classifier2[index])
                sorted_diffs.append(abs(score - classifier2[index]))
        return differences, sorted_diffs

    @staticmethod
    def _calculate_position_differences(differences, sorted_diffs):
        position_diffs = []
        for index, diff in enumerate(differences):
            count = sorted_diffs.count(abs(diff))
            if count > 1:
                median = sorted_diffs.index(abs(diff)) + 1
                for i in range(1, count):
                    median += sorted_diffs.index(abs(diff)) + i + 1
                    position_diffs.append(float(median)/count)
            else:
                position_diffs.append(sorted_diffs.index(abs(diff)) + 1)
        return position_diffs

    @staticmethod
    def _calculate_positive_negative_sum(position_diffs):
        sum_positive = 0
        sum_negative = 0
        for score in position_diffs:
            if score < 0:
                sum_negative += abs(score)
            else:
                sum_positive += score
        return sum_positive, sum_negative

    @staticmethod
    def perform_wilcoxon_validation(series1, series2):
        """This method performs Wilcoxon validation. The boolean output means that the null hypothesis
        was rejected or not. Implemented following the steps on http://www.leg.ufpr.br/lib/exe/fetch.php/disciplinas:ce001:wilcoxon_1_.pdf"""
        differences, sorted_diffs = ExperimentUtil._calculate_differences(series1, series2)
        sorted_diffs.sort()
        position_diffs = ExperimentUtil._calculate_position_differences(differences, sorted_diffs)

        for index, score in enumerate(differences):
            if score < 0:
                position_diffs[index] = position_diffs[index] * -1

        sum_positive, sum_negative = ExperimentUtil._calculate_positive_negative_sum(position_diffs)
        T = min(sum_positive, sum_negative)
        # TODO: Se o tamanho de n for maior que 30, é preciso usar a tabela T-Student
        if len(position_diffs) <= 30:
            # TODO: Com o valor de T, precisamos ver qual o valor critico e elaborar melhor a resposta no relatório
            return T < ExperimentUtil.wilcox_table[len(position_diffs)]











