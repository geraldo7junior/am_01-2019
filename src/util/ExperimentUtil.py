import statistics
import math
import scipy.stats as st


class ExperimentUtil:

    @staticmethod
    def calculate_error_margin(input_data, confidence_coefficient):
        critical_value = st.norm.ppf(1 - (1 - confidence_coefficient) / 2)
        std_error = statistics.stdev(input_data) / math.sqrt(len(input_data))
        return critical_value * std_error
        return error_margin

    @staticmethod
    def calculate_confidence_interval(input_data, confidence_coeficient=0.95):
        """"The method calculates de confidence interval superior and inferior limits
        The code implementation was based on this link: https://pt.wikihow.com/Calcular-o-Intervalo-de-Confian%C3%A7a
        """
        error_margin = ExperimentUtil.calculate_error_margin(input_data, confidence_coeficient)
        superior_limit = statistics.mean(input_data) + error_margin
        inferior_limit = statistics.mean(input_data) - error_margin
        return superior_limit, inferior_limit

