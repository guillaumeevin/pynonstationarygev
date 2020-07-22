import numpy as np

from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit
from extreme_fit.model.utils import r


def cramer_von_mises_and_anderson_darling_tests_pvalues_for_gumbel_distribution(data):
    res = r.gnfit_fixed(data, "gum")
    res = AbstractResultFromModelFit.get_python_dictionary(res)
    res = {k: np.array(v)[0] for k, v in res.items()}
    return res['Wpval'], res['Apval']


def goodness_of_fit_anderson(quantiles, significance_level=0.05):
    test = cramer_von_mises_and_anderson_darling_tests_pvalues_for_gumbel_distribution(quantiles)
    _, ander_darling_test_pvalue = test
    return ander_darling_test_pvalue > significance_level


if __name__ == '__main__':
    cramer_von_mises_and_anderson_darling_tests_pvalues_for_gumbel_distribution(np.arange(50))
