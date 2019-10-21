from multiprocessing.pool import Pool

import numpy as np

from experiment.trend_analysis.univariate_test.abstract_gev_trend_test import AbstractGevTrendTest
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from root_utils import NB_CORES


def compute_gev_change_point_test_result(smooth_maxima, starting_year, trend_test_class, years, fit_method=AbstractTemporalLinearMarginModel.ISMEV_GEV_FIT_METHOD_STR):
    trend_test = trend_test_class(years, smooth_maxima, starting_year)  # type: AbstractGevTrendTest
    assert isinstance(trend_test, AbstractGevTrendTest)
    return trend_test.test_trend_type, \
           trend_test.test_trend_slope_strength, \
           trend_test.unconstained_nllh, \
           trend_test.test_trend_constant_quantile, \
           trend_test.mean_difference_same_sign_as_slope_strenght, \
           trend_test.variance_difference_same_sign_as_slope_strenght, \
           trend_test.unconstrained_model_deviance, \
           trend_test.constrained_model_deviance


def compute_gev_change_point_test_results(multiprocessing, maxima, starting_years, trend_test_class,
                                          years):
    if multiprocessing:
        list_args = [(maxima, starting_year, trend_test_class, years) for starting_year in
                     starting_years]
        with Pool(NB_CORES) as p:
            trend_test_res = p.starmap(compute_gev_change_point_test_result, list_args)
    else:
        trend_test_res = [
            compute_gev_change_point_test_result(maxima, starting_year, trend_test_class, years)
            for starting_year in starting_years]
    # Keep only the most likely starting year
    # (i.e. the starting year that minimizes its negative log likelihood)
    # (set all the other data to np.nan so that they will not be taken into account in mean function)
    best_idx = list(np.argmin(trend_test_res, axis=0))[2]
    # print(best_idx, trend_test_res)
    best_idxs = [best_idx]
    # todo: by doing a sorting on the deviance, I could get the nb_top_likelihood_values values
    # best_idxs = list(np.argmax(trend_test_res, axis=0))[-nb_top_likelihood_values:]

    return trend_test_res, best_idxs
