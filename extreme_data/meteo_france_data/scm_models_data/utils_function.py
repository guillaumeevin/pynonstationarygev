import datetime
import math
import time
import warnings
from itertools import chain
from multiprocessing import Pool
from typing import Tuple, Dict

import numpy as np
from sklearn.utils import resample

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.margin_estimator.utils import _fitted_stationary_gev
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.eurocode_return_level_uncertainties import \
    EurocodeConfidenceIntervalFromExtremes
from root_utils import NB_CORES, batch


def fitted_stationary_gev_with_uncertainty_interval(x_gev, fit_method=MarginFitMethod.is_mev_gev_fit,
                                                    model_class=StationaryTemporalModel,
                                                    starting_year=None,
                                                    quantile_level=0.98,
                                                    confidence_interval_based_on_delta_method=True):
    estimator, gev_param = _fitted_stationary_gev(fit_method, model_class, starting_year, x_gev)
    if quantile_level is not None:
        EurocodeConfidenceIntervalFromExtremes.quantile_level = quantile_level
        coordinate = estimator.dataset.coordinates.df_all_coordinates.iloc[0].values
        if confidence_interval_based_on_delta_method:
            confidence_interval = EurocodeConfidenceIntervalFromExtremes.from_estimator_extremes(estimator,
                                                                                                 ConfidenceIntervalMethodFromExtremes.ci_mle,
                                                                                                 coordinate)
            mean_estimate = confidence_interval.mean_estimate
            confidence_interval = confidence_interval.confidence_interval
            return_level_list = None
        else:
            # Bootstrap method
            return_level_list = ReturnLevelBootstrap(fit_method, model_class, starting_year, x_gev,
                                                     quantile_level).compute_all_return_level()
            return_level_list = np.array(return_level_list)
            # Remove infinite return levels and return level
            # percentage_of_inf = 100 * sum([np.isinf(r) for r in return_level_list]) / len(return_level_list)
            # print('Percentage of fit with inf = {} \%'.format(percentage_of_inf))
            confidence_interval = tuple([np.quantile(return_level_list, q)
                                         for q in AbstractExtractEurocodeReturnLevel.bottom_and_upper_quantile])
            mean_estimate = gev_param.quantile(quantile_level)
    else:
        confidence_interval = None
        mean_estimate = None
        return_level_list = None
    return gev_param, mean_estimate, confidence_interval, return_level_list


class ReturnLevelBootstrap(object):
    only_physically_plausible_fits = False
    multiprocess = None

    def __init__(self, fit_method, model_class, starting_year, x_gev, quantile_level):
        self.nb_bootstrap = AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP
        self.quantile_level = quantile_level
        self.x_gev = x_gev
        self.starting_year = starting_year
        self.model_class = model_class
        self.fit_method = fit_method

    def compute_all_return_level(self):
        idxs = list(range(self.nb_bootstrap))
        multiprocess = self.multiprocess
        if AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP <= 10:
            multiprocess = False

        if multiprocess is None:

            with Pool(NB_CORES) as p:
                batchsize = math.ceil(AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP / NB_CORES)
                list_return_level = p.map(self.compute_return_level_batch, batch(idxs, batchsize=batchsize))
                return_level_list = list(chain.from_iterable(list_return_level))

        elif multiprocess:
            f = self.compute_return_level_physically_plausible if self.only_physically_plausible_fits else self.compute_return_level
            with Pool(NB_CORES) as p:
                return_level_list = p.map(f, idxs)
        else:
            f = self.compute_return_level_physically_plausible if self.only_physically_plausible_fits else self.compute_return_level
            return_level_list = [f(idx) for idx in idxs]

        return return_level_list

    def compute_return_level_batch(self, idxs):
        if self.only_physically_plausible_fits:
            return [self.compute_return_level_physically_plausible(idx) for idx in idxs]
        else:
            return [self.compute_return_level(idx) for idx in idxs]

    def compute_return_level(self, idx):
        x = resample(self.x_gev)
        with warnings.catch_warnings():
            gev_params = _fitted_stationary_gev(self.fit_method, self.model_class, self.starting_year, x)[1]
        return gev_params.quantile(self.quantile_level)

    def compute_return_level_physically_plausible(self, idx):
        gev_params = GevParams(0, 1, 0.6)
        while not (-0.5 <= gev_params.shape < 0.5):
            x = resample(self.x_gev)
            with warnings.catch_warnings():
                gev_params = _fitted_stationary_gev(self.fit_method, self.model_class, self.starting_year, x)[1]
        return gev_params.quantile(self.quantile_level)
