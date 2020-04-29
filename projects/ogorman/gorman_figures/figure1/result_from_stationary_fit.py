from typing import Dict

import numpy as np
from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.margin_estimator.utils import fitted_stationary_gev


class ResultFromDoubleStationaryFit(object):

    def __init__(self, study_before, study_after, fit_method, return_period):
        self.return_period = return_period
        self.fit_method = fit_method
        self.result_before = ResultFromSingleStationaryFit(study_before, fit_method, return_period)
        self.result_after = ResultFromSingleStationaryFit(study_after, fit_method, return_period)

    @property
    def massif_names(self):
        return self.re

    @cached_property
    def massif_name_to_difference_return_level(self):
        return {m: v - self.result_before.massif_name_to_return_level[m]
                for m, v in self.result_after.massif_name_to_return_level.items()}

    @cached_property
    def massif_name_to_relative_difference_return_level(self):
        return {m: 100 * v / self.result_before.massif_name_to_return_level[m]
                for m, v in self.massif_name_to_difference_return_level.items()}

    @cached_property
    def return_level_list_couple(self):
        return [(v, self.result_after.massif_name_to_return_level[m])
                for m, v in self.result_before.massif_name_to_return_level.items()]

    @cached_property
    def shape_list_couple(self):
        return [(v, self.result_after.massif_name_to_shape[m])
                for m, v in self.result_before.massif_name_to_shape.items()]


class ResultFromSingleStationaryFit(object):

    def __init__(self, study: AbstractStudy, fit_method, return_period):
        self.study = study
        self.fit_method = fit_method
        self.return_period = return_period

    @property
    def massif_name_to_maxima(self):
        return self.study.massif_name_to_annual_maxima

    @cached_property
    def massif_name_to_gev_param_fitted(self) -> Dict[str, GevParams]:
        return {m: fitted_stationary_gev(maxima, fit_method=self.fit_method) for m, maxima in
                self.massif_name_to_maxima.items()}

    @property
    def massif_name_to_shape(self):
        return {m: gev_param.shape for m, gev_param in self.massif_name_to_gev_param_fitted.items()}

    @cached_property
    def massif_name_to_return_level(self):
        return {m: gev_param.return_level(return_period=self.return_period)
                for m, gev_param in self.massif_name_to_gev_param_fitted.items()}

    @property
    def massif_name_to_difference_return_level_and_maxima(self):
        return {m: r - np.max(self.massif_name_to_maxima[m]) for m, r in self.massif_name_to_return_level.items()}
